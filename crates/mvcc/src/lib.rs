//! Transactions, isolation levels, and enforced constraints for ordinary Rust
//! structs.
//!
//! This is a database's **concurrency control** without a database's **storage
//! layer**: atomicity, consistency, and isolation, but no durability. Nothing is
//! ever written to disk. That is a deliberate trade — see
//! [What this is not](#what-this-is-not).
//!
//! Add [`derive@Mvcc`] to a struct, register it with a [`Database`], and several
//! threads can read and write it through [`Transaction`]s that each see a
//! consistent snapshot of the whole database.
//!
//! # The problem it solves
//!
//! The honest comparison is not against Postgres, it is against
//! `RwLock<HashMap<K, V>>`.
//!
//! A lock gives you one map at a time. The moment an invariant spans *two* maps
//! — or two records in one map — you need a consistent view across both, and the
//! only way to get one from a lock is to hold it globally, which serialises
//! every reader against every writer. Long reads make it worse: an analytics
//! scan holding a read lock blocks writers for its whole duration.
//!
//! Multi-version concurrency control removes that trade. Reach for this crate
//! when you have **shared mutable state that several threads read and write, and
//! invariants that span more than one record**. If you have one map and no
//! cross-record invariant, the lock is simpler and you should use it.
//!
//! | you need | use |
//! | --- | --- |
//! | data to survive a restart | an embedded database — `redb`, `sled`, `SQLite` |
//! | more data than fits in RAM | anything with a buffer pool |
//! | multiple processes | a real database server |
//! | one map, one thread at a time | `RwLock<HashMap<K, V>>` |
//! | queries by shape rather than by key | a query engine; this has no planner |
//!
//! # How it works
//!
//! The core idea: **never overwrite data, and never block a reader.** An update
//! writes a *new version* and leaves the old one in place. Each record is a slot
//! holding a chain of versions, newest first:
//!
//! ```text
//! index ──► Slot ──► Version { begin: 40, end: MAX, value }   ← current
//!                         │
//!                         ▼
//!                    Version { begin: 20, end: 40,  value }
//!                         │
//!                         ▼
//!                    Version { begin:  5, end: 20,  value }
//! ```
//!
//! A version is visible to a snapshot `s` when `begin <= s < end`. That is the
//! whole visibility rule, and it is why the [isolation levels](#isolation-levels)
//! differ only in *which snapshot they pass in*.
//!
//! A transaction takes a snapshot timestamp when it begins and reads at it for
//! its whole life, so its view never changes underneath it, no matter what
//! commits alongside. Reading is a pointer walk and a comparison — no locks, no
//! reference counts, and no writes to shared memory at all, which is why
//! contended reads scale rather than collapse.
//!
//! Writing is **first-updater-wins**: a writer claims the slot, and a second
//! writer fails immediately with [`Error::WriteConflict`] rather than waiting.
//! Waiting would reintroduce deadlock detection, which is one of the things MVCC
//! removes. A delete installs a *tombstone* version rather than removing
//! anything, so a reader at an older snapshot still finds the record alive.
//!
//! # Getting started
//!
//! Declare the record, register it, and do the work inside a transaction:
//!
//! ```rust
//! use mvcc::{Config, Database, Mvcc, Serializable};
//!
//! #[derive(Mvcc, Clone, Debug)]
//! #[mvcc(table = "accounts")]
//! struct Account {
//!     #[mvcc(primary_key)]
//!     id: u64,
//!
//!     /// No two accounts may share an owner; the engine enforces it.
//!     #[mvcc(index(unique))]
//!     owner: String,
//!
//!     balance: i64,
//! }
//!
//! let db = Database::open(Config::in_memory())?;
//! db.register::<Account>()?;
//!
//! db.transaction(|tx| {
//!     tx.insert(Account { id: 1, owner: "ada".into(), balance: 100 })?;
//!     tx.insert(Account { id: 2, owner: "bob".into(), balance: 0 })
//! })?;
//!
//! // This transfer's *write* depends on a balance it *read*, so it needs the
//! // strongest level — see below.
//! let moved = db.transaction_with::<Serializable, _, _>(|tx| {
//!     let balance = tx.get::<Account>(&1)?.map_or(0, |a| a.balance);
//!     if balance < 50 {
//!         return Ok(false);
//!     }
//!     tx.update::<Account>(&1, |a| a.balance -= 50)?;
//!     tx.update::<Account>(&2, |a| a.balance += 50)?;
//!     Ok(true)
//! })?;
//!
//! assert!(moved);
//!
//! let mut tx = db.begin();
//! assert_eq!(tx.get::<Account>(&2)?.unwrap().balance, 50);
//! # Ok::<(), mvcc::Error>(())
//! ```
//!
//! [`Database::transaction`] is the API to reach for: it runs the closure,
//! commits it, and **re-runs it on a retriable conflict**. Because it may run
//! more than once, the closure must not have side effects outside the
//! transaction — return the value and let the caller act on the committed
//! result, as above. For manual control, [`Database::begin`] hands back a
//! transaction you commit yourself, and **dropping it without committing rolls
//! it back**.
//!
//! # Isolation levels
//!
//! The level is a **type parameter, not a runtime flag**, so the cost of the
//! strongest never leaks into the weakest: a [`ReadCommitted`] transaction
//! records no read set and allocates nothing.
//!
//! | level | sees | permits |
//! | --- | --- | --- |
//! | [`ReadCommitted`] | a fresh snapshot per statement | non-repeatable reads, phantoms |
//! | [`RepeatableRead`] | one snapshot | write skew |
//! | [`Snapshot`] *(default)* | one snapshot | write skew |
//! | [`Serializable`] | one snapshot + conflict detection | nothing |
//!
//! **Default to [`Snapshot`].** Reads never block and never abort, and lost
//! updates are impossible.
//!
//! **Reach for [`Serializable`] when a transaction's *write* depends on
//! something it merely *read*** — balance checks, capacity limits, "at least one
//! of these must remain true". That is the write-skew shape, and snapshot
//! isolation will not catch it: two transfers can each read a sufficient balance
//! and both withdraw, because on the write side they touch different rows and so
//! nothing conflicts. Expect retriable aborts in exchange.
//!
//! Isolation behaviour is verified against [Hermitage], Martin Kleppmann's
//! isolation test suite: all ten anomalies, each asserted *present or absent per
//! level*.
//!
//! # Conflicts are normal
//!
//! A conflict is the engine reporting that two transactions could not both
//! happen. [`Error::is_retriable`] separates the two cases: [`WriteConflict`]
//! and [`SerializationFailure`] mean re-run, and everything else is a
//! programming mistake that will fail again identically.
//! [`Database::transaction`] already loops on the retriable ones, so most code
//! never matches on this at all.
//!
//! When scanning, prefer [`Transaction::scan_where`] over
//! [`scan`][Transaction::scan] plus `.filter()`. The first hands the predicate to
//! the engine, which re-evaluates it at commit and so can detect a row that
//! *appears* and matches — a phantom. The second says only that you read the
//! entire table, so any concurrent write to it aborts you.
//!
//! # Memory growth
//!
//! Superseded versions are reclaimed. When a write commits it also prunes the
//! record's chain, freeing every version no live transaction can still reach,
//! so steady-state memory tracks live data rather than cumulative writes.
//!
//! **Reclamation is bounded below by the oldest live transaction.** The cutoff
//! is a *minimum* over live snapshots, so one forgotten transaction — a REPL
//! session, a leaked handle, a long-running scan — pins it and version chains
//! grow without limit for as long as it is open. It presents as a memory leak
//! rather than as a transaction problem, and it is the most common way real
//! MVCC systems fall over. [`Database::stats`] exposes the watermark and the
//! live transaction count; watch them, and see [`stats::GcStats`].
//!
//! Deleting a record eventually returns everything it held: once the tombstone
//! is itself below the watermark, the whole chain goes. And records that stop
//! being written are collected too, by a sweep that rides on other commits — so
//! a record written once and then only read does not keep its history forever.
//!
//! What is **not** reclaimed is the per-key slot: roughly **180 bytes for every
//! distinct key** the database has ever held, measured with a counting
//! allocator. That figure is flat in the size of the record — everything that
//! scales with your type lives in the version, which is freed — so a workload
//! churning through unboundedly many distinct keys still grows, but at a fixed
//! cost per key rather than per byte written.
//!
//! [`Database::compact`] gives those bytes back. Call it in a
//! quiet moment if your key space is unbounded.
//!
//! # What this is not
//!
//! **It is not durable, by design.** Everything lives in memory and nothing
//! survives the process — no log, no checkpoint, no `data_dir`, no recovery.
//!
//! It is also not distributed — one process, one machine — and the dataset must
//! fit in memory. There is no query planner: records are reached by primary key,
//! by predicate, or by a range over a declared index.
//!
//! # Where to look next
//!
//! - [`derive@Mvcc`] — declaring a record, and the full attribute reference.
//! - [`Transaction`] — reading, writing, and scanning.
//! - [`Database`] — opening, registering types, running transactions.
//! - [`Error`] — what can fail, and which failures are worth retrying.
//! - The `examples/` directory in the repository is the fastest way in. Each
//!   example ends by asserting that the world it built is still consistent;
//!   `game.rs` is the end-to-end tour, covering a write conflict, write skew and
//!   its fix, a phantom, an atomic trade, a long report reading while the world
//!   moves, and a four-thread raid.
//!
//! [Hermitage]: https://github.com/ept/hermitage
//! [`WriteConflict`]: Error::WriteConflict
//! [`SerializationFailure`]: Error::SerializationFailure

#![doc(html_no_source)]

// The derive emits absolute `::mvcc::__private` paths, which have no crate to
// resolve against inside `mvcc` itself. Unit tests here use the derive like any
// user would, so give the crate its own name to answer them.
#[cfg(test)]
extern crate self as mvcc;

// Both modules are private: everything users touch is re-exported below, so the
// internal layout can change without it being a breaking change.
mod core;
mod engine;

pub use crate::core::{
    Error, Index, IsolationLevel, ReadCommitted, RepeatableRead, Result, Serializable, Snapshot,
    Timestamp, TxnId, Versioned,
};
pub use crate::engine::store::Config;
pub use crate::engine::txn::Ref;
pub use crate::engine::{Database, Transaction};

/// Make a struct storable in a `Database`, by implementing `Versioned` for it.
///
/// Exactly one field must be marked `#[mvcc(primary_key)]`. Any others may be
/// indexed. The struct must also be `Clone`, because an update copies the record
/// before mutating it into a new version.
///
/// ```rust
/// use mvcc::{Config, Database, Mvcc};
///
/// #[derive(Mvcc, Clone, Debug)]
/// #[mvcc(table = "accounts")]
/// pub struct Account {
///     #[mvcc(primary_key)]
///     pub id: u64,
///
///     /// No two accounts may share an owner.
///     #[mvcc(index(unique))]
///     pub owner: String,
///
///     /// Range-scannable, duplicates allowed.
///     #[mvcc(index)]
///     pub branch: u32,
///
///     pub balance: i64,
/// }
///
/// let db = Database::open(Config::in_memory())?;
/// db.register::<Account>()?;
///
/// db.transaction(|tx| {
///     tx.insert(Account { id: 1, owner: "ada".into(), branch: 10, balance: 500 })
/// })?;
///
/// // The derive emits `Account::BRANCH` for the indexed field, and that const —
/// // not a string — is what a scan takes.
/// let mut tx = db.begin();
/// let at_branch_10 = tx.scan_index(Account::BRANCH, 10u32..=10)?;
/// assert_eq!(at_branch_10.len(), 1);
/// # Ok::<(), mvcc::Error>(())
/// ```
///
/// # Attribute reference
///
/// | attribute | position | meaning |
/// |---|---|---|
/// | `table = "name"` | struct | table name, used only in error messages; defaults to the type name |
/// | `primary_key` | field | required, exactly one |
/// | `index` | field | secondary index on this field |
/// | `index(unique)` | field | unique secondary index |
///
/// An index is always named after its field. There is no rename knob: the name
/// is not a string anyone types, it is the associated const above, so renaming
/// it would only decouple the const from the field it reads.
///
/// There is no `skip`: nothing is serialised, so every field is simply carried
/// along in the struct. Fields need no traits beyond what `Versioned` requires
/// of the struct as a whole — a record can hold a `HashMap`, an `Instant`, or a
/// function pointer.
///
/// # What it expands to
///
/// - `impl Versioned for Account` — key type and extraction, `memcmp` key
///   encoding, and the index descriptor table.
/// - a private `static` holding the index descriptors, const-constructed with
///   `extract` as a plain `fn` pointer.
/// - a private `static OnceLock<TableId>`, filled by `Database::register`.
/// - `Account::OWNER: Index<Account, String>` — one associated const per
///   indexed field, named after the field in upper case, which is how scans
///   name an index. Carrying the field's type is the point: it is what makes
///   `tx.scan_index(Account::OWNER, 1u64..=2)` a type error rather than a scan
///   that matches nothing.
///
/// Everything is emitted inside a `const _: () = { … };` block so the generated
/// statics cannot collide with user items or with a second derive in the same
/// module.
///
/// # What it deliberately does not do
///
/// It does not make the struct itself transactional. `Account` gains no interior
/// mutability, no `Drop`, and no hidden fields — it stays a plain Rust struct
/// you can construct, match on, and pass around. All transactional behaviour
/// lives on `Transaction`, which is where errors can actually be returned.
///
/// # Compile errors
///
/// Missing a primary key is rejected at expansion time rather than producing a
/// type that fails obscurely later:
///
/// ```compile_fail
/// # use mvcc::Mvcc;
/// #[derive(Mvcc, Clone)]
/// struct NoKey {
///     name: String,
/// }
/// ```
///
/// So is declaring two of them, and so is applying the derive to an enum,
/// a union, or a tuple struct.
///
/// The macro itself lives in the `mvcc-derive` crate. It is documented here, on
/// the re-export, because a compiling example needs the [`Versioned`] trait it
/// implements — which lives in this crate, so documenting it from the other
/// direction would mean a dependency cycle.
#[cfg(feature = "derive")]
pub use mvcc_derive::Mvcc;

/// Tuning knobs.
pub mod config {
    pub use crate::engine::oracle::OracleConfig;
}

/// Runtime statistics. Watch `watermark` and `active_transactions` — see the
/// [garbage collection](crate::stats::GcStats) notes for why.
pub mod stats {
    pub use crate::engine::gc::GcStats;
}

/// Used by `#[derive(Mvcc)]`. Not a stable API.
///
/// Generated code refers to everything through this module so that renaming an
/// internal path is not a breaking change.
#[doc(hidden)]
pub mod __private {
    pub use crate::core::{Encodable, Index, IndexDesc, IndexKey, TableId, Versioned};
}
