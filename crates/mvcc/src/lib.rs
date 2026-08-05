//! Multi-version concurrency control for ordinary Rust structs.
//!
//! Add `#[derive(Mvcc)]` to a struct and it becomes a transactional, versioned,
//! in-memory table. Readers never block writers and writers never block readers.
//! Nothing is written to disk — see [What this is not](#what-this-is-not).
//!
//! ```
//! use mvcc::{Config, Database, Mvcc, Serializable};
//!
//! #[derive(Mvcc, Clone, Debug)]
//! #[mvcc(table = "accounts")]
//! struct Account {
//!     #[mvcc(primary_key)] id: u64,
//!     #[mvcc(index)] owner: String,
//!     balance: i64,
//! }
//!
//! let db = Database::open(Config::in_memory())?;
//! db.register::<Account>()?;
//!
//! // Snapshot isolation by default; retries on conflict.
//! db.transaction(|tx| {
//!     tx.insert(Account { id: 1, owner: "ada".into(), balance: 100 })?;
//!     tx.insert(Account { id: 2, owner: "bob".into(), balance: 0 })?;
//!     Ok(())
//! })?;
//!
//! // A transfer needs the stronger level: under snapshot isolation two
//! // concurrent transfers can each read a balance the other is about to
//! // change (write skew).
//! db.transaction_with::<Serializable, _, _>(|tx| {
//!     tx.update::<Account>(&1, |a| a.balance -= 50)?;
//!     tx.update::<Account>(&2, |a| a.balance += 50)?;
//!     Ok(())
//! })?;
//! # Ok::<(), mvcc::Error>(())
//! ```
//!
//! # Choosing an isolation level
//!
//! [`Snapshot`] is the default and the right answer for most workloads. Reach
//! for [`Serializable`] when a transaction's *write* depends on a value it
//! merely *read* — balance checks, capacity limits, uniqueness enforced in
//! application code. That is the write-skew shape, and snapshot isolation will
//! not catch it. Expect retriable aborts in exchange.
//!
//! # What this gives you
//!
//! - Snapshot isolation and serializability behave as documented per level; the
//!   anomaly suite in `tests/isolation_anomalies.rs` pins this down.
//! - Readers never block on writers, and never abort. Following a version chain
//!   takes no locks and writes nothing to shared memory.
//! - A transaction dropped without `commit()` rolls back.
//! - Fields may be of any type: nothing is serialised, so a record can hold a
//!   `HashMap`, an `Instant`, a function pointer — whatever your struct holds.
//!
//! # What this is not
//!
//! **It is not durable, by design.** Everything lives in memory and nothing
//! survives the process. There is no log, no checkpoint, no `data_dir`. Think
//! of it as a transactional, concurrent, versioned in-memory store — the
//! semantics of a database's concurrency control without a database's storage
//! layer. `DESIGN.md` §5 records what adding durability would take and why it
//! was left out.
//!
//! It is also not distributed, and the dataset must fit in memory.

#![doc(html_no_source)]

// Both modules are private: everything users touch is re-exported below, so the
// internal layout can change without it being a breaking change.
mod core;
mod engine;

pub use crate::core::{
    Error, IsolationLevel, ReadCommitted, RepeatableRead, Result, Serializable, Snapshot,
    Timestamp, TxnId, Versioned,
};
pub use crate::engine::store::Config;
pub use crate::engine::txn::Ref;
pub use crate::engine::{Database, Transaction};

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
    pub use crate::core::{Encodable, IndexDesc, IndexKey, TableId, Versioned};
}
