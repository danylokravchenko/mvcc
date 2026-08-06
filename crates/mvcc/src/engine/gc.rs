//! Version reclamation.
//!
//! Two problems, often conflated:
//!
//! 1. **When is a version logically dead?** When `end < gc_watermark`, i.e. no
//!    live transaction's snapshot can reach it.
//! 2. **When is it safe to free the memory?** Later — a reader may already hold
//!    a pointer to it, obtained before the watermark moved.
//!
//! (1) is MVCC bookkeeping; (2) is a memory-reclamation problem, and it is
//! answered by epoch-based reclamation: a reader publishes its epoch once on
//! entry, and freeing waits until every thread has moved past the epoch in
//! which a version was retired. See [`crate::engine::store`] for the chain it
//! protects.
//!
//! The predecessor was an `Arc` per version, dropped when the last reference to
//! it went away. Correct, and correct first — but it cost an atomic increment
//! per read, a write to a shared cache line on the read path, which is exactly
//! what MVCC exists to avoid. Reference counting loses to EBR here for the same
//! reason the timestamp oracle avoids a shared counter.
//!
//! # Implementation status: chains are not pruned
//!
//! The above describes the mechanism the watermark exists for. **It is not
//! wired up.** Nothing in the engine walks a version chain and frees the
//! versions below the watermark:
//!
//! - `Version::prev` is written once, when the version is constructed, and is
//!   only ever loaded afterwards. No code truncates a chain.
//! - The one place a version is freed is `SlotWrite::abort`, which unlinks a
//!   version whose transaction never committed. Epoch reclamation otherwise
//!   covers `SlotMap`'s superseded bucket arrays, not versions.
//! - [`Oracle::gc_watermark`](crate::engine::oracle::Oracle::gc_watermark) has
//!   two consumers, and neither frees anything: expiring SIREAD locks in
//!   `crate::engine::ssi`, and [`GcStats`].
//!
//! So a committed version stays reachable from its chain for the lifetime of
//! the process. Memory grows with the total number of writes rather than with
//! the amount of live data — a record updated a million times holds a million
//! versions. Reads do not degrade with it, because the newest version is at the
//! head of the chain.
//!
//! # The failure mode this becomes
//!
//! Once a reclaimer exists it will be gated on the watermark, which is a
//! *minimum* over live snapshots. One forgotten transaction — a REPL session, a
//! leaked handle, an analytics query — will pin it and stop reclamation for
//! everyone. That is the most common way a real MVCC system falls over, and it
//! presents as a memory leak rather than as a transaction problem.
//!
//! [`GcStats::active_transactions`] and [`GcStats::watermark`] are already
//! reported, so the instrumentation is in place ahead of the mechanism: a
//! watermark that stops advancing while writes continue is the signal.

use crate::core::Timestamp;

/// A snapshot of reclamation state, from [`Database::stats`].
///
/// These report the state a reclaimer *would* be gated on. As the module docs
/// explain, version chains are not pruned yet, so a stalled watermark is not
/// what makes memory grow today — it grows with cumulative writes regardless.
///
/// [`watermark`] and [`active_transactions`] are still the pair to watch: they
/// are what will matter the moment reclamation lands, and a transaction count
/// that only climbs is a leaked handle in its own right. The signal is a flat
/// watermark while writes continue. Neither number means much alone — a flat
/// watermark on an idle database is just an idle database.
///
/// ```rust
/// # use mvcc::{Config, Database, Mvcc};
/// # #[derive(Mvcc, Clone)]
/// # struct Account {
/// #     #[mvcc(primary_key)] id: u64,
/// #     balance: i64,
/// # }
/// let db = Database::open(Config::in_memory())?;
/// db.register::<Account>()?;
///
/// // A transaction held open pins the watermark for everyone.
/// let held = db.begin();
/// assert_eq!(db.stats().active_transactions, 1);
///
/// drop(held);
/// assert_eq!(db.stats().active_transactions, 0);
/// # Ok::<(), mvcc::Error>(())
/// ```
///
/// [`Database::stats`]: crate::Database::stats
/// [`watermark`]: GcStats::watermark
/// [`active_transactions`]: GcStats::active_transactions
#[derive(Clone, Copy, Debug)]
pub struct GcStats {
    /// The oldest snapshot any live transaction still reads at, or the read
    /// watermark when there are none. Nothing is gated on it yet — see the
    /// module docs — but it is what a reclaimer would advance behind.
    pub watermark: Timestamp,
    /// Live transactions. A number that only grows is the leak.
    pub active_transactions: usize,
}
