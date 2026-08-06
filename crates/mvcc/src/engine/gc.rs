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
//! # How chains are pruned
//!
//! [`Slot::prune`](crate::engine::store::Slot::prune) does the reclaiming, and
//! it runs on the **write path** — inside `SlotWrite::commit`, while that
//! writer still holds the slot lock. That placement is the whole design:
//!
//! - The lock is already held, so no other writer can be mutating the chain and
//!   pruning needs no synchronisation of its own.
//! - The write that just lengthened the chain is what pays to shorten it, so
//!   cost lands on the workload creating the garbage.
//! - No background thread, and no sweep to find which slots are worth visiting.
//!
//! Dead versions are always a *suffix*. `SlotWrite::commit` stamps a displaced
//! version's `end` with its successor's `begin`, so `end` decreases walking
//! down a chain, which makes `end <= watermark` downward-closed. Pruning is
//! therefore a tail truncation: null one `prev` and the whole dead suffix
//! detaches at once. No interior node is ever unlinked, so a reader mid-walk
//! never has the chain rearranged underneath it.
//!
//! What this does **not** reclaim: the `Slot` itself, and the tombstone left by
//! a delete. Slots are immortal on purpose — it is what lets `SlotMap::get`
//! hand out a `&Slot<T>` borrowed from the map with no reclamation argument at
//! all. A workload that inserts and deletes many *distinct* keys still grows.
//!
//! # The watermark is a hint
//!
//! Pruning reads [`Database::gc_hint`](crate::engine::store::Database::gc_hint)
//! rather than calling [`Oracle::gc_watermark`](crate::engine::oracle::Oracle::gc_watermark),
//! which takes all sixteen shard locks and is far too expensive per commit. The
//! hint is recomputed every `Database::GC_HINT_INTERVAL` commits.
//!
//! Staleness is safe in exactly one direction and this errs in it: the true
//! watermark never moves backwards, so a stale hint is always a lower bound —
//! it prunes less than it could, never more.
//!
//! # The failure mode to instrument
//!
//! Reclamation is bounded below by the oldest live snapshot, because the
//! watermark is a *minimum* over them. One forgotten transaction — a REPL
//! session, a leaked handle, an analytics query — pins it, and from that moment
//! version chains grow without limit again. That is the most common way a real
//! MVCC system falls over, and it presents as a memory leak rather than as a
//! transaction problem.
//!
//! Watch [`GcStats::active_transactions`] and [`GcStats::watermark`]: a
//! watermark that stops advancing while writes continue is the signal.

use crate::core::Timestamp;

/// A snapshot of reclamation state, from [`Database::stats`].
///
/// These two are what version reclamation is gated on, which is why they are
/// the pair to alert on. Pruning cannot pass the oldest live snapshot, so a
/// single forgotten transaction stops it for every record at once and chains
/// start growing without limit again.
///
/// The signal is [`watermark`] flat while writes continue, usually alongside an
/// [`active_transactions`] that only climbs. Neither number means much alone —
/// a flat watermark on an idle database is just an idle database.
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
    /// watermark when there are none. Versions superseded at or before this are
    /// reclaimable, so if it stops advancing, nothing else matters.
    pub watermark: Timestamp,
    /// Live transactions. A number that only grows is the leak.
    pub active_transactions: usize,
}
