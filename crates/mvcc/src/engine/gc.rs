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
//! An `Arc` per version, dropped when the last reference goes away, answers (2)
//! correctly and is the obvious way to do it. It costs an atomic increment per
//! read — a write to a shared cache line on the read path, which is exactly what
//! MVCC exists to avoid. Reference counting loses to EBR here for the same
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
//! - No background thread. A slot being written is exactly a slot worth
//!   pruning, so the common case needs no search for work at all — see below
//!   for the sweep that covers the records this misses.
//!
//! Dead versions are always a *suffix*. `SlotWrite::commit` stamps a displaced
//! version's `end` with its successor's `begin`, so `end` decreases walking
//! down a chain, which makes `end <= watermark` downward-closed. Pruning is
//! therefore a tail truncation: null one `prev` and the whole dead suffix
//! detaches at once. No interior node is ever unlinked, so a reader mid-walk
//! never has the chain rearranged underneath it.
//!
//! A **deleted** record is emptied rather than trimmed. Once its tombstone is
//! itself below the watermark every reader sees the record as absent, so the
//! whole chain goes and the slot is left holding nothing. That returns the
//! record's data — the part that scales with `T`.
//!
//! What survives is the `Record` — the slot and its key — plus that key's bucket
//! entry and its slot in the iteration chunk list. Measured with a counting
//! allocator over 50,000 insert-then-delete keys: **about 180 bytes retained per
//! distinct key**, of which `Record` is 128 (a `Slot<T>` is one cache line by
//! `repr(align(64))`, and the key rounds it to two). The figure does not move
//! with the size of `T`: a record with a 512-byte payload retains the same 180
//! bytes, because everything that scales with `T` lives in the `Version`, which
//! *is* freed. In that measurement 39% of live bytes came back on delete.
//!
//! Those 180 bytes are reclaimable, but only with exclusive access:
//! [`Database::compact`](crate::engine::store::Database::compact) takes
//! `&mut self`, which is a compile-time proof that no transaction exists — a
//! `Transaction` borrows the database — and so that no slot resolved by one is
//! still in use. It rebuilds each shard around its survivors, which drops the
//! retained bytes to about 2 per key.
//!
//! Requiring exclusivity is a performance decision.
//! Freeing slots concurrently means revalidating each write against the key
//! map, or draining transactions behind a lock they all touch — either one puts
//! synchronisation back on a path that has none today, and the append-only map
//! is worth 1.9x on uniform point reads and 9.8x on contended ones at four
//! threads — the shipped before-and-after, not the throwaway lock-deletion
//! experiment `crate::engine::slotmap` tabulates.
//!
//! Those records are immortal during normal operation, and it is not a detail
//! that can be changed here — `SlotMap::get` hands out a `&Slot<T>` borrowed from the *map*
//! rather than from the guard, which is sound only because records outlive every
//! epoch. Freeing them means changing that borrow, and with it every caller.
//! So a workload churning through many *distinct* keys still grows, now at a
//! fixed cost per key ever used rather than per byte ever written.
//!
//! # Records nobody writes any more
//!
//! Pruning on the write path only ever reaches slots that are written. A record
//! written once and then only read would keep whatever versions it had at its
//! last write, so [`Database::sweep`](crate::engine::store::Database::sweep)
//! walks one shard of every table every `Database::SWEEP_INTERVAL` commits and
//! prunes what it finds.
//!
//! It runs there — on a commit, not on a thread and not on reads — for two
//! reasons. A background thread is a lifecycle to own and join. And pruning from
//! reads would put a lock acquisition, which is a *shared write*, back onto the
//! read path: the one property the whole engine is built around. A database with
//! no writes sweeps never, and needs nothing swept, because nothing is producing
//! versions.
//!
//! Two details keep it off the write path's critical cost. It checks
//! `Slot::needs_prune` before locking anything, so a slot with no garbage is
//! never contended — the slot lock is first-updater-wins, and a sweeper that
//! grabbed locks speculatively would make user transactions fail with
//! `WriteConflict`. And it is paced separately from the watermark refresh,
//! because tying the two together made sweep cost scale with the commit rate and
//! cost a third of serializable write throughput.
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
    /// reclaimable.
    pub watermark: Timestamp,
    /// Live transactions. A number that only grows is the leak.
    pub active_transactions: usize,
}
