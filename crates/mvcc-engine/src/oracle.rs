//! The timestamp oracle — and the single most important scalability decision in
//! the engine.
//!
//! Every transaction takes a snapshot at begin and a timestamp at commit, and
//! both used to run through a global mutex. Measurement put those two mutexes at
//! roughly a third of the write path's cost at 8 threads; removing them
//! entirely, as a throwaway experiment, moved write throughput from 1.23M to
//! 1.79M ops/sec. Neither uses a lock now.
//!
//! # The read watermark, without a mutex
//!
//! The watermark is the largest `T` such that every commit at or below `T` has
//! finished installing its versions. Readers use it rather than the raw counter,
//! because the counter can name a commit whose versions are still being stamped
//! — a reader at that timestamp would see a torn commit.
//!
//! Computing it used to mean a `BTreeSet` of in-flight timestamps under a mutex,
//! walked on every commit. It is really a sequence-completion problem, so it is
//! now a ring of atomics (the LMAX Disruptor sequencer pattern):
//!
//! ```text
//!   completed[ts % RING] = ts        once `ts` has finished installing
//!   watermark            advances    while completed[watermark + 1] == watermark + 1
//! ```
//!
//! An in-flight commit leaves a hole and the watermark stops there — exactly the
//! old semantics, with no lock and no allocation.
//!
//! This needs commit timestamps to be **gap-free**, which is why transaction ids
//! now come from their own counter. The two can no longer be told apart by
//! value, and do not need to be: the tag bit in `mvcc_core::time` is what
//! distinguishes an in-flight writer from a commit timestamp.
//!
//! # The GC watermark, sharded
//!
//! Live snapshots still need a min-reduction, but nothing needs it to be exact
//! at every instant — it gates version reclamation, which is a background
//! concern. It is sharded by transaction id, so registering and releasing touch
//! one shard's lock instead of a single global one, and the min is taken across
//! shards only when someone asks.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

use mvcc_core::{Timestamp, TxnId};

/// Slots in the completion ring. A commit only stalls if it runs this far ahead
/// of the watermark, which takes this many commits installing at once.
const RING: usize = 4096;
const RING_MASK: u64 = RING as u64 - 1;

/// Independent shards for the live-snapshot set. More than the core count buys
/// nothing; fewer leaves contention on the table.
const ACTIVE_SHARDS: usize = 16;

/// How timestamps are handed out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OracleConfig {
    /// One global atomic counter for commit timestamps.
    ///
    /// Exact and totally ordered. The counter is still a contended cache line —
    /// see `Batched` and `Epoch` for the ways out — but it is now a bare
    /// `fetch_add` with no lock behind it.
    #[default]
    Centralised,

    /// Each thread claims a block of `stride` timestamps with one `fetch_add`
    /// and hands them out locally. Cuts coherence traffic by `stride`, at the
    /// cost of timestamps no longer being densely allocated — which the
    /// completion ring currently relies on, so this needs a different watermark.
    ///
    /// Not yet implemented.
    Batched { stride: u64 },

    /// Silo-style epochs: a global epoch advances on a timer; transactions are
    /// ordered *between* epochs but unordered *within* one, which removes the
    /// shared counter from the commit path entirely.
    ///
    /// Costs a latency floor of `epoch_micros` on commit and gives up strict
    /// serializability (serializability itself holds).
    ///
    /// Not yet implemented.
    Epoch { epoch_micros: u64 },
}

#[repr(align(64))] // one shard per cache line
struct ActiveShard {
    /// Snapshot to the number of transactions holding it.
    ///
    /// A multiset, not a set: transactions that begin before any commit lands
    /// all share a snapshot value, and a plain set would let the first of them
    /// to finish deregister the others.
    snapshots: Mutex<BTreeMap<u64, usize>>,
}

/// Hands out transaction ids and commit timestamps, and maintains the two
/// watermarks.
pub struct Oracle {
    config: OracleConfig,
    /// Commit timestamps. Gap-free, because the completion ring depends on it.
    next_ts: AtomicU64,
    /// Transaction ids. A separate sequence, so ids do not punch holes in the
    /// commit timestamps.
    next_id: AtomicU64,
    /// Highest timestamp below which every commit is fully installed.
    read_watermark: AtomicU64,
    /// `completed[ts & RING_MASK] == ts` once `ts` has finished installing.
    completed: Box<[AtomicU64]>,
    active: Box<[ActiveShard]>,
}

impl Oracle {
    pub fn new(config: OracleConfig) -> Self {
        Oracle {
            config,
            // Start at 1: timestamp 0 means "before everything", and is the
            // watermark's initial value.
            next_ts: AtomicU64::new(1),
            next_id: AtomicU64::new(1),
            read_watermark: AtomicU64::new(0),
            completed: (0..RING).map(|_| AtomicU64::new(0)).collect(),
            active: (0..ACTIVE_SHARDS)
                .map(|_| ActiveShard {
                    snapshots: Mutex::new(BTreeMap::new()),
                })
                .collect(),
        }
    }

    pub fn config(&self) -> OracleConfig {
        self.config
    }

    /// Restart the counters above `ts`.
    pub fn resume_after(&self, ts: Timestamp) {
        self.next_ts.store(ts.raw() + 1, Ordering::Release);
        self.read_watermark.store(ts.raw(), Ordering::Release);
    }

    /// Allocate a transaction id.
    ///
    /// Ids and commit timestamps come from different counters and may coincide
    /// numerically. That is safe: a version's `begin` field is tagged (see
    /// `mvcc_core::time`), and the tag — not the value — says whether it holds
    /// an in-flight writer or a commit timestamp.
    pub fn next_txn_id(&self) -> TxnId {
        TxnId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Take a snapshot for a beginning transaction and register it as live.
    ///
    /// Returns the *read watermark*, not the raw counter, so a transaction
    /// cannot see a commit whose versions are still being stamped.
    pub fn begin_snapshot(&self, id: TxnId) -> Timestamp {
        let ts = Timestamp(self.read_watermark.load(Ordering::Acquire));
        *self.shard(id).snapshots.lock().entry(ts.raw()).or_insert(0) += 1;
        ts
    }

    /// A snapshot for a statement in a `ReadCommitted` transaction, and for
    /// commit-time revalidation. Registers nothing: the caller's begin snapshot
    /// already pins the watermark.
    pub fn statement_snapshot(&self) -> Timestamp {
        Timestamp(self.read_watermark.load(Ordering::Acquire))
    }

    /// Drop a transaction's registration when it commits or aborts.
    pub fn release_snapshot(&self, id: TxnId, ts: Timestamp) {
        let mut shard = self.shard(id).snapshots.lock();
        if let std::collections::btree_map::Entry::Occupied(mut e) = shard.entry(ts.raw()) {
            *e.get_mut() -= 1;
            if *e.get() == 0 {
                e.remove();
            }
        }
    }

    /// Allocate a commit timestamp.
    ///
    /// Must be paired with [`Oracle::publish`], or the read watermark stops at
    /// this timestamp and every later snapshot freezes.
    pub fn begin_commit(&self) -> Timestamp {
        let ts = self.next_ts.fetch_add(1, Ordering::Relaxed);

        // The ring can only describe `RING` timestamps at once, so running
        // further ahead than that would overwrite a slot the watermark has not
        // reached. It takes `RING` commits installing simultaneously, so in
        // practice this never spins.
        while ts.saturating_sub(self.read_watermark.load(Ordering::Acquire)) >= RING as u64 {
            std::hint::spin_loop();
        }
        Timestamp(ts)
    }

    /// Publish that `ts` is fully installed, and advance the read watermark over
    /// whatever contiguous run of completed commits that exposes.
    pub fn publish(&self, ts: Timestamp) {
        self.completed[(ts.raw() & RING_MASK) as usize].store(ts.raw(), Ordering::Release);

        // Several threads may run this at once. The CAS decides which of them
        // moves the watermark; a loser re-reads and continues. The watermark
        // only ever moves forward and only over completed slots, so no commit
        // can be skipped and none can be exposed twice.
        loop {
            let current = self.read_watermark.load(Ordering::Acquire);
            let candidate = current + 1;
            if self.completed[(candidate & RING_MASK) as usize].load(Ordering::Acquire) != candidate
            {
                return; // a hole: some earlier commit is still installing
            }
            if self
                .read_watermark
                .compare_exchange_weak(current, candidate, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                std::hint::spin_loop();
            }
        }
    }

    /// The oldest snapshot any live transaction can still see. Versions whose
    /// `end` is at or below this are unreachable and may be reclaimed.
    ///
    /// A single long-running reader pins this and stalls all reclamation — the
    /// classic MVCC failure mode. See [`crate::gc`].
    pub fn gc_watermark(&self) -> Timestamp {
        let oldest = self
            .active
            .iter()
            .filter_map(|s| s.snapshots.lock().keys().next().copied())
            .min();
        match oldest {
            Some(ts) => Timestamp(ts),
            None => Timestamp(self.read_watermark.load(Ordering::Acquire)),
        }
    }

    /// Number of live transactions, for [`crate::gc::GcStats`].
    pub fn active_count(&self) -> usize {
        self.active
            .iter()
            .map(|s| s.snapshots.lock().values().sum::<usize>())
            .sum()
    }

    fn shard(&self, id: TxnId) -> &ActiveShard {
        &self.active[(id.0 as usize) % ACTIVE_SHARDS]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watermark_waits_for_out_of_order_installs() {
        let o = Oracle::new(OracleConfig::Centralised);
        let first = o.begin_commit();
        let second = o.begin_commit();
        assert!(first < second);

        // The later commit finishes installing first. The watermark must not
        // advance past it, or a reader would see `second` without `first`.
        o.publish(second);
        assert!(
            o.statement_snapshot() < first,
            "watermark passed a commit that is still installing"
        );

        o.publish(first);
        assert!(
            o.statement_snapshot() >= second,
            "watermark should now cover both"
        );
    }

    #[test]
    fn watermark_crosses_a_long_run_in_one_go() {
        let o = Oracle::new(OracleConfig::Centralised);
        let stamps: Vec<_> = (0..100).map(|_| o.begin_commit()).collect();
        // Published in reverse, so only the very last call can move anything.
        for ts in stamps.iter().rev() {
            o.publish(*ts);
        }
        assert_eq!(o.statement_snapshot(), *stamps.last().unwrap());
    }

    #[test]
    fn gc_watermark_is_pinned_by_the_oldest_reader() {
        let o = Oracle::new(OracleConfig::Centralised);
        let ts = o.begin_commit();
        o.publish(ts);

        let old = o.next_txn_id();
        let old_reader = o.begin_snapshot(old);
        let commit = o.begin_commit();
        o.publish(commit);
        let new = o.next_txn_id();
        let _new_reader = o.begin_snapshot(new);

        assert_eq!(o.gc_watermark(), old_reader, "the oldest reader pins GC");

        o.release_snapshot(old, old_reader);
        assert!(o.gc_watermark() > old_reader, "releasing it lets GC advance");
    }

    #[test]
    fn transactions_sharing_a_snapshot_are_counted_separately() {
        // Regression: every transaction that begins before the first commit
        // gets the same snapshot value. Tracking them in a set rather than a
        // multiset let the first to finish deregister the rest, which advanced
        // the GC watermark past snapshots that were still live.
        //
        // The two ids are chosen to land in the *same* shard, so this exercises
        // the collision rather than accidentally testing two shards.
        let o = Oracle::new(OracleConfig::Centralised);
        let a = TxnId(1);
        let b = TxnId(1 + ACTIVE_SHARDS as u64);
        let sa = o.begin_snapshot(a);
        let sb = o.begin_snapshot(b);
        assert_eq!(sa, sb, "both began before anything committed");
        assert_eq!(o.active_count(), 2);

        let ts = o.begin_commit();
        o.publish(ts);

        o.release_snapshot(a, sa);
        assert_eq!(o.active_count(), 1, "b is still running");
        assert_eq!(o.gc_watermark(), sb, "b's snapshot must still pin GC");

        o.release_snapshot(b, sb);
        assert_eq!(o.active_count(), 0);
        assert!(o.gc_watermark() > sb, "now GC may advance");
    }

    #[test]
    fn concurrent_commits_leave_the_watermark_consistent() {
        use std::sync::Arc;
        use std::thread;

        let o = Arc::new(Oracle::new(OracleConfig::Centralised));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let o = Arc::clone(&o);
                thread::spawn(move || {
                    for _ in 0..2_000 {
                        let ts = o.begin_commit();
                        o.publish(ts);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().expect("worker panicked");
        }

        // Every timestamp handed out was published, so the watermark must have
        // caught all the way up. A stall here would mean the CAS loop dropped a
        // completed slot.
        let next = o.next_ts.load(Ordering::Acquire);
        assert_eq!(
            o.statement_snapshot(),
            Timestamp(next - 1),
            "watermark stalled below a fully published sequence"
        );
    }
}
