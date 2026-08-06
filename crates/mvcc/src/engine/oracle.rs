//! The timestamp oracle.
//!
//! Every transaction takes a snapshot at begin and a timestamp at commit, and
//! neither takes a lock. A global mutex on each is the obvious implementation
//! and costs roughly a third of the write path at 8 threads: deleting both as a
//! throwaway experiment moves write throughput from 1.23M to 1.79M ops/sec.
//!
//! # The read watermark, without a mutex
//!
//! The watermark is the largest `T` such that every commit at or below `T` has
//! finished installing its versions. Readers use it rather than the raw counter,
//! because the counter can name a commit whose versions are still being stamped
//! — a reader at that timestamp would see a torn commit.
//!
//! Computing it from a `BTreeSet` of in-flight timestamps under a mutex, walked
//! on every commit, is the direct approach. But it is really a sequence-
//! completion problem, so this is a ring of atomics instead (the LMAX Disruptor
//! sequencer pattern):
//!
//! ```text
//!   completed[ts % RING] = ts        once `ts` has finished installing
//!   watermark            advances    while completed[watermark + 1] == watermark + 1
//! ```
//!
//! An in-flight commit leaves a hole and the watermark stops there — the same
//! semantics the `BTreeSet` gives, with no lock and no allocation.
//!
//! This needs commit timestamps to be **gap-free**, which is why transaction ids
//! come from their own counter. The two cannot be told apart by value, and do
//! not need to be: the tag bit in `crate::core::time` is what distinguishes an
//! in-flight writer from a commit timestamp.
//!
//! # The GC watermark, sharded
//!
//! Live snapshots still need a min-reduction, but nothing needs it to be exact
//! at every instant — it gates version reclamation, which is a background
//! concern. It is sharded by transaction id, so registering and releasing touch
//! one shard's lock instead of a single global one, and the min is taken across
//! shards only when someone asks.

use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

use crate::core::{Timestamp, TxnId};

/// Slots in the completion ring. A commit only stalls if it runs this far ahead
/// of the watermark, which takes this many commits installing at once.
const RING: usize = 4096;
const RING_MASK: u64 = RING as u64 - 1;

/// Independent shards for the live-snapshot set and the transaction-id counter.
/// More than the core count buys nothing; fewer leaves contention on the table.
///
/// The two use the same number deliberately. Ids are handed out as
/// `1 + slot + SHARDS * n`, so `id % SHARDS` recovers the thread's slot — which
/// means a thread's transactions always land in *its own* snapshot shard, and
/// the two structures contend as one instead of interleaving randomly.
const SHARDS: usize = 16;

/// A stable per-thread index into the sharded counters.
///
/// One atomic increment per thread for the lifetime of the process, then a
/// thread-local read. Deliberately process-global rather than per-`Database`:
/// it is only a hint for which shard to prefer, and every `Database` has its own
/// counters, so sharing it across them cannot produce a duplicate id.
fn thread_slot() -> usize {
    use std::cell::Cell;
    static NEXT: AtomicU64 = AtomicU64::new(0);
    thread_local! {
        static SLOT: Cell<Option<usize>> = const { Cell::new(None) };
    }
    SLOT.with(|slot| match slot.get() {
        Some(s) => s,
        None => {
            let s = (NEXT.fetch_add(1, Ordering::Relaxed) as usize) % SHARDS;
            slot.set(Some(s));
            s
        }
    })
}

/// How timestamps are handed out.
///
/// One strategy, deliberately. Two others are described in the literature and
/// were the obvious next steps; both turned out to conflict with the completion
/// ring that computes the read watermark, and neither is offered as a setting
/// that silently does nothing.
///
/// **Batched** — each thread claims a block of `stride` timestamps with one
/// `fetch_add` and hands them out locally. It cannot work here: the ring needs
/// commit timestamps to be *dense*, and a reserved-but-unused timestamp is an
/// permanent hole. Tried as an experiment, it did not merely run slowly, it
/// deadlocked — every committer blocked waiting for a watermark that could
/// never advance. Making it work needs the watermark to track reserved ranges,
/// at which point an idle thread holding a block freezes every snapshot in the
/// system.
///
/// **Epoch** (Silo-style) — a global epoch advances on a timer and transactions
/// are ordered between epochs but not within one, which removes the shared
/// counter from the commit path entirely. The obstacle is not the counter but
/// the two things built on top of it here: visibility would lag by up to an
/// epoch, so "commit, then read it back" would stop working without an extra
/// wait; and SSI revalidates read sets against the read watermark, which would
/// no longer be able to see same-epoch commits. It remains the right answer for
/// a much larger machine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum OracleConfig {
    /// One global atomic counter for commit timestamps.
    ///
    /// Exact and totally ordered, and now a bare `fetch_add` with no lock
    /// behind it. The counter is still a shared cache line, but measurement put
    /// the watermark's compare-exchange well ahead of it as the write path's
    /// cost — see `Oracle::publish`.
    #[default]
    Centralised,
}

#[repr(align(64))] // one counter per cache line
struct IdShard {
    next: AtomicU64,
}

#[repr(align(64))] // one shard per cache line
struct ActiveShard {
    /// Snapshots of the transactions live in this shard.
    ///
    /// A flat `Vec`, and a multiset: duplicates are separate entries, because
    /// transactions that begin before any commit lands all share a snapshot
    /// value and one finishing must not deregister the others.
    ///
    /// Ordered structures are the obvious choice for "give me the minimum" and
    /// the wrong one here. A shard holds a handful of live transactions, so a
    /// `BTreeMap` spent a tree descent and sometimes a node allocation on every
    /// begin and every release — measurable per-transaction cost — to make a
    /// scan over three elements marginally cheaper. Push, `swap_remove`, and a
    /// linear minimum beat it comfortably at this size, and the `Vec` keeps its
    /// capacity so the steady state does not allocate at all.
    snapshots: Mutex<Vec<u64>>,
}

/// Hands out transaction ids and commit timestamps, and maintains the two
/// watermarks.
pub(crate) struct Oracle {
    /// Commit timestamps. Gap-free, because the completion ring depends on it.
    next_ts: AtomicU64,
    /// Transaction ids, one counter per shard.
    ///
    /// Unlike commit timestamps these need only be *unique*, not dense — the
    /// completion ring does not track them — which is what makes sharding them
    /// legal. A single global `fetch_add` here was the last per-transaction
    /// shared write, and it was enough on its own to make read-only
    /// transactions scale negatively.
    next_id: Box<[IdShard]>,
    /// Highest timestamp below which every commit is fully installed.
    read_watermark: AtomicU64,
    /// `completed[ts & RING_MASK] == ts` once `ts` has finished installing.
    completed: Box<[AtomicU64]>,
    active: Box<[ActiveShard]>,
}

impl Oracle {
    /// Takes the strategy even though [`OracleConfig`] has exactly one variant,
    /// for the reasons its own docs give: there is nothing to select between
    /// yet, and keeping the parameter means adding a second strategy is a
    /// change inside this file rather than to `Database::open`'s signature.
    pub(crate) fn new(_config: OracleConfig) -> Self {
        Oracle {
            // Start at 1: timestamp 0 means "before everything", and is the
            // watermark's initial value.
            next_ts: AtomicU64::new(1),
            next_id: (0..SHARDS)
                .map(|_| IdShard {
                    next: AtomicU64::new(0),
                })
                .collect(),
            read_watermark: AtomicU64::new(0),
            completed: (0..RING).map(|_| AtomicU64::new(0)).collect(),
            active: (0..SHARDS)
                .map(|_| ActiveShard {
                    snapshots: Mutex::new(Vec::new()),
                })
                .collect(),
        }
    }

    /// Allocate a transaction id.
    ///
    /// Ids and commit timestamps come from different counters and may coincide
    /// numerically. That is safe: a version's `begin` field is tagged (see
    /// `crate::core::time`), and the tag — not the value — says whether it holds
    /// an in-flight writer or a commit timestamp.
    pub(crate) fn next_txn_id(&self) -> TxnId {
        let slot = thread_slot();
        let n = self.next_id[slot].next.fetch_add(1, Ordering::Relaxed);
        // `1 +` because 0 is `TxnId::NONE`, and because `Slot::lock` uses 0 to
        // mean "free". Striding by `SHARDS` keeps ids from different shards
        // distinct without any coordination between them.
        TxnId(1 + slot as u64 + SHARDS as u64 * n)
    }

    /// Take a snapshot for a beginning transaction and register it as live.
    ///
    /// Returns the *read watermark*, not the raw counter, so a transaction
    /// cannot see a commit whose versions are still being stamped.
    pub(crate) fn begin_snapshot(&self, id: TxnId) -> Timestamp {
        // Bring the watermark current, since this is the point at which its
        // staleness would become observable. One load and a comparison when
        // there is nothing to do; the compare-exchange runs only when it can
        // make progress. This is what keeps "commit, then read it back" working
        // while commits themselves stay off the watermark.
        self.advance();
        let ts = Timestamp(self.read_watermark.load(Ordering::Acquire));
        self.shard(id).snapshots.lock().push(ts.raw());
        ts
    }

    /// A snapshot for a statement in a `ReadCommitted` transaction, and for
    /// commit-time revalidation. Registers nothing: the caller's begin snapshot
    /// already pins the watermark.
    pub(crate) fn statement_snapshot(&self) -> Timestamp {
        Timestamp(self.read_watermark.load(Ordering::Acquire))
    }

    /// Drop a transaction's registration when it commits or aborts.
    pub(crate) fn release_snapshot(&self, id: TxnId, ts: Timestamp) {
        let mut shard = self.shard(id).snapshots.lock();
        // Removes one entry, not every equal one — several live transactions
        // may share this snapshot value.
        if let Some(at) = shard.iter().position(|&s| s == ts.raw()) {
            shard.swap_remove(at);
        }
    }

    /// Allocate a commit timestamp.
    ///
    /// Must be paired with [`Oracle::publish`], or the read watermark stops at
    /// this timestamp and every later snapshot freezes.
    pub(crate) fn begin_commit(&self) -> Timestamp {
        let ts = self.next_ts.fetch_add(1, Ordering::Relaxed);

        // The ring describes `RING` timestamps at once, so running further
        // ahead would overwrite a slot the watermark has not consumed. Waiting
        // here also drags the watermark along: a transaction that has taken a
        // timestamp is itself the hole everyone else is stuck behind, so it must
        // not block without trying to make progress.
        //
        // Reaching this needs `RING` commits installing simultaneously, so in
        // practice it never spins.
        while ts.saturating_sub(self.read_watermark.load(Ordering::Acquire)) >= RING as u64 {
            self.advance();
            std::hint::spin_loop();
        }
        Timestamp(ts)
    }

    /// Publish that `ts` is fully installed.
    ///
    /// Storing the completion is the whole obligation. Moving the watermark is
    /// left to whichever commit happens to fill the current hole, plus a
    /// catch-up in [`Oracle::begin_snapshot`].
    ///
    /// That split is sound because the watermark exists only to serve
    /// snapshots: if nobody is taking one, how far behind it has fallen is
    /// unobservable. So it is brought current where it is about to be read
    /// rather than on every commit.
    ///
    /// And it matters, because the watermark is a single word that every commit
    /// was fighting over. Only the commit completing `watermark + 1` can move
    /// anything; every other one ran a compare-exchange loop that failed and
    /// achieved nothing except invalidating that cache line for everybody else.
    /// Removing the loop entirely, as a throwaway experiment, took four-thread
    /// write throughput from 5.45M to 8.98M ops/sec — so one load and a
    /// comparison to skip it is worth having.
    pub(crate) fn publish(&self, ts: Timestamp) {
        self.completed[(ts.raw() & RING_MASK) as usize].store(ts.raw(), Ordering::Release);

        if self.read_watermark.load(Ordering::Acquire) + 1 == ts.raw() {
            self.advance();
        }
    }

    /// Move the read watermark over any contiguous run of completed commits.
    ///
    /// Racing callers are harmless: the watermark only moves forward and only
    /// over slots already marked complete, so a loser re-reads and continues. A
    /// completion that lands just after a scan has passed it is not lost — the
    /// next `publish` that fills a hole, or the next `begin_snapshot`, picks it
    /// up.
    fn advance(&self) {
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
    /// classic MVCC failure mode. See [`crate::engine::gc`].
    pub(crate) fn gc_watermark(&self) -> Timestamp {
        let oldest = self
            .active
            .iter()
            .filter_map(|s| s.snapshots.lock().iter().copied().min())
            .min();
        match oldest {
            Some(ts) => Timestamp(ts),
            None => Timestamp(self.read_watermark.load(Ordering::Acquire)),
        }
    }

    /// Number of live transactions, for [`crate::engine::gc::GcStats`].
    pub(crate) fn active_count(&self) -> usize {
        self.active.iter().map(|s| s.snapshots.lock().len()).sum()
    }

    /// The snapshot shard for `id`.
    ///
    /// Ids are `1 + slot + SHARDS * n`, so this recovers the allocating
    /// thread's slot and every transaction of a given thread lands in the same
    /// shard as its siblings and no one else's.
    fn shard(&self, id: TxnId) -> &ActiveShard {
        &self.active[(id.0 as usize) % SHARDS]
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
        assert!(
            o.gc_watermark() > old_reader,
            "releasing it lets GC advance"
        );
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
        let b = TxnId(1 + SHARDS as u64);
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

        // Every timestamp handed out was published, so a transaction beginning
        // now must see all of them. `begin_snapshot` is where the watermark is
        // brought current — a commit chases it only when it happens to fill the
        // current hole — so that is what this asks for.
        let next = o.next_ts.load(Ordering::Acquire);
        assert_eq!(
            o.begin_snapshot(TxnId(1)),
            Timestamp(next - 1),
            "watermark stalled below a fully published sequence"
        );
    }
}
