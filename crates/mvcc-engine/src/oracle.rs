//! The timestamp oracle — and the single most important scalability decision in
//! the engine.
//!
//! Every transaction takes a timestamp at begin and another at commit. The
//! obvious implementation is one global `AtomicU64::fetch_add`. That is also the
//! thing that will cap this engine's throughput long before the disk does: a
//! contended cache line must be acquired exclusively by each core, and the cost
//! *grows* with core count rather than staying flat.
//!
//! [`OracleConfig`] documents the escape routes. Only `Centralised` is
//! implemented so far — see the build order in `DESIGN.md`.
//!
//! # Why `parking_lot` here
//!
//! The two `Mutex<BTreeSet<u64>>` below are the hottest contended locks in the
//! engine: every transaction touches `active` twice (begin and release) and
//! every commit touches `installing` twice. Their critical sections are tiny —
//! one `u64` insert or remove into a small `BTreeSet`, on the order of tens of
//! nanoseconds.
//!
//! That is precisely the shape where `std::sync::Mutex` does badly. It parks a
//! contended waiter in the kernel almost immediately, and a kernel round trip
//! costs far more than the critical section it is waiting on. `parking_lot`
//! spins briefly before parking, so a waiter usually acquires the lock without
//! ever descending into the kernel.
//!
//! Measured on `examples/bench.rs`, 8 threads (medians of 3):
//!
//! ```text
//!                                    std      parking_lot
//!   point reads (100 per txn)      3.08M/s      3.19M/s     +4%
//!   read-only txns (1 read each)   1.77M/s      3.47M/s    +96%
//!   write txns                     0.87M/s      1.36M/s    +56%
//! ```
//!
//! The point-read row is the control: it amortises the oracle over 100 reads,
//! so it barely moves. Nearly all of the win is these two mutexes.
//!
//! **This is a mitigation, not a fix.** Making a contended lock cheaper is not
//! the same as not contending. Both sets are per-transaction global state, and
//! removing them is what `OracleConfig::Batched` and `OracleConfig::Epoch` are
//! for. `parking_lot` buys roughly 2x; deleting the sets buys the rest.

use parking_lot::Mutex;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};

use mvcc_core::{Timestamp, TxnId};

/// How timestamps are handed out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OracleConfig {
    /// One global atomic counter.
    ///
    /// Simple, exact, and totally ordered. Correct at any core count and the
    /// right default up to roughly a socket's worth of cores.
    #[default]
    Centralised,

    /// Each thread claims a block of `stride` timestamps with one `fetch_add`
    /// and hands them out locally. Cuts coherence traffic by `stride`, at the
    /// cost of timestamps no longer being densely allocated in commit order.
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

/// Hands out begin and commit timestamps and maintains the two watermarks.
pub struct Oracle {
    config: OracleConfig,
    /// Source of both transaction ids and commit timestamps, so the two can
    /// never collide in a version's tagged `begin` field.
    next: AtomicU64,
    /// Highest timestamp below which every commit is fully installed.
    read_watermark: AtomicU64,
    /// Commit timestamps allocated but not yet fully installed.
    installing: Mutex<BTreeSet<u64>>,
    /// Snapshots of live transactions, for the GC watermark.
    active: Mutex<BTreeSet<u64>>,
}

impl Oracle {
    pub fn new(config: OracleConfig) -> Self {
        Oracle {
            config,
            // Start at 1: timestamp 0 means "before everything", and is used as
            // the `begin` of bootstrap data.
            next: AtomicU64::new(1),
            read_watermark: AtomicU64::new(0),
            installing: Mutex::new(BTreeSet::new()),
            active: Mutex::new(BTreeSet::new()),
        }
    }

    pub fn config(&self) -> OracleConfig {
        self.config
    }

    /// Restart the counter above `ts`, after recovery has replayed the log.
    ///
    /// Must not reuse a timestamp: a recovered transaction sharing a timestamp
    /// with a new one would be invisible to a snapshot taken between them.
    pub fn resume_after(&self, ts: Timestamp) {
        let next = ts.raw() + 1;
        self.next.store(next, Ordering::Release);
        self.read_watermark.store(ts.raw(), Ordering::Release);
    }

    /// Allocate a transaction id.
    pub fn next_txn_id(&self) -> TxnId {
        TxnId(self.next.fetch_add(1, Ordering::Relaxed))
    }

    /// Take a snapshot for a beginning transaction, and register it as active.
    ///
    /// Returns the *read watermark*, not the raw counter. Using the counter
    /// would let a transaction see a commit timestamp whose versions are still
    /// being stamped, and so observe a torn commit.
    pub fn begin_snapshot(&self) -> Timestamp {
        let mut active = self.active.lock();
        let ts = Timestamp(self.read_watermark.load(Ordering::Acquire));
        active.insert(ts.raw());
        ts
    }

    /// A snapshot for a statement in a `ReadCommitted` transaction. Does not
    /// register — the transaction's begin snapshot already pins the watermark.
    pub fn statement_snapshot(&self) -> Timestamp {
        Timestamp(self.read_watermark.load(Ordering::Acquire))
    }

    /// Drop a transaction's registration when it commits or aborts.
    pub fn release_snapshot(&self, ts: Timestamp) {
        let mut active = self.active.lock();
        active.remove(&ts.raw());
    }

    /// Allocate a commit timestamp and mark it as installing.
    ///
    /// Must be paired with [`Oracle::publish`], or the read watermark stops
    /// advancing and every later snapshot is frozen.
    pub fn begin_commit(&self) -> Timestamp {
        let mut installing = self.installing.lock();
        let ts = Timestamp(self.next.fetch_add(1, Ordering::Relaxed));
        installing.insert(ts.raw());
        ts
    }

    /// Publish that `ts` is fully installed and advance the read watermark.
    ///
    /// The watermark can only pass a timestamp once every *lower* commit has
    /// also finished installing, so this is a min-reduction over the in-flight
    /// set rather than a plain store. Committing out of order is therefore
    /// safe: an early finisher does not expose a later-numbered commit.
    pub fn publish(&self, ts: Timestamp) {
        let mut installing = self.installing.lock();
        installing.remove(&ts.raw());

        // Everything below the lowest still-installing commit is settled. With
        // nothing installing, everything allocated so far is settled.
        let settled = match installing.first() {
            Some(&lowest) => lowest - 1,
            None => self.next.load(Ordering::Relaxed) - 1,
        };
        self.read_watermark.fetch_max(settled, Ordering::Release);
    }

    /// The oldest snapshot any live transaction can still see. Versions whose
    /// `end` is at or below this are unreachable and may be reclaimed.
    ///
    /// A single long-running reader pins this and stalls all reclamation — the
    /// classic MVCC failure mode. See [`crate::gc`].
    pub fn gc_watermark(&self) -> Timestamp {
        let active = self.active.lock();
        match active.first() {
            Some(&oldest) => Timestamp(oldest),
            None => Timestamp(self.read_watermark.load(Ordering::Acquire)),
        }
    }

    /// Number of live transactions, for [`crate::gc::GcStats`].
    pub fn active_count(&self) -> usize {
        self.active.lock().len()
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
    fn gc_watermark_is_pinned_by_the_oldest_reader() {
        let o = Oracle::new(OracleConfig::Centralised);
        let ts = o.begin_commit();
        o.publish(ts);

        let old_reader = o.begin_snapshot();
        let commit = o.begin_commit();
        o.publish(commit);
        let _new_reader = o.begin_snapshot();

        assert_eq!(o.gc_watermark(), old_reader, "the oldest reader pins GC");

        o.release_snapshot(old_reader);
        assert!(
            o.gc_watermark() > old_reader,
            "releasing it lets GC advance"
        );
    }

    #[test]
    fn txn_ids_and_timestamps_never_collide() {
        let o = Oracle::new(OracleConfig::Centralised);
        let id = o.next_txn_id();
        let ts = o.begin_commit();
        assert_ne!(id.0, ts.raw());
    }
}
