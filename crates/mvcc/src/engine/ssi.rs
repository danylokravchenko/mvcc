//! Serializable Snapshot Isolation: rw-antidependency tracking.
//!
//! # The rule
//!
//! Snapshot isolation's anomalies all share one structure (Fekete, Cahill et
//! al.): every cycle in the serialization graph contains a transaction with
//! *two consecutive rw-antidependency edges* — one coming in, one going out.
//! Such a transaction is a **pivot**. Aborting pivots is sufficient to
//! guarantee serializability, and it aborts far less than "abort if anything I
//! read changed", which is what this engine did before.
//!
//! An rw-antidependency `T →rw W` means: T read a version, and W overwrote it.
//! T must be ordered before W, even though W wrote second.
//!
//! # How each edge is found
//!
//! The two directions need different machinery, and this engine already had
//! half of it:
//!
//! - **Outgoing** (`self →rw W`): re-run the read set at commit. Anything that
//!   changed was changed by someone, and because revalidation reads at the
//!   current read watermark, that someone has necessarily *committed*. This is
//!   the check that used to abort on its own; now it only sets a flag.
//!
//! - **Incoming** (`R →rw self`): needs to know who read the rows this
//!   transaction wrote — a *SIREAD lock*. [`Slot`](crate::engine::store::Slot) keeps a
//!   list of transactions that read it, and [`Table`](crate::engine::store::Table)
//!   keeps registered predicates so an *insert* can be matched against reads of
//!   rows that did not exist yet. The second is what makes G2 detectable.
//!
//! # Retention
//!
//! A committed transaction's SIREAD locks cannot be dropped at commit: a
//! transaction that starts later but with an older snapshot can still form an
//! edge with it. They are held until the GC watermark passes the commit
//! timestamp — the same watermark that governs version reclamation — and purged
//! lazily whenever a list is walked.
//!
//! # Verification status
//!
//! Verified empirically against the Hermitage suite (`tests/hermitage.rs`) and
//! the anomaly suite, including tests that assert the *absence* of false
//! aborts. Not formally proven. Postgres's SSI took years to harden and this is
//! a far smaller implementation; treat `Serializable` as well-tested rather
//! than as proven.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::core::{Timestamp, TxnId};

/// Shared, concurrently-updatable state for one transaction.
///
/// Lives behind an `Arc` because other transactions set flags on it: a writer
/// marks its readers, and a reader marks the writer it conflicts with. It
/// outlives the `Transaction` itself, since SIREAD locks and version records
/// keep referring to it after commit.
#[derive(Debug)]
pub(crate) struct TxnState {
    id: TxnId,
    /// A concurrent transaction overwrote something this one read.
    out_conflict: AtomicBool,
    /// A concurrent transaction read something this one overwrote.
    in_conflict: AtomicBool,
    /// Commit timestamp, or 0 while running.
    committed_at: AtomicU64,
    aborted: AtomicBool,
}

impl TxnState {
    pub(crate) fn new(id: TxnId) -> Arc<Self> {
        Arc::new(TxnState {
            id,
            out_conflict: AtomicBool::new(false),
            in_conflict: AtomicBool::new(false),
            committed_at: AtomicU64::new(0),
            aborted: AtomicBool::new(false),
        })
    }

    pub(crate) fn id(&self) -> TxnId {
        self.id
    }

    pub(crate) fn set_out_conflict(&self) {
        self.out_conflict.store(true, Ordering::Release);
    }

    pub(crate) fn set_in_conflict(&self) {
        self.in_conflict.store(true, Ordering::Release);
    }

    /// Whether a concurrent transaction overwrote something this one read.
    pub(crate) fn has_out_conflict(&self) -> bool {
        self.out_conflict.load(Ordering::Acquire)
    }

    /// Both edges present: this transaction is a pivot and cannot be allowed to
    /// commit.
    pub(crate) fn is_pivot(&self) -> bool {
        self.in_conflict.load(Ordering::Acquire) && self.out_conflict.load(Ordering::Acquire)
    }

    pub(crate) fn is_committed(&self) -> bool {
        self.committed_at.load(Ordering::Acquire) != 0
    }

    pub(crate) fn is_aborted(&self) -> bool {
        self.aborted.load(Ordering::Acquire)
    }

    pub(crate) fn mark_committed(&self, ts: Timestamp) {
        self.committed_at.store(ts.raw(), Ordering::Release);
    }

    pub(crate) fn mark_aborted(&self) {
        self.aborted.store(true, Ordering::Release);
    }

    /// Whether this transaction's SIREAD locks can be discarded.
    ///
    /// An aborted transaction's reads never mattered. A committed one's stop
    /// mattering once no live snapshot predates its commit, which is exactly
    /// what the GC watermark tracks.
    pub(crate) fn is_expired(&self, gc_watermark: Timestamp) -> bool {
        if self.is_aborted() {
            return true;
        }
        match self.committed_at.load(Ordering::Acquire) {
            0 => false, // still running
            ts => Timestamp(ts) <= gc_watermark,
        }
    }
}

/// The set of transactions holding a SIREAD lock on one slot.
///
/// Deliberately a `Vec`: read sets per slot are almost always tiny, and a
/// linear scan over a handful of `Arc`s beats a hash set's allocation.
#[derive(Default, Debug)]
pub(crate) struct Readers(Vec<Arc<TxnState>>);

impl Readers {
    /// Register `state` as having read this slot, if it is not already there.
    pub(crate) fn register(&mut self, state: &Arc<TxnState>) {
        if !self.0.iter().any(|r| Arc::ptr_eq(r, state)) {
            self.0.push(Arc::clone(state));
        }
    }

    /// Drop expired entries, then return every live reader other than `writer`.
    ///
    /// Purging happens here rather than on a timer because this is the only
    /// place the list is walked, so the work is already paid for.
    pub(crate) fn others(&mut self, writer: TxnId, gc_watermark: Timestamp) -> Vec<Arc<TxnState>> {
        self.0.retain(|r| !r.is_expired(gc_watermark));
        self.0.iter().filter(|r| r.id != writer).cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(id: u64) -> Arc<TxnState> {
        TxnState::new(TxnId(id))
    }

    /// Live readers other than `writer`. `others` is the only accessor the
    /// engine uses, so the tests go through it too rather than through helpers
    /// that exist for their benefit alone.
    fn live(readers: &mut Readers, writer: TxnId) -> Vec<Arc<TxnState>> {
        readers.others(writer, Timestamp(0))
    }

    #[test]
    fn a_transaction_is_a_pivot_only_with_both_edges() {
        let t = state(1);
        assert!(!t.is_pivot());
        t.set_out_conflict();
        assert!(!t.is_pivot(), "an outgoing edge alone is not a cycle");
        t.set_in_conflict();
        assert!(t.is_pivot());
    }

    #[test]
    fn committed_readers_expire_only_once_the_watermark_passes() {
        let t = state(1);
        assert!(!t.is_expired(Timestamp(100)), "still running");

        t.mark_committed(Timestamp(50));
        assert!(
            !t.is_expired(Timestamp(49)),
            "a live snapshot still predates it"
        );
        assert!(t.is_expired(Timestamp(50)));
        assert!(t.is_expired(Timestamp(51)));
    }

    #[test]
    fn aborted_readers_expire_immediately() {
        let t = state(1);
        t.mark_aborted();
        assert!(t.is_expired(Timestamp(0)));
    }

    #[test]
    fn registration_is_idempotent_and_excludes_the_writer() {
        let mut readers = Readers::default();
        let a = state(1);
        let b = state(2);
        assert!(
            live(&mut readers, TxnId::NONE).is_empty(),
            "no readers initially"
        );
        readers.register(&a);
        readers.register(&a);
        readers.register(&b);
        assert_eq!(
            live(&mut readers, TxnId::NONE).len(),
            2,
            "duplicate registration"
        );

        assert!(!live(&mut readers, TxnId(1)).is_empty(), "b reads it");
        assert!(!live(&mut readers, TxnId(3)).is_empty());

        // With b expired, only the writer's own read remains.
        b.mark_aborted();
        assert!(live(&mut readers, TxnId(1)).is_empty());
    }
}
