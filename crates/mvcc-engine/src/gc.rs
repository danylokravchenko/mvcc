//! Version reclamation.
//!
//! Two problems, often conflated:
//!
//! 1. **When is a version logically dead?** When `end < gc_watermark`, i.e. no
//!    live transaction's snapshot can reach it.
//! 2. **When is it safe to free the memory?** Later — a reader may already hold
//!    a pointer to it, obtained before the watermark moved.
//!
//! (1) is MVCC bookkeeping; (2) is a memory-reclamation problem. Today the
//! `Arc`-based chain in [`crate::store`] answers (2) for free: a version is
//! dropped when the last `Ref` to it goes away. That is the correct-first
//! implementation, and it costs an atomic increment per read — a write to a
//! shared cache line on the read path, which is exactly what MVCC exists to
//! avoid.
//!
//! The planned replacement is epoch-based reclamation: a reader publishes its
//! epoch once on entry, and freeing waits until every thread has moved past the
//! epoch in which a version was retired. Reference counting each version read
//! is the alternative, and it is the wrong one for the same reason — it puts an
//! atomic RMW on the read path.
//!
//! # The failure mode to instrument
//!
//! Reclamation is bounded below by the oldest live snapshot. One forgotten
//! transaction — a REPL session, a leaked handle, an analytics query — pins the
//! watermark and version chains grow without limit. This is the most common way
//! a real MVCC system falls over, and it presents as a memory leak rather than
//! as a transaction problem.
//!
//! Watch [`GcStats::active_transactions`] and [`GcStats::watermark`]: a
//! watermark that stops advancing while writes continue is the signal.

use mvcc_core::Timestamp;

#[derive(Clone, Copy, Debug)]
pub struct GcStats {
    /// Versions unlinked from chains but not yet freed. Always 0 while
    /// reclamation is `Arc`-driven.
    pub pending_reclaim: u64,
    /// Versions freed since start. Not tracked yet.
    pub reclaimed_total: u64,
    /// Current watermark. If this is not advancing, nothing else matters.
    pub watermark: Timestamp,
    /// Live transactions. A number that only grows is the leak.
    pub active_transactions: usize,
}

/// Background reclamation.
///
/// Not yet implemented: version lifetime is currently handled by `Arc`. This
/// becomes real alongside the lock-free chain migration, and will run per shard
/// so collection touches only that shard's cache lines.
pub struct Collector {
    _private: (),
}
