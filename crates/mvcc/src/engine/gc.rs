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

use crate::core::Timestamp;

#[derive(Clone, Copy, Debug)]
pub struct GcStats {
    /// Versions unlinked from chains but not yet freed. Not tracked yet —
    /// crossbeam's deferred destruction owns this queue and does not expose its
    /// depth, so this reads 0.
    pub pending_reclaim: u64,
    /// Versions freed since start. Not tracked yet, for the same reason.
    pub reclaimed_total: u64,
    /// Current watermark. If this is not advancing, nothing else matters.
    pub watermark: Timestamp,
    /// Live transactions. A number that only grows is the leak.
    pub active_transactions: usize,
}
