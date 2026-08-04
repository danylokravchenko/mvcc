//! Isolation levels as typestates.
//!
//! The isolation level is a type parameter on `Transaction`, not a runtime enum.
//! Each level supplies a [`ReadTracker`] associated type; the weaker levels use
//! [`NoTracking`], a zero-sized type whose methods are empty. A Read Committed
//! transaction therefore contains no read-set field and emits no read-tracking
//! instructions at all, while a Serializable transaction pays for exactly the
//! bookkeeping its algorithm requires.
//!
//! The four levels differ along two axes only:
//!
//! | level             | snapshot taken     | read tracking |
//! |-------------------|--------------------|---------------|
//! | [`ReadCommitted`] | per statement      | none          |
//! | [`RepeatableRead`]| once, at begin     | none          |
//! | [`Snapshot`]      | once, at begin     | none          |
//! | [`Serializable`]  | once, at begin     | read set, revalidated at commit |
//!
//! [`RepeatableRead`] and [`Snapshot`] are the same mechanism here — under MVCC
//! a repeatable-read snapshot *is* a snapshot — but they are distinct types so
//! that `RepeatableRead` can be given phantom-read semantics later without a
//! breaking change, and so user code documents its intent.
//!
//! # What each level admits
//!
//! - `ReadCommitted` — no dirty reads; non-repeatable reads and phantoms allowed.
//! - `RepeatableRead` / `Snapshot` — no non-repeatable reads; **write skew is
//!   allowed**. This is the classic snapshot-isolation anomaly and the reason
//!   `Serializable` exists.
//! - `Serializable` — no anomalies. Currently enforced by revalidating the read
//!   set at commit; SSI's pivot detection is the planned refinement. See
//!   `DESIGN.md` §4.

use crate::time::{Timestamp, TxnId};

mod sealed {
    pub trait Sealed {}
}

/// A transaction isolation level.
///
/// Sealed: implementing a new level requires engine support, so this trait is
/// not extensible from outside the crate.
pub trait IsolationLevel: sealed::Sealed + Copy + Default + Send + Sync + 'static {
    /// Human-readable name, used in traces and error messages.
    const NAME: &'static str;

    /// Whether a fresh snapshot is taken before each statement rather than once
    /// at `begin`.
    const REFRESH_SNAPSHOT_PER_STATEMENT: bool;

    /// Whether the engine must check for write-write conflicts against versions
    /// committed *after* this transaction's snapshot. False only for
    /// `ReadCommitted`, which re-reads the latest version before writing.
    const FIRST_COMMITTER_WINS: bool;

    /// Whether reads are recorded and re-checked at commit. True only for
    /// `Serializable`.
    ///
    /// A `const` rather than a property of the tracker so that the engine's
    /// read path can branch on it and have the branch compiled away — the
    /// weaker levels emit no read-recording code at all.
    const VALIDATES_READS: bool;

    /// Per-transaction read bookkeeping. `NoTracking` for every level except
    /// `Serializable`.
    type ReadTracker: ReadTracker;
}

/// Read-set bookkeeping hooked into every visible-version read.
///
/// The engine calls [`ReadTracker::observe`] on each read and
/// [`ReadTracker::validate`] at commit. The [`NoTracking`] implementation
/// compiles to nothing.
pub trait ReadTracker: Default + Send + Sync + 'static {
    /// Record that `reader` observed the version of `slot` valid at `snapshot`.
    fn observe(&mut self, slot: SlotRef, snapshot: Timestamp);

    /// Record that a concurrent transaction overwrote something we read
    /// (an rw-antidependency *into* us).
    fn note_inbound_conflict(&mut self, writer: TxnId);

    /// Record that we overwrote something a concurrent transaction read
    /// (an rw-antidependency *out of* us).
    fn note_outbound_conflict(&mut self, reader: TxnId);

    /// Called at commit. `false` means abort with
    /// [`crate::Error::SerializationFailure`].
    fn validate(&self) -> bool;
}

/// Opaque handle to a version slot, used as the identity of a read.
///
/// Carries the slot's address rather than the key so that read tracking costs
/// one word regardless of key size.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SlotRef(pub usize);

/// The no-op read tracker used by every level below `Serializable`.
#[derive(Default, Clone, Copy, Debug)]
pub struct NoTracking;

impl ReadTracker for NoTracking {
    #[inline(always)]
    fn observe(&mut self, _slot: SlotRef, _snapshot: Timestamp) {}
    #[inline(always)]
    fn note_inbound_conflict(&mut self, _writer: TxnId) {}
    #[inline(always)]
    fn note_outbound_conflict(&mut self, _reader: TxnId) {}
    #[inline(always)]
    fn validate(&self) -> bool {
        true
    }
}

/// Read-set tracking for Serializable Snapshot Isolation (Cahill et al.).
///
/// SSI runs snapshot isolation and additionally watches for the one structure
/// that every serialization anomaly under SI contains: a transaction that has
/// an rw-antidependency both coming in and going out — a *pivot*. Aborting
/// pivots is sufficient to guarantee serializability, at the cost of some false
/// positives (transactions aborted that would in fact have been safe).
#[derive(Default, Debug)]
pub struct SsiTracker {
    /// Slots this transaction read, for writers to consult.
    reads: Vec<(SlotRef, Timestamp)>,
    /// A concurrent transaction wrote something we read.
    in_conflict: bool,
    /// We wrote something a concurrent transaction read.
    out_conflict: bool,
}

impl SsiTracker {
    /// The slots read so far, for the engine to register with the conflict
    /// manager at commit.
    pub fn reads(&self) -> &[(SlotRef, Timestamp)] {
        &self.reads
    }

    /// True when this transaction is a pivot in a dangerous structure.
    pub fn is_pivot(&self) -> bool {
        self.in_conflict && self.out_conflict
    }
}

impl ReadTracker for SsiTracker {
    fn observe(&mut self, slot: SlotRef, snapshot: Timestamp) {
        self.reads.push((slot, snapshot));
    }

    fn note_inbound_conflict(&mut self, _writer: TxnId) {
        self.in_conflict = true;
    }

    fn note_outbound_conflict(&mut self, _reader: TxnId) {
        self.out_conflict = true;
    }

    fn validate(&self) -> bool {
        !self.is_pivot()
    }
}

macro_rules! define_level {
    (
        $(#[$meta:meta])*
        $name:ident {
            refresh: $refresh:expr,
            first_committer_wins: $fcw:expr,
            validates_reads: $validates:expr,
            tracker: $tracker:ty $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Default, Debug)]
        pub struct $name;

        impl sealed::Sealed for $name {}

        impl IsolationLevel for $name {
            const NAME: &'static str = stringify!($name);
            const REFRESH_SNAPSHOT_PER_STATEMENT: bool = $refresh;
            const FIRST_COMMITTER_WINS: bool = $fcw;
            const VALIDATES_READS: bool = $validates;
            type ReadTracker = $tracker;
        }
    };
}

define_level! {
    /// Each statement sees everything committed before that statement started.
    ///
    /// Cheapest level: no read set, and a writer never aborts on a stale
    /// snapshot because it re-reads the latest version first. Non-repeatable
    /// reads and phantoms are visible to the application.
    ReadCommitted {
        refresh: true,
        first_committer_wins: false,
        validates_reads: false,
        tracker: NoTracking,
    }
}

define_level! {
    /// A single snapshot for the whole transaction; repeated reads of the same
    /// key return the same value.
    ///
    /// Under MVCC this is implemented identically to [`Snapshot`]; it exists as
    /// a separate type to document intent and to leave room for SQL-standard
    /// phantom semantics later.
    RepeatableRead {
        refresh: false,
        first_committer_wins: true,
        validates_reads: false,
        tracker: NoTracking,
    }
}

define_level! {
    /// Full snapshot isolation. The default.
    ///
    /// Reads never block and never abort. Writes abort with
    /// [`crate::Error::WriteConflict`] if another transaction committed a change
    /// to the same key after this transaction's snapshot (first-committer-wins).
    /// Permits write skew.
    Snapshot {
        refresh: false,
        first_committer_wins: true,
        validates_reads: false,
        tracker: NoTracking,
    }
}

define_level! {
    /// Serializable, via SSI on top of snapshot isolation.
    ///
    /// Adds read-set tracking and rw-antidependency detection. Transactions may
    /// abort with [`crate::Error::SerializationFailure`] even when no two of
    /// them touched the same key for writing — retry is expected.
    Serializable {
        refresh: false,
        first_committer_wins: true,
        validates_reads: true,
        tracker: SsiTracker,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_tracking_is_zero_sized() {
        assert_eq!(size_of::<NoTracking>(), 0);
    }

    #[test]
    fn ssi_aborts_only_pivots() {
        let mut t = SsiTracker::default();
        assert!(t.validate());

        t.note_inbound_conflict(TxnId(1));
        assert!(t.validate(), "an inbound edge alone is not a cycle");

        t.note_outbound_conflict(TxnId(2));
        assert!(!t.validate(), "in + out is a dangerous structure");
    }

    #[test]
    fn the_levels_differ_only_in_their_declared_constants() {
        // The engine branches on these, so they are the whole behavioural
        // difference between the levels. Asserting them here means a typo in a
        // `define_level!` invocation fails a test rather than silently
        // downgrading someone's isolation.
        assert!(ReadCommitted::REFRESH_SNAPSHOT_PER_STATEMENT);
        assert!(!Snapshot::REFRESH_SNAPSHOT_PER_STATEMENT);
        assert!(!RepeatableRead::REFRESH_SNAPSHOT_PER_STATEMENT);
        assert!(!Serializable::REFRESH_SNAPSHOT_PER_STATEMENT);

        assert!(!ReadCommitted::FIRST_COMMITTER_WINS);
        assert!(Snapshot::FIRST_COMMITTER_WINS);
        assert!(Serializable::FIRST_COMMITTER_WINS);

        assert!(!Snapshot::VALIDATES_READS);
        assert!(!RepeatableRead::VALIDATES_READS);
        assert!(
            Serializable::VALIDATES_READS,
            "this is what stops write skew"
        );
    }
}
