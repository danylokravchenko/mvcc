//! Isolation levels as typestates.
//!
//! The isolation level is a type parameter on `Transaction`, not a runtime enum,
//! so the cost of the strongest level does not leak into the weakest. Every
//! behavioural difference is a `const` on this trait, which means the engine's
//! branches on them fold away at compile time: a `ReadCommitted` transaction
//! records no read set, registers no SIREAD locks and allocates nothing, while a
//! `Serializable` one pays for exactly the machinery SSI needs.
//!
//! The four levels differ along two axes only:
//!
//! | level             | snapshot taken     | read tracking |
//! |-------------------|--------------------|---------------|
//! | [`ReadCommitted`] | per statement      | none          |
//! | [`RepeatableRead`]| once, at begin     | none          |
//! | [`Snapshot`]      | once, at begin     | none          |
//! | [`Serializable`]  | once, at begin     | read set + SIREAD locks (SSI) |
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
//! - `Serializable` — no anomalies, via SSI. The mechanism lives in
//!   `crate::engine::ssi`: a transaction aborts only when it has an
//!   rw-antidependency in both directions.

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
    /// A `const` so the engine's read path can branch on it and have the
    /// branch compiled away — the weaker levels emit no read-recording code,
    /// register no SIREAD locks, and allocate nothing.
    const VALIDATES_READS: bool;
}

macro_rules! define_level {
    (
        $(#[$meta:meta])*
        $name:ident {
            refresh: $refresh:expr,
            first_committer_wins: $fcw:expr,
            validates_reads: $validates:expr $(,)?
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::assertions_on_constants)]
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
