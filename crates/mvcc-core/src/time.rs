//! Logical time: transaction ids, commit timestamps, and log sequence numbers.
//!
//! There is exactly one logical clock in the system, a monotonically increasing
//! `u64` handed out by the engine's timestamp oracle. Both transaction ids and
//! commit timestamps are drawn from it, which is what lets a single `u64` field
//! on a version record encode "this version was created by in-flight txn N" and
//! "this version became visible at time T" without a discriminant.
//!
//! The high bit is the tag:
//!
//! ```text
//!   0b0_xxxx…xxx   committed, value is a commit timestamp
//!   0b1_xxxx…xxx   in flight, low 63 bits are the creating transaction id
//! ```

/// Marks a [`Timestamp`] slot as holding an in-flight transaction id.
const IN_FLIGHT: u64 = 1 << 63;
const ID_MASK: u64 = !IN_FLIGHT;

/// A point on the logical commit timeline.
///
/// Snapshots are taken as a single `Timestamp`; a version is visible to a
/// snapshot `s` if `begin <= s < end`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Timestamp(pub u64);

impl Timestamp {
    /// Before every possible commit. Used as `begin` for bootstrap data.
    pub const ZERO: Timestamp = Timestamp(0);
    /// After every possible commit. Used as `end` for the current version.
    pub const MAX: Timestamp = Timestamp(ID_MASK);

    #[inline]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Identifies a running transaction. Drawn from the same counter as
/// [`Timestamp`] so that ids and timestamps never collide.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct TxnId(pub u64);

impl TxnId {
    /// A id belonging to no transaction.
    ///
    /// Reading with this id makes every in-flight version invisible, including
    /// the reader's own — which is what commit-time read validation wants: it
    /// must compare against what *other* transactions have committed, not
    /// against its own uncommitted writes. `0` is never handed out, because the
    /// oracle's counter starts at 1.
    pub const NONE: TxnId = TxnId(0);

    /// Encode this id into the tagged representation stored in a version's
    /// `begin`/`end` field while the transaction is still in flight.
    #[inline]
    pub const fn tagged(self) -> u64 {
        self.0 | IN_FLIGHT
    }
}

/// Decoded form of a version's tagged `begin`/`end` field.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Visibility {
    /// The version was committed at this timestamp.
    CommittedAt(Timestamp),
    /// The version is still being written by this transaction. Only that
    /// transaction may read it; everyone else must follow the version chain
    /// to the previous record.
    InFlight(TxnId),
}

impl Visibility {
    /// Decode a raw tagged field read out of a version record.
    #[inline]
    pub const fn decode(raw: u64) -> Visibility {
        if raw & IN_FLIGHT != 0 {
            Visibility::InFlight(TxnId(raw & ID_MASK))
        } else {
            Visibility::CommittedAt(Timestamp(raw))
        }
    }

    /// Whether a reader at `snapshot` belonging to transaction `reader` should
    /// treat this endpoint as "already passed".
    ///
    /// This is the single visibility predicate for the whole engine: a version
    /// is visible when `begin.reached(..) && !end.reached(..)`. Keeping it in
    /// one place is what makes the four isolation levels differ only in *which
    /// snapshot they pass in*, not in how visibility is computed.
    #[inline]
    pub fn reached(self, snapshot: Timestamp, reader: TxnId) -> bool {
        match self {
            Visibility::CommittedAt(ts) => ts <= snapshot,
            Visibility::InFlight(id) => id == reader,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tagged_txn_ids_round_trip() {
        let id = TxnId(12345);
        assert_eq!(Visibility::decode(id.tagged()), Visibility::InFlight(id));
    }

    #[test]
    fn commit_timestamps_round_trip() {
        let ts = Timestamp(999);
        assert_eq!(Visibility::decode(ts.raw()), Visibility::CommittedAt(ts));
    }

    #[test]
    fn own_writes_are_visible_to_their_author() {
        let me = TxnId(7);
        let someone_else = TxnId(8);
        let v = Visibility::decode(me.tagged());
        assert!(v.reached(Timestamp::ZERO, me));
        assert!(!v.reached(Timestamp::MAX, someone_else));
    }

    #[test]
    fn max_timestamp_does_not_collide_with_in_flight_tag() {
        assert!(matches!(
            Visibility::decode(Timestamp::MAX.raw()),
            Visibility::CommittedAt(_)
        ));
    }
}
