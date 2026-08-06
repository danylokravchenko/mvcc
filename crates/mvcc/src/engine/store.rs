//! Version storage.
//!
//! # Layout: newest in place, older versions behind it
//!
//! ```text
//!   index ──► Slot ────────────────────────────────┐
//!             ├ latest ──────────────────────────► Version { begin, end, prev, value }  v3 current
//!             └ lock                                        │
//!                                                           ▼
//!                                                 Version { … }  v2
//!                                                           │
//!                                                           ▼
//!                                                 Version { … }  v1 oldest live
//! ```
//!
//! The current version is one hop from the index; older versions hang off it,
//! newest to oldest. Chosen over Postgres-style append-only because OLTP reads
//! overwhelmingly want the newest version, and because the slot address is
//! stable — an update touches only the indexes whose key actually changed.
//!
//! # Implementation status
//!
//! The chain is lock-free. `Slot::latest` and `Version::prev` are epoch-managed
//! `Atomic`s, so following a chain is plain pointer loads and a visibility
//! comparison — no lock, no reference count, no write to shared memory. A
//! transaction pins one epoch for its whole life, which is what lets a `Ref`
//! borrow a version rather than count it.
//!
//! The map in front of it is lock-free too. [`SlotMap`] resolves a primary key
//! to a slot by probing an epoch-managed bucket array, so **a point read now
//! performs no write to shared memory at any point on the path** — which is the
//! property this whole design is built around, and it took removing two things
//! to get: the `Arc` refcount on the chain, and the shard `RwLock` in front of
//! it, whose *read* acquire was still an atomic read-modify-write.
//!
//! Measured, not assumed, at four threads:
//!
//! ```text
//!                            Arc + RwLock    sharded RwLock    lock-free
//!   point reads, uniform          3.7M           111.9M          221.2M
//!   point reads, 4 hot rows          —            49.0M          475.2M
//! ```
//!
//! The middle column is where sharding had hidden the problem for uniform keys
//! and could not for hot ones: four keys land in at most four shards however
//! many shards there are, so that workload was the only one in the benchmark
//! that went *backwards* with more threads. See `crate::engine::slotmap`.
//!
//! What a read still touches: **`Database::tables`**, one registry `RwLock`,
//! amortised by the transaction's `table_cache` to about once per table per
//! transaction.
//!
//! And what only writers touch: the `RwLock` on each secondary index (taken
//! exclusively, but only by writes that actually change that index's key — see
//! [`Table::index_record`]), `Table::predicate_locks`, and `Slot::readers`,
//! which stays empty below `Serializable`.
//!
//! # Locks
//!
//! `parking_lot` rather than `std::sync`, for the reasons measured in
//! `crate::engine::oracle`. The `RwLock`s in this file benefit far less than the oracle's
//! mutexes do — their contention is spread across many slots rather than
//! concentrated on one — but using one lock type throughout is worth more than
//! the handful of bytes a split would save.
//!
//! Dropping poisoning costs nothing here: **no user code runs while an engine
//! lock is held.** The closure passed to `Transaction::update` is called before
//! any lock is taken, and index key extraction runs on cloned candidates rather
//! than under the index lock. There is no panic path that could leave a chain
//! torn, so there was never anything for poisoning to protect — the 18 sites it
//! required were pure noise.

use parking_lot::{Mutex, RwLock};
use std::any::{Any, TypeId};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Bound;
use std::sync::Arc;
use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};

use crate::core::{
    Error, IndexKey, IsolationLevel, Result, Snapshot as SnapshotLevel, TableId, Timestamp, TxnId,
    Versioned, Visibility,
};

use crossbeam_epoch::{Atomic, Guard, Shared};

use crate::engine::oracle::{Oracle, OracleConfig};
use crate::engine::slotmap::SlotMap;
use crate::engine::ssi::{Readers, TxnState};
use crate::engine::txn::Transaction;

/// One version of a record.
///
/// `value` is `None` for a tombstone: a delete installs a version like any
/// other write, so that a reader at an older snapshot still finds the record
/// alive behind it.
pub(crate) struct Version<T> {
    pub(crate) begin: AtomicU64,
    pub(crate) end: AtomicU64,
    /// The version this one displaced. Written once at construction.
    pub(crate) prev: Atomic<Version<T>>,
    pub(crate) value: Option<T>,
    /// The transaction that created this version.
    ///
    /// Kept so that a reader discovering at commit that its read set changed
    /// can name the transaction that changed it, and mark that transaction's
    /// incoming edge. `None` only for versions that predate the process, of
    /// which there are currently none.
    pub(crate) writer: Option<Arc<TxnState>>,
}

impl<T> Version<T> {
    /// Whether this version is the one visible at `snapshot` to `reader`.
    pub(crate) fn visible_to(&self, snapshot: Timestamp, reader: TxnId) -> bool {
        let begin = Visibility::decode(self.begin.load(Ordering::Acquire));
        let end = Visibility::decode(self.end.load(Ordering::Acquire));
        begin.reached(snapshot, reader) && !end.reached(snapshot, reader)
    }
}

/// The stable identity of a logical record. Index entries point here, so they
/// survive every update.
#[repr(align(64))] // own cache line: `lock` is contended
pub(crate) struct Slot<T> {
    /// Head of the version chain.
    ///
    /// An epoch-managed `Atomic`, so a reader follows the chain by plain
    /// pointer loads: no lock, no reference count, no write to shared memory of
    /// any kind. Writes are already serialised by `lock` below, so the only job
    /// left is making the load atomic against a concurrent store.
    ///
    /// The reference count this replaced was the last shared write on the read
    /// path, and it was expensive out of all proportion to its size. Isolated,
    /// four threads reading four hot records ran 37x faster through a plain
    /// reference than through `Arc::clone` — every read was dirtying a cache
    /// line that every other reader of that record needed.
    pub(crate) latest: Atomic<Version<T>>,
    /// Transaction id currently writing this slot, or 0 if free.
    pub(crate) lock: AtomicU64,
    /// SIREAD locks: transactions that have read this slot and may still form
    /// an rw-antidependency with a future writer. Only `Serializable`
    /// transactions ever register, so this stays empty under other levels.
    pub(crate) readers: Mutex<Readers>,
}

impl<T> Slot<T> {
    pub(crate) fn new() -> Self {
        Slot {
            latest: Atomic::null(),
            lock: AtomicU64::new(0),
            readers: Mutex::new(Readers::default()),
        }
    }

    /// Walk the chain for the version visible at `snapshot`.
    ///
    /// Returns a borrow valid for as long as `guard` is pinned — which, for a
    /// transaction, is its whole life. Nothing is cloned and nothing shared is
    /// written.
    pub(crate) fn read<'g>(
        &self,
        snapshot: Timestamp,
        reader: TxnId,
        guard: &'g Guard,
    ) -> Option<&'g Version<T>> {
        let mut cur = self.latest.load(Ordering::Acquire, guard);
        loop {
            // SAFETY: a version leaves a chain only by being retired with
            // `defer_destroy`, and epoch reclamation cannot run the destructor
            // while `guard` is pinned. So any pointer reachable from `latest`
            // during this pin stays allocated for the pin's duration.
            let v = unsafe { cur.as_ref() }?;
            if v.visible_to(snapshot, reader) {
                return Some(v);
            }
            cur = v.prev.load(Ordering::Acquire, guard);
        }
    }

    /// The versions visible at `snapshot` to `reader` and to *nobody*, from a
    /// single descent.
    ///
    /// `Serializable` needs both on every predicate read: the first is what the
    /// caller gets back, the second is what goes into the read set — recorded
    /// as seen by nobody so a transaction's own uncommitted writes stay out of
    /// its own read set, or every read-modify-write would abort itself.
    ///
    /// One walk suffices because the second answer can only ever sit at or
    /// below the first in the chain. `Visibility::reached` for a real reader is
    /// a superset of `reached` for nobody — an in-flight version counts only
    /// for its own author — so the versions the two disagree about are exactly
    /// the ones `reader` wrote, and those are nearer the head than the
    /// committed versions they displaced.
    pub(crate) fn read_pair<'g>(
        &self,
        snapshot: Timestamp,
        reader: TxnId,
        guard: &'g Guard,
    ) -> (Option<&'g Version<T>>, Option<&'g Version<T>>) {
        let mut seen = None;
        let mut cur = self.latest.load(Ordering::Acquire, guard);
        loop {
            // SAFETY: as in `Slot::read`.
            let Some(v) = (unsafe { cur.as_ref() }) else {
                return (seen, None);
            };
            if seen.is_none() && v.visible_to(snapshot, reader) {
                seen = Some(v);
            }
            if v.visible_to(snapshot, TxnId::NONE) {
                // `seen` is necessarily set by now: a version visible to nobody
                // but not to `reader` must have been ended by `reader`, which
                // means `reader` installed a newer one, which is visible to it.
                return (seen, Some(v));
            }
            cur = v.prev.load(Ordering::Acquire, guard);
        }
    }

    /// The newest version that is committed, or written by `reader` itself —
    /// at no snapshot in particular.
    ///
    /// Every other read in the engine is snapshot-relative; this one
    /// deliberately is not, and it exists for exactly one caller: unique index
    /// enforcement. A constraint is a property of the *database*, not of a
    /// transaction's view of it, so checking it at the caller's snapshot asks
    /// the wrong question — the row that would collide may have been committed
    /// after that snapshot was taken, and is no less real for it. Postgres
    /// makes the same departure, checking uniqueness against the live index
    /// rather than the reader's MVCC snapshot.
    ///
    /// Returns the head-most version whose `begin` is committed or ours.
    /// Timestamps increase toward the head among committed versions — the slot
    /// lock serialises installation, and a commit stamp is taken after the
    /// install it belongs to — so the first such version is the current one,
    /// and its `end` needs no examination: whatever ended it would be nearer
    /// the head and would have been returned instead.
    pub(crate) fn read_committed_now<'g>(
        &self,
        reader: TxnId,
        guard: &'g Guard,
    ) -> Option<&'g Version<T>> {
        let mut cur = self.latest.load(Ordering::Acquire, guard);
        loop {
            // SAFETY: as in `Slot::read`.
            let v = unsafe { cur.as_ref() }?;
            match Visibility::decode(v.begin.load(Ordering::Acquire)) {
                Visibility::CommittedAt(_) => return Some(v),
                Visibility::InFlight(id) if id == reader => return Some(v),
                // Someone else's uncommitted write. Skipping it is what makes
                // the *claim* rather than this walk the thing that keeps two
                // in-flight writers off one key.
                Visibility::InFlight(_) => {}
            }
            cur = v.prev.load(Ordering::Acquire, guard);
        }
    }

    /// Take the write lock, first-updater-wins.
    ///
    /// Returns `false` if another transaction holds it. We abort rather than
    /// wait: waiting reintroduces deadlock detection, which is one of the
    /// things MVCC just removed.
    pub(crate) fn try_lock(&self, txn: TxnId) -> bool {
        self.lock
            .compare_exchange(0, txn.0, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
            || self.lock.load(Ordering::Acquire) == txn.0 // already ours
    }

    pub(crate) fn unlock(&self, txn: TxnId) {
        let _ = self
            .lock
            .compare_exchange(txn.0, 0, Ordering::AcqRel, Ordering::Acquire);
    }
}

/// Rows a scan returned: primary key and the version it matched through, in
/// primary key order. Borrowed from the caller's epoch pin, never cloned.
type Matches<'g, T> = Vec<(<T as Versioned>::Key, &'g Version<T>)>;

/// A registered table: the primary map plus any secondary indexes.
pub(crate) struct Table<T: Versioned> {
    /// Primary key to slot. Lock-free on the read path — see [`SlotMap`].
    slots: SlotMap<T>,
    /// One per `#[mvcc(index)]` field, in declaration order.
    ///
    /// Entries are *candidates*, not answers: an index maps a key to primary
    /// keys that had that value at some point. A scan resolves each candidate's
    /// visible version and re-checks its extracted key. That makes index
    /// maintenance append-only — no entry ever has to be removed on update or
    /// rollback — at the cost of a recheck per candidate, which the scan is
    /// doing anyway to establish visibility.
    secondary: Vec<RwLock<BTreeMap<IndexKey, BTreeSet<T::Key>>>>,
    /// Predicate SIREAD locks: predicates that `Serializable` transactions have
    /// evaluated and that a future insert or update might come to satisfy.
    ///
    /// A per-slot reader list cannot express a read of a row that does not
    /// exist yet, which is exactly what a phantom is. This can: a writer checks
    /// its new row against every registered predicate, and a match is an
    /// incoming rw-antidependency.
    predicate_locks: Mutex<Vec<PredicateLock<T>>>,
    /// Unique index keys claimed by in-flight transactions, one map per index.
    ///
    /// This is first-updater-wins applied to an index key instead of a slot,
    /// and it is what makes a unique constraint hold under concurrency.
    /// [`Table::unique_key_taken`] can only see what is already committed; two
    /// transactions inserting the same key into two *different* slots are
    /// invisible to each other by construction, because neither one's version
    /// is committed while the other is looking. Nothing in the slot lock, the
    /// snapshot, or SSI covers that — the collision is between rows that do not
    /// share a slot, and for an inserted row there is no earlier version to
    /// have read.
    ///
    /// Claims are taken before the check and released after the writes are
    /// stamped, so a claim's whole lifetime covers the window in which the
    /// claimant's rows are invisible to everyone else. Whoever claims next
    /// therefore sees them.
    ///
    /// Non-unique indexes keep an entry so positions line up with
    /// [`Versioned::indexes`]; it is never touched.
    unique_claims: Vec<Mutex<HashMap<IndexKey, TxnId>>>,
}

/// What [`Table::try_claim_unique`] did.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Claim {
    /// The key was free and is now ours. The caller owns releasing it.
    Acquired,
    /// We already held it, from an earlier write in the same transaction.
    /// Releasing it is the earlier claim's job, not this one's.
    Held,
    /// Another in-flight transaction holds it.
    Contended,
}

impl<T: Versioned> Drop for Table<T> {
    /// Free every version chain.
    ///
    /// Necessary because `Atomic` does not own its pointee — that is the whole
    /// point of epoch reclamation, and it is what the `Arc` chain this replaced
    /// used to do implicitly. Without this, dropping a `Database` leaks every
    /// version it ever held.
    fn drop(&mut self) {
        // SAFETY: `&mut self` proves there is no concurrent reader, so the
        // chains can be freed directly rather than deferred. `unprotected` is
        // exactly the escape hatch for that situation.
        let guard = unsafe { crossbeam_epoch::unprotected() };
        // Runs before the field itself drops, so the records are still there.
        // `SlotMap::drop` then frees the records; the chains hanging off them
        // are this table's to free, because `Atomic` does not own its pointee.
        self.slots.for_each(guard, |record| {
            let mut cur = record
                .slot
                .latest
                .swap(Shared::null(), Ordering::Relaxed, guard);
            while !cur.is_null() {
                // SAFETY: each version belongs to exactly one chain and is
                // reached once, so this takes ownership exactly once.
                let owned = unsafe { cur.into_owned() };
                cur = owned.prev.load(Ordering::Relaxed, guard);
                drop(owned);
            }
        });
    }
}

/// One registered predicate read, held until its transaction expires.
struct PredicateLock<T> {
    state: Arc<TxnState>,
    predicate: Arc<dyn Fn(&T) -> bool + Send + Sync>,
}

impl<T: Versioned> Table<T> {
    fn new() -> Self {
        Table {
            slots: SlotMap::new(),
            secondary: T::indexes()
                .iter()
                .map(|_| RwLock::new(BTreeMap::new()))
                .collect(),
            predicate_locks: Mutex::new(Vec::new()),
            unique_claims: T::indexes()
                .iter()
                .map(|_| Mutex::new(HashMap::new()))
                .collect(),
        }
    }

    /// The slot for `key`, borrowed for as long as the table is.
    ///
    /// Takes no locks and writes nothing to shared memory — the map is
    /// append-only, so the record this borrows lives as long as the table
    /// does. See [`SlotMap`] for why that is the whole design.
    ///
    /// A delete does not remove a slot; it installs a tombstone version, so
    /// readers at older snapshots still find the record. If garbage collection
    /// ever starts reclaiming empty slots, that reasoning breaks and the map
    /// needs real reclamation rather than an append-only argument.
    pub(crate) fn slot(&self, key: &T::Key, guard: &Guard) -> Option<&Slot<T>> {
        self.slots.get(key, guard)
    }

    /// The slot for `key`, creating an empty one if absent.
    pub(crate) fn slot_or_create(&self, key: &T::Key, guard: &Guard) -> &Slot<T> {
        self.slots.get_or_create(key, guard)
    }

    /// Record `record`'s index keys as candidates, skipping any index whose key
    /// is unchanged from `previous` — the version this write displaces, or
    /// `None` for an insert.
    ///
    /// Safe to skip because index maintenance is append-only (see
    /// [`Table::secondary`]): an unchanged key's entry is already there from
    /// the write that first set it, and nothing ever removes one. What it buys
    /// is the lock. Without the check, every write to a table takes the
    /// *exclusive* lock on every one of its secondary indexes, whether or not
    /// the indexed column was touched — which is what actually made the module
    /// header's "an update touches only the indexes whose key actually changed"
    /// true of the slot address but not of the index maps.
    pub(crate) fn index_record(&self, record: &T, previous: Option<&T>) {
        let key = record.key();
        for (desc, map) in T::indexes().iter().zip(&self.secondary) {
            let index_key = (desc.extract)(record);
            if previous.is_some_and(|prev| (desc.extract)(prev) == index_key) {
                continue;
            }
            map.write()
                .entry(index_key)
                .or_default()
                .insert(key.clone());
        }
    }

    /// Candidate primary keys in an index range.
    pub(crate) fn index_candidates(
        &self,
        position: usize,
        lo: Bound<IndexKey>,
        hi: Bound<IndexKey>,
    ) -> Vec<T::Key> {
        let map = self.secondary[position].read();
        map.range((lo, hi))
            .flat_map(|(_, keys)| keys.iter().cloned())
            .collect()
    }

    /// Visit every slot in the table, taking no locks at all.
    ///
    /// A scan is the one operation that pays per-row costs on every row in the
    /// table rather than on one, so the difference between this and looking
    /// each key up again — a hash and a lock acquisition per row — is the
    /// difference the scan benchmarks measure. Since [`SlotMap`] hands out
    /// record borrows that outlive any epoch, there is nothing left to release
    /// and nothing to batch.
    fn visit_slots<'s>(&'s self, guard: &Guard, mut f: impl FnMut(&'s Slot<T>)) {
        self.slots.for_each(guard, |record| f(&record.slot));
    }

    /// Every record visible at `snapshot` that satisfies `predicate`, in
    /// primary key order.
    ///
    /// This is the primitive behind both `Transaction::scan_where` and its
    /// revalidation at commit. They must agree exactly — a predicate read is
    /// validated by re-running it and comparing — so they call the same code
    /// rather than two implementations that could drift.
    pub(crate) fn matching<'g>(
        &self,
        snapshot: Timestamp,
        reader: TxnId,
        predicate: &dyn Fn(&T) -> bool,
        guard: &'g Guard,
    ) -> Matches<'g, T> {
        let mut out = Vec::new();
        self.visit_slots(guard, |slot| {
            if let Some(version) = slot.read(snapshot, reader, guard)
                && let Some(value) = version.value.as_ref()
                && predicate(value)
            {
                out.push((value.key(), version));
            }
        });
        // Sorting the matches rather than the whole key space is the point of
        // deriving keys from records: a selective predicate sorts a handful of
        // rows where collecting keys up front sorted every row in the table.
        // Unstable: primary keys are unique, so there are no equal elements
        // for stability to order — and the stable sort's `n/2` scratch buffer
        // is a 160KB allocation on a scan of this size. The sort is the
        // dominant cost of an unselective scan, 72% of it when measured by
        // deleting it.
        out.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// [`Table::matching`] as `reader` sees it *and* as nobody sees it, from
    /// one pass over the table.
    ///
    /// `Serializable` needs both — see [`Slot::read_pair`] for which is which
    /// and why one chain descent answers both. Calling `matching` twice would
    /// be two full table passes for a difference confined to the rows `reader`
    /// itself wrote.
    pub(crate) fn matching_pair<'g>(
        &self,
        snapshot: Timestamp,
        reader: TxnId,
        predicate: &dyn Fn(&T) -> bool,
        guard: &'g Guard,
    ) -> (Matches<'g, T>, Matches<'g, T>) {
        let mut seen = Vec::new();
        let mut committed = Vec::new();
        self.visit_slots(guard, |slot| {
            let (a, b) = slot.read_pair(snapshot, reader, guard);
            let same = match (a, b) {
                (Some(a), Some(b)) => std::ptr::eq(a, b),
                (None, None) => true,
                _ => false,
            };
            if let Some(version) = a
                && let Some(value) = version.value.as_ref()
                && predicate(value)
            {
                let key = value.key();
                if same {
                    committed.push((key.clone(), version));
                }
                seen.push((key, version));
            }
            // Only reachable for a row `reader` has written, so the second
            // predicate evaluation is paid on those rows alone.
            if !same
                && let Some(version) = b
                && let Some(value) = version.value.as_ref()
                && predicate(value)
            {
                committed.push((value.key(), version));
            }
        });
        seen.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        committed.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        (seen, committed)
    }

    /// Every record visible at `snapshot` whose key for index `position` falls
    /// in `[lo, hi]`, in index key order.
    pub(crate) fn matching_in_index<'g>(
        &self,
        position: usize,
        lo: &Bound<IndexKey>,
        hi: &Bound<IndexKey>,
        snapshot: Timestamp,
        reader: TxnId,
        guard: &'g Guard,
    ) -> Matches<'g, T> {
        let desc = &T::indexes()[position];
        let in_range = |k: &IndexKey| {
            let lo_ok = match lo {
                Bound::Included(b) => k >= b,
                Bound::Excluded(b) => k > b,
                Bound::Unbounded => true,
            };
            let hi_ok = match hi {
                Bound::Included(b) => k <= b,
                Bound::Excluded(b) => k < b,
                Bound::Unbounded => true,
            };
            lo_ok && hi_ok
        };

        // Index entries are candidates, not answers — see `Table::secondary`.
        // Each is resolved to its visible version and its key re-extracted,
        // which is what filters out entries left behind by updates and
        // rolled-back writes.
        let mut out: Vec<(IndexKey, T::Key, &'g Version<T>)> = Vec::new();
        for key in self.index_candidates(position, lo.clone(), hi.clone()) {
            let Some(slot) = self.slot(&key, guard) else {
                continue;
            };
            let Some(version) = slot.read(snapshot, reader, guard) else {
                continue;
            };
            let Some(value) = version.value.as_ref() else {
                continue;
            };
            let actual = (desc.extract)(value);
            if in_range(&actual) {
                out.push((actual, key, version));
            }
        }
        out.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        out.into_iter().map(|(_, k, v)| (k, v)).collect()
    }

    /// Register that `state` evaluated `predicate` over this table.
    pub(crate) fn register_predicate(
        &self,
        state: &Arc<TxnState>,
        predicate: Arc<dyn Fn(&T) -> bool + Send + Sync>,
    ) {
        self.predicate_locks.lock().push(PredicateLock {
            state: Arc::clone(state),
            predicate,
        });
    }

    /// Transactions other than `writer` whose registered predicate `record`
    /// satisfies — that is, whose predicate read this write turns into a
    /// phantom.
    ///
    /// Purges expired locks while it walks, which is the only place they are
    /// walked, so the cleanup is already paid for.
    pub(crate) fn predicate_readers_of(
        &self,
        record: &T,
        writer: TxnId,
        gc_watermark: Timestamp,
    ) -> Vec<Arc<TxnState>> {
        let mut locks = self.predicate_locks.lock();
        locks.retain(|l| !l.state.is_expired(gc_watermark));
        locks
            .iter()
            .filter(|l| l.state.id() != writer && (l.predicate)(record))
            .map(|l| Arc::clone(&l.state))
            .collect()
    }

    /// Claim `index_key` on unique index `position` for `txn`, unless another
    /// in-flight transaction holds it.
    ///
    /// See [`Table::unique_claims`] for why this exists at all.
    pub(crate) fn try_claim_unique(
        &self,
        position: usize,
        index_key: &IndexKey,
        txn: TxnId,
    ) -> Claim {
        let mut claims = self.unique_claims[position].lock();
        match claims.get(index_key) {
            Some(&owner) if owner == txn => Claim::Held,
            Some(_) => Claim::Contended,
            None => {
                claims.insert(index_key.clone(), txn);
                Claim::Acquired
            }
        }
    }

    /// Drop a claim taken by [`Table::try_claim_unique`].
    ///
    /// Checks the owner rather than removing blindly, so that releasing a claim
    /// this transaction does not hold cannot evict someone else's.
    pub(crate) fn release_unique(&self, position: usize, index_key: &IndexKey, txn: TxnId) {
        let mut claims = self.unique_claims[position].lock();
        if claims.get(index_key) == Some(&txn) {
            claims.remove(index_key);
        }
    }

    /// Whether some record other than `own_key` currently holds `index_key`.
    ///
    /// "Currently" as in [`Slot::read_committed_now`]: committed, at whatever
    /// timestamp, plus `reader`'s own uncommitted writes. Only sound as half of
    /// a claim — on its own it cannot see a concurrent in-flight writer of the
    /// same key.
    pub(crate) fn unique_key_taken(
        &self,
        position: usize,
        index_key: &IndexKey,
        own_key: &T::Key,
        reader: TxnId,
        guard: &Guard,
    ) -> bool {
        let extract = T::indexes()[position].extract;
        let candidates = {
            let map = self.secondary[position].read();
            map.get(index_key).cloned().unwrap_or_default()
        };
        candidates.into_iter().any(|candidate| {
            if candidate == *own_key {
                return false;
            }
            // Recheck: index entries are candidates, not answers — the
            // candidate may have moved off this key, been deleted, or belong to
            // a write that rolled back.
            self.slot(&candidate, guard)
                .and_then(|slot| slot.read_committed_now(reader, guard))
                .and_then(|version| version.value.as_ref())
                .is_some_and(|value| extract(value) == *index_key)
        })
    }
}

/// Configuration for a [`Database`].
///
/// There is no `data_dir`: this engine is in-memory by design, not by omission.
///
/// There is no shard count either: the oracle and the slot map each pick their
/// own, as compile-time constants, because the oracle's is part of how
/// transaction ids are encoded rather than a tuning knob.
#[derive(Debug, Default)]
pub struct Config {
    /// Timestamp oracle tuning. The defaults are the measured ones; see
    /// [`OracleConfig`](crate::config::OracleConfig).
    pub oracle: OracleConfig,
}

impl Config {
    /// The default configuration.
    ///
    /// Named rather than relying on `Default` so that call sites say plainly
    /// what this database is.
    pub fn in_memory() -> Self {
        Config::default()
    }
}

/// The handle everything hangs off.
pub struct Database {
    oracle: Oracle,
    /// Type-erased tables. **Append-only** — see the safety note on
    /// [`Database::table`], which depends on entries never being removed.
    tables: RwLock<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>,
    next_table_id: AtomicU16,
    /// Serialises the validate-allocate-stamp phase of commit.
    ///
    /// Held only for the duration of that phase, never across user code or I/O.
    /// It is a real scalability limit and the next thing to shard; see the note
    /// in `Transaction::commit`.
    commit_lock: Mutex<()>,
}

impl Database {
    /// Create an empty database.
    ///
    /// Nothing is read from disk and nothing will be written to it — see the
    /// [crate docs](crate#what-this-is-not). Every type it will store must then
    /// be passed to [`Database::register`] before use.
    ///
    /// ```
    /// use mvcc::{Config, Database};
    ///
    /// let db = Database::open(Config::in_memory())?;
    /// # Ok::<(), mvcc::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Currently infallible; the `Result` is here so that configuration that
    /// can fail may be added without a breaking change.
    // Takes `Config` by value even though only `oracle` is read out of it: this
    // is the public constructor, and handing ownership over is what lets fields
    // be added later without changing the signature.
    #[allow(clippy::needless_pass_by_value)]
    pub fn open(config: Config) -> Result<Self> {
        Ok(Database {
            oracle: Oracle::new(config.oracle),
            tables: RwLock::new(HashMap::new()),
            next_table_id: AtomicU16::new(0),
            commit_lock: Mutex::new(()),
        })
    }

    /// The timestamp oracle.
    ///
    /// `pub(crate)`: `Oracle` is not re-exported, so a `pub` accessor here
    /// promised users a type they could neither name nor read the docs for.
    /// [`OracleConfig`](crate::config::OracleConfig) is the supported way to
    /// influence it, and [`Database::stats`] the supported way to observe it.
    pub(crate) fn oracle(&self) -> &Oracle {
        &self.oracle
    }

    pub(crate) fn commit_lock(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.commit_lock.lock()
    }

    /// Snapshot of engine statistics. See [`crate::stats`] for what to watch.
    pub fn stats(&self) -> crate::engine::gc::GcStats {
        crate::engine::gc::GcStats {
            watermark: self.oracle.gc_watermark(),
            active_transactions: self.oracle.active_count(),
        }
    }

    /// Register a type. Assigns its [`TableId`] and builds its indexes.
    ///
    /// Must happen before any transaction touches `T`. Registering twice is a
    /// no-op rather than an error, so a library that registers its own types
    /// defensively does not conflict with an application that did the same.
    pub fn register<T: Versioned>(&self) -> Result<()> {
        let type_id = TypeId::of::<T>();
        let mut tables = self.tables.write();
        if tables.contains_key(&type_id) {
            return Ok(());
        }

        let id = TableId(self.next_table_id.fetch_add(1, Ordering::Relaxed));
        // Fails only if another thread won the race; either way the cell now
        // holds a valid id for this type.
        let _ = T::table_id_cell().set(id);
        tables.insert(type_id, Arc::new(Table::<T>::new()));
        Ok(())
    }

    /// Borrow a registered table for as long as the `Database` is borrowed.
    ///
    /// Type-erased, because the only caller — `Transaction::table` — caches the
    /// result without naming `T`. Returning a reference rather than an `Arc`
    /// matters more than it looks: this is on *every* operation, and cloning the
    /// `Arc` meant an atomic increment and decrement on one refcount shared by
    /// every core in the process.
    pub(crate) fn table_erased(
        &self,
        type_id: TypeId,
        name: &'static str,
    ) -> Result<&(dyn Any + Send + Sync)> {
        let tables = self.tables.read();
        let entry = tables
            .get(&type_id)
            .ok_or(Error::TableNotRegistered { table: name })?;
        let erased: &(dyn Any + Send + Sync) = &**entry;

        // SAFETY: the borrow is extended from the read guard to `&self`.
        //
        // Sound because the registry is *append-only*: `register` inserts and
        // never removes or replaces an entry, and there is no public API that
        // does either. The `Arc` holding this `Table` is therefore owned by
        // `self` for all of `self`'s life, and the `Table` itself lives on the
        // heap and never moves. Dropping the guard releases the lock but cannot
        // drop or relocate the referent.
        //
        // If a `deregister` is ever added, this becomes unsound and must go
        // back to returning an `Arc`.
        Ok(unsafe { &*(erased as *const (dyn Any + Send + Sync)) })
    }

    /// Begin a transaction at the default isolation level (snapshot isolation).
    pub fn begin(&self) -> Transaction<'_, SnapshotLevel> {
        self.begin_with()
    }

    /// Begin a transaction at a chosen isolation level.
    ///
    /// The level is a type parameter, so the cost of the strongest never leaks
    /// into the weakest — see [`IsolationLevel`].
    ///
    /// ```
    /// # use mvcc::{Config, Database, Mvcc, ReadCommitted, Serializable};
    /// # #[derive(Mvcc, Clone)]
    /// # struct Account {
    /// #     #[mvcc(primary_key)] id: u64,
    /// #     balance: i64,
    /// # }
    /// # let db = Database::open(Config::in_memory())?;
    /// # db.register::<Account>()?;
    /// let strict = db.begin_with::<Serializable>();
    /// let cheap = db.begin_with::<ReadCommitted>();
    /// # Ok::<(), mvcc::Error>(())
    /// ```
    ///
    /// Prefer [`Database::transaction_with`] unless you need manual control:
    /// this hands back a transaction you must commit yourself, and **dropping
    /// it without committing rolls it back**.
    pub fn begin_with<I: IsolationLevel>(&self) -> Transaction<'_, I> {
        Transaction::new(self)
    }

    /// Run `f` in a transaction at snapshot isolation, retrying while it fails
    /// retriably, and commit.
    ///
    /// This is the API most users should reach for. Under snapshot isolation,
    /// and especially serializable, an abort is a normal outcome rather than an
    /// error condition — every caller would otherwise write this loop.
    ///
    /// `f` may run more than once, so it must not have side effects outside the
    /// transaction.
    ///
    /// Split from [`Database::transaction_with`] rather than defaulting a type
    /// parameter, because Rust cannot infer a defaulted type parameter on a
    /// function — `db.transaction(|tx| …)` would not compile.
    pub fn transaction<R, F>(&self, f: F) -> Result<R>
    where
        F: FnMut(&mut Transaction<'_, SnapshotLevel>) -> Result<R>,
    {
        self.transaction_with::<SnapshotLevel, R, F>(f)
    }

    /// Run `f` in a transaction at isolation level `I`, retrying while it fails
    /// retriably, and commit.
    ///
    /// Reach for [`Serializable`](crate::Serializable) here when a transaction's
    /// *write* depends on a value it merely *read* — a balance check, a capacity
    /// limit. That is the write-skew shape, and snapshot isolation does not
    /// catch it.
    ///
    /// ```
    /// # use mvcc::{Config, Database, Mvcc, Serializable};
    /// # #[derive(Mvcc, Clone)]
    /// # struct Account {
    /// #     #[mvcc(primary_key)] id: u64,
    /// #     balance: i64,
    /// # }
    /// # let db = Database::open(Config::in_memory())?;
    /// # db.register::<Account>()?;
    /// # db.transaction(|tx| {
    /// #     tx.insert(Account { id: 1, balance: 100 })?;
    /// #     tx.insert(Account { id: 2, balance: 0 })
    /// # })?;
    /// // Withdraw only if the balance covers it. The check and the write must
    /// // be serializable together, or two concurrent transfers each see a
    /// // sufficient balance and both withdraw.
    /// let moved = db.transaction_with::<Serializable, _, _>(|tx| {
    ///     let balance = tx.get::<Account>(&1)?.map_or(0, |a| a.balance);
    ///     if balance < 50 {
    ///         return Ok(false);
    ///     }
    ///     tx.update::<Account>(&1, |a| a.balance -= 50)?;
    ///     tx.update::<Account>(&2, |a| a.balance += 50)?;
    ///     Ok(true)
    /// })?;
    ///
    /// assert!(moved);
    /// # Ok::<(), mvcc::Error>(())
    /// ```
    ///
    /// As with [`Database::transaction`], `f` may run more than once, so it
    /// must not have side effects outside the transaction. Return the value and
    /// let the caller act on the committed result, as above.
    pub fn transaction_with<I, R, F>(&self, mut f: F) -> Result<R>
    where
        I: IsolationLevel,
        F: FnMut(&mut Transaction<'_, I>) -> Result<R>,
    {
        const MAX_ATTEMPTS: u32 = 100;

        let mut attempt = 0;
        loop {
            attempt += 1;
            let mut tx = self.begin_with::<I>();
            let outcome = f(&mut tx).and_then(|r| tx.commit().map(|_| r));

            match outcome {
                Ok(r) => return Ok(r),
                Err(e) if e.is_retriable() && attempt < MAX_ATTEMPTS => {
                    // Exponential backoff with a ceiling. Without it, two
                    // conflicting transactions can livelock by retrying in
                    // lockstep and colliding at the same point every time.
                    let backoff = 1u64 << attempt.min(10);
                    std::thread::sleep(std::time::Duration::from_micros(backoff));
                }
                Err(e) => return Err(e),
            }
        }
    }
}
