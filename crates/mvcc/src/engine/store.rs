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

    /// Resolve an index name to its position, for scans.
    pub(crate) fn index_position(&self, name: &str) -> Option<usize> {
        T::indexes().iter().position(|d| d.name == name)
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

    /// Whether a unique index would be violated by `record`, given a reader's
    /// snapshot.
    pub(crate) fn unique_violation(
        &self,
        record: &T,
        snapshot: Timestamp,
        reader: TxnId,
        guard: &Guard,
    ) -> Option<&'static str> {
        let own_key = record.key();
        for (position, desc) in T::indexes().iter().enumerate() {
            if !desc.unique {
                continue;
            }
            let index_key = (desc.extract)(record);
            let candidates = {
                let map = self.secondary[position].read();
                map.get(&index_key).cloned().unwrap_or_default()
            };
            for candidate in candidates {
                if candidate == own_key {
                    continue;
                }
                // Recheck: the candidate may have moved off this key, or its
                // visible version may not exist at this snapshot.
                let Some(slot) = self.slot(&candidate, guard) else {
                    continue;
                };
                let Some(version) = slot.read(snapshot, reader, guard) else {
                    continue;
                };
                let Some(value) = version.value.as_ref() else {
                    continue;
                };
                if (desc.extract)(value) == index_key {
                    return Some(desc.name);
                }
            }
        }
        None
    }
}

/// Configuration for a [`Database`].
///
/// There is no `data_dir`: this engine is in-memory by design, not by
/// omission. See `DESIGN.md` §5.
#[derive(Debug)]
pub struct Config {
    /// Number of independent shards. Not yet used — see `DESIGN.md` §6.
    pub shards: usize,
    pub oracle: OracleConfig,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            shards: std::thread::available_parallelism().map_or(4, std::num::NonZero::get),
            oracle: Default::default(),
        }
    }
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

    pub fn oracle(&self) -> &Oracle {
        &self.oracle
    }

    pub(crate) fn commit_lock(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.commit_lock.lock()
    }

    /// Snapshot of engine statistics. See [`crate::engine::gc`] for what to watch.
    pub fn stats(&self) -> crate::engine::gc::GcStats {
        crate::engine::gc::GcStats {
            pending_reclaim: 0,
            reclaimed_total: 0,
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
    /// every core in the process. See `DESIGN.md` §6.
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
    /// ```ignore
    /// let tx = db.begin_with::<Serializable>();
    /// ```
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
    /// ```ignore
    /// db.transaction_with::<Serializable, _, _>(|tx| { … })?;
    /// ```
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
