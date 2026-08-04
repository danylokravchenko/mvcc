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
//! The chain is built from `Arc` and `RwLock` rather than the `AtomicPtr`
//! design in `DESIGN.md`. That is deliberate sequencing, not an accident:
//! correctness first, in safe Rust, with the semantics pinned down by tests.
//! Swapping in lock-free chains and epoch-based reclamation is an internal
//! change that the public API cannot observe.
//!
//! The `Arc` version does hold one real cost worth naming: cloning an `Arc` on
//! read is an atomic increment on a shared cache line, which is exactly the
//! "reads must not write to shared memory" rule that `crate::gc` argues for.
//! That is what the eventual EBR migration buys back.
//!
//! # Locks
//!
//! `parking_lot` rather than `std::sync`, for the reasons measured in
//! `crate::oracle`. The `RwLock`s in this file benefit far less than the oracle's
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
use std::sync::Arc;
use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};

use mvcc_core::{
    Error, IndexKey, IsolationLevel, Result, Snapshot as SnapshotLevel, TableId, Timestamp, TxnId,
    Versioned, Visibility,
};

use crate::oracle::{Oracle, OracleConfig};
use crate::txn::Transaction;

/// One version of a record.
///
/// `value` is `None` for a tombstone: a delete installs a version like any
/// other write, so that a reader at an older snapshot still finds the record
/// alive behind it.
pub struct Version<T> {
    pub(crate) begin: AtomicU64,
    pub(crate) end: AtomicU64,
    pub(crate) prev: Option<Arc<Version<T>>>,
    pub(crate) value: Option<T>,
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
pub struct Slot<T> {
    pub(crate) latest: RwLock<Option<Arc<Version<T>>>>,
    /// Transaction id currently writing this slot, or 0 if free.
    pub(crate) lock: AtomicU64,
}

impl<T> Slot<T> {
    fn new() -> Self {
        Slot {
            latest: RwLock::new(None),
            lock: AtomicU64::new(0),
        }
    }

    /// Walk the chain for the version visible at `snapshot`.
    pub(crate) fn read(&self, snapshot: Timestamp, reader: TxnId) -> Option<Arc<Version<T>>> {
        let mut cur = self.latest.read().clone();
        while let Some(v) = cur {
            if v.visible_to(snapshot, reader) {
                return Some(v);
            }
            cur = v.prev.clone();
        }
        None
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

/// A registered table: the primary map plus any secondary indexes.
pub struct Table<T: Versioned> {
    id: TableId,
    slots: RwLock<HashMap<T::Key, Arc<Slot<T>>>>,
    /// One per `#[mvcc(index)]` field, in declaration order.
    ///
    /// Entries are *candidates*, not answers: an index maps a key to primary
    /// keys that had that value at some point. A scan resolves each candidate's
    /// visible version and re-checks its extracted key. That makes index
    /// maintenance append-only — no entry ever has to be removed on update or
    /// rollback — at the cost of a recheck per candidate, which the scan is
    /// doing anyway to establish visibility.
    secondary: Vec<RwLock<BTreeMap<IndexKey, BTreeSet<T::Key>>>>,
}

impl<T: Versioned> Table<T> {
    fn new(id: TableId) -> Self {
        Table {
            id,
            slots: RwLock::new(HashMap::new()),
            secondary: T::indexes()
                .iter()
                .map(|_| RwLock::new(BTreeMap::new()))
                .collect(),
        }
    }

    pub fn id(&self) -> TableId {
        self.id
    }

    pub(crate) fn slot(&self, key: &T::Key) -> Option<Arc<Slot<T>>> {
        self.slots.read().get(key).cloned()
    }

    /// Fetch the slot for `key`, creating an empty one if absent.
    pub(crate) fn slot_or_create(&self, key: &T::Key) -> Arc<Slot<T>> {
        if let Some(s) = self.slot(key) {
            return s;
        }
        let mut slots = self.slots.write();
        slots
            .entry(key.clone())
            .or_insert_with(|| Arc::new(Slot::new()))
            .clone()
    }

    /// Record `record`'s index keys as candidates.
    pub(crate) fn index_record(&self, record: &T) {
        let key = record.key();
        for (desc, map) in T::indexes().iter().zip(&self.secondary) {
            let index_key = (desc.extract)(record);
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
        lo: std::ops::Bound<IndexKey>,
        hi: std::ops::Bound<IndexKey>,
    ) -> Vec<T::Key> {
        let map = self.secondary[position].read();
        map.range((lo, hi))
            .flat_map(|(_, keys)| keys.iter().cloned())
            .collect()
    }

    /// Every primary key, for a full scan.
    pub(crate) fn all_keys(&self) -> Vec<T::Key> {
        let slots = self.slots.read();
        let mut keys: Vec<_> = slots.keys().cloned().collect();
        keys.sort();
        keys
    }

    /// Whether a unique index would be violated by `record`, given a reader's
    /// snapshot.
    pub(crate) fn unique_violation(
        &self,
        record: &T,
        snapshot: Timestamp,
        reader: TxnId,
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
                let Some(slot) = self.slot(&candidate) else {
                    continue;
                };
                let Some(version) = slot.read(snapshot, reader) else {
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
            shards: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
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

    /// Snapshot of engine statistics. See [`crate::gc`] for what to watch.
    pub fn stats(&self) -> crate::gc::GcStats {
        crate::gc::GcStats {
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
        tables.insert(type_id, Arc::new(Table::<T>::new(T::table_id())));
        Ok(())
    }

    pub(crate) fn table<T: Versioned>(&self) -> Result<Arc<Table<T>>> {
        let tables = self.tables.read();
        tables
            .get(&TypeId::of::<T>())
            .and_then(|t| t.clone().downcast::<Table<T>>().ok())
            .ok_or(Error::TableNotRegistered {
                table: T::TABLE_NAME,
            })
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
