//! The transaction handle — the type users actually touch.

use std::any::{Any, TypeId};
use std::marker::PhantomData;
use std::ops::{Bound, RangeBounds};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::core::{
    Encodable, Error, IndexKey, IsolationLevel, Result, Timestamp, TxnId, Versioned, Visibility,
};

use crossbeam_epoch::{Guard, Owned, Shared};

use crate::engine::ssi::TxnState;
use crate::engine::store::{Database, Slot, Table, Version};

/// A write this transaction has installed but not yet committed.
///
/// Type-erased so one `Transaction` can hold writes to many different tables.
trait WriteOp<'db> {
    /// Stamp the new version visible at `ts` and retire the one it replaced.
    fn commit(&self, ts: Timestamp);
    /// Unlink the new version, restoring the slot to what it was.
    fn abort(&self);
    /// Transactions that read what this write overwrites — the incoming half of
    /// the SSI pivot test.
    ///
    /// Two sources, and both are needed. The slot's SIREAD locks cover readers
    /// of the row as it was. The table's registered predicates cover readers of
    /// rows that did not exist when they looked, which is what an insert turns
    /// into a phantom.
    fn conflicting_readers(&self, txn: TxnId, gc: Timestamp) -> Vec<Arc<TxnState>>;
}

struct SlotWrite<'db, T: Versioned> {
    table: &'db Table<T>,
    slot: &'db Slot<T>,
    /// The version installed, and the one it displaced.
    ///
    /// Raw pointers rather than references because the allocations they name are
    /// kept alive by the *transaction's own pin*, and a struct cannot borrow
    /// from a sibling field. The invariant that makes dereferencing them sound:
    /// `Transaction::guard` is declared last, so it is dropped after the write
    /// set, and every write op is consumed or discarded before that.
    installed: *const Version<T>,
    replaced: *const Version<T>,
    txn: TxnId,
}

impl<'db, T: Versioned> WriteOp<'db> for SlotWrite<'db, T> {
    fn commit(&self, ts: Timestamp) {
        // SAFETY: both pointers name versions kept alive by this transaction's
        // pin — see the note on the fields.
        unsafe {
            // Order matters. The new version becomes visible only once `begin`
            // holds a real timestamp, so stamping the old version's `end` first
            // would briefly leave the record invisible to concurrent readers.
            if let Some(prev) = self.replaced.as_ref() {
                prev.end.store(ts.raw(), Ordering::Release);
            }
            (*self.installed).begin.store(ts.raw(), Ordering::Release);
        }
        self.slot.unlock(self.txn);
    }

    fn abort(&self) {
        // Unlink first, then retire. Between the two the version is unreachable
        // from the chain but still allocated, which is exactly the window
        // epoch reclamation exists to cover: a reader that loaded the pointer
        // before the store keeps dereferencing it safely until its pin ends.
        let guard = &crossbeam_epoch::pin();
        self.slot
            .latest
            .store(Shared::from(self.replaced), Ordering::Release);

        // SAFETY: `installed` was allocated by this transaction and has just
        // been unlinked, so no new reader can reach it. `defer_destroy` runs
        // the drop only once every thread pinned at retirement has unpinned,
        // which covers readers that loaded the pointer before the store above.
        unsafe {
            guard.defer_destroy(Shared::from(self.installed));
        }
        self.slot.unlock(self.txn);
    }

    fn conflicting_readers(&self, txn: TxnId, gc: Timestamp) -> Vec<Arc<TxnState>> {
        let mut readers = self.slot.readers.lock().others(txn, gc);
        // SAFETY: see the note on the fields.
        unsafe {
            if let Some(value) = (*self.installed).value.as_ref() {
                readers.extend(self.table.predicate_readers_of(value, txn, gc));
            }
            // A delete makes a row vanish from predicates that matched it, so
            // the version it replaced has to be checked too.
            if let Some(previous) = self.replaced.as_ref().and_then(|v| v.value.as_ref()) {
                readers.extend(self.table.predicate_readers_of(previous, txn, gc));
            }
        }
        readers
    }
}

/// Re-checks at commit that what this transaction read has not changed.
///
/// Returns the transactions responsible for any change rather than a bare
/// `bool`, because SSI needs to mark *their* incoming edge as well as this
/// transaction's outgoing one. An empty result means the read is still valid.
///
/// Anything this reports has necessarily committed: revalidation reads at the
/// current read watermark, which by construction only covers commits that are
/// fully installed. That is what satisfies Cahill's "the transaction on the
/// outgoing edge commits first" condition without tracking commit order
/// separately.
trait ReadValidation<'db> {
    fn revalidate(&self, now: Timestamp, guard: &Guard) -> Revalidation;
}

/// The outcome of re-checking one read at commit.
///
/// `changed` and `writers` are separate on purpose. A writer below
/// `Serializable` has no [`TxnState`] — it does no SSI bookkeeping of its own,
/// and allocating one per transaction for nobody to read cost an allocation on
/// the hot path — so its versions are anonymous. The change it made still has to
/// set *this* transaction's outgoing edge.
///
/// Collapsing the two, and treating "no writers named" as "nothing changed",
/// silently drops that edge: a `Serializable` transaction whose read was
/// overwritten by a `Snapshot` one would see a clean read set and commit. It is
/// a quiet failure, and `both_edges_together_abort` is the test that catches it.
#[derive(Default)]
struct Revalidation {
    changed: bool,
    /// Writers that can be marked. May be empty even when `changed` is true.
    writers: Vec<Arc<TxnState>>,
}

impl Revalidation {
    fn unchanged() -> Self {
        Revalidation::default()
    }

    fn changed_by(writers: Vec<Arc<TxnState>>) -> Self {
        Revalidation {
            changed: true,
            writers,
        }
    }
}

/// One point read, recorded for commit-time revalidation.
///
/// `observed` is an *address*, not a pointer or a reference, and deliberately
/// so: revalidation only ever compares it for identity. Storing it as a `usize`
/// makes "never dereferenced" a property of the type rather than a comment, and
/// keeps the read set free of anything that could outlive the pin.
struct SlotRead<'db, T> {
    slot: &'db Slot<T>,
    observed: Option<usize>,
}

impl<'db, T: Send + Sync> ReadValidation<'db> for SlotRead<'db, T> {
    fn revalidate(&self, now: Timestamp, guard: &Guard) -> Revalidation {
        // Read as nobody, so this transaction's own in-flight writes stay
        // invisible. Validating as ourselves would compare what we read against
        // what we then wrote, and every read-modify-write would abort itself.
        let current = self.slot.read(now, TxnId::NONE, guard);
        let unchanged = match (self.observed, current) {
            (None, None) => true,
            (Some(a), Some(b)) => a == b as *const Version<T> as usize,
            _ => false,
        };
        if unchanged {
            Revalidation::unchanged()
        } else {
            Revalidation::changed_by(current.and_then(|v| v.writer.clone()).into_iter().collect())
        }
    }
}

/// A predicate a transaction evaluated, and the rows that satisfied it.
///
/// This is what closes the phantom hole. A [`SlotRead`] can only speak for a row
/// that existed when it was read; a row inserted afterwards has no slot to
/// compare against, so a read set built only from slots cannot express *"and
/// nothing else matched"*. Re-running the predicate at commit can.
///
/// Validation compares both the set of matching keys and the identity of each
/// matching version, so one entry covers phantoms (a key appears or disappears)
/// and item changes (a key's version was replaced) together.
struct PredicateRead<'db, T: Versioned> {
    table: &'db Table<T>,
    predicate: Arc<dyn Fn(&T) -> bool + Send + Sync>,
    /// Matching keys and the *address* of the version each matched through.
    /// Identity only — see [`SlotRead`].
    observed: Vec<(T::Key, usize)>,
}

impl<'db, T: Versioned> ReadValidation<'db> for PredicateRead<'db, T> {
    fn revalidate(&self, now: Timestamp, guard: &Guard) -> Revalidation {
        let current = self
            .table
            .matching(now, TxnId::NONE, &*self.predicate, guard);
        writers_of_change(self.table, &self.observed, &current, now, guard)
    }
}

/// An index range a transaction scanned, and the rows it returned.
///
/// The range *is* the predicate, so revalidation re-runs the range scan rather
/// than a full table scan — the same reason the index exists in the first place.
struct IndexRangeRead<'db, T: Versioned> {
    table: &'db Table<T>,
    position: usize,
    lo: Bound<IndexKey>,
    hi: Bound<IndexKey>,
    observed: Vec<(T::Key, usize)>,
}

impl<'db, T: Versioned> ReadValidation<'db> for IndexRangeRead<'db, T> {
    fn revalidate(&self, now: Timestamp, guard: &Guard) -> Revalidation {
        let current = self.table.matching_in_index(
            self.position,
            &self.lo,
            &self.hi,
            now,
            TxnId::NONE,
            guard,
        );
        writers_of_change(self.table, &self.observed, &current, now, guard)
    }
}

/// The transactions responsible for a predicate's result set changing, or empty
/// if it is unchanged in both membership and version identity.
///
/// A result set can change in two directions and both matter:
///
/// - a row **entered** it, or an existing row's version was replaced. The
///   responsible transaction wrote the version now in the set.
/// - a row **left** it, because an update moved it out of the predicate. The
///   responsible transaction wrote a version that is *not* in the set at all,
///   so it has to be fetched from the slot.
///
/// Missing the second case is easy and quiet: the predicate correctly reports
/// that it changed, but names nobody, so no rw-edge is recorded and the pivot
/// test never fires.
fn writers_of_change<T: Versioned>(
    table: &Table<T>,
    observed: &[(T::Key, usize)],
    current: &[(T::Key, &Version<T>)],
    now: Timestamp,
    guard: &Guard,
) -> Revalidation {
    let addr = |v: &Version<T>| v as *const Version<T> as usize;
    let identical = observed.len() == current.len()
        && observed
            .iter()
            .zip(current)
            .all(|((ko, vo), (kc, vc))| ko == kc && *vo == addr(vc));
    if identical {
        return Revalidation::unchanged();
    }

    let mut writers = Vec::new();

    for (key, version) in current {
        let same_as_observed = observed
            .iter()
            .any(|(ko, vo)| ko == key && *vo == addr(version));
        if !same_as_observed {
            writers.extend(version.writer.clone());
        }
    }

    for (key, _) in observed {
        if current.iter().any(|(kc, _)| kc == key) {
            continue;
        }
        if let Some(slot) = table.slot(key, guard)
            && let Some(version) = slot.read(now, TxnId::NONE, guard)
        {
            writers.extend(version.writer.clone());
        }
    }

    Revalidation::changed_by(writers)
}

/// A transaction at isolation level `I`.
///
/// `I` is a typestate: `Transaction<'_, ReadCommitted>` and
/// `Transaction<'_, Serializable>` are different types with different generated
/// code. See `crate::core::isolation`.
pub struct Transaction<'db, I: IsolationLevel> {
    db: &'db Database,
    id: TxnId,
    snapshot: Timestamp,
    /// Shared with other transactions, which set this one's conflict flags.
    /// Outlives the `Transaction`: SIREAD locks and version records keep
    /// referring to it after commit. See [`crate::engine::ssi`].
    ///
    /// `None` below `Serializable`. Only SSI reads these flags, so allocating
    /// one per transaction at every level meant a `malloc` and `free` on the
    /// hot path for a value nobody would look at.
    state: Option<Arc<TxnState>>,
    /// Epoch pin, held for the transaction's whole life.
    ///
    /// Pinning once here rather than once per read is what makes the reads
    /// free: inside the pin, following a version chain is plain pointer loads.
    /// It also gives every version this transaction observes a lifetime — the
    /// pin's — which is what `Ref<'txn, T>` borrows from.
    ///
    /// The cost is the same shape as the GC watermark's: a long-running
    /// transaction holds an epoch and defers reclamation for everyone. That is
    /// the failure mode `crate::engine::gc` already documents, now with a second way to
    /// trigger it.
    ///
    /// Declared last so it drops last: the read and write sets hold addresses
    /// into allocations this pin protects.
    guard: Guard,
    writes: Vec<Box<dyn WriteOp<'db> + 'db>>,
    /// Empty, and never pushed to, unless `I::VALIDATES_READS`.
    reads: Vec<Box<dyn ReadValidation<'db> + 'db>>,
    /// Tables already resolved by this transaction.
    ///
    /// The registry lock is uncontended in isolation but it is a *single*
    /// `RwLock` taken on every operation, so at eight threads it is a cache line
    /// ping-ponging between cores. A transaction touches one or two tables, so a
    /// linear scan over this beats going back to the registry every time.
    ///
    /// A `&'db dyn Any` rather than a raw pointer, so the transaction stays
    /// `Send` and no further `unsafe` is needed here.
    table_cache: Vec<(TypeId, &'db (dyn Any + Send + Sync))>,
    done: bool,
    /// `I` appears in no field: every difference between the levels is a
    /// `const` on [`IsolationLevel`], so the type parameter exists purely to
    /// select those constants at compile time.
    _level: PhantomData<fn() -> I>,
}

impl<'db, I: IsolationLevel> Transaction<'db, I> {
    pub(crate) fn new(db: &'db Database) -> Self {
        let id = db.oracle().next_txn_id();
        let snapshot = db.oracle().begin_snapshot(id);
        Transaction {
            id,
            snapshot,
            state: I::VALIDATES_READS.then(|| TxnState::new(id)),
            db,
            writes: Vec::new(),
            reads: Vec::new(),
            table_cache: Vec::new(),
            done: false,
            _level: PhantomData,
            guard: crossbeam_epoch::pin(),
        }
    }

    pub fn id(&self) -> TxnId {
        self.id
    }

    /// The snapshot this transaction reads at.
    pub fn snapshot(&self) -> Timestamp {
        self.snapshot
    }

    /// The snapshot for the statement about to run.
    ///
    /// Identical to [`Transaction::snapshot`] at every level except
    /// `ReadCommitted`, which takes a fresh one per statement and so can see
    /// commits that landed after it began.
    fn statement_snapshot(&self) -> Timestamp {
        if I::REFRESH_SNAPSHOT_PER_STATEMENT {
            self.db.oracle().statement_snapshot()
        } else {
            self.snapshot
        }
    }

    /// This transaction's SSI state.
    ///
    /// # Panics
    ///
    /// Below `Serializable`, where it is never allocated. Every caller is
    /// already inside a `I::VALIDATES_READS` branch, which the compiler folds
    /// away for the other levels.
    fn ssi(&self) -> &Arc<TxnState> {
        self.state
            .as_ref()
            .expect("SSI state is only touched when I::VALIDATES_READS")
    }

    /// Resolve `T`'s table, caching the registry lookup for the rest of this
    /// transaction. See [`Transaction::table_cache`].
    fn table<T: Versioned>(&mut self) -> Result<&'db Table<T>> {
        let type_id = TypeId::of::<T>();
        if let Some((_, erased)) = self.table_cache.iter().find(|(id, _)| *id == type_id) {
            return Ok(erased
                .downcast_ref::<Table<T>>()
                .expect("cache is keyed by TypeId"));
        }
        let erased = self.db.table_erased(type_id, T::TABLE_NAME)?;
        self.table_cache.push((type_id, erased));
        Ok(erased
            .downcast_ref::<Table<T>>()
            .expect("registry is keyed by TypeId"))
    }

    fn ensure_live(&self) -> Result<()> {
        if self.done {
            Err(Error::Aborted)
        } else {
            Ok(())
        }
    }

    /// Read by primary key.
    ///
    /// Returns a guard over the version in place rather than a copy. Reads take
    /// no locks and never block, whatever else is happening to the record.
    pub fn get<T: Versioned>(&mut self, key: &T::Key) -> Result<Option<Ref<'_, T>>> {
        self.ensure_live()?;
        let table = self.table::<T>()?;
        let snapshot = self.statement_snapshot();

        let Some(slot) = table.slot(key, &self.guard) else {
            // Record the absence too: a `Serializable` transaction that acted
            // on "this key does not exist" must abort if someone creates it.
            if I::VALIDATES_READS {
                let slot = table.slot_or_create(key, &self.guard);
                let state = self
                    .state
                    .as_ref()
                    .expect("SSI state is only touched when I::VALIDATES_READS");
                slot.readers.lock().register(state);
                self.reads.push(Box::new(SlotRead::<T> {
                    slot,
                    observed: None,
                }));
            }
            return Ok(None);
        };

        let version = slot.read(snapshot, self.id, &self.guard);

        if I::VALIDATES_READS {
            // SIREAD lock: a later writer of this slot needs to know we read it.
            let state = self
                .state
                .as_ref()
                .expect("SSI state is only touched when I::VALIDATES_READS");
            slot.readers.lock().register(state);
            self.reads.push(Box::new(SlotRead::<T> {
                slot,
                observed: version.map(|v| v as *const Version<T> as usize),
            }));
        }

        // A visible version with no value is a tombstone: the record was
        // deleted at or before this snapshot.
        Ok(version
            .filter(|v| v.value.is_some())
            .map(|v| Ref { version: v }))
    }

    /// Install a new version of `key`'s record, or fail if the slot is
    /// contended or was committed under us.
    fn write<T: Versioned>(
        &mut self,
        table: &'db Table<T>,
        key: &T::Key,
        value: Option<T>,
    ) -> Result<()> {
        let slot = table.slot_or_create(key, &self.guard);

        // First-updater-wins, part one: whoever takes the lock owns the slot
        // until it commits or aborts.
        if !slot.try_lock(self.id) {
            return Err(Error::WriteConflict {
                table: T::TABLE_NAME,
            });
        }

        let replaced = {
            let shared = slot.latest.load(Ordering::Acquire, &self.guard);
            // SAFETY: reachable from the chain under our pin, so still alive.
            unsafe { shared.as_ref() }
        };

        // First-updater-wins, part two: someone may have committed and released
        // the lock between our snapshot and now. Writing on top of that version
        // would silently lose their update.
        if I::FIRST_COMMITTER_WINS
            && let Some(current) = &replaced
            && let Visibility::CommittedAt(ts) =
                Visibility::decode(current.begin.load(Ordering::Acquire))
            && ts > self.snapshot
        {
            slot.unlock(self.id);
            return Err(Error::WriteConflict {
                table: T::TABLE_NAME,
            });
        }

        if let Some(record) = &value {
            table.index_record(record, replaced.and_then(|v| v.value.as_ref()));
        }

        let installed = Owned::new(Version {
            // Tagged as in-flight: invisible to everyone but us until commit
            // overwrites this with a real timestamp.
            begin: std::sync::atomic::AtomicU64::new(self.id.tagged()),
            end: std::sync::atomic::AtomicU64::new(Timestamp::MAX.raw()),
            prev: crossbeam_epoch::Atomic::null(),
            value,
            writer: self.state.clone(),
        });
        if let Some(prev) = replaced {
            installed
                .prev
                .store(Shared::from(prev as *const Version<T>), Ordering::Relaxed);
        }

        let installed = installed.into_shared(&self.guard);
        slot.latest.store(installed, Ordering::Release);

        self.writes.push(Box::new(SlotWrite {
            table,
            slot,
            installed: installed.as_raw(),
            replaced: replaced
                .map(|v| v as *const Version<T>)
                .unwrap_or(std::ptr::null()),
            txn: self.id,
        }));
        Ok(())
    }

    /// Insert a new record.
    ///
    /// Fails with [`Error::DuplicateKey`] if the primary key already has a
    /// visible, non-deleted version, or if a unique index would be violated.
    pub fn insert<T: Versioned>(&mut self, value: T) -> Result<()> {
        self.ensure_live()?;
        let table = self.table::<T>()?;
        let key = value.key();
        let snapshot = self.statement_snapshot();

        if let Some(slot) = table.slot(&key, &self.guard)
            && slot
                .read(snapshot, self.id, &self.guard)
                .is_some_and(|v| v.value.is_some())
        {
            return Err(Error::DuplicateKey {
                table: T::TABLE_NAME,
                index: "primary_key",
            });
        }

        if let Some(index) = table.unique_violation(&value, snapshot, self.id, &self.guard) {
            return Err(Error::DuplicateKey {
                table: T::TABLE_NAME,
                index,
            });
        }

        self.write(table, &key, Some(value))
    }

    /// Read-modify-write by primary key. Returns `false` if the record does not
    /// exist at this snapshot.
    ///
    /// A closure rather than a `&mut` guard, deliberately: the engine needs to
    /// know exactly when mutation ends so it can install the version and update
    /// the indexes. A guard doing that work in `Drop` could not report a write
    /// conflict, because `Drop` has nowhere to return an error to.
    pub fn update<T: Versioned>(&mut self, key: &T::Key, f: impl FnOnce(&mut T)) -> Result<bool> {
        self.ensure_live()?;
        let table = self.table::<T>()?;
        let snapshot = self.statement_snapshot();

        let Some(slot) = table.slot(key, &self.guard) else {
            return Ok(false);
        };
        let Some(version) = slot.read(snapshot, self.id, &self.guard) else {
            return Ok(false);
        };
        let Some(current) = version.value.as_ref() else {
            return Ok(false);
        };

        let mut next = current.clone();
        f(&mut next);

        if next.key() != *key {
            // Changing the primary key would move the record to a different
            // slot, leaving the old one holding a stale value under a key that
            // no longer matches it.
            return Err(Error::PrimaryKeyChanged {
                table: T::TABLE_NAME,
            });
        }

        if let Some(index) = table.unique_violation(&next, snapshot, self.id, &self.guard) {
            return Err(Error::DuplicateKey {
                table: T::TABLE_NAME,
                index,
            });
        }

        self.write(table, key, Some(next))?;
        Ok(true)
    }

    /// Delete by primary key. Returns `false` if the record does not exist at
    /// this snapshot.
    ///
    /// Installs a tombstone version rather than removing anything, so readers
    /// at older snapshots still see the record alive.
    pub fn delete<T: Versioned>(&mut self, key: &T::Key) -> Result<bool> {
        self.ensure_live()?;
        let table = self.table::<T>()?;
        let snapshot = self.statement_snapshot();

        let Some(slot) = table.slot(key, &self.guard) else {
            return Ok(false);
        };
        if slot
            .read(snapshot, self.id, &self.guard)
            .is_none_or(|v| v.value.is_none())
        {
            return Ok(false);
        }

        self.write::<T>(table, key, None)?;
        Ok(true)
    }

    /// Every record of `T` visible at this snapshot, in primary key order.
    ///
    /// Equivalent to [`Transaction::scan_where`] with a predicate that accepts
    /// everything. Under `Serializable` that means the *whole table* becomes
    /// part of the read set, so any concurrent insert or update anywhere in it
    /// will abort this transaction. Prefer `scan_where` when you have a
    /// predicate: it is both faster to validate and far less likely to abort.
    pub fn scan<T: Versioned>(&mut self) -> Result<Vec<Ref<'_, T>>> {
        self.scan_where::<T, _>(|_| true)
    }

    /// Every record of `T` visible at this snapshot that satisfies `predicate`,
    /// in primary key order.
    ///
    /// The predicate is handed to the engine rather than applied by the caller
    /// afterwards, and that is the entire point under `Serializable`: the
    /// engine records it, re-evaluates it at commit, and aborts if the set of
    /// matching rows changed. That is what makes phantoms visible — see
    /// `PredicateRead` — and it is why
    ///
    /// ```ignore
    /// tx.scan_where::<Account, _>(|a| a.balance < 0)?
    /// ```
    ///
    /// is a materially stronger statement than
    ///
    /// ```ignore
    /// tx.scan::<Account>()?.into_iter().filter(|a| a.balance < 0)
    /// ```
    ///
    /// The second form tells the engine only that you read the whole table.
    ///
    /// `predicate` must be `Send + Sync + 'static` because it is retained for
    /// the transaction's lifetime and re-run at commit. It should be pure:
    /// it will be called more than once, and on rows this call never returns.
    pub fn scan_where<T, P>(&mut self, predicate: P) -> Result<Vec<Ref<'_, T>>>
    where
        T: Versioned,
        P: Fn(&T) -> bool + Send + Sync + 'static,
    {
        self.ensure_live()?;
        let table = self.table::<T>()?;
        let snapshot = self.statement_snapshot();

        let predicate: Arc<dyn Fn(&T) -> bool + Send + Sync> = Arc::new(predicate);

        let matched = if I::VALIDATES_READS {
            // The read set is recorded as seen by *nobody*, so this
            // transaction's own uncommitted writes stay out of it — otherwise
            // inserting a row that matches your own predicate would abort you.
            // `matching_pair` produces that view alongside the caller's from a
            // single table pass; they differ only on rows written here.
            let (matched, committed) =
                table.matching_pair(snapshot, self.id, &*predicate, &self.guard);
            let observed: Vec<(T::Key, usize)> = committed
                .iter()
                .map(|(k, v)| (k.clone(), *v as *const Version<T> as usize))
                .collect();
            // Predicate SIREAD lock: a later insert has to be checked against
            // this predicate, since a row that does not exist yet has no slot
            // to register on. This is what makes phantoms detectable.
            table.register_predicate(self.ssi(), Arc::clone(&predicate));
            self.reads.push(Box::new(PredicateRead {
                table,
                predicate,
                observed,
            }));
            matched
        } else {
            table.matching(snapshot, self.id, &*predicate, &self.guard)
        };

        Ok(matched
            .into_iter()
            .map(|(_, version)| Ref { version })
            .collect())
    }

    /// Range scan over a secondary index, in index key order.
    ///
    /// The index is named by the `&'static str` the derive produced, so a typo
    /// is an error rather than an empty result.
    ///
    /// Under `Serializable` the range is recorded and re-scanned at commit, so
    /// a concurrent insert *into the range* aborts this transaction while one
    /// outside it does not.
    pub fn scan_index<T, K, R>(&mut self, index: &str, range: R) -> Result<Vec<Ref<'_, T>>>
    where
        T: Versioned,
        K: Encodable,
        R: RangeBounds<K>,
    {
        self.ensure_live()?;
        let table = self.table::<T>()?;
        let snapshot = self.statement_snapshot();

        let position = table
            .index_position(index)
            .ok_or_else(|| Error::NoSuchIndex {
                table: T::TABLE_NAME,
                index: index.to_string(),
            })?;

        let encode_bound = |b: Bound<&K>| match b {
            Bound::Included(k) => Bound::Included(k.encode()),
            Bound::Excluded(k) => Bound::Excluded(k.encode()),
            Bound::Unbounded => Bound::Unbounded,
        };
        let lo = encode_bound(range.start_bound());
        let hi = encode_bound(range.end_bound());

        let matched = table.matching_in_index(position, &lo, &hi, snapshot, self.id, &self.guard);

        if I::VALIDATES_READS {
            let observed: Vec<(T::Key, usize)> = table
                .matching_in_index(position, &lo, &hi, snapshot, TxnId::NONE, &self.guard)
                .iter()
                .map(|(k, v)| (k.clone(), *v as *const Version<T> as usize))
                .collect();

            // The range is a predicate over the indexed column, so it registers
            // like any other: an insert landing inside the range is a phantom
            // for this read.
            let extract = T::indexes()[position].extract;
            let (plo, phi) = (lo.clone(), hi.clone());
            table.register_predicate(
                self.ssi(),
                Arc::new(move |record: &T| {
                    let k = extract(record);
                    let lo_ok = match &plo {
                        Bound::Included(b) => k >= *b,
                        Bound::Excluded(b) => k > *b,
                        Bound::Unbounded => true,
                    };
                    let hi_ok = match &phi {
                        Bound::Included(b) => k <= *b,
                        Bound::Excluded(b) => k < *b,
                        Bound::Unbounded => true,
                    };
                    lo_ok && hi_ok
                }),
            );

            self.reads.push(Box::new(IndexRangeRead {
                table,
                position,
                lo,
                hi,
                observed,
            }));
        }

        Ok(matched
            .into_iter()
            .map(|(_, version)| Ref { version })
            .collect())
    }

    /// Find this transaction's rw-antidependency edges and record them on the
    /// shared [`TxnState`]s involved.
    ///
    /// Returns `false` if the transaction must abort. **Runs without any global
    /// lock**, which is what keeps the expensive half of commit — predicate
    /// re-scans especially — off the serialized path.
    ///
    /// Three things make that safe:
    ///
    /// - Revalidation reads at a *fixed* `now`, so it is a snapshot read and
    ///   cannot tear no matter what commits alongside it.
    /// - The decision variable is a pair of atomic flags, not the scan result,
    ///   and **both parties to an edge set both flags**. An edge that forms
    ///   after this scan is still recorded, by the other side.
    /// - The final [`TxnState::is_pivot`] check happens later, under the commit
    ///   lock, and flags only ever go false → true.
    ///
    /// The predecessor to this was "abort if anything I read changed", i.e. the
    /// outgoing edge alone. That is sound but aborts transactions that are in
    /// no cycle at all — a transaction that read a row someone else updated,
    /// but whose own writes nobody read, can always be ordered before that
    /// someone. Requiring both edges is Cahill's rule and is still sufficient:
    /// every cycle contains a transaction with two consecutive rw edges.
    fn detect_conflicts(&self) -> bool {
        if self.ssi().is_aborted() {
            return false;
        }

        let now = self.db.oracle().statement_snapshot();

        // --- outgoing edges: did anything I read change under me? -----------
        for read in &self.reads {
            let outcome = read.revalidate(now, &self.guard);
            if !outcome.changed {
                continue;
            }
            self.ssi().set_out_conflict();

            for writer in outcome.writers {
                // The other side of the same edge. It may already have
                // committed, in which case marking it is how a *later*
                // transaction learns the structure exists.
                writer.set_in_conflict();

                // If naming that edge just turned an already-committed
                // transaction into a pivot, the cycle runs through a
                // transaction we can no longer abort. Abort ourselves instead:
                // we are a participant, so removing us breaks it too.
                if writer.is_pivot() && writer.is_committed() {
                    return false;
                }
            }
        }

        // --- incoming edges: did anyone read what I overwrote? --------------
        // Only worth asking if we have an outgoing edge, since a pivot needs
        // both and this half is the expensive one.
        if self.ssi().is_pivot() {
            return false;
        }
        if self.ssi().has_out_conflict() {
            // Computed here rather than up front: `gc_watermark` locks every
            // one of the oracle's shards, and this branch is the only thing
            // that wants it. A transaction whose read set did not change — the
            // common case — was paying 16 mutex acquisitions on cache lines
            // other threads are actively writing, for a value it discarded.
            let gc = self.db.oracle().gc_watermark();
            for write in &self.writes {
                for reader in write.conflicting_readers(self.id, gc) {
                    reader.set_out_conflict();
                    self.ssi().set_in_conflict();
                }
            }
        }

        !self.ssi().is_pivot()
    }

    /// Commit.
    ///
    /// Consumes the transaction, so use-after-commit is a compile error rather
    /// than a runtime check.
    pub fn commit(mut self) -> Result<()> {
        self.ensure_live()?;

        // A transaction that wrote nothing never needs validating.
        //
        // It writes no versions, so it creates no dependency edge out of itself
        // and cannot be part of a cycle. That much is true of any scheme. What
        // makes *skipping the check* sound here specifically:
        //
        // - every read-write transaction is validated, so each is serializable
        //   at its own commit timestamp;
        // - `Oracle::publish` advances the read watermark by a min-reduction
        //   over in-flight commits, so the watermark never passes a commit that
        //   is still installing;
        // - therefore a snapshot `s` is exactly the set of transactions with
        //   `commit_ts <= s` — a *prefix* of the commit order, and so a state
        //   some serial execution actually produces.
        //
        // Reading a consistent prefix and writing nothing is serializable by
        // construction, ordered before everything committed after `s`.
        //
        // This does not contradict the read-only anomaly of Fekete et al.: that
        // anomaly needs the two read-write transactions to interleave
        // non-serializably, which plain snapshot isolation permits and this
        // validation does not.
        let needs_validation = I::VALIDATES_READS && !self.writes.is_empty();

        // Conflict detection runs *outside* the critical section. See
        // `detect_conflicts` for why that is safe.
        if needs_validation && !self.detect_conflicts() {
            self.rollback();
            return Err(Error::SerializationFailure);
        }

        {
            // Only `Serializable` needs global coordination, and only for the
            // decision itself.
            //
            // Everything else in this block is already safe without it:
            // `begin_commit` is one atomic increment; each version is stamped in
            // a slot this transaction holds exclusively via first-updater-wins;
            // and the read watermark does not advance past `ts` until `publish`,
            // so no reader can observe a half-stamped commit. That last property
            // is the whole job of the watermark, and it is what lets the weaker
            // levels commit with no global lock at all.
            //
            // What the lock still buys `Serializable`: two transactions that
            // are pivots for *each other* must not both pass their own
            // `is_pivot` check and commit. Serializing the decision-and-stamp
            // step means whichever goes second sees the first's flags.
            let _decision = needs_validation.then(|| self.db.commit_lock());

            // Re-read under the lock. Flags only go false → true, so an edge
            // recorded since `detect_conflicts` ran is caught here.
            if needs_validation && self.ssi().is_pivot() {
                drop(_decision);
                self.rollback();
                return Err(Error::SerializationFailure);
            }

            let ts = self.db.oracle().begin_commit();
            for write in &self.writes {
                write.commit(ts);
            }
            self.db.oracle().publish(ts);
            if let Some(state) = &self.state {
                state.mark_committed(ts);
            }
        }

        self.db.oracle().release_snapshot(self.id, self.snapshot);
        self.done = true;
        Ok(())
    }

    /// Abort and discard every write.
    ///
    /// Infallible: rollback unlinks version pointers, which cannot fail.
    pub fn abort(mut self) {
        self.rollback();
    }

    fn rollback(&mut self) {
        if self.done {
            return;
        }
        // Reverse order, so that a slot written more than once by this
        // transaction is restored to the version it had before the first write.
        for write in self.writes.iter().rev() {
            write.abort();
        }
        // Frees this transaction's SIREAD locks: an aborted transaction's reads
        // never constrained anyone. See `TxnState::is_expired`.
        if let Some(state) = &self.state {
            state.mark_aborted();
        }
        self.db.oracle().release_snapshot(self.id, self.snapshot);
        self.done = true;
    }
}

impl<'db, I: IsolationLevel> Drop for Transaction<'db, I> {
    fn drop(&mut self) {
        // Reaching here without `done` means `commit` was never called — an
        // early return, a `?`, or a panic. Rolling back silently is the only
        // safe default; silently committing would be far worse.
        self.rollback();
    }
}

/// A snapshot-consistent view of a record.
///
/// Holds the version alive, so it stays valid for as long as you keep it, even
/// as other transactions overwrite the record.
pub struct Ref<'txn, T> {
    version: &'txn Version<T>,
}

impl<T> std::ops::Deref for Ref<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.version
            .value
            .as_ref()
            .expect("a Ref is only constructed for a version with a value")
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Ref<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
    }
}

impl<T: Clone> Ref<'_, T> {
    /// Copy the record out, detaching it from the version chain.
    pub fn to_owned(&self) -> T {
        (**self).clone()
    }
}
