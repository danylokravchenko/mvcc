//! The transaction handle — the type users actually touch.

use std::marker::PhantomData;
use std::ops::{Bound, RangeBounds};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use mvcc_core::{
    Encodable, Error, IndexKey, IsolationLevel, Result, Timestamp, TxnId, Versioned, Visibility,
};

use crate::ssi::TxnState;
use crate::store::{Database, Slot, Table, Version};

/// A write this transaction has installed but not yet committed.
///
/// Type-erased so one `Transaction` can hold writes to many different tables.
trait WriteOp: Send + Sync {
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

struct SlotWrite<T: Versioned> {
    table: Arc<Table<T>>,
    slot: Arc<Slot<T>>,
    installed: Arc<Version<T>>,
    /// The version `installed` displaced, whose `end` must be stamped on commit
    /// and which the slot is restored to on abort.
    replaced: Option<Arc<Version<T>>>,
    txn: TxnId,
}

impl<T: Versioned> WriteOp for SlotWrite<T> {
    fn commit(&self, ts: Timestamp) {
        // Order matters. The new version becomes visible only once `begin`
        // holds a real timestamp, so stamping the old version's `end` first
        // would briefly leave the record invisible to concurrent readers.
        if let Some(prev) = &self.replaced {
            prev.end.store(ts.raw(), Ordering::Release);
        }
        self.installed.begin.store(ts.raw(), Ordering::Release);
        self.slot.unlock(self.txn);
    }

    fn abort(&self) {
        *self.slot.latest.write() = self.replaced.clone();
        self.slot.unlock(self.txn);
    }

    fn conflicting_readers(&self, txn: TxnId, gc: Timestamp) -> Vec<Arc<TxnState>> {
        let mut readers = self.slot.readers.lock().others(txn, gc);
        if let Some(value) = self.installed.value.as_ref() {
            readers.extend(self.table.predicate_readers_of(value, txn, gc));
        }
        // A delete makes a row vanish from predicates that matched it, so the
        // version it replaced has to be checked too.
        if let Some(previous) = self.replaced.as_ref().and_then(|v| v.value.as_ref()) {
            readers.extend(self.table.predicate_readers_of(previous, txn, gc));
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
trait ReadValidation: Send + Sync {
    fn conflicting_writers(&self, now: Timestamp) -> Vec<Arc<TxnState>>;
}

struct SlotRead<T> {
    slot: Arc<Slot<T>>,
    observed: Option<Arc<Version<T>>>,
}

impl<T: Send + Sync> ReadValidation for SlotRead<T> {
    fn conflicting_writers(&self, now: Timestamp) -> Vec<Arc<TxnState>> {
        // Read as nobody, so this transaction's own in-flight writes stay
        // invisible. Validating as ourselves would compare what we read against
        // what we then wrote, and every read-modify-write would abort itself.
        let current = self.slot.read(now, TxnId::NONE);
        let unchanged = match (&self.observed, &current) {
            (None, None) => true,
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        };
        if unchanged {
            Vec::new()
        } else {
            current.and_then(|v| v.writer.clone()).into_iter().collect()
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
struct PredicateRead<T: Versioned> {
    table: Arc<Table<T>>,
    predicate: Arc<dyn Fn(&T) -> bool + Send + Sync>,
    observed: Vec<(T::Key, Arc<Version<T>>)>,
}

impl<T: Versioned> ReadValidation for PredicateRead<T> {
    fn conflicting_writers(&self, now: Timestamp) -> Vec<Arc<TxnState>> {
        let current = self.table.matching(now, TxnId::NONE, &*self.predicate);
        writers_of_change(&self.table, &self.observed, &current, now)
    }
}

/// An index range a transaction scanned, and the rows it returned.
///
/// The range *is* the predicate, so revalidation re-runs the range scan rather
/// than a full table scan — the same reason the index exists in the first place.
struct IndexRangeRead<T: Versioned> {
    table: Arc<Table<T>>,
    position: usize,
    lo: Bound<IndexKey>,
    hi: Bound<IndexKey>,
    observed: Vec<(T::Key, Arc<Version<T>>)>,
}

impl<T: Versioned> ReadValidation for IndexRangeRead<T> {
    fn conflicting_writers(&self, now: Timestamp) -> Vec<Arc<TxnState>> {
        let current = self.table.matching_in_index(
            self.position,
            self.lo.clone(),
            self.hi.clone(),
            now,
            TxnId::NONE,
        );
        writers_of_change(&self.table, &self.observed, &current, now)
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
    observed: &[(T::Key, Arc<Version<T>>)],
    current: &[(T::Key, Arc<Version<T>>)],
    now: Timestamp,
) -> Vec<Arc<TxnState>> {
    let mut writers = Vec::new();

    for (key, version) in current {
        let same_as_observed = observed
            .iter()
            .any(|(ko, vo)| ko == key && Arc::ptr_eq(vo, version));
        if !same_as_observed {
            writers.extend(version.writer.clone());
        }
    }

    for (key, _) in observed {
        if current.iter().any(|(kc, _)| kc == key) {
            continue;
        }
        if let Some(slot) = table.slot(key)
            && let Some(version) = slot.read(now, TxnId::NONE)
        {
            writers.extend(version.writer.clone());
        }
    }

    writers
}

/// A transaction at isolation level `I`.
///
/// `I` is a typestate: `Transaction<'_, ReadCommitted>` and
/// `Transaction<'_, Serializable>` are different types with different generated
/// code. See `mvcc_core::isolation`.
pub struct Transaction<'db, I: IsolationLevel> {
    db: &'db Database,
    id: TxnId,
    snapshot: Timestamp,
    /// Shared with other transactions, which set this one's conflict flags.
    /// Outlives the `Transaction`: SIREAD locks and version records keep
    /// referring to it after commit. See [`crate::ssi`].
    state: Arc<TxnState>,
    writes: Vec<Box<dyn WriteOp>>,
    /// Empty, and never pushed to, unless `I::VALIDATES_READS`.
    reads: Vec<Box<dyn ReadValidation>>,
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
            state: TxnState::new(id, snapshot),
            db,
            writes: Vec::new(),
            reads: Vec::new(),
            done: false,
            _level: PhantomData,
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
    pub fn get<T: Versioned>(&mut self, key: &T::Key) -> Result<Option<Ref<T>>> {
        self.ensure_live()?;
        let table = self.db.table::<T>()?;
        let snapshot = self.statement_snapshot();

        let Some(slot) = table.slot(key) else {
            // Record the absence too: a `Serializable` transaction that acted
            // on "this key does not exist" must abort if someone creates it.
            if I::VALIDATES_READS {
                let slot = table.slot_or_create(key);
                slot.readers.lock().register(&self.state);
                self.reads.push(Box::new(SlotRead::<T> {
                    slot,
                    observed: None,
                }));
            }
            return Ok(None);
        };

        let version = slot.read(snapshot, self.id);

        if I::VALIDATES_READS {
            // SIREAD lock: a later writer of this slot needs to know we read it.
            slot.readers.lock().register(&self.state);
            self.reads.push(Box::new(SlotRead::<T> {
                slot: slot.clone(),
                observed: version.clone(),
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
        table: &Arc<Table<T>>,
        key: &T::Key,
        value: Option<T>,
    ) -> Result<()> {
        let slot = table.slot_or_create(key);

        // First-updater-wins, part one: whoever takes the lock owns the slot
        // until it commits or aborts.
        if !slot.try_lock(self.id) {
            return Err(Error::WriteConflict {
                table: T::TABLE_NAME,
            });
        }

        let replaced = slot.latest.read().clone();

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
            table.index_record(record);
        }

        let installed = Arc::new(Version {
            // Tagged as in-flight: invisible to everyone but us until commit
            // overwrites this with a real timestamp.
            begin: std::sync::atomic::AtomicU64::new(self.id.tagged()),
            end: std::sync::atomic::AtomicU64::new(Timestamp::MAX.raw()),
            prev: replaced.clone(),
            value,
            writer: Some(Arc::clone(&self.state)),
        });

        *slot.latest.write() = Some(installed.clone());

        self.writes.push(Box::new(SlotWrite {
            table: table.clone(),
            slot,
            installed,
            replaced,
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
        let table = self.db.table::<T>()?;
        let key = value.key();
        let snapshot = self.statement_snapshot();

        if let Some(slot) = table.slot(&key)
            && slot
                .read(snapshot, self.id)
                .is_some_and(|v| v.value.is_some())
        {
            return Err(Error::DuplicateKey {
                table: T::TABLE_NAME,
                index: "primary_key",
            });
        }

        if let Some(index) = table.unique_violation(&value, snapshot, self.id) {
            return Err(Error::DuplicateKey {
                table: T::TABLE_NAME,
                index,
            });
        }

        self.write(&table, &key, Some(value))
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
        let table = self.db.table::<T>()?;
        let snapshot = self.statement_snapshot();

        let Some(slot) = table.slot(key) else {
            return Ok(false);
        };
        let Some(version) = slot.read(snapshot, self.id) else {
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

        if let Some(index) = table.unique_violation(&next, snapshot, self.id) {
            return Err(Error::DuplicateKey {
                table: T::TABLE_NAME,
                index,
            });
        }

        self.write(&table, key, Some(next))?;
        Ok(true)
    }

    /// Delete by primary key. Returns `false` if the record does not exist at
    /// this snapshot.
    ///
    /// Installs a tombstone version rather than removing anything, so readers
    /// at older snapshots still see the record alive.
    pub fn delete<T: Versioned>(&mut self, key: &T::Key) -> Result<bool> {
        self.ensure_live()?;
        let table = self.db.table::<T>()?;
        let snapshot = self.statement_snapshot();

        let Some(slot) = table.slot(key) else {
            return Ok(false);
        };
        if slot
            .read(snapshot, self.id)
            .is_none_or(|v| v.value.is_none())
        {
            return Ok(false);
        }

        self.write::<T>(&table, key, None)?;
        Ok(true)
    }

    /// Every record of `T` visible at this snapshot, in primary key order.
    ///
    /// Equivalent to [`Transaction::scan_where`] with a predicate that accepts
    /// everything. Under `Serializable` that means the *whole table* becomes
    /// part of the read set, so any concurrent insert or update anywhere in it
    /// will abort this transaction. Prefer `scan_where` when you have a
    /// predicate: it is both faster to validate and far less likely to abort.
    pub fn scan<T: Versioned>(&mut self) -> Result<Vec<Ref<T>>> {
        self.scan_where::<T, _>(|_| true)
    }

    /// Every record of `T` visible at this snapshot that satisfies `predicate`,
    /// in primary key order.
    ///
    /// The predicate is handed to the engine rather than applied by the caller
    /// afterwards, and that is the entire point under `Serializable`: the
    /// engine records it, re-evaluates it at commit, and aborts if the set of
    /// matching rows changed. That is what makes phantoms visible — see
    /// [`PredicateRead`] — and it is why
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
    pub fn scan_where<T, P>(&mut self, predicate: P) -> Result<Vec<Ref<T>>>
    where
        T: Versioned,
        P: Fn(&T) -> bool + Send + Sync + 'static,
    {
        self.ensure_live()?;
        let table = self.db.table::<T>()?;
        let snapshot = self.statement_snapshot();

        let predicate: Arc<dyn Fn(&T) -> bool + Send + Sync> = Arc::new(predicate);
        let matched = table.matching(snapshot, self.id, &*predicate);

        if I::VALIDATES_READS {
            // Recorded as seen by *nobody*, so this transaction's own
            // uncommitted writes stay out of its own read set — otherwise
            // inserting a row that matches your own predicate would abort you.
            // With nothing written yet the two views coincide, which is the
            // common case and saves a second pass over the table.
            let observed = if self.writes.is_empty() {
                matched.clone()
            } else {
                table.matching(snapshot, TxnId::NONE, &*predicate)
            };
            // Predicate SIREAD lock: a later insert has to be checked against
            // this predicate, since a row that does not exist yet has no slot
            // to register on. This is what makes phantoms detectable.
            table.register_predicate(&self.state, Arc::clone(&predicate));
            self.reads.push(Box::new(PredicateRead {
                table: table.clone(),
                predicate,
                observed,
            }));
        }

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
    pub fn scan_index<T, K, R>(&mut self, index: &str, range: R) -> Result<Vec<Ref<T>>>
    where
        T: Versioned,
        K: Encodable,
        R: RangeBounds<K>,
    {
        self.ensure_live()?;
        let table = self.db.table::<T>()?;
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

        let matched =
            table.matching_in_index(position, lo.clone(), hi.clone(), snapshot, self.id);

        if I::VALIDATES_READS {
            let observed =
                table.matching_in_index(position, lo.clone(), hi.clone(), snapshot, TxnId::NONE);

            // The range is a predicate over the indexed column, so it registers
            // like any other: an insert landing inside the range is a phantom
            // for this read.
            let extract = T::indexes()[position].extract;
            let (plo, phi) = (lo.clone(), hi.clone());
            table.register_predicate(
                &self.state,
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
                table: table.clone(),
                position,
                lo,
                hi,
                observed,
            }));
        }

        Ok(matched.into_iter().map(|(_, version)| Ref { version }).collect())
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
        if self.state.is_aborted() {
            return false;
        }

        let now = self.db.oracle().statement_snapshot();
        let gc = self.db.oracle().gc_watermark();

        // --- outgoing edges: did anything I read change under me? -----------
        for read in &self.reads {
            for writer in read.conflicting_writers(now) {
                self.state.set_out_conflict();
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
        if self.state.is_pivot() {
            return false;
        }
        if self.state.has_out_conflict() {
            for write in &self.writes {
                for reader in write.conflicting_readers(self.id, gc) {
                    reader.set_out_conflict();
                    self.state.set_in_conflict();
                }
            }
        }

        !self.state.is_pivot()
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
            if needs_validation && self.state.is_pivot() {
                drop(_decision);
                self.rollback();
                return Err(Error::SerializationFailure);
            }

            let ts = self.db.oracle().begin_commit();
            for write in &self.writes {
                write.commit(ts);
            }
            self.db.oracle().publish(ts);
            self.state.mark_committed(ts);
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
        self.state.mark_aborted();
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
pub struct Ref<T> {
    version: Arc<Version<T>>,
}

impl<T> std::ops::Deref for Ref<T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.version
            .value
            .as_ref()
            .expect("a Ref is only constructed for a version with a value")
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Ref<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
    }
}

impl<T: Clone> Ref<T> {
    /// Copy the record out, detaching it from the version chain.
    pub fn to_owned(&self) -> T {
        (**self).clone()
    }
}
