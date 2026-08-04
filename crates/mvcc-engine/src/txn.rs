//! The transaction handle — the type users actually touch.

use std::ops::{Bound, RangeBounds};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use mvcc_core::isolation::{ReadTracker, SlotRef};
use mvcc_core::{
    Encodable, Error, IsolationLevel, Result, Timestamp, TxnId, Versioned, Visibility,
};

use crate::store::{Database, Slot, Version};

/// A write this transaction has installed but not yet committed.
///
/// Type-erased so one `Transaction` can hold writes to many different tables.
trait WriteOp: Send + Sync {
    /// Stamp the new version visible at `ts` and retire the one it replaced.
    fn commit(&self, ts: Timestamp);
    /// Unlink the new version, restoring the slot to what it was.
    fn abort(&self);
}

struct SlotWrite<T> {
    slot: Arc<Slot<T>>,
    installed: Arc<Version<T>>,
    /// The version `installed` displaced, whose `end` must be stamped on commit
    /// and which the slot is restored to on abort.
    replaced: Option<Arc<Version<T>>>,
    txn: TxnId,
}

impl<T: Send + Sync> WriteOp for SlotWrite<T> {
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
}

/// Re-checks at commit that a value this transaction read has not changed.
trait ReadValidation: Send + Sync {
    fn still_valid(&self, now: Timestamp) -> bool;
}

struct SlotRead<T> {
    slot: Arc<Slot<T>>,
    observed: Option<Arc<Version<T>>>,
}

impl<T: Send + Sync> ReadValidation for SlotRead<T> {
    fn still_valid(&self, now: Timestamp) -> bool {
        // Read as nobody, so this transaction's own in-flight writes stay
        // invisible. Validating as ourselves would compare what we read against
        // what we then wrote, and every read-modify-write would abort itself.
        let current = self.slot.read(now, TxnId::NONE);
        match (&self.observed, &current) {
            (None, None) => true,
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
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
    tracker: I::ReadTracker,
    writes: Vec<Box<dyn WriteOp>>,
    /// Empty, and never pushed to, unless `I::VALIDATES_READS`.
    reads: Vec<Box<dyn ReadValidation>>,
    done: bool,
}

impl<'db, I: IsolationLevel> Transaction<'db, I> {
    pub(crate) fn new(db: &'db Database) -> Self {
        Transaction {
            id: db.oracle().next_txn_id(),
            snapshot: db.oracle().begin_snapshot(),
            db,
            tracker: I::ReadTracker::default(),
            writes: Vec::new(),
            reads: Vec::new(),
            done: false,
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
                self.reads.push(Box::new(SlotRead::<T> {
                    slot,
                    observed: None,
                }));
            }
            return Ok(None);
        };

        let version = slot.read(snapshot, self.id);

        if I::VALIDATES_READS {
            self.tracker
                .observe(SlotRef(Arc::as_ptr(&slot) as usize), snapshot);
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
    fn write<T: Versioned>(&mut self, key: &T::Key, value: Option<T>) -> Result<()> {
        let table = self.db.table::<T>()?;
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
        });

        *slot.latest.write() = Some(installed.clone());

        self.writes.push(Box::new(SlotWrite {
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

        self.write(&key, Some(value))
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

        self.write(key, Some(next))?;
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

        self.write::<T>(key, None)?;
        Ok(true)
    }

    /// Every record of `T` visible at this snapshot, in primary key order.
    pub fn scan<T: Versioned>(&mut self) -> Result<Vec<Ref<T>>> {
        self.ensure_live()?;
        let table = self.db.table::<T>()?;
        let snapshot = self.statement_snapshot();

        let mut out = Vec::new();
        for key in table.all_keys() {
            let Some(slot) = table.slot(&key) else {
                continue;
            };
            let Some(version) = slot.read(snapshot, self.id) else {
                continue;
            };
            if version.value.is_none() {
                continue;
            }
            if I::VALIDATES_READS {
                self.reads.push(Box::new(SlotRead::<T> {
                    slot: slot.clone(),
                    observed: Some(version.clone()),
                }));
            }
            out.push(Ref { version });
        }
        Ok(out)
    }

    /// Range scan over a secondary index, in index key order.
    ///
    /// The index is named by the `&'static str` the derive produced, so a typo
    /// is an error rather than an empty result.
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
        let desc = &T::indexes()[position];

        let encode_bound = |b: Bound<&K>| match b {
            Bound::Included(k) => Bound::Included(k.encode()),
            Bound::Excluded(k) => Bound::Excluded(k.encode()),
            Bound::Unbounded => Bound::Unbounded,
        };

        let candidates = table.index_candidates(
            position,
            encode_bound(range.start_bound()),
            encode_bound(range.end_bound()),
        );

        // Index entries are candidates, not answers — see `Table::secondary`.
        // Each one is resolved to its visible version and its key re-extracted,
        // which also filters out entries left behind by updates and rollbacks.
        let mut out = Vec::new();
        for key in candidates {
            let Some(slot) = table.slot(&key) else {
                continue;
            };
            let Some(version) = slot.read(snapshot, self.id) else {
                continue;
            };
            let Some(value) = version.value.as_ref() else {
                continue;
            };

            let actual = (desc.extract)(value);
            let (lo, hi) = (range.start_bound(), range.end_bound());
            let in_range = {
                let lo_ok = match lo {
                    Bound::Included(k) => actual >= k.encode(),
                    Bound::Excluded(k) => actual > k.encode(),
                    Bound::Unbounded => true,
                };
                let hi_ok = match hi {
                    Bound::Included(k) => actual <= k.encode(),
                    Bound::Excluded(k) => actual < k.encode(),
                    Bound::Unbounded => true,
                };
                lo_ok && hi_ok
            };
            if !in_range {
                continue;
            }

            if I::VALIDATES_READS {
                self.reads.push(Box::new(SlotRead::<T> {
                    slot: slot.clone(),
                    observed: Some(version.clone()),
                }));
            }
            out.push(Ref { version });
        }

        out.sort_by_key(|r| {
            (desc.extract)(r.version.value.as_ref().expect("filtered above"))
                .0
                .clone()
        });
        Ok(out)
    }

    /// Commit.
    ///
    /// Consumes the transaction, so use-after-commit is a compile error rather
    /// than a runtime check.
    pub fn commit(mut self) -> Result<()> {
        self.ensure_live()?;

        {
            // One global critical section covering validation, timestamp
            // allocation and stamping. Correct but not scalable: this is the
            // next thing to shard, and the reason the oracle already handles
            // out-of-order installs — removing this lock must not change the
            // semantics.
            let _guard = self.db.commit_lock();

            if I::VALIDATES_READS {
                let now = self.db.oracle().statement_snapshot();
                // Conservative OCC validation: abort if anything we read has
                // changed. More aggressive than SSI's pivot detection, which
                // would let through some of these, but it is sound and it is
                // what makes write skew impossible at this level.
                if !self.reads.iter().all(|r| r.still_valid(now)) || !self.tracker.validate() {
                    drop(_guard);
                    self.rollback();
                    return Err(Error::SerializationFailure);
                }
            }

            let ts = self.db.oracle().begin_commit();
            for write in &self.writes {
                write.commit(ts);
            }
            // Publish while still holding the lock, so that the next
            // transaction to validate its read set is guaranteed to see this
            // commit. The oracle's min-reduction handles out-of-order publishes
            // correctly on its own; this is belt-and-braces for as long as the
            // global lock exists, and the two must stay consistent when it goes.
            self.db.oracle().publish(ts);
        };

        self.db.oracle().release_snapshot(self.snapshot);
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
        self.db.oracle().release_snapshot(self.snapshot);
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
