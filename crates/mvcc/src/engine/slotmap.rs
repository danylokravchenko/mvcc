//! The primary key to slot map, with a read path that writes nothing.
//!
//! # Why not a sharded `RwLock<HashMap>`
//!
//! That is the obvious structure for this map, and it puts a shared write back
//! on the read path. A `parking_lot` *read* acquire is still an atomic
//! read-modify-write on the lock word, so every "lock-free" chain walk is
//! preceded by two RMWs on a line shared with every other reader of that shard.
//! Sharding 64 ways hides it when keys are uniform and does nothing at all when
//! they are not: four hot keys land in at most four shards however many shards
//! exist.
//!
//! Measured, by deleting the lock as a throwaway experiment and re-running
//! `cargo bench -- 4`:
//!
//! ```text
//!                          with lock    without
//!   uniform keys, 1 thread    49.3M       49.7M
//!   uniform keys, 4 threads  111.9M      177.0M
//!   4 hot rows,   1 thread    91.0M       99.0M
//!   4 hot rows,   4 threads    49.0M     351.8M
//! ```
//!
//! At one thread the lock costs nothing measurable. All of it is contention —
//! and under it the hot-row workload is the only one in the benchmark that runs
//! *backwards* with more threads, 91M at one to 49M at four. This is the same
//! defect the version chain avoids: see `Slot::latest`, where keeping an `Arc`
//! refcount off the read path is worth 37x on the same workload.
//!
//! # The design
//!
//! Open addressing with linear probing, per shard, behind an epoch-managed
//! pointer. **A lookup performs no writes of any kind**: load the bucket array,
//! probe, compare, done.
//!
//! What makes that sound is the one property this map has and a general-purpose
//! map does not — it is **append-only**. `get_or_create` inserts and nothing
//! ever removes: a delete installs a tombstone version rather than dropping a
//! slot. So:
//!
//! - **Records are immortal.** Each [`Record`] — a slot and the key it belongs
//!   to — is allocated once, never moved, and freed only when the whole map is
//!   dropped. That is what lets `get` hand out a `&Slot<T>` borrowed from the
//!   *map* rather than from a lock guard or a bucket array, and it is why a
//!   resize can copy raw pointers without disturbing anyone holding one.
//! - **Only the bucket array is reclaimed**, and only through the epoch already
//!   pinned for the caller's whole transaction. A reader that loaded the old
//!   array keeps using it; it stays allocated and its contents stay valid,
//!   because a resize copies entries rather than rewriting them.
//!
//! Writers serialise per shard on a `Mutex`, so entry publication has a single
//! writer and needs no compare-exchange. Readers never take it. Inserts keep
//! the 64-way parallelism the sharded `RwLock` had.
//!
//! # Publication
//!
//! An entry is published by storing its `hash` last, with `Release`. A reader
//! that observes a non-zero `hash` with `Acquire` therefore also observes the
//! `record` pointer stored before it, and everything written into the `Record`
//! before *that*. `hash == 0` means empty and terminates a probe, which is why
//! a real hash of zero is remapped to one.
//!
//! # Staleness
//!
//! A reader holding a superseded bucket array cannot see keys inserted after
//! the resize, and will report them absent. That is not a new race: the
//! `RwLock` version had the same one, narrower. It is harmless because a slot
//! that has just been created holds only an in-flight version, which is
//! invisible to every transaction but its author — and the author reaches it
//! through `get_or_create`, which re-probes the current array under the shard
//! lock.

use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

use crossbeam_epoch::{Atomic, Guard, Owned, Shared};
use parking_lot::Mutex;

use crate::core::Versioned;
use crate::engine::hash::hash_one;
use crate::engine::store::Slot;

/// Independent shards, to keep insert parallelism after the read path stopped
/// needing shards at all. Reads pick a shard without synchronising.
pub(crate) const SHARDS: usize = 64;

/// Entries per shard at construction. Grows by doubling.
const INITIAL_CAPACITY: usize = 16;

/// A slot and the primary key it belongs to.
///
/// Allocated once and never moved or freed while the map lives, which is the
/// property every borrow handed out by this module rests on.
pub(crate) struct Record<T: Versioned> {
    pub(crate) slot: Slot<T>,
    pub(crate) key: T::Key,
}

/// One bucket. Empty when `hash` is zero.
struct Entry<T: Versioned> {
    /// The key's hash, remapped away from zero. Published last, with `Release`.
    hash: AtomicU64,
    /// Never null once `hash` is non-zero. Not owning — see [`Record`].
    record: AtomicPtr<Record<T>>,
}

impl<T: Versioned> Entry<T> {
    fn empty() -> Self {
        Entry {
            hash: AtomicU64::new(0),
            record: AtomicPtr::new(std::ptr::null_mut()),
        }
    }
}

/// One generation of a shard's bucket array. `entries.len()` is a power of two.
struct Buckets<T: Versioned> {
    mask: usize,
    entries: Box<[Entry<T>]>,
}

impl<T: Versioned> Buckets<T> {
    fn with_capacity(capacity: usize) -> Self {
        debug_assert!(capacity.is_power_of_two());
        Buckets {
            mask: capacity - 1,
            entries: (0..capacity).map(|_| Entry::empty()).collect(),
        }
    }
}

/// Records in insertion order, for iteration.
///
/// A scan wants every record; the bucket array is the wrong shape to give it
/// one, because it is at most half full by construction and each live entry is
/// a pointer chase away from the next. Walking it cost the scan benchmarks 21
/// to 33% when this module first replaced the `HashMap`.
///
/// So iteration gets its own structure: a chunked append-only list, which is
/// dense, in cache order, and needs no emptiness test per element. Records are
/// immortal and never removed, so "append-only list" is the whole data
/// structure — and it doubles as the ownership record that [`SlotMap::drop`]
/// frees from.
const CHUNK: usize = 64;

struct Chunk<T: Versioned> {
    /// Filled left to right. A null terminates iteration: entries are appended
    /// under the writer lock in index order, so there are never holes.
    records: [AtomicPtr<Record<T>>; CHUNK],
    /// Allocated only once `records` is full, so a chunk with a null in it is
    /// always the last one.
    next: Atomic<Chunk<T>>,
}

impl<T: Versioned> Chunk<T> {
    fn new() -> Self {
        Chunk {
            records: [const { AtomicPtr::new(std::ptr::null_mut()) }; CHUNK],
            next: Atomic::null(),
        }
    }
}

struct Shard<T: Versioned> {
    buckets: Atomic<Buckets<T>>,
    /// Head of the insertion-order list. See [`CHUNK`].
    chunks: Atomic<Chunk<T>>,
    /// Serialises inserts and resizes, and holds the live entry count — which
    /// is also the next free index in the chunk list. Readers never take it.
    writers: Mutex<usize>,
}

pub(crate) struct SlotMap<T: Versioned> {
    shards: Box<[Shard<T>]>,
}

impl<T: Versioned> SlotMap<T> {
    pub(crate) fn new() -> Self {
        SlotMap {
            shards: (0..SHARDS)
                .map(|_| Shard {
                    buckets: Atomic::new(Buckets::with_capacity(INITIAL_CAPACITY)),
                    chunks: Atomic::new(Chunk::new()),
                    writers: Mutex::new(0),
                })
                .collect(),
        }
    }

    /// The key's hash, computed **once** for both shard selection and probing.
    ///
    /// The sharded `HashMap` this replaced hashed twice: once here to pick the
    /// shard, then again inside `HashMap::get`. Owning the table is what makes
    /// one hash possible; `std`'s map does not expose lookup by precomputed
    /// hash on stable.
    ///
    /// Zero is reserved for "empty bucket", so a key that hashes to it is
    /// remapped. The collision that creates costs one extra key comparison and
    /// nothing else.
    #[inline]
    fn hash(key: &T::Key) -> u64 {
        match hash_one(key) {
            0 => 1,
            h => h,
        }
    }

    #[inline]
    fn shard(&self, hash: u64) -> &Shard<T> {
        // The low bits pick the bucket, so take the shard from the high ones —
        // otherwise every key in a shard collides in its first probe position.
        &self.shards[(hash >> 32) as usize % SHARDS]
    }

    /// The slot for `key`, or `None`. Takes no locks and writes nothing.
    ///
    /// The borrow is from `&self`, not from `guard`: the [`Record`] outlives
    /// every epoch, and only the bucket array read on the way to it is
    /// epoch-managed.
    pub(crate) fn get<'a>(&'a self, key: &T::Key, guard: &Guard) -> Option<&'a Slot<T>> {
        let hash = Self::hash(key);
        Some(&Self::probe(self.shard(hash), hash, key, guard)?.slot)
    }

    /// Find `key`'s record in `shard`'s current bucket array.
    fn probe<'a>(
        shard: &Shard<T>,
        hash: u64,
        key: &T::Key,
        guard: &Guard,
    ) -> Option<&'a Record<T>> {
        // SAFETY: the array is replaced only by `grow`, which defers the old
        // one's destruction to the epoch. `guard` is pinned, so whichever
        // generation this load returns stays allocated for the probe.
        let buckets = unsafe { shard.buckets.load(Ordering::Acquire, guard).deref() };

        let mut i = hash as usize & buckets.mask;
        loop {
            let entry = &buckets.entries[i];
            match entry.hash.load(Ordering::Acquire) {
                // Terminates: `grow` keeps the array at most half full, so
                // every probe reaches an empty bucket.
                0 => return None,
                h if h == hash => {
                    let record = entry.record.load(Ordering::Acquire);
                    // SAFETY: a non-zero `hash` was published after `record`
                    // with `Release`, so the pointer is visible and non-null;
                    // and records are never freed while the map lives.
                    let record = unsafe { &*record };
                    if record.key == *key {
                        return Some(record);
                    }
                }
                _ => {}
            }
            i = (i + 1) & buckets.mask;
        }
    }

    /// The slot for `key`, creating an empty one if absent.
    pub(crate) fn get_or_create<'a>(&'a self, key: &T::Key, guard: &Guard) -> &'a Slot<T> {
        let hash = Self::hash(key);
        let shard = self.shard(hash);
        if let Some(record) = Self::probe(shard, hash, key, guard) {
            return &record.slot;
        }

        let mut count = shard.writers.lock();
        // Re-probe under the lock. Without this two threads racing on a new key
        // would each install a record, and a key with two slots is a lost
        // update rather than a merely wasted allocation.
        if let Some(record) = Self::probe(shard, hash, key, guard) {
            return &record.slot;
        }

        // Half full at most, so a probe always terminates on an empty bucket.
        // SAFETY: only this section replaces the array, and it holds the lock.
        let buckets = unsafe { shard.buckets.load(Ordering::Acquire, guard).deref() };
        if (*count + 1) * 2 > buckets.entries.len() {
            self.grow(shard, guard);
        }
        // SAFETY: as above; reloaded because `grow` may have replaced it.
        let buckets = unsafe { shard.buckets.load(Ordering::Acquire, guard).deref() };

        let record = Box::into_raw(Box::new(Record {
            slot: Slot::new(),
            key: key.clone(),
        }));
        Self::place(buckets, hash, record);
        Self::append(shard, *count, record, guard);
        *count += 1;

        // SAFETY: just allocated, published, and never freed until the map is
        // dropped — which cannot happen while `&'a self` is borrowed.
        &unsafe { &*record }.slot
    }

    /// Append `record` at position `index` of the shard's chunk list.
    ///
    /// Caller must hold the shard's writer lock, so `index` is this shard's
    /// live count and chunks fill strictly left to right with no holes.
    fn append(shard: &Shard<T>, index: usize, record: *mut Record<T>, guard: &Guard) {
        let mut chunk = shard.chunks.load(Ordering::Acquire, guard);
        for _ in 0..index / CHUNK {
            // SAFETY: chunk `n` exists because appending the record at
            // `n * CHUNK - 1` linked it, and chunks are never freed while the
            // map lives.
            let next = unsafe { chunk.deref() }.next.load(Ordering::Acquire, guard);
            chunk = if next.is_null() {
                let fresh = Owned::new(Chunk::new()).into_shared(guard);
                // SAFETY: as above. Only the lock holder links a chunk, so this
                // store cannot race another writer.
                unsafe { chunk.deref() }
                    .next
                    .store(fresh, Ordering::Release);
                fresh
            } else {
                next
            };
        }
        // SAFETY: as above.
        unsafe { chunk.deref() }.records[index % CHUNK].store(record, Ordering::Release);
    }

    /// Install `record` in the first empty bucket on its probe path.
    ///
    /// Caller must hold the shard's writer lock and have ensured there is room.
    /// Storing `hash` last is what publishes the entry; see the module docs.
    fn place(buckets: &Buckets<T>, hash: u64, record: *mut Record<T>) {
        let mut i = hash as usize & buckets.mask;
        loop {
            let entry = &buckets.entries[i];
            if entry.hash.load(Ordering::Relaxed) == 0 {
                entry.record.store(record, Ordering::Release);
                entry.hash.store(hash, Ordering::Release);
                return;
            }
            i = (i + 1) & buckets.mask;
        }
    }

    /// Double the shard's bucket array and republish it.
    ///
    /// Copies pointers, not records, so every borrow already handed out stays
    /// valid and readers still probing the old array find exactly what they
    /// would have found before. Caller must hold the shard's writer lock.
    fn grow(&self, shard: &Shard<T>, guard: &Guard) {
        let old = shard.buckets.load(Ordering::Acquire, guard);
        // SAFETY: the lock is held, so nothing else can have replaced it.
        let old_ref = unsafe { old.deref() };

        let grown = Buckets::with_capacity(old_ref.entries.len() * 2);
        for entry in &old_ref.entries {
            let hash = entry.hash.load(Ordering::Relaxed);
            if hash != 0 {
                Self::place(&grown, hash, entry.record.load(Ordering::Relaxed));
            }
        }

        shard.buckets.store(Owned::new(grown), Ordering::Release);
        // SAFETY: unreachable from `buckets` now, and `defer_destroy` waits for
        // every thread pinned at this moment to unpin — which covers readers
        // that loaded the old pointer just before the store. Dropping a
        // `Buckets` frees only the bucket array; the records it pointed at are
        // owned by `SlotMap::drop` and are still live in the new array.
        unsafe { guard.defer_destroy(old) };
    }

    /// Free the records whose slots hold no version, and rebuild around what is
    /// left. Returns how many were reclaimed.
    ///
    /// This is the one operation that breaks the append-only rule the whole
    /// module rests on — and it is confined to exclusive access precisely so
    /// that the rule holds everywhere else. Removing records concurrently would
    /// cost a probe-and-revalidate on every write, or a lock every transaction
    /// has to announce itself to; either gives back the measurement at the top
    /// of this file, which is the reason the map is shaped this way at all.
    ///
    /// # Safety contract
    ///
    /// **The caller must hold `&mut Database`.** That is what makes freeing
    /// records sound rather than merely careful: a `Transaction<'db>` borrows
    /// `&'db Database`, so an exclusive borrow of the database is a
    /// compile-time proof that no transaction exists and therefore that no
    /// `&Slot<T>` handed out by [`SlotMap::get`] is still outstanding. Nothing
    /// is deferred and nothing needs to be — there is no reader to race, which
    /// is why this can free records outright when normal operation cannot.
    ///
    /// It takes `&self` only because the map is reached through an `Arc` from
    /// the registry; `Database::compact` is the sole caller and is where the
    /// exclusivity is actually proven.
    ///
    /// An empty slot means the record was deleted and its tombstone has already
    /// been collected by [`Slot::prune`](crate::engine::store::Slot::prune) — so
    /// this reclaims keys that are gone, never ones merely idle.
    ///
    /// Both structures are rebuilt rather than patched. The bucket array is open
    /// addressing with linear probing, where clearing an entry severs the probe
    /// chains running through it; re-placing the survivors into a fresh array
    /// avoids that class of bug entirely. The chunk list is rebuilt for the same
    /// kind of reason — iteration stops at the first null, so it has to stay
    /// dense.
    pub(crate) fn compact(&self) -> usize {
        // SAFETY: the caller holds `&mut Database`, so there is no concurrent
        // reader and nothing needs deferring — the same shape of argument
        // `SlotMap::drop` makes from `&mut self`.
        let guard = unsafe { crossbeam_epoch::unprotected() };
        let mut reclaimed = 0;

        for shard in &self.shards {
            let mut count = shard.writers.lock();
            let mut survivors: Vec<*mut Record<T>> = Vec::with_capacity(*count);

            // Take the whole chunk list; it is rebuilt below from the survivors.
            let mut chunk = shard.chunks.swap(Shared::null(), Ordering::Relaxed, guard);
            while !chunk.is_null() {
                // SAFETY: each chunk is linked into exactly one list and is
                // reached once here, so this takes ownership exactly once.
                let owned = unsafe { chunk.into_owned() };
                for entry in &owned.records {
                    let record = entry.load(Ordering::Relaxed);
                    if record.is_null() {
                        break;
                    }
                    // SAFETY: appended by `get_or_create` and not yet freed.
                    let empty = unsafe { &*record }
                        .slot
                        .latest
                        .load(Ordering::Relaxed, guard)
                        .is_null();
                    if empty {
                        // SAFETY: reached exactly once, and unreachable by
                        // anyone else — see the contract above.
                        drop(unsafe { Box::from_raw(record) });
                        reclaimed += 1;
                    } else {
                        survivors.push(record);
                    }
                }
                chunk = owned.next.load(Ordering::Relaxed, guard);
                drop(owned);
            }

            // Rebuild the index over exactly the survivors, at a capacity that
            // keeps `get_or_create`'s "half full at most" probe invariant.
            let capacity = (survivors.len() * 2)
                .next_power_of_two()
                .max(INITIAL_CAPACITY);
            let buckets = Buckets::with_capacity(capacity);
            let fresh = Owned::new(Chunk::new()).into_shared(guard);
            shard.chunks.store(fresh, Ordering::Relaxed);

            for (index, &record) in survivors.iter().enumerate() {
                // SAFETY: a survivor, so still allocated and owned by this map.
                let hash = Self::hash(&unsafe { &*record }.key);
                Self::place(&buckets, hash, record);
                Self::append(shard, index, record, guard);
            }

            let old = shard
                .buckets
                .swap(Owned::new(buckets), Ordering::Relaxed, guard);
            // SAFETY: taken out of the map, and no one is probing it — see the
            // contract above. Dropping `Buckets` frees the array, not records.
            drop(unsafe { old.into_owned() });

            *count = survivors.len();
        }

        reclaimed
    }

    /// Visit the records of one shard, chosen by `round`.
    ///
    /// This is what lets a caller walk the whole map a slice at a time. Records
    /// are spread across [`SHARDS`] shards by hash, so sweeping one shard per
    /// round touches about `1/SHARDS` of the map and completes a full pass in
    /// that many rounds — no cursor to store, and no skipping past records that
    /// were already visited.
    pub(crate) fn for_each_in_shard<'a>(
        &'a self,
        round: usize,
        guard: &Guard,
        f: impl FnMut(&'a Record<T>),
    ) {
        Self::walk(&self.shards[round % SHARDS], guard, f);
    }

    /// Visit every record in the map, in unspecified order.
    ///
    /// Walks the chunk lists, not the bucket arrays — see [`CHUNK`] for why
    /// that distinction is worth a second data structure.
    ///
    /// Lock-free like `get`, and for the same reason. A record inserted while
    /// this runs may or may not be seen, which is the same guarantee a scan had
    /// when it collected keys shard by shard under a read lock.
    pub(crate) fn for_each<'a>(&'a self, guard: &Guard, mut f: impl FnMut(&'a Record<T>)) {
        for shard in &self.shards {
            Self::walk(shard, guard, &mut f);
        }
    }

    /// Walk one shard's chunk list, oldest record first.
    fn walk<'a>(shard: &'a Shard<T>, guard: &Guard, mut f: impl FnMut(&'a Record<T>)) {
        let mut chunk = shard.chunks.load(Ordering::Acquire, guard);
        'chunks: while !chunk.is_null() {
            // SAFETY: chunks are linked once and never freed while the map
            // lives, so any non-null pointer here stays valid.
            let current = unsafe { chunk.deref() };
            for entry in &current.records {
                let record = entry.load(Ordering::Acquire);
                if record.is_null() {
                    // No holes, and `next` is linked only once a chunk is
                    // full — so this is the end of the list, not a gap.
                    break 'chunks;
                }
                // SAFETY: published with `Release` by `append`, and records
                // outlive every epoch.
                f(unsafe { &*record });
            }
            chunk = current.next.load(Ordering::Acquire, guard);
        }
    }
}

impl<T: Versioned> Drop for SlotMap<T> {
    /// Free every record, and the live bucket arrays.
    ///
    /// Superseded arrays were handed to `defer_destroy` and are crossbeam's to
    /// free; they own no records, so whether its queue drains before exit
    /// changes nothing that could be observed as a leak of user data.
    fn drop(&mut self) {
        // SAFETY: `&mut self` proves there is no concurrent reader, so nothing
        // needs deferring. `unprotected` is the escape hatch for exactly that.
        let guard = unsafe { crossbeam_epoch::unprotected() };
        for shard in &mut self.shards {
            // The bucket array is only an index into the chunk list, so freeing
            // it reclaims no records.
            let buckets = shard.buckets.swap(Shared::null(), Ordering::Relaxed, guard);
            // SAFETY: taken out of the map, so ownership is ours exactly once.
            drop(unsafe { buckets.into_owned() });

            let mut chunk = shard.chunks.swap(Shared::null(), Ordering::Relaxed, guard);
            while !chunk.is_null() {
                // SAFETY: each chunk is linked into exactly one list and is
                // reached once, so this takes ownership exactly once.
                let owned = unsafe { chunk.into_owned() };
                for entry in &owned.records {
                    let record = entry.load(Ordering::Relaxed);
                    if record.is_null() {
                        break;
                    }
                    // SAFETY: every record is appended to exactly one position
                    // in one chunk, so this reclaims it exactly once.
                    drop(unsafe { Box::from_raw(record) });
                }
                chunk = owned.next.load(Ordering::Relaxed, guard);
                drop(owned);
            }
        }
    }
}
