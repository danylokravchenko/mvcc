//! Indexes: key → `Slot`.
//!
//! Indexes map keys to *slots*, never to versions. Version visibility is
//! resolved after the lookup, by walking the slot's chain. Keeping the index
//! version-oblivious means it never has to be MVCC-aware, and an update that
//! does not change an indexed column does not touch the index at all.
//!
//! # Structure
//!
//! An **adaptive radix tree** (ART) with optimistic lock coupling:
//!
//! - Radix rather than comparison-based: lookups are O(key length) with no key
//!   comparisons, and node fan-out adapts (4/16/48/256 children) so it stays
//!   compact for sparse key spaces. This is the standard choice for in-memory
//!   OLTP and the reason `IndexKey` is `memcmp`-ordered bytes.
//! - Optimistic lock coupling for concurrency: readers take no locks. Each node
//!   carries a version counter; a reader reads the counter, reads the node,
//!   re-reads the counter, and retries if it changed. Writers lock only the
//!   nodes they modify.
//!
//! The point of OLC is that lookups perform **no writes to shared memory**.
//! A read-write lock, even an uncontended one, dirties the lock's cache line on
//! every acquisition, so N cores reading the same hot root node serialise on
//! coherence traffic. This is the same argument as the timestamp oracle
//! (`crate::oracle`) and the same argument as EBR over refcounting
//! (`crate::gc`): on a read-mostly in-memory engine, the scalability limit is
//! *shared writes*, not lock contention as such.
//!
//! A hash index is also provided for primary keys that are never range-scanned,
//! since that case is common and a hash table is faster still.

use mvcc_core::IndexKey;

pub trait Index<V>: Send + Sync {
    fn get(&self, key: &IndexKey) -> Option<*const V>;
    fn insert(&self, key: &IndexKey, value: *const V) -> Option<*const V>;
    fn remove(&self, key: &IndexKey) -> Option<*const V>;
    fn range(&self, lo: &IndexKey, hi: &IndexKey) -> Box<dyn Iterator<Item = *const V> + '_>;
}
