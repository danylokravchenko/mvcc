//! A fast, non-cryptographic hasher for the engine's internal maps.
//!
//! `std`'s default is SipHash-1-3, chosen so that a `HashMap` exposed to
//! untrusted keys cannot be pushed into its worst case. That costs on the order
//! of ten nanoseconds for a small key, and the slot lookup pays it *twice* —
//! once to pick a shard, once inside the shard's map. Against a point read of
//! roughly thirty nanoseconds, that was most of the budget.
//!
//! Primary keys here come from the application's own records, not from the
//! network, so the collision-resistance `SipHash` buys is not doing any work.
//! This is the same trade `rustc` makes for its own interning tables, and this
//! is the same construction (`FxHash`): multiply-xor-rotate, a few instructions
//! per word.
//!
//! It is deliberately *not* exposed in the public API. If a caller ever needs
//! adversarial-input resistance for its keys, the answer is to make the map's
//! hasher configurable, not to hand this one out.

use std::hash::{BuildHasherDefault, Hasher};

/// The odd multiplier from `FxHash` — the fractional bits of the golden ratio.
const SEED: u64 = 0x517c_c1b7_2722_0a95;

#[derive(Default, Clone, Copy)]
pub(crate) struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        // Rotate so that high bits of the previous word reach the low bits,
        // then mix. The multiply is what actually spreads the entropy.
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            self.add(u64::from_le_bytes(
                chunk.try_into().expect("chunk is 8 bytes"),
            ));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(buf));
        }
    }

    // Without these, `Hasher`'s defaults route every integer through `write`,
    // which means a slice construction and a loop for what should be one mix.
    #[inline]
    fn write_u8(&mut self, n: u8) {
        self.add(n as u64)
    }
    #[inline]
    fn write_u16(&mut self, n: u16) {
        self.add(n as u64)
    }
    #[inline]
    fn write_u32(&mut self, n: u32) {
        self.add(n as u64)
    }
    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.add(n)
    }
    #[inline]
    fn write_u128(&mut self, n: u128) {
        self.add(n as u64);
        self.add((n >> 64) as u64);
    }
    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.add(n as u64)
    }
    #[inline]
    fn write_i8(&mut self, n: i8) {
        self.add(n as u64)
    }
    #[inline]
    fn write_i16(&mut self, n: i16) {
        self.add(n as u64)
    }
    #[inline]
    fn write_i32(&mut self, n: i32) {
        self.add(n as u64)
    }
    #[inline]
    fn write_i64(&mut self, n: i64) {
        self.add(n as u64)
    }
    #[inline]
    fn write_isize(&mut self, n: isize) {
        self.add(n as u64)
    }
}

pub(crate) type FxBuildHasher = BuildHasherDefault<FxHasher>;

/// Hash one key, for shard selection.
#[inline]
pub(crate) fn hash_one<K: std::hash::Hash + ?Sized>(key: &K) -> u64 {
    let mut hasher = FxHasher::default();
    key.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn distinct_keys_mostly_land_in_distinct_shards() {
        // Not a quality claim about the hash, just a check that shard selection
        // is not degenerate — a hasher that ignored the high bits, or returned
        // the key unchanged, would pile sequential keys into few shards.
        const SHARDS: usize = 64;
        let mut counts = [0usize; SHARDS];
        for k in 0u64..64_000 {
            counts[(hash_one(&k) as usize) % SHARDS] += 1;
        }
        let (min, max) = (
            *counts.iter().min().expect("non-empty"),
            *counts.iter().max().expect("non-empty"),
        );
        assert!(
            max < min * 2,
            "shard occupancy is lopsided: {min}..{max} across {SHARDS} shards"
        );
    }

    #[test]
    fn sequential_keys_do_not_collide() {
        let hashes: HashSet<u64> = (0u64..10_000).map(|k| hash_one(&k)).collect();
        assert_eq!(hashes.len(), 10_000, "sequential keys collided");
    }

    #[test]
    fn strings_and_integers_both_mix() {
        assert_ne!(hash_one("alpha"), hash_one("beta"));
        assert_ne!(hash_one(&1u64), hash_one(&2u64));
        // A single-bit difference must not survive into a single-bit difference.
        assert!((hash_one(&1u64) ^ hash_one(&3u64)).count_ones() > 8);
    }
}
