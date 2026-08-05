//! Schema description produced by `#[derive(Mvcc)]`.
//!
//! The derive emits a `const` description of the table — its id, its primary
//! key, and its secondary indexes — so the engine can build storage without
//! reflection and without a runtime schema registry lookup on the hot path.

/// Dense identifier for a table, assigned at registration.
///
/// Dense and small so the engine can index tables with an array rather than a
/// hash map.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct TableId(pub u16);

/// An order-preserving encoding of an index key.
///
/// Secondary index keys are flattened to bytes whose `memcmp` order matches the
/// logical order of the original values. That lets every index share one
/// radix-tree implementation instead of being generic over key type, and lets
/// range scans work on composite keys.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct IndexKey(pub Vec<u8>);

impl IndexKey {
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// A secondary index over `T`, described at compile time.
pub struct IndexDesc<T> {
    /// Field name, or a user-supplied name for composite indexes.
    pub name: &'static str,
    /// Whether inserts must fail on a duplicate.
    pub unique: bool,
    /// Projects a record to its index key. A function pointer rather than a
    /// closure so the whole descriptor stays `const`-constructible.
    pub extract: fn(&T) -> IndexKey,
}

impl<T> Clone for IndexDesc<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for IndexDesc<T> {}

/// Types usable as index keys, encoded to `memcmp`-comparable bytes.
///
/// Implemented by the crate for primitives and `String`; the derive calls it to
/// build [`IndexKey`]s.
pub trait Encodable {
    fn encode_to(&self, out: &mut Vec<u8>);

    fn encode(&self) -> IndexKey {
        let mut out = Vec::new();
        self.encode_to(&mut out);
        IndexKey(out)
    }
}

macro_rules! impl_encodable_uint {
    ($($t:ty),*) => {$(
        impl Encodable for $t {
            #[inline]
            fn encode_to(&self, out: &mut Vec<u8>) {
                // Big-endian: byte order matches numeric order.
                out.extend_from_slice(&self.to_be_bytes());
            }
        }
    )*};
}
impl_encodable_uint!(u8, u16, u32, u64, u128);

macro_rules! impl_encodable_int {
    ($($t:ty => $u:ty),*) => {$(
        impl Encodable for $t {
            #[inline]
            fn encode_to(&self, out: &mut Vec<u8>) {
                // Flip the sign bit so negatives sort below positives.
                let biased = (*self as $u) ^ (1 << (<$u>::BITS - 1));
                out.extend_from_slice(&biased.to_be_bytes());
            }
        }
    )*};
}
impl_encodable_int!(i8 => u8, i16 => u16, i32 => u32, i64 => u64, i128 => u128);

impl Encodable for String {
    fn encode_to(&self, out: &mut Vec<u8>) {
        self.as_str().encode_to(out);
    }
}

impl Encodable for str {
    fn encode_to(&self, out: &mut Vec<u8>) {
        // NUL-terminated with escaping, so that "ab" sorts before "abc" and no
        // value can forge a separator in a composite key.
        for &b in self.as_bytes() {
            if b == 0x00 {
                out.extend_from_slice(&[0x00, 0xff]);
            } else {
                out.push(b);
            }
        }
        out.extend_from_slice(&[0x00, 0x00]);
    }
}

impl<T: Encodable> Encodable for &T {
    fn encode_to(&self, out: &mut Vec<u8>) {
        (**self).encode_to(out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key<T: Encodable>(v: T) -> Vec<u8> {
        v.encode().0
    }

    #[test]
    fn unsigned_keys_sort_numerically() {
        assert!(key(1u64) < key(2u64));
        assert!(key(255u64) < key(256u64));
    }

    #[test]
    fn signed_keys_sort_with_negatives_first() {
        assert!(key(-1i64) < key(0i64));
        assert!(key(i64::MIN) < key(i64::MAX));
        assert!(key(-5i64) < key(-4i64));
    }

    #[test]
    fn string_prefixes_sort_before_extensions() {
        assert!(key("ab".to_string()) < key("abc".to_string()));
        assert!(key("a".to_string()) < key("b".to_string()));
    }

    #[test]
    fn embedded_nul_cannot_forge_a_separator() {
        // "a\0b" must not compare equal to the composite ("a", "b").
        assert_ne!(key("a\0b".to_string()), {
            let mut v = key("a".to_string());
            v.extend_from_slice(&key("b".to_string()));
            v
        });
    }
}
