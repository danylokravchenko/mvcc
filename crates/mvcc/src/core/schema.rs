//! Schema description produced by `#[derive(Mvcc)]`.
//!
//! The derive emits a `const` description of the table — its id, its primary
//! key, and its secondary indexes — so the engine can build storage without
//! reflection and without a runtime schema registry lookup on the hot path.

use core::{fmt, marker::PhantomData};

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

/// A handle to one of `T`'s secondary indexes, naming the field it covers and
/// the type of that field.
///
/// `#[derive(Mvcc)]` emits one of these as an associated const per indexed
/// field, named after the field in upper case — `#[mvcc(index)] owner: u64`
/// becomes `Item::OWNER: Index<Item, u64>`. Scans take the const rather than a
/// string, so both halves of the old `scan_index("owner", …)` are checked: a
/// wrong name is an unresolved item, and a range whose type does not match the
/// indexed field fails to unify with `K` instead of silently matching nothing.
///
/// It is `Copy` and holds only a position, so passing one costs nothing.
pub struct Index<T, K> {
    /// Position in [`Versioned::indexes`](crate::Versioned::indexes).
    pub position: usize,
    /// The indexed field's name, for diagnostics.
    pub name: &'static str,
    /// Ties the handle to its record and key type without owning either.
    /// `fn(&T) -> K` rather than `(T, K)` so the handle stays `Send + Sync`
    /// whatever the record is — it describes the extraction, not a value.
    _record: PhantomData<fn(&T) -> K>,
}

impl<T, K> Index<T, K> {
    /// Called by the derive. Constructing one by hand with a position that is
    /// not this type's index is a logic error, not a memory-safety one: the
    /// position is bounds-checked against `T::indexes()` on use.
    #[doc(hidden)]
    pub const fn new(position: usize, name: &'static str) -> Self {
        Index {
            position,
            name,
            _record: PhantomData,
        }
    }
}

impl<T, K> Clone for Index<T, K> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T, K> Copy for Index<T, K> {}

impl<T, K> fmt::Debug for Index<T, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Index({})", self.name)
    }
}

/// A secondary index over `T`, described at compile time.
pub struct IndexDesc<T> {
    /// The indexed field's name. Used in `DuplicateKey` errors and by
    /// [`Index`], which is how user code refers to it.
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
