//! The storage engine.
//!
//! Version chains, transactions, the timestamp oracle, garbage collection. This
//! is where the unsafe code lives; see [`crate::core`] for the half that has
//! none.
//!
//! The module is private. Everything users touch is re-exported from the crate
//! root, so the paths here can be rearranged without it being a breaking
//! change.

pub(crate) mod gc;
pub(crate) mod hash;
pub(crate) mod oracle;
pub(crate) mod ssi;
pub(crate) mod store;
pub(crate) mod txn;

pub use store::Database;
pub use txn::Transaction;
