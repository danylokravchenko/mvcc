//! Core vocabulary.
//!
//! The types that cross every layer boundary — timestamps, the visibility
//! rules, the isolation-level typestates, and the [`Versioned`] trait that
//! `#[derive(Mvcc)]` implements.
//!
//! Nothing here knows how versions are stored, and that separation is what lets
//! the module ban unsafe code outright. It is worth preserving: the definitions
//! the derive macro and the engine have to agree on are exactly the ones with
//! no pointers in them.

#![forbid(unsafe_code)]

pub(crate) mod error;
pub(crate) mod isolation;
pub(crate) mod schema;
pub(crate) mod time;
pub(crate) mod versioned;

pub use error::{Error, Result};
pub use isolation::{IsolationLevel, ReadCommitted, RepeatableRead, Serializable, Snapshot};
pub use schema::{Encodable, Index, IndexDesc, IndexKey, TableId};
pub use time::{Timestamp, TxnId};
// Internal: the decoded form of a version stamp, never named in a public
// signature.
pub(crate) use time::Visibility;
pub use versioned::Versioned;
