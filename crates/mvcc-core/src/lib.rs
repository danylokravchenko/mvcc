//! Core vocabulary for the MVCC engine.
//!
//! This crate holds the types that cross every layer boundary — timestamps, the
//! visibility rules, the isolation-level typestates, and the [`Versioned`] trait
//! that `#[derive(Mvcc)]` implements. It has no dependencies, so the macro crate
//! and the engine can agree on these definitions without a dependency cycle.

#![forbid(unsafe_code)]

pub mod error;
pub mod isolation;
pub mod schema;
pub mod time;
pub mod versioned;

pub use error::{Error, Result};
pub use isolation::{IsolationLevel, ReadCommitted, RepeatableRead, Serializable, Snapshot};
pub use schema::{Encodable, IndexDesc, IndexKey, TableId};
pub use time::{Timestamp, TxnId, Visibility};
pub use versioned::Versioned;
