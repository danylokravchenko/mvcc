//! The storage engine.

pub mod gc;
mod hash;
pub mod index;
pub mod oracle;
pub mod ssi;
pub mod store;
pub mod txn;

pub use oracle::Oracle;
pub use store::{Database, Table};
pub use txn::Transaction;
