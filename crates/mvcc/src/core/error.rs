//! Errors surfaced to user code.

use core::fmt;

pub type Result<T, E = Error> = core::result::Result<T, E>;

#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Another transaction committed a conflicting write to the same key after
    /// our snapshot. Retriable: re-run the transaction.
    WriteConflict { table: &'static str },

    /// Serializable validation found that something this transaction read has
    /// since changed. Retriable, and expected under contention — this does not
    /// indicate a bug.
    SerializationFailure,

    /// A unique index, or the primary key, already holds this value.
    DuplicateKey {
        table: &'static str,
        index: &'static str,
    },

    /// The transaction was aborted, explicitly or by being dropped, and can no
    /// longer be used.
    Aborted,

    /// The type was never passed to `Database::register`.
    TableNotRegistered { table: &'static str },

    /// An update tried to change the primary key, which would move the record
    /// to a different slot and leave the old one holding a value that no longer
    /// matches its key. Delete and re-insert instead.
    PrimaryKeyChanged { table: &'static str },

    /// No index of that name is declared on the table.
    NoSuchIndex { table: &'static str, index: String },
}

impl Error {
    /// Whether re-running the transaction from the top is a reasonable
    /// response. `Database::transaction` loops on exactly this.
    pub fn is_retriable(&self) -> bool {
        matches!(
            self,
            Error::WriteConflict { .. } | Error::SerializationFailure
        )
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::WriteConflict { table } => {
                write!(f, "write-write conflict on table `{table}`")
            }
            Error::SerializationFailure => {
                f.write_str("could not serialize access due to read/write dependencies")
            }
            Error::DuplicateKey { table, index } => {
                write!(f, "duplicate key on `{table}`.`{index}`")
            }
            Error::Aborted => f.write_str("transaction is aborted"),
            Error::TableNotRegistered { table } => {
                write!(
                    f,
                    "`{table}` was not registered: call `db.register::<{table}>()` at startup"
                )
            }
            Error::PrimaryKeyChanged { table } => write!(
                f,
                "an update may not change the primary key of `{table}`; delete and insert instead"
            ),
            Error::NoSuchIndex { table, index } => {
                write!(f, "no index `{index}` on table `{table}`")
            }
        }
    }
}

impl std::error::Error for Error {}
