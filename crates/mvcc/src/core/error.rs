//! Errors surfaced to user code.

use core::fmt;

/// `Result` with this crate's [`Error`] as the default error type.
///
/// Every fallible engine operation returns this, so `-> Result<()>` in user
/// code that has `use mvcc::Result` means the same thing it does here.
pub type Result<T, E = Error> = core::result::Result<T, E>;

/// Everything an engine operation can fail with.
///
/// The split that matters is [`Error::is_retriable`]: [`WriteConflict`] and
/// [`SerializationFailure`] are the engine reporting that two transactions
/// could not both happen, and re-running is the correct response. The rest are
/// programming mistakes that will fail again identically.
///
/// [`Database::transaction`] already loops on the retriable ones, so most code
/// never matches on this at all.
///
/// ```
/// use mvcc::{Config, Database, Error, Mvcc};
///
/// #[derive(Mvcc, Clone)]
/// struct Account {
///     #[mvcc(primary_key)] id: u64,
///     balance: i64,
/// }
///
/// let db = Database::open(Config::in_memory())?;
/// db.register::<Account>()?;
/// db.transaction(|tx| tx.insert(Account { id: 1, balance: 0 }))?;
///
/// // Inserting the same primary key twice is a programming error, not a race:
/// // it is not retriable, and `transaction` returns it rather than looping.
/// let err = db
///     .transaction(|tx| tx.insert(Account { id: 1, balance: 0 }))
///     .unwrap_err();
///
/// assert!(matches!(err, Error::DuplicateKey { .. }));
/// assert!(!err.is_retriable());
/// # Ok::<(), mvcc::Error>(())
/// ```
///
/// `#[non_exhaustive]`: matching must include a `_` arm, so that a new failure
/// mode is not a breaking change.
///
/// [`WriteConflict`]: Error::WriteConflict
/// [`SerializationFailure`]: Error::SerializationFailure
/// [`Database::transaction`]: crate::Database::transaction
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Another transaction committed a conflicting write to the same key after
    /// our snapshot. Retriable: re-run the transaction.
    WriteConflict {
        /// The table whose record was contended.
        table: &'static str,
    },

    /// Serializable validation found that something this transaction read has
    /// since changed. Retriable, and expected under contention — this does not
    /// indicate a bug.
    SerializationFailure,

    /// A unique index, or the primary key, already holds this value.
    DuplicateKey {
        /// The table the value would have been written to.
        table: &'static str,
        /// The index that already holds it, or the primary key.
        index: &'static str,
    },

    /// The transaction was aborted, explicitly or by being dropped, and can no
    /// longer be used.
    Aborted,

    /// The type was never passed to `Database::register`.
    TableNotRegistered {
        /// The unregistered type's table name.
        table: &'static str,
    },

    /// An update tried to change the primary key, which would move the record
    /// to a different slot and leave the old one holding a value that no longer
    /// matches its key. Delete and re-insert instead.
    PrimaryKeyChanged {
        /// The table whose primary key the update tried to change.
        table: &'static str,
    },
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
        }
    }
}

impl std::error::Error for Error {}
