//! The trait `#[derive(Mvcc)]` implements.

use std::sync::OnceLock;

use crate::schema::{IndexDesc, IndexKey, TableId};

/// A struct that the engine can store and version.
///
/// The bounds are the whole requirement: `Clone` because an update copies the
/// record before mutating it into a new version, `Send + Sync + 'static` because
/// versions outlive the transaction that wrote them. Fields are otherwise
/// unconstrained — there is no serialisation trait to satisfy.
///
/// Users never implement this by hand — `#[derive(Mvcc)]` generates it from
/// field attributes. It is public because it appears in the bounds of every
/// engine method, and because a hand implementation is a legitimate escape
/// hatch for exotic layouts.
pub trait Versioned: Sized + Clone + Send + Sync + 'static {
    /// The primary key type, from the `#[mvcc(primary_key)]` field.
    type Key: Ord + Clone + core::hash::Hash + Send + Sync + 'static;

    /// Table name, used in diagnostics and error messages.
    const TABLE_NAME: &'static str;

    /// Storage for this type's table id.
    ///
    /// The derive emits a private `static OnceLock<TableId>` per type and
    /// returns it here; `Database::register` fills it in. A `OnceLock` rather
    /// than a `const` because ids are assigned in registration order, which is
    /// not known until runtime.
    fn table_id_cell() -> &'static OnceLock<TableId>;

    /// This type's table id.
    ///
    /// # Panics
    ///
    /// If the type was never registered. This is a programming error rather
    /// than a runtime condition — every engine method that could reach it
    /// already requires a `Database` the caller had to register with.
    fn table_id() -> TableId {
        *Self::table_id_cell().get().unwrap_or_else(|| {
            panic!(
                "`{}` was not registered: call `db.register::<{}>()` before using it",
                Self::TABLE_NAME,
                Self::TABLE_NAME,
            )
        })
    }

    /// Extract the primary key.
    fn key(&self) -> Self::Key;

    /// The primary key in `memcmp` order, for the primary index.
    fn key_bytes(&self) -> IndexKey;

    /// Secondary indexes declared with `#[mvcc(index)]`.
    fn indexes() -> &'static [IndexDesc<Self>];
}
