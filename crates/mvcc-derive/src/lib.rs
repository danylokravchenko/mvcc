//! `#[derive(Mvcc)]`.
//!
//! # What the user writes
//!
//! ```ignore
//! #[derive(Mvcc, Clone, Debug)]
//! #[mvcc(table = "accounts")]
//! pub struct Account {
//!     #[mvcc(primary_key)]
//!     pub id: u64,
//!     #[mvcc(index)]
//!     pub owner: String,
//!     pub balance: i64,
//! }
//! ```
//!
//! # What it expands to
//!
//! - `impl Versioned for Account` — key type and extraction, `memcmp` key
//!   encoding, and the index descriptor table.
//! - a private `static` holding the index descriptors, const-constructed with
//!   `extract` as a plain `fn` pointer.
//! - a private `static OnceLock<TableId>`, filled by `Database::register`.
//! - `Account::OWNER: Index<Account, String>` — one associated const per
//!   indexed field, named after the field in upper case, which is how scans
//!   name an index. Carrying the field's type is the point: it is what makes
//!   `tx.scan_index(Account::OWNER, 1u64..=2)` a type error rather than a scan
//!   that matches nothing.
//!
//! Everything is emitted inside a `const _: () = { … };` block so the generated
//! statics cannot collide with user items or with a second derive in the same
//! module.
//!
//! # What it deliberately does not do
//!
//! It does not make the struct itself transactional. `Account` gains no interior
//! mutability, no `Drop`, and no hidden fields — it stays a plain Rust struct
//! you can construct, match on, and pass around. All transactional behaviour
//! lives on `Transaction`, which is where errors can actually be returned.
//!
//! # Attribute reference
//!
//! | attribute | position | meaning |
//! |---|---|---|
//! | `table = "name"` | struct | table name, used only in error messages; defaults to the type name |
//! | `primary_key` | field | required, exactly one |
//! | `index` | field | secondary index on this field |
//! | `index(unique)` | field | unique secondary index |
//!
//! An index is always named after its field. There is no rename knob: the name
//! is not a string anyone types, it is the associated const above, so renaming
//! it would only decouple the const from the field it reads.
//!
//! There is no `skip`: nothing is serialised, so every field is simply carried
//! along in the struct. Fields need no traits beyond what `Versioned` requires
//! of the struct as a whole.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    Data, DeriveInput, Field, Fields, Ident, LitStr, Type, parse_macro_input, spanned::Spanned,
};

/// Derive `Versioned`. See the module docs.
#[proc_macro_derive(Mvcc, attributes(mvcc))]
pub fn derive_mvcc(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand(&input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Per-field configuration parsed from `#[mvcc(...)]`.
#[derive(Default)]
struct FieldOpts {
    primary_key: bool,
    index: bool,
    unique: bool,
}

fn parse_field_opts(field: &Field) -> syn::Result<FieldOpts> {
    let mut opts = FieldOpts::default();

    for attr in field.attrs.iter().filter(|a| a.path().is_ident("mvcc")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("primary_key") {
                opts.primary_key = true;
            } else if meta.path.is_ident("index") {
                opts.index = true;
                // `index` may be bare or carry `(unique)`.
                if meta.input.peek(syn::token::Paren) {
                    meta.parse_nested_meta(|inner| {
                        if inner.path.is_ident("unique") {
                            opts.unique = true;
                            Ok(())
                        } else {
                            Err(inner.error("expected `unique`"))
                        }
                    })?;
                }
            } else {
                return Err(meta.error("expected `primary_key`, `index`, or `index(unique)`"));
            }
            Ok(())
        })?;
    }

    Ok(opts)
}

fn expand(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let ty = &input.ident;

    // --- struct-level attributes -------------------------------------------
    let mut table_name = ty.to_string();
    for attr in input.attrs.iter().filter(|a| a.path().is_ident("mvcc")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("table") {
                table_name = meta.value()?.parse::<LitStr>()?.value();
                Ok(())
            } else {
                Err(meta.error("expected `table = \"…\"`"))
            }
        })?;
    }

    // --- fields -------------------------------------------------------------
    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(named) => &named.named,
            _ => {
                return Err(syn::Error::new(
                    input.span(),
                    "`Mvcc` requires a struct with named fields: index declarations \
                     refer to fields by name",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new(
                input.span(),
                "`Mvcc` can only be derived for structs",
            ));
        }
    };

    let mut primary: Option<(Ident, Type)> = None;
    let mut indexes: Vec<(Ident, Type, bool)> = Vec::new();

    for field in fields {
        let name = field.ident.clone().expect("named fields checked above");
        let opts = parse_field_opts(field)?;

        if opts.primary_key {
            if primary.is_some() {
                return Err(syn::Error::new(
                    field.span(),
                    "a second `#[mvcc(primary_key)]`: exactly one is allowed",
                ));
            }
            primary = Some((name.clone(), field.ty.clone()));
        }

        if opts.index {
            indexes.push((name, field.ty.clone(), opts.unique));
        }
    }

    let (key_field, key_type) = primary.ok_or_else(|| {
        syn::Error::new(
            input.span(),
            format!(
                "`{ty}` has no primary key: mark exactly one field with `#[mvcc(primary_key)]`"
            ),
        )
    })?;

    // --- generated pieces ---------------------------------------------------
    let index_count = indexes.len();
    let index_entries = indexes.iter().map(|(field, _, unique)| {
        let index_name = field.to_string();
        quote! {
            _mvcc::IndexDesc {
                name: #index_name,
                unique: #unique,
                extract: |record: &#ty| _mvcc::Encodable::encode(&record.#field),
            }
        }
    });

    // One associated const per index, named after the field in upper case, so
    // `tx.scan_index(Item::OWNER, …)` is resolved and type-checked by the
    // compiler rather than matched against a string at runtime. `pub` so the
    // const is as visible as the struct; the `allow`s are because generated
    // code cannot know whether the struct is public or whether every index is
    // actually scanned.
    let index_consts = indexes
        .iter()
        .enumerate()
        .map(|(position, (field, key_ty, _))| {
            let const_name =
                format_ident!("{}", field.to_string().to_uppercase(), span = field.span());
            let index_name = field.to_string();
            let doc = format!("The `{index_name}` secondary index. See [`Index`](::mvcc::Index).");
            quote! {
                #[doc = #doc]
                #[allow(dead_code, unreachable_pub)]
                pub const #const_name: _mvcc::Index<#ty, #key_ty> =
                    _mvcc::Index::new(#position, #index_name);
            }
        });

    let table_id_cell = format_ident!("__MVCC_TABLE_ID");
    let index_table = format_ident!("__MVCC_INDEXES");

    Ok(quote! {
        const _: () = {
            use ::mvcc::__private as _mvcc;

            static #table_id_cell: ::std::sync::OnceLock<_mvcc::TableId> =
                ::std::sync::OnceLock::new();

            static #index_table: [_mvcc::IndexDesc<#ty>; #index_count] = [ #(#index_entries),* ];

            // An inherent impl inside a `const _` block still attaches to the
            // type globally, so these consts are reachable as `#ty::NAME` from
            // anywhere the type is, without leaking anything else.
            impl #ty {
                #(#index_consts)*
            }

            impl _mvcc::Versioned for #ty {
                type Key = #key_type;

                const TABLE_NAME: &'static str = #table_name;

                #[inline]
                fn table_id_cell() -> &'static ::std::sync::OnceLock<_mvcc::TableId> {
                    &#table_id_cell
                }

                #[inline]
                fn key(&self) -> Self::Key {
                    ::core::clone::Clone::clone(&self.#key_field)
                }

                #[inline]
                fn key_bytes(&self) -> _mvcc::IndexKey {
                    _mvcc::Encodable::encode(&self.#key_field)
                }

                #[inline]
                fn indexes() -> &'static [_mvcc::IndexDesc<Self>] {
                    &#index_table
                }
            }
        };
    })
}
