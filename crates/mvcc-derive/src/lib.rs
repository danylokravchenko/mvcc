//! The proc-macro behind `#[derive(Mvcc)]`, which turns a plain struct into a
//! table the `mvcc` engine can store.
//!
//! **Do not depend on this crate directly.** Depend on `mvcc-core`, which re-exports
//! the derive and is the only thing the generated code refers to. The
//! user-facing documentation — what the attributes mean, what you get — lives on
//! that re-export, so that it renders at `mvcc::Mvcc` where people arrive, and
//! so that its examples are real doctests. They could not live here: a compiling
//! example needs the `Versioned` trait this macro implements, and depending on
//! `mvcc-core` to get it, even as a dev-dependency, is a cycle. This page is for
//! reading or changing the expansion.
//!
//! # What it generates
//!
//! Given a struct with a primary key and one indexed field:
//!
//! ```text
//! #[derive(Mvcc, Clone)]
//! struct Account {
//!     #[mvcc(primary_key)] id: u64,
//!     #[mvcc(index)]       owner: String,
//!     balance: i64,
//! }
//! ```
//!
//! the expansion is, in outline:
//!
//! ```text
//! const _: () = {
//!     use ::mvcc::__private as _mvcc;
//!
//!     // Filled in by `Database::register`; ids are assigned in registration
//!     // order, so this cannot be a `const`.
//!     static __MVCC_TABLE_ID: OnceLock<TableId> = OnceLock::new();
//!
//!     // Index descriptors, const-constructed. `extract` is a plain `fn`
//!     // pointer, so the table needs no allocation and no lazy init.
//!     static __MVCC_INDEXES: [IndexDesc<Account>; 1] = [ … ];
//!
//!     impl Account {
//!         pub const OWNER: Index<Account, String> = Index::new(0, "owner");
//!     }
//!
//!     impl Versioned for Account { type Key = u64; … }
//! };
//! ```
//!

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    Data, DeriveInput, Field, Fields, Ident, LitStr, Type, parse_macro_input, spanned::Spanned,
};

/// Derive `Versioned`. Documented at `mvcc::Mvcc`, which is how users reach it.
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
