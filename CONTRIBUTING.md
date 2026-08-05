# Contributing

Thanks for looking. This is a small, opinionated crate: an MVCC engine with no storage layer. Most of what follows is about **what belongs here** and **what a change has to prove** before it lands, because on a lock-free concurrency engine those are the two things that are expensive to get wrong.

---

## Contents

- [Before you write code](#before-you-write-code) — scope, and what will be declined
- [Setting up](#setting-up)
- [The checks](#the-checks)
- [The unsafe boundary](#the-unsafe-boundary)
- [Testing a change](#testing-a-change)
- [Performance claims](#performance-claims)
- [Style](#style)
- [Sending the change](#sending-the-change)

---

## Before you write code

Read [`README.md`](README.md) for what the crate does.

**Open an issue first for anything beyond a bug fix or a doc fix.** Not ceremony — this codebase has a fairly specific shape, and the fastest way to waste a weekend on it is to implement something that was deliberately cut.

Things that will be declined, with the reasoning already written down:

| proposal | why not |
| --- | --- |
| a WAL, checkpoints, `data_dir`, recovery | durability was cut on purpose; it roughly doubles the surface area and none of it is needed to get the MVCC model right |
| a buffer pool, larger-than-memory | records live as your actual struct; a disk-primary design makes the derive an ORM |
| networking, multi-process, replication | one process, one machine |
| a query planner | there is no query language to plan |
| `deregister` for tables | the registry being append-only is load-bearing for an `unsafe` borrow extension in `Database::table_erased` |
| `serde` bounds on record fields | nothing is serialised, and that is exactly why fields need no traits at all |

Things that are wanted, roughly in the order they would help:

- **Bugs in isolation behaviour.** A case where a level permits an anomaly it should forbid, or forbids one it should permit, is the most valuable report possible. Ideally as a test in `tests/hermitage.rs` or `tests/isolation_anomalies.rs` that fails.
- **ART with optimistic lock coupling**, replacing the sharded `HashMap` — the last lock a read takes, and the thing that makes ordered scans cheap
- **An epoch-based oracle** for the commit timestamp counter. That single shared cache line is the current write-throughput floor. The obvious batched variant cannot work.
- **YCSB A–C and TPC-C new-order benchmarks.** Nothing in "make it fast" list should start before there is a benchmark to prove it did something.
- **Docs and examples.** Examples here are documentation that executes and asserts — see below.

## Setting up

Rust edition 2024. No system libraries, no code generation step, no database to install.

```sh
git clone https://github.com/danylokravchenko/mvcc
cd mvcc
cargo test --workspace
```

The workspace is two crates, and the split is not negotiable — a proc-macro crate can export nothing but macros:

```text
crates/mvcc/          the library
├── src/core/         timestamps, visibility, isolation typestates, Versioned
│                     — #![forbid(unsafe_code)]
├── src/engine/       version chains, transactions, oracle, indexes, GC
│                     — where the unsafe lives
├── tests/            isolation suites, stress, public API surface
├── benches/          throughput.rs
└── examples/         basic, concurrent, isolation, game

crates/mvcc-derive/   #[derive(Mvcc)]
```

`core` and `engine` are private modules. Everything users touch is re-exported from the crate root, so internal paths can be rearranged freely — but moving an item *out* of the root is a breaking change, and `tests/public_api.rs` exists to make that fail to compile rather than fail to notice.

## The checks

Run checks yourself; a PR that fails them is not ready.

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features
cargo test --workspace
cargo bench                    # only if you touched the engine
```

**Clippy must be at zero warnings.** The lint set in the root `Cargo.toml` is deliberately not `clippy::pedantic` wholesale — on an epoch-based lock-free engine that group is ~150 warnings on code that is that way on purpose, and a lint set nobody can keep at zero stops being a signal. What is enabled is every lint `clippy.toml` sets a threshold for, plus the structural ones the codebase already satisfies.

If a threshold in `clippy.toml` blocks you, the answer is almost always to split the function rather than raise the number. Each threshold has a comment saying what it is protecting; raise one only by editing that comment to explain why the old reasoning stopped holding.

`cargo test --workspace` includes `tests/examples_run.rs`, which runs every shipped example in release mode — so a full test run compiles the examples twice. That is intentional: `cargo test` builds examples but never runs them, so an example can compile happily while asserting something no longer true.

## The unsafe boundary

`src/core/` is `#![forbid(unsafe_code)]` at the module level and stays that way. It holds the definitions the derive macro and the engine have to agree on, and those are exactly the ones with no pointers in them. If a change needs `unsafe` in `core`, the change is in the wrong module.

All `unsafe` lives in `src/engine/`, and every block needs a `// SAFETY:` comment that names the invariant keeping it sound — not a restatement of what the code does. The two that carry the most weight:

- **Epoch pinning.** A transaction pins one epoch for its whole life, which is what lets versions be borrowed rather than reference-counted. A borrow thatoutlives its guard is a use-after-free, and it will not reproduce reliably.
- **The append-only registry.** `Database::table_erased` extends a borrow from a read guard to `&self`, sound only because `register` never removes or replaces an entry. Anything that breaks that has to put the `Arc` back.

Changes to reclamation, the slot CAS protocol, or the watermark handshake need `tests/concurrent_stress.rs` to pass **and** a run under a sanitizer. Those tests cannot prove absence of a use-after-free; they make the window very likely to be hit.

## Testing a change

Match the test to what you touched. This is the part that gets a PR merged quickly.

| you changed | the test that has to move |
| --- | --- |
| the visibility predicate or a snapshot rule | `tests/isolation_anomalies.rs`, then `tests/hermitage.rs` |
| SSI, conflict detection, the commit path | `tests/hermitage.rs` — G2-item and G2 in particular |
| version chains, GC, epoch reclamation | `tests/concurrent_stress.rs`, plus a sanitizer run |
| anything re-exported from the crate root | `tests/public_api.rs` |
| the derive | a compile-level test with the attribute in question |
| behaviour a user would read about | the example that demonstrates it |

Two conventions worth knowing:

**Isolation tests assert in both directions.** A level that forbids too much is as wrong as one that forbids too little, so each anomaly is asserted *present or absent per level*, not just absent at `Serializable`. `hermitage.rs` quotes the original SQL transcript from [Hermitage](https://github.com/ept/hermitage) alongside each port so the translation can be checked against the source. If you add a case, quote its source too.

**Examples end by asserting the world they built is still consistent.**
`examples/game.rs` checks that gold is conserved, that no hero exceeds their carry limit, and that the party never overfills. An example that only prints is not pulling its weight — `examples_run.rs` runs it, so make the run mean something.

If you find a bug, the ideal first commit is the failing test alone.

## Performance claims

The benchmark is `benches/throughput.rs`, and its workloads are separated by which shared structure each one touches so a result points at a cause rather than just a number. Keep that property when adding one.

Three rules for any number in a PR description:

- **Always state the thread count**, and the machine. These workloads behave very differently at 1 thread and at 4, and on a hybrid CPU the default `available_parallelism` counts efficiency cores — on an M1 the 4 → 8 step measures cores getting slower, not contention getting worse. Pass the performance-core count explicitly: `cargo bench -- 4`.
- **Quote the contended figure.** Uniformly random keys leave per-record synchronization uncontended and therefore invisible; the four-hot-row workload exists because that is where design choices show.
- **Report before and after from the same run of the same binary.** Numbers measured on different days on a laptop are not comparable.

An optimisation PR with no measurement will be asked for one before review.

## Style

- `rustfmt.toml` and `clippy.toml` are the style guide; don't hand-format around them.
- **Comments explain why, not what.** The existing ones say what a design choice is protecting or what alternative was tried and rejected. Match that density — it is higher than usual, deliberately, because reasoning about lock-free code from the code alone is not realistic.
- Public items get doc comments with a runnable example where the API is not obvious from its signature. Doc examples are compiled by `cargo test`.
- When a change invalidates something in `DESIGN.md`, update `DESIGN.md` in the same commit. It is a record of decisions, and a stale one is worse than none.

## Sending the change

Commits follow [Conventional Commits](https://www.conventionalcommits.org), lowercase subject, present tense:

```text
feat: sharded live-snapshot set in the oracle
fix: g2 gap and ssi
test: hermitage suite
bench: throughput
docs: isolation level table
```

Keep a PR to one concern. A refactor bundled with a behaviour change is two PRs — on this codebase the review question is always "what invariant does this rely on", and a mixed diff makes that unanswerable.

In the description, say:

- what changes for a user of the crate, if anything;
- which invariant the change relies on, if it touches `engine/`;
- the before/after numbers, if it claims to be faster.

By contributing you agree your work is licensed under **MIT OR Apache-2.0**, the same as the crate.
