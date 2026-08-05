# AGENTS.md

Working notes for coding agents on this repo. [`CONTRIBUTING.md`](CONTRIBUTING.md) is the authority on process and [`DESIGN.md`](DESIGN.md) on why the engine looks the way it does; this file is the short version plus the things that are easy to get wrong when you cannot run the code in your head.

## What this is

An in-memory MVCC engine: a database's concurrency control with no storage layer. Transactions, four isolation levels, version chains, SSI conflict detection. **No durability, no WAL, no `data_dir`, no query planner, one process.** Those were cut on purpose — do not add them, and do not add a `serde` bound to record fields (nothing is serialised, which is exactly why fields need no traits).

Rust edition 2024, MSRV 1.96. No system libraries, no codegen step, nothing to install.

## How to work here

**Think before coding.** State your assumptions before you write. If a change touches `engine/`, name the invariant you are relying on — that is the review question here, and answering it after the fact usually means answering it wrong. If something is unclear, stop and say what is confusing rather than picking an interpretation silently. If there is a simpler approach, say so; push back when warranted.

**Simplicity first.** The minimum code that solves the problem, nothing speculative. No abstractions for single-use code, no configurability nobody asked for, no error handling for impossible states. On a lock-free engine every added moving part is another thing a sanitizer run has to cover — speculative generality is not free here, it is a liability.

**Surgical changes.** Touch only what you must. Don't "improve" adjacent code, comments, or formatting; the comment density in this codebase is deliberately high and its wording is often load-bearing. Match the existing style even where you would do it differently. Remove imports and helpers that *your* change orphaned; if you spot unrelated dead code, mention it instead of deleting it. Every changed line should trace to the request.

**Goal-driven execution.** Turn the task into something verifiable before starting, and loop until it verifies:

- "fix the anomaly" → a failing case in `tests/isolation_anomalies.rs`, then make it pass
- "make it faster" → a `cargo bench -- <threads>` number on the contended workload, before and after
- "add the attribute" → a compile-level derive test that fails without it

Reasoning about this code in your head is not a substitute for running it. The commands below are the verification, and a concurrency bug that "looks fixed" is the failure mode
this whole section exists to prevent.

## Layout

```text
crates/mvcc/          the library
├── src/core/         timestamps, visibility, isolation typestates, Versioned
│                     — #![forbid(unsafe_code)] at the module level
├── src/engine/       version chains, transactions, oracle, indexes, GC — all unsafe lives here
├── tests/            isolation suites, stress, examples runner, public API surface
├── benches/          throughput.rs (harness = false, its own fn main)
└── examples/         basic, concurrent, isolation, game

crates/mvcc-derive/   #[derive(Mvcc)] — a proc-macro crate can export nothing else
```

`core` and `engine` are private modules. Everything a user touches is re-exported from the crate root, so internal paths can be rearranged freely — but moving an item *out* of the root is a breaking change, and `tests/public_api.rs` makes that fail to compile.

## Commands

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features   # must be zero warnings
cargo test --workspace
cargo bench -- 4                                        # only if you touched the engine
```

Run all four before reporting a change as done. There is no CI in this repo, so the local run is the only check there is.

`cargo test --workspace` includes `tests/examples_run.rs`, which shells out and runs every example in release mode — so a full test run compiles the examples twice and takes noticeably longer than the test count suggests. That is intentional; do not "optimise" it away.

`cargo bench` takes the thread count as a positional argument. Passing it matters: the default is `available_parallelism`, which on a hybrid CPU counts efficiency cores, so an unqualified 4 → 8 comparison measures cores getting slower rather than contention getting worse.

## Invariants worth checking before you edit

- **`src/core/` stays unsafe-free.** If a change there needs `unsafe`, it belongs in `engine/`.
- **Every `unsafe` block gets a `// SAFETY:` comment naming the invariant that keeps it  sound** — not a restatement of what the code does.
- **Epoch pinning.** A transaction pins one epoch for its whole life, which is what lets versions be borrowed instead of reference-counted. A borrow that outlives its guard is a use-after-free that will not reproduce reliably.
- **The registry is append-only.** `Database::table_erased` extends a borrow from a read guard to `&self`, sound only because `register` never removes or replaces an entry. This is why there is no `deregister`.
- **Visibility lives in one function** (`begin <= s < end`). The isolation levels differ only in which snapshot they pass in. Fixing a level by special-casing the predicate is the wrong fix.

## Which test has to move

| you changed | the test |
| --- | --- |
| the visibility predicate or a snapshot rule | `tests/isolation_anomalies.rs`, then `tests/hermitage.rs` |
| SSI, conflict detection, the commit path | `tests/hermitage.rs` — G2-item and G2 in particular |
| version chains, GC, epoch reclamation | `tests/concurrent_stress.rs`, plus a sanitizer run |
| anything re-exported from the crate root | `tests/public_api.rs` |
| the derive | a compile-level test with the attribute in question |
| behaviour a user would read about | the example that demonstrates it |

Isolation tests assert **in both directions** — each anomaly is asserted present or
absent *per level*, because a level that forbids too much is as wrong as one that forbids
too little. `hermitage.rs` quotes the original SQL transcript beside each port; if you add
a case, quote its source too.

Examples end by asserting the world they built is still consistent (`game.rs` checks gold
is conserved and no hero exceeds their carry limit). An example that only prints is not
pulling its weight.

For a bug, the ideal first commit is the failing test alone.

## Style

- `rustfmt.toml` and `clippy.toml` are the style guide. Don't hand-format around them.
- If a `clippy.toml` threshold blocks you, split the function — do not raise the number. Each threshold carries a comment saying what it protects; raising one means editing that comment to explain why the old reasoning stopped holding.
- **Comments explain why, not what**, at a higher density than usual: what a design choice protects, or what alternative was tried and rejected. Reasoning about lock-free code from the code alone is not realistic. Match the surrounding density.
- Public items get doc comments, with a runnable example where the signature is not self-explanatory. Doc examples are compiled by `cargo test`.
- When a change invalidates something in `DESIGN.md`, update `DESIGN.md` in the same commit.

## Commits

[Conventional Commits](https://www.conventionalcommits.org), lowercase subject, present tense — `feat: sharded live-snapshot set in the oracle`, `fix: g2 gap and ssi`.

One concern per commit or PR. A refactor bundled with a behaviour change is two, because the review question here is always "what invariant does this rely on" and a mixed diff makes that unanswerable.

For a performance claim: state the thread count and the machine, quote the contended (four-hot-row) workload rather than the uniform-random one, and report before and after from the same run of the same binary.

---

**This file is working if:** diffs contain nothing that isn't traceable to the request, `unsafe` changes arrive with the invariant already named, and the questions come before the implementation rather than after the sanitizer run.
