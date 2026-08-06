---
name: write-docs
description: Write or revise documentation — doc comments on public API, module-level docs, README, CONTRIBUTING, or any prose explaining code. Use when asked to document, add docs/docstrings/doc comments, write or rewrite a README or CONTRIBUTING, explain a module, or clean up existing docs that are too long or too vague.
---

# Writing documentation

Documentation is a **reference under time pressure**. The reader is mid-task, scanning, and will stop reading the moment they have what they came for. Write for that reader: precise, compact, no filler. This is not a literature book — no scene-setting, no throat-clearing, no useless sentences or idioms, no restating the signature in English.

The bar for every sentence: **would the reader be worse off if it were deleted?** If not, delete it.

Applies equally to doc comments, module docs, README, and CONTRIBUTING.

## The rules

**Say what it does, then what the caller must know.** Behaviour, then constraints — errors, panics, blocking, ordering, thread-safety, allocation, complexity. Those are the things a reader cannot recover from the signature.

**Cut sentences that carry no information.** Delete on sight:

- "This function is used to…" → start with the verb: "Returns…", "Inserts…"
- "As you can see…", "It is important to note that…", "Simply…", "Basically…"
- Restatements of the name: `fn parse_config` → *"Parses the config."*
- Praise for the design ("elegant", "powerful", "robust", "seamless")
- Parameter lists that repeat the type signature and add nothing
- **Aphorisms and rhetorical closers.** The antithesis ("the register changes; the discipline does not", "it is not X, it is Y"), the punchy one-liner ending a section, the em-dash flourish that restates the sentence before it. They read as insight and carry none. If a closing line does not tell the reader to do something different, cut it.

**No hedging and no marketing.** State facts. "Returns `None` if the key is absent" — not "may possibly return `None` in some cases". A README sells by being accurate about scope, including what the thing does *not* do.

**Mention implementation details only when they change what the caller writes.** A detail earns its place if it affects correctness, performance, or the choice between two APIs. Costs O(n) — say so. Takes a lock the caller might already hold — say so. Uses a hash map internally — silence, unless that leaks into ordering guarantees.

**Explain the reasoning, briefly.** One or two sentences on *why* a design is the way it is — what it protects, what was tried and rejected, what the trade is. This is what stops the next reader from "fixing" it. Keep it to the trade, not the history: *"Requires `&mut self` so the compiler proves no reader is live; a concurrent version would put synchronisation back on the read path."* Reasoning belongs in module docs and DESIGN/CONTRIBUTING far more than in per-item docs.

**Document the current state, never the history of changes.** Docs describe what the code does now, for a reader who has never seen an earlier version. Cut:

- "Previously this took a `String`; it now takes `&str`" → just document what it takes
- "Renamed from `fetch_all`", "moved here from `util.rs`", "refactored in v2.1"
- Changelog entries, dated notes, "added in the recent rewrite"
- "TODO: was going to remove this" and other notes to yourself

Git history, the changelog, and release notes are where that belongs. So does a
document whose subject *is* the sequence of decisions — a design record, an ADR,
a roadmap with done-markers. Those are exempt; a doc comment or a README is not.

The exception is the part of the past that still constrains the present: **an alternative that was tried and rejected**, when knowing it stops the next reader from re-trying it. That is reasoning about the current design, not a record of changes — write it in the present tense and about the trade, not the event. *"Reference counting costs an atomic increment per read, which is what this exists to avoid"* — not *"we used to use `Arc` and switched in March"*. Version-dependent facts a caller must act on (deprecations, MSRV, since-tags) are API contract and stay.

**Prefer a code example over a paragraph** whenever the example is shorter or clearer.

## Code examples

An example is worth including when the call is non-obvious: setup order, builder chains, error handling, lifetimes, a pattern the type is really for. Skip it for `fn len(&self) -> usize`.

Make examples **descriptive and real**: names from the domain, not `foo`/`bar`. Show the shape of actual use, not a toy. Keep them minimal. Make them compile, and run them in the test suite if the language supports it (Rust doctests, etc.).

```rust
/// Applies `f` to the record under `key` and stages the result.
///
/// The closure sees the version visible to this transaction's snapshot, so a
/// read-modify-write needs no explicit re-read. Returns [`Error::NotFound`] if
/// no visible version exists; conflicts surface at commit, not here.
///
/// ```
/// tx.update::<Account>(&payer, |a| a.balance -= amount)?;
/// tx.update::<Account>(&payee, |a| a.balance += amount)?;
/// ```
pub fn update<T: Mvcc>(&self, key: &T::Key, f: impl FnOnce(&mut T)) -> Result<()>
```

Versus the version to avoid — longer, and says less:

```rust
/// This is the update function. It is a very useful and powerful method that
/// allows you to update a record in the database.
///
/// # Arguments
/// * `key` - The key of the record to update.
/// * `f` - A closure that will be called with the record.
///
/// # Returns
/// A Result indicating success or failure.
```

## By document type

**Item docs (functions, types, methods).** One summary line, in the imperative or third person, ending in a period. Then only what is needed: errors, panics, invariants, complexity, an example if the call is non-obvious. Most items are two to four lines total. Document every public item.

**Module docs.** The one place reasoning belongs at length. What the module is responsible for, the model the reader needs in their head to use or edit it, and the design trades — what was rejected and why. Structure with headings once it exceeds a screen. Link to the types that implement what you describe.

**README.** For someone deciding whether to use this, in about sixty seconds. Order: what it is in one or two sentences → a code example that shows the real thing working → scope, explicitly including what it does *not* do → install → the rest. Lead with the honest limitation rather than burying it. Tables beat paragraphs for feature/level/comparison matrices.

**CONTRIBUTING.** For someone about to spend a weekend on a change. What belongs in the project and what will be declined, with the reasoning written down — a table of *proposal → why not* saves more time than anything else on the page. Then setup, the exact commands that constitute "checked", what a change has to prove (which test moves), style, and commit/PR conventions. Every claim should be runnable or checkable.

## Match the codebase

Read neighbouring docs before writing. Comment density, tone, heading conventions, whether examples are doctested, British vs American spelling — follow what is there, even where you would do it differently.

If the project has a CLAUDE.md, AGENTS.md, CONTRIBUTING.md, or DESIGN.md with a documented style rule, that rule wins over this skill.

## Revising existing docs

When cleaning up rather than writing fresh:

1. Delete the filler first — the result is often already correct and half the length.
2. Check every factual claim against the code. Stale docs are worse than missing ones.
3. Strip accumulated history — migration notes, "previously", version annotations that no longer bind anyone. Rewrite the page as if the current code were the only version there had ever been.
4. Add what is missing: errors, panics, complexity, the one sentence of reasoning.
4. Do not rewrite wording that is fine. Load-bearing phrasing often looks casual.

**Done when:** every sentence carries information the reader cannot get from the signature, the reasoning behind non-obvious design is one or two sentences long, examples compile, and nothing claims more than the code delivers.
