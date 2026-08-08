//! Does version reclamation stay O(1) per commit as a chain grows?
//!
//! ```text
//! cargo bench --bench reclamation
//! ```
//!
//! `Slot::prune` runs on the write path, searching down the chain for the
//! boundary between live and dead versions. The cost of that search is the
//! whole question here, because it is paid by every commit and the thing it
//! searches is unbounded.
//!
//! Two columns, and **the second is the one that matters**:
//!
//! - **free** — nothing holds the GC watermark down, so each commit finds
//!   something to collect and chains stay short.
//! - **pinned** — one transaction is held open for the whole run, so the
//!   watermark cannot advance and *there is nothing to collect*. The chain grows
//!   by one version per commit and every prune is doomed to fail.
//!
//! Both columns must stay flat as `updates` rises. A pinned column that climbs
//! with the row count means the search is walking the chain it cannot prune —
//! O(chain) per commit, O(n²) over the run.
//!
//! That is not hypothetical: it is what the first implementation did, and it
//! turned a long-running transaction from a memory problem into a throughput
//! collapse. Measured before [`Slot::PRUNE_PROBE`] bounded the search, on one
//! hot row:
//!
//! ```text
//!    updates     free ns/upd   pinned ns/upd
//!       2000             317            2261
//!       8000             313           10127
//!      32000             305           39023      ← 128x the unpinned cost
//! ```
//!
//! # Why one row, and one thread
//!
//! The quantity under test is chain length, and a chain belongs to a single
//! slot. Spreading writes over many rows shortens every chain and hides the
//! effect; adding threads puts them in `WriteConflict` retries against each
//! other, which measures first-updater-wins rather than pruning. `throughput.rs`
//! is where contention is measured — this benchmark deliberately holds it at
//! zero so the only variable left is how far `prune` walks.

use std::time::{Duration, Instant};

use mvcc::{Config, Database, Mvcc, Result};

/// Chain lengths to measure at. Each is a full run against a fresh database.
const SIZES: [u64; 5] = [2_000, 4_000, 8_000, 16_000, 32_000];

/// Runs per cell; the median is reported. Odd, so the median is a real sample.
const SAMPLES: usize = 5;

#[derive(Mvcc, Clone)]
#[mvcc(table = "rows")]
struct Row {
    #[mvcc(primary_key)]
    id: u64,
    n: u64,
}

fn main() -> Result<()> {
    println!("{SAMPLES} samples per cell, one row, one thread\n");
    println!(
        "  {:>9}  {:>13}  {:>13}   pinned vs free",
        "updates", "free ns/upd", "pinned ns/upd"
    );

    let mut first_pinned = 0.0;
    for (i, updates) in SIZES.into_iter().enumerate() {
        let free = median(|| bump(updates, false));
        let pinned = median(|| bump(updates, true));
        if i == 0 {
            first_pinned = pinned;
        }
        println!(
            "  {updates:>9}  {free:>13.0}  {pinned:>13.0}   {:>6.1}x",
            pinned / free
        );
    }

    // The regression this benchmark exists to catch shows up here rather than in
    // any single row: a bounded search holds this near 1, a walk of the whole
    // chain makes it scale with the largest size.
    let growth = median(|| bump(SIZES[SIZES.len() - 1], true)) / first_pinned;
    println!(
        "\n  pinned cost at {} rows vs {} rows: {growth:.1}x  ({})",
        SIZES[SIZES.len() - 1],
        SIZES[0],
        if growth < 2.0 {
            "flat — the search is bounded"
        } else {
            "GROWING — prune is walking the chain it cannot prune"
        }
    );
    Ok(())
}

/// Nanoseconds per update, applying `updates` of them to a single row.
///
/// A fresh database each time: the chain under test has to start empty, and
/// reusing one would carry the previous run's versions into this one.
///
/// When `pin` is set a transaction is held open for the duration, which stops
/// the GC watermark advancing and so stops anything from being reclaimed.
fn bump(updates: u64, pin: bool) -> f64 {
    let db = Database::open(Config::in_memory()).expect("open");
    db.register::<Row>().expect("register");
    db.transaction(|tx| tx.insert(Row { id: 1, n: 0 }))
        .expect("seed");

    let held = pin.then(|| db.begin());

    let start = Instant::now();
    for _ in 0..updates {
        db.transaction(|tx| tx.update::<Row>(&1, |r| r.n += 1))
            .expect("update");
    }
    let elapsed = start.elapsed();

    // After the timer: dropping it releases the watermark, and that must not be
    // charged to the loop.
    drop(held);

    nanos_each(elapsed, updates)
}

fn nanos_each(elapsed: Duration, updates: u64) -> f64 {
    elapsed.as_secs_f64() * 1e9 / updates as f64
}

/// Median of [`SAMPLES`] runs. Median rather than mean because a single
/// scheduler hiccup should not move the number.
fn median(mut run: impl FnMut() -> f64) -> f64 {
    let mut runs: Vec<f64> = (0..SAMPLES).map(|_| run()).collect();
    runs.sort_by(f64::total_cmp);
    runs[runs.len() / 2]
}
