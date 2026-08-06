//! What does a deleted key leave behind, and how much of it does `compact` get
//! back?
//!
//! ```text
//! cargo bench --bench retention
//! ```
//!
//! Not a throughput benchmark. It measures **bytes**, through a counting global
//! allocator, because the numbers quoted in the crate docs and the README —
//! roughly 180 bytes retained per distinct key, nearly all of it reclaimable —
//! are the kind that drift silently once written down.
//!
//! Three points, in the order a long-running process meets them:
//!
//! - **populated** — every key live. The baseline.
//! - **after delete+sweep** — every key deleted, and enough commits driven past
//!   it for the watermark to advance and the sweep to collect the tombstones.
//!   The versions are gone; what remains is the per-key `Record`, its bucket
//!   entry, and its slot in the iteration chunk list.
//! - **after `compact()`** — [`Database::compact`] has rebuilt the map around the
//!   survivors, which is the only thing that frees a `Record`.
//!
//! The gap between the second and third lines is the whole reason `compact`
//! exists. The second line is what a process that never calls it pays forever.
//!
//! Deletes are driven on 50,000 distinct keys while commits are driven on one
//! sentinel key, so the churn that advances the watermark does not itself
//! allocate slots and pollute the measurement.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use mvcc::{Config, Database, Mvcc, Result};

/// Distinct keys inserted and then deleted.
const KEYS: u64 = 50_000;

/// Commits needed for the sweep's round-robin to reach every shard, with margin.
/// Mirrors `Database::SWEEP_INTERVAL * SlotMap::SHARDS`, which are internal.
const FULL_PASS: u64 = 128 * 32 * 70;

/// Committed on repeatedly to advance the watermark. One key, so the churn
/// allocates no records of its own.
const CHURN_KEY: u64 = u64::MAX;

static LIVE: AtomicUsize = AtomicUsize::new(0);

/// Passes everything through to the system allocator, and tracks the live total.
struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LIVE.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new: usize) -> *mut u8 {
        LIVE.fetch_add(new, Ordering::Relaxed);
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

#[derive(Mvcc, Clone)]
#[mvcc(table = "rows")]
struct Row {
    #[mvcc(primary_key)]
    id: u64,
    /// Deliberately chunky, to keep what scales with `T` distinguishable from
    /// the fixed per-key overhead this benchmark is about.
    payload: [u64; 8],
}

fn main() -> Result<()> {
    let base = LIVE.load(Ordering::Relaxed);

    let mut db = Database::open(Config::in_memory())?;
    db.register::<Row>()?;
    db.transaction(|tx| {
        tx.insert(Row {
            id: CHURN_KEY,
            payload: [0; 8],
        })
    })?;

    for id in 0..KEYS {
        db.transaction(|tx| {
            tx.insert(Row {
                id,
                payload: [id; 8],
            })
        })?;
    }
    let populated = LIVE.load(Ordering::Relaxed);

    for id in 0..KEYS {
        db.transaction(|tx| tx.delete::<Row>(&id))?;
    }
    for _ in 0..FULL_PASS {
        db.transaction(|tx| tx.update::<Row>(&CHURN_KEY, |r| r.payload[0] += 1))?;
    }
    settle();
    let swept = LIVE.load(Ordering::Relaxed);

    let freed = db.compact();
    settle();
    let compacted = LIVE.load(Ordering::Relaxed);

    println!("{KEYS} keys, inserted then deleted\n");
    report("populated", populated - base);
    report("after delete+sweep", swept - base);
    report("after compact()", compacted - base);
    println!("\n  records freed by compact : {freed}");
    println!(
        "  reclaimed overall        : {:.1}%",
        (populated - compacted) as f64 / (populated - base) as f64 * 100.0
    );

    drop(db);
    Ok(())
}

fn report(stage: &str, bytes: usize) {
    println!(
        "  {stage:<19}: {bytes:>10} bytes  ({:>6.1} per key)",
        bytes as f64 / KEYS as f64
    );
}

/// Epoch reclamation defers destructors; run them before reading the total, or
/// the measurement charges freed memory to whatever came next.
fn settle() {
    for _ in 0..64 {
        crossbeam_epoch::pin().flush();
    }
}
