//! Throughput benchmark, split by which locks each workload exercises.
//!
//! ```text
//! cargo bench                     # all workloads, default thread count
//! cargo bench -- 4                # 4 threads
//! ```
//!
//! Three workloads, deliberately separated so a result points at a cause:
//!
//! - **point reads** — the `tables` `RwLock`, the `slots` `RwLock`, and an
//!   atomic load of `Slot::latest`.
//!   One transaction per 100 reads, so per-transaction costs amortise away.
//! - **point reads (hot)** — the same, but every thread on the same four rows.
//!   Uniform keys leave per-slot synchronisation uncontended and therefore
//!   invisible; this is the workload that shows it.
//! - **read-only txns** — one transaction per read. The difference from the
//!   first row is the `Oracle`'s per-transaction cost.
//! - **write txns** — adds the global commit lock and the rest of the commit
//!   path.
//! - **predicate scans** — the only workloads that visit every slot, so the
//!   only ones where `Table::matching`'s per-row costs are visible at all. Run
//!   at two selectivities because they stress different halves: 1% is dominated
//!   by *visiting* rows, 100% by *returning* them.
//! - **update+scan (serializable)** — the same scan in the shape that pays full
//!   price: a pending write forces the read set to be built from a separate
//!   view, and commit re-runs the predicate. Read-only transactions skip
//!   validation entirely, so the write is what makes this row measure it.
//!
//! Reading the two *differences* rather than the three absolutes is the point:
//! that is what localised the `parking_lot` win to the oracle (see
//! `engine::oracle`) rather than to the version-chain locks.
//!
//! # Choosing a thread count
//!
//! The default is `available_parallelism`, which on a hybrid CPU counts
//! efficiency cores too. On an Apple M1 that is 8, of which only 4 are
//! performance cores — so the 4 → 8 step measures cores getting slower, not
//! contention getting worse, and reads appear to stop scaling when they have
//! not. Pass the performance-core count explicitly when comparing runs:
//!
//! ```text
//! sysctl -n hw.perflevel0.logicalcpu    # macOS: performance cores
//! cargo bench -- 4
//! ```
//!
//! # Why `harness = false`, and no criterion
//!
//! Criterion is built for single-threaded latency micro-benchmarks: it calls a
//! closure many times and does outlier analysis on per-call duration. These
//! workloads are multi-threaded *throughput* over a fixed wall-clock window,
//! which does not fit that shape. Rather than fight it, each workload runs
//! `SAMPLES` times and reports the median with its spread — enough to tell a
//! real change from noise, which is the only question this benchmark is asked.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use mvcc::{Config, Database, Mvcc, Result, Serializable, Snapshot};

#[derive(Mvcc, Clone, Debug)]
#[mvcc(table = "rows")]
struct Row {
    #[mvcc(primary_key)]
    id: u64,
    value: i64,
}

const ROWS: u64 = 10_000;
/// Keys every thread contends on, for the hot-row read workload.
const HOT_ROWS: u64 = 4;
const RUN: Duration = Duration::from_millis(700);
/// Odd, so the median is a measured value rather than the mean of two.
const SAMPLES: usize = 5;

fn main() -> Result<()> {
    // Cargo passes `--bench` to `harness = false` targets, so skip flags and
    // take the first positional argument as the thread count.
    let threads: usize = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with('-'))
        .and_then(|a| a.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, std::num::NonZero::get));

    let db = Arc::new(Database::open(Config::in_memory())?);
    db.register::<Row>()?;
    db.transaction(|tx| {
        for id in 0..ROWS {
            tx.insert(Row { id, value: 0 })?;
        }
        Ok(())
    })?;

    println!("{threads} threads, {ROWS} rows, {SAMPLES} samples of {RUN:?} per workload\n");
    println!("  {:<28} {:>14}   range", "workload", "median ops/s");

    let reads = sample(threads, &db, |db, rng| {
        let mut tx = db.begin_with::<Snapshot>();
        for _ in 0..100 {
            let _ = tx.get::<Row>(&(rng() % ROWS)).expect("get");
        }
        100
    });
    reads.report("point reads (100 per txn)");

    // Every thread hammers the same handful of rows. Uniform keys spread across
    // 10k slots leave per-slot synchronisation uncontended and so invisible;
    // this is where it shows.
    let hot = sample(threads, &db, |db, rng| {
        let mut tx = db.begin_with::<Snapshot>();
        for _ in 0..100 {
            let _ = tx.get::<Row>(&(rng() % HOT_ROWS)).expect("get");
        }
        100
    });
    hot.report("point reads (4 hot rows)");

    let read_txns = sample(threads, &db, |db, rng| {
        let mut tx = db.begin_with::<Snapshot>();
        let _ = tx.get::<Row>(&(rng() % ROWS)).expect("get");
        1
    });
    read_txns.report("read-only txns (1 read)");

    let writes = sample(threads, &db, |db, rng| {
        let key = rng() % ROWS;
        db.transaction(|tx| tx.update::<Row>(&key, |r| r.value += 1).map(|_| ()))
            .expect("update");
        1
    });
    writes.report("write txns (snapshot)");

    // The same work at `Serializable`, which adds SIREAD lock registration on
    // every read and a reader/predicate scan per write at commit. The gap
    // between this row and the one above is the price of SSI.
    let ssi_writes = sample(threads, &db, |db, rng| {
        let key = rng() % ROWS;
        db.transaction_with::<Serializable, _, _>(|tx| {
            let current = tx.get::<Row>(&key)?.map(|r| r.value).unwrap_or(0);
            tx.update::<Row>(&key, |r| r.value = current + 1)
                .map(|_| ())
        })
        .expect("update");
        1
    });
    ssi_writes.report("write txns (serializable)");

    // Predicate scans. These are the only workloads that visit every slot in
    // the table, so they are where `Table::matching`'s per-row costs show up at
    // all — the point-read rows above touch exactly one slot and hide them.
    //
    // Selective and unselective are separated because they stress different
    // halves: the 1% variant is dominated by the cost of *visiting* rows, the
    // 100% variant adds the cost of *returning* them. A change that only helps
    // one of the two will show up as exactly that.
    let scan_selective = sample(threads, &db, |db, _| {
        let mut tx = db.begin_with::<Snapshot>();
        tx.scan_where::<Row, _>(|r| r.id.is_multiple_of(100))
            .expect("scan");
        1
    });
    scan_selective.report("scan_where 1% (snapshot)");

    let scan_all = sample(threads, &db, |db, _| {
        let mut tx = db.begin_with::<Snapshot>();
        tx.scan_where::<Row, _>(|_| true).expect("scan");
        1
    });
    scan_all.report("scan_where 100% (snapshot)");

    // The same selective scan at `Serializable`, in the shape that actually
    // pays for it: **write first, then scan**.
    //
    // Both halves of that matter. A read-only transaction skips validation
    // entirely (see `Transaction::commit`), so without the update this row
    // would measure SIREAD registration and nothing else. And with a write
    // already pending, `scan_where` cannot reuse its result as the observed
    // set — it has to scan a second time as nobody — so this is the worst case
    // for a predicate read, at three table passes: two here, one at commit.
    let scan_ssi = sample(threads, &db, |db, rng| {
        let key = rng() % ROWS;
        db.transaction_with::<Serializable, _, _>(|tx| {
            tx.update::<Row>(&key, |r| r.value += 1)?;
            tx.scan_where::<Row, _>(|r| r.id.is_multiple_of(100))?;
            Ok(())
        })
        .expect("scan+update");
        1
    });
    scan_ssi.report("update+scan 1% (serializable)");

    // Per-transaction costs, derived from the differences. A value at or below
    // zero means the two workloads are within noise of each other.
    println!(
        "\n  oracle ≈ {:.0} ns/txn, commit path ≈ {:.0} ns/txn, SSI ≈ {:.0} ns/txn",
        nanos_each(read_txns.median()) - nanos_each(reads.median()),
        nanos_each(writes.median()) - nanos_each(read_txns.median()),
        nanos_each(ssi_writes.median()) - nanos_each(writes.median()),
    );
    Ok(())
}

fn nanos_each(ops_per_sec: u64) -> f64 {
    1e9 / ops_per_sec as f64
}

/// The samples for one workload, sorted ascending.
struct Samples(Vec<u64>);

impl Samples {
    fn median(&self) -> u64 {
        self.0[self.0.len() / 2]
    }

    /// Spread is printed as a percentage of the median. More than a few percent
    /// means the machine is too noisy to trust a small difference between runs.
    fn report(&self, name: &str) {
        let (lo, hi) = (self.0[0], self.0[self.0.len() - 1]);
        let median = self.median();
        let spread = (hi - lo) as f64 / median as f64 * 100.0;
        println!(
            "  {name:<28} {:>14}   {} … {}  ±{spread:.0}%",
            thousands(median),
            thousands(lo),
            thousands(hi),
        );
    }
}

fn sample(
    threads: usize,
    db: &Arc<Database>,
    op: impl Fn(&Database, &mut dyn FnMut() -> u64) -> u64 + Send + Sync + Copy + 'static,
) -> Samples {
    let mut runs: Vec<u64> = (0..SAMPLES).map(|_| run(threads, db, op)).collect();
    runs.sort_unstable();
    Samples(runs)
}

/// Run `op` on `threads` threads for `RUN`, returning operations per second.
fn run(
    threads: usize,
    db: &Arc<Database>,
    op: impl Fn(&Database, &mut dyn FnMut() -> u64) -> u64 + Send + Sync + Copy + 'static,
) -> u64 {
    let total = Arc::new(AtomicU64::new(0));
    let start = Instant::now();

    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let db = Arc::clone(db);
            let total = Arc::clone(&total);
            thread::spawn(move || {
                let mut seed = 0x9e37_79b9_7f4a_7c15u64 ^ (t as u64 + 1);
                let mut rng = move || {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    seed
                };
                let mut done = 0u64;
                while start.elapsed() < RUN {
                    for _ in 0..64 {
                        done += op(&db, &mut rng);
                    }
                }
                total.fetch_add(done, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().expect("worker panicked");
    }
    (total.load(Ordering::Relaxed) as f64 / start.elapsed().as_secs_f64()) as u64
}

fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}
