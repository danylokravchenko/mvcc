//! Throughput benchmark, split by which locks each workload exercises.
//!
//! ```text
//! cargo bench                     # all workloads, default thread count
//! cargo bench -- 4                # 4 threads
//! ```
//!
//! Three workloads, deliberately separated so a result points at a cause:
//!
//! - **point reads** — `tables` RwLock, `slots` RwLock, `Slot::latest` RwLock.
//!   One transaction per 100 reads, so per-transaction costs amortise away.
//! - **read-only txns** — the same reads, one transaction each. The difference
//!   from the row above is the `Oracle`'s two mutexes.
//! - **write txns** — adds the global commit lock and the rest of the commit
//!   path.
//!
//! Reading the two *differences* rather than the three absolutes is the point:
//! that is what localised the `parking_lot` win to the oracle (see
//! `mvcc_engine::oracle`) rather than to the version-chain locks.
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

use mvcc::{Config, Database, Mvcc, Result, Snapshot};

#[derive(Mvcc, Clone, Debug)]
#[mvcc(table = "rows")]
struct Row {
    #[mvcc(primary_key)]
    id: u64,
    value: i64,
}

const ROWS: u64 = 10_000;
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
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });

    let db = Arc::new(Database::open(Config::in_memory())?);
    db.register::<Row>()?;
    db.transaction(|tx| {
        for id in 0..ROWS {
            tx.insert(Row { id, value: 0 })?;
        }
        Ok(())
    })?;

    println!("{threads} threads, {ROWS} rows, {SAMPLES} samples of {RUN:?} per workload\n");
    println!("  {:<28} {:>14}   {}", "workload", "median ops/s", "range");

    let reads = sample(threads, &db, |db, rng| {
        let mut tx = db.begin_with::<Snapshot>();
        for _ in 0..100 {
            let _ = tx.get::<Row>(&(rng() % ROWS)).expect("get");
        }
        100
    });
    reads.report("point reads (100 per txn)");

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
    writes.report("write txns");

    // Per-transaction costs, derived from the differences. A value at or below
    // zero means the two workloads are within noise of each other.
    println!(
        "\n  oracle ≈ {:.0} ns/txn, commit path ≈ {:.0} ns/txn",
        nanos_each(read_txns.median()) - nanos_each(reads.median()),
        nanos_each(writes.median()) - nanos_each(read_txns.median()),
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
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}
