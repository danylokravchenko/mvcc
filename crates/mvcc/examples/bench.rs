//! Throughput benchmark, split by which locks each workload exercises.
//!
//!     cargo run --release --example bench [threads]
//!
//! Three workloads, deliberately separated:
//!
//! - **point reads** — `tables` RwLock, `slots` RwLock, `Slot::latest` RwLock.
//!   No oracle mutexes beyond begin/end, no commit lock. This is where pure
//!   lock overhead shows up most clearly.
//! - **read-only transactions** — adds the two `Oracle` mutexes per transaction.
//! - **writes** — adds the global commit lock and the whole commit path.
//!
//! Comparing the first two isolates the oracle's cost; comparing the second two
//! isolates the commit path's.

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
const RUN: Duration = Duration::from_millis(1500);

fn main() -> Result<()> {
    let threads: usize = std::env::args()
        .nth(1)
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

    println!("{threads} threads, {ROWS} rows, {RUN:?} per workload\n");

    let reads_in_one_txn = run(threads, &db, |db, rng| {
        // One transaction, many gets: amortises the oracle over 100 reads.
        let mut tx = db.begin_with::<Snapshot>();
        for _ in 0..100 {
            let _ = tx.get::<Row>(&(rng() % ROWS)).expect("get");
        }
        100
    });

    let txn_per_read = run(threads, &db, |db, rng| {
        // One transaction per read: pays the oracle mutexes every time.
        let mut tx = db.begin_with::<Snapshot>();
        let _ = tx.get::<Row>(&(rng() % ROWS)).expect("get");
        1
    });

    let writes = run(threads, &db, |db, rng| {
        let key = rng() % ROWS;
        db.transaction(|tx| tx.update::<Row>(&key, |r| r.value += 1).map(|_| ()))
            .expect("update");
        1
    });

    report("point reads (100 per txn)", reads_in_one_txn);
    report("read-only txns (1 read each)", txn_per_read);
    report("write txns", writes);

    println!(
        "\noracle cost per txn ≈ {:.0} ns",
        1e9 / txn_per_read as f64 - 1e9 / reads_in_one_txn as f64
    );
    println!(
        "commit path cost per txn ≈ {:.0} ns",
        1e9 / writes as f64 - 1e9 / txn_per_read as f64
    );
    Ok(())
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
    let elapsed = start.elapsed();
    (total.load(Ordering::Relaxed) as f64 / elapsed.as_secs_f64()) as u64
}

fn report(name: &str, per_sec: u64) {
    println!("  {name:<30} {:>12} ops/sec", format_thousands(per_sec));
}

fn format_thousands(n: u64) -> String {
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
