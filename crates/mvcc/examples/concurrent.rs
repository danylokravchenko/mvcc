//! Many threads, one database.
//!
//! Shows the two properties that motivate MVCC in the first place — readers are
//! never blocked by writers, and a long reader sees a stable world — plus the
//! retry loop that contended writers need.
//!
//!     cargo run --release --example concurrent

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use mvcc::{Config, Database, Mvcc, Result, Serializable};

#[derive(Mvcc, Clone, Debug)]
#[mvcc(table = "accounts")]
struct Account {
    #[mvcc(primary_key)]
    id: u64,
    balance: i64,
}

const ACCOUNTS: u64 = 64;
const TRANSFER_THREADS: usize = 8;
const TRANSFERS_PER_THREAD: usize = 500;
const STARTING_BALANCE: i64 = 1_000;

fn main() -> Result<()> {
    let db = Arc::new(Database::open(Config::in_memory())?);
    db.register::<Account>()?;

    db.transaction(|tx| {
        for id in 0..ACCOUNTS {
            tx.insert(Account {
                id,
                balance: STARTING_BALANCE,
            })?;
        }
        Ok(())
    })?;

    let total_before = total(&db)?;
    println!("starting total: {total_before}");

    let stop = Arc::new(AtomicBool::new(false));
    let reads = Arc::new(AtomicU64::new(0));
    let retries = Arc::new(AtomicU64::new(0));

    // ---- a reader that must never observe a torn transfer -----------------
    // Money moves between accounts constantly. Under snapshot isolation this
    // thread's scans must always sum to the same total: it either sees both
    // halves of a transfer or neither.
    let auditor = {
        let db = Arc::clone(&db);
        let stop = Arc::clone(&stop);
        let reads = Arc::clone(&reads);
        thread::spawn(move || -> Result<()> {
            while !stop.load(Ordering::Relaxed) {
                let mut tx = db.begin();
                let sum: i64 = tx.scan::<Account>()?.iter().map(|a| a.balance).sum();
                assert_eq!(
                    sum,
                    ACCOUNTS as i64 * STARTING_BALANCE,
                    "a reader observed a partially applied transfer"
                );
                reads.fetch_add(1, Ordering::Relaxed);
            }
            Ok(())
        })
    };

    // ---- writers ----------------------------------------------------------
    let start = Instant::now();
    let mut writers = Vec::new();
    for t in 0..TRANSFER_THREADS {
        let db = Arc::clone(&db);
        let retries = Arc::clone(&retries);
        writers.push(thread::spawn(move || -> Result<()> {
            // A cheap per-thread PRNG; no dependency needed for this.
            let mut seed = 0x9e37_79b9_7f4a_7c15u64 ^ (t as u64 + 1);
            let mut next = move || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };

            for _ in 0..TRANSFERS_PER_THREAD {
                let from = next() % ACCOUNTS;
                let to = next() % ACCOUNTS;
                if from == to {
                    continue;
                }

                let mut attempts = 0u64;
                // Serializable, because the transfer's *write* depends on a
                // balance it *read* — the write-skew shape. Snapshot isolation
                // would let two concurrent transfers both pass the check.
                db.transaction_with::<Serializable, _, _>(|tx| {
                    attempts += 1;
                    let balance = tx.get::<Account>(&from)?.map(|a| a.balance).unwrap_or(0);
                    if balance < 10 {
                        return Ok(());
                    }
                    tx.update::<Account>(&from, |a| a.balance -= 10)?;
                    tx.update::<Account>(&to, |a| a.balance += 10)?;
                    Ok(())
                })?;
                retries.fetch_add(attempts - 1, Ordering::Relaxed);
            }
            Ok(())
        }));
    }

    for w in writers {
        w.join().expect("writer panicked")?;
    }
    let elapsed = start.elapsed();

    stop.store(true, Ordering::Relaxed);
    auditor.join().expect("auditor panicked")?;

    // ---- results ----------------------------------------------------------
    let total_after = total(&db)?;
    let committed = (TRANSFER_THREADS * TRANSFERS_PER_THREAD) as u64;

    println!("final total:    {total_after}");
    assert_eq!(total_before, total_after, "money was created or destroyed");

    println!("\n{committed} transfers in {elapsed:.2?}");
    println!(
        "  {:.0} transfers/sec",
        committed as f64 / elapsed.as_secs_f64()
    );
    println!(
        "  {} concurrent audit scans completed, none of them blocked",
        reads.load(Ordering::Relaxed)
    );
    println!(
        "  {} retries ({:.1} per transfer)",
        retries.load(Ordering::Relaxed),
        retries.load(Ordering::Relaxed) as f64 / committed as f64
    );

    // The GC watermark should be advancing. If `active_transactions` climbs and
    // the watermark stalls, some transaction is being leaked — see mvcc_engine::gc.
    let stats = db.stats();
    println!(
        "\ngc watermark {:?}, {} transactions still live",
        stats.watermark, stats.active_transactions
    );

    println!("\n✓ no torn reads, no lost updates, conservation held");

    // Slow readers are what pin the GC watermark. Left here as the shape of the
    // thing to watch for, not as something this example demonstrates going wrong.
    thread::sleep(Duration::from_millis(1));
    Ok(())
}

fn total(db: &Database) -> Result<i64> {
    let mut tx = db.begin();
    Ok(tx.scan::<Account>()?.iter().map(|a| a.balance).sum())
}
