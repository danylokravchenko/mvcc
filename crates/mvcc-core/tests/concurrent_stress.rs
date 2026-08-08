//! Concurrency stress, aimed at the epoch-reclamation machinery.
//!
//! Aborting is the only path that frees a version (`defer_destroy`), so it is
//! the only path where a reader can be traversing a chain whose head is being
//! unlinked underneath it. These tests exist to hammer exactly that overlap.
//!
//! They cannot *prove* absence of a use-after-free; run under a sanitizer for
//! that. What they do is make the window very likely to be hit, so that a
//! mistake shows up as a crash or wrong answer in CI rather than in production.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;

use mvcc::{Config, Database, Error, Mvcc, ReadCommitted, Result, Serializable, Snapshot};

#[derive(Mvcc, Clone, Debug)]
#[mvcc(table = "rows")]
struct Row {
    #[mvcc(primary_key)]
    id: u64,
    value: i64,
    /// A heap field, so a freed version reached through a stale pointer is
    /// likely to fault or read garbage rather than silently look plausible.
    label: String,
}

const KEYS: u64 = 8;

fn seeded(db: &Database) -> Result<()> {
    db.transaction(|tx| {
        for id in 0..KEYS {
            tx.insert(Row {
                id,
                value: 0,
                label: format!("row-{id}"),
            })?;
        }
        Ok(())
    })
}

#[test]
fn readers_survive_writers_aborting_underneath_them() -> Result<()> {
    let db = Arc::new(Database::open(Config::in_memory())?);
    db.register::<Row>()?;
    seeded(&db)?;

    let stop = Arc::new(AtomicBool::new(false));
    let reads = Arc::new(AtomicU64::new(0));
    let aborts = Arc::new(AtomicU64::new(0));

    // Writers that install a version and then always roll it back, so every
    // write ends in `defer_destroy` while readers are mid-traversal.
    let writers: Vec<_> = (0..4)
        .map(|t| {
            let db = Arc::clone(&db);
            let stop = Arc::clone(&stop);
            let aborts = Arc::clone(&aborts);
            thread::spawn(move || {
                let mut seed = 0x9e37_79b9_7f4a_7c15u64 ^ (t + 1);
                while !stop.load(Ordering::Relaxed) {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let key = seed % KEYS;

                    let mut tx = db.begin_with::<Snapshot>();
                    match tx.update::<Row>(&key, |r| {
                        r.value += 1;
                        r.label = format!("pending-{key}");
                    }) {
                        Ok(_) => {
                            tx.abort();
                            aborts.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(e) => assert!(e.is_retriable(), "unexpected: {e}"),
                    }
                }
            })
        })
        .collect();

    // Readers traversing chains the writers are churning.
    let readers: Vec<_> = (0..4)
        .map(|t| {
            let db = Arc::clone(&db);
            let stop = Arc::clone(&stop);
            let reads = Arc::clone(&reads);
            thread::spawn(move || -> Result<()> {
                let mut seed = 0xdead_beef_0bad_f00du64 ^ (t + 1);
                while !stop.load(Ordering::Relaxed) {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let mut tx = db.begin_with::<Snapshot>();
                    let row = tx.get::<Row>(&(seed % KEYS))?.expect("seeded");
                    // Every abort is rolled back, so no committed value ever
                    // changes. Reading anything else means a stale or freed
                    // version was reachable.
                    assert_eq!(row.value, 0, "observed an aborted write");
                    assert_eq!(row.label, format!("row-{}", row.id), "torn version");
                    reads.fetch_add(1, Ordering::Relaxed);
                }
                Ok(())
            })
        })
        .collect();

    thread::sleep(std::time::Duration::from_millis(400));
    stop.store(true, Ordering::Relaxed);
    for w in writers {
        w.join().expect("writer panicked");
    }
    for r in readers {
        r.join().expect("reader panicked")?;
    }

    assert!(
        aborts.load(Ordering::Relaxed) > 100,
        "too few aborts to be meaningful"
    );
    assert!(
        reads.load(Ordering::Relaxed) > 100,
        "too few reads to be meaningful"
    );
    Ok(())
}

#[test]
fn concurrent_inserts_survive_the_slot_map_growing_under_readers() -> Result<()> {
    // Aimed at `engine::slotmap`. The existing tests here work over eight keys
    // seeded up front, so they never grow a bucket array and never race an
    // insert against a lookup — which is precisely the part of that module with
    // no lock to hide behind.
    //
    // 12k keys over 64 shards is roughly 190 per shard, so every shard doubles
    // its array four times from `INITIAL_CAPACITY`, with readers probing
    // throughout. Two failures are in scope: a reader following a bucket array
    // being replaced, and two inserters racing on the same key and each
    // installing a record — a key with two slots, which reads as a lost update
    // rather than as a crash.
    const N: u64 = 12_000;
    const WRITERS: u64 = 4;

    let db = Arc::new(Database::open(Config::in_memory())?);
    db.register::<Row>()?;

    let stop = Arc::new(AtomicBool::new(false));
    let probes = Arc::new(AtomicU64::new(0));

    // Readers probing for keys that are appearing underneath them. A key is
    // either absent or fully formed; a torn label or a missing row that was
    // already observed present would mean a bad publication.
    let readers: Vec<_> = (0..4)
        .map(|t| {
            let db = Arc::clone(&db);
            let stop = Arc::clone(&stop);
            let probes = Arc::clone(&probes);
            thread::spawn(move || -> Result<()> {
                let mut seed = 0xfeed_face_cafe_d00du64 ^ (t + 1);
                while !stop.load(Ordering::Relaxed) {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let id = seed % N;
                    let mut tx = db.begin_with::<Snapshot>();
                    if let Some(row) = tx.get::<Row>(&id)? {
                        assert_eq!(row.id, id, "lookup returned another key's slot");
                        assert_eq!(row.label, format!("row-{id}"), "torn version");
                    }
                    probes.fetch_add(1, Ordering::Relaxed);
                }
                Ok(())
            })
        })
        .collect();

    // Every key is offered by *two* writers, so the same-key insert race is hit
    // continuously rather than incidentally.
    let writers: Vec<_> = (0..WRITERS)
        .map(|t| {
            let db = Arc::clone(&db);
            thread::spawn(move || -> Result<()> {
                for id in (t % 2..N).step_by(2) {
                    let outcome = db.transaction(|tx| {
                        tx.insert(Row {
                            id,
                            value: 0,
                            label: format!("row-{id}"),
                        })
                    });
                    // Losing the race is the expected outcome for one of the
                    // two writers offering each key, either way round.
                    if let Err(e) = outcome
                        && !matches!(e, Error::DuplicateKey { .. } | Error::WriteConflict { .. })
                    {
                        return Err(e);
                    }
                }
                Ok(())
            })
        })
        .collect();

    for w in writers {
        w.join().expect("writer panicked")?;
    }
    stop.store(true, Ordering::Relaxed);
    for r in readers {
        r.join().expect("reader panicked")?;
    }

    // Every key present exactly once. A duplicated slot shows up here as a
    // count above `N`, and a lost record as one below.
    let mut tx = db.begin();
    let rows = tx.scan::<Row>()?;
    assert_eq!(
        rows.len(),
        N as usize,
        "slot count wrong after concurrent growth"
    );
    let mut ids: Vec<u64> = rows.iter().map(|r| r.id).collect();
    ids.dedup();
    assert_eq!(ids.len(), N as usize, "a key was installed in two slots");
    drop(tx);

    // And each one is reachable by the index, not merely present in a scan —
    // a record appended to the chunk list but lost by the bucket array during a
    // resize would pass the assertions above and fail here.
    let mut tx = db.begin();
    for id in 0..N {
        assert!(
            tx.get::<Row>(&id)?.is_some(),
            "key {id} unreachable by lookup"
        );
    }

    assert!(
        probes.load(Ordering::Relaxed) > 1_000,
        "too few concurrent probes to be meaningful"
    );
    Ok(())
}

#[test]
fn contended_serializable_transfers_conserve_and_terminate() -> Result<()> {
    // Every transfer touches two of eight keys at `Serializable`, so aborts,
    // retries, SIREAD registration and reclamation all run flat out and overlap.
    let db = Arc::new(Database::open(Config::in_memory())?);
    db.register::<Row>()?;
    db.transaction(|tx| {
        for id in 0..KEYS {
            tx.insert(Row {
                id,
                value: 1_000,
                label: format!("row-{id}"),
            })?;
        }
        Ok(())
    })?;

    let threads: Vec<_> = (0..4)
        .map(|t| {
            let db = Arc::clone(&db);
            thread::spawn(move || -> Result<()> {
                let mut seed = 0x51_7c_c1_b7u64 ^ (t + 1);
                for _ in 0..300 {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let (from, to) = (seed % KEYS, (seed >> 8) % KEYS);
                    if from == to {
                        continue;
                    }
                    let outcome = db.transaction_with::<Serializable, _, _>(|tx| {
                        let balance = tx.get::<Row>(&from)?.map(|r| r.value).unwrap_or(0);
                        if balance < 10 {
                            return Ok(());
                        }
                        tx.update::<Row>(&from, |r| r.value -= 10)?;
                        tx.update::<Row>(&to, |r| r.value += 10)?;
                        Ok(())
                    });
                    if let Err(e) = outcome {
                        // Retries are bounded, so exhausting them under this
                        // much contention is a legitimate outcome; anything
                        // else is a real failure.
                        match e {
                            Error::SerializationFailure | Error::WriteConflict { .. } => {}
                            e => return Err(e),
                        }
                    }
                }
                Ok(())
            })
        })
        .collect();

    for t in threads {
        t.join().expect("worker panicked")?;
    }

    let mut tx = db.begin();
    let total: i64 = tx.scan::<Row>()?.iter().map(|r| r.value).sum();
    assert_eq!(total, KEYS as i64 * 1_000, "money was created or destroyed");
    Ok(())
}

/// A unique index under a genuine race, rather than a hand-built interleaving.
///
/// Every thread tries to take the same key at the same time, once per round,
/// each with its own primary key — so no two of them touch a common slot and
/// nothing in first-updater-wins, the snapshot, or SSI is between them. The
/// only thing that is, is the claim on the index key.
///
/// The failure this guards is quiet: a duplicate is not a crash and not a torn
/// read, so a test that only watched for those would pass while the constraint
/// silently did nothing. Checked by breaking it — with the claim removed, every
/// round admits two to four winners.
#[test]
fn only_one_transaction_may_take_a_unique_key() -> Result<()> {
    use std::sync::Barrier;

    #[derive(Mvcc, Clone, Debug)]
    #[mvcc(table = "unique_rows")]
    struct Member {
        #[mvcc(primary_key)]
        id: u64,
        #[mvcc(index(unique))]
        email: String,
    }

    const THREADS: u64 = 4;
    const ROUNDS: u64 = 200;

    let db = Arc::new(Database::open(Config::in_memory())?);
    db.register::<Member>()?;
    let barrier = Arc::new(Barrier::new(THREADS as usize));
    let winners = Arc::new(AtomicU64::new(0));

    let threads: Vec<_> = (0..THREADS)
        .map(|t| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let winners = Arc::clone(&winners);
            thread::spawn(move || -> Result<()> {
                for round in 0..ROUNDS {
                    barrier.wait();
                    let email = format!("round-{round}@example.com");
                    let outcome = db.transaction(|tx| {
                        tx.insert(Member {
                            id: round * THREADS + t,
                            email: email.clone(),
                        })
                    });
                    match outcome {
                        Ok(()) => {
                            winners.fetch_add(1, Ordering::Relaxed);
                        }
                        // The two ways of losing: the key was already committed
                        // by the winner, or the winner was still in flight and
                        // the retries ran out.
                        Err(Error::DuplicateKey { .. } | Error::WriteConflict { .. }) => {}
                        Err(e) => return Err(e),
                    }
                }
                Ok(())
            })
        })
        .collect();

    for t in threads {
        t.join().expect("worker panicked")?;
    }

    // One winner per round, and — the assertion that actually catches a lost
    // constraint — one row per key in the table.
    let mut tx = db.begin();
    let rows = tx.scan::<Member>()?;
    assert_eq!(
        rows.len(),
        ROUNDS as usize,
        "{} rows for {ROUNDS} unique keys",
        rows.len()
    );
    let mut emails: Vec<String> = rows.iter().map(|m| m.email.clone()).collect();
    emails.sort();
    let distinct = emails.len();
    emails.dedup();
    assert_eq!(emails.len(), distinct, "a unique key was taken twice");
    assert_eq!(
        winners.load(Ordering::Relaxed),
        ROUNDS,
        "exactly one transaction per round may win"
    );
    Ok(())
}

/// An insert must never quietly become an update.
///
/// The primary key check in `insert` runs before the slot lock, so under a race
/// it is stale by the time the write lands: every thread here looks at a key
/// that does not exist yet, and only then contends for the slot. Above
/// `ReadCommitted` first-committer-wins catches the loser, which is why this
/// runs at `ReadCommitted` — the level with no such check, and the only one
/// where the stale look was the last word.
///
/// The symptom is a lost insert rather than a crash: two transactions both
/// report success and one row is left holding one of the two values.
#[test]
fn a_racing_insert_does_not_overwrite_a_committed_row() -> Result<()> {
    use std::sync::Barrier;

    const THREADS: u64 = 4;
    const ROUNDS: u64 = 200;

    let db = Arc::new(Database::open(Config::in_memory())?);
    db.register::<Row>()?;
    let barrier = Arc::new(Barrier::new(THREADS as usize));
    let winners = Arc::new(AtomicU64::new(0));

    let threads: Vec<_> = (0..THREADS)
        .map(|t| {
            let db = Arc::clone(&db);
            let barrier = Arc::clone(&barrier);
            let winners = Arc::clone(&winners);
            thread::spawn(move || -> Result<()> {
                for round in 0..ROUNDS {
                    barrier.wait();
                    // One fresh key per round, contended by every thread.
                    let outcome = db.transaction_with::<ReadCommitted, _, _>(|tx| {
                        tx.insert(Row {
                            id: round,
                            value: t as i64,
                            label: format!("by-{t}"),
                        })
                    });
                    match outcome {
                        Ok(()) => {
                            winners.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(Error::DuplicateKey { .. } | Error::WriteConflict { .. }) => {}
                        Err(e) => return Err(e),
                    }
                }
                Ok(())
            })
        })
        .collect();

    for t in threads {
        t.join().expect("worker panicked")?;
    }

    assert_eq!(
        winners.load(Ordering::Relaxed),
        ROUNDS,
        "every key must have exactly one successful inserter"
    );
    let mut tx = db.begin();
    assert_eq!(tx.scan::<Row>()?.len(), ROUNDS as usize);
    Ok(())
}
