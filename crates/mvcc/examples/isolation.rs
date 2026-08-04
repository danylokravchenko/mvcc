//! What each isolation level actually gives you.
//!
//! Every scenario below interleaves two transactions by hand, so the outcomes
//! are deterministic rather than timing-dependent.
//!
//!     cargo run --example isolation

use mvcc::{Config, Database, Mvcc, ReadCommitted, Result, Serializable, Snapshot};

#[derive(Mvcc, Clone, Debug)]
#[mvcc(table = "doctors")]
struct Doctor {
    #[mvcc(primary_key)]
    id: u64,
    name: String,
    on_call: bool,
}

#[derive(Mvcc, Clone, Debug)]
#[mvcc(table = "counters")]
struct Counter {
    #[mvcc(primary_key)]
    id: u64,
    value: i64,
}

fn banner(title: &str) {
    println!("\n\x1b[1m{title}\x1b[0m");
    println!("{}", "─".repeat(title.len()));
}

fn main() -> Result<()> {
    let db = Database::open(Config::in_memory())?;
    db.register::<Doctor>()?;
    db.register::<Counter>()?;

    db.transaction(|tx| {
        tx.insert(Doctor {
            id: 1,
            name: "ada".into(),
            on_call: true,
        })?;
        tx.insert(Doctor {
            id: 2,
            name: "bob".into(),
            on_call: true,
        })?;
        tx.insert(Counter { id: 1, value: 0 })
    })?;

    // ------------------------------------------------------------------
    banner("1. Snapshot: a reader is unaffected by concurrent commits");
    {
        let mut reader = db.begin_with::<Snapshot>();
        let before = reader.get::<Counter>(&1)?.unwrap().value;

        // Someone else commits, start to finish, while `reader` is open.
        db.transaction(|tx| tx.update::<Counter>(&1, |c| c.value = 42).map(|_| ()))?;

        let after = reader.get::<Counter>(&1)?.unwrap().value;
        println!("  reader saw {before} before, {after} after a concurrent commit");
        assert_eq!(before, after, "snapshot isolation must be stable");
        println!("  ✓ the snapshot held: readers never block and never change under you");
    }

    // ------------------------------------------------------------------
    banner("2. ReadCommitted: each statement sees a fresh snapshot");
    {
        let mut reader = db.begin_with::<ReadCommitted>();
        let before = reader.get::<Counter>(&1)?.unwrap().value;

        db.transaction(|tx| tx.update::<Counter>(&1, |c| c.value = 99).map(|_| ()))?;

        let after = reader.get::<Counter>(&1)?.unwrap().value;
        println!("  reader saw {before} before, {after} after a concurrent commit");
        assert_ne!(
            before, after,
            "read committed should observe the new commit"
        );
        println!("  ✓ non-repeatable read — the tradeoff this level makes for cheapness");
    }

    // ------------------------------------------------------------------
    banner("3. Write-write conflict: first committer wins");
    {
        let mut first = db.begin_with::<Snapshot>();
        let mut second = db.begin_with::<Snapshot>();

        first.update::<Counter>(&1, |c| c.value += 1)?;
        first.commit()?;

        // `second` took its snapshot before `first` committed, so writing here
        // would silently drop `first`'s update.
        let result = second.update::<Counter>(&1, |c| c.value += 1);
        match result {
            Err(e) => println!(
                "  second transaction: {e}  (retriable: {})",
                e.is_retriable()
            ),
            Ok(_) => unreachable!("the stale write should have been rejected"),
        }
        println!("  ✓ no lost update");
    }

    // ------------------------------------------------------------------
    banner("4. Write skew: allowed under Snapshot");
    {
        // The rule: at least one doctor must stay on call. Each transaction
        // checks the rule, sees it satisfied, and takes a *different* doctor
        // off call — so there is no write-write conflict to catch them.
        db.transaction(|tx| {
            tx.update::<Doctor>(&1, |d| d.on_call = true)?;
            tx.update::<Doctor>(&2, |d| d.on_call = true).map(|_| ())
        })?;

        let mut t1 = db.begin_with::<Snapshot>();
        let mut t2 = db.begin_with::<Snapshot>();

        let t1_sees = t1.get::<Doctor>(&1)?.unwrap().on_call as u8
            + t1.get::<Doctor>(&2)?.unwrap().on_call as u8;
        let t2_sees = t2.get::<Doctor>(&1)?.unwrap().on_call as u8
            + t2.get::<Doctor>(&2)?.unwrap().on_call as u8;
        let names: Vec<String> = {
            let mut tx = db.begin();
            tx.scan::<Doctor>()?
                .iter()
                .map(|d| d.name.clone())
                .collect()
        };
        println!("  on the rota: {}", names.join(", "));
        println!("  t1 counts {t1_sees} on call, t2 counts {t2_sees} — both think it is safe");

        t1.update::<Doctor>(&1, |d| d.on_call = false)?;
        t2.update::<Doctor>(&2, |d| d.on_call = false)?;
        t1.commit()?;
        t2.commit()?;

        let mut check = db.begin();
        let remaining = check.get::<Doctor>(&1)?.unwrap().on_call as u8
            + check.get::<Doctor>(&2)?.unwrap().on_call as u8;
        println!("  ✗ {remaining} doctors on call — the invariant is broken");
        println!("    Both transactions were individually legal. This is write skew,");
        println!("    and it is why Snapshot is not Serializable.");
    }

    // ------------------------------------------------------------------
    banner("5. Write skew: prevented under Serializable");
    {
        db.transaction(|tx| {
            tx.update::<Doctor>(&1, |d| d.on_call = true)?;
            tx.update::<Doctor>(&2, |d| d.on_call = true).map(|_| ())
        })?;

        let mut t1 = db.begin_with::<Serializable>();
        let mut t2 = db.begin_with::<Serializable>();

        // Both read both rows, exactly as before.
        let _ = t1.get::<Doctor>(&1)?;
        let _ = t1.get::<Doctor>(&2)?;
        let _ = t2.get::<Doctor>(&1)?;
        let _ = t2.get::<Doctor>(&2)?;

        t1.update::<Doctor>(&1, |d| d.on_call = false)?;
        t2.update::<Doctor>(&2, |d| d.on_call = false)?;

        t1.commit()?;
        match t2.commit() {
            Err(e) => println!("  t2: {e}  (retriable: {})", e.is_retriable()),
            Ok(()) => unreachable!("t2 read a row t1 changed; it must not commit"),
        }

        let mut check = db.begin();
        let remaining = check.get::<Doctor>(&1)?.unwrap().on_call as u8
            + check.get::<Doctor>(&2)?.unwrap().on_call as u8;
        println!("  ✓ {remaining} doctor still on call — the invariant held");
        println!("    t2 read doctor 1, which t1 changed, so t2 could not be serialized.");
    }

    // ------------------------------------------------------------------
    banner("6. Retries are the normal way to use Serializable");
    {
        // `transaction_with` reruns the closure on a retriable failure, so the
        // abort in scenario 5 becomes invisible to the caller.
        let mut attempts = 0;
        db.transaction_with::<Serializable, _, _>(|tx| {
            attempts += 1;
            let ada = tx.get::<Doctor>(&1)?.unwrap().on_call;
            let bob = tx.get::<Doctor>(&2)?.unwrap().on_call;
            if ada && bob {
                tx.update::<Doctor>(&1, |d| d.on_call = false)?;
            }
            Ok(())
        })?;
        println!("  committed after {attempts} attempt(s)");
        println!("  ✓ write your transaction as if it runs alone; let the engine retry it");
    }

    banner("Summary");
    println!("  ReadCommitted   cheapest; non-repeatable reads and phantoms");
    println!("  RepeatableRead  stable snapshot; write skew possible");
    println!("  Snapshot        stable snapshot; write skew possible   ← default");
    println!("  Serializable    no anomalies; expect retriable aborts");

    Ok(())
}
