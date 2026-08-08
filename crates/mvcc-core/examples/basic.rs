//! Quickstart: derive, register, and run transactions.
//!
//!     cargo run --example basic

use mvcc::{Config, Database, Mvcc, Result};

#[derive(Mvcc, Clone, Debug)]
#[mvcc(table = "accounts")]
struct Account {
    #[mvcc(primary_key)]
    id: u64,

    /// Two accounts may not share an owner name.
    #[mvcc(index(unique))]
    owner: String,

    /// Range-scannable.
    #[mvcc(index)]
    branch: u32,

    balance: i64,
}

fn main() -> Result<()> {
    let db = Database::open(Config::in_memory())?;
    db.register::<Account>()?;

    // ---- insert -----------------------------------------------------------
    // `transaction` runs the closure, commits it, and retries it if it hits a
    // retriable conflict. Snapshot isolation by default.
    db.transaction(|tx| {
        tx.insert(Account {
            id: 1,
            owner: "ada".into(),
            branch: 10,
            balance: 500,
        })?;
        tx.insert(Account {
            id: 2,
            owner: "bob".into(),
            branch: 10,
            balance: 250,
        })?;
        tx.insert(Account {
            id: 3,
            owner: "cleo".into(),
            branch: 20,
            balance: 900,
        })?;
        Ok(())
    })?;

    // ---- read -------------------------------------------------------------
    let mut tx = db.begin();
    let ada = tx.get::<Account>(&1)?.expect("just inserted");
    println!("ada: branch {}, balance {}", ada.branch, ada.balance);

    // A read-only transaction has nothing to commit; dropping it rolls back,
    // which for a reader means simply releasing its snapshot.
    drop(tx);

    // ---- update -----------------------------------------------------------
    db.transaction(|tx| {
        tx.update::<Account>(&1, |a| a.balance -= 100)?;
        tx.update::<Account>(&2, |a| a.balance += 100)?;
        Ok(())
    })?;

    // ---- scan by primary key ----------------------------------------------
    let mut tx = db.begin();
    println!("\nall accounts:");
    for account in tx.scan::<Account>()? {
        println!(
            "  {:>4}  {:<6} branch {}  {:>5}",
            account.id, account.owner, account.branch, account.balance
        );
    }

    // ---- scan by secondary index ------------------------------------------
    println!("\nbranch 10:");
    for account in tx.scan_index(Account::BRANCH, 10u32..=10)? {
        println!("  {} ({})", account.owner, account.balance);
    }
    drop(tx);

    // ---- constraint violations --------------------------------------------
    let mut tx = db.begin();
    let duplicate = tx.insert(Account {
        id: 99,
        owner: "ada".into(),
        branch: 30,
        balance: 0,
    });
    println!("\nreusing owner 'ada': {}", duplicate.unwrap_err());
    drop(tx);

    // ---- delete -----------------------------------------------------------
    db.transaction(|tx| {
        let existed = tx.delete::<Account>(&3)?;
        println!("deleted cleo: {existed}");
        Ok(())
    })?;

    let mut tx = db.begin();
    println!(
        "cleo now: {:?}",
        tx.get::<Account>(&3)?.map(|r| r.to_owned())
    );

    Ok(())
}
