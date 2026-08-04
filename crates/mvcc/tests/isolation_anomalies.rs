//! The standard anomaly suite, asserted *present or absent per level*.
//!
//! A level that forbids too much is as wrong as one that forbids too little, so
//! each anomaly is tested in both directions: it must occur at the levels that
//! permit it and must not occur at the levels that do not.

use mvcc::{Config, Database, Mvcc, ReadCommitted, Result, Serializable, Snapshot};

#[derive(Mvcc, Clone, Debug, PartialEq)]
#[mvcc(table = "items")]
struct Item {
    #[mvcc(primary_key)]
    id: u64,
    #[mvcc(index)]
    tag: String,
    value: i64,
}

fn db_with(items: &[(u64, &str, i64)]) -> Result<Database> {
    let db = Database::open(Config::in_memory())?;
    db.register::<Item>()?;
    db.transaction(|tx| {
        for (id, tag, value) in items {
            tx.insert(Item {
                id: *id,
                tag: (*tag).into(),
                value: *value,
            })?;
        }
        Ok(())
    })?;
    Ok(db)
}

#[test]
fn dirty_reads_are_impossible_at_every_level() -> Result<()> {
    let db = db_with(&[(1, "a", 10)])?;

    let mut writer = db.begin();
    writer.update::<Item>(&1, |i| i.value = 999)?;

    // The write is installed but uncommitted. Nobody else may see it.
    for observed in [
        db.begin_with::<ReadCommitted>()
            .get::<Item>(&1)?
            .unwrap()
            .value,
        db.begin_with::<Snapshot>().get::<Item>(&1)?.unwrap().value,
        db.begin_with::<Serializable>()
            .get::<Item>(&1)?
            .unwrap()
            .value,
    ] {
        assert_eq!(observed, 10, "read uncommitted data");
    }

    writer.abort();
    assert_eq!(
        db.begin().get::<Item>(&1)?.unwrap().value,
        10,
        "abort must roll back"
    );
    Ok(())
}

#[test]
fn a_transaction_sees_its_own_writes() -> Result<()> {
    let db = db_with(&[(1, "a", 10)])?;
    let mut tx = db.begin();
    tx.update::<Item>(&1, |i| i.value = 42)?;
    assert_eq!(tx.get::<Item>(&1)?.unwrap().value, 42);
    tx.commit()
}

#[test]
fn non_repeatable_read_occurs_only_at_read_committed() -> Result<()> {
    // Permitted at ReadCommitted.
    let db = db_with(&[(1, "a", 10)])?;
    let mut rc = db.begin_with::<ReadCommitted>();
    assert_eq!(rc.get::<Item>(&1)?.unwrap().value, 10);
    db.transaction(|tx| tx.update::<Item>(&1, |i| i.value = 20).map(|_| ()))?;
    assert_eq!(
        rc.get::<Item>(&1)?.unwrap().value,
        20,
        "read committed should see it"
    );

    // Forbidden at Snapshot.
    let db = db_with(&[(1, "a", 10)])?;
    let mut si = db.begin_with::<Snapshot>();
    assert_eq!(si.get::<Item>(&1)?.unwrap().value, 10);
    db.transaction(|tx| tx.update::<Item>(&1, |i| i.value = 20).map(|_| ()))?;
    assert_eq!(
        si.get::<Item>(&1)?.unwrap().value,
        10,
        "snapshot must be stable"
    );
    Ok(())
}

#[test]
fn phantoms_do_not_appear_under_snapshot_isolation() -> Result<()> {
    let db = db_with(&[(1, "a", 10)])?;
    let mut tx = db.begin_with::<Snapshot>();
    assert_eq!(tx.scan::<Item>()?.len(), 1);

    db.transaction(|t| {
        t.insert(Item {
            id: 2,
            tag: "a".into(),
            value: 20,
        })
    })?;

    assert_eq!(tx.scan::<Item>()?.len(), 1, "a phantom row appeared");
    assert_eq!(
        db.begin().scan::<Item>()?.len(),
        2,
        "a new transaction should see it"
    );
    Ok(())
}

#[test]
fn lost_update_is_prevented_by_first_committer_wins() -> Result<()> {
    let db = db_with(&[(1, "a", 100)])?;

    let mut first = db.begin_with::<Snapshot>();
    let mut second = db.begin_with::<Snapshot>();

    first.update::<Item>(&1, |i| i.value += 10)?;
    first.commit()?;

    let err = second
        .update::<Item>(&1, |i| i.value += 10)
        .expect_err("the stale write must be rejected");
    assert!(err.is_retriable());

    assert_eq!(db.begin().get::<Item>(&1)?.unwrap().value, 110);
    Ok(())
}

#[test]
fn write_skew_occurs_under_snapshot_isolation() -> Result<()> {
    // Both rows are non-zero; each transaction zeroes a different one after
    // checking that the *other* is non-zero. Neither writes what the other
    // wrote, so no write-write conflict catches them.
    let db = db_with(&[(1, "a", 1), (2, "a", 1)])?;

    let mut t1 = db.begin_with::<Snapshot>();
    let mut t2 = db.begin_with::<Snapshot>();

    assert_eq!(t1.get::<Item>(&2)?.unwrap().value, 1);
    assert_eq!(t2.get::<Item>(&1)?.unwrap().value, 1);

    t1.update::<Item>(&1, |i| i.value = 0)?;
    t2.update::<Item>(&2, |i| i.value = 0)?;
    t1.commit()?;
    t2.commit()?;

    let mut check = db.begin();
    let sum = check.get::<Item>(&1)?.unwrap().value + check.get::<Item>(&2)?.unwrap().value;
    assert_eq!(
        sum, 0,
        "this is the anomaly Snapshot is documented to permit"
    );
    Ok(())
}

#[test]
fn write_skew_is_prevented_under_serializable() -> Result<()> {
    let db = db_with(&[(1, "a", 1), (2, "a", 1)])?;

    let mut t1 = db.begin_with::<Serializable>();
    let mut t2 = db.begin_with::<Serializable>();

    assert_eq!(t1.get::<Item>(&2)?.unwrap().value, 1);
    assert_eq!(t2.get::<Item>(&1)?.unwrap().value, 1);

    t1.update::<Item>(&1, |i| i.value = 0)?;
    t2.update::<Item>(&2, |i| i.value = 0)?;

    t1.commit()?;
    let err = t2.commit().expect_err("t2 read a row t1 changed");
    assert!(err.is_retriable());

    let mut check = db.begin();
    let sum = check.get::<Item>(&1)?.unwrap().value + check.get::<Item>(&2)?.unwrap().value;
    assert_eq!(sum, 1, "the invariant must hold");
    Ok(())
}

#[test]
fn serializable_read_modify_write_does_not_abort_itself() -> Result<()> {
    // Regression: commit validation re-reads the read set, and must not see the
    // transaction's own in-flight writes when it does.
    let db = db_with(&[(1, "a", 5)])?;
    let mut tx = db.begin_with::<Serializable>();
    let current = tx.get::<Item>(&1)?.unwrap().value;
    tx.update::<Item>(&1, |i| i.value = current + 1)?;
    tx.commit()?;
    assert_eq!(db.begin().get::<Item>(&1)?.unwrap().value, 6);
    Ok(())
}

#[test]
fn a_dropped_transaction_rolls_back() -> Result<()> {
    let db = db_with(&[(1, "a", 10)])?;
    {
        let mut tx = db.begin();
        tx.update::<Item>(&1, |i| i.value = 999)?;
        // No commit; `tx` is dropped here.
    }
    assert_eq!(db.begin().get::<Item>(&1)?.unwrap().value, 10);
    Ok(())
}

#[test]
fn deletes_are_invisible_to_older_snapshots() -> Result<()> {
    let db = db_with(&[(1, "a", 10)])?;
    let mut old_reader = db.begin_with::<Snapshot>();

    db.transaction(|tx| tx.delete::<Item>(&1).map(|_| ()))?;

    assert!(
        old_reader.get::<Item>(&1)?.is_some(),
        "the old snapshot must still see it"
    );
    assert!(
        db.begin().get::<Item>(&1)?.is_none(),
        "a new transaction must not"
    );
    Ok(())
}

#[test]
fn unique_indexes_reject_duplicates_and_allow_reuse_after_delete() -> Result<()> {
    #[derive(Mvcc, Clone, Debug)]
    #[mvcc(table = "users")]
    struct User {
        #[mvcc(primary_key)]
        id: u64,
        #[mvcc(index(unique))]
        email: String,
    }

    let db = Database::open(Config::in_memory())?;
    db.register::<User>()?;
    db.transaction(|tx| {
        tx.insert(User {
            id: 1,
            email: "a@example.com".into(),
        })
    })?;

    let mut tx = db.begin();
    assert!(
        tx.insert(User {
            id: 2,
            email: "a@example.com".into()
        })
        .is_err()
    );
    drop(tx);

    db.transaction(|tx| tx.delete::<User>(&1).map(|_| ()))?;
    db.transaction(|tx| {
        tx.insert(User {
            id: 2,
            email: "a@example.com".into(),
        })
    })?;
    Ok(())
}

#[test]
fn secondary_index_scans_respect_visibility() -> Result<()> {
    let db = db_with(&[(1, "red", 1), (2, "red", 2), (3, "blue", 3)])?;

    let mut reader = db.begin_with::<Snapshot>();
    assert_eq!(
        reader
            .scan_index::<Item, _, _>("tag", "red".to_string()..="red".to_string())?
            .len(),
        2
    );

    // Move item 2 out of the "red" group.
    db.transaction(|tx| {
        tx.update::<Item>(&2, |i| i.tag = "green".into())
            .map(|_| ())
    })?;

    // The old snapshot still sees two; a new transaction sees one. The stale
    // index entry for item 2 is filtered by the recheck, not by index cleanup.
    assert_eq!(
        reader
            .scan_index::<Item, _, _>("tag", "red".to_string()..="red".to_string())?
            .len(),
        2
    );
    assert_eq!(
        db.begin()
            .scan_index::<Item, _, _>("tag", "red".to_string()..="red".to_string())?
            .len(),
        1
    );
    Ok(())
}

#[test]
fn updating_the_primary_key_is_rejected() -> Result<()> {
    let db = db_with(&[(1, "a", 10)])?;
    let mut tx = db.begin();
    assert!(tx.update::<Item>(&1, |i| i.id = 2).is_err());
    Ok(())
}

#[test]
fn fields_may_be_any_type_the_struct_itself_supports() -> Result<()> {
    // Nothing is serialised, so a field needs no trait of its own. A record can
    // hold whatever the struct as a whole can hold.
    use std::collections::HashMap;

    #[derive(Mvcc, Clone, Debug)]
    #[mvcc(table = "sessions")]
    struct Session {
        #[mvcc(primary_key)]
        id: u64,
        headers: HashMap<String, Vec<u8>>,
        started: std::time::Instant,
        handler: fn(u8) -> u8,
    }

    let db = Database::open(Config::in_memory())?;
    db.register::<Session>()?;
    db.transaction(|tx| {
        tx.insert(Session {
            id: 1,
            headers: HashMap::from([("accept".to_string(), b"*/*".to_vec())]),
            started: std::time::Instant::now(),
            handler: |b| b + 1,
        })
    })?;

    let mut tx = db.begin();
    let session = tx.get::<Session>(&1)?.expect("inserted");
    assert_eq!(session.headers["accept"], b"*/*");
    assert_eq!((session.handler)(1), 2);
    assert!(session.started.elapsed().as_secs() < 60);
    Ok(())
}
