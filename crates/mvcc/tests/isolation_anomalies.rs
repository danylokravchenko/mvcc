//! The standard anomaly suite, asserted *present or absent per level*.
//!
//! A level that forbids too much is as wrong as one that forbids too little, so
//! each anomaly is tested in both directions: it must occur at the levels that
//! permit it and must not occur at the levels that do not.

use mvcc::{Config, Database, Error, Mvcc, ReadCommitted, Result, Serializable, Snapshot};

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
fn read_only_serializable_transactions_never_abort() -> Result<()> {
    // A transaction that wrote nothing adds no dependency edge and reads a
    // prefix of the commit order, so it is serializable by construction. It
    // must not be aborted just because something it read was concurrently
    // updated — see the soundness note in `Transaction::commit`.
    let db = db_with(&[(1, "a", 10), (2, "a", 20)])?;
    let mut reader = db.begin_with::<Serializable>();

    assert_eq!(reader.get::<Item>(&1)?.unwrap().value, 10);
    assert_eq!(reader.scan::<Item>()?.len(), 2);

    // Every row the reader touched is overwritten, twice over.
    db.transaction(|tx| tx.update::<Item>(&1, |i| i.value = 999).map(|_| ()))?;
    db.transaction(|tx| tx.update::<Item>(&2, |i| i.value = 888).map(|_| ()))?;

    // Still a stable snapshot, and still commits.
    assert_eq!(reader.get::<Item>(&1)?.unwrap().value, 10);
    reader.commit()?;

    Ok(())
}

#[test]
fn an_outgoing_edge_alone_does_not_abort() -> Result<()> {
    // This is what SSI buys over "abort if anything I read changed".
    //
    // T reads row 1, a concurrent transaction overwrites row 1, and T then
    // writes row 2 — which nobody has read. T has an outgoing rw-edge and no
    // incoming one, so it sits in no cycle: order T before the other
    // transaction and the schedule is serial. The old rule aborted T anyway.
    let db = db_with(&[(1, "a", 10), (2, "a", 20)])?;

    let mut t = db.begin_with::<Serializable>();
    assert_eq!(t.get::<Item>(&1)?.unwrap().value, 10);

    db.transaction(|tx| tx.update::<Item>(&1, |i| i.value = 999).map(|_| ()))?;

    t.update::<Item>(&2, |i| i.value = 2)?;
    t.commit()?;

    let mut check = db.begin();
    assert_eq!(check.get::<Item>(&1)?.unwrap().value, 999);
    assert_eq!(check.get::<Item>(&2)?.unwrap().value, 2);
    Ok(())
}

#[test]
fn an_incoming_edge_alone_does_not_abort() -> Result<()> {
    // The mirror image: a transaction writes a row that a concurrent
    // transaction read, but nothing it read was overwritten. Incoming edge
    // only, so no cycle.
    let db = db_with(&[(1, "a", 10), (2, "a", 20)])?;

    let mut reader = db.begin_with::<Serializable>();
    assert_eq!(reader.get::<Item>(&1)?.unwrap().value, 10);

    let mut writer = db.begin_with::<Serializable>();
    writer.update::<Item>(&1, |i| i.value = 11)?;
    writer.commit()?;

    reader.commit()?;
    Ok(())
}

#[test]
fn both_edges_together_abort() -> Result<()> {
    // Add the second edge to `an_outgoing_edge_alone_does_not_abort` and the
    // same transaction must now fail: someone read what it wrote, and someone
    // overwrote what it read.
    let db = db_with(&[(1, "a", 10), (2, "a", 20)])?;

    // `peer` reads row 2, which becomes T's incoming edge once T writes row 2.
    let mut peer = db.begin_with::<Serializable>();
    assert_eq!(peer.get::<Item>(&2)?.unwrap().value, 20);

    let mut t = db.begin_with::<Serializable>();
    assert_eq!(t.get::<Item>(&1)?.unwrap().value, 10);

    // T's outgoing edge: row 1 is overwritten under it.
    //
    // Note the overwriting transaction runs at *snapshot* isolation, so it has
    // no SSI state and its version names no writer. T must still notice the
    // change — "nobody to blame" is not "nothing happened". This is the case
    // that broke when SSI state stopped being allocated below `Serializable`.
    db.transaction(|tx| tx.update::<Item>(&1, |i| i.value = 999).map(|_| ()))?;

    t.update::<Item>(&2, |i| i.value = 2)?;

    let err = t.commit().expect_err("in + out is a dangerous structure");
    assert!(matches!(err, Error::SerializationFailure), "{err}");
    drop(peer);
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

/// A unique index is a constraint on the *database*, so the tests below assert
/// it at every isolation level rather than only the strongest. Nothing about
/// snapshot isolation is supposed to make a duplicate reachable.
mod unique_under_concurrency {
    use super::*;
    use mvcc::{IsolationLevel, RepeatableRead};

    #[derive(Mvcc, Clone, Debug)]
    #[mvcc(table = "members")]
    struct Member {
        #[mvcc(primary_key)]
        id: u64,
        #[mvcc(index(unique))]
        email: String,
    }

    fn member(id: u64, email: &str) -> Member {
        Member {
            id,
            email: email.into(),
        }
    }

    fn empty_db() -> Result<Database> {
        let db = Database::open(Config::in_memory())?;
        db.register::<Member>()?;
        Ok(db)
    }

    fn holders_of(db: &Database, email: &str) -> usize {
        let owned = email.to_string();
        db.begin()
            .scan_where::<Member, _>(move |m| m.email == owned)
            .expect("scan")
            .len()
    }

    /// Two transactions in flight at once, each inserting a different primary
    /// key with the same unique email. Neither can see the other's row — that
    /// is what snapshot isolation is — so the check alone cannot catch this and
    /// the claim has to.
    fn overlapping_inserts_collide<I: IsolationLevel>() -> Result<()> {
        let db = empty_db()?;

        let mut a = db.begin_with::<I>();
        let mut b = db.begin_with::<I>();

        a.insert(member(1, "dup@example.com"))?;
        let second = b.insert(member(2, "dup@example.com"));

        assert!(
            matches!(second, Err(Error::WriteConflict { .. })),
            "{}: the second inserter must be turned away while the first is in \
             flight, got {second:?}",
            I::NAME
        );

        a.commit()?;
        drop(b);

        assert_eq!(
            holders_of(&db, "dup@example.com"),
            1,
            "{}: exactly one row may hold the key",
            I::NAME
        );
        Ok(())
    }

    #[test]
    fn overlapping_inserts_collide_at_every_level() -> Result<()> {
        overlapping_inserts_collide::<ReadCommitted>()?;
        overlapping_inserts_collide::<RepeatableRead>()?;
        overlapping_inserts_collide::<Snapshot>()?;
        overlapping_inserts_collide::<Serializable>()
    }

    /// The other half: the winner has already committed, but the loser's
    /// snapshot predates it. Checking uniqueness at that snapshot would report
    /// the key as free, so the check reads the current committed state instead
    /// — deliberately outside the transaction's own view.
    fn insert_after_someone_elses_commit_collides<I: IsolationLevel>() -> Result<()> {
        let db = empty_db()?;

        // Begins first, so its snapshot cannot include the insert below.
        let mut late = db.begin_with::<I>();
        db.transaction(|tx| tx.insert(member(1, "dup@example.com")))?;

        let outcome = late.insert(member(2, "dup@example.com"));
        assert!(
            matches!(outcome, Err(Error::DuplicateKey { .. })),
            "{}: the key is taken in the committed database, whatever this \
             transaction's snapshot shows, got {outcome:?}",
            I::NAME
        );
        drop(late);

        assert_eq!(holders_of(&db, "dup@example.com"), 1, "{}", I::NAME);
        Ok(())
    }

    #[test]
    fn insert_after_someone_elses_commit_collides_at_every_level() -> Result<()> {
        insert_after_someone_elses_commit_collides::<ReadCommitted>()?;
        insert_after_someone_elses_commit_collides::<RepeatableRead>()?;
        insert_after_someone_elses_commit_collides::<Snapshot>()?;
        insert_after_someone_elses_commit_collides::<Serializable>()
    }

    /// An update moving a row onto a key is the same collision as an insert
    /// taking it, and goes through the same claim.
    #[test]
    fn overlapping_updates_onto_one_key_collide() -> Result<()> {
        let db = empty_db()?;
        db.transaction(|tx| {
            tx.insert(member(1, "a@example.com"))?;
            tx.insert(member(2, "b@example.com"))?;
            Ok(())
        })?;

        let mut a = db.begin();
        let mut b = db.begin();

        assert!(a.update::<Member>(&1, |m| m.email = "c@example.com".into())?);
        let second = b.update::<Member>(&2, |m| m.email = "c@example.com".into());
        assert!(
            matches!(second, Err(Error::WriteConflict { .. })),
            "got {second:?}"
        );

        a.commit()?;
        drop(b);
        assert_eq!(holders_of(&db, "c@example.com"), 1);
        Ok(())
    }

    /// A claim is a reservation, not a grave. Everything a failed or abandoned
    /// transaction claimed has to come back, or one bad insert would burn a key
    /// for the life of the process.
    #[test]
    fn an_abandoned_claim_is_released() -> Result<()> {
        let db = empty_db()?;

        let mut doomed = db.begin();
        doomed.insert(member(1, "free@example.com"))?;
        drop(doomed);

        db.transaction(|tx| tx.insert(member(2, "free@example.com")))?;
        assert_eq!(holders_of(&db, "free@example.com"), 1);

        // Also when the transaction survives the failure: a rejected insert
        // must not leave the key it was refused holding a claim.
        let mut rejected = db.begin();
        assert!(rejected.insert(member(3, "free@example.com")).is_err());
        drop(rejected);

        db.transaction(|tx| tx.delete::<Member>(&2).map(|_| ()))?;
        db.transaction(|tx| tx.insert(member(4, "free@example.com")))?;
        assert_eq!(holders_of(&db, "free@example.com"), 1);
        Ok(())
    }

    /// One transaction may take a key, release it by deleting the row, and take
    /// it again — the claim is per transaction, not per statement, so its own
    /// second write must not be turned away as a conflict with itself.
    #[test]
    fn a_transaction_does_not_collide_with_itself() -> Result<()> {
        let db = empty_db()?;
        db.transaction(|tx| {
            tx.insert(member(1, "self@example.com"))?;
            assert!(matches!(
                tx.insert(member(2, "self@example.com")),
                Err(Error::DuplicateKey { .. })
            ));
            tx.delete::<Member>(&1)?;
            tx.insert(member(2, "self@example.com"))
        })?;
        assert_eq!(holders_of(&db, "self@example.com"), 1);
        Ok(())
    }

    /// The primary key is a unique constraint too, and had the same hole at
    /// `ReadCommitted`, which has no first-committer-wins check to fall back
    /// on: the insert would land on top of the committed row as if it were an
    /// update.
    fn insert_over_a_committed_primary_key_fails<I: IsolationLevel>() -> Result<()> {
        let db = empty_db()?;

        let mut late = db.begin_with::<I>();
        db.transaction(|tx| tx.insert(member(7, "first@example.com")))?;

        let outcome = late.insert(member(7, "second@example.com"));
        assert!(
            outcome.is_err(),
            "{}: inserting over a committed primary key must fail, got \
             {outcome:?}",
            I::NAME
        );
        drop(late);

        let mut check = db.begin();
        assert_eq!(
            check.get::<Member>(&7)?.map(|m| m.email.clone()),
            Some("first@example.com".to_string()),
            "{}: and must not have overwritten it",
            I::NAME
        );
        Ok(())
    }

    /// The other direction of that check, and the thing it could plausibly have
    /// broken: a deleted key is free again. A tombstone is a version like any
    /// other, so "the slot holds a live record" is not the same question as
    /// "the slot exists", and answering the second would make every primary key
    /// single-use.
    #[test]
    fn a_deleted_primary_key_can_be_reused() -> Result<()> {
        let db = empty_db()?;

        db.transaction(|tx| tx.insert(member(1, "first@example.com")))?;
        db.transaction(|tx| tx.delete::<Member>(&1).map(|_| ()))?;
        db.transaction(|tx| tx.insert(member(1, "second@example.com")))?;

        // Including within a single transaction, where the tombstone the
        // re-insert has to see past is the transaction's own uncommitted one.
        db.transaction(|tx| {
            tx.delete::<Member>(&1)?;
            tx.insert(member(1, "third@example.com"))
        })?;

        let mut tx = db.begin();
        assert_eq!(
            tx.get::<Member>(&1)?.map(|m| m.email.clone()),
            Some("third@example.com".to_string())
        );
        Ok(())
    }

    #[test]
    fn insert_over_a_committed_primary_key_fails_at_every_level() -> Result<()> {
        insert_over_a_committed_primary_key_fails::<ReadCommitted>()?;
        insert_over_a_committed_primary_key_fails::<RepeatableRead>()?;
        insert_over_a_committed_primary_key_fails::<Snapshot>()?;
        insert_over_a_committed_primary_key_fails::<Serializable>()
    }
}

#[test]
fn secondary_index_scans_respect_visibility() -> Result<()> {
    let db = db_with(&[(1, "red", 1), (2, "red", 2), (3, "blue", 3)])?;

    let mut reader = db.begin_with::<Snapshot>();
    assert_eq!(
        reader
            .scan_index(Item::TAG, "red".to_string()..="red".to_string())?
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
            .scan_index(Item::TAG, "red".to_string()..="red".to_string())?
            .len(),
        2
    );
    assert_eq!(
        db.begin()
            .scan_index(Item::TAG, "red".to_string()..="red".to_string())?
            .len(),
        1
    );
    Ok(())
}

#[test]
fn index_entries_survive_updates_that_leave_the_key_alone() -> Result<()> {
    // Index maintenance skips any index whose key is unchanged from the version
    // being displaced, on the grounds that the entry is already there from the
    // write that first set it and nothing ever removes one.
    //
    // Worth a test of its own because the failure mode is silent: a row that
    // lost its own index entry does not error, it just stops being returned by
    // scans that should find it, which looks exactly like the visibility
    // recheck doing its job.
    let red = || "red".to_string()..="red".to_string();
    let db = db_with(&[(1, "red", 1), (2, "red", 2)])?;

    // An update touching only a non-indexed field takes the skip path.
    db.transaction(|tx| tx.update::<Item>(&1, |i| i.value += 10).map(|_| ()))?;
    assert_eq!(db.begin().scan_index(Item::TAG, red())?.len(), 2);

    // Repeatedly, so a stale entry cannot be masked by one fresh insert.
    for _ in 0..3 {
        db.transaction(|tx| tx.update::<Item>(&2, |i| i.value += 1).map(|_| ()))?;
    }
    assert_eq!(db.begin().scan_index(Item::TAG, red())?.len(), 2);

    // Out of the group and back: the return write does *not* skip, because the
    // version it displaces has key "green".
    db.transaction(|tx| {
        tx.update::<Item>(&2, |i| i.tag = "green".into())
            .map(|_| ())
    })?;
    assert_eq!(db.begin().scan_index(Item::TAG, red())?.len(), 1);
    db.transaction(|tx| tx.update::<Item>(&2, |i| i.tag = "red".into()).map(|_| ()))?;
    assert_eq!(db.begin().scan_index(Item::TAG, red())?.len(), 2);

    // A delete installs a tombstone, so the write after it sees no previous
    // value and re-indexes rather than skipping.
    db.transaction(|tx| tx.delete::<Item>(&1).map(|_| ()))?;
    db.transaction(|tx| {
        tx.insert(Item {
            id: 1,
            tag: "red".into(),
            value: 1,
        })
    })?;
    assert_eq!(db.begin().scan_index(Item::TAG, red())?.len(), 2);
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

/// Dropping a `Database` must free every version, not just the live ones.
///
/// Regression guard for the epoch-reclamation conversion. The version chain
/// used to be `Arc`-linked, so dropping a slot freed the whole chain for free.
/// An epoch-managed `Atomic` deliberately does *not* own its pointee — that is
/// what lets readers traverse without touching a refcount — so the chains have
/// to be freed explicitly. Miss it and every database ever opened leaks every
/// row it ever held, silently.
#[test]
fn dropping_a_database_frees_every_version() -> Result<()> {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone, Debug)]
    struct Tracked(Arc<AtomicUsize>);

    impl Drop for Tracked {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[derive(Mvcc, Clone, Debug)]
    #[mvcc(table = "tracked")]
    struct Row {
        #[mvcc(primary_key)]
        id: u64,
        payload: Tracked,
    }

    let drops = Arc::new(AtomicUsize::new(0));

    {
        let db = Database::open(Config::in_memory())?;
        db.register::<Row>()?;

        // 100 rows, each updated twice, so each key carries a three-version
        // chain and only the head is reachable by a reader.
        db.transaction(|tx| {
            for id in 0..100 {
                tx.insert(Row {
                    id,
                    payload: Tracked(Arc::clone(&drops)),
                })?;
            }
            Ok(())
        })?;
        for _ in 0..2 {
            db.transaction(|tx| {
                for id in 0..100 {
                    tx.update::<Row>(&id, |r| r.payload = Tracked(Arc::clone(&drops)))?;
                }
                Ok(())
            })?;
        }

        drops.store(0, Ordering::Relaxed);
    } // `db` dropped here

    // 300 versions: 100 keys x 3 versions each. Every one must be freed, not
    // just the 100 still at the head of their chain.
    assert_eq!(
        drops.load(Ordering::Relaxed),
        300,
        "versions leaked when the database was dropped"
    );
    Ok(())
}
