//! The Hermitage suite, ported to this engine.
//!
//! <https://github.com/ept/hermitage> — Martin Kleppmann's isolation-level
//! verification suite. Each test below is a faithful translation of a SQL
//! transcript from `postgres.md`, with the original quoted in the doc comment so
//! the port can be checked against the source.
//!
//! # Results for this engine
//!
//! ✓ = prevented, — = possible (anomaly occurs)
//!
//! | level            | G0 | G1a | G1b | G1c | OTV | PMP | P4 | G-single | G2-item | G2 |
//! |:-----------------|:--:|:---:|:---:|:---:|:---:|:---:|:--:|:--------:|:-------:|:--:|
//! | `ReadCommitted`  | ✓  | ✓   | ✓   | ✓   | ✓   | —   | ✓* | —        | —       | —  |
//! | `RepeatableRead` | ✓  | ✓   | ✓   | ✓   | ✓   | ✓   | ✓  | ✓        | —       | —  |
//! | `Snapshot`       | ✓  | ✓   | ✓   | ✓   | ✓   | ✓   | ✓  | ✓        | —       | —  |
//! | `Serializable`   | ✓  | ✓   | ✓   | ✓   | ✓   | ✓   | ✓  | ✓        | ✓       | ✓  |
//!
//! Read-only transactions never abort at any level, including `Serializable`
//! — see `read_only_serializable_transactions_never_abort` in
//! `isolation_anomalies.rs` and the soundness note in `Transaction::commit`.
//! Several transcripts below (G1b, G-single) depend on that: their T1 or T2 is
//! read-only, and an engine that validated it anyway would abort a transaction
//! Postgres commits.
//!
//! Two entries deviate from what Postgres reports, both deliberately:
//!
//! `P4*` — Postgres "read committed" *permits* lost update, because T2 blocks on
//! the row lock and then overwrites. This engine has no blocking mode: a writer
//! that finds a slot locked aborts immediately (`WriteConflict`). That makes
//! `ReadCommitted` here strictly stronger than SQL read committed for this
//! anomaly. [`p4_lost_update_is_prevented_by_aborting_instead_of_blocking`]
//! pins that down, and
//! [`p4_lost_update_via_stale_write_is_possible_at_read_committed`] shows the
//! one shape that does still get through.
//!
//! `Serializable` is SSI: a transaction aborts only when it has an
//! rw-antidependency edge in *both* directions (see `engine::ssi`). An
//! outgoing edge alone — something you read was overwritten — is no longer
//! enough, because such a transaction sits in no cycle. Several transcripts
//! below rely on that.
//!
//! **`G2` is prevented, but only for predicates the engine can see.** Use
//! [`Transaction::scan_where`], not `scan().filter(..)`: the first records the
//! predicate and re-evaluates it at commit, which is what makes a phantom
//! detectable. The second records "I read the whole table" — still sound, but it
//! aborts on any concurrent write to the table at all. See
//! [`scan_then_filter_is_sound_but_far_more_likely_to_abort`].
//!
//! # Translating "BLOCKS"
//!
//! Hermitage's transcripts assume blocking locks — `-- T2, BLOCKS` means the
//! database parks T2 until T1 commits. This engine is first-updater-wins and
//! aborts instead (see `Slot::try_lock`), so wherever a transcript says BLOCKS,
//! the port asserts `WriteConflict` at that point and retries the transaction
//! afterwards. The anomaly outcome is unchanged; only the mechanism differs.

use mvcc::{
    Config, Database, Error, IsolationLevel, Mvcc, ReadCommitted, RepeatableRead, Result,
    Serializable, Snapshot, Transaction,
};

/// Hermitage's fixture: `create table test (id int primary key, value int)`
/// with rows `(1, 10), (2, 20)`.
#[derive(Mvcc, Clone, Debug, PartialEq)]
#[mvcc(table = "test")]
struct Test {
    #[mvcc(primary_key)]
    id: u64,
    value: i64,
}

fn setup() -> Result<Database> {
    let db = Database::open(Config::in_memory())?;
    db.register::<Test>()?;
    db.transaction(|tx| {
        tx.insert(Test { id: 1, value: 10 })?;
        tx.insert(Test { id: 2, value: 20 })
    })?;
    Ok(db)
}

/// `select * from test where id = ?`
fn value<I: IsolationLevel>(tx: &mut Transaction<'_, I>, id: u64) -> Result<Option<i64>> {
    Ok(tx.get::<Test>(&id)?.map(|r| r.value))
}

/// `select * from test where <predicate>`, as `(id, value)` pairs in key order.
///
/// Uses `scan_where` rather than `scan(..).filter(..)` so the predicate reaches
/// the engine. That distinction is the whole of G2: a filter applied afterwards
/// tells the engine only that the transaction read the entire table, whereas a
/// predicate it holds can be re-evaluated at commit to detect phantoms.
fn matching<I: IsolationLevel>(
    tx: &mut Transaction<'_, I>,
    predicate: impl Fn(i64) -> bool + Send + Sync + 'static,
) -> Result<Vec<(u64, i64)>> {
    Ok(tx
        .scan_where::<Test, _>(move |r| predicate(r.value))?
        .iter()
        .map(|r| (r.id, r.value))
        .collect())
}

fn is_write_conflict(e: &Error) -> bool {
    matches!(e, Error::WriteConflict { .. })
}

// ===========================================================================
// G0 — Write Cycles (dirty writes)
// ===========================================================================

/// ```sql
/// update test set value = 11 where id = 1; -- T1
/// update test set value = 12 where id = 1; -- T2, BLOCKS
/// update test set value = 21 where id = 2; -- T1
/// commit; -- T1. This unblocks T2
/// update test set value = 22 where id = 2; -- T2
/// commit; -- T2
/// select * from test; -- either. Shows 1 => 12, 2 => 22
/// ```
fn g0<I: IsolationLevel>() -> Result<()> {
    let db = setup()?;
    let mut t1 = db.begin_with::<I>();
    let mut t2 = db.begin_with::<I>();

    t1.update::<Test>(&1, |r| r.value = 11)?;

    // BLOCKS in Postgres; WriteConflict here.
    let blocked = t2.update::<Test>(&1, |r| r.value = 12).unwrap_err();
    assert!(is_write_conflict(&blocked), "{}: {blocked}", I::NAME);
    t2.abort();

    t1.update::<Test>(&2, |r| r.value = 21)?;
    t1.commit()?;

    // "This unblocks T2" — the retry.
    let mut t2 = db.begin_with::<I>();
    t2.update::<Test>(&1, |r| r.value = 12)?;
    t2.update::<Test>(&2, |r| r.value = 22)?;
    t2.commit()?;

    let mut check = db.begin();
    assert_eq!(value(&mut check, 1)?, Some(12), "{}", I::NAME);
    assert_eq!(value(&mut check, 2)?, Some(22), "{}", I::NAME);
    Ok(())
}

#[test]
fn g0_write_cycles_prevented_at_every_level() -> Result<()> {
    g0::<ReadCommitted>()?;
    g0::<RepeatableRead>()?;
    g0::<Snapshot>()?;
    g0::<Serializable>()
}

// ===========================================================================
// G1a — Aborted Reads
// ===========================================================================

/// ```sql
/// update test set value = 101 where id = 1; -- T1
/// select * from test; -- T2. Still shows 1 => 10
/// abort;  -- T1
/// select * from test; -- T2. Still shows 1 => 10
/// commit; -- T2
/// ```
fn g1a<I: IsolationLevel>() -> Result<()> {
    let db = setup()?;
    let mut t1 = db.begin_with::<I>();
    let mut t2 = db.begin_with::<I>();

    t1.update::<Test>(&1, |r| r.value = 101)?;

    assert_eq!(
        value(&mut t2, 1)?,
        Some(10),
        "{}: read uncommitted",
        I::NAME
    );

    t1.abort();

    assert_eq!(value(&mut t2, 1)?, Some(10), "{}: read aborted", I::NAME);
    t2.commit()
}

#[test]
fn g1a_aborted_reads_prevented_at_every_level() -> Result<()> {
    g1a::<ReadCommitted>()?;
    g1a::<RepeatableRead>()?;
    g1a::<Snapshot>()?;
    g1a::<Serializable>()
}

// ===========================================================================
// G1b — Intermediate Reads
// ===========================================================================

/// ```sql
/// update test set value = 101 where id = 1; -- T1
/// select * from test; -- T2. Still shows 1 => 10
/// update test set value = 11 where id = 1; -- T1
/// commit; -- T1
/// select * from test; -- T2. Now shows 1 => 11   (read committed)
/// commit; -- T2
/// ```
///
/// `after_commit` is what T2's second read must return: 11 for `ReadCommitted`,
/// which refreshes its snapshot per statement, and 10 for the levels that hold
/// one snapshot. Neither may ever be 101 — that is the anomaly.
fn g1b<I: IsolationLevel>(after_commit: i64) -> Result<()> {
    let db = setup()?;
    let mut t1 = db.begin_with::<I>();
    let mut t2 = db.begin_with::<I>();

    t1.update::<Test>(&1, |r| r.value = 101)?;
    assert_eq!(value(&mut t2, 1)?, Some(10), "{}", I::NAME);

    t1.update::<Test>(&1, |r| r.value = 11)?;
    t1.commit()?;

    let seen = value(&mut t2, 1)?;
    assert_ne!(seen, Some(101), "{}: saw the intermediate value", I::NAME);
    assert_eq!(seen, Some(after_commit), "{}", I::NAME);
    t2.commit()
}

#[test]
fn g1b_intermediate_reads_prevented_at_every_level() -> Result<()> {
    g1b::<ReadCommitted>(11)?;
    g1b::<RepeatableRead>(10)?;
    g1b::<Snapshot>(10)?;
    g1b::<Serializable>(10)
}

// ===========================================================================
// G1c — Circular Information Flow
// ===========================================================================

/// ```sql
/// update test set value = 11 where id = 1; -- T1
/// update test set value = 22 where id = 2; -- T2
/// select * from test where id = 2; -- T1. Still shows 2 => 20
/// select * from test where id = 1; -- T2. Still shows 1 => 10
/// commit; -- T1
/// commit; -- T2
/// ```
fn g1c<I: IsolationLevel>() -> Result<()> {
    let db = setup()?;
    let mut t1 = db.begin_with::<I>();
    let mut t2 = db.begin_with::<I>();

    t1.update::<Test>(&1, |r| r.value = 11)?;
    t2.update::<Test>(&2, |r| r.value = 22)?;

    assert_eq!(value(&mut t1, 2)?, Some(20), "{}: T1 saw T2", I::NAME);
    assert_eq!(value(&mut t2, 1)?, Some(10), "{}: T2 saw T1", I::NAME);

    t1.commit()?;
    let outcome = t2.commit();

    // G1c asks only that neither transaction reads the other's uncommitted
    // write, which is asserted above and holds at every level.
    //
    // Whether both then *commit* is a separate question, and at `Serializable`
    // the answer is no. Written out, this interleaving is:
    //
    //   T1: W(1) R(2)        T2: W(2) R(1)
    //
    // T1 read row 2, which T2 overwrites; T2 read row 1, which T1 overwrote.
    // Two rw-antidependencies pointing at each other — the same cycle as
    // G2-item, wearing different clothes. Aborting one is correct, not a false
    // positive.
    if I::VALIDATES_READS {
        let err = outcome.expect_err("serializable must break the rw cycle");
        assert!(matches!(err, Error::SerializationFailure), "{err}");
    } else {
        outcome?;
    }
    Ok(())
}

#[test]
fn g1c_circular_information_flow_prevented_at_every_level() -> Result<()> {
    g1c::<ReadCommitted>()?;
    g1c::<RepeatableRead>()?;
    g1c::<Snapshot>()?;
    g1c::<Serializable>()
}

// ===========================================================================
// OTV — Observed Transaction Vanishes
// ===========================================================================

/// ```sql
/// update test set value = 11 where id = 1; -- T1
/// update test set value = 19 where id = 2; -- T1
/// update test set value = 12 where id = 1; -- T2. BLOCKS
/// commit; -- T1. This unblocks T2
/// select * from test where id = 1; -- T3. Shows 1 => 11
/// update test set value = 18 where id = 2; -- T2
/// select * from test where id = 2; -- T3. Shows 2 => 19
/// commit; -- T2
/// select * from test where id = 2; -- T3. Shows 2 => 18
/// select * from test where id = 1; -- T3. Shows 1 => 12
/// ```
///
/// The anomaly would be T3 observing *half* of T1 — id 1 already at 11 while id
/// 2 is still 20. T3 runs at `ReadCommitted` precisely so it can see committed
/// transactions arrive; the requirement is that each one arrives whole.
#[test]
fn otv_observed_transaction_vanishes_prevented() -> Result<()> {
    let db = setup()?;
    let mut t1 = db.begin_with::<ReadCommitted>();
    let mut t2 = db.begin_with::<ReadCommitted>();
    let mut t3 = db.begin_with::<ReadCommitted>();

    t1.update::<Test>(&1, |r| r.value = 11)?;
    t1.update::<Test>(&2, |r| r.value = 19)?;

    let blocked = t2.update::<Test>(&1, |r| r.value = 12).unwrap_err();
    assert!(is_write_conflict(&blocked));
    t2.abort();

    // T3 sees none of T1 while T1 is in flight — not a half-applied T1.
    assert_eq!(
        (value(&mut t3, 1)?, value(&mut t3, 2)?),
        (Some(10), Some(20))
    );

    t1.commit()?;

    // Now T3 sees all of T1.
    assert_eq!(
        (value(&mut t3, 1)?, value(&mut t3, 2)?),
        (Some(11), Some(19)),
        "T3 observed a partially applied T1"
    );

    let mut t2 = db.begin_with::<ReadCommitted>();
    t2.update::<Test>(&1, |r| r.value = 12)?;
    t2.update::<Test>(&2, |r| r.value = 18)?;

    // T2 in flight: T3 still sees all of T1 and none of T2.
    assert_eq!(
        (value(&mut t3, 1)?, value(&mut t3, 2)?),
        (Some(11), Some(19))
    );

    t2.commit()?;

    assert_eq!(
        (value(&mut t3, 1)?, value(&mut t3, 2)?),
        (Some(12), Some(18)),
        "T3 observed a partially applied T2"
    );
    t3.commit()
}

/// Under `Snapshot`, OTV cannot arise at all: T3 sees its own snapshot
/// throughout and neither T1 nor T2 ever becomes visible to it.
#[test]
fn otv_prevented_trivially_under_snapshot_isolation() -> Result<()> {
    let db = setup()?;
    let mut t3 = db.begin_with::<Snapshot>();

    db.transaction(|tx| {
        tx.update::<Test>(&1, |r| r.value = 11)?;
        tx.update::<Test>(&2, |r| r.value = 19).map(|_| ())
    })?;

    assert_eq!(
        (value(&mut t3, 1)?, value(&mut t3, 2)?),
        (Some(10), Some(20))
    );
    t3.commit()
}

// ===========================================================================
// PMP — Predicate-Many-Preceders
// ===========================================================================

/// ```sql
/// select * from test where value = 30; -- T1. Returns nothing
/// insert into test (id, value) values(3, 30); -- T2
/// commit; -- T2
/// select * from test where value % 3 = 0; -- T1. Still returns nothing (RR+)
/// commit; -- T1
/// ```
fn pmp<I: IsolationLevel>(sees_the_insert: bool) -> Result<()> {
    let db = setup()?;
    let mut t1 = db.begin_with::<I>();

    assert!(matching(&mut t1, |v| v == 30)?.is_empty(), "{}", I::NAME);

    db.transaction(|tx| tx.insert(Test { id: 3, value: 30 }))?;

    let found = matching(&mut t1, |v| v % 3 == 0)?;
    if sees_the_insert {
        assert_eq!(found, vec![(3, 30)], "{}: should see the new row", I::NAME);
    } else {
        assert!(found.is_empty(), "{}: predicate-many-preceders", I::NAME);
    }
    t1.commit()
}

#[test]
fn pmp_possible_at_read_committed_prevented_above() -> Result<()> {
    pmp::<ReadCommitted>(true)?;
    pmp::<RepeatableRead>(false)?;
    pmp::<Snapshot>(false)?;
    pmp::<Serializable>(false)
}

/// PMP for write predicates.
///
/// ```sql
/// update test set value = value + 10; -- T1
/// delete from test where value = 20;  -- T2, BLOCKS
/// commit; -- T1
/// -- read committed: T2's delete proceeds against the new state
/// -- repeatable read: "ERROR: could not serialize access due to concurrent update"
/// ```
/// `matches_20_after_t1` is what T2's `where value = 20` returns once T1 has
/// committed. T1 adds 10 to both rows, so `(1, 10), (2, 20)` becomes
/// `(1, 20), (2, 30)`:
///
/// - `ReadCommitted` re-reads and matches **row 1**, which qualifies only
///   because of T1's update — the anomaly Postgres calls out as "returns
///   1 => 20 (despite ostensibly having been deleted)".
/// - the snapshot levels keep T2's original snapshot, still see
///   `(1, 10), (2, 20)`, and match **row 2**.
///
/// Note that T2 is the *same* transaction throughout, as in the transcript. A
/// write that loses the first-updater-wins race returns `WriteConflict` without
/// installing anything and without killing the transaction, so T2 can carry on
/// and its snapshot is preserved — which is what makes the distinction above
/// observable.
fn pmp_write<I: IsolationLevel>(matches_20_after_t1: (u64, i64)) -> Result<()> {
    let db = setup()?;
    let mut t1 = db.begin_with::<I>();
    let mut t2 = db.begin_with::<I>();

    t1.update::<Test>(&1, |r| r.value += 10)?;
    t1.update::<Test>(&2, |r| r.value += 10)?;

    // T2's predicate matches the pre-T1 state: T1 is uncommitted.
    assert_eq!(
        matching(&mut t2, |v| v == 20)?,
        vec![(2, 20)],
        "{}",
        I::NAME
    );

    // `delete from test where value = 20` — BLOCKS in Postgres, conflicts here.
    let blocked = t2.delete::<Test>(&2).unwrap_err();
    assert!(is_write_conflict(&blocked), "{}", I::NAME);

    t1.commit()?; // "This unblocks T2"

    assert_eq!(
        matching(&mut t2, |v| v == 20)?,
        vec![matches_20_after_t1],
        "{}",
        I::NAME
    );
    Ok(())
}

#[test]
fn pmp_write_predicates() -> Result<()> {
    // Read committed matches a row that qualifies only because of T1.
    pmp_write::<ReadCommitted>((1, 20))?;
    // The snapshot levels match the row that qualified all along.
    pmp_write::<RepeatableRead>((2, 20))?;
    pmp_write::<Snapshot>((2, 20))?;
    pmp_write::<Serializable>((2, 20))
}

// ===========================================================================
// P4 — Lost Update
// ===========================================================================

/// ```sql
/// select * from test where id = 1; -- T1
/// select * from test where id = 1; -- T2
/// update test set value = 11 where id = 1; -- T1
/// update test set value = 11 where id = 1; -- T2, BLOCKS
/// commit; -- T1. read committed: T1's update is overwritten
/// ```
///
/// Postgres permits this at read committed because T2 blocks and then
/// overwrites. This engine aborts instead of blocking, so the update is never
/// lost — `ReadCommitted` here is stronger than SQL read committed for P4.
fn p4<I: IsolationLevel>() -> Result<()> {
    let db = setup()?;
    let mut t1 = db.begin_with::<I>();
    let mut t2 = db.begin_with::<I>();

    assert_eq!(value(&mut t1, 1)?, Some(10));
    assert_eq!(value(&mut t2, 1)?, Some(10));

    t1.update::<Test>(&1, |r| r.value = 11)?;

    let blocked = t2.update::<Test>(&1, |r| r.value = 11).unwrap_err();
    assert!(is_write_conflict(&blocked), "{}: {blocked}", I::NAME);
    assert!(blocked.is_retriable(), "{}", I::NAME);

    t1.commit()?;
    t2.abort();

    assert_eq!(value(&mut db.begin(), 1)?, Some(11), "{}", I::NAME);
    Ok(())
}

#[test]
fn p4_lost_update_is_prevented_by_aborting_instead_of_blocking() -> Result<()> {
    p4::<ReadCommitted>()?;
    p4::<RepeatableRead>()?;
    p4::<Snapshot>()?;
    p4::<Serializable>()
}

/// The one lost-update shape that still gets through `ReadCommitted`.
///
/// Not from Hermitage — added because the transcript above cannot distinguish
/// the levels on this engine, and something should.
///
/// T2 reads, T1 commits and *releases the slot lock*, then T2 writes a value
/// derived from its stale read. There is no lock to conflict on any more, so
/// only a snapshot check can catch it — which is exactly what
/// `FIRST_COMMITTER_WINS` is, and `ReadCommitted` is the one level that does
/// not set it.
#[test]
fn p4_lost_update_via_stale_write_is_possible_at_read_committed() -> Result<()> {
    // ReadCommitted: the increment computed from the stale read wins.
    let db = setup()?;
    let mut t2 = db.begin_with::<ReadCommitted>();
    let stale = value(&mut t2, 1)?.unwrap();

    db.transaction(|tx| tx.update::<Test>(&1, |r| r.value = 100).map(|_| ()))?;

    t2.update::<Test>(&1, |r| r.value = stale + 1)?;
    t2.commit()?;

    assert_eq!(
        value(&mut db.begin(), 1)?,
        Some(11),
        "read committed lost the concurrent update to 100"
    );

    // Snapshot: first-committer-wins rejects the same write.
    let db = setup()?;
    let mut t2 = db.begin_with::<Snapshot>();
    let stale = value(&mut t2, 1)?.unwrap();

    db.transaction(|tx| tx.update::<Test>(&1, |r| r.value = 100).map(|_| ()))?;

    let err = t2.update::<Test>(&1, |r| r.value = stale + 1).unwrap_err();
    assert!(is_write_conflict(&err), "{err}");
    Ok(())
}

// ===========================================================================
// G-single — Read Skew
// ===========================================================================

/// ```sql
/// select * from test where id = 1; -- T1. Shows 1 => 10
/// select * from test where id = 1; -- T2
/// select * from test where id = 2; -- T2
/// update test set value = 12 where id = 1; -- T2
/// update test set value = 18 where id = 2; -- T2
/// commit; -- T2
/// select * from test where id = 2; -- T1. Shows 2 => 18 (RC) or 2 => 20 (RR+)
/// commit; -- T1
/// ```
fn g_single<I: IsolationLevel>(t1_sees: i64) -> Result<()> {
    let db = setup()?;
    let mut t1 = db.begin_with::<I>();

    assert_eq!(value(&mut t1, 1)?, Some(10), "{}", I::NAME);

    db.transaction(|tx| {
        tx.update::<Test>(&1, |r| r.value = 12)?;
        tx.update::<Test>(&2, |r| r.value = 18).map(|_| ())
    })?;

    assert_eq!(value(&mut t1, 2)?, Some(t1_sees), "{}: read skew", I::NAME);

    // T1 wrote nothing. Its snapshot is a prefix of the commit order, so
    // ordering it before T2 is a valid serial schedule and it commits at every
    // level — including Serializable, which skips validation for read-only
    // transactions. See the soundness note in `Transaction::commit`.
    t1.commit()
}

#[test]
fn g_single_read_skew_possible_at_read_committed_prevented_above() -> Result<()> {
    g_single::<ReadCommitted>(18)?;
    g_single::<RepeatableRead>(20)?;
    g_single::<Snapshot>(20)?;
    g_single::<Serializable>(20)
}

/// G-single using predicate dependencies.
///
/// ```sql
/// select * from test where value % 5 = 0; -- T1
/// update test set value = 12 where value = 10; -- T2
/// commit; -- T2
/// select * from test where value % 3 = 0; -- T1. Returns nothing
/// commit; -- T1
/// ```
#[test]
fn g_single_predicate_read_skew_prevented_under_snapshot() -> Result<()> {
    let db = setup()?;
    let mut t1 = db.begin_with::<Snapshot>();

    assert_eq!(matching(&mut t1, |v| v % 5 == 0)?, vec![(1, 10), (2, 20)]);

    db.transaction(|tx| tx.update::<Test>(&1, |r| r.value = 12).map(|_| ()))?;

    assert!(
        matching(&mut t1, |v| v % 3 == 0)?.is_empty(),
        "T1's snapshot should not contain 12"
    );
    t1.commit()
}

/// G-single using a write predicate.
///
/// ```sql
/// select * from test where id = 1; -- T1. Shows 1 => 10
/// select * from test; -- T2
/// update test set value = 12 where id = 1; -- T2
/// update test set value = 18 where id = 2; -- T2
/// commit; -- T2
/// delete from test where value = 20; -- T1. "could not serialize access"
/// abort; -- T1
/// ```
#[test]
fn g_single_write_predicate_conflicts_under_snapshot() -> Result<()> {
    let db = setup()?;
    let mut t1 = db.begin_with::<Snapshot>();

    assert_eq!(value(&mut t1, 1)?, Some(10));

    db.transaction(|tx| {
        tx.update::<Test>(&1, |r| r.value = 12)?;
        tx.update::<Test>(&2, |r| r.value = 18).map(|_| ())
    })?;

    // T1 still sees 2 => 20 and would delete it; the row has moved under it.
    assert_eq!(matching(&mut t1, |v| v == 20)?, vec![(2, 20)]);
    let err = t1.delete::<Test>(&2).unwrap_err();
    assert!(is_write_conflict(&err), "{err}");
    Ok(())
}

// ===========================================================================
// G2-item — Item Anti-dependency Cycles (write skew on disjoint read)
// ===========================================================================

/// ```sql
/// select * from test where id in (1,2); -- T1
/// select * from test where id in (1,2); -- T2
/// update test set value = 11 where id = 1; -- T1
/// update test set value = 21 where id = 2; -- T2
/// commit; -- T1
/// commit; -- T2. serializable: "could not serialize access due to
/// --                            read/write dependencies among transactions"
/// ```
fn g2_item<I: IsolationLevel>() -> Result<()> {
    let db = setup()?;
    let mut t1 = db.begin_with::<I>();
    let mut t2 = db.begin_with::<I>();

    for id in [1, 2] {
        value(&mut t1, id)?;
        value(&mut t2, id)?;
    }

    t1.update::<Test>(&1, |r| r.value = 11)?;
    t2.update::<Test>(&2, |r| r.value = 21)?;

    t1.commit()?;
    let outcome = t2.commit();

    if I::VALIDATES_READS {
        let err = outcome.expect_err("serializable must prevent write skew");
        assert!(matches!(err, Error::SerializationFailure), "{err}");
        // The invariant "not both changed" holds.
        let mut check = db.begin();
        assert_eq!(
            (value(&mut check, 1)?, value(&mut check, 2)?),
            (Some(11), Some(20))
        );
    } else {
        outcome?;
        let mut check = db.begin();
        assert_eq!(
            (value(&mut check, 1)?, value(&mut check, 2)?),
            (Some(11), Some(21)),
            "{}: write skew is documented as possible here",
            I::NAME
        );
    }
    Ok(())
}

#[test]
fn g2_item_write_skew_possible_below_serializable() -> Result<()> {
    g2_item::<ReadCommitted>()?;
    g2_item::<RepeatableRead>()?;
    g2_item::<Snapshot>()?;
    g2_item::<Serializable>()
}

// ===========================================================================
// G2 — Anti-Dependency Cycles (write skew on predicate read)
// ===========================================================================

/// ```sql
/// select * from test where value % 3 = 0; -- T1
/// select * from test where value % 3 = 0; -- T2
/// insert into test (id, value) values(3, 30); -- T1
/// insert into test (id, value) values(4, 42); -- T2
/// commit; -- T1
/// commit; -- T2. Postgres serializable: "could not serialize access ..."
/// select * from test where value % 3 = 0; -- Either. Returns 3 => 30, 4 => 42
/// ```
///
/// # How this is prevented
///
/// A read set built from slots cannot express "and nothing else matched": rows
/// 3 and 4 did not exist when the predicates were evaluated, so there was no
/// slot to record. `scan_where` therefore records the *predicate itself*
/// alongside its result set, and commit re-evaluates it — see `PredicateRead`
/// in `engine::txn`.
///
/// T2 read `value % 3 = 0` and saw nothing. By the time T2 commits, T1's row 3
/// matches. The result set changed, so T2 cannot be serialized.
#[test]
fn g2_predicate_write_skew_is_prevented() -> Result<()> {
    let db = setup()?;
    let mut t1 = db.begin_with::<Serializable>();
    let mut t2 = db.begin_with::<Serializable>();

    assert!(matching(&mut t1, |v| v % 3 == 0)?.is_empty());
    assert!(matching(&mut t2, |v| v % 3 == 0)?.is_empty());

    t1.insert(Test { id: 3, value: 30 })?;
    t2.insert(Test { id: 4, value: 42 })?;

    t1.commit()?;
    let err = t2
        .commit()
        .expect_err("T1's insert became a phantom in T2's predicate");
    assert!(matches!(err, Error::SerializationFailure), "{err}");

    // Only T1's row is present, which is the serial outcome T1-then-T2 with T2
    // rolled back — not the "both believed nothing matched" state.
    let mut check = db.begin();
    assert_eq!(matching(&mut check, |v| v % 3 == 0)?, vec![(3, 30)]);
    Ok(())
}

/// The complement, and the reason predicates are recorded rather than whole
/// tables: an insert that does **not** satisfy the predicate must not abort
/// anyone.
///
/// Without this, "prevent G2" could be achieved trivially and uselessly by
/// aborting on any concurrent insert at all.
#[test]
fn g2_prevention_does_not_abort_on_irrelevant_inserts() -> Result<()> {
    let db = setup()?;
    let mut t1 = db.begin_with::<Serializable>();
    let mut t2 = db.begin_with::<Serializable>();

    assert!(matching(&mut t1, |v| v % 3 == 0)?.is_empty());
    assert!(matching(&mut t2, |v| v % 3 == 0)?.is_empty());

    // 31 and 32 are not multiples of 3: outside both predicates.
    t1.insert(Test { id: 3, value: 31 })?;
    t2.insert(Test { id: 4, value: 32 })?;

    t1.commit()?;
    t2.commit()?;

    let mut check = db.begin();
    assert!(matching(&mut check, |v| v % 3 == 0)?.is_empty());
    Ok(())
}

/// A filter applied *after* `scan` is not a predicate the engine can see, so it
/// falls back to "I read the whole table" — sound, but it aborts on any
/// concurrent write anywhere in the table.
///
/// Documented as a test because it is the performance trap in this API.
#[test]
fn scan_then_filter_is_sound_but_far_more_likely_to_abort() -> Result<()> {
    let db = setup()?;
    let mut t1 = db.begin_with::<Serializable>();
    let mut t2 = db.begin_with::<Serializable>();

    // The filter never reaches the engine; both see "the whole table".
    let _ = t1
        .scan::<Test>()?
        .iter()
        .filter(|r| r.value % 3 == 0)
        .count();
    let _ = t2
        .scan::<Test>()?
        .iter()
        .filter(|r| r.value % 3 == 0)
        .count();

    // Inserts that match nobody's predicate.
    t1.insert(Test { id: 3, value: 31 })?;
    t2.insert(Test { id: 4, value: 32 })?;

    t1.commit()?;
    let err = t2
        .commit()
        .expect_err("the whole-table read set is invalidated");
    assert!(matches!(err, Error::SerializationFailure), "{err}");
    Ok(())
}

/// The item-level half of G2 *is* caught, which localises the gap precisely:
/// phantoms are invisible, existing rows are not.
#[test]
fn g2_is_prevented_when_the_predicate_covers_existing_rows() -> Result<()> {
    let db = setup()?;
    let mut t1 = db.begin_with::<Serializable>();
    let mut t2 = db.begin_with::<Serializable>();

    // Same predicate shape as G2, but matching rows that already exist.
    assert_eq!(matching(&mut t1, |v| v % 10 == 0)?, vec![(1, 10), (2, 20)]);
    assert_eq!(matching(&mut t2, |v| v % 10 == 0)?, vec![(1, 10), (2, 20)]);

    t1.update::<Test>(&1, |r| r.value = 15)?;
    t2.update::<Test>(&2, |r| r.value = 25)?;

    t1.commit()?;
    let err = t2.commit().expect_err("T2 read row 1, which T1 changed");
    assert!(matches!(err, Error::SerializationFailure), "{err}");
    Ok(())
}
