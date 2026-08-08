//! Names every item the crate root promises, so a merge that quietly drops one
//! fails to compile.
use mvcc::config::OracleConfig;
use mvcc::stats::GcStats;
use mvcc::{
    Config, Database, Error, IsolationLevel, Mvcc, ReadCommitted, Ref, RepeatableRead, Result,
    Serializable, Snapshot, Timestamp, Transaction, TxnId, Versioned,
};

#[derive(Mvcc, Clone, Debug)]
struct Row {
    #[mvcc(primary_key)]
    id: u64,
    #[mvcc(index)]
    tag: String,
}

#[test]
fn public_surface() -> Result<()> {
    fn _assert<T>() {}
    _assert::<Error>();
    _assert::<Timestamp>();
    _assert::<TxnId>();
    _assert::<GcStats>();
    _assert::<OracleConfig>();
    fn _levels<I: IsolationLevel>() {}
    _levels::<ReadCommitted>();
    _levels::<RepeatableRead>();
    _levels::<Snapshot>();
    _levels::<Serializable>();
    fn _versioned<T: Versioned>() {}
    _versioned::<Row>();

    let db = Database::open(Config::in_memory())?;
    db.register::<Row>()?;
    db.transaction(|tx: &mut Transaction<'_, Snapshot>| {
        tx.insert(Row {
            id: 1,
            tag: "a".into(),
        })
    })?;
    db.transaction_with::<Serializable, _, _>(|tx| {
        let rows: Vec<Ref<'_, Row>> = tx.scan_where::<Row, _>(|r| r.tag == "a")?;
        assert_eq!(rows.len(), 1);
        Ok(())
    })?;
    let _stats: GcStats = db.stats();
    Ok(())
}
