# MVCC

**ACI, hold the D.** Three quarters of a database, for ordinary Rust structs. Add `#[derive(Mvcc)]` and get atomic transactions, enforced constraints, and four isolation levels including a serializable one that actually is.

```rust
#[derive(Mvcc, Clone, Debug)]
#[mvcc(table = "accounts")]
struct Account {
    #[mvcc(primary_key)]
    id: u64,
    #[mvcc(index(unique))]
    owner: String,
    balance: i64,
}

let db = Database::open(Config::in_memory())?;
db.register::<Account>()?;

db.transaction(|tx| {
    tx.update::<Account>(&1, |a| a.balance -= 50)?;
    tx.update::<Account>(&2, |a| a.balance += 50)?;
    Ok(())
})?;
```

| | | |
| --- | --- | --- |
| **A**tomicity | ✔ | a transaction commits whole or not at all; a dropped one rolls back |
| **C**onsistency | ✔ | primary keys and unique indexes are enforced against the committed database, and `Serializable` preserves any invariant your code checks |
| **I**solation | ✔ | four levels, verified anomaly by anomaly against [Hermitage](#verified-not-asserted) |
| **D**urability | ✘ | nothing is written to disk, ever |

**The missing letter is the one to decide on first.** Everything lives in memory and nothing survives the process — no WAL, no recovery, no `data_dir`. See [Scope](#scope) before going further.

---

## Contents

- [Scope](#scope) — what this is and is not
- [When to use it](#when-to-use-it)
- [How MVCC works here](#how-mvcc-works-here)
- [Using it](#using-it)
- [Isolation levels](#isolation-levels) — and how to choose
- [Performance](#performance)
- [Status](#status)

---

## Scope

This is a database's **concurrency control** without a database's **storage layer** — atomicity, consistency and isolation, but no durability. Transactions, isolation levels, version chains, conflict detection are part of the project. The project doesn't come with WAL, checkpoints, `data_dir`, nor recovery.

Which is a deliberate trade rather than an unfinished one: durability is what forces `fsync` onto the commit path, and dropping it is why a commit here costs a few atomic operations instead of a disk round-trip. What you lose is everything, on process exit.

## When to use it

Reach for this when you have **shared mutable state that several threads read and write, and invariants that span more than one record.**

Good fits:

- An in-process store where readers must never block writers — config state, session tables, routing tables, a simulation's world state, an order book.
- Anywhere you are reaching for `RwLock<HashMap<K, V>>` and finding that a consistent view across *two* maps is impossible without a global lock.
- Enforcing invariants across records ("at least one doctor on call", "balance never negative") where you want the engine to catch the race rather than hand-rolling the locking.
- Long analytical reads over data that is being written concurrently. A reader sees a stable snapshot and never blocks a writer, however long it runs.

Use something else when:

| you need | use |
| --- | --- |
| data to survive a restart | an embedded database — `redb`, `sled`, SQLite |
| more data than fits in RAM | anything with a buffer pool |
| multiple processes | a real database server |
| one map, one thread at a time | `RwLock<HashMap<K, V>>` — genuinely |
| queries by shape rather than by key | a query engine; this has no planner |

The honest comparison is not against Postgres, it is against `RwLock<HashMap<_, _>>`. This wins when *consistency across records* matters or when readers must not be blocked. If neither is true, the lock is simpler and you should use it.

## How MVCC works here

The core idea: **never overwrite data, and never block a reader.** An update writes a *new version* and leaves the old one in place. Readers pick the version that was current when they started, so a reader and a writer touching the same record never wait on each other.

### Version chains

Each record is a `Slot` with a chain of versions, newest first:

```text
index ──► Slot ──► Version { begin: 40, end: MAX, value }   ← current
                        │
                        ▼
                   Version { begin: 20, end: 40,  value }
                        │
                        ▼
                   Version { begin:  5, end: 20,  value }
```

A version is visible to a snapshot `s` when `begin <= s < end`. That is the whole visibility rule, and it lives in exactly one function in the codebase — which is what makes the four isolation levels differ only in *which snapshot they pass in*.

The newest version sits one hop from the index rather than at the end of a chain, because OLTP reads overwhelmingly want the current value. It also keeps the index stable: the slot address never changes, so an update touches only the indexes whose key actually changed.

### Reading

Take a snapshot timestamp, walk the chain until a version is visible at it. No locks, no reference counts, no writes to shared memory — a read is a pointer load and a comparison. A transaction pins one epoch for its whole life, which is what lets versions be borrowed rather than counted.

That holds for finding the record as well as for reading it: the primary key map is append-only, so a lookup probes it without taking a lock. Nothing about a read is visible to any other core, which is why hot rows scale rather than collapse.

Because the snapshot is fixed, a transaction's view never changes underneath it, no matter what commits alongside.

### Writing

First-updater-wins. A writer takes the slot's lock with a compare-exchange; if another transaction already holds it, the write fails immediately with `WriteConflict` rather than waiting. Waiting would reintroduce deadlock detection, which is one of the things MVCC removes.

New versions are tagged as in-flight — invisible to everyone but their author — until commit stamps them with a real timestamp. That makes a commit atomic to readers: they see all of a transaction or none of it, never half.

### Unique constraints

A unique index is enforced against the **committed database**, not against your snapshot, at every isolation level. Two consequences worth knowing:

- An insert fails with `DuplicateKey` if the key is taken by a row committed *after* your snapshot — a row you cannot otherwise see. A constraint is a property of the database; asking whether your snapshot happens to show the collision would be asking the wrong question. Postgres makes the same departure.
- Two transactions in flight at once cannot both take the same key. Neither can see the other's rows, so instead each claims the key for the duration; the second gets a retriable `WriteConflict`, exactly as it would for a contended row.

### Deleting

A delete installs a *tombstone* version rather than removing anything, so a reader at an older snapshot still finds the record alive.

### The failure mode to know about

Old versions can only be reclaimed once no live transaction can still reach them. **One forgotten transaction — a REPL session, a leaked handle, a long-running scan — pins that watermark and version chains grow without limit.** It presents as a memory leak rather than as a transaction problem, and it is the most common way real MVCC systems fall over. `db.stats()` exposes the watermark and the live transaction count; watch them.

## Using it

### Defining a record

```rust
use mvcc::{Config, Database, Mvcc, Result, Serializable};

#[derive(Mvcc, Clone, Debug)]
#[mvcc(table = "accounts")]
struct Account {
    #[mvcc(primary_key)]
    id: u64,

    /// No two accounts may share an owner.
    #[mvcc(index(unique))]
    owner: String,

    /// Range-scannable.
    #[mvcc(index)]
    branch: u32,

    balance: i64,
}
```

| attribute | on | meaning |
| --- | --- | --- |
| `#[mvcc(table = "name")]` | struct | table name in error messages; defaults to the type name |
| `#[mvcc(primary_key)]` | field | required, exactly one |
| `#[mvcc(index)]` | field | secondary index |
| `#[mvcc(index(unique))]` | field | unique secondary index |

An index is always named after its field, and the derive emits a handle for it as an associated const in upper case — `#[mvcc(index)] branch: u32` gives `Account::BRANCH`. That const, not a string, is what a scan takes, so the index name and the range's type are both checked at compile time.

The derive does **not** make the struct itself transactional. `Account` stays a plain struct with no interior mutability, no `Drop` and no hidden fields — all transactional behaviour is on `Transaction`, which is where errors can actually be returned.

### Running transactions

`transaction` runs the closure, commits it, and **retries it** on a retriable conflict. This is the API to reach for: under snapshot isolation, and especially serializable, an abort is a normal outcome rather than an error.

```rust
let db = Database::open(Config::in_memory())?;
db.register::<Account>()?;

db.transaction(|tx| {
    tx.insert(Account { id: 1, owner: "ada".into(), branch: 10, balance: 500 })?;
    tx.insert(Account { id: 2, owner: "bob".into(), branch: 10, balance: 250 })?;
    Ok(())
})?;
```

Because the closure may run more than once, it must not have side effects outside the transaction.

For manual control, `db.begin()` returns a transaction you commit or drop. **Dropping without committing rolls back** — silently committing would be far worse.

```rust
let mut tx = db.begin();
tx.update::<Account>(&1, |a| a.balance += 10)?;
tx.commit()?;
```

### Reading a record

```rust
let mut tx = db.begin();

// By primary key.
if let Some(account) = tx.get::<Account>(&1)? {
    println!("{} has {}", account.owner, account.balance);
}

// Everything, in primary key order.
for account in tx.scan::<Account>()? {
    println!("{}", account.owner);
}

// With a predicate the engine can see — see the note below.
for account in tx.scan_where::<Account, _>(|a| a.balance < 0)? {
    println!("overdrawn: {}", account.owner);
}

// Over a secondary index. `Account::BRANCH` is `Index<Account, u32>`, so a
// misspelled index or a range of the wrong type is a compile error.
for account in tx.scan_index(Account::BRANCH, 10u32..=20)? {
    println!("{}", account.owner);
}
```

**`scan_where` vs `scan().filter()` matters at `Serializable`.** The first hands the predicate to the engine, which re-evaluates it at commit and so can detect a row that *appears* and matches — a phantom. The second tells the engine only that you read the entire table: still correct, but it will abort on any concurrent write to the table at all.

### Modifying a record

```rust
tx.insert(Account { id: 3, owner: "cleo".into(), branch: 20, balance: 0 })?;
tx.update::<Account>(&1, |a| a.balance -= 50)?;   // Ok(false) if absent
tx.delete::<Account>(&2)?;                         // Ok(false) if absent
```

`update` takes a closure rather than handing you a `&mut`, deliberately: the engine needs to know exactly when mutation ends so it can install the version and work out which indexes changed. It may not change the primary key — delete and re-insert instead.

### Handling conflicts and retries

Under MVCC, a conflict is not an error condition — it is the normal way the engine tells you two transactions could not both happen. Code that treats it as a failure will be wrong about half the time.

**The default is that you do not handle it.** `db.transaction` runs the closure, commits, and reruns it from the top if it hit a retriable conflict:

```rust
// If another transaction commits a change to account 1 first, this closure
// simply runs again against a fresh snapshot.
db.transaction(|tx| {
    tx.update::<Account>(&1, |a| a.balance += 10)?;
    Ok(())
})?;
```

Up to 100 attempts, with exponential backoff capped at about a millisecond — so roughly 90ms of retrying before it gives up and returns the last error. The backoff is not decoration: without it two conflicting transactions retry in lockstep and collide at the same point every time.

#### Which errors retry

```rust
if err.is_retriable() { /* WriteConflict or SerializationFailure */ }
```

| error | retriable | meaning |
| --- | --- | --- |
| `WriteConflict` | ✔ | someone else wrote this record, or claimed this unique key, first |
| `SerializationFailure` | ✔ | committing would not have been serializable |
| `DuplicateKey` | ✘ | the value violates a unique index |
| `PrimaryKeyChanged` | ✘ | an `update` tried to change the primary key |
| `TableNotRegistered` | ✘ | `db.register::<T>()` was never called |
| `Aborted` | ✘ | the transaction was already finished |

The non-retriable ones are programming mistakes. Retrying them just burns time before reporting the same thing.

#### The one rule for retried closures

**The closure may run more than once, so it must not have side effects outside the transaction.** Everything it does through `tx` is rolled back on a retry; everything else is not.

```rust
// Wrong: `charged` is incremented once per attempt, not once per transfer.
let mut charged = 0;
db.transaction(|tx| {
    charged += 1;
    tx.update::<Account>(&1, |a| a.balance -= 10)?;
    Ok(())
})?;

// Right: return the value and let the caller act on the committed result.
let amount = db.transaction(|tx| {
    tx.update::<Account>(&1, |a| a.balance -= 10)?;
    Ok(10)
})?;
```

The same applies to sending on a channel, writing a log line that implies the work happened, or incrementing a metric. Do those *after* `transaction` returns `Ok`.

#### Where conflicts surface

They can appear in two places, and which one depends on the isolation level:

- **At the write**, immediately. First-updater-wins: if another transaction already holds that record, `insert`/`update`/`delete` returns `WriteConflict` right away rather than blocking. This is the common case under `Snapshot`.
- **At `commit()`**, as `SerializationFailure`. Only `Serializable` does this — it is SSI deciding the transaction cannot be ordered.

So manual code has to handle both. A failed *write* leaves the transaction usable, so you may continue and commit what you have; a failed *commit* consumes it and has already rolled back.

#### Retrying by hand

`db.transaction` is the right answer almost always. Write the loop yourself when you need a different retry policy, want to give up early, or need to do something between attempts:

```rust
use mvcc::Error;

let mut attempts = 0;
let outcome = loop {
    attempts += 1;
    let mut tx = db.begin_with::<Serializable>();

    // Conflicts can come from the operations…
    let result = (|| {
        let balance = tx.get::<Account>(&1)?.map(|a| a.balance).unwrap_or(0);
        tx.update::<Account>(&1, |a| a.balance = balance - 10)?;
        Ok(())
    })();

    // …or from the commit, so check both.
    match result.and_then(|()| tx.commit()) {
        Ok(()) => break Ok(attempts),
        Err(e) if e.is_retriable() && attempts < 5 => continue,
        Err(e) => break Err(e),
    }
};

match outcome {
    Ok(n) => println!("committed after {n} attempt(s)"),
    Err(Error::SerializationFailure) => println!("gave up; too much contention"),
    Err(e) => return Err(e),
}
```

Note the closure around the operations: `?` needs somewhere to return to, and you want the conflict in your hands rather than propagating out of the loop.

#### If retries keep failing

Exhausting retries means genuine contention, not a bug. The usual causes, in order of likelihood:

- **Too much work in one transaction.** A long transaction has a bigger read set and more time to be invalidated. Split it.
- **`Serializable` where `Snapshot` would do.** Only reach for it when a write depends on something you read — see [Isolation levels](#isolation-levels).
- **`scan()` where `scan_where()` would do.** An unfiltered scan puts the whole table in the read set, so any concurrent write anywhere aborts you.
- **A genuine hotspot.** Every transaction updating one counter row will serialize no matter what the engine does. That needs a different data model — sharded counters, or aggregation outside the transaction.

## Isolation levels

```rust
db.transaction_with::<Serializable, _, _>(|tx| { /* … */ })?;
let tx = db.begin_with::<Serializable>();
```

The level is a **type parameter, not a runtime flag**, so the cost of the strongest never leaks into the weakest: a `ReadCommitted` transaction records no read set, registers no conflict locks and allocates nothing.

| level | sees | permits |
| --- | --- | --- |
| `ReadCommitted` | a fresh snapshot per statement | non-repeatable reads, phantoms |
| `RepeatableRead` | one snapshot | write skew |
| `Snapshot` *(default)* | one snapshot | write skew |
| `Serializable` | one snapshot + conflict detection | nothing |

### Choosing

**Default to `Snapshot`.** It is the right answer for most work: reads never block and never abort, and lost updates are impossible.

**Reach for `Serializable` when a transaction's *write* depends on something it merely *read*.** That is the write-skew shape, and snapshot isolation will not catch it:

```rust
// Two concurrent transfers both read the balance, both see it is sufficient,
// and both withdraw. Under Snapshot they touch different rows on the write
// side, so nothing conflicts and the account goes negative.
db.transaction_with::<Serializable, _, _>(|tx| {
    let balance = tx.get::<Account>(&from)?.map(|a| a.balance).unwrap_or(0);
    if balance < amount { return Ok(()); }
    tx.update::<Account>(&from, |a| a.balance -= amount)?;
    tx.update::<Account>(&to, |a| a.balance += amount)?;
    Ok(())
})?;
```

Balance checks, capacity limits, "at least one of these must remain true", uniqueness enforced in application code — all the same shape. Expect retriable aborts in exchange, and let `transaction_with` handle them.

`Serializable` uses SSI: a transaction aborts only when it has a read/write conflict in **both** directions. An outgoing conflict alone — something you read was overwritten — is not enough, because such a transaction can always be ordered before the one that overwrote it.

**`ReadCommitted` is rarely what you want.** It exists for read-mostly work where seeing the freshest committed value matters more than seeing a consistent one.

### Verified, not asserted

Isolation behaviour is checked against [Hermitage](https://github.com/ept/hermitage), Martin Kleppmann's isolation verification suite: all ten anomalies (G0, G1a/b/c, OTV, PMP, P4, G-single, G2-item, G2), each asserted **present or absent per level**, with the original SQL transcript quoted alongside the port. A level that forbids too much is as wrong as one that forbids too little, so both directions are tested.

Results match PostgreSQL, with one deliberate difference: **lost update (P4) is prevented even at `ReadCommitted`**, because this engine aborts rather than blocking on a locked row, which makes it stronger than SQL read committed there.

See [`crates/mvcc/tests/hermitage.rs`](crates/mvcc/tests/hermitage.rs).

## Performance

`cargo bench`, Apple M1 (4 performance + 4 efficiency cores), so **4 threads is the meaningful number** — an eighth thread lands on an efficiency core and drags the average down for reasons unrelated to contention.

| ops/sec | 1 thread | 4 threads | |
| --- | --- | --- | --- |
| point reads (uniform keys) | 62.2M | **217.2M** | 3.5× |
| point reads (4 hot rows) | 140.4M | **480.8M** | 3.4× |
| read-only transactions | 14.7M | 16.1M | |
| write transactions (snapshot) | 6.2M | 6.1M | |
| write transactions (serializable) | 1.59M | 2.96M | |
| `scan_where`, 1% of 10k rows | 17.0k | 60.6k | |
| `scan_where`, all of 10k rows | 6.2k | 22.8k | |

**Reads scale, including contended ones.** Nothing on the read path writes to shared memory — not a lock word, not a refcount — so four cores reading the same four rows do not fight over a cache line. That took removing two things: the `Arc` refcount on the version chain, and the sharded `RwLock` in front of it, whose *read* acquire was still an atomic read-modify-write. The hot-row row used to run *backwards*, 91M at one thread to 49M at four.

**Writes are roughly flat.** The commit timestamp counter is a single shared cache line and is the current floor.

Three things worth knowing when reading any number here:

- **Always state the thread count.** These workloads behave very differently at
  one thread and at four, and contended figures are the honest ones to quote.
- **Hot rows are the realistic case.** Uniformly random keys leave per-record
  synchronization uncontended and therefore invisible; the benchmark includes a
  four-hot-row workload specifically because that is where design choices show.
- **Scans are per-scan, not per-row**, over a 10,000-row table — so the 1% row
  is roughly 600M rows/sec visited at four threads. An unselective scan is
  dominated by sorting its result, not by reading it.

`benches/throughput.rs` separates workloads by which shared structure they touch, so a result points at a cause rather than just a number.

## Status

Working and tested: version chains, all four isolation levels, SSI, secondary indexes with unique constraints, ordered range scans over them, predicate reads, epoch-based reclamation, a lock-free primary key map.

The examples are the fastest way in, and each one ends by asserting the world it
built is still consistent. [`examples/game.rs`](crates/mvcc/examples/game.rs) is the end-to-end tour: two heroes reaching for the same sword (write conflict), a hero overloading herself (write skew, and the fix), a party filling up (a phantom), an atomic trade, a long report reading while the world moves, and a four-thread raid.

Not implemented, in rough order of how much they would change things:

- **Durability.** No log, no recovery. See [Scope](#scope).
- **Larger-than-memory.** The dataset must fit in RAM.
- **A distributed anything.** One process, one machine.

## License

MIT OR Apache-2.0
