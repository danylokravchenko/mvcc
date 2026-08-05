//! A game world: heroes, loot, inventories, trades, a party roster.
//!
//!     cargo run --release --example game
//!
//! An end-to-end tour built around a workload MVCC is genuinely good at. A game
//! server is shared mutable state touched by many players at once, and almost
//! every rule that makes it a *game* is an invariant spanning more than one
//! record: an item has exactly one owner, a hero cannot carry more than their
//! strength allows, gold is conserved by a trade, a party has a maximum size.
//!
//! Those are the rules that a plain `RwLock<HashMap<_, _>>` makes you enforce by
//! hand, and that snapshot isolation alone quietly lets you break. Each act
//! below picks one and shows what the engine does with it.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::Instant;

use mvcc::{Config, Database, Error, Mvcc, Result, Serializable, Snapshot};

// ===========================================================================
// The world
// ===========================================================================

#[derive(Mvcc, Clone, Debug)]
#[mvcc(table = "heroes")]
struct Hero {
    #[mvcc(primary_key)]
    id: u64,
    /// Two heroes may not share a name — a unique index, enforced by the engine.
    #[mvcc(index(unique))]
    name: String,
    gold: i64,
    /// Total weight this hero can carry.
    strength: i32,
}

/// Items on the ground have this owner. A sentinel rather than an `Option`,
/// because secondary indexes are built over ordered byte keys and `u64` has one.
const GROUND: u64 = 0;

#[derive(Mvcc, Clone, Debug)]
#[mvcc(table = "items")]
struct Item {
    #[mvcc(primary_key)]
    id: u64,
    /// Who holds it, or [`GROUND`]. Indexed, so "show me a hero's inventory" is
    /// a range scan rather than a table scan.
    #[mvcc(index)]
    owner: u64,
    name: String,
    weight: i32,
    value: i64,
    /// An ordinary Rust enum. Nothing here is serialised, so a record's fields
    /// need no traits of their own — see `flavour` below for the extreme case.
    kind: Kind,
    /// Arbitrary per-item state. A `HashMap` in a database row, with no
    /// serialisation format to agree on, because the row never leaves memory.
    flavour: HashMap<String, String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Weapon,
    Potion,
    Trinket,
}

#[derive(Mvcc, Clone, Debug)]
#[mvcc(table = "party_members")]
struct PartyMember {
    #[mvcc(primary_key)]
    hero: u64,
    #[mvcc(index)]
    party: u64,
    role: String,
}

const PARTY: u64 = 1;
const PARTY_LIMIT: usize = 4;

// ===========================================================================
// Queries — the vocabulary the acts are written in
// ===========================================================================

/// Everything a hero is carrying, by way of the `owner` index.
fn inventory<I: mvcc::IsolationLevel>(
    tx: &mut mvcc::Transaction<'_, I>,
    hero: u64,
) -> Result<Vec<(u64, String, i32)>> {
    Ok(tx
        .scan_index::<Item, _, _>("owner", hero..=hero)?
        .iter()
        .map(|i| (i.id, i.name.clone(), i.weight))
        .collect())
}

/// How much a hero is carrying. Uses the same index read, so under
/// `Serializable` it is *this* that a concurrent pickup invalidates.
fn carried<I: mvcc::IsolationLevel>(tx: &mut mvcc::Transaction<'_, I>, hero: u64) -> Result<i32> {
    Ok(tx
        .scan_index::<Item, _, _>("owner", hero..=hero)?
        .iter()
        .map(|i| i.weight)
        .sum())
}

fn gold_of<I: mvcc::IsolationLevel>(tx: &mut mvcc::Transaction<'_, I>, hero: u64) -> Result<i64> {
    Ok(tx.get::<Hero>(&hero)?.map(|h| h.gold).unwrap_or(0))
}

fn name_of<I: mvcc::IsolationLevel>(
    tx: &mut mvcc::Transaction<'_, I>,
    hero: u64,
) -> Result<String> {
    Ok(tx
        .get::<Hero>(&hero)?
        .map(|h| h.name.clone())
        .unwrap_or_else(|| "?".into()))
}

fn party_size<I: mvcc::IsolationLevel>(tx: &mut mvcc::Transaction<'_, I>) -> Result<usize> {
    Ok(tx.scan_where::<PartyMember, _>(|m| m.party == PARTY)?.len())
}

fn total_gold(db: &Database) -> Result<i64> {
    let mut tx = db.begin();
    Ok(tx.scan::<Hero>()?.iter().map(|h| h.gold).sum())
}

fn act(n: u32, title: &str) {
    println!("\n\x1b[1m── Act {n}. {title}\x1b[0m");
}

// A narrative example: one linear script of acts, read top to bottom. Splitting
// it into helpers to satisfy the length and complexity budgets would make it
// harder to follow, which is the only thing this file is for.
#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
fn main() -> Result<()> {
    let db = Arc::new(Database::open(Config::in_memory())?);
    db.register::<Hero>()?;
    db.register::<Item>()?;
    db.register::<PartyMember>()?;

    let (ada, bram, cleo) = (1u64, 2u64, 3u64);

    // -----------------------------------------------------------------------
    act(1, "The world is created");

    db.transaction(|tx| {
        tx.insert(Hero {
            id: ada,
            name: "Ada".into(),
            gold: 120,
            strength: 100,
        })?;
        tx.insert(Hero {
            id: bram,
            name: "Bram".into(),
            gold: 80,
            strength: 100,
        })?;
        tx.insert(Hero {
            id: cleo,
            name: "Cleo".into(),
            gold: 200,
            strength: 60,
        })?;

        let mut item = |id, owner, name: &str, weight, value, kind| {
            tx.insert(Item {
                id,
                owner,
                name: name.into(),
                weight,
                value,
                kind,
                flavour: HashMap::new(),
            })
        };
        item(10, ada, "Rusty Sword", 30, 15, Kind::Weapon)?;
        item(11, ada, "Health Potion", 5, 20, Kind::Potion)?;
        item(12, bram, "Oak Shield", 40, 35, Kind::Weapon)?;
        item(13, cleo, "Lucky Charm", 2, 90, Kind::Trinket)?;
        // Loot lying in the dungeon, owned by nobody.
        item(20, GROUND, "Flaming Greatsword", 55, 500, Kind::Weapon)?;
        item(21, GROUND, "Elven Cloak", 20, 240, Kind::Trinket)?;
        item(22, GROUND, "Iron Helm", 35, 60, Kind::Weapon)?;

        // Arbitrary per-item state, in a database row, with no schema to
        // declare and no serialisation format to agree on.
        tx.update::<Item>(&20, |i| {
            i.flavour.insert("enchantment".into(), "flame".into());
            i.flavour.insert("forged_by".into(), "Durin".into());
        })?;
        Ok(())
    })?;

    // A whole transaction that fails partway leaves nothing behind. The unique
    // index on `name` refuses the second Ada, and the gold change goes with it.
    let before = gold_of(&mut db.begin(), ada)?;
    let doomed = db.transaction(|tx| {
        tx.update::<Hero>(&ada, |h| h.gold += 10_000)?;
        tx.insert(Hero {
            id: 99,
            name: "Ada".into(),
            gold: 0,
            strength: 10,
        })
    });
    println!("  a duplicate hero name: {}", doomed.unwrap_err());
    println!(
        "  Ada's gold is still {} — the whole transaction rolled back, not just the insert",
        gold_of(&mut db.begin(), ada)?
    );
    assert_eq!(gold_of(&mut db.begin(), ada)?, before);

    let mut tx = db.begin();
    for hero in [ada, bram, cleo] {
        let name = name_of(&mut tx, hero)?;
        let carried = carried(&mut tx, hero)?;
        let items = inventory(&mut tx, hero)?;
        let names: Vec<_> = items.iter().map(|(_, n, _)| n.as_str()).collect();
        println!("  {name:<5} carries {carried:>3} — {}", names.join(", "));
    }
    drop(tx);

    // -----------------------------------------------------------------------
    act(2, "Two heroes reach for the same greatsword");

    // Both begin, both see the sword unclaimed. The first to write takes it;
    // the second is told immediately rather than being made to wait.
    let mut ada_grabs = db.begin_with::<Snapshot>();
    let mut bram_grabs = db.begin_with::<Snapshot>();

    ada_grabs.update::<Item>(&20, |i| i.owner = ada)?;

    match bram_grabs.update::<Item>(&20, |i| i.owner = bram) {
        Err(e @ Error::WriteConflict { .. }) => {
            println!("  Bram: {e}");
            println!(
                "  ...retriable: {} — he can try for something else",
                e.is_retriable()
            );
        }
        other => unreachable!("expected a conflict, got {other:?}"),
    }

    // A failed write does not end the transaction. Bram takes the cloak instead.
    bram_grabs.update::<Item>(&21, |i| i.owner = bram)?;
    ada_grabs.commit()?;
    bram_grabs.commit()?;
    println!("  Ada takes the greatsword, Bram takes the cloak. Nobody blocked.");

    // -----------------------------------------------------------------------
    act(3, "Cleo overloads herself — write skew, and the fix");

    // Cleo can carry 60 and is carrying 2. Either 35kg helm fits; both do not.
    // Each transaction reads her inventory, checks the total, and picks up a
    // *different* item — so the two writes never touch the same record, and
    // first-updater-wins has nothing to catch.
    db.transaction(|tx| {
        tx.update::<Item>(&22, |i| i.owner = GROUND)?;
        tx.insert(Item {
            id: 23,
            owner: GROUND,
            name: "Steel Helm".into(),
            weight: 35,
            value: 70,
            kind: Kind::Weapon,
            flavour: HashMap::new(),
        })
    })?;

    let pick_up = |item: u64| {
        move |tx: &mut mvcc::Transaction<'_, Serializable>| -> Result<bool> {
            let hero = tx.get::<Hero>(&cleo)?.expect("Cleo exists");
            let strength = hero.strength;
            let load = carried(tx, cleo)?;
            let weight = tx.get::<Item>(&item)?.map(|i| i.weight).unwrap_or(0);
            if load + weight > strength {
                return Ok(false);
            }
            tx.update::<Item>(&item, |i| i.owner = cleo)?;
            Ok(true)
        }
    };

    // Under Snapshot, both checks pass against a stale inventory.
    {
        let mut t1 = db.begin_with::<Snapshot>();
        let mut t2 = db.begin_with::<Snapshot>();
        let (load1, load2) = (carried(&mut t1, cleo)?, carried(&mut t2, cleo)?);
        println!(
            "  Snapshot:     both transactions see {load1}/{load2} carried, both think a 35kg helm fits"
        );
        t1.update::<Item>(&22, |i| i.owner = cleo)?;
        t2.update::<Item>(&23, |i| i.owner = cleo)?;
        t1.commit()?;
        t2.commit()?;
        let over = carried(&mut db.begin(), cleo)?;
        println!(
            "  ✗ Cleo now carries {over} of a possible 60. Two legal transactions, one broken rule."
        );
    }

    // Put it back and try again at Serializable.
    db.transaction(|tx| {
        tx.update::<Item>(&22, |i| i.owner = GROUND)?;
        tx.update::<Item>(&23, |i| i.owner = GROUND).map(|_| ())
    })?;

    {
        let mut t1 = db.begin_with::<Serializable>();
        let mut t2 = db.begin_with::<Serializable>();
        // Both read the inventory, as before.
        let _ = carried(&mut t1, cleo)?;
        let _ = carried(&mut t2, cleo)?;
        t1.update::<Item>(&22, |i| i.owner = cleo)?;
        t2.update::<Item>(&23, |i| i.owner = cleo)?;
        t1.commit()?;
        match t2.commit() {
            Err(e @ Error::SerializationFailure) => println!("  Serializable: second pickup {e}"),
            other => unreachable!("expected a serialization failure, got {other:?}"),
        }
        println!(
            "  ✓ Cleo carries {} — the second pickup read an inventory the first changed.",
            carried(&mut db.begin(), cleo)?
        );
    }

    // In real code you would not hand-roll that. `transaction_with` retries,
    // and the retry re-reads the inventory and correctly declines.
    let took_it = db.transaction_with::<Serializable, _, _>(pick_up(23))?;
    println!("  ...and on retry the second helm is refused on its merits: picked_up = {took_it}");

    // -----------------------------------------------------------------------
    act(4, "The party fills up — a phantom, not a conflict");

    db.transaction(|tx| {
        tx.insert(PartyMember {
            hero: ada,
            party: PARTY,
            role: "Vanguard".into(),
        })?;
        tx.insert(PartyMember {
            hero: bram,
            party: PARTY,
            role: "Shield".into(),
        })?;
        tx.insert(PartyMember {
            hero: cleo,
            party: PARTY,
            role: "Scout".into(),
        })
    })?;

    // Two newcomers apply at once, with three of four slots taken. Neither
    // writes a row the other wrote — they insert *different* rows — so there is
    // no write conflict to detect. What they collide on is a row that did not
    // exist when either of them counted.
    let (dara, finn) = (4u64, 5u64);
    db.transaction(|tx| {
        tx.insert(Hero {
            id: dara,
            name: "Dara".into(),
            gold: 40,
            strength: 80,
        })?;
        tx.insert(Hero {
            id: finn,
            name: "Finn".into(),
            gold: 40,
            strength: 80,
        })
    })?;

    let mut d = db.begin_with::<Serializable>();
    let mut f = db.begin_with::<Serializable>();
    println!(
        "  Dara counts {} members, Finn counts {} — both see a free slot",
        party_size(&mut d)?,
        party_size(&mut f)?
    );
    d.insert(PartyMember {
        hero: dara,
        party: PARTY,
        role: "Healer".into(),
    })?;
    f.insert(PartyMember {
        hero: finn,
        party: PARTY,
        role: "Healer".into(),
    })?;
    d.commit()?;
    match f.commit() {
        Err(e @ Error::SerializationFailure) => println!("  Finn: {e}"),
        other => unreachable!("expected a serialization failure, got {other:?}"),
    }
    let mut tx = db.begin();
    let mut roster: Vec<_> = tx
        .scan_where::<PartyMember, _>(|m| m.party == PARTY)?
        .iter()
        .map(|m| (m.hero, m.role.clone()))
        .collect();
    drop(tx);
    roster.sort();
    let size = roster.len();
    let mut tx = db.begin();
    let listed: Vec<String> = roster
        .iter()
        .map(|(h, role)| Ok(format!("{} the {role}", name_of(&mut tx, *h)?)))
        .collect::<Result<_>>()?;
    drop(tx);
    println!("  party: {}", listed.join(", "));
    println!("  ✓ {size}/{PARTY_LIMIT} filled. Finn's insert became a phantom in Dara's count.");
    println!("    (`scan_where` hands the engine the predicate, so it can re-check it at commit.)");
    assert!(size <= PARTY_LIMIT);

    // -----------------------------------------------------------------------
    act(5, "A trade — two records, one atomic step");

    let before = total_gold(&db)?;

    let trade = |seller: u64, buyer: u64, item: u64, price: i64| {
        let db = Arc::clone(&db);
        move || -> Result<bool> {
            db.transaction_with::<Serializable, _, _>(|tx| {
                let buyer_gold = gold_of(tx, buyer)?;
                if buyer_gold < price {
                    return Ok(false);
                }
                let owned_by_seller = tx
                    .get::<Item>(&item)?
                    .map(|i| i.owner == seller)
                    .unwrap_or(false);
                if !owned_by_seller {
                    return Ok(false);
                }
                tx.update::<Hero>(&buyer, |h| h.gold -= price)?;
                tx.update::<Hero>(&seller, |h| h.gold += price)?;
                tx.update::<Item>(&item, |i| i.owner = buyer)?;
                Ok(true)
            })
        }
    };

    let sold = trade(bram, cleo, 21, 150)()?;
    println!("  Bram sells the Elven Cloak to Cleo for 150g: {sold}");
    let mut tx = db.begin();
    let cloak_owner = tx.get::<Item>(&21)?.expect("cloak").owner;
    println!(
        "  Bram {}g, Cleo {}g, cloak owner is {}",
        gold_of(&mut tx, bram)?,
        gold_of(&mut tx, cleo)?,
        name_of(&mut tx, cloak_owner)?
    );
    drop(tx);
    assert_eq!(total_gold(&db)?, before, "gold must be conserved");
    println!("  ✓ total gold unchanged at {before} — no step of that was separately visible");

    // -----------------------------------------------------------------------
    act(6, "The chronicler reads while the world moves");

    // A long report scanning the whole world while combat rewrites it. It never
    // blocks a writer, and its view never shifts underneath it.
    let mut chronicler = db.begin_with::<Snapshot>();
    let opening = total_gold(&db)?;
    let seen_first = chronicler.scan::<Hero>()?.len();

    for _ in 0..50 {
        db.transaction(|tx| {
            tx.update::<Hero>(&ada, |h| h.gold += 1)?;
            tx.update::<Hero>(&bram, |h| h.gold -= 1)?;
            Ok(())
        })?;
    }
    db.transaction(|tx| {
        tx.insert(Hero {
            id: 6,
            name: "Mira".into(),
            gold: 0,
            strength: 50,
        })
    })?;

    let chron_gold: i64 = chronicler.scan::<Hero>()?.iter().map(|h| h.gold).sum();
    let seen_after = chronicler.scan::<Hero>()?.len();
    let in_world = db.begin().scan::<Hero>()?.len();
    println!("  chronicler saw {seen_first} heroes at the start and still sees {seen_after};");
    println!("  the world now holds {in_world}, after 50 commits and one new arrival");
    println!(
        "  chronicler totals {chron_gold}g; the world now totals {}g",
        total_gold(&db)?
    );
    assert_eq!(chron_gold, opening, "a snapshot must not move");
    chronicler.commit()?;
    println!("  ✓ a reader that ran across 51 commits saw exactly one consistent world");

    // -----------------------------------------------------------------------
    act(7, "The raid — many adventurers at once");

    let stop = Arc::new(AtomicBool::new(false));
    let trades = Arc::new(AtomicU64::new(0));
    let retries = Arc::new(AtomicU64::new(0));
    let audits = Arc::new(AtomicU64::new(0));
    let opening = total_gold(&db)?;
    let heroes = [ada, bram, cleo, dara, finn];

    // An auditor that must never catch a trade half-applied.
    let auditor = {
        let (db, stop, audits) = (Arc::clone(&db), Arc::clone(&stop), Arc::clone(&audits));
        thread::spawn(move || -> Result<()> {
            while !stop.load(Ordering::Relaxed) {
                let mut tx = db.begin();
                let total: i64 = tx.scan::<Hero>()?.iter().map(|h| h.gold).sum();
                assert_eq!(total, opening, "auditor saw a half-finished trade");
                audits.fetch_add(1, Ordering::Relaxed);
            }
            Ok(())
        })
    };

    let start = Instant::now();
    let raiders: Vec<_> = (0..4)
        .map(|t| {
            let (db, trades, retries) =
                (Arc::clone(&db), Arc::clone(&trades), Arc::clone(&retries));
            thread::spawn(move || -> Result<()> {
                let mut seed = 0x9e37_79b9_7f4a_7c15u64 ^ (t + 1);
                for _ in 0..400 {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let from = heroes[(seed % heroes.len() as u64) as usize];
                    let to = heroes[((seed >> 8) % heroes.len() as u64) as usize];
                    if from == to {
                        continue;
                    }

                    let mut attempts = 0u64;
                    // Serializable, because the payment depends on a balance we
                    // read — the same shape as Cleo's carry limit in Act 3.
                    db.transaction_with::<Serializable, _, _>(|tx| {
                        attempts += 1;
                        let purse = gold_of(tx, from)?;
                        if purse < 5 {
                            return Ok(());
                        }
                        tx.update::<Hero>(&from, |h| h.gold -= 5)?;
                        tx.update::<Hero>(&to, |h| h.gold += 5)?;
                        Ok(())
                    })?;
                    trades.fetch_add(1, Ordering::Relaxed);
                    retries.fetch_add(attempts - 1, Ordering::Relaxed);
                }
                Ok(())
            })
        })
        .collect();

    for r in raiders {
        r.join().expect("raider panicked")?;
    }
    let elapsed = start.elapsed();
    stop.store(true, Ordering::Relaxed);
    auditor.join().expect("auditor panicked")?;

    let done = trades.load(Ordering::Relaxed);
    println!("  {done} trades across 4 threads in {elapsed:.2?}");
    println!(
        "  {} audit sweeps completed alongside them, none blocked, none torn",
        audits.load(Ordering::Relaxed)
    );
    println!(
        "  {} retries ({:.2} per trade) — contention on 5 heroes, handled by the engine",
        retries.load(Ordering::Relaxed),
        retries.load(Ordering::Relaxed) as f64 / done.max(1) as f64
    );
    assert_eq!(total_gold(&db)?, opening, "gold was created or destroyed");
    println!("  ✓ total gold still {opening}");

    // -----------------------------------------------------------------------
    act(8, "Closing the ledger");

    let mut tx = db.begin();
    let mut roster: Vec<_> = tx
        .scan::<Hero>()?
        .iter()
        .map(|h| (h.id, h.name.clone(), h.gold))
        .collect();
    roster.sort_by_key(|(id, _, _)| *id);
    drop(tx);

    println!(
        "  {:<6} {:>6} {:>5} {:>7}  inventory",
        "hero", "gold", "load", "worth"
    );
    let mut tx = db.begin();
    for (id, name, gold) in &roster {
        let held = tx.scan_index::<Item, _, _>("owner", *id..=*id)?;
        let load: i32 = held.iter().map(|i| i.weight).sum();
        let worth: i64 = held.iter().map(|i| i.value).sum();
        let names: Vec<String> = held
            .iter()
            .map(|i| match i.kind {
                Kind::Weapon => format!("[wpn] {}", i.name),
                Kind::Potion => format!("[pot] {}", i.name),
                Kind::Trinket => format!("[trk] {}", i.name),
            })
            .collect();
        let names = if names.is_empty() {
            "—".to_string()
        } else {
            names.join(", ")
        };
        println!("  {name:<6} {gold:>5}g {load:>4}kg {worth:>6}g  {names}");
    }

    // Potions anywhere in the world, found by predicate rather than by key —
    // and `kind` is an ordinary Rust enum the engine knows nothing about.
    let potions = tx.scan_where::<Item, _>(|i| i.kind == Kind::Potion)?.len();
    let sword = tx.get::<Item>(&20)?.expect("greatsword");
    let enchantment = sword
        .flavour
        .get("enchantment")
        .cloned()
        .unwrap_or_default();
    let forged_by = sword.flavour.get("forged_by").cloned().unwrap_or_default();
    drop(tx);
    // The rule Act 3 was about, checked rather than asserted in prose.
    let mut tx = db.begin();
    for (id, name, _) in &roster {
        let strength = tx.get::<Hero>(id)?.map(|h| h.strength).unwrap_or(0);
        let load = carried(&mut tx, *id)?;
        assert!(
            load <= strength,
            "{name} is over their carry limit: {load} > {strength}"
        );
    }
    drop(tx);

    println!("\n  potions in the world: {potions}");
    println!("  the greatsword is {enchantment}-enchanted, forged by {forged_by}");

    // Nothing is reclaimed while a transaction that could still reach it is
    // alive. This is the number to watch in a long-lived process: a forgotten
    // transaction pins it and version chains grow without limit.
    let stats = db.stats();
    println!(
        "\n  gc watermark {:?}, {} transactions still live",
        stats.watermark, stats.active_transactions
    );
    println!("  (a watermark that stops moving while writes continue is the leak to look for)");

    Ok(())
}
