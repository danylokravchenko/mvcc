//! Runs the shipped examples end to end.
//!
//! Examples are documentation that executes, and documentation rots. `cargo
//! test` builds them but never runs them, so an example can compile happily
//! while asserting something that is no longer true. These invoke them for
//! real, and every example ends in assertions about the world it built — the
//! game example checks that gold is conserved, that no hero exceeds their carry
//! limit, and that the party never overfills.

use std::process::Command;

fn run(example: &str) {
    let output = Command::new(env!("CARGO"))
        .args(["run", "--quiet", "--release", "--example", example])
        .output()
        .unwrap_or_else(|e| panic!("could not launch `{example}`: {e}"));

    assert!(
        output.status.success(),
        "example `{example}` failed ({}):\n{}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn game_example_runs() {
    run("game");
}

#[test]
fn basic_example_runs() {
    run("basic");
}

#[test]
fn isolation_example_runs() {
    run("isolation");
}
