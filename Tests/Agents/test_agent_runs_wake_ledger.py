"""PR3a-2 Task 5: the durable per-run wake-delivery ledger.

Auto-wake needs exactly-once delivery that survives screen teardown AND app
restart. The conversation-level ``FLEET_UNSEEN`` mark cannot carry per-run
identity (one drain can mix children settled minutes apart, so no timestamp
rule against the mark recovers which runs a wake already carried), and the
in-memory pending state dies with the process. The ledger is therefore ON
the run row itself: ``agent_runs.wake_delivered_at``, NULL while
undelivered, stamped only after a wake turn was actually accepted.

Under test:

1. ``undelivered_wake_runs`` -- the durable definition of "owed to the
   supervisor": terminal sub-agent runs (never ``superseded``) that
   settled no earlier than their parent's own terminal write (the
   survivor discriminator, expressed in timestamps because only survivors
   settle after their turn) and carry no delivery stamp;
2. ``mark_wake_delivered`` -- first-writer-wins stamping that never moves
   an existing timestamp and never runs at compose/schedule time;
3. the idempotent-ALTER migration -- an old file without the column gains
   it on reopen, exactly like ``assistant_message_id`` before it.
"""
from __future__ import annotations

import sqlite3

import pytest

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


@pytest.fixture
def db(tmp_path):
    handle = AgentRunsDB(tmp_path / "runs.db", client_id="wake-test")
    yield handle
    handle.close()


def _spawned_turn(db, conversation_id="conv-wake"):
    """One finished primary turn that spawned one child, child still running."""
    parent_id = db.create_run(conversation_id=conversation_id, agent_kind="primary")
    child_id = db.create_run(
        conversation_id=conversation_id,
        agent_kind="subagent",
        task="long job",
        parent_run_id=parent_id,
    )
    return parent_id, child_id


def test_a_survivor_is_owed_and_a_within_turn_child_is_not(db):
    """The discriminator: a child terminal BEFORE its parent's terminal
    write was collected in-turn (the turn's own news); one terminal AFTER
    is a survivor whose result reached nobody -- exactly what a wake owes."""
    parent_id, within_turn_id = _spawned_turn(db)
    survivor_id = db.create_run(
        conversation_id="conv-wake",
        agent_kind="subagent",
        task="slow job",
        parent_run_id=parent_id,
    )
    # Settle order IS the discriminator: within-turn child, then the
    # parent's turn ends, then the survivor settles.
    db.set_status(within_turn_id, "done", result="quick answer")
    db.set_status(parent_id, "done", result="turn final")
    db.set_status(survivor_id, "done", result="slow answer")

    owed = db.undelivered_wake_runs("conv-wake")

    assert [run["id"] for run in owed] == [survivor_id]
    assert owed[0]["result"] == "slow answer"
    assert owed[0]["wake_delivered_at"] is None


def test_a_child_of_a_still_running_turn_is_not_owed_yet(db):
    """While the parent turn is live its children are in-turn collectible;
    nothing is owed to a wake until the parent itself is terminal."""
    parent_id, child_id = _spawned_turn(db)
    db.set_status(child_id, "done", result="early answer")

    assert db.undelivered_wake_runs("conv-wake") == []


def test_error_and_cancelled_survivors_are_owed_but_superseded_is_not(db):
    """Honest statuses ride the wake (the supervisor may act on a failure);
    a superseded row is retracted work and must never be announced."""
    parent_id, errored = _spawned_turn(db)
    cancelled = db.create_run(
        conversation_id="conv-wake", agent_kind="subagent", parent_run_id=parent_id
    )
    superseded = db.create_run(
        conversation_id="conv-wake", agent_kind="subagent", parent_run_id=parent_id
    )
    db.set_status(parent_id, "done")
    db.set_status(errored, "error", result="boom")
    db.set_status(cancelled, "cancelled")
    db.set_status(superseded, "superseded")

    owed = {run["id"]: run["status"] for run in db.undelivered_wake_runs("conv-wake")}

    assert owed == {errored: "error", cancelled: "cancelled"}


def test_parented_local_command_can_never_enter_model_wake_results(db):
    parent_id, child_id = _spawned_turn(db)
    local_id = db.create_run(
        conversation_id="conv-wake",
        agent_kind="local_command",
        task="Local command",
        parent_run_id=parent_id,
    )
    db.set_status(parent_id, "done")
    db.set_status(child_id, "done", result="sub-agent answer")
    db.set_status(local_id, "done", result="LOCAL_COMMAND_SECRET")

    owed = db.undelivered_wake_runs("conv-wake")

    assert [run["id"] for run in owed] == [child_id]
    assert local_id not in {run["id"] for run in owed}
    assert "LOCAL_COMMAND_SECRET" not in repr(owed)


def test_owed_runs_come_back_in_settle_order(db):
    parent_id, first = _spawned_turn(db)
    second = db.create_run(
        conversation_id="conv-wake", agent_kind="subagent", parent_run_id=parent_id
    )
    db.set_status(parent_id, "done")
    db.set_status(first, "done", result="first answer")
    db.set_status(second, "done", result="second answer")

    assert [run["id"] for run in db.undelivered_wake_runs("conv-wake")] == [
        first,
        second,
    ]


def test_marking_delivered_removes_a_run_from_the_owed_set_exactly_once(db):
    """The exactly-once ledger: a stamped run is never owed again -- across
    a FRESH handle on the same file (restart shape) -- and a duplicate
    stamp neither counts nor moves the recorded delivery instant."""
    parent_id, within_turn_id = _spawned_turn(db)
    survivor_id = db.create_run(
        conversation_id="conv-wake", agent_kind="subagent", parent_run_id=parent_id
    )
    db.set_status(within_turn_id, "done")
    db.set_status(parent_id, "done")
    db.set_status(survivor_id, "done", result="slow answer")

    assert db.mark_wake_delivered([survivor_id]) == 1
    stamped_at = db.get_run(survivor_id)["wake_delivered_at"]
    assert stamped_at, "delivery must record its instant on the row"
    assert db.undelivered_wake_runs("conv-wake") == []

    assert db.mark_wake_delivered([survivor_id]) == 0, (
        "a second stamp must be a no-op, not a re-delivery record"
    )
    assert db.get_run(survivor_id)["wake_delivered_at"] == stamped_at

    fresh = AgentRunsDB(db.db_path_str, client_id="wake-test-2")
    try:
        assert fresh.undelivered_wake_runs("conv-wake") == [], (
            "the delivered ledger must be durable: a fresh handle on the "
            "same file (the restart shape) may not re-owe a delivered run"
        )
    finally:
        fresh.close()


def test_marking_delivered_does_not_disturb_the_runs_lifecycle_timestamp(db):
    """``updated_at`` is the survivor comparison's input; the ledger stamp
    must never rewrite the run's own lifecycle history."""
    parent_id, survivor_id = _spawned_turn(db)
    db.set_status(parent_id, "done")
    db.set_status(survivor_id, "done", result="answer")
    settled_at = db.get_run(survivor_id)["updated_at"]

    db.mark_wake_delivered([survivor_id])

    assert db.get_run(survivor_id)["updated_at"] == settled_at


def test_mark_wake_delivered_tolerates_empty_and_blank_ids(db):
    assert db.mark_wake_delivered([]) == 0
    assert db.mark_wake_delivered(["", None]) == 0


def test_an_old_file_without_the_ledger_column_gains_it_on_reopen(db, tmp_path):
    """The idempotent-ALTER migration, proven on a genuinely old-shaped
    file: strip the column with raw SQL, reopen through AgentRunsDB, and
    the ledger is back -- the ``assistant_message_id`` precedent."""
    parent_id, survivor_id = _spawned_turn(db)
    db.set_status(parent_id, "done")
    db.set_status(survivor_id, "done", result="answer")
    db.close()

    raw = sqlite3.connect(db.db_path_str)
    try:
        raw.execute("ALTER TABLE agent_runs DROP COLUMN wake_delivered_at")
        raw.commit()
        columns = {
            row[1] for row in raw.execute("PRAGMA table_info(agent_runs)").fetchall()
        }
        assert "wake_delivered_at" not in columns, "precondition: old shape"
    finally:
        raw.close()

    reopened = AgentRunsDB(db.db_path_str, client_id="wake-test-3")
    try:
        assert [run["id"] for run in reopened.undelivered_wake_runs("conv-wake")] == [
            survivor_id
        ]
        assert reopened.mark_wake_delivered([survivor_id]) == 1
    finally:
        reopened.close()
