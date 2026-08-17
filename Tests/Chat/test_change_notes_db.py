"""TASK-16800 Task 1: `change_notes` persistence in AgentRuns_DB.

Persistence foundation for the turn-file-card annotate/feedback loop
(spec `Docs/superpowers/specs/2026-08-17-console-turn-file-annotate-design.md`
§1). Tests run against a real FILE-BACKED ``AgentRunsDB`` — never
``:memory:`` (thread-affinity trap, V1 lesson carried into this spec).
"""
from __future__ import annotations

import sqlite3

import pytest

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


@pytest.fixture()
def db(tmp_path):
    return AgentRunsDB(tmp_path / "runs.db", client_id="t")


def _make_run(db: AgentRunsDB, conversation_id: str = "conv1") -> str:
    return db.create_run(conversation_id=conversation_id, agent_kind="primary")


def test_add_change_note_returns_id_and_round_trips_all_fields(db):
    run_id = _make_run(db)
    note_id = db.add_change_note(
        run_id=run_id,
        root="/repo",
        path="a.py",
        hunk_index=0,
        hunk_header="@@ -1,4 +1,6 @@",
        hunk_excerpt="-old\n+new",
        note="use the cached value here",
    )
    assert isinstance(note_id, int)

    notes = db.notes_for_run(run_id)
    assert len(notes) == 1
    row = notes[0]
    assert row["id"] == note_id
    assert row["run_id"] == run_id
    assert row["root"] == "/repo"
    assert row["path"] == "a.py"
    assert row["hunk_index"] == 0
    assert row["hunk_header"] == "@@ -1,4 +1,6 @@"
    assert row["hunk_excerpt"] == "-old\n+new"
    assert row["note"] == "use the cached value here"
    assert row["created_at"]
    assert row["delivered_at"] is None


def test_notes_for_run_oldest_first(db):
    run_id = _make_run(db)
    first = db.add_change_note(
        run_id=run_id, root="/r", path="a.py", hunk_index=0,
        hunk_header="@@ -1,1 +1,1 @@", hunk_excerpt="x", note="first",
    )
    second = db.add_change_note(
        run_id=run_id, root="/r", path="b.py", hunk_index=1,
        hunk_header="@@ -2,1 +2,1 @@", hunk_excerpt="y", note="second",
    )
    ids = [row["id"] for row in db.notes_for_run(run_id)]
    assert ids == [first, second]


def test_pending_notes_for_conversation_joins_both_runs_oldest_first(db):
    run_a = _make_run(db, conversation_id="conv1")
    run_b = _make_run(db, conversation_id="conv1")
    other_conv_run = _make_run(db, conversation_id="conv2")

    note_a = db.add_change_note(
        run_id=run_a, root="/r", path="a.py", hunk_index=0,
        hunk_header="@@ -1,1 +1,1 @@", hunk_excerpt="x", note="note-a",
    )
    note_b = db.add_change_note(
        run_id=run_b, root="/r", path="b.py", hunk_index=0,
        hunk_header="@@ -2,1 +2,1 @@", hunk_excerpt="y", note="note-b",
    )
    db.add_change_note(
        run_id=other_conv_run, root="/r", path="c.py", hunk_index=0,
        hunk_header="@@ -3,1 +3,1 @@", hunk_excerpt="z", note="note-other-conv",
    )

    pending = db.pending_notes_for_conversation("conv1")
    assert [row["id"] for row in pending] == [note_a, note_b]
    assert {row["run_id"] for row in pending} == {run_a, run_b}


def test_pending_notes_for_conversation_excludes_delivered(db):
    run_id = _make_run(db)
    note_id = db.add_change_note(
        run_id=run_id, root="/r", path="a.py", hunk_index=0,
        hunk_header="@@ -1,1 +1,1 @@", hunk_excerpt="x", note="note",
    )
    db.mark_notes_delivered([note_id])
    assert db.pending_notes_for_conversation("conv1") == []


def test_mark_notes_delivered_stamps_only_given_ids_and_sets_timestamp(db):
    run_id = _make_run(db)
    note_1 = db.add_change_note(
        run_id=run_id, root="/r", path="a.py", hunk_index=0,
        hunk_header="@@ -1,1 +1,1 @@", hunk_excerpt="x", note="one",
    )
    note_2 = db.add_change_note(
        run_id=run_id, root="/r", path="b.py", hunk_index=1,
        hunk_header="@@ -2,1 +2,1 @@", hunk_excerpt="y", note="two",
    )

    db.mark_notes_delivered([note_1])

    notes = {row["id"]: row for row in db.notes_for_run(run_id)}
    assert notes[note_1]["delivered_at"] is not None
    assert notes[note_2]["delivered_at"] is None


def test_mark_notes_delivered_mid_run_race_leaves_later_note_pending(db):
    """Spec §4: the attach step captures the exact pending-id list at
    attach time; a note added AFTER that capture — while the run is still
    in flight — must not be swept up by stamping the captured list, even
    though it is technically still "pending" when the stamp runs. This is
    load-bearing for Task 5's delivery seam.
    """
    run_id = _make_run(db)
    note_1 = db.add_change_note(
        run_id=run_id, root="/r", path="a.py", hunk_index=0,
        hunk_header="@@ -1,1 +1,1 @@", hunk_excerpt="x", note="captured-before-run",
    )

    # Attach seam captures the pending id list at this instant.
    captured_ids = [row["id"] for row in db.pending_notes_for_conversation("conv1")]
    assert captured_ids == [note_1]

    # A note lands on an older turn's card while the run is in flight.
    note_2 = db.add_change_note(
        run_id=run_id, root="/r", path="b.py", hunk_index=1,
        hunk_header="@@ -2,1 +2,1 @@", hunk_excerpt="y", note="mid-run-race",
    )

    # Completion stamps exactly the captured list, not "all pending now".
    db.mark_notes_delivered(captured_ids)

    notes = {row["id"]: row for row in db.notes_for_run(run_id)}
    assert notes[note_1]["delivered_at"] is not None
    assert notes[note_2]["delivered_at"] is None

    still_pending = [row["id"] for row in db.pending_notes_for_conversation("conv1")]
    assert still_pending == [note_2]


def test_delete_change_note_deletes_pending_note(db):
    run_id = _make_run(db)
    note_id = db.add_change_note(
        run_id=run_id, root="/r", path="a.py", hunk_index=0,
        hunk_header="@@ -1,1 +1,1 @@", hunk_excerpt="x", note="deleteme",
    )
    assert db.delete_change_note(note_id) is True
    assert db.notes_for_run(run_id) == []


def test_delete_change_note_returns_false_for_delivered(db):
    run_id = _make_run(db)
    note_id = db.add_change_note(
        run_id=run_id, root="/r", path="a.py", hunk_index=0,
        hunk_header="@@ -1,1 +1,1 @@", hunk_excerpt="x", note="delivered",
    )
    db.mark_notes_delivered([note_id])
    assert db.delete_change_note(note_id) is False
    # Delivered note survives -- it is record, not deletable.
    assert len(db.notes_for_run(run_id)) == 1


def test_delete_change_note_returns_false_for_missing(db):
    assert db.delete_change_note(999999) is False


def test_mark_notes_delivered_empty_list_is_noop(db):
    run_id = _make_run(db)
    note_id = db.add_change_note(
        run_id=run_id, root="/r", path="a.py", hunk_index=0,
        hunk_header="@@ -1,1 +1,1 @@", hunk_excerpt="x", note="untouched",
    )
    db.mark_notes_delivered([])
    assert db.notes_for_run(run_id)[0]["delivered_at"] is None


# --- Migration: change_notes DDL + audit version 8 --------------------


def test_migration_creates_table_and_appends_audit_version_8(tmp_path):
    db_path = tmp_path / "migrate.db"

    # Open once with the current code -- this already creates change_notes
    # and appends schema_version 8, since CREATE TABLE IF NOT EXISTS /
    # INSERT OR IGNORE run unconditionally on every open (this DB's own
    # migration mechanism -- there is no separate "old" binary to run
    # here). Simulate a pre-migration file by tearing the table/version
    # row back out via a raw sqlite3 connection, then reopening through
    # AgentRunsDB and asserting the migration reapplies.
    AgentRunsDB(db_path, client_id="t").close()

    raw = sqlite3.connect(str(db_path))
    try:
        raw.execute("DROP TABLE change_notes")
        raw.execute("DELETE FROM schema_version WHERE version = 8")
        raw.commit()
    finally:
        raw.close()

    # Confirm the simulated pre-migration state actually took.
    raw = sqlite3.connect(str(db_path))
    try:
        tables = {
            row[0]
            for row in raw.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "change_notes" not in tables
        versions = {
            row[0] for row in raw.execute("SELECT version FROM schema_version").fetchall()
        }
        assert 8 not in versions
    finally:
        raw.close()

    reopened = AgentRunsDB(db_path, client_id="t")
    try:
        raw = sqlite3.connect(str(db_path))
        try:
            tables = {
                row[0]
                for row in raw.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "change_notes" in tables
            versions = {
                row[0]
                for row in raw.execute("SELECT version FROM schema_version").fetchall()
            }
            assert 8 in versions
        finally:
            raw.close()

        # The reopened instance's own API must work against the
        # freshly-recreated table.
        run_id = reopened.create_run(conversation_id="c", agent_kind="primary")
        note_id = reopened.add_change_note(
            run_id=run_id, root="/r", path="a.py", hunk_index=0,
            hunk_header="@@ -1,1 +1,1 @@", hunk_excerpt="x", note="post-migration",
        )
        assert reopened.notes_for_run(run_id)[0]["id"] == note_id
    finally:
        reopened.close()


def test_migration_reopening_twice_is_idempotent(tmp_path):
    db_path = tmp_path / "double_open.db"

    first = AgentRunsDB(db_path, client_id="t")
    first.close()
    second = AgentRunsDB(db_path, client_id="t")
    try:
        raw = sqlite3.connect(str(db_path))
        try:
            version_8_rows = raw.execute(
                "SELECT COUNT(*) FROM schema_version WHERE version = 8"
            ).fetchone()[0]
            assert version_8_rows == 1  # INSERT OR IGNORE -- never duplicated

            tables = [
                row[0]
                for row in raw.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name = 'change_notes'"
                ).fetchall()
            ]
            assert tables == ["change_notes"]
        finally:
            raw.close()

        # And the API still works normally on the third open.
        run_id = second.create_run(conversation_id="c", agent_kind="primary")
        note_id = second.add_change_note(
            run_id=run_id, root="/r", path="a.py", hunk_index=0,
            hunk_header="@@ -1,1 +1,1 @@", hunk_excerpt="x", note="third-open",
        )
        assert note_id
    finally:
        second.close()
