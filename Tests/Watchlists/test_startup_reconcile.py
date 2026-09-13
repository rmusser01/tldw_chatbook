"""Tests for the startup sweep that un-wedges interrupted work (task-19561).

A real file-backed `SubscriptionsDB` throughout: the sweep's whole point is
what it does to rows an earlier *process* left behind, and a mock cannot
disagree with the schema.

The sweep is scoped by a row-id boundary captured before this process can
write anything (Qodo review of PR #1972). Every test here therefore captures
the boundary at the point the *production* caller does -- after the previous
process's rows exist, before this process creates any -- and the boundary
tests at the bottom pin the property that makes the scoping sound.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.startup_reconcile import (
    INTERRUPTED_RUN_ERROR,
    PriorProcessBoundary,
    capture_prior_process_boundary,
    fail_interrupted_watchlist_runs,
    reconcile_interrupted_subscription_work,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def db(tmp_path) -> SubscriptionsDB:
    # File-backed, not `:memory:`: `SubscriptionsDB` connections are
    # thread-local, and the production caller reaches this through
    # `asyncio.to_thread`.
    return SubscriptionsDB(tmp_path / "subs.db", "test")


def _source(db: SubscriptionsDB, name: str = "src") -> int:
    return db.add_subscription(
        name=name, type="rss", source=f"https://{name}.example/feed.xml"
    )


def _run(db: SubscriptionsDB, source_id: int, status: str) -> int:
    with db.transaction() as conn:
        return conn.execute(
            "INSERT INTO local_watchlist_runs "
            "(source_id, status, created_at, updated_at) "
            "VALUES (?, ?, datetime('now'), datetime('now'))",
            (source_id, status),
        ).lastrowid


def _run_row(db: SubscriptionsDB, run_id: int) -> dict:
    with db.transaction() as conn:
        row = conn.execute(
            "SELECT status, error_msg, finished_at FROM local_watchlist_runs "
            "WHERE id = ?",
            (run_id,),
        ).fetchone()
    return dict(row)


def _watchlist(db: SubscriptionsDB, name: str = "wl") -> int:
    with db.transaction() as conn:
        return conn.execute(
            "INSERT INTO watchlists (name, created_at, updated_at) "
            "VALUES (?, datetime('now'), datetime('now'))",
            (name,),
        ).lastrowid


def _briefing(db: SubscriptionsDB, watchlist_id: int, status: str) -> int:
    with db.transaction() as conn:
        return conn.execute(
            "INSERT INTO briefings (watchlist_id, status) VALUES (?, ?)",
            (watchlist_id, status),
        ).lastrowid


def _script(db: SubscriptionsDB, briefing_id: int, status: str) -> int:
    with db.transaction() as conn:
        return conn.execute(
            "INSERT INTO briefing_scripts "
            "(briefing_id, preset_name, roster_snapshot_json, status) "
            "VALUES (?, 'preset', '[]', ?)",
            (briefing_id, status),
        ).lastrowid


def _audio(db: SubscriptionsDB, script_id: int, status: str) -> int:
    with db.transaction() as conn:
        return conn.execute(
            "INSERT INTO briefing_audio (script_id, voice_snapshot_json, status) "
            "VALUES (?, '{}', ?)",
            (script_id, status),
        ).lastrowid


def _briefing_status(db: SubscriptionsDB, briefing_id: int) -> str:
    with db.transaction() as conn:
        return conn.execute(
            "SELECT status FROM briefings WHERE id = ?", (briefing_id,)
        ).fetchone()["status"]


def _status(db: SubscriptionsDB, table: str, row_id: int) -> str:
    with db.transaction() as conn:
        return conn.execute(
            f"SELECT status FROM {table} WHERE id = ?", (row_id,)
        ).fetchone()["status"]


@pytest.mark.parametrize("stranded_status", ["running", "queued"])
def test_a_stranded_run_is_failed_with_a_distinguishable_reason(db, stranded_status):
    """`queued` matters as much as `running`: nothing will ever dispatch it."""
    source_id = _source(db)
    run_id = _run(db, source_id, stranded_status)
    boundary = capture_prior_process_boundary(db)

    assert fail_interrupted_watchlist_runs(db, boundary.runs) == 1

    row = _run_row(db, run_id)
    assert row["status"] == "failed"
    assert row["error_msg"] == INTERRUPTED_RUN_ERROR
    assert row["finished_at"], "a failed run must carry an end time"


def test_finished_history_is_never_rewritten(db):
    source_id = _source(db)
    finished = {
        status: _run(db, source_id, status)
        for status in ("completed", "failed", "cancelled")
    }
    stranded = _run(db, source_id, "running")
    boundary = capture_prior_process_boundary(db)

    assert fail_interrupted_watchlist_runs(db, boundary.runs) == 1

    for status, run_id in finished.items():
        assert _run_row(db, run_id)["status"] == status
    assert _run_row(db, stranded)["status"] == "failed"


def test_an_existing_error_message_is_kept(db):
    """A run that recorded WHY it failed keeps its own text."""
    source_id = _source(db)
    run_id = _run(db, source_id, "running")
    with db.transaction() as conn:
        conn.execute(
            "UPDATE local_watchlist_runs SET error_msg = ? WHERE id = ?",
            ("upstream returned 503", run_id),
        )
    boundary = capture_prior_process_boundary(db)

    fail_interrupted_watchlist_runs(db, boundary.runs)

    assert _run_row(db, run_id)["error_msg"] == "upstream returned 503"


def test_a_clean_database_sweeps_to_zero(db):
    source_id = _source(db)
    _run(db, source_id, "completed")
    boundary = capture_prior_process_boundary(db)
    assert fail_interrupted_watchlist_runs(db, boundary.runs) == 0


def test_the_sweep_is_idempotent(db):
    source_id = _source(db)
    _run(db, source_id, "running")
    boundary = capture_prior_process_boundary(db)
    assert fail_interrupted_watchlist_runs(db, boundary.runs) == 1
    assert fail_interrupted_watchlist_runs(db, boundary.runs) == 0


def test_reconcile_covers_runs_and_briefings_together(db):
    source_id = _source(db)
    run_id = _run(db, source_id, "running")
    watchlist_id = _watchlist(db)
    stranded = _briefing(db, watchlist_id, "generating")
    finished = _briefing(db, watchlist_id, "complete")
    boundary = capture_prior_process_boundary(db)

    reconciled = reconcile_interrupted_subscription_work(db, boundary)

    assert reconciled["runs"] == 1
    assert reconciled["briefings"] == 1
    # The other two tables are swept too, with nothing to do here.
    assert reconciled["scripts"] == 0
    assert reconciled["audio"] == 0
    assert _run_row(db, run_id)["status"] == "failed"
    assert _briefing_status(db, stranded) == "failed"
    assert _briefing_status(db, finished) == "complete"


def test_one_failing_sweep_does_not_veto_the_others(db, monkeypatch):
    """A wedged guard in any one table is a feature the user cannot use."""
    watchlist_id = _watchlist(db)
    stranded = _briefing(db, watchlist_id, "generating")
    boundary = capture_prior_process_boundary(db)

    import tldw_chatbook.Subscriptions.startup_reconcile as module

    def _boom(_db, _max_row_id):
        raise RuntimeError("this table is unhappy")

    monkeypatch.setattr(module, "fail_interrupted_watchlist_runs", _boom)

    reconciled = reconcile_interrupted_subscription_work(db, boundary)

    assert "runs" not in reconciled, "the failing sweep reports nothing"
    assert reconciled["briefings"] == 1
    assert _briefing_status(db, stranded) == "failed"


# --- the boundary that keeps this process's own rows out of reach ----------
#
# Qodo review of PR #1972: `on_mount` starts the scheduler worker and the
# sweep runs later, as a deferred startup task, so an unscoped sweep failed
# live rows this process had just created. The end-to-end proof is in
# `test_startup_reconcile_scheduler_race.py`; these pin the mechanism.


def test_a_row_created_after_the_boundary_is_never_swept(db):
    """The whole point: this process's own in-flight work is out of reach."""
    source_id = _source(db)
    stranded = _run(db, source_id, "running")
    boundary = capture_prior_process_boundary(db)

    # Everything from here stands in for work this process starts after the
    # database was opened -- a scheduled check, a Check Now press, anything.
    live = _run(db, source_id, "running")

    reconciled = reconcile_interrupted_subscription_work(db, boundary)

    assert reconciled["runs"] == 1, "only the row that predates this process"
    assert _run_row(db, stranded)["status"] == "failed"
    assert _run_row(db, live)["status"] == "running", (
        "the sweep failed a run this process launched after opening the DB"
    )


def test_the_boundary_protects_every_table_the_sweep_touches(db):
    """Briefings, scripts and audio get the same protection as runs."""
    source_id = _source(db)
    watchlist_id = _watchlist(db)
    old_run = _run(db, source_id, "running")
    old_briefing = _briefing(db, watchlist_id, "generating")
    old_script = _script(db, old_briefing, "generating")
    old_audio = _audio(db, old_script, "generating")
    boundary = capture_prior_process_boundary(db)

    new_run = _run(db, source_id, "running")
    new_briefing = _briefing(db, watchlist_id, "generating")
    new_script = _script(db, new_briefing, "generating")
    new_audio = _audio(db, new_script, "generating")

    reconciled = reconcile_interrupted_subscription_work(db, boundary)

    assert reconciled == {"runs": 1, "briefings": 1, "scripts": 1, "audio": 1}
    for table, row_id in (
        ("local_watchlist_runs", old_run),
        ("briefings", old_briefing),
        ("briefing_scripts", old_script),
        ("briefing_audio", old_audio),
    ):
        assert _status(db, table, row_id) == "failed", table
    for table, row_id, alive in (
        ("local_watchlist_runs", new_run, "running"),
        ("briefings", new_briefing, "generating"),
        ("briefing_scripts", new_script, "generating"),
        ("briefing_audio", new_audio, "generating"),
    ):
        assert _status(db, table, row_id) == alive, (
            f"{table} row created after the boundary was swept anyway"
        )


def test_an_empty_boundary_sweeps_nothing_at_all(db):
    """No boundary is a refusal to sweep, not a licence to sweep everything.

    A table that could not be read contributes `None`. Leaving a row wedged
    is recoverable on the next launch; failing a live one is not.
    """
    source_id = _source(db)
    stranded = _run(db, source_id, "running")
    watchlist_id = _watchlist(db)
    briefing = _briefing(db, watchlist_id, "generating")

    reconciled = reconcile_interrupted_subscription_work(db, PriorProcessBoundary())

    assert reconciled == {"runs": 0, "briefings": 0, "scripts": 0, "audio": 0}
    assert _run_row(db, stranded)["status"] == "running"
    assert _briefing_status(db, briefing) == "generating"


def test_an_empty_table_yields_no_boundary_and_so_no_sweep(db):
    """`MAX(id)` of an empty table is NULL, which is the same refusal.

    Correct by coincidence rather than by luck: an empty table holds nothing
    a previous process could have stranded, so there is nothing to sweep --
    and every row that appears later belongs to this process.
    """
    boundary = capture_prior_process_boundary(db)
    assert boundary == PriorProcessBoundary(None, None, None, None)

    source_id = _source(db)
    live = _run(db, source_id, "running")

    assert reconcile_interrupted_subscription_work(db, boundary)["runs"] == 0
    assert _run_row(db, live)["status"] == "running"


def test_deleting_the_highest_row_cannot_let_a_later_row_slip_under(db):
    """The AUTOINCREMENT guarantee the boundary rests on, asserted directly.

    A plain `INTEGER PRIMARY KEY` reuses the highest freed rowid, so deleting
    the boundary row would let the *next* insert land at or below the
    boundary and be swept as if it were old. `local_watchlist_runs` declares
    `AUTOINCREMENT`, whose `sqlite_sequence` counter never goes backwards. If
    that declaration is ever dropped, this test is the alarm.
    """
    source_id = _source(db)
    highest = _run(db, source_id, "completed")
    boundary = capture_prior_process_boundary(db)
    assert boundary.runs == highest

    with db.transaction() as conn:
        conn.execute("DELETE FROM local_watchlist_runs WHERE id = ?", (highest,))
    live = _run(db, source_id, "running")

    assert live > boundary.runs, "a reused rowid would break the boundary"
    assert reconcile_interrupted_subscription_work(db, boundary)["runs"] == 0
    assert _run_row(db, live)["status"] == "running"


def test_every_bounded_table_really_is_autoincrement(db):
    """The guarantee above, asserted for all four tables, not just one.

    `test_deleting_the_highest_row_cannot_let_a_later_row_slip_under` proves
    the behaviour for `local_watchlist_runs`. This is the cheap schema-level
    alarm for the other three: if any of them is ever rebuilt without
    `AUTOINCREMENT`, its rowids become reusable and the boundary silently
    stops meaning what `_BOUNDED_TABLES` says it means.
    """
    from tldw_chatbook.Subscriptions.startup_reconcile import _BOUNDED_TABLES

    with db.transaction() as conn:
        schemas = {
            row["name"]: row["sql"]
            for row in conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'table'"
            )
        }

    for _key, table in _BOUNDED_TABLES:
        assert table in schemas, f"{table} is missing from the schema"
        assert "AUTOINCREMENT" in schemas[table].upper(), (
            f"{table} no longer declares AUTOINCREMENT, so its rowids can be "
            f"reused after a delete and the startup-reconcile boundary can "
            f"sweep a row this process created"
        )


def test_the_reconcile_cannot_be_called_without_a_boundary(db):
    """The scoping must not be droppable by an innocent edit.

    Both entry points take the boundary as a required positional argument, so
    a caller that forgets it fails loudly at the call rather than quietly
    reverting to the unscoped sweep that failed live rows.
    """
    with pytest.raises(TypeError):
        reconcile_interrupted_subscription_work(db)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        fail_interrupted_watchlist_runs(db)  # type: ignore[call-arg]
