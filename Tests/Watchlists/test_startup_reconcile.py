"""Tests for the startup sweep that un-wedges interrupted work (task-19561).

A real file-backed `SubscriptionsDB` throughout: the sweep's whole point is
what it does to rows an earlier *process* left behind, and a mock cannot
disagree with the schema.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.startup_reconcile import (
    INTERRUPTED_RUN_ERROR,
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


def _briefing_status(db: SubscriptionsDB, briefing_id: int) -> str:
    with db.transaction() as conn:
        return conn.execute(
            "SELECT status FROM briefings WHERE id = ?", (briefing_id,)
        ).fetchone()["status"]


@pytest.mark.parametrize("stranded_status", ["running", "queued"])
def test_a_stranded_run_is_failed_with_a_distinguishable_reason(db, stranded_status):
    """`queued` matters as much as `running`: nothing will ever dispatch it."""
    source_id = _source(db)
    run_id = _run(db, source_id, stranded_status)

    assert fail_interrupted_watchlist_runs(db) == 1

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

    assert fail_interrupted_watchlist_runs(db) == 1

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

    fail_interrupted_watchlist_runs(db)

    assert _run_row(db, run_id)["error_msg"] == "upstream returned 503"


def test_a_clean_database_sweeps_to_zero(db):
    source_id = _source(db)
    _run(db, source_id, "completed")
    assert fail_interrupted_watchlist_runs(db) == 0


def test_the_sweep_is_idempotent(db):
    source_id = _source(db)
    _run(db, source_id, "running")
    assert fail_interrupted_watchlist_runs(db) == 1
    assert fail_interrupted_watchlist_runs(db) == 0


def test_reconcile_covers_runs_and_briefings_together(db):
    source_id = _source(db)
    run_id = _run(db, source_id, "running")
    watchlist_id = _watchlist(db)
    stranded = _briefing(db, watchlist_id, "generating")
    finished = _briefing(db, watchlist_id, "complete")

    reconciled = reconcile_interrupted_subscription_work(db)

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

    import tldw_chatbook.Subscriptions.startup_reconcile as module

    def _boom(_db):
        raise RuntimeError("this table is unhappy")

    monkeypatch.setattr(module, "fail_interrupted_watchlist_runs", _boom)

    reconciled = reconcile_interrupted_subscription_work(db)

    assert "runs" not in reconciled, "the failing sweep reports nothing"
    assert reconciled["briefings"] == 1
    assert _briefing_status(db, stranded) == "failed"
