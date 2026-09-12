"""The startup sweep must not fail work this process's own scheduler started.

Qodo review of PR #1972 (task-19561). `TldwCli.on_mount` starts the scheduler
worker; the startup reconcile is created *later*, as a deferred startup task
after post-mount setup. `SchedulerLoop.run()` ticks immediately after loading
its queue, so a due watchlist check can have launched a real `queued`/
`running` row before the sweep ever runs -- and the unscoped sweep then failed
that live row as "interrupted".

This is not the two-instance corner case the module documented as accepted:
it is single-process, on every launch, with nothing anywhere enforcing the
ordering it depended on.

Nothing here is hand-built. The row under test is created by the real
`SchedulerLoop` dispatching to the real `WatchlistCheckHandler`, which routes
through the real `LocalWatchlistsService.launch_run`/`execute_run` against a
real file-backed `SubscriptionsDB`. Only the HTTP fetch is replaced -- with a
block, so the check is genuinely still in flight when the sweep looks at it.

Against the merge base this test fails on the last assertion: the live run
comes back `failed` with `INTERRUPTED_RUN_ERROR`.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Scheduling.scheduler.handlers.watchlist_check_handler import (
    WatchlistCheckHandler,
)
from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop
from tldw_chatbook.Scheduling.services.watchlist_projection import WatchlistProjection
from tldw_chatbook.Subscriptions.startup_reconcile import (
    capture_prior_process_boundary,
    reconcile_interrupted_subscription_work,
)

pytestmark = pytest.mark.unit

#: Every await here is bounded: the subject is a shutdown path, and a test
#: that hangs is a test that tells nobody anything.
_TIMEOUT_SECONDS = 20.0


def _due_source(db: SubscriptionsDB) -> int:
    """A source whose cadence has already elapsed, so the first tick fires."""
    subscription_id = db.add_subscription(
        name="Watched page",
        type="url",
        source="https://example.com/watched",
        check_frequency=3600,
    )
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    with db.transaction() as conn:
        conn.execute(
            "UPDATE subscriptions SET last_checked = ? WHERE id = ?",
            (stale, subscription_id),
        )
    return subscription_id


def _run_rows(db: SubscriptionsDB) -> list[dict]:
    with db.transaction() as conn:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT id, status, error_msg FROM local_watchlist_runs ORDER BY id"
            )
        ]


@pytest.mark.asyncio
async def test_the_startup_sweep_spares_a_run_the_scheduler_just_launched(
    tmp_path, monkeypatch
):
    db = SubscriptionsDB(tmp_path / "subs.db", "test")

    # === what `TldwCli.__init__` does when it opens the database ===
    # Captured here, before any loop exists, which is the whole reason the fix
    # survives a reordering of `on_mount`.
    boundary = capture_prior_process_boundary(db)

    _due_source(db)

    fetch_started = asyncio.Event()
    release_fetch = asyncio.Event()

    async def blocking_fetch(url, *, client, max_bytes, **kwargs):
        fetch_started.set()
        await release_fetch.wait()
        raise RuntimeError("released")

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.guarded_fetch_httpx_async",
        blocking_fetch,
    )

    tasks_db = MagicMock()
    tasks_db.list_reminder_tasks.return_value = []
    loop = SchedulerLoop(
        tasks_db,
        handlers={"watchlist_job": WatchlistCheckHandler(subscriptions_db=db)},
        watchlist_projection=WatchlistProjection(db),
        poll_interval=0.05,
    )

    # === what `on_mount` does: the scheduler starts BEFORE the sweep runs ===
    scheduler_worker = asyncio.create_task(loop.run(), name="scheduler_worker")
    try:
        await asyncio.wait_for(fetch_started.wait(), timeout=_TIMEOUT_SECONDS)

        before = _run_rows(db)
        assert len(before) == 1, f"the scheduler should have launched one run: {before}"
        live_id = before[0]["id"]
        assert before[0]["status"] == "running", (
            f"the run must be genuinely in flight when the sweep looks: {before}"
        )

        # === what the deferred startup task does, moments later ===
        reconciled = await asyncio.wait_for(
            asyncio.to_thread(reconcile_interrupted_subscription_work, db, boundary),
            timeout=_TIMEOUT_SECONDS,
        )

        after = {row["id"]: row for row in _run_rows(db)}
        assert reconciled["runs"] == 0, (
            f"the sweep had nothing legitimate to reconcile, but claimed "
            f"{reconciled}"
        )
        assert after[live_id]["status"] == "running", (
            f"the startup sweep failed a LIVE scheduled check: {after[live_id]}"
        )
        assert after[live_id]["error_msg"] is None
    finally:
        loop.running = False
        release_fetch.set()
        scheduler_worker.cancel()
        try:
            await asyncio.wait_for(scheduler_worker, timeout=_TIMEOUT_SECONDS)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass


@pytest.mark.asyncio
async def test_the_sweep_still_fails_a_run_the_previous_process_stranded(
    tmp_path, monkeypatch
):
    """The counterpart: sparing live rows must not disarm the sweep.

    Same harness, but the stranded row exists *before* the boundary is taken,
    which is what a run left behind by a killed process looks like on the next
    launch. It must still be failed, while the scheduler's own new run is
    spared -- the two behaviours in one database, at the same moment.
    """
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    subscription_id = _due_source(db)

    # The previous process died here, mid-run.
    with db.transaction() as conn:
        stranded_id = conn.execute(
            "INSERT INTO local_watchlist_runs "
            "(source_id, status, created_at, updated_at) "
            "VALUES (?, 'running', datetime('now'), datetime('now'))",
            (subscription_id,),
        ).lastrowid

    boundary = capture_prior_process_boundary(db)

    fetch_started = asyncio.Event()
    release_fetch = asyncio.Event()

    async def blocking_fetch(url, *, client, max_bytes, **kwargs):
        fetch_started.set()
        await release_fetch.wait()
        raise RuntimeError("released")

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.guarded_fetch_httpx_async",
        blocking_fetch,
    )

    tasks_db = MagicMock()
    tasks_db.list_reminder_tasks.return_value = []
    loop = SchedulerLoop(
        tasks_db,
        handlers={"watchlist_job": WatchlistCheckHandler(subscriptions_db=db)},
        watchlist_projection=WatchlistProjection(db),
        poll_interval=0.05,
    )
    scheduler_worker = asyncio.create_task(loop.run(), name="scheduler_worker")
    try:
        await asyncio.wait_for(fetch_started.wait(), timeout=_TIMEOUT_SECONDS)
        live_id = [
            row["id"] for row in _run_rows(db) if row["id"] != stranded_id
        ][0]

        reconciled = await asyncio.wait_for(
            asyncio.to_thread(reconcile_interrupted_subscription_work, db, boundary),
            timeout=_TIMEOUT_SECONDS,
        )

        after = {row["id"]: row for row in _run_rows(db)}
        assert reconciled["runs"] == 1
        assert after[stranded_id]["status"] == "failed", (
            "the row the previous process stranded must still be un-wedged"
        )
        assert after[live_id]["status"] == "running"
    finally:
        loop.running = False
        release_fetch.set()
        scheduler_worker.cancel()
        try:
            await asyncio.wait_for(scheduler_worker, timeout=_TIMEOUT_SECONDS)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass


def test_the_app_captures_the_boundary_before_any_loop_exists():
    """Pins the fix where a reordering of `on_mount` cannot reach it.

    Constructing `TldwCli` runs `__init__` only -- no `on_mount`, no event
    loop, no scheduler worker. If the boundary is present at that point then
    every row this process later creates is above it by construction, whatever
    order `on_mount` does things in. If someone moves the capture into
    `on_mount` (or into the deferred startup task, where the sweep lives),
    this fails.
    """
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Subscriptions.startup_reconcile import PriorProcessBoundary

    app = TldwCli()
    boundary = getattr(app, "_subscriptions_prior_process_boundary", None)

    assert isinstance(boundary, PriorProcessBoundary), (
        "the startup-reconcile boundary must be captured in __init__, before "
        "an event loop (and therefore the scheduler) can exist"
    )
