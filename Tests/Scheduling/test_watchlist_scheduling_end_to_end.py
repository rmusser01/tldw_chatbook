"""End-to-end cover for scheduled watchlist execution.

Every existing test in this area exercises one component in isolation: the
projection builds ``ScheduledTask``s, the handler checks a feed, the loop
dispatches to a handler. All of them passed while the app shipped with the
handler unregistered and shadow mode on, so nothing checked a watchlist on a
schedule at all (TASK-1210).

These tests join the seams: a real ``SubscriptionsDB`` row, projected, queued,
dispatched by a real ``SchedulerLoop``, landing back in the database.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Scheduling.scheduler.handlers.watchlist_check_handler import (
    WatchlistCheckHandler,
)
from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop
from tldw_chatbook.Scheduling.services.watchlist_projection import WatchlistProjection
from tldw_chatbook.Subscriptions import LocalWatchlistsService

pytestmark = pytest.mark.unit


def _due_subscription(subs_db: SubscriptionsDB) -> int:
    """Add an active feed whose cadence has already elapsed."""
    subscription_id = subs_db.add_subscription(
        name="Due feed",
        type="rss",
        source="https://example.com/feed.xml",
        check_frequency=3600,
    )
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    with subs_db.transaction() as conn:
        conn.execute(
            "UPDATE subscriptions SET last_checked = ? WHERE id = ?",
            (stale, subscription_id),
        )
    return subscription_id


def _loop(subs_db: SubscriptionsDB, handler: WatchlistCheckHandler) -> SchedulerLoop:
    """Build a loop whose only source of work is the watchlist projection."""
    tasks_db = MagicMock()
    tasks_db.list_reminder_tasks.return_value = []
    return SchedulerLoop(
        tasks_db,
        handlers={"watchlist_job": handler},
        watchlist_projection=WatchlistProjection(subs_db),
    )


@pytest.mark.asyncio
async def test_due_watchlist_is_dispatched_and_persisted(tmp_path):
    """A due subscription is checked and its result written back to the DB.

    TASK-1383: the handler now executes through ``LocalWatchlistsService``, so
    the stub goes in at the service's ``run_executor`` seam rather than at the
    handler's monitors -- which keeps the real ``launch_run``/``execute_run``
    persistence in the loop and lets this test assert the run row the Runs pane
    reads, which is what was missing.
    """
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    subscription_id = _due_subscription(subs_db)
    before = subs_db.get_subscription(subscription_id)["last_checked"]

    checked: list[str] = []

    def fake_executor(subscription):
        checked.append(subscription["name"])
        return [
            {
                "url": "https://example.com/post-1",
                "title": "Post 1",
                "content": "hello",
            }
        ]

    handler = WatchlistCheckHandler(
        subscriptions_db=subs_db,
        shadow_mode=False,
        watchlists_service=LocalWatchlistsService(
            db_factory=lambda: subs_db, run_executor=fake_executor
        ),
    )

    loop = _loop(subs_db, handler)
    loop.queue.load()
    await loop.tick()

    assert checked == ["Due feed"], "the due subscription was not checked"
    after = subs_db.get_subscription(subscription_id)["last_checked"]
    assert after != before, "the check result was not persisted to Subscriptions_DB"

    runs = await LocalWatchlistsService(db_factory=lambda: subs_db).list_runs(
        source_id=subscription_id
    )
    assert len(runs) == 1, "a scheduled check must be visible to the Runs pane"
    assert runs[0]["status"] == "completed"
    assert runs[0]["stats"]["new_items_found"] == 1


@pytest.mark.asyncio
async def test_shadow_mode_checks_without_persisting(tmp_path):
    """Shadow mode stays available for diagnostics and must not mutate the DB."""
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    subscription_id = _due_subscription(subs_db)
    before = subs_db.get_subscription(subscription_id)["last_checked"]

    feed_monitor = AsyncMock()
    feed_monitor.check_feed.return_value = [{"url": "https://example.com/post-1"}]
    handler = WatchlistCheckHandler(
        subscriptions_db=subs_db,
        feed_monitor=feed_monitor,
        url_monitor=AsyncMock(),
        shadow_mode=True,
    )

    loop = _loop(subs_db, handler)
    loop.queue.load()
    await loop.tick()

    feed_monitor.check_feed.assert_awaited_once()
    after = subs_db.get_subscription(subscription_id)["last_checked"]
    assert after == before, "shadow mode must not write check results"


@pytest.mark.asyncio
async def test_not_yet_due_watchlist_is_not_dispatched(tmp_path):
    """A subscription inside its cadence window is left alone."""
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    subs_db.add_subscription(
        name="Fresh feed",
        type="rss",
        source="https://example.com/fresh.xml",
        check_frequency=86_400,
    )

    feed_monitor = AsyncMock()
    handler = WatchlistCheckHandler(
        subscriptions_db=subs_db,
        feed_monitor=feed_monitor,
        url_monitor=AsyncMock(),
        shadow_mode=False,
    )

    loop = _loop(subs_db, handler)
    loop.queue.load()
    await loop.tick()

    feed_monitor.check_feed.assert_not_awaited()


@pytest.mark.asyncio
async def test_watchlist_handler_absent_leaves_task_undispatched(tmp_path):
    """Guards the actual TASK-1210 failure: no registered handler for the type.

    ``app.py`` used to omit the ``watchlist_job`` entry entirely when the feature
    flag was off, which the loop handles by logging and moving on -- silently.
    """
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    _due_subscription(subs_db)

    tasks_db = MagicMock()
    tasks_db.list_reminder_tasks.return_value = []
    loop = SchedulerLoop(
        tasks_db,
        handlers={},
        watchlist_projection=WatchlistProjection(subs_db),
    )
    loop.queue.load()

    assert len(loop.queue) == 1, "the projection should still queue the due task"
    await loop.tick()
    assert len(loop.queue) == 0, "the task is consumed with no handler to run it"
