"""Tests for the watchlist projection service."""

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Scheduling.models import ScheduledTask, TaskStatus
from tldw_chatbook.Scheduling.services.watchlist_projection import WatchlistProjection


def test_watchlist_projection_from_subscriptions_db(tmp_path):
    """Subscriptions_DB rows are projected into ScheduledTask objects."""
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    subs_db.add_subscription(
        name="Hacker News RSS",
        type="rss",
        source="https://news.ycombinator.com/rss",
        check_frequency=3600,
    )

    projection = WatchlistProjection(subs_db)
    tasks = projection.list_jobs(owner_id="local")

    assert len(tasks) == 1
    task = tasks[0]
    assert isinstance(task, ScheduledTask)
    assert task.title == "Hacker News RSS"
    assert task.type == "watchlist_job"
    assert task.status is TaskStatus.WAITING
    assert task.schedule_summary == "Every 1h"
    assert task.owner_id == "local"
    assert task.source == "https://news.ycombinator.com/rss"
    assert task.id.startswith("watchlist:")
    assert task.next_run_at is not None


def test_watchlist_projection_status_found_results(tmp_path):
    """A subscription with a last_checked timestamp maps to FOUND_RESULTS."""
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    subscription_id = subs_db.add_subscription(
        name="Checked feed",
        type="atom",
        source="https://example.com/feed.atom",
        check_frequency=1800,
    )
    subs_db.record_check_result(subscription_id, items=[])

    projection = WatchlistProjection(subs_db)
    tasks = projection.list_jobs()

    assert len(tasks) == 1
    assert tasks[0].status is TaskStatus.FOUND_RESULTS
    assert tasks[0].next_run_at is not None


def test_watchlist_projection_paused_subscription(tmp_path):
    """Paused or inactive subscriptions map to PAUSED."""
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    subscription_id = subs_db.add_subscription(
        name="Paused feed",
        type="url",
        source="https://example.com/",
        check_frequency=60,
    )
    subs_db.update_subscription(subscription_id, is_paused=1)

    projection = WatchlistProjection(subs_db)
    tasks = projection.list_jobs()

    assert len(tasks) == 1
    assert tasks[0].status is TaskStatus.PAUSED


def test_watchlist_projection_empty_db(tmp_path):
    """An empty subscription database projects to an empty task list."""
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    projection = WatchlistProjection(subs_db)
    assert projection.list_jobs() == []
