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


def test_watchlist_projection_waiting_after_successful_check(tmp_path):
    """A successfully checked subscription remains WAITING in the projection."""
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
    assert tasks[0].status is TaskStatus.WAITING
    assert tasks[0].next_run_at is not None


def test_watchlist_projection_paused_subscription(tmp_path):
    """Paused subscriptions map to PAUSED."""
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


def test_watchlist_projection_inactive_subscription(tmp_path):
    """Inactive subscriptions map to DISABLED."""
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    subscription_id = subs_db.add_subscription(
        name="Inactive feed",
        type="rss",
        source="https://example.com/feed.rss",
        check_frequency=300,
    )
    subs_db.update_subscription(subscription_id, is_active=0)

    projection = WatchlistProjection(subs_db)
    tasks = projection.list_jobs()

    assert len(tasks) == 1
    assert tasks[0].status is TaskStatus.DISABLED


def test_watchlist_projection_error_needs_attention(tmp_path):
    """Subscriptions with an error and no successful check map to NEEDS_ATTENTION."""
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    subscription_id = subs_db.add_subscription(
        name="Failing feed",
        type="rss",
        source="https://example.com/failing.rss",
        check_frequency=300,
    )
    subs_db.record_check_error(subscription_id, error="Connection refused")

    projection = WatchlistProjection(subs_db)
    tasks = projection.list_jobs()

    assert len(tasks) == 1
    assert tasks[0].status is TaskStatus.NEEDS_ATTENTION


def test_watchlist_projection_error_with_successful_check_is_waiting(tmp_path):
    """A subscription that had at least one successful check stays WAITING despite later errors."""
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    subscription_id = subs_db.add_subscription(
        name="Recovering feed",
        type="rss",
        source="https://example.com/recover.rss",
        check_frequency=300,
    )
    subs_db.record_check_result(subscription_id, items=[])
    subs_db.record_check_error(subscription_id, error="Temporary timeout")

    projection = WatchlistProjection(subs_db)
    tasks = projection.list_jobs()

    assert len(tasks) == 1
    assert tasks[0].status is TaskStatus.WAITING


def test_watchlist_projection_source_fallback(tmp_path):
    """When source is missing, the projection falls back to the subscription type."""
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    subs_db.add_subscription(
        name="No source feed",
        type="rss",
        source="",
        check_frequency=300,
    )

    projection = WatchlistProjection(subs_db)
    tasks = projection.list_jobs()

    assert len(tasks) == 1
    assert tasks[0].source == "rss"


def test_watchlist_projection_schedule_summary_boundaries():
    """Schedule summaries use the largest whole unit for the check frequency."""
    from tldw_chatbook.Scheduling.services.watchlist_projection import _build_schedule_summary

    assert _build_schedule_summary(30) == "Every 30s"
    assert _build_schedule_summary(90) == "Every 1m"
    assert _build_schedule_summary(3600) == "Every 1h"
    assert _build_schedule_summary(86400) == "Every 1d"
    assert _build_schedule_summary(172800) == "Every 2d"
    assert _build_schedule_summary(None) is None


def test_watchlist_projection_empty_db(tmp_path):
    """An empty subscription database projects to an empty task list."""
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    projection = WatchlistProjection(subs_db)
    assert projection.list_jobs() == []
