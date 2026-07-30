import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tldw_chatbook.Scheduling.scheduler.handlers.watchlist_check_handler import (
    WatchlistCheckHandler,
)
from tldw_chatbook.Subscriptions import LocalWatchlistsService


def _task(subscription_id: int | str = 42, **overrides) -> dict:
    """Return a task dict matching the shape produced by ``WatchlistProjection``.

    Fields not read by the handler (``title``, ``status``, ``schedule_summary``,
    ``next_run_at``, ``owner_id``, ``source``) are included so the fixture stays
    representative of the projection output.
    """
    return {
        "id": f"watchlist:{subscription_id}",
        "title": "My Feed",
        "type": "watchlist_job",
        "status": "waiting",
        "schedule_summary": "Every 1h",
        "next_run_at": None,
        "owner_id": "local",
        "source": "http://example.com/feed",
        **overrides,
    }


def _subscription(sub_type: str = "rss", **overrides) -> dict:
    return {
        "id": 42,
        "name": "My Feed",
        "type": sub_type,
        "source": "http://example.com/feed",
        "is_paused": False,
        "is_active": True,
        **overrides,
    }


def _service_mock(*, status: str = "completed", stats: dict | None = None):
    """A stand-in for `LocalWatchlistsService`'s two-call execution contract.

    TASK-1383: the handler no longer executes checks itself, so these unit
    tests assert the delegation. What the service then *does* -- run rows,
    dispositions, per-URL checks -- is covered against real objects in
    `test_scheduled_watchlist_runs.py`, which is the point of that module.
    """
    service = AsyncMock()
    service.launch_run.return_value = {"run_id": 7, "source_id": 42}
    service.execute_run.return_value = {
        "run_id": 7,
        "status": status,
        "stats": stats if stats is not None else {"new_items_found": 1},
    }
    return service


@pytest.fixture
def handler():
    db = MagicMock()
    feed_monitor = AsyncMock()
    url_monitor = AsyncMock()
    return WatchlistCheckHandler(
        subscriptions_db=db,
        feed_monitor=feed_monitor,
        url_monitor=url_monitor,
        shadow_mode=False,
        watchlists_service=_service_mock(),
    )


@pytest.fixture
def metrics_patch():
    with (
        patch(
            "tldw_chatbook.Scheduling.scheduler.handlers.watchlist_check_handler.log_counter"
        ) as counter,
        patch(
            "tldw_chatbook.Scheduling.scheduler.handlers.watchlist_check_handler.log_histogram"
        ) as histogram,
    ):
        yield counter, histogram


def _assert_metrics(counter, histogram, *, status, subscription_type, shadow=None):
    counter.assert_called_once()
    histogram.assert_called_once()
    counter_args, counter_kwargs = counter.call_args
    histogram_args, histogram_kwargs = histogram.call_args
    assert counter_kwargs["labels"]["status"] == status
    assert counter_kwargs["labels"]["subscription_type"] == subscription_type
    assert histogram_kwargs["labels"]["status"] == status
    assert histogram_kwargs["labels"]["subscription_type"] == subscription_type
    assert isinstance(histogram_args[1], (int, float))
    if shadow:
        assert counter_kwargs["labels"]["shadow"] == shadow
        assert histogram_kwargs["labels"]["shadow"] == shadow
    else:
        assert "shadow" not in counter_kwargs["labels"]
        assert "shadow" not in histogram_kwargs["labels"]


@pytest.mark.parametrize(
    "sub_type", ["rss", "atom", "json_feed", "podcast", "url", "url_list", "sitemap", "api"]
)
@pytest.mark.asyncio
async def test_every_executable_type_is_launched_as_a_run(handler, sub_type):
    """TASK-1383: every storable type goes through the run seam, `sitemap` included.

    `sitemap` is listed here deliberately: the handler's own `_URL_TYPES` tuple
    omitted it, so a scheduled sitemap source was declared an unknown type and
    never checked at all.
    """
    handler.subscriptions_db.get_subscription.return_value = _subscription(sub_type)

    await handler.handle(_task())

    handler.watchlists_service.launch_run.assert_awaited_once_with(source_id=42)
    handler.watchlists_service.execute_run.assert_awaited_once_with(7)
    # The service records the check; a second write here would double-count it
    # into `subscription_stats` and re-bump the auto-pause counter.
    handler.subscriptions_db.record_check_result.assert_not_called()
    handler.subscriptions_db.record_check_error.assert_not_called()
    # The handler's own monitors belong to the shadow path now.
    handler.feed_monitor.check_feed.assert_not_awaited()
    handler.url_monitor.check_url.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_run_is_reported_without_a_second_error_record(handler):
    """`execute_run` handles its own fetch failures and does not re-raise."""
    handler.watchlists_service.execute_run.return_value = {
        "run_id": 7,
        "status": "failed",
        "stats": {"error_msg": "boom"},
    }
    handler.subscriptions_db.get_subscription.return_value = _subscription("url")

    await handler.handle(_task())

    handler.watchlists_service.record_run_failure.assert_not_awaited()
    handler.subscriptions_db.record_check_error.assert_not_called()


@pytest.mark.asyncio
async def test_paused_subscription_is_skipped(handler):
    handler.subscriptions_db.get_subscription.return_value = _subscription(
        is_paused=True
    )

    await handler.handle(_task())

    handler.feed_monitor.check_feed.assert_not_awaited()
    handler.url_monitor.check_url.assert_not_awaited()
    handler.watchlists_service.launch_run.assert_not_awaited()
    handler.subscriptions_db.record_check_result.assert_not_called()
    handler.subscriptions_db.record_check_error.assert_not_called()


@pytest.mark.asyncio
async def test_inactive_subscription_is_skipped(handler):
    handler.subscriptions_db.get_subscription.return_value = _subscription(
        is_active=False
    )

    await handler.handle(_task())

    handler.feed_monitor.check_feed.assert_not_awaited()
    handler.url_monitor.check_url.assert_not_awaited()
    handler.watchlists_service.launch_run.assert_not_awaited()
    handler.subscriptions_db.record_check_result.assert_not_called()
    handler.subscriptions_db.record_check_error.assert_not_called()


@pytest.mark.asyncio
async def test_failure_escaping_execute_run_marks_the_launched_run_failed(handler):
    """A fault around execution must not leave the launched row at `queued`.

    `record_run_failure` also calls `SubscriptionsDB.record_check_error`, which
    is the same call this handler used to make itself -- so the auto-pause
    counter still advances, exactly once.
    """
    error = RuntimeError("feed unreachable")
    handler.watchlists_service.execute_run.side_effect = error
    handler.subscriptions_db.get_subscription.return_value = _subscription("rss")

    await handler.handle(_task())

    handler.watchlists_service.record_run_failure.assert_awaited_once_with(
        7, source_id=42, error=error
    )
    handler.subscriptions_db.record_check_result.assert_not_called()
    handler.subscriptions_db.record_check_error.assert_not_called()


@pytest.mark.asyncio
async def test_failure_before_a_run_exists_falls_back_to_record_check_error(handler):
    """With no run row to fail, the error still reaches the subscription."""
    handler.watchlists_service.launch_run.side_effect = RuntimeError("launch failed")
    handler.subscriptions_db.get_subscription.return_value = _subscription("rss")

    await handler.handle(_task())

    handler.watchlists_service.record_run_failure.assert_not_awaited()
    handler.subscriptions_db.record_check_error.assert_called_once_with(
        42, "launch failed"
    )


@pytest.mark.asyncio
async def test_shadow_mode_executes_without_db_writes(handler):
    handler.shadow_mode = True
    items = [{"title": "Shadow post"}]
    handler.feed_monitor.check_feed.return_value = items
    handler.subscriptions_db.get_subscription.return_value = _subscription("rss")

    await handler.handle(_task())

    handler.feed_monitor.check_feed.assert_awaited_once()
    handler.subscriptions_db.record_check_result.assert_not_called()
    handler.subscriptions_db.record_check_error.assert_not_called()


@pytest.mark.asyncio
async def test_shadow_mode_does_not_record_errors(handler):
    handler.shadow_mode = True
    handler.feed_monitor.check_feed.side_effect = RuntimeError("boom")
    handler.subscriptions_db.get_subscription.return_value = _subscription("rss")

    await handler.handle(_task())

    handler.subscriptions_db.record_check_result.assert_not_called()
    handler.subscriptions_db.record_check_error.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_subscription_type_logs_and_returns(handler):
    """A type with no executor is reported, not launched into a failed run.

    `sitemap` used to be this test's example; it is a real, executable type and
    the handler was simply refusing to check it (TASK-1383). No type the schema
    accepts reaches this branch any more -- see
    `test_executable_types_match_every_type_the_db_accepts`.
    """
    handler.subscriptions_db.get_subscription.return_value = _subscription("gopher")

    await handler.handle(_task())

    handler.watchlists_service.launch_run.assert_not_awaited()
    handler.feed_monitor.check_feed.assert_not_awaited()
    handler.url_monitor.check_url.assert_not_awaited()
    handler.subscriptions_db.record_check_result.assert_not_called()
    handler.subscriptions_db.record_check_error.assert_not_called()


@pytest.mark.asyncio
async def test_missing_task_id_logs_and_returns(handler):
    await handler.handle(_task(subscription_id=""))

    handler.subscriptions_db.get_subscription.assert_not_called()
    handler.feed_monitor.check_feed.assert_not_awaited()
    handler.url_monitor.check_url.assert_not_awaited()


@pytest.mark.asyncio
async def test_bad_task_id_prefix_logs_and_returns(handler):
    await handler.handle({**_task(), "id": "reminder:42"})

    handler.subscriptions_db.get_subscription.assert_not_called()


@pytest.mark.asyncio
async def test_non_integer_task_id_logs_and_returns(handler):
    await handler.handle(_task(subscription_id="not-a-number"))

    handler.subscriptions_db.get_subscription.assert_not_called()


@pytest.mark.asyncio
async def test_missing_subscription_logs_and_returns(handler):
    handler.subscriptions_db.get_subscription.return_value = None

    await handler.handle(_task())

    handler.feed_monitor.check_feed.assert_not_awaited()
    handler.url_monitor.check_url.assert_not_awaited()
    handler.subscriptions_db.record_check_result.assert_not_called()


@pytest.mark.asyncio
async def test_handler_is_callable(handler):
    handler.subscriptions_db.get_subscription.return_value = _subscription("rss")

    await handler(_task())

    handler.watchlists_service.launch_run.assert_awaited_once_with(source_id=42)
    handler.watchlists_service.execute_run.assert_awaited_once_with(7)


def test_default_monitors_are_built_lazily_not_at_construction():
    """The monitors serve the shadow path alone, so the normal path skips them.

    Shadow mode is off by default, so building both at construction meant every
    handler paid -- at app start -- for two objects it never touched.
    """
    db = MagicMock()
    with (
        patch(
            "tldw_chatbook.Scheduling.scheduler.handlers.watchlist_check_handler.FeedMonitor"
        ) as feed_cls,
        patch(
            "tldw_chatbook.Scheduling.scheduler.handlers.watchlist_check_handler.URLMonitor"
        ) as url_cls,
    ):
        handler = WatchlistCheckHandler(subscriptions_db=db)
        feed_cls.assert_not_called()
        url_cls.assert_not_called()

        assert handler.feed_monitor is feed_cls.return_value
        assert handler.url_monitor is url_cls.return_value
        feed_cls.assert_called_once_with()
        url_cls.assert_called_once_with(db=db, persist_snapshots=True)

        # Cached, not rebuilt per access.
        assert handler.feed_monitor is feed_cls.return_value
        assert handler.url_monitor is url_cls.return_value
        feed_cls.assert_called_once_with()
        url_cls.assert_called_once_with(db=db, persist_snapshots=True)


def test_shadow_url_monitor_does_not_persist_snapshots():
    """`persist_snapshots` follows the mode actually in force at first use."""
    db = MagicMock()
    with patch(
        "tldw_chatbook.Scheduling.scheduler.handlers.watchlist_check_handler.URLMonitor"
    ) as url_cls:
        handler = WatchlistCheckHandler(subscriptions_db=db, shadow_mode=True)
        assert handler.url_monitor is url_cls.return_value
        url_cls.assert_called_once_with(db=db, persist_snapshots=False)


def test_default_service_is_bound_to_the_handlers_db():
    """Production wiring stays zero-config: `app.py` passes only the db."""
    db = MagicMock()
    handler = WatchlistCheckHandler(subscriptions_db=db)
    assert isinstance(handler.watchlists_service, LocalWatchlistsService)
    assert handler.watchlists_service.db_factory() is db


@pytest.mark.asyncio
async def test_metrics_success_path(handler, metrics_patch):
    counter, histogram = metrics_patch
    handler.subscriptions_db.get_subscription.return_value = _subscription("rss")

    await handler.handle(_task())

    _assert_metrics(
        counter,
        histogram,
        status="success",
        subscription_type="rss",
    )


@pytest.mark.asyncio
async def test_metrics_shadow_path(handler, metrics_patch):
    counter, histogram = metrics_patch
    handler.shadow_mode = True
    handler.feed_monitor.check_feed.return_value = [{"title": "Post"}]
    handler.subscriptions_db.get_subscription.return_value = _subscription("rss")

    await handler.handle(_task())

    _assert_metrics(
        counter,
        histogram,
        status="success",
        subscription_type="rss",
        shadow="true",
    )


@pytest.mark.asyncio
async def test_metrics_skipped_subscription(handler, metrics_patch):
    counter, histogram = metrics_patch
    handler.subscriptions_db.get_subscription.return_value = _subscription(
        "rss", is_paused=True
    )

    await handler.handle(_task())

    _assert_metrics(
        counter,
        histogram,
        status="skipped",
        subscription_type="rss",
    )


@pytest.mark.asyncio
async def test_metrics_unknown_type(handler, metrics_patch):
    counter, histogram = metrics_patch
    handler.subscriptions_db.get_subscription.return_value = _subscription("gopher")

    await handler.handle(_task())

    _assert_metrics(
        counter,
        histogram,
        status="unknown_type",
        subscription_type="gopher",
    )


@pytest.mark.asyncio
async def test_metrics_error_path_for_a_failed_run(handler, metrics_patch):
    """A run the service marked failed is an errored check, not a successful one."""
    counter, histogram = metrics_patch
    handler.watchlists_service.execute_run.return_value = {
        "run_id": 7,
        "status": "failed",
        "stats": {},
    }
    handler.subscriptions_db.get_subscription.return_value = _subscription("url")

    await handler.handle(_task())

    _assert_metrics(counter, histogram, status="error", subscription_type="url")


@pytest.mark.asyncio
async def test_metrics_missing_task_id(handler, metrics_patch):
    counter, histogram = metrics_patch

    await handler.handle(_task(subscription_id=""))

    _assert_metrics(
        counter,
        histogram,
        status="missing",
        subscription_type="unknown",
    )


@pytest.mark.asyncio
async def test_metrics_error_path(handler, metrics_patch):
    counter, histogram = metrics_patch
    handler.watchlists_service.execute_run.side_effect = RuntimeError("feed unreachable")
    handler.subscriptions_db.get_subscription.return_value = _subscription("rss")

    await handler.handle(_task())

    _assert_metrics(
        counter,
        histogram,
        status="error",
        subscription_type="rss",
    )
