import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tldw_chatbook.Scheduling.scheduler.handlers.watchlist_check_handler import (
    WatchlistCheckHandler,
)


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


@pytest.mark.asyncio
async def test_feed_check_records_result(handler):
    items = [{"title": "New post", "url": "http://example.com/1"}]
    handler.feed_monitor.check_feed.return_value = items
    handler.subscriptions_db.get_subscription.return_value = _subscription("rss")

    await handler.handle(_task())

    handler.feed_monitor.check_feed.assert_awaited_once_with(
        handler.subscriptions_db.get_subscription.return_value
    )
    handler.url_monitor.check_url.assert_not_awaited()
    handler.subscriptions_db.record_check_result.assert_called_once()
    call_args = handler.subscriptions_db.record_check_result.call_args
    assert call_args.args[0] == 42
    assert call_args.kwargs["items"] == items
    assert call_args.kwargs["stats"]["new_items_found"] == 1
    assert isinstance(call_args.kwargs["stats"]["response_time_ms"], int)


@pytest.mark.asyncio
async def test_url_check_records_result(handler):
    result = {"changed": True, "url": "http://example.com/page"}
    handler.url_monitor.check_url.return_value = result
    handler.subscriptions_db.get_subscription.return_value = _subscription("url")

    await handler.handle(_task())

    handler.url_monitor.check_url.assert_awaited_once_with(
        handler.subscriptions_db.get_subscription.return_value
    )
    handler.feed_monitor.check_feed.assert_not_awaited()
    handler.subscriptions_db.record_check_result.assert_called_once()
    call_args = handler.subscriptions_db.record_check_result.call_args
    assert call_args.args[0] == 42
    assert call_args.kwargs["items"] == [result]


@pytest.mark.asyncio
async def test_url_check_with_none_result(handler):
    handler.url_monitor.check_url.return_value = None
    handler.subscriptions_db.get_subscription.return_value = _subscription("url")

    await handler.handle(_task())

    handler.subscriptions_db.record_check_result.assert_called_once()
    call_args = handler.subscriptions_db.record_check_result.call_args
    assert call_args.kwargs["items"] == []


@pytest.mark.asyncio
async def test_paused_subscription_is_skipped(handler):
    handler.subscriptions_db.get_subscription.return_value = _subscription(
        is_paused=True
    )

    await handler.handle(_task())

    handler.feed_monitor.check_feed.assert_not_awaited()
    handler.url_monitor.check_url.assert_not_awaited()
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
    handler.subscriptions_db.record_check_result.assert_not_called()
    handler.subscriptions_db.record_check_error.assert_not_called()


@pytest.mark.asyncio
async def test_monitor_error_records_check_error(handler):
    handler.feed_monitor.check_feed.side_effect = RuntimeError("feed unreachable")
    handler.subscriptions_db.get_subscription.return_value = _subscription("rss")

    await handler.handle(_task())

    handler.subscriptions_db.record_check_result.assert_not_called()
    handler.subscriptions_db.record_check_error.assert_called_once_with(
        42, "feed unreachable"
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
    handler.subscriptions_db.get_subscription.return_value = _subscription("sitemap")

    await handler.handle(_task())

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
    handler.feed_monitor.check_feed.return_value = []
    handler.subscriptions_db.get_subscription.return_value = _subscription("rss")

    await handler(_task())

    handler.feed_monitor.check_feed.assert_awaited_once()
    handler.subscriptions_db.record_check_result.assert_called_once()


@pytest.mark.asyncio
async def test_default_monitors_are_constructed():
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
        assert handler.feed_monitor is feed_cls.return_value
        assert handler.url_monitor is url_cls.return_value
        feed_cls.assert_called_once_with()
        url_cls.assert_called_once_with(db=db)


@pytest.mark.asyncio
async def test_metrics_success_path(handler, metrics_patch):
    counter, histogram = metrics_patch
    handler.feed_monitor.check_feed.return_value = [{"title": "Post"}]
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
    handler.subscriptions_db.get_subscription.return_value = _subscription("sitemap")

    await handler.handle(_task())

    _assert_metrics(
        counter,
        histogram,
        status="unknown_type",
        subscription_type="sitemap",
    )


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
    handler.feed_monitor.check_feed.side_effect = RuntimeError("feed unreachable")
    handler.subscriptions_db.get_subscription.return_value = _subscription("rss")

    await handler.handle(_task())

    _assert_metrics(
        counter,
        histogram,
        status="error",
        subscription_type="rss",
    )
