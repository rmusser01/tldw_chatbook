from types import SimpleNamespace

import pytest
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.watchlist_preview_service import WatchlistPreviewService


@pytest.mark.asyncio
async def test_preview_uses_run_executor_when_provided():
    async def fake_executor(subscription):
        return {
            "items": [
                {"url": "https://example.com/post", "title": "Post", "content_hash": "hash-1"},
            ],
        }

    svc = WatchlistPreviewService(run_executor=fake_executor)
    result = await svc.preview({"source_type": "rss", "url": "https://example.com/feed"})

    assert result["items"][0]["url"] == "https://example.com/post"
    assert "Preview completed" in result["log_text"]


@pytest.mark.asyncio
async def test_preview_closes_its_database_on_success(monkeypatch):
    """Since task-689 made the ``:memory:`` DB a real, functional connection
    (rather than the inert, schema-less object it used to be), ``preview()``
    must close it deterministically -- otherwise every preview call leaks a
    thread-local sqlite3 connection and its in-memory database."""
    closed = []
    real_close = SubscriptionsDB.close

    def spy_close(self):
        closed.append(self)
        real_close(self)

    monkeypatch.setattr(SubscriptionsDB, "close", spy_close)

    async def fake_executor(subscription):
        return {"items": []}

    svc = WatchlistPreviewService(run_executor=fake_executor)
    await svc.preview({"source_type": "rss", "url": "https://example.com/feed"})

    assert len(closed) == 1


@pytest.mark.asyncio
async def test_preview_closes_its_database_even_when_the_work_raises(monkeypatch):
    """A failed preview is exactly when the resource must come back: the
    connection must not be leaked on the exception path either."""
    closed = []
    real_close = SubscriptionsDB.close

    def spy_close(self):
        closed.append(self)
        real_close(self)

    monkeypatch.setattr(SubscriptionsDB, "close", spy_close)

    async def raising_executor(subscription):
        raise RuntimeError("boom")

    svc = WatchlistPreviewService(run_executor=raising_executor)

    with pytest.raises(RuntimeError, match="boom"):
        await svc.preview({"source_type": "rss", "url": "https://example.com/feed"})

    assert len(closed) == 1


@pytest.mark.asyncio
async def test_preview_url_source_completes_end_to_end_without_raising(monkeypatch):
    """Regression for task-689, exercising the real (non-overridden)
    execution path -- no ``run_executor`` override, so this drives the
    default executor's ``URLMonitor(db).check_url()`` for real, including
    its write to ``url_snapshots``.

    Before the fix this raised twice over: first ``OperationalError: no
    such table: subscriptions`` (the in-memory schema bug), and once that
    was fixed on its own, ``IntegrityError`` (the ``url_snapshots`` write
    referencing the synthetic subscription id ``-1``, which has no parent
    row once foreign_keys=ON actually has a real schema to enforce against).
    """

    async def fake_guarded(
        url, *, client, max_bytes, trusted_origins=frozenset(), headers=None, params=None, auth=None
    ):
        return SimpleNamespace(
            status_code=200,
            headers={"content-type": "text/html"},
            text="<html><body>hello world</body></html>",
            final_url=url,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.guarded_fetch_httpx_async",
        fake_guarded,
    )

    svc = WatchlistPreviewService()
    result = await svc.preview({"source_type": "url", "url": "https://example.com/page"})

    # First check for a URL source stores a baseline snapshot and reports no
    # change yet -- the point of this test is that it completes at all.
    assert result["items"] == []
    assert "Preview completed" in result["log_text"]
