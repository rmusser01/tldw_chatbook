from types import SimpleNamespace

import pytest
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
