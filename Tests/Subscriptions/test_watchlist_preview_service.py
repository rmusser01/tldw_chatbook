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
