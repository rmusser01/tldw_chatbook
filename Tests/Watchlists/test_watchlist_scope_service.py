import pytest
from unittest.mock import AsyncMock, patch

from tldw_chatbook.Subscriptions.watchlist_scope_service import WatchlistBackend, WatchlistScopeService


def make_scope_service():
    local_service = AsyncMock()
    local_service.run_executor = AsyncMock(return_value={"items": []})
    local_service.execute_run = None
    server_service = AsyncMock()
    return WatchlistScopeService(local_service=local_service, server_service=server_service), local_service, server_service


@pytest.mark.asyncio
async def test_preview_source_local_uses_preview_service():
    scope_service, local_service, _ = make_scope_service()
    preview_result = {"items": [{"title": "Post"}], "log_text": "Previewed 1 item."}

    with patch(
        "tldw_chatbook.Subscriptions.watchlist_scope_service.WatchlistPreviewService"
    ) as MockPreviewService:
        instance = MockPreviewService.return_value
        instance.preview = AsyncMock(return_value=preview_result)

        result = await scope_service.preview_source(
            runtime_backend=WatchlistBackend.LOCAL,
            source_config={"source_type": "rss", "url": "http://example.com/feed"},
        )

        MockPreviewService.assert_called_once_with(run_executor=local_service.run_executor)
        instance.preview.assert_awaited_once_with({"source_type": "rss", "url": "http://example.com/feed"})
        assert result == preview_result


@pytest.mark.asyncio
async def test_preview_source_rejects_server():
    scope_service, _, _ = make_scope_service()
    with pytest.raises(ValueError, match="Preview is only supported for the local backend"):
        await scope_service.preview_source(
            runtime_backend=WatchlistBackend.SERVER,
            source_config={"source_type": "rss", "url": "http://example.com/feed"},
        )


@pytest.mark.asyncio
async def test_check_now_delegates_to_launch_run():
    scope_service, local_service, _ = make_scope_service()
    local_service.launch_run = AsyncMock(return_value={"run_id": 7, "status": "queued"})

    result = await scope_service.check_now(runtime_backend=WatchlistBackend.LOCAL, source_id=42)

    local_service.launch_run.assert_awaited_once_with(job_id=None, source_id=42)
    assert result == {"run_id": 7, "status": "queued"}


@pytest.mark.asyncio
async def test_import_opml_creates_local_sources():
    scope_service, local_service, _ = make_scope_service()
    local_service.create_source = AsyncMock(side_effect=[{"id": 1, "name": "A"}, {"id": 2, "name": "B"}])
    xml_text = """<?xml version="1.0"?>
    <opml version="2.0">
        <body>
            <outline text="A" title="A" type="rss" xmlUrl="http://a.com/feed"/>
            <outline text="B" title="B" type="rss" xmlUrl="http://b.com/feed"/>
        </body>
    </opml>
    """

    result = await scope_service.import_opml(runtime_backend=WatchlistBackend.LOCAL, xml_text=xml_text)

    assert local_service.create_source.await_count == 2
    assert result["created"] == 2
    assert result["sources"] == [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]


@pytest.mark.asyncio
async def test_import_opml_rejects_server():
    scope_service, _, _ = make_scope_service()
    with pytest.raises(ValueError, match="OPML import is only supported for the local backend"):
        await scope_service.import_opml(runtime_backend=WatchlistBackend.SERVER, xml_text="<opml/>")


@pytest.mark.asyncio
async def test_export_opml_lists_sources_and_returns_xml():
    scope_service, local_service, _ = make_scope_service()
    local_service.list_sources = AsyncMock(return_value=[{"name": "A", "url": "http://a.com", "source_type": "rss"}])

    result = await scope_service.export_opml(runtime_backend=WatchlistBackend.LOCAL)

    local_service.list_sources.assert_awaited_once_with(limit=10000, offset=0)
    assert "<opml" in result
    assert "http://a.com" in result
    assert "A" in result


@pytest.mark.asyncio
async def test_save_alert_rule_create_path():
    scope_service, local_service, _ = make_scope_service()
    local_service.create_alert_rule = AsyncMock(return_value={"id": 3, "name": "New Rule"})
    payload = {"name": "New Rule", "condition": "always"}

    result = await scope_service.save_alert_rule(runtime_backend=WatchlistBackend.LOCAL, payload=payload)

    local_service.create_alert_rule.assert_awaited_once_with(name="New Rule", condition="always")
    local_service.update_alert_rule.assert_not_awaited()
    assert result == {"id": 3, "name": "New Rule"}


@pytest.mark.asyncio
async def test_save_alert_rule_update_path():
    scope_service, local_service, _ = make_scope_service()
    local_service.update_alert_rule = AsyncMock(return_value={"id": 7, "name": "Updated Rule"})
    payload = {"id": 7, "name": "Updated Rule", "condition": "never"}

    result = await scope_service.save_alert_rule(runtime_backend=WatchlistBackend.LOCAL, payload=payload)

    local_service.update_alert_rule.assert_awaited_once_with("7", name="Updated Rule", condition="never")
    local_service.create_alert_rule.assert_not_awaited()
    assert result == {"id": 7, "name": "Updated Rule"}


@pytest.mark.asyncio
async def test_save_alert_rule_none_id_treated_as_create_and_stripped():
    scope_service, local_service, _ = make_scope_service()
    local_service.create_alert_rule = AsyncMock(return_value={"id": 9, "name": "Rule from None"})
    payload = {"id": None, "name": "Rule from None", "condition": "sometimes"}

    result = await scope_service.save_alert_rule(runtime_backend=WatchlistBackend.LOCAL, payload=payload)

    local_service.create_alert_rule.assert_awaited_once_with(name="Rule from None", condition="sometimes")
    local_service.update_alert_rule.assert_not_awaited()
    assert result == {"id": 9, "name": "Rule from None"}
