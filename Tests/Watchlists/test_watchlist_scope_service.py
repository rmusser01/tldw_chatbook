import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService
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
async def test_list_items_delegates_to_local_service():
    scope_service, local_service, _ = make_scope_service()
    local_service.list_items = AsyncMock(return_value=[{"id": "local:watchlist_item:1", "title": "Post"}])

    result = await scope_service.list_items(runtime_backend=WatchlistBackend.LOCAL, status="new")

    # `run_id` joined the delegation in TASK-2306: the Runs tab's Items
    # sub-region asks for one run's items through this same `items.list` route.
    local_service.list_items.assert_awaited_once_with(
        source_id=None, status="new", limit=100, offset=0, run_id=None,
        watchlist_id=None, unassigned_only=False, statuses=None,
    )
    assert len(result) == 1


@pytest.mark.asyncio
async def test_list_items_forwards_watchlist_scope():
    scope_service, local_service, _ = make_scope_service()
    local_service.list_items = AsyncMock(return_value=[])

    await scope_service.list_items(
        runtime_backend=WatchlistBackend.LOCAL,
        watchlist_id=3,
        statuses=["new", "reviewed"],
    )

    local_service.list_items.assert_awaited_once_with(
        source_id=None, status=None, limit=100, offset=0, run_id=None,
        watchlist_id=3, unassigned_only=False, statuses=["new", "reviewed"],
    )


@pytest.mark.asyncio
async def test_list_items_watchlist_scope_server_backend_rejected():
    scope_service, _, _ = make_scope_service()
    with pytest.raises(ValueError, match="Item listing is only supported for the local backend"):
        await scope_service.list_items(runtime_backend=WatchlistBackend.SERVER, watchlist_id=3)


def test_bundle_service_delegates_source_counts():
    db = MagicMock()
    expected = {7: {"total": 5, "unread": 2}}
    db.get_source_item_counts.return_value = expected
    bundle_service = WatchlistBundleService(db)

    result = bundle_service.get_source_item_counts()

    db.get_source_item_counts.assert_called_once_with()
    assert result == expected


@pytest.mark.asyncio
async def test_list_items_rejects_server():
    scope_service, _, _ = make_scope_service()
    with pytest.raises(ValueError, match="Item listing is only supported for the local backend"):
        await scope_service.list_items(runtime_backend=WatchlistBackend.SERVER)


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


@pytest.mark.asyncio
async def test_mark_all_read_forwards_scope_kwargs_to_local_service():
    scope_service, local_service, _ = make_scope_service()
    local_service.mark_all_read = AsyncMock(return_value=[1, 2, 3])

    result = await scope_service.mark_all_read(
        runtime_backend=WatchlistBackend.LOCAL,
        watchlist_id=3,
        unassigned_only=True,
    )

    local_service.mark_all_read.assert_awaited_once_with(
        source_id=None, watchlist_id=3, unassigned_only=True
    )
    assert result == [1, 2, 3]


@pytest.mark.asyncio
async def test_mark_all_read_server_backend_rejected():
    scope_service, local_service, _ = make_scope_service()
    with pytest.raises(ValueError, match="Item status updates are only supported for the local backend"):
        await scope_service.mark_all_read(runtime_backend=WatchlistBackend.SERVER)
    local_service.mark_all_read.assert_not_awaited()


@pytest.mark.asyncio
async def test_restore_items_new_forwards_denamespaced_ids_to_local_service():
    scope_service, local_service, _ = make_scope_service()
    local_service.restore_items_new = AsyncMock(return_value=2)

    result = await scope_service.restore_items_new(
        runtime_backend=WatchlistBackend.LOCAL,
        item_ids=["local:watchlist_item:1", 2],
    )

    # `_source_id_from_item_id` strips the display namespace, exactly as
    # `update_item` already does for its single item id; bare ids pass through.
    local_service.restore_items_new.assert_awaited_once_with(item_ids=["1", 2])
    assert result == 2


@pytest.mark.asyncio
async def test_restore_items_new_server_backend_rejected():
    scope_service, local_service, _ = make_scope_service()
    with pytest.raises(ValueError, match="Item status updates are only supported for the local backend"):
        await scope_service.restore_items_new(
            runtime_backend=WatchlistBackend.SERVER, item_ids=[1]
        )
    local_service.restore_items_new.assert_not_awaited()
