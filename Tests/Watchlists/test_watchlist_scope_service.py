import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService
from tldw_chatbook.Subscriptions.watchlist_opml_service import WatchlistOpmlService
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
    # TASK-3604: import dedupes by URL first -- a bare AsyncMock would
    # auto-truthy this lookup and skip creation, so the fixture names the
    # "nothing exists yet" case explicitly.
    local_service.find_source_id_by_url = AsyncMock(return_value=None)
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
async def test_export_opml_nests_watchlist_structure_for_local():
    """TASK-3604 plan task 4: the local export assembles watchlists with
    member rows and unassigned rows, and the serializer nests them."""
    scope_service, local_service, _ = make_scope_service()
    local_service.list_watchlists = AsyncMock(
        return_value=[{"id": 9, "name": "Tech"}]
    )
    local_service.list_watchlist_source_rows = AsyncMock(
        return_value=[{"name": "AI", "url": "http://a.com", "source_type": "rss"}]
    )
    local_service.list_unassigned_source_rows = AsyncMock(
        return_value=[{"name": "Loose", "url": "http://b.com", "source_type": "rss"}]
    )

    result = await scope_service.export_opml(runtime_backend=WatchlistBackend.LOCAL)

    local_service.list_watchlist_source_rows.assert_awaited_once_with(watchlist_id=9)
    assert "<opml" in result
    assert "http://a.com" in result and "http://b.com" in result
    # The member feed is nested UNDER the folder; the loose one is not.
    items = WatchlistOpmlService().parse(result)
    by_url = {item["url"]: item for item in items}
    assert by_url["http://a.com"]["folder"] == "Tech"
    assert by_url["http://b.com"]["folder"] is None


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
        is_flagged=None, search=None, since=None,
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
        is_flagged=None, search=None, since=None,
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


# --- TASK-3072 plan task 7: the star write ------------------------------------


@pytest.mark.asyncio
async def test_set_item_flagged_forwards_denamespaced_id_to_local_service():
    scope_service, local_service, _ = make_scope_service()
    local_service.set_item_flagged = AsyncMock(return_value=None)

    await scope_service.set_item_flagged(
        runtime_backend=WatchlistBackend.LOCAL,
        item_id="local:watchlist_item:7",
        flagged=True,
    )

    # `_source_id_from_item_id` strips the display namespace, exactly as
    # `update_item` and `restore_items_new` already do.
    local_service.set_item_flagged.assert_awaited_once_with(item_id="7", flagged=True)


@pytest.mark.asyncio
async def test_set_item_flagged_server_backend_rejected():
    scope_service, local_service, _ = make_scope_service()
    local_service.set_item_flagged = AsyncMock(return_value=None)
    with pytest.raises(ValueError, match="only supported for the local backend"):
        await scope_service.set_item_flagged(
            runtime_backend=WatchlistBackend.SERVER, item_id=7, flagged=True
        )
    local_service.set_item_flagged.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_items_forwards_search_and_since():
    """TASK-3791 plan task 2: the reader's search/since terms reach the local
    service untouched, the same contract as every other list_items kwarg."""
    scope_service, local_service, _ = make_scope_service()
    local_service.list_items = AsyncMock(return_value=[])

    await scope_service.list_items(
        runtime_backend=WatchlistBackend.LOCAL,
        search="retrieval rubric",
        since="2026-08-08T00:00:00+00:00",
    )

    local_service.list_items.assert_awaited_once_with(
        source_id=None, status=None, limit=100, offset=0, run_id=None,
        watchlist_id=None, unassigned_only=False, statuses=None,
        is_flagged=None,
        search="retrieval rubric", since="2026-08-08T00:00:00+00:00",
    )


# --- TASK-3604 plan task 3: import assigns folder membership -------------------


_FOLDERED_OPML = (
    '<?xml version="1.0"?><opml version="2.0"><body>'
    '<outline text="Tech">'
    '<outline text="AI" type="rss" xmlUrl="http://example.com/ai"/>'
    '<outline text="ML" type="rss" xmlUrl="http://example.com/ml"/>'
    '</outline>'
    '<outline text="Loose" type="rss" xmlUrl="http://example.com/loose"/>'
    "</body></opml>"
)


@pytest.mark.asyncio
async def test_import_opml_assigns_folder_membership():
    """Foldered feeds join the folder's watchlist; top-level feeds stay
    Unassigned; the summary says what happened."""
    scope_service, local_service, _ = make_scope_service()
    local_service.find_source_id_by_url = AsyncMock(return_value=None)
    local_service.create_source = AsyncMock(
        side_effect=[
            {"id": "local:subscription:1", "source_id": 1},
            {"id": "local:subscription:2", "source_id": 2},
            {"id": "local:subscription:3", "source_id": 3},
        ]
    )
    local_service.resolve_or_create_watchlist = AsyncMock(
        return_value=({"id": 9, "name": "Tech"}, True)
    )
    local_service.add_source_to_watchlist = AsyncMock()

    result = await scope_service.import_opml(
        runtime_backend=WatchlistBackend.LOCAL, xml_text=_FOLDERED_OPML
    )

    local_service.resolve_or_create_watchlist.assert_awaited_once_with("Tech")
    assigned = {
        call.kwargs["source_id"]
        for call in local_service.add_source_to_watchlist.await_args_list
    }
    assert assigned == {1, 2}, "the two foldered feeds, and only they"
    assert all(
        call.kwargs["watchlist_id"] == 9
        for call in local_service.add_source_to_watchlist.await_args_list
    )
    assert result["created"] == 3
    assert result["existing"] == 0
    assert result["watchlists_created"] == ["Tech"]
    assert result["watchlists_reused"] == []
    assert result["assignments"] == 2


@pytest.mark.asyncio
async def test_import_opml_reuses_existing_sources_by_url():
    """A feed URL already in the DB is reused, never duplicated -- the
    additive-only round-trip no-op (ADR-043 rule 6) depends on it."""
    scope_service, local_service, _ = make_scope_service()
    local_service.find_source_id_by_url = AsyncMock(return_value=7)
    local_service.create_source = AsyncMock(
        side_effect=AssertionError("must not create for a known URL")
    )
    local_service.resolve_or_create_watchlist = AsyncMock(
        return_value=({"id": 9, "name": "Tech"}, False)
    )
    local_service.add_source_to_watchlist = AsyncMock()

    one_feed = (
        '<?xml version="1.0"?><opml version="2.0"><body>'
        '<outline text="Tech"><outline text="AI" type="rss" xmlUrl="http://example.com/ai"/>'
        "</outline></body></opml>"
    )
    result = await scope_service.import_opml(
        runtime_backend=WatchlistBackend.LOCAL, xml_text=one_feed
    )

    local_service.create_source.assert_not_awaited()
    local_service.add_source_to_watchlist.assert_awaited_once_with(
        watchlist_id=9, source_id=7
    )
    assert result["created"] == 0
    assert result["existing"] == 1
    assert result["watchlists_reused"] == ["Tech"]
