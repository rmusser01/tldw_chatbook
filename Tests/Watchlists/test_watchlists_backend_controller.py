import pytest
from unittest.mock import AsyncMock

from tldw_chatbook.UI.Watchlists_Modules.watchlists_backend_controller import WatchlistsBackendController


class FakeScopeService:
    async def list_watch_items(self, *, runtime_backend, **kwargs):
        return [{"id": 1, "title": "Source"}]

    async def create_watch_item(self, *, runtime_backend, payload):
        return {"id": 1, **payload}


def test_controller_normalizes_backend():
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=FakeScopeService(), server_service=None)
    assert ctrl._normalize_backend("server") == "server"
    assert ctrl._normalize_backend(None) == "local"


@pytest.mark.asyncio
async def test_list_sources_routes_to_scope_service():
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=FakeScopeService(), server_service=None)
    items = await ctrl.list_sources(runtime_backend="local")
    assert len(items) == 1
    assert items[0]["title"] == "Source"


@pytest.mark.asyncio
async def test_create_source_routes_to_scope_service():
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=FakeScopeService(), server_service=None)
    result = await ctrl.create_source(payload={"name": "New"})
    assert result["name"] == "New"


@pytest.mark.asyncio
async def test_preview_source_routes_to_scope_service():
    scope_service = AsyncMock()
    scope_service.preview_source = AsyncMock(return_value={"items": ["a"], "log_text": "ok"})
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=scope_service, server_service=None)

    result = await ctrl.preview_source(runtime_backend="local", source_config={"url": "http://example.com/feed"})

    scope_service.preview_source.assert_awaited_once_with(
        runtime_backend="local", source_config={"url": "http://example.com/feed"}
    )
    assert result == {"items": ["a"], "log_text": "ok"}


@pytest.mark.asyncio
async def test_check_now_routes_to_scope_service():
    scope_service = AsyncMock()
    scope_service.check_now = AsyncMock(return_value={"run_id": "42", "status": "queued"})
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=scope_service, server_service=None)

    result = await ctrl.check_now(runtime_backend="local", source_id="1")

    scope_service.check_now.assert_awaited_once_with(runtime_backend="local", source_id="1")
    assert result["run_id"] == "42"


@pytest.mark.asyncio
async def test_import_opml_routes_to_scope_service():
    scope_service = AsyncMock()
    scope_service.import_opml = AsyncMock(return_value={"created": 2, "sources": [{"id": 1}, {"id": 2}]})
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=scope_service, server_service=None)

    result = await ctrl.import_opml(runtime_backend="local", xml_text="<opml></opml>")

    scope_service.import_opml.assert_awaited_once_with(runtime_backend="local", xml_text="<opml></opml>")
    assert result["created"] == 2
    assert len(result["sources"]) == 2


@pytest.mark.asyncio
async def test_export_opml_routes_to_scope_service():
    scope_service = AsyncMock()
    scope_service.export_opml = AsyncMock(return_value="<opml></opml>")
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=scope_service, server_service=None)

    result = await ctrl.export_opml(runtime_backend="local")

    scope_service.export_opml.assert_awaited_once_with(runtime_backend="local")
    assert result == "<opml></opml>"


@pytest.mark.asyncio
async def test_list_items_routes_to_scope_service():
    scope_service = AsyncMock()
    scope_service.list_items = AsyncMock(return_value=[{"id": "local:watchlist_item:1", "title": "Post"}])
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=scope_service, server_service=None)

    result = await ctrl.list_items(runtime_backend="local")

    scope_service.list_items.assert_awaited_once_with(runtime_backend="local")
    assert len(result) == 1
    assert result[0]["title"] == "Post"


@pytest.mark.asyncio
async def test_cancel_run_routes_to_scope_service():
    scope_service = AsyncMock()
    scope_service.cancel_run = AsyncMock(return_value={"run_id": "42", "status": "cancelled"})
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=scope_service, server_service=None)

    result = await ctrl.cancel_run(runtime_backend="local", run_id="42")

    scope_service.cancel_run.assert_awaited_once_with(runtime_backend="local", run_id="42")
    assert result["run_id"] == "42"
    assert result["status"] == "cancelled"


@pytest.mark.asyncio
async def test_save_alert_rule_create_routes_to_scope_service():
    scope_service = AsyncMock()
    scope_service.save_alert_rule = AsyncMock(return_value={"rule_id": "7", "name": "New Rule"})
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=scope_service, server_service=None)

    result = await ctrl.save_alert_rule(runtime_backend="local", payload={"name": "New Rule"})

    scope_service.save_alert_rule.assert_awaited_once_with(
        runtime_backend="local", payload={"name": "New Rule"}
    )
    assert result["rule_id"] == "7"


@pytest.mark.asyncio
async def test_save_alert_rule_update_routes_to_scope_service():
    scope_service = AsyncMock()
    scope_service.save_alert_rule = AsyncMock(return_value={"rule_id": "7", "name": "Updated Rule"})
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=scope_service, server_service=None)

    result = await ctrl.save_alert_rule(runtime_backend="local", payload={"id": "7", "name": "Updated Rule"})

    scope_service.save_alert_rule.assert_awaited_once_with(
        runtime_backend="local", payload={"id": "7", "name": "Updated Rule"}
    )
    assert result["name"] == "Updated Rule"
