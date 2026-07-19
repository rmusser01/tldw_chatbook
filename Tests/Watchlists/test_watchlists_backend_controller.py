import pytest
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
