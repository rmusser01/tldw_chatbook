from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static, TextArea

from tldw_chatbook.UI.Embeddings_Management_Window import EmbeddingsManagementWindow


def _normalized_collection(
    backend: str,
    name: str,
    *,
    count: int = 0,
    dimension: Optional[int] = None,
    metadata: Optional[dict] = None,
    status: str = "ready",
) -> dict:
    payload_metadata = dict(metadata or {})
    if dimension is not None and "embedding_dimension" not in payload_metadata:
        payload_metadata["embedding_dimension"] = dimension
    return {
        "record_id": f"{backend}:embedding_collection:{name}",
        "record_type": "embedding_collection",
        "backend": backend,
        "backing_collection_name": name,
        "name": name,
        "count": count,
        "embedding_dimension": dimension,
        "metadata": payload_metadata,
        "provider": payload_metadata.get("provider"),
        "status": status,
    }


class FakeScopeService:
    def __init__(
        self,
        *,
        local_collections=None,
        server_collections=None,
        local_details=None,
        server_details=None,
    ):
        self.local_collections = list(local_collections or [])
        self.server_collections = list(server_collections or [])
        self.local_details = dict(local_details or {})
        self.server_details = dict(server_details or {})
        self.calls = []
        self.deleted_collections = []

    def _records_for_mode(self, mode: str) -> list[dict]:
        return self.server_collections if mode == "server" else self.local_collections

    def _details_for_mode(self, mode: str) -> dict[str, dict]:
        return self.server_details if mode == "server" else self.local_details

    async def list_collections(self, *, mode):
        self.calls.append(("list_collections", mode))
        return list(self._records_for_mode(mode))

    async def get_collection_detail(self, *, mode, collection_name):
        self.calls.append(("get_collection_detail", mode, collection_name))
        detail = self._details_for_mode(mode).get(collection_name)
        if detail is not None:
            return dict(detail)
        for record in self._records_for_mode(mode):
            if record.get("name") == collection_name:
                return dict(record)
        return _normalized_collection(mode, collection_name)

    async def delete_collection(self, *, mode, collection_name):
        self.calls.append(("delete_collection", mode, collection_name))
        self.deleted_collections.append((mode, collection_name))
        active_records = self._records_for_mode(mode)
        active_records[:] = [
            record for record in active_records if record.get("name") != collection_name
        ]
        self._details_for_mode(mode).pop(collection_name, None)


class _EmbeddingsManagementTestApp(App):
    def __init__(self, *, runtime_backend: str = "local", scope_service=None):
        super().__init__()
        self.current_runtime_backend = runtime_backend
        self.media_runtime_state = SimpleNamespace(runtime_backend=runtime_backend)
        self.rag_admin_scope_service = scope_service
        self.notify = Mock()
        self.push_screen = AsyncMock()
        self.push_screen_wait = AsyncMock(return_value=True)
        self.chachanotes_db = None
        self.media_db = None
        self.window = EmbeddingsManagementWindow(app_instance=self)
        self.window._initialize_embeddings = AsyncMock()
        self.window._load_models_list = AsyncMock()

    def compose(self) -> ComposeResult:
        yield self.window


@pytest.mark.asyncio
async def test_embeddings_window_loads_local_collections_from_scope_service():
    scope = FakeScopeService(
        local_collections=[_normalized_collection("local", "notes_embeddings", count=5)]
    )
    app = _EmbeddingsManagementTestApp(runtime_backend="local", scope_service=scope)

    async with app.run_test() as pilot:
        await pilot.pause()

        assert [record["name"] for record in app.window.collections] == ["notes_embeddings"]
        assert scope.calls == [("list_collections", "local")]


@pytest.mark.asyncio
async def test_embeddings_window_loads_server_collection_stats_on_selection():
    scope = FakeScopeService(
        server_collections=[_normalized_collection("server", "remote_embeddings")],
        server_details={
            "remote_embeddings": _normalized_collection(
                "server",
                "remote_embeddings",
                count=12,
                dimension=1536,
                metadata={"provider": "openai", "source": "server"},
            )
        },
    )
    app = _EmbeddingsManagementTestApp(runtime_backend="server", scope_service=scope)

    async with app.run_test() as pilot:
        await pilot.pause()
        await app.window._update_collection_info("remote_embeddings")
        await pilot.pause()

        count_widget = app.window.query_one("#embeddings-collection-count", Static)
        metadata_widget = app.window.query_one("#embeddings-collection-metadata", TextArea)

        assert "12" in str(count_widget.renderable)
        assert "1536" in metadata_widget.text
        assert ("get_collection_detail", "server", "remote_embeddings") in scope.calls


@pytest.mark.asyncio
async def test_embeddings_window_deletes_collection_via_scope_service():
    scope = FakeScopeService(
        local_collections=[_normalized_collection("local", "demo", count=3)]
    )
    app = _EmbeddingsManagementTestApp(runtime_backend="local", scope_service=scope)

    async with app.run_test() as pilot:
        await pilot.pause()
        app.window.selected_collection = "demo"

        with patch(
            "tldw_chatbook.Widgets.delete_confirmation_dialog.create_delete_confirmation",
            return_value=object(),
        ):
            event = Mock()
            await app.window.on_delete_collection(event)
            await pilot.pause()

        event.stop.assert_called_once()
        assert scope.deleted_collections == [("local", "demo")]
        assert app.window.selected_collection is None
        assert app.window.collections == []


@pytest.mark.asyncio
async def test_embeddings_window_batch_deletes_collections_via_scope_service():
    scope = FakeScopeService(
        server_collections=[
            _normalized_collection("server", "alpha", count=2),
            _normalized_collection("server", "beta", count=4),
        ]
    )
    app = _EmbeddingsManagementTestApp(runtime_backend="server", scope_service=scope)

    async with app.run_test() as pilot:
        await pilot.pause()
        app.window.selected_collections = {"beta", "alpha"}

        with patch(
            "tldw_chatbook.Widgets.delete_confirmation_dialog.create_delete_confirmation",
            return_value=object(),
        ):
            event = Mock()
            await app.window.on_delete_selected_collections(event)
            await pilot.pause()

        event.stop.assert_called_once()
        assert sorted(scope.deleted_collections) == [("server", "alpha"), ("server", "beta")]
        assert app.window.selected_collections == set()
        assert app.window.collections == []
