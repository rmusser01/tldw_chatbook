"""
Tests for notes, workspaces, and media endpoint wiring on the shared TLDW API client.
"""

from unittest.mock import AsyncMock

import httpx
import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient
from tldw_chatbook.tldw_api.exceptions import APIResponseError
from tldw_chatbook.tldw_api.notes_workspace_schemas import (
    EdgeType,
    GraphFormat,
    MediaSearchRequest,
    MediaListResponse,
    NoteGraphRequest,
    NoteLinkCreate,
    NoteCreateRequest,
    NoteListResponse,
    NoteUpdateRequest,
    WorkspaceArtifactCreateRequest,
    WorkspaceArtifactUpdateRequest,
    WorkspaceCreateRequest,
    WorkspaceNoteCreateRequest,
    WorkspaceNoteUpdateRequest,
    WorkspaceResponse,
    WorkspaceSourceCreateRequest,
    WorkspaceSourceUpdateRequest,
    WorkspaceUpdateRequest,
)


def _assert_request_call(call_args, expected_method, expected_endpoint, expected_kwargs):
    args, kwargs = call_args
    assert args[:2] == (expected_method, expected_endpoint)
    for key, value in expected_kwargs.items():
        assert kwargs[key] == value


class _FakeHTTPClient:
    def __init__(self, response):
        self.response = response

    async def request(self, *args, **kwargs):
        return self.response


@pytest.mark.asyncio
class TestNotesWorkspaceClient:
    @pytest.mark.parametrize(
        "method_name, call_args, call_kwargs, expected_method, expected_endpoint, expected_kwargs",
        [
            (
                "list_server_notes",
                (),
                {"limit": 25, "offset": 0},
                "GET",
                "/api/v1/notes/",
                {"params": {"limit": 25, "offset": 0, "include_keywords": "true"}},
            ),
            (
                "search_server_notes",
                (),
                {"query": "alpha", "tokens": ["beta", "gamma"], "limit": 5, "offset": 2, "include_keywords": True},
                "GET",
                "/api/v1/notes/search",
                {"params": {"query": "alpha", "tokens": ["beta", "gamma"], "limit": 5, "offset": 2, "include_keywords": "true"}},
            ),
            (
                "get_server_note",
                ("note-1",),
                {},
                "GET",
                "/api/v1/notes/note-1",
                {},
            ),
            (
                "create_server_note",
                (NoteCreateRequest(title=None, content="Body", keywords="alpha, beta"),),
                {},
                "POST",
                "/api/v1/notes/",
                {"json_data": {"content": "Body", "keywords": "alpha, beta"}},
            ),
            (
                "update_server_note",
                ("note-2", NoteUpdateRequest(title="Updated", keywords=["alpha", "beta"]), 7),
                {},
                "PUT",
                "/api/v1/notes/note-2",
                {
                    "json_data": {"title": "Updated", "keywords": ["alpha", "beta"]},
                    "headers": {"expected-version": "7"},
                },
            ),
            (
                "delete_server_note",
                ("note-3", 11),
                {},
                "DELETE",
                "/api/v1/notes/note-3",
                {"headers": {"expected-version": "11"}},
            ),
        ],
    )
    async def test_notes_endpoint_wiring(
        self,
        monkeypatch,
        method_name,
        call_args,
        call_kwargs,
        expected_method,
        expected_endpoint,
        expected_kwargs,
    ):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"ok": True})
        monkeypatch.setattr(client, "_request", mocked)

        await getattr(client, method_name)(*call_args, **call_kwargs)

        _assert_request_call(mocked.await_args, expected_method, expected_endpoint, expected_kwargs)

    async def test_notes_graph_endpoint_wiring(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"ok": True})
        monkeypatch.setattr(client, "_request", mocked)

        await client.get_notes_graph(
            NoteGraphRequest(
                center_note_id="note:123",
                radius=2,
                edge_types=[EdgeType.manual, EdgeType.wikilink],
                max_nodes=200,
                format=GraphFormat.cytoscape,
                allow_heavy=True,
            )
        )
        await client.get_note_neighbors(
            "note:123",
            edge_types=["manual", "backlink"],
            max_edges=100,
        )
        await client.create_note_link(
            "note:123",
            NoteLinkCreate(
                to_note_id="note:456",
                directed=True,
                weight=2.5,
                metadata={"label": "related"},
            ),
        )
        await client.delete_note_link("e:edge-1")

        expected_calls = [
            (
                "GET",
                "/api/v1/notes/graph",
                {
                    "params": {
                        "center_note_id": "note:123",
                        "radius": 2,
                        "edge_types": "manual,wikilink",
                        "max_nodes": 200,
                        "format": "cytoscape",
                        "allow_heavy": "true",
                    }
                },
            ),
            (
                "GET",
                "/api/v1/notes/note:123/neighbors",
                {"params": {"edge_types": "manual,backlink", "max_edges": 100}},
            ),
            (
                "POST",
                "/api/v1/notes/note:123/links",
                {
                    "json_data": {
                        "to_note_id": "note:456",
                        "directed": True,
                        "weight": 2.5,
                        "metadata": {"label": "related"},
                    }
                },
            ),
            ("DELETE", "/api/v1/notes/links/e:edge-1", {}),
        ]
        assert len(mocked.await_args_list) == len(expected_calls)
        for call_args, expected in zip(mocked.await_args_list, expected_calls):
            _assert_request_call(call_args, *expected)

    @pytest.mark.parametrize(
        "method_name, call_args, call_kwargs, expected_method, expected_endpoint, expected_kwargs",
        [
            (
                "list_workspaces",
                (),
                {},
                "GET",
                "/api/v1/workspaces/",
                {},
            ),
            (
                "get_workspace",
                ("ws-1",),
                {},
                "GET",
                "/api/v1/workspaces/ws-1",
                {},
            ),
            (
                "create_workspace",
                ("ws-1", WorkspaceCreateRequest(name="Workspace", archived=True, study_materials_policy="workspace")),
                {},
                "PUT",
                "/api/v1/workspaces/ws-1",
                {"json_data": {"name": "Workspace", "archived": True, "study_materials_policy": "workspace"}},
            ),
            (
                "update_workspace",
                ("ws-1", WorkspaceUpdateRequest(name="Renamed", version=3)),
                {},
                "PATCH",
                "/api/v1/workspaces/ws-1",
                {"json_data": {"name": "Renamed", "version": 3}},
            ),
            (
                "delete_workspace",
                ("ws-1",),
                {},
                "DELETE",
                "/api/v1/workspaces/ws-1",
                {},
            ),
        ],
    )
    async def test_workspace_endpoint_wiring(
        self,
        monkeypatch,
        method_name,
        call_args,
        call_kwargs,
        expected_method,
        expected_endpoint,
        expected_kwargs,
    ):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"ok": True})
        monkeypatch.setattr(client, "_request", mocked)

        await getattr(client, method_name)(*call_args, **call_kwargs)

        _assert_request_call(mocked.await_args, expected_method, expected_endpoint, expected_kwargs)

    @pytest.mark.parametrize(
        "method_name, call_args, call_kwargs, expected_method, expected_endpoint, expected_kwargs",
        [
            (
                "list_workspace_notes",
                ("ws-1",),
                {},
                "GET",
                "/api/v1/workspaces/ws-1/notes",
                {},
            ),
            (
                "create_workspace_note",
                ("ws-1", WorkspaceNoteCreateRequest(title="Draft", content="Body", keywords=["alpha", "beta"])),
                {},
                "POST",
                "/api/v1/workspaces/ws-1/notes",
                {"json_data": {"title": "Draft", "content": "Body", "keywords": ["alpha", "beta"]}},
            ),
            (
                "update_workspace_note",
                ("ws-1", 4, WorkspaceNoteUpdateRequest(title="Updated", keywords_json='["alpha"]', version=2)),
                {},
                "PUT",
                "/api/v1/workspaces/ws-1/notes/4",
                {"json_data": {"title": "Updated", "keywords_json": '["alpha"]', "version": 2}},
            ),
            (
                "delete_workspace_note",
                ("ws-1", 4),
                {},
                "DELETE",
                "/api/v1/workspaces/ws-1/notes/4",
                {},
            ),
        ],
    )
    async def test_workspace_note_endpoint_wiring(
        self,
        monkeypatch,
        method_name,
        call_args,
        call_kwargs,
        expected_method,
        expected_endpoint,
        expected_kwargs,
    ):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"ok": True})
        monkeypatch.setattr(client, "_request", mocked)

        await getattr(client, method_name)(*call_args, **call_kwargs)

        _assert_request_call(mocked.await_args, expected_method, expected_endpoint, expected_kwargs)

    @pytest.mark.parametrize(
        "method_name, call_args, call_kwargs, expected_method, expected_endpoint, expected_kwargs",
        [
            (
                "list_workspace_sources",
                ("ws-1",),
                {},
                "GET",
                "/api/v1/workspaces/ws-1/sources",
                {},
            ),
            (
                "create_workspace_source",
                ("ws-1", WorkspaceSourceCreateRequest(id="src-1", media_id=9, title="Paper", source_type="pdf")),
                {},
                "POST",
                "/api/v1/workspaces/ws-1/sources",
                {
                    "json_data": {
                        "id": "src-1",
                        "media_id": 9,
                        "title": "Paper",
                        "source_type": "pdf",
                        "url": None,
                        "position": 0,
                        "selected": True,
                    }
                },
            ),
            (
                "update_workspace_source",
                ("ws-1", "src-1", WorkspaceSourceUpdateRequest(title="Reordered", position=1, version=3)),
                {},
                "PUT",
                "/api/v1/workspaces/ws-1/sources/src-1",
                {"json_data": {"title": "Reordered", "position": 1, "version": 3}},
            ),
            (
                "delete_workspace_source",
                ("ws-1", "src-1"),
                {},
                "DELETE",
                "/api/v1/workspaces/ws-1/sources/src-1",
                {},
            ),
        ],
    )
    async def test_workspace_source_endpoint_wiring(
        self,
        monkeypatch,
        method_name,
        call_args,
        call_kwargs,
        expected_method,
        expected_endpoint,
        expected_kwargs,
    ):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"ok": True})
        monkeypatch.setattr(client, "_request", mocked)

        await getattr(client, method_name)(*call_args, **call_kwargs)

        _assert_request_call(mocked.await_args, expected_method, expected_endpoint, expected_kwargs)

    @pytest.mark.parametrize(
        "method_name, call_args, call_kwargs, expected_method, expected_endpoint, expected_kwargs",
        [
            (
                "list_workspace_artifacts",
                ("ws-1",),
                {},
                "GET",
                "/api/v1/workspaces/ws-1/artifacts",
                {},
            ),
            (
                "create_workspace_artifact",
                ("ws-1", WorkspaceArtifactCreateRequest(id="art-1", artifact_type="summary", title="Summary")),
                {},
                "POST",
                "/api/v1/workspaces/ws-1/artifacts",
                {
                    "json_data": {
                        "id": "art-1",
                        "artifact_type": "summary",
                        "title": "Summary",
                        "status": "pending",
                        "content": None,
                    }
                },
            ),
            (
                "update_workspace_artifact",
                ("ws-1", "art-1", WorkspaceArtifactUpdateRequest(title="Summary v2", total_tokens=10, version=2)),
                {},
                "PUT",
                "/api/v1/workspaces/ws-1/artifacts/art-1",
                {"json_data": {"title": "Summary v2", "total_tokens": 10, "version": 2}},
            ),
            (
                "delete_workspace_artifact",
                ("ws-1", "art-1"),
                {},
                "DELETE",
                "/api/v1/workspaces/ws-1/artifacts/art-1",
                {},
            ),
        ],
    )
    async def test_workspace_artifact_endpoint_wiring(
        self,
        monkeypatch,
        method_name,
        call_args,
        call_kwargs,
        expected_method,
        expected_endpoint,
        expected_kwargs,
    ):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"ok": True})
        monkeypatch.setattr(client, "_request", mocked)

        await getattr(client, method_name)(*call_args, **call_kwargs)

        _assert_request_call(mocked.await_args, expected_method, expected_endpoint, expected_kwargs)

    @pytest.mark.parametrize(
        "method_name, call_args, call_kwargs, expected_method, expected_endpoint, expected_kwargs",
        [
            (
                "list_media_items",
                (),
                {"page": 2, "results_per_page": 25, "include_keywords": True},
                "GET",
                "/api/v1/media/",
                {"params": {"page": 2, "results_per_page": 25, "include_keywords": "true"}},
            ),
            (
                "search_media_items",
                (MediaSearchRequest(query="paper"),),
                {"page": 3, "results_per_page": 20},
                "POST",
                "/api/v1/media/search",
                {
                    "json_data": MediaSearchRequest(query="paper").model_dump(exclude_none=True, mode="json"),
                    "params": {"page": 3, "results_per_page": 20},
                },
            ),
        ],
    )
    async def test_media_endpoint_wiring(
        self,
        monkeypatch,
        method_name,
        call_args,
        call_kwargs,
        expected_method,
        expected_endpoint,
        expected_kwargs,
    ):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"ok": True})
        monkeypatch.setattr(client, "_request", mocked)

        await getattr(client, method_name)(*call_args, **call_kwargs)

        _assert_request_call(mocked.await_args, expected_method, expected_endpoint, expected_kwargs)

    async def test_request_returns_empty_dict_for_204_no_content(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        request = httpx.Request("DELETE", "http://localhost:8000/api/v1/notes/note-1")
        response = httpx.Response(204, request=request)
        monkeypatch.setattr(client, "_get_client", AsyncMock(return_value=_FakeHTTPClient(response)))

        result = await client._request("DELETE", "/api/v1/notes/note-1")

        assert result == {}

    async def test_request_raises_on_empty_200_response(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        request = httpx.Request("GET", "http://localhost:8000/api/v1/notes")
        response = httpx.Response(200, request=request, content=b"")
        monkeypatch.setattr(client, "_get_client", AsyncMock(return_value=_FakeHTTPClient(response)))

        with pytest.raises(APIResponseError) as exc_info:
            await client._request("GET", "/api/v1/notes")

        assert "Failed to decode JSON response" in str(exc_info.value)

    async def test_note_list_response_validates_wrapper_payload(self):
        payload = {
            "notes": [{"id": "note-1", "title": "Alpha", "content": "Body", "version": 1}],
            "items": [{"id": "note-1", "title": "Alpha", "content": "Body", "version": 1}],
            "results": [{"id": "note-1", "title": "Alpha", "content": "Body", "version": 1}],
            "count": 1,
            "limit": 25,
            "offset": 0,
            "total": 1,
        }

        model = NoteListResponse.model_validate(payload)

        assert model.count == 1
        assert model.notes[0].id == "note-1"
        assert model.items[0].title == "Alpha"

    async def test_media_list_response_validates_search_payload(self):
        payload = {
            "items": [
                {"id": 9, "title": "Paper", "url": "/api/v1/media/9", "type": "pdf"},
                {"id": 10, "title": "Video", "url": "/api/v1/media/10", "type": "video"},
            ],
            "pagination": {
                "page": 1,
                "results_per_page": 20,
                "total_pages": 1,
                "total_items": 2,
            },
        }

        model = MediaListResponse.model_validate(payload)

        assert model.items[0].url == "/api/v1/media/9"
        assert model.pagination.total_items == 2

    async def test_workspace_response_validates_workspace_payload(self):
        payload = {
            "id": "ws-1",
            "name": "Workspace",
            "archived": True,
            "study_materials_policy": "workspace",
            "deleted": False,
            "created_at": "2026-04-19T00:00:00Z",
            "last_modified": "2026-04-19T00:01:00Z",
            "version": 4,
        }

        model = WorkspaceResponse.model_validate(payload)

        assert model.id == "ws-1"
        assert model.archived is True
        assert model.version == 4

    async def test_note_create_request_accepts_optional_title_and_keyword_string(self):
        model = NoteCreateRequest.model_validate(
            {
                "title": None,
                "content": "Body",
                "keywords": "alpha, beta",
            }
        )

        assert model.title is None
        assert model.normalized_keywords == ["alpha", "beta"]
