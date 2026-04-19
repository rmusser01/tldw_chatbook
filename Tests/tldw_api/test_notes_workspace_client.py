"""
Tests for notes, workspaces, and media endpoint wiring on the shared TLDW API client.
"""

from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api.client import TLDWAPIClient
from tldw_chatbook.tldw_api.notes_workspace_schemas import (
    MediaSearchRequest,
    NoteCreateRequest,
    NoteUpdateRequest,
    WorkspaceArtifactCreateRequest,
    WorkspaceArtifactUpdateRequest,
    WorkspaceCreateRequest,
    WorkspaceNoteCreateRequest,
    WorkspaceNoteUpdateRequest,
    WorkspaceSourceCreateRequest,
    WorkspaceSourceUpdateRequest,
    WorkspaceUpdateRequest,
)


@pytest.mark.asyncio
class TestNotesWorkspaceClient:
    async def test_list_server_notes_hits_notes_endpoint(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"notes": [], "count": 0})
        monkeypatch.setattr(client, "_request", mocked)

        await client.list_server_notes(limit=25, offset=0)

        args, kwargs = mocked.await_args
        assert args[:2] == ("GET", "/api/v1/notes/")
        assert kwargs["params"] == {"limit": 25, "offset": 0, "include_keywords": "true"}

    async def test_search_media_items_posts_to_media_search(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"items": [], "pagination": {"total": 0}})
        monkeypatch.setattr(client, "_request", mocked)

        await client.search_media_items(MediaSearchRequest(query="paper"))

        args, kwargs = mocked.await_args
        assert args[:2] == ("POST", "/api/v1/media/search")

    async def test_workspace_note_create_serializes_keywords(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"id": 1})
        monkeypatch.setattr(client, "_request", mocked)

        await client.create_workspace_note("ws-1", WorkspaceNoteCreateRequest(title="Draft", keywords=["alpha", "beta"]))

        args, kwargs = mocked.await_args
        assert args[:2] == ("POST", "/api/v1/workspaces/ws-1/notes")
        assert kwargs["json_data"] == {"title": "Draft", "content": "", "keywords": ["alpha", "beta"]}

    async def test_workspace_update_uses_workspace_path(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"id": "ws-1"})
        monkeypatch.setattr(client, "_request", mocked)

        await client.update_workspace("ws-1", WorkspaceUpdateRequest(name="Renamed", version=3))

        args, kwargs = mocked.await_args
        assert args[:2] == ("PATCH", "/api/v1/workspaces/ws-1")
        assert kwargs["json_data"] == {"name": "Renamed", "version": 3}

    async def test_workspace_source_create_uses_sources_collection(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"id": "src-1"})
        monkeypatch.setattr(client, "_request", mocked)

        await client.create_workspace_source(
            "ws-1",
            WorkspaceSourceCreateRequest(id="src-1", media_id=9, title="Paper", source_type="pdf"),
        )

        args, kwargs = mocked.await_args
        assert args[:2] == ("POST", "/api/v1/workspaces/ws-1/sources")
        assert kwargs["json_data"] == {
            "id": "src-1",
            "media_id": 9,
            "title": "Paper",
            "source_type": "pdf",
            "url": None,
            "position": 0,
            "selected": True,
        }

    async def test_workspace_artifact_update_uses_workspace_path(self, monkeypatch):
        client = TLDWAPIClient("http://localhost:8000")
        mocked = AsyncMock(return_value={"id": "art-1"})
        monkeypatch.setattr(client, "_request", mocked)

        await client.update_workspace_artifact(
            "ws-1",
            "art-1",
            WorkspaceArtifactUpdateRequest(title="Summary", version=2),
        )

        args, kwargs = mocked.await_args
        assert args[:2] == ("PUT", "/api/v1/workspaces/ws-1/artifacts/art-1")
        assert kwargs["json_data"] == {"title": "Summary", "version": 2}
