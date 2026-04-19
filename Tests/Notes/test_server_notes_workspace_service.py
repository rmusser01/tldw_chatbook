import pytest

from tldw_chatbook.Notes.server_notes_workspace_service import (
    ServerNotesWorkspaceService,
)


class FakeClient:
    def __init__(self):
        self.search_calls = []
        self.workspace_calls = []

    async def search_server_notes(self, *, query=None, limit=10, offset=0, include_keywords=True):
        self.search_calls.append(
            {
                "query": query,
                "limit": limit,
                "offset": offset,
                "include_keywords": include_keywords,
            }
        )
        return {
            "notes": [
                {
                    "id": "note-1",
                    "title": "Remote",
                    "content": "Body",
                    "version": 4,
                    "keywords": [{"keyword": "alpha"}, {"keyword": "beta"}],
                }
            ],
            "count": 1,
        }

    async def get_workspace(self, workspace_id):
        self.workspace_calls.append(("workspace", workspace_id))
        return {
            "id": workspace_id,
            "name": "Research",
            "version": 7,
            "archived": False,
        }

    async def list_workspace_notes(self, workspace_id):
        self.workspace_calls.append(("notes", workspace_id))
        return {
            "items": [
                {
                    "id": 11,
                    "workspace_id": workspace_id,
                    "title": "Draft",
                    "content": "Workspace body",
                    "keywords_json": '["alpha"]',
                    "version": 3,
                }
            ]
        }

    async def list_workspace_sources(self, workspace_id):
        self.workspace_calls.append(("sources", workspace_id))
        return {
            "items": [
                {
                    "id": "src-1",
                    "workspace_id": workspace_id,
                    "media_id": 55,
                    "title": "Paper",
                    "source_type": "pdf",
                    "version": 5,
                }
            ]
        }

    async def list_workspace_artifacts(self, workspace_id):
        self.workspace_calls.append(("artifacts", workspace_id))
        return {
            "items": [
                {
                    "id": "artifact-1",
                    "workspace_id": workspace_id,
                    "artifact_type": "summary",
                    "title": "Summary",
                    "status": "complete",
                    "version": 9,
                }
            ]
        }


@pytest.mark.asyncio
async def test_service_serializes_workspace_note_keywords_for_update():
    service = ServerNotesWorkspaceService(client=FakeClient())

    payload = service.build_workspace_note_update_payload(
        title="Draft",
        content="Body",
        keywords=["alpha", "beta"],
        version=3,
    )

    assert payload.keywords_json == '["alpha", "beta"]'


def test_service_filters_workspace_notes_within_active_workspace_only():
    service = ServerNotesWorkspaceService(client=None)
    notes = [
        {"id": 1, "workspace_id": "ws-1", "title": "Alpha", "content": "One"},
        {"id": 2, "workspace_id": "ws-2", "title": "Alpha", "content": "Two"},
    ]

    filtered = service.filter_workspace_notes(notes, workspace_id="ws-1", query="alpha")

    assert [note["id"] for note in filtered] == [1]


def test_service_builds_workspace_source_create_payload_with_media_id():
    service = ServerNotesWorkspaceService(client=None)

    payload = service.build_workspace_source_create_payload(
        source_id="src-1",
        media_id=42,
        title="Paper",
        source_type="pdf",
        url="https://example.com/paper",
        position=1,
        selected=True,
    )

    assert payload.media_id == 42
    assert payload.id == "src-1"


def test_service_normalizes_workspace_note_editor_payload_with_version():
    service = ServerNotesWorkspaceService(client=None)

    normalized = service.normalize_workspace_note(
        {
            "id": 11,
            "workspace_id": "ws-1",
            "title": "Draft",
            "content": "Body",
            "keywords_json": '["alpha", "beta"]',
            "version": 6,
        }
    )

    assert normalized == {
        "id": 11,
        "workspace_id": "ws-1",
        "title": "Draft",
        "content": "Body",
        "keywords": ["alpha", "beta"],
        "version": 6,
    }


def test_service_normalizes_server_note_editor_payload_with_version():
    service = ServerNotesWorkspaceService(client=None)

    normalized = service.normalize_server_note(
        {
            "id": "note-1",
            "title": "Remote",
            "content": "Body",
            "keywords": [{"keyword": "alpha"}, {"keyword": "beta"}],
            "version": 4,
        }
    )

    assert normalized == {
        "id": "note-1",
        "title": "Remote",
        "content": "Body",
        "keywords": ["alpha", "beta"],
        "version": 4,
    }


@pytest.mark.asyncio
async def test_service_uses_api_backed_search_for_user_space_notes():
    client = FakeClient()
    service = ServerNotesWorkspaceService(client=client)

    result = await service.search_server_notes(query="remote", limit=25)

    assert client.search_calls == [
        {
            "query": "remote",
            "limit": 25,
            "offset": 0,
            "include_keywords": True,
        }
    ]
    assert result["items"][0]["version"] == 4
    assert result["items"][0]["keywords"] == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_service_loads_workspace_context_with_versions():
    client = FakeClient()
    service = ServerNotesWorkspaceService(client=client)

    context = await service.load_workspace_context("ws-1")

    assert client.workspace_calls == [
        ("workspace", "ws-1"),
        ("notes", "ws-1"),
        ("sources", "ws-1"),
        ("artifacts", "ws-1"),
    ]
    assert context["workspace"]["version"] == 7
    assert context["notes"][0]["version"] == 3
    assert context["sources"][0]["version"] == 5
    assert context["artifacts"][0]["version"] == 9
