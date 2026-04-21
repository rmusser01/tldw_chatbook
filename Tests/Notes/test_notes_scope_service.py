import pytest

from tldw_chatbook.Notes.notes_scope_service import NotesScopeService, ScopeType


class FakeLocalNotes:
    def __init__(self):
        self.add_calls = []
        self.update_calls = []
        self.delete_calls = []
        self.search_calls = []

    def add_note(self, user_id, title, content, note_id=None):
        self.add_calls.append(
            {
                "user_id": user_id,
                "title": title,
                "content": content,
                "note_id": note_id,
            }
        )
        return note_id or "local-new"

    def update_note(self, user_id, note_id, update_data, expected_version):
        self.update_calls.append(
            {
                "user_id": user_id,
                "note_id": note_id,
                "update_data": update_data,
                "expected_version": expected_version,
            }
        )
        return True

    def soft_delete_note(self, user_id, note_id, expected_version):
        self.delete_calls.append(
            {
                "user_id": user_id,
                "note_id": note_id,
                "expected_version": expected_version,
            }
        )
        return True

    def search_notes(self, user_id, search_term, limit=10):
        self.search_calls.append(
            {
                "user_id": user_id,
                "search_term": search_term,
                "limit": limit,
            }
        )
        return [{"id": "local-1", "title": "Local"}]


class FakeServerNotes:
    def __init__(self):
        self.saved_ids = []
        self.workspace_saved = []
        self.deleted_ids = []
        self.server_queries = []
        self.workspace_queries = []
        self.loaded_workspaces = []

    async def save_server_note(self, **kwargs):
        self.saved_ids.append(kwargs["note_id"])
        return {"id": kwargs["note_id"], "version": kwargs["version"]}

    async def save_workspace_note(self, **kwargs):
        self.workspace_saved.append((kwargs["workspace_id"], kwargs["note_id"]))
        return {"id": kwargs["note_id"], "workspace_id": kwargs["workspace_id"]}

    async def delete_server_note(self, note_id, version):
        self.deleted_ids.append(("server", note_id, version))
        return {"deleted": True}

    async def delete_workspace_note(self, workspace_id, note_id, version):
        self.deleted_ids.append(("workspace", workspace_id, note_id, version))
        return {"deleted": True}

    async def search_server_notes(self, query, limit=10, offset=0):
        self.server_queries.append((query, limit, offset))
        return {"items": [{"id": "server-1"}], "count": 1}

    async def search_workspace_notes(self, workspace_id, query, notes=None):
        self.workspace_queries.append((workspace_id, query, notes))
        return [{"id": 5, "workspace_id": workspace_id}]

    async def load_workspace_context(self, workspace_id):
        self.loaded_workspaces.append(workspace_id)
        return {"workspace": {"id": workspace_id}}


@pytest.mark.asyncio
async def test_scope_service_routes_server_note_save_to_server_service():
    scope_service = NotesScopeService(
        local_notes_service=FakeLocalNotes(),
        server_service=FakeServerNotes(),
    )

    await scope_service.save_note(
        scope=ScopeType.SERVER_NOTE,
        note_id="note-1",
        title="Remote",
        content="Body",
        version=2,
    )

    assert scope_service.server_service.saved_ids == ["note-1"]


@pytest.mark.asyncio
async def test_scope_service_routes_workspace_note_save_to_workspace_service():
    scope_service = NotesScopeService(
        local_notes_service=FakeLocalNotes(),
        server_service=FakeServerNotes(),
    )

    await scope_service.save_note(
        scope=ScopeType.WORKSPACE,
        workspace_id="ws-1",
        note_id=11,
        title="Draft",
        content="Body",
        version=4,
    )

    assert scope_service.server_service.workspace_saved == [("ws-1", 11)]


@pytest.mark.asyncio
async def test_scope_service_routes_local_note_save_to_local_service():
    local = FakeLocalNotes()
    scope_service = NotesScopeService(
        local_notes_service=local,
        server_service=FakeServerNotes(),
    )

    result = await scope_service.save_note(
        scope=ScopeType.LOCAL_NOTE,
        user_id="user-1",
        note_id="local-1",
        title="Local",
        content="Body",
        version=3,
    )

    assert result is True
    assert local.update_calls == [
        {
            "user_id": "user-1",
            "note_id": "local-1",
            "update_data": {"title": "Local", "content": "Body"},
            "expected_version": 3,
        }
    ]


@pytest.mark.asyncio
async def test_scope_service_keeps_server_note_search_api_backed():
    server = FakeServerNotes()
    scope_service = NotesScopeService(
        local_notes_service=FakeLocalNotes(),
        server_service=server,
    )

    result = await scope_service.search_notes(
        scope=ScopeType.SERVER_NOTE,
        query="remote",
        limit=25,
    )

    assert server.server_queries == [("remote", 25, 0)]
    assert result["items"] == [{"id": "server-1"}]


@pytest.mark.asyncio
async def test_scope_service_keeps_workspace_note_search_client_side():
    server = FakeServerNotes()
    scope_service = NotesScopeService(
        local_notes_service=FakeLocalNotes(),
        server_service=server,
    )

    workspace_notes = [{"id": 5, "workspace_id": "ws-1", "title": "Draft"}]
    result = await scope_service.search_notes(
        scope=ScopeType.WORKSPACE,
        workspace_id="ws-1",
        query="draft",
        workspace_notes=workspace_notes,
    )

    assert server.workspace_queries == [("ws-1", "draft", workspace_notes)]
    assert result == [{"id": 5, "workspace_id": "ws-1"}]


@pytest.mark.asyncio
async def test_scope_service_loads_workspace_context_from_server_service():
    server = FakeServerNotes()
    scope_service = NotesScopeService(
        local_notes_service=FakeLocalNotes(),
        server_service=server,
    )

    result = await scope_service.load_workspace_context(
        scope=ScopeType.WORKSPACE,
        workspace_id="ws-7",
    )

    assert server.loaded_workspaces == ["ws-7"]
    assert result == {"workspace": {"id": "ws-7"}}
