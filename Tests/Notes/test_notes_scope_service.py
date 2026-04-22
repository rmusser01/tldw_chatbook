import pytest

from tldw_chatbook.Notes.notes_scope_service import NotesScopeService, ScopeType
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeLocalNotes:
    def __init__(self):
        self.add_calls = []
        self.update_calls = []
        self.delete_calls = []
        self.search_calls = []
        self.link_calls = []
        self.unlink_calls = []
        self.add_keyword_calls = []
        self.keyword_rows = {
            "existing": {"id": 1, "keyword": "existing"},
            "stale": {"id": 2, "keyword": "stale"},
        }
        self.note_keywords = {
            "local-1": [
                {"id": 1, "keyword": "existing"},
                {"id": 2, "keyword": "stale"},
            ]
        }

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

    def get_keywords_for_note(self, user_id, note_id):
        return list(self.note_keywords.get(note_id, []))

    def get_keyword_by_text(self, user_id, keyword_text):
        return self.keyword_rows.get(keyword_text)

    def add_keyword(self, user_id, keyword_text):
        self.add_keyword_calls.append((user_id, keyword_text))
        new_id = len(self.keyword_rows) + 1
        self.keyword_rows[keyword_text] = {"id": new_id, "keyword": keyword_text}
        return new_id

    def link_note_to_keyword(self, user_id, note_id, keyword_id):
        self.link_calls.append(
            {
                "user_id": user_id,
                "note_id": note_id,
                "keyword_id": keyword_id,
            }
        )
        return True

    def unlink_note_from_keyword(self, user_id, note_id, keyword_id):
        self.unlink_calls.append(
            {
                "user_id": user_id,
                "note_id": note_id,
                "keyword_id": keyword_id,
            }
        )
        return True


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


class FakePolicyEnforcer:
    def __init__(self, denied_reason: str | None = None):
        self.denied_reason = denied_reason
        self.calls = []

    @classmethod
    def deny(cls, reason_code: str) -> "FakePolicyEnforcer":
        return cls(denied_reason=reason_code)

    def require_allowed(self, *, action_id: str) -> None:
        self.calls.append(action_id)
        if self.denied_reason is None:
            return
        raise PolicyDeniedError(
            action_id=action_id,
            reason_code=self.denied_reason,
            user_message=f"{action_id} denied",
            effective_source="local",
            authority_owner="server",
        )


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
async def test_notes_scope_service_denies_server_create_when_active_source_is_local():
    policy_enforcer = FakePolicyEnforcer.deny("wrong_source")
    scope_service = NotesScopeService(
        local_notes_service=FakeLocalNotes(),
        server_service=FakeServerNotes(),
        policy_enforcer=policy_enforcer,
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await scope_service.save_note(
            scope=ScopeType.SERVER_NOTE,
            title="Remote",
            content="Body",
        )

    assert exc.value.reason_code == "wrong_source"
    assert policy_enforcer.calls == ["notes.create.server"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "kwargs", "expected_action_id"),
    [
        (ScopeType.LOCAL_NOTE, {"user_id": "user-1"}, "notes.create.local"),
        (ScopeType.SERVER_NOTE, {}, "notes.create.server"),
        (ScopeType.WORKSPACE, {"workspace_id": "ws-1"}, "notes.create.workspace"),
    ],
)
async def test_notes_scope_service_uses_distinct_create_policy_actions_per_scope(
    scope,
    kwargs,
    expected_action_id,
):
    policy_enforcer = FakePolicyEnforcer()
    scope_service = NotesScopeService(
        local_notes_service=FakeLocalNotes(),
        server_service=FakeServerNotes(),
        policy_enforcer=policy_enforcer,
    )

    await scope_service.save_note(
        scope=scope,
        title="Draft",
        content="Body",
        **kwargs,
    )

    assert policy_enforcer.calls == [expected_action_id]


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
async def test_scope_service_routes_local_note_keywords_through_local_service():
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
        keywords=["existing", "fresh"],
    )

    assert result == {
        "id": "local-1",
        "version": 4,
        "title": "Local",
        "content": "Body",
        "keywords": ["existing", "fresh"],
    }
    assert local.link_calls == [
        {
            "user_id": "user-1",
            "note_id": "local-1",
            "keyword_id": 3,
        }
    ]
    assert local.unlink_calls == [
        {
            "user_id": "user-1",
            "note_id": "local-1",
            "keyword_id": 2,
        }
    ]
    assert local.add_keyword_calls == [("user-1", "fresh")]


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
