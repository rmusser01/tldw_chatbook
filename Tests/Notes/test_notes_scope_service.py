import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
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
        self.notes_for_keyword = {
            1: [
                {"id": "local-1", "title": "Local", "content": "Body", "version": 1},
                {"id": "local-2", "title": "Related", "content": "More", "version": 1},
            ],
            2: [{"id": "local-1", "title": "Local", "content": "Body", "version": 1}],
        }
        self.manual_links = []

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

    def get_note_by_id(self, user_id, note_id):
        if note_id == "local-1":
            return {"id": "local-1", "title": "Local", "content": "Body", "version": 1}
        if note_id == "local-2":
            return {"id": "local-2", "title": "Related", "content": "More", "version": 1}
        return None

    def list_notes(self, user_id, limit=100, offset=0):
        return [
            {"id": "local-1", "title": "Local", "content": "Body", "version": 1},
            {"id": "local-2", "title": "Related", "content": "More", "version": 1},
        ][offset : offset + limit]

    def get_keywords_for_note(self, user_id, note_id):
        return list(self.note_keywords.get(note_id, []))

    def get_notes_for_keyword(self, user_id, keyword_id, limit=50, offset=0):
        return list(self.notes_for_keyword.get(keyword_id, []))[offset : offset + limit]

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

    def create_note_link(self, user_id, note_id, to_note_id, directed=False, weight=None, metadata=None):
        edge = {
            "id": f"local:manual:{len(self.manual_links) + 1}",
            "source": note_id,
            "target": to_note_id,
            "type": "manual",
            "directed": directed,
            "weight": 1.0 if weight is None else weight,
            "metadata": metadata or {},
        }
        self.manual_links.append(edge)
        return edge

    def list_note_links(self, user_id, center_note_id=None, limit=200):
        links = list(self.manual_links)
        if center_note_id:
            links = [
                edge
                for edge in links
                if edge["source"] == center_note_id or edge["target"] == center_note_id
            ]
        return links[:limit]

    def delete_note_link(self, user_id, edge_id):
        before = len(self.manual_links)
        self.manual_links = [edge for edge in self.manual_links if edge["id"] != edge_id]
        return {"deleted": len(self.manual_links) != before, "edge_id": edge_id}


class FakeServerNotes:
    def __init__(self):
        self.saved_ids = []
        self.workspace_saved = []
        self.workspace_record_saves = []
        self.workspace_deletes = []
        self.workspace_source_saves = []
        self.workspace_source_deletes = []
        self.workspace_artifact_saves = []
        self.workspace_artifact_deletes = []
        self.deleted_ids = []
        self.server_queries = []
        self.workspace_queries = []
        self.loaded_workspaces = []
        self.graph_calls = []

    async def save_server_note(self, **kwargs):
        self.saved_ids.append(kwargs["note_id"])
        return {"id": kwargs["note_id"], "version": kwargs["version"]}

    async def save_workspace_note(self, **kwargs):
        self.workspace_saved.append((kwargs["workspace_id"], kwargs["note_id"]))
        return {"id": kwargs["note_id"], "workspace_id": kwargs["workspace_id"]}

    async def save_workspace(self, **kwargs):
        self.workspace_record_saves.append(kwargs)
        return {"id": kwargs["workspace_id"], "version": kwargs.get("version") or 1}

    async def delete_workspace(self, workspace_id):
        self.workspace_deletes.append(workspace_id)
        return {"deleted": True, "workspace_id": workspace_id}

    async def list_workspace_sources(self, workspace_id):
        self.workspace_queries.append(("sources", workspace_id))
        return [{"id": "src-1", "workspace_id": workspace_id}]

    async def save_workspace_source(self, **kwargs):
        self.workspace_source_saves.append(kwargs)
        return {"id": kwargs["source_id"], "workspace_id": kwargs["workspace_id"]}

    async def delete_workspace_source(self, workspace_id, source_id):
        self.workspace_source_deletes.append((workspace_id, source_id))
        return {"deleted": True, "source_id": source_id}

    async def list_workspace_artifacts(self, workspace_id):
        self.workspace_queries.append(("artifacts", workspace_id))
        return [{"id": "artifact-1", "workspace_id": workspace_id}]

    async def save_workspace_artifact(self, **kwargs):
        self.workspace_artifact_saves.append(kwargs)
        return {"id": kwargs["artifact_id"], "workspace_id": kwargs["workspace_id"]}

    async def delete_workspace_artifact(self, workspace_id, artifact_id):
        self.workspace_artifact_deletes.append((workspace_id, artifact_id))
        return {"deleted": True, "artifact_id": artifact_id}

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

    async def get_notes_graph(self, **kwargs):
        self.graph_calls.append(("graph", kwargs))
        return {"nodes": [], "edges": []}

    async def get_note_neighbors(self, note_id, **kwargs):
        self.graph_calls.append(("neighbors", note_id, kwargs))
        return {"nodes": [{"id": note_id}], "edges": []}

    async def create_note_link(self, note_id, **kwargs):
        self.graph_calls.append(("create_link", note_id, kwargs))
        return {"status": "created"}

    async def delete_note_link(self, edge_id):
        self.graph_calls.append(("delete_link", edge_id))
        return {"deleted": True}


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


class FakeSyncScopeService:
    def __init__(self):
        self.calls = []

    def record_dry_run_mirror_report(self, **kwargs):
        self.calls.append(kwargs)
        return {"backend": kwargs["mode"], "domain": kwargs["domain"]}


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
async def test_scope_service_routes_workspace_record_crud_to_server_service():
    server = FakeServerNotes()
    policy_enforcer = FakePolicyEnforcer()
    scope_service = NotesScopeService(
        local_notes_service=FakeLocalNotes(),
        server_service=server,
        policy_enforcer=policy_enforcer,
    )

    created = await scope_service.save_workspace(
        workspace_id="ws-1",
        name="Research",
    )
    updated = await scope_service.save_workspace(
        workspace_id="ws-1",
        name="Research Updated",
        version=3,
        archived=True,
    )
    deleted = await scope_service.delete_workspace(workspace_id="ws-1")

    assert created == {"id": "ws-1", "version": 1}
    assert updated == {"id": "ws-1", "version": 3}
    assert deleted == {"deleted": True, "workspace_id": "ws-1"}
    assert server.workspace_record_saves == [
        {"workspace_id": "ws-1", "name": "Research"},
        {"workspace_id": "ws-1", "name": "Research Updated", "version": 3, "archived": True},
    ]
    assert server.workspace_deletes == ["ws-1"]
    assert policy_enforcer.calls == [
        "notes.workspace.create.server",
        "notes.workspace.update.server",
        "notes.workspace.delete.server",
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_workspace_source_crud_to_server_service():
    server = FakeServerNotes()
    policy_enforcer = FakePolicyEnforcer()
    scope_service = NotesScopeService(
        local_notes_service=FakeLocalNotes(),
        server_service=server,
        policy_enforcer=policy_enforcer,
    )

    sources = await scope_service.list_workspace_sources(workspace_id="ws-1")
    saved = await scope_service.save_workspace_source(
        workspace_id="ws-1",
        source_id="src-1",
        media_id=42,
        title="Paper",
        source_type="pdf",
    )
    deleted = await scope_service.delete_workspace_source(
        workspace_id="ws-1",
        source_id="src-1",
    )

    assert sources == [{"id": "src-1", "workspace_id": "ws-1"}]
    assert saved == {"id": "src-1", "workspace_id": "ws-1"}
    assert deleted == {"deleted": True, "source_id": "src-1"}
    assert server.workspace_source_saves == [
        {
            "workspace_id": "ws-1",
            "source_id": "src-1",
            "media_id": 42,
            "title": "Paper",
            "source_type": "pdf",
        }
    ]
    assert server.workspace_source_deletes == [("ws-1", "src-1")]
    assert policy_enforcer.calls == [
        "notes.workspace.detail.server",
        "notes.workspace.update.server",
        "notes.workspace.update.server",
    ]


@pytest.mark.asyncio
async def test_scope_service_routes_workspace_artifact_crud_to_server_service():
    server = FakeServerNotes()
    policy_enforcer = FakePolicyEnforcer()
    scope_service = NotesScopeService(
        local_notes_service=FakeLocalNotes(),
        server_service=server,
        policy_enforcer=policy_enforcer,
    )

    artifacts = await scope_service.list_workspace_artifacts(workspace_id="ws-1")
    saved = await scope_service.save_workspace_artifact(
        workspace_id="ws-1",
        artifact_id="artifact-1",
        artifact_type="summary",
        title="Summary",
    )
    deleted = await scope_service.delete_workspace_artifact(
        workspace_id="ws-1",
        artifact_id="artifact-1",
    )

    assert artifacts == [{"id": "artifact-1", "workspace_id": "ws-1"}]
    assert saved == {"id": "artifact-1", "workspace_id": "ws-1"}
    assert deleted == {"deleted": True, "artifact_id": "artifact-1"}
    assert server.workspace_artifact_saves == [
        {
            "workspace_id": "ws-1",
            "artifact_id": "artifact-1",
            "artifact_type": "summary",
            "title": "Summary",
        }
    ]
    assert server.workspace_artifact_deletes == [("ws-1", "artifact-1")]
    assert policy_enforcer.calls == [
        "notes.workspace.detail.server",
        "notes.workspace.update.server",
        "notes.workspace.update.server",
    ]


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


@pytest.mark.asyncio
async def test_scope_service_routes_server_notes_graph_operations():
    server = FakeServerNotes()
    scope_service = NotesScopeService(
        local_notes_service=FakeLocalNotes(),
        server_service=server,
    )

    graph = await scope_service.get_notes_graph(
        scope=ScopeType.SERVER_NOTE,
        center_note_id="note:123",
        edge_types=["manual"],
    )
    neighbors = await scope_service.get_note_neighbors(
        scope=ScopeType.SERVER_NOTE,
        note_id="note:123",
        edge_types=["manual", "backlink"],
    )
    created = await scope_service.create_note_link(
        scope=ScopeType.SERVER_NOTE,
        note_id="note:123",
        to_note_id="note:456",
    )
    deleted = await scope_service.delete_note_link(
        scope=ScopeType.SERVER_NOTE,
        edge_id="e:1",
    )

    assert graph == {"nodes": [], "edges": []}
    assert neighbors["nodes"] == [{"id": "note:123"}]
    assert created == {"status": "created"}
    assert deleted == {"deleted": True}
    assert server.graph_calls == [
        ("graph", {"center_note_id": "note:123", "edge_types": ["manual"]}),
        ("neighbors", "note:123", {"edge_types": ["manual", "backlink"]}),
        ("create_link", "note:123", {"to_note_id": "note:456"}),
        ("delete_link", "e:1"),
    ]


@pytest.mark.asyncio
async def test_scope_service_rejects_local_notes_graph_operations_explicitly():
    scope_service = NotesScopeService(
        local_notes_service=FakeLocalNotes(),
        server_service=FakeServerNotes(),
    )

    with pytest.raises(ValueError, match="Notes graph operations are currently server-backed"):
        await scope_service.get_notes_graph(scope=ScopeType.LOCAL_NOTE)


@pytest.mark.asyncio
async def test_scope_service_rejects_all_local_notes_graph_operations_before_backend_dispatch():
    server = FakeServerNotes()
    scope_service = NotesScopeService(
        local_notes_service=FakeLocalNotes(),
        server_service=server,
    )

    with pytest.raises(ValueError, match="Notes graph operations are currently server-backed"):
        await scope_service.get_note_neighbors(scope=ScopeType.LOCAL_NOTE, note_id="local-1")
    with pytest.raises(ValueError, match="Notes graph operations are currently server-backed"):
        await scope_service.create_note_link(
            scope=ScopeType.LOCAL_NOTE,
            note_id="local-1",
            to_note_id="local-2",
        )
    with pytest.raises(ValueError, match="Notes graph operations are currently server-backed"):
        await scope_service.delete_note_link(scope=ScopeType.LOCAL_NOTE, edge_id="local:manual:1")

    assert server.graph_calls == []


@pytest.mark.asyncio
async def test_scope_service_rejects_all_workspace_notes_graph_operations_before_backend_dispatch():
    server = FakeServerNotes()
    scope_service = NotesScopeService(
        local_notes_service=FakeLocalNotes(),
        server_service=server,
    )

    with pytest.raises(ValueError, match="Notes graph operations are currently server-backed"):
        await scope_service.get_notes_graph(scope=ScopeType.WORKSPACE, center_note_id="note:123")
    with pytest.raises(ValueError, match="Notes graph operations are currently server-backed"):
        await scope_service.get_note_neighbors(scope=ScopeType.WORKSPACE, note_id="note:123")
    with pytest.raises(ValueError, match="Notes graph operations are currently server-backed"):
        await scope_service.create_note_link(
            scope=ScopeType.WORKSPACE,
            note_id="note:123",
            to_note_id="note:456",
        )
    with pytest.raises(ValueError, match="Notes graph operations are currently server-backed"):
        await scope_service.delete_note_link(scope=ScopeType.WORKSPACE, edge_id="e:1")

    assert server.graph_calls == []


def test_scope_service_reports_known_notes_graph_capability_gaps():
    scope_service = NotesScopeService(
        local_notes_service=FakeLocalNotes(),
        server_service=FakeServerNotes(),
    )

    local_report = scope_service.list_unsupported_capabilities(scope=ScopeType.LOCAL_NOTE)
    workspace_report = scope_service.list_unsupported_capabilities(scope=ScopeType.WORKSPACE)
    server_report = scope_service.list_unsupported_capabilities(scope=ScopeType.SERVER_NOTE)

    assert local_report == [
        {
            "operation_id": "notes.graph.local",
            "source": "local",
            "supported": False,
            "reason_code": "local_contract_missing",
            "user_message": "Local/offline notes graph generation and manual graph links are deferred; graph operations are server-backed today.",
            "affected_action_ids": [
                "notes.graph.list.server",
                "notes.graph.detail.server",
                "notes.graph.create.server",
                "notes.graph.delete.server",
            ],
        }
    ]
    assert workspace_report == [
        {
            "operation_id": "notes.graph.workspace",
            "source": "workspace",
            "supported": False,
            "reason_code": "scope_not_supported",
            "user_message": "Workspace-scoped notes remain isolated from the global notes graph until sync/graph semantics are designed.",
            "affected_action_ids": [
                "notes.graph.list.server",
                "notes.graph.detail.server",
                "notes.graph.create.server",
                "notes.graph.delete.server",
            ],
        }
    ]
    assert server_report == []


def test_notes_scope_service_routes_server_note_sync_mirror_report_to_sync_scope():
    sync_scope = FakeSyncScopeService()
    scope_service = NotesScopeService(
        local_notes_service=FakeLocalNotes(),
        server_service=FakeServerNotes(),
        sync_scope_service=sync_scope,
    )

    result = scope_service.record_sync_mirror_report(
        scope=ScopeType.SERVER_NOTE,
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        local_records=[{"id": "local-note-1"}],
        remote_records=[{"id": "remote-note-1"}],
    )

    assert result == {"backend": "server", "domain": "notes"}
    assert sync_scope.calls == [
        {
            "mode": "server",
            "domain": "notes",
            "entity_type": "note",
            "server_profile_id": "server-a",
            "authenticated_principal_id": "user-a",
            "workspace_scope": None,
            "local_records": [{"id": "local-note-1"}],
            "remote_records": [{"id": "remote-note-1"}],
        }
    ]


def test_notes_scope_service_routes_workspace_note_sync_mirror_report_to_sync_scope():
    sync_scope = FakeSyncScopeService()
    scope_service = NotesScopeService(
        local_notes_service=FakeLocalNotes(),
        server_service=FakeServerNotes(),
        sync_scope_service=sync_scope,
    )

    result = scope_service.record_sync_mirror_report(
        scope=ScopeType.WORKSPACE,
        server_profile_id="server-a",
        workspace_id="workspace-1",
    )

    assert result == {"backend": "server", "domain": "workspace_notes"}
    assert sync_scope.calls[0]["domain"] == "workspace_notes"
    assert sync_scope.calls[0]["entity_type"] == "workspace_note"
    assert sync_scope.calls[0]["workspace_scope"] == "workspace-1"
