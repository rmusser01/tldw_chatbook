import inspect
from unittest.mock import Mock

import pytest

import tldw_chatbook.Notes.server_notes_workspace_service as notes_workspace_module
from tldw_chatbook.Notes.server_notes_workspace_service import (
    ServerNotesWorkspaceService,
)
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeClient:
    def __init__(self):
        self.search_calls = []
        self.workspace_calls = []
        self.graph_calls = []

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
            "audio_provider": "openai",
            "audio_model": "gpt-4o-mini-tts",
            "audio_voice": "alloy",
            "audio_speed": 1.25,
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

    async def get_notes_graph(self, request_data):
        self.graph_calls.append(("graph", request_data))
        return {
            "nodes": [{"id": "note:123", "type": "note", "label": "Note"}],
            "edges": [],
            "limits": {"max_nodes": 200, "max_edges": 400, "max_degree": 40},
        }

    async def get_note_neighbors(self, note_id, **kwargs):
        self.graph_calls.append(("neighbors", note_id, kwargs))
        return {
            "nodes": [{"id": note_id, "type": "note", "label": "Note"}],
            "edges": [],
            "limits": {"max_nodes": 200, "max_edges": 400, "max_degree": 40},
        }

    async def create_note_link(self, note_id, request_data):
        self.graph_calls.append(("create_link", note_id, request_data))
        return {"status": "created", "edge": {"id": "e:1", "source": note_id}}

    async def delete_note_link(self, edge_id):
        self.graph_calls.append(("delete_link", edge_id))
        return {"deleted": True, "edge_id": edge_id}


class FakeClientProvider:
    def __init__(self, client):
        self.client = client
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        return self.client


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
            authority_owner="shared",
        )


def test_server_notes_workspace_service_module_does_not_reference_legacy_config_client_builders():
    source = inspect.getsource(notes_workspace_module)

    assert "build_runtime_api_client_from_config" not in source
    assert "build_runtime_api_client(app_config" not in source


@pytest.mark.asyncio
async def test_service_uses_provider_backed_client_when_no_direct_client():
    client = FakeClient()
    provider = FakeClientProvider(client)
    service = ServerNotesWorkspaceService(client=None, client_provider=provider)

    result = await service.search_server_notes(query="remote", limit=25)

    assert service.client is None
    assert provider.build_calls == 1
    assert client.search_calls == [
        {
            "query": "remote",
            "limit": 25,
            "offset": 0,
            "include_keywords": True,
        }
    ]
    assert result["items"][0]["title"] == "Remote"


@pytest.mark.asyncio
async def test_service_prefers_direct_client_over_provider():
    direct_client = FakeClient()
    provider_client = FakeClient()
    provider = FakeClientProvider(provider_client)
    service = ServerNotesWorkspaceService(client=direct_client, client_provider=provider)

    await service.search_server_notes(query="remote")

    assert provider.build_calls == 0
    assert len(direct_client.search_calls) == 1
    assert provider_client.search_calls == []


@pytest.mark.asyncio
async def test_service_from_server_context_provider_uses_provider_client():
    client = FakeClient()
    provider = FakeClientProvider(client)
    policy_enforcer = FakePolicyEnforcer()
    service = ServerNotesWorkspaceService.from_server_context_provider(
        provider,
        policy_enforcer=policy_enforcer,
    )

    assert isinstance(service, ServerNotesWorkspaceService)
    assert service.client is None
    assert provider.build_calls == 0

    await service.search_server_notes(query="remote")

    assert service.client is None
    assert provider.build_calls == 1
    assert len(client.search_calls) == 1
    assert policy_enforcer.calls == ["notes.list.server"]


@pytest.mark.asyncio
async def test_service_denied_policy_does_not_build_provider_client():
    policy_enforcer = FakePolicyEnforcer.deny("wrong_source")
    provider = FakeClientProvider(FakeClient())
    service = ServerNotesWorkspaceService(
        client=None,
        client_provider=provider,
        policy_enforcer=policy_enforcer,
    )

    with pytest.raises(PolicyDeniedError):
        await service.search_server_notes(query="remote")

    assert provider.build_calls == 0
    assert policy_enforcer.calls == ["notes.list.server"]


@pytest.mark.asyncio
async def test_service_from_config_uses_shared_provider_lazily(monkeypatch):
    sentinel_client = FakeClient()
    build_client = Mock(return_value=sentinel_client)
    monkeypatch.setattr(
        "tldw_chatbook.runtime_policy.bootstrap.build_runtime_api_client_from_config",
        build_client,
    )

    service = ServerNotesWorkspaceService.from_config({"tldw_api": {"base_url": "https://example.com"}})

    assert isinstance(service, ServerNotesWorkspaceService)
    assert service.client is None
    assert service.client_provider is not None
    build_client.assert_not_called()

    result = await service.search_server_notes(query="remote")

    assert result["items"][0]["title"] == "Remote"
    assert service.client is None
    build_client.assert_called_once_with({"tldw_api": {"base_url": "https://example.com"}})


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


@pytest.mark.asyncio
async def test_server_notes_workspace_service_denies_workspace_mutation_when_source_is_wrong():
    policy_enforcer = FakePolicyEnforcer.deny("wrong_source")
    service = ServerNotesWorkspaceService(
        client=FakeClient(),
        policy_enforcer=policy_enforcer,
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await service.save_workspace_note(
            workspace_id="ws-1",
            title="Draft",
            content="Body",
        )

    assert exc.value.reason_code == "wrong_source"
    assert policy_enforcer.calls == ["notes.create.workspace"]


def test_service_server_note_update_payload_omits_keywords_when_not_supplied():
    service = ServerNotesWorkspaceService(client=None)

    payload = service.build_server_note_update_payload(title="Remote", content="Body")

    assert payload.model_dump(exclude_unset=True) == {
        "title": "Remote",
        "content": "Body",
    }


def test_service_workspace_note_update_payload_omits_unset_fields():
    service = ServerNotesWorkspaceService(client=None)

    payload = service.build_workspace_note_update_payload(version=3)

    assert payload.model_dump(exclude_unset=True) == {"version": 3}


def test_service_workspace_rename_only_update_payload_omits_unrelated_fields():
    service = ServerNotesWorkspaceService(client=None)

    payload = service.build_workspace_update_payload(name="Renamed", version=8)

    assert payload.model_dump(exclude_unset=True) == {
        "name": "Renamed",
        "version": 8,
    }


def test_service_workspace_update_payload_preserves_explicit_null_clear():
    service = ServerNotesWorkspaceService(client=None)

    payload = service.build_workspace_update_payload(
        name="Renamed",
        banner_title=None,
        version=8,
    )

    assert payload.model_dump(exclude_unset=True) == {
        "name": "Renamed",
        "banner_title": None,
        "version": 8,
    }


def test_service_workspace_update_payload_includes_audio_fields():
    service = ServerNotesWorkspaceService(client=None)

    payload = service.build_workspace_update_payload(
        name="Renamed",
        audio_provider="openai",
        audio_model="gpt-4o-mini-tts",
        audio_voice="alloy",
        audio_speed=1.25,
        version=8,
    )

    assert payload.model_dump(exclude_unset=True) == {
        "name": "Renamed",
        "audio_provider": "openai",
        "audio_model": "gpt-4o-mini-tts",
        "audio_voice": "alloy",
        "audio_speed": 1.25,
        "version": 8,
    }


def test_service_workspace_source_update_payload_omits_unset_fields():
    service = ServerNotesWorkspaceService(client=None)

    payload = service.build_workspace_source_update_payload(version=5)

    assert payload.model_dump(exclude_unset=True) == {"version": 5}


def test_service_workspace_artifact_update_payload_omits_unset_fields():
    service = ServerNotesWorkspaceService(client=None)

    payload = service.build_workspace_artifact_update_payload(version=9)

    assert payload.model_dump(exclude_unset=True) == {"version": 9}


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


def test_service_normalizes_workspace_with_audio_fields():
    service = ServerNotesWorkspaceService(client=None)

    normalized = service.normalize_workspace(
        {
            "id": "ws-1",
            "name": "Research",
            "version": 7,
            "archived": False,
            "audio_provider": "openai",
            "audio_model": "gpt-4o-mini-tts",
            "audio_voice": "alloy",
            "audio_speed": 1.25,
        }
    )

    assert normalized == {
        "id": "ws-1",
        "name": "Research",
        "archived": False,
        "study_materials_policy": "general",
        "audio_provider": "openai",
        "audio_model": "gpt-4o-mini-tts",
        "audio_voice": "alloy",
        "audio_speed": 1.25,
        "version": 7,
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
    assert context["workspace"]["audio_provider"] == "openai"
    assert context["workspace"]["audio_model"] == "gpt-4o-mini-tts"
    assert context["workspace"]["audio_voice"] == "alloy"
    assert context["workspace"]["audio_speed"] == 1.25
    assert context["notes"][0]["version"] == 3
    assert context["sources"][0]["version"] == 5
    assert context["artifacts"][0]["version"] == 9


@pytest.mark.asyncio
async def test_service_delegates_notes_graph_operations_to_server_client():
    client = FakeClient()
    service = ServerNotesWorkspaceService(client=client)

    graph = await service.get_notes_graph(center_note_id="note:123", edge_types=["manual"])
    neighbors = await service.get_note_neighbors("note:123", edge_types=["manual", "backlink"])
    created = await service.create_note_link(
        "note:123",
        to_note_id="note:456",
        directed=True,
        weight=2.5,
        metadata={"label": "related"},
    )
    deleted = await service.delete_note_link("e:1")

    assert graph["nodes"][0]["id"] == "note:123"
    assert neighbors["nodes"][0]["id"] == "note:123"
    assert created["status"] == "created"
    assert deleted["deleted"] is True
    assert client.graph_calls[0][0] == "graph"
    assert client.graph_calls[1] == ("neighbors", "note:123", {"edge_types": ["manual", "backlink"]})
    assert client.graph_calls[2][0:2] == ("create_link", "note:123")
    assert client.graph_calls[3] == ("delete_link", "e:1")


@pytest.mark.asyncio
async def test_service_enforces_notes_graph_policy_actions():
    policy_enforcer = FakePolicyEnforcer()
    service = ServerNotesWorkspaceService(
        client=FakeClient(),
        policy_enforcer=policy_enforcer,
    )

    await service.get_notes_graph(center_note_id="note:123")
    await service.get_note_neighbors("note:123")
    await service.create_note_link("note:123", to_note_id="note:456")
    await service.delete_note_link("e:1")

    assert policy_enforcer.calls == [
        "notes.graph.list.server",
        "notes.graph.detail.server",
        "notes.graph.create.server",
        "notes.graph.delete.server",
    ]
