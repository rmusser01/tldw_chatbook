from unittest.mock import Mock

import pytest

from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    LocalPromptService,
    PromptBackend,
    PromptScopeService,
    ServerPromptService,
    build_prompt_scope_service,
)
from tldw_chatbook.tldw_api.prompt_chatbook_schemas import (
    PaginatedPromptsResponse,
    PromptBriefResponse,
    PromptCollectionCreateResponse,
    PromptCollectionListResponse,
    PromptCollectionResponse,
    PromptResponse,
    PromptVersionResponse,
)


class FakePolicyEnforcer:
    def __init__(self, denied_reason=None):
        self.actions = []
        self.denied_reason = denied_reason

    @classmethod
    def deny(cls, reason="blocked"):
        return cls(denied_reason=reason)

    def require_allowed(self, *, action_id):
        self.actions.append(action_id)
        if self.denied_reason:
            raise PermissionError(self.denied_reason)


class FakeLocalPromptService:
    def __init__(self):
        self.calls = []
        self.prompt = {
            "id": 7,
            "uuid": "local-uuid-7",
            "name": "Local Prompt",
            "author": "Local Writer",
            "details": "Local details",
            "system_prompt": "Local system",
            "user_prompt": "Local user",
            "keywords": ["draft"],
            "prompt_format": "legacy",
            "version": 3,
            "deleted": False,
        }

    def list_prompts(self, *, page=1, per_page=10, include_deleted=False, **_kwargs):
        self.calls.append(("list_prompts", page, per_page, include_deleted))
        return {
            "items": [self.prompt],
            "total_pages": 1,
            "current_page": page,
            "total_items": 1,
        }

    def get_prompt(self, prompt_identifier, *, include_deleted=False):
        self.calls.append(("get_prompt", prompt_identifier, include_deleted))
        return self.prompt

    def create_prompt(self, payload):
        self.calls.append(("create_prompt", payload))
        return {**self.prompt, **payload, "id": 8, "uuid": "local-uuid-8", "version": 1}

    def update_prompt(self, prompt_identifier, payload):
        self.calls.append(("update_prompt", prompt_identifier, payload))
        return {**self.prompt, **payload}

    def delete_prompt(self, prompt_identifier):
        self.calls.append(("delete_prompt", prompt_identifier))
        return True

    def create_prompt_collection(self, payload):
        self.calls.append(("create_prompt_collection", payload))
        return {"collection_id": 3}

    def list_prompt_collections(self, *, limit=200, offset=0):
        self.calls.append(("list_prompt_collections", limit, offset))
        return {
            "collections": [
                {
                    "collection_id": 3,
                    "name": "Local Collection",
                    "description": "Offline prompts",
                    "prompt_ids": [7],
                }
            ],
            "limit": limit,
            "offset": offset,
            "total": 1,
        }

    def get_prompt_collection(self, collection_id):
        self.calls.append(("get_prompt_collection", collection_id))
        return {
            "collection_id": collection_id,
            "name": "Local Collection",
            "description": "Offline prompts",
            "prompt_ids": [7],
        }

    def update_prompt_collection(self, collection_id, payload):
        self.calls.append(("update_prompt_collection", collection_id, payload))
        return {
            "collection_id": collection_id,
            "name": payload.get("name") or "Local Collection",
            "description": payload.get("description"),
            "prompt_ids": payload.get("prompt_ids") or [],
        }


class FakeServerPromptService:
    def __init__(self):
        self.calls = []
        self.prompt = PromptResponse(
            id=9,
            uuid="server-uuid-9",
            name="Server Prompt",
            author="Server Writer",
            details="Server details",
            system_prompt="Server system",
            user_prompt="Server user",
            keywords=["remote"],
            prompt_format="structured",
            prompt_schema_version=1,
            prompt_definition={"schema_version": 1, "messages": []},
            version=5,
            usage_count=11,
            deleted=False,
        )

    async def list_prompts(self, *, page=1, per_page=10, include_deleted=False, sort_by="last_modified", sort_order="desc"):
        self.calls.append(("list_prompts", page, per_page, include_deleted, sort_by, sort_order))
        return PaginatedPromptsResponse(
            items=[
                PromptBriefResponse(
                    id=9,
                    uuid="server-uuid-9",
                    name="Server Prompt",
                    author="Server Writer",
                    usage_count=11,
                )
            ],
            total_pages=2,
            current_page=page,
            total_items=12,
        )

    async def get_prompt(self, prompt_identifier, *, include_deleted=False):
        self.calls.append(("get_prompt", prompt_identifier, include_deleted))
        return self.prompt

    async def create_prompt(self, payload):
        self.calls.append(("create_prompt", payload))
        return self.prompt.model_copy(update={"name": payload["name"]})

    async def update_prompt(self, prompt_identifier, payload):
        self.calls.append(("update_prompt", prompt_identifier, payload))
        return self.prompt.model_copy(update=payload)

    async def delete_prompt(self, prompt_identifier):
        self.calls.append(("delete_prompt", prompt_identifier))
        return {}

    async def record_prompt_usage(self, prompt_identifier):
        self.calls.append(("record_prompt_usage", prompt_identifier))
        return self.prompt

    async def list_prompt_versions(self, prompt_identifier):
        self.calls.append(("list_prompt_versions", prompt_identifier))
        return [
            PromptVersionResponse(
                version=5,
                created_at="2026-04-22T00:00:00Z",
                comment="current",
                name="Server Prompt",
            )
        ]

    async def restore_prompt_version(self, prompt_identifier, version):
        self.calls.append(("restore_prompt_version", prompt_identifier, version))
        return self.prompt.model_copy(update={"version": version})

    async def create_prompt_collection(self, payload):
        self.calls.append(("create_prompt_collection", payload))
        return PromptCollectionCreateResponse(collection_id=7)

    async def list_prompt_collections(self, *, limit=200, offset=0):
        self.calls.append(("list_prompt_collections", limit, offset))
        return PromptCollectionListResponse(
            collections=[
                PromptCollectionResponse(
                    collection_id=7,
                    name="Server Collection",
                    description="Remote prompts",
                    prompt_ids=[9],
                )
            ]
        )

    async def get_prompt_collection(self, collection_id):
        self.calls.append(("get_prompt_collection", collection_id))
        return PromptCollectionResponse(
            collection_id=collection_id,
            name="Server Collection",
            description="Remote prompts",
            prompt_ids=[9],
        )

    async def update_prompt_collection(self, collection_id, payload):
        self.calls.append(("update_prompt_collection", collection_id, payload))
        return PromptCollectionResponse(
            collection_id=collection_id,
            name=payload.get("name") or "Server Collection",
            description=payload.get("description"),
            prompt_ids=payload.get("prompt_ids") or [],
        )


@pytest.mark.asyncio
async def test_prompt_scope_lists_local_and_server_prompts_with_stable_ids():
    policy = FakePolicyEnforcer()
    local = FakeLocalPromptService()
    server = FakeServerPromptService()
    service = PromptScopeService(local_service=local, server_service=server, policy_enforcer=policy)

    local_result = await service.list_prompts(mode=PromptBackend.LOCAL, page=1, per_page=10)
    server_result = await service.list_prompts(
        mode=PromptBackend.SERVER,
        page=2,
        per_page=25,
        include_deleted=True,
        sort_by="name",
        sort_order="asc",
    )

    assert local_result["items"][0]["id"] == "local:prompt:local-uuid-7"
    assert local_result["items"][0]["backend"] == "local"
    assert server_result["items"][0]["id"] == "server:prompt:server-uuid-9"
    assert server_result["items"][0]["backend"] == "server"
    assert server_result["current_page"] == 2
    assert policy.actions == ["prompts.list.local", "prompts.list.server"]


@pytest.mark.asyncio
async def test_prompt_scope_saves_and_deletes_against_selected_backend():
    policy = FakePolicyEnforcer()
    local = FakeLocalPromptService()
    server = FakeServerPromptService()
    service = PromptScopeService(local_service=local, server_service=server, policy_enforcer=policy)

    created = await service.save_prompt(
        mode="local",
        name="New Local",
        author="Me",
        details="Details",
        system_prompt="System",
        user_prompt="User",
        keywords=["local"],
    )
    updated = await service.save_prompt(
        mode="server",
        prompt_identifier="server-uuid-9",
        name="Updated Server",
        prompt_format="structured",
        prompt_schema_version=1,
        prompt_definition={"schema_version": 1, "messages": []},
    )
    deleted = await service.delete_prompt(mode="server", prompt_identifier="server-uuid-9")

    assert created["id"] == "local:prompt:local-uuid-8"
    assert local.calls[0][0] == "create_prompt"
    assert updated["id"] == "server:prompt:server-uuid-9"
    assert updated["name"] == "Updated Server"
    assert server.calls[-2][0] == "update_prompt"
    assert deleted is True
    assert policy.actions == [
        "prompts.create.local",
        "prompts.update.server",
        "prompts.delete.server",
    ]


def test_build_prompt_scope_service_wires_local_and_server_backends(monkeypatch):
    client = object()
    calls = []

    def fake_client_builder(app_config):
        calls.append(app_config)
        return client

    monkeypatch.setattr(
        "tldw_chatbook.Prompt_Management.prompt_scope_service.build_tldw_api_client_from_config",
        fake_client_builder,
    )

    prompt_db = object()
    app_config = {"tldw_api": {"base_url": "http://server.test", "api_key": "token"}}
    service = build_prompt_scope_service(
        prompt_db=prompt_db,
        app_config=app_config,
        policy_enforcer="policy",
    )

    assert isinstance(service, PromptScopeService)
    assert isinstance(service.local_service, LocalPromptService)
    assert service.local_service.prompt_db is prompt_db
    assert isinstance(service.server_service, ServerPromptService)
    assert service.server_service.client is client
    assert service.policy_enforcer == "policy"
    assert calls == [app_config]


def test_build_prompt_scope_service_keeps_server_backend_unavailable_without_config():
    service = build_prompt_scope_service(prompt_db=None, app_config={}, policy_enforcer=None)

    assert service.local_service is None
    assert isinstance(service.server_service, ServerPromptService)
    assert service.server_service.client is None


@pytest.mark.asyncio
async def test_scope_server_prompt_service_from_config_can_use_provider_backed_client(monkeypatch):
    build_client = Mock(side_effect=AssertionError("legacy config builder should not run"))
    monkeypatch.setattr(
        "tldw_chatbook.Prompt_Management.prompt_scope_service.build_tldw_api_client_from_config",
        build_client,
    )

    class FakeClientProvider:
        def __init__(self, client):
            self.client = client
            self.build_calls = 0

        def build_client(self):
            self.build_calls += 1
            return self.client

    provider = FakeClientProvider(FakeServerPromptService())
    service = ServerPromptService.from_config(
        {"tldw_api": {"base_url": "https://example.com"}},
        client_provider=provider,
    )

    result = await service.list_prompts(page=2, per_page=3)

    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 1
    assert result.items[0].id == 9


@pytest.mark.asyncio
async def test_prompt_scope_routes_server_usage_versions_and_restore():
    policy = FakePolicyEnforcer()
    server = FakeServerPromptService()
    service = PromptScopeService(
        local_service=FakeLocalPromptService(),
        server_service=server,
        policy_enforcer=policy,
    )

    used = await service.record_prompt_usage(mode="server", prompt_identifier="server-uuid-9")
    versions = await service.list_prompt_versions(mode="server", prompt_identifier="server-uuid-9")
    restored = await service.restore_prompt_version(
        mode="server",
        prompt_identifier="server-uuid-9",
        version=3,
    )

    assert used["id"] == "server:prompt:server-uuid-9"
    assert versions == [
        {
            "backend": "server",
            "version": 5,
            "created_at": "2026-04-22T00:00:00Z",
            "comment": "current",
            "name": "Server Prompt",
            "author": None,
            "details": None,
            "system_prompt": None,
            "user_prompt": None,
            "prompt_uuid": None,
            "prompt_format": "legacy",
            "prompt_schema_version": None,
            "prompt_definition": None,
        }
    ]
    assert restored["version"] == 3
    assert server.calls[-3:] == [
        ("record_prompt_usage", "server-uuid-9"),
        ("list_prompt_versions", "server-uuid-9"),
        ("restore_prompt_version", "server-uuid-9", 3),
    ]
    assert policy.actions[-3:] == [
        "prompts.use.server",
        "prompts.versions.server",
        "prompts.restore_version.server",
    ]


@pytest.mark.asyncio
async def test_prompt_scope_rejects_local_version_history_until_supported():
    service = PromptScopeService(
        local_service=FakeLocalPromptService(),
        server_service=FakeServerPromptService(),
    )

    with pytest.raises(ValueError, match="Local prompt version history is unavailable"):
        await service.list_prompt_versions(mode="local", prompt_identifier="local-uuid-7")

    with pytest.raises(ValueError, match="Local prompt version restore is unavailable"):
        await service.restore_prompt_version(
            mode="local",
            prompt_identifier="local-uuid-7",
            version=1,
        )


@pytest.mark.asyncio
async def test_prompt_scope_routes_server_prompt_collections_with_policy():
    policy = FakePolicyEnforcer()
    server = FakeServerPromptService()
    service = PromptScopeService(
        local_service=FakeLocalPromptService(),
        server_service=server,
        policy_enforcer=policy,
    )

    created = await service.create_prompt_collection(
        mode="server",
        name="Server Collection",
        description="Remote prompts",
        prompt_ids=[9],
    )
    listed = await service.list_prompt_collections(mode="server", limit=50, offset=5)
    fetched = await service.get_prompt_collection(mode="server", collection_id=7)
    updated = await service.update_prompt_collection(
        mode="server",
        collection_id=7,
        name="Renamed",
        description="Updated",
        prompt_ids=[9, 10],
    )

    assert created == {"id": "server:prompt_collection:7", "backend": "server", "collection_id": 7}
    assert listed["collections"][0]["id"] == "server:prompt_collection:7"
    assert fetched["name"] == "Server Collection"
    assert updated["name"] == "Renamed"
    assert server.calls[-4:] == [
        ("create_prompt_collection", {"name": "Server Collection", "description": "Remote prompts", "prompt_ids": [9]}),
        ("list_prompt_collections", 50, 5),
        ("get_prompt_collection", 7),
        ("update_prompt_collection", 7, {"name": "Renamed", "description": "Updated", "prompt_ids": [9, 10]}),
    ]
    assert policy.actions[-4:] == [
        "prompts.collections.create.server",
        "prompts.collections.list.server",
        "prompts.collections.detail.server",
        "prompts.collections.update.server",
    ]


@pytest.mark.asyncio
async def test_prompt_scope_routes_local_prompt_collections_with_policy():
    policy = FakePolicyEnforcer()
    local = FakeLocalPromptService()
    service = PromptScopeService(
        local_service=local,
        server_service=FakeServerPromptService(),
        policy_enforcer=policy,
    )

    created = await service.create_prompt_collection(
        mode="local",
        name="Local Collection",
        description="Offline prompts",
        prompt_ids=[7],
    )
    listed = await service.list_prompt_collections(mode="local", limit=50, offset=5)
    fetched = await service.get_prompt_collection(mode="local", collection_id=3)
    updated = await service.update_prompt_collection(
        mode="local",
        collection_id=3,
        name="Renamed",
        description="Updated",
        prompt_ids=[7, 8],
    )

    assert created == {"id": "local:prompt_collection:3", "backend": "local", "collection_id": 3}
    assert listed["collections"][0]["id"] == "local:prompt_collection:3"
    assert fetched["name"] == "Local Collection"
    assert updated["name"] == "Renamed"
    assert local.calls[-4:] == [
        ("create_prompt_collection", {"name": "Local Collection", "description": "Offline prompts", "prompt_ids": [7]}),
        ("list_prompt_collections", 50, 5),
        ("get_prompt_collection", 3),
        ("update_prompt_collection", 3, {"name": "Renamed", "description": "Updated", "prompt_ids": [7, 8]}),
    ]
    assert policy.actions[-4:] == [
        "prompts.collections.create.local",
        "prompts.collections.list.local",
        "prompts.collections.detail.local",
        "prompts.collections.update.local",
    ]


def test_local_prompt_service_persists_prompt_collections(tmp_path):
    prompt_db = PromptsDatabase(tmp_path / "prompts.db", client_id="test_client")
    prompt_id, _prompt_uuid, _ = prompt_db.add_prompt(
        name="Local Prompt",
        author="Writer",
        details="Details",
        system_prompt="System",
        user_prompt="User",
        keywords=["draft"],
        overwrite=False,
    )
    second_prompt_id, _second_prompt_uuid, _ = prompt_db.add_prompt(
        name="Second Local Prompt",
        author="Writer",
        details="More details",
        system_prompt="Second system",
        user_prompt="Second user",
        keywords=["draft"],
        overwrite=False,
    )
    service = LocalPromptService(prompt_db)

    created = service.create_prompt_collection(
        {
            "name": "Local Collection",
            "description": "Offline prompts",
            "prompt_ids": [prompt_id],
        }
    )
    listed = service.list_prompt_collections(limit=10, offset=0)
    fetched = service.get_prompt_collection(created["collection_id"])
    updated = service.update_prompt_collection(
        created["collection_id"],
        {
            "name": "Renamed",
            "description": "Updated",
            "prompt_ids": [prompt_id],
        },
    )
    membership_updated = service.update_prompt_collection(
        created["collection_id"],
        {"prompt_ids": [second_prompt_id]},
    )

    assert listed["total"] == 1
    assert fetched["prompt_ids"] == [prompt_id]
    assert updated["name"] == "Renamed"
    assert updated["prompt_ids"] == [prompt_id]
    assert membership_updated["name"] == "Renamed"
    assert membership_updated["prompt_ids"] == [second_prompt_id]
