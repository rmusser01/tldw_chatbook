import pytest

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
    PromptResponse,
    PromptVersionResponse,
)


class FakePolicyEnforcer:
    def __init__(self):
        self.actions = []

    def require_allowed(self, *, action_id):
        self.actions.append(action_id)


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
