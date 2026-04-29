import inspect
from unittest.mock import Mock

import pytest

import tldw_chatbook.Prompt_Management.server_prompt_service as server_prompt_module
from tldw_chatbook.Prompt_Management.server_prompt_service import ServerPromptService
from tldw_chatbook.runtime_policy import PolicyDecision, PolicyDeniedError


class FakePromptClient:
    def __init__(self):
        self.calls = []

    async def list_prompts(self, **kwargs):
        self.calls.append(("list_prompts", kwargs))
        return {"prompts": []}

    async def create_prompt(self, request_data):
        self.calls.append(("create_prompt", request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": 7, "name": request_data.name}

    async def preview_prompt(self, request_data):
        self.calls.append(("preview_prompt", request_data.model_dump(exclude_none=True, mode="json")))
        return {"rendered": "Hello Ada"}

    async def update_prompt(self, prompt_id, request_data):
        self.calls.append(("update_prompt", prompt_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": prompt_id, "name": request_data.name}

    async def delete_prompt(self, prompt_id):
        self.calls.append(("delete_prompt", prompt_id))
        return {"deleted": True}

    async def list_prompt_versions(self, prompt_id):
        self.calls.append(("list_prompt_versions", prompt_id))
        return [{"version": 3, "prompt_uuid": "prompt-uuid"}]

    async def restore_prompt_version(self, prompt_id, version):
        self.calls.append(("restore_prompt_version", prompt_id, version))
        return {"id": prompt_id, "uuid": "prompt-uuid", "version": version, "name": "Restored"}

    async def get_prompts_health(self):
        self.calls.append(("get_prompts_health",))
        return {"status": "healthy"}

    async def get_prompt_sync_log(self, **kwargs):
        self.calls.append(("get_prompt_sync_log", kwargs))
        return {"changes": []}

    async def search_prompts(self, **kwargs):
        self.calls.append(("search_prompts", kwargs))
        return {"items": []}

    async def create_prompt_keyword(self, keyword_text):
        self.calls.append(("create_prompt_keyword", keyword_text))
        return {"keyword_text": keyword_text}

    async def list_prompt_keywords(self):
        self.calls.append(("list_prompt_keywords",))
        return ["drafting"]

    async def delete_prompt_keyword(self, keyword_text):
        self.calls.append(("delete_prompt_keyword", keyword_text))
        return None

    async def export_prompts(self, **kwargs):
        self.calls.append(("export_prompts", kwargs))
        return {"message": "exported"}

    async def export_prompt_keywords(self):
        self.calls.append(("export_prompt_keywords",))
        return {"message": "exported"}

    async def import_prompts(self, payload):
        self.calls.append(("import_prompts", payload))
        return {"imported": 1}

    async def extract_prompt_template_variables(self, template):
        self.calls.append(("extract_prompt_template_variables", template))
        return {"variables": ["name"]}

    async def render_prompt_template(self, template, variables):
        self.calls.append(("render_prompt_template", template, variables))
        return {"rendered": "Hello Ada"}

    async def convert_prompt(self, payload):
        self.calls.append(("convert_prompt", payload))
        return {"prompt_definition": {"blocks": []}}

    async def bulk_delete_prompts(self, prompt_ids):
        self.calls.append(("bulk_delete_prompts", prompt_ids))
        return {"deleted": len(prompt_ids)}

    async def bulk_update_prompt_keywords(self, prompt_ids, keywords, mode="add"):
        self.calls.append(("bulk_update_prompt_keywords", prompt_ids, keywords, mode))
        return {"updated": len(prompt_ids)}

    async def record_prompt_usage(self, prompt_identifier):
        self.calls.append(("record_prompt_usage", prompt_identifier))
        return {"usage_count": 1}

    async def create_prompt_collection(self, **kwargs):
        self.calls.append(("create_prompt_collection", kwargs))
        return {"collection_id": 3}

    async def list_prompt_collections(self, **kwargs):
        self.calls.append(("list_prompt_collections", kwargs))
        return {"collections": []}

    async def get_prompt_collection(self, collection_id):
        self.calls.append(("get_prompt_collection", collection_id))
        return {"collection_id": collection_id}

    async def update_prompt_collection(self, collection_id, **kwargs):
        self.calls.append(("update_prompt_collection", collection_id, kwargs))
        return {"collection_id": collection_id, **kwargs}


class FakeClientProvider:
    def __init__(self, client):
        self.client = client
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        return self.client


class ExplodingClientProvider:
    def __init__(self):
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        raise AssertionError("provider should not build a client")


def test_server_prompt_service_module_does_not_reference_legacy_config_client_builders():
    source = inspect.getsource(server_prompt_module)

    assert "build_runtime_api_client_from_config" not in source
    assert "build_runtime_api_client(app_config" not in source


@pytest.mark.asyncio
async def test_server_prompt_service_from_config_uses_shared_provider_lazily(monkeypatch):
    sentinel_client = FakePromptClient()
    direct_builder = Mock(side_effect=AssertionError("service should not call direct legacy builder"))
    provider_builder = Mock(return_value=sentinel_client)
    monkeypatch.setattr("tldw_chatbook.runtime_policy.bootstrap.build_runtime_api_client", direct_builder)
    monkeypatch.setattr(
        "tldw_chatbook.runtime_policy.bootstrap.build_runtime_api_client_from_config",
        provider_builder,
    )

    service = ServerPromptService.from_config({"tldw_api": {"base_url": "https://example.com"}})

    assert isinstance(service, ServerPromptService)
    assert service.client is None
    assert service.client_provider is not None
    direct_builder.assert_not_called()
    provider_builder.assert_not_called()

    result = await service.get_prompts_health()

    assert result == {"status": "healthy"}
    assert service.client is None
    provider_builder.assert_called_once_with({"tldw_api": {"base_url": "https://example.com"}})


@pytest.mark.asyncio
async def test_server_prompt_service_from_config_can_use_provider_backed_client(monkeypatch):
    build_client = Mock(side_effect=AssertionError("legacy config builder should not run"))
    monkeypatch.setattr(
        "tldw_chatbook.runtime_policy.bootstrap.build_runtime_api_client",
        build_client,
    )
    client = FakePromptClient()
    provider = FakeClientProvider(client)
    policy = Mock()

    service = ServerPromptService.from_config(
        {"tldw_api": {"base_url": "https://example.com"}},
        client_provider=provider,
        policy_enforcer=policy,
    )

    result = await service.get_prompts_health()

    assert result == {"status": "healthy"}
    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 1
    policy.require_allowed.assert_called_once_with(action_id="prompts.health.detail.server")


@pytest.mark.asyncio
async def test_server_prompt_service_direct_client_takes_precedence_over_provider():
    client = FakePromptClient()
    provider = ExplodingClientProvider()
    service = ServerPromptService(client=client, client_provider=provider)

    result = await service.get_prompts_health()

    assert result == {"status": "healthy"}
    assert provider.build_calls == 0
    assert client.calls == [("get_prompts_health",)]


@pytest.mark.asyncio
async def test_server_prompt_service_enforces_policy_actions():
    client = FakePromptClient()
    policy = Mock()
    service = ServerPromptService(client=client, policy_enforcer=policy)

    await service.list_prompts()
    await service.create_prompt(name="Greeting", user_prompt="Hello {{name}}")
    await service.preview_prompt(name="Greeting", user_prompt="Hello {{name}}")
    await service.update_prompt(7, name="Greeting", user_prompt="Hello {{name}}")
    await service.delete_prompt(7)

    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "prompts.list.server",
        "prompts.create.server",
        "prompts.preview.server",
        "prompts.update.server",
        "prompts.delete.server",
    ]


@pytest.mark.asyncio
async def test_server_prompt_service_routes_prompt_version_controls():
    client = FakePromptClient()
    policy = Mock()
    service = ServerPromptService(client=client, policy_enforcer=policy)

    versions = await service.list_prompt_versions("prompt-uuid")
    restored = await service.restore_prompt_version("prompt-uuid", 3)

    assert versions == [{"version": 3, "prompt_uuid": "prompt-uuid"}]
    assert restored["name"] == "Restored"
    assert client.calls[-2:] == [
        ("list_prompt_versions", "prompt-uuid"),
        ("restore_prompt_version", "prompt-uuid", 3),
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list][-2:] == [
        "prompts.versions.list.server",
        "prompts.versions.restore.server",
    ]


@pytest.mark.asyncio
async def test_server_prompt_service_routes_server_prompt_utility_surfaces():
    client = FakePromptClient()
    policy = Mock()
    service = ServerPromptService(client=client, policy_enforcer=policy)

    await service.get_prompts_health()
    await service.get_prompt_sync_log(since_change_id=5, limit=25)
    await service.search_prompts(search_query="rag", search_fields=["name"], page=2)
    await service.create_prompt_keyword("Drafting")
    await service.list_prompt_keywords()
    await service.delete_prompt_keyword("Drafting")
    await service.export_prompts(export_format="markdown", filter_keywords=["drafting"])
    await service.export_prompt_keywords()
    await service.import_prompts({"prompts": [{"name": "Draft", "content": "Body"}]})
    await service.extract_prompt_template_variables("Hello {{name}}")
    await service.render_prompt_template("Hello {{name}}", {"name": "Ada"})
    await service.convert_prompt({"system_prompt": "S", "user_prompt": "U"})
    await service.bulk_delete_prompts([1])
    await service.bulk_update_prompt_keywords([1], ["drafting"], mode="replace")
    await service.record_prompt_usage("prompt-1")
    await service.create_prompt_collection(name="Pack", prompt_ids=[1])
    await service.list_prompt_collections(limit=25)
    await service.get_prompt_collection(7)
    await service.update_prompt_collection(7, name="Updated")

    assert client.calls[-19:] == [
        ("get_prompts_health",),
        ("get_prompt_sync_log", {"since_change_id": 5, "limit": 25}),
        ("search_prompts", {"search_query": "rag", "search_fields": ["name"], "page": 2}),
        ("create_prompt_keyword", "Drafting"),
        ("list_prompt_keywords",),
        ("delete_prompt_keyword", "Drafting"),
        ("export_prompts", {"export_format": "markdown", "filter_keywords": ["drafting"]}),
        ("export_prompt_keywords",),
        ("import_prompts", {"prompts": [{"name": "Draft", "content": "Body"}]}),
        ("extract_prompt_template_variables", "Hello {{name}}"),
        ("render_prompt_template", "Hello {{name}}", {"name": "Ada"}),
        ("convert_prompt", {"system_prompt": "S", "user_prompt": "U"}),
        ("bulk_delete_prompts", [1]),
        ("bulk_update_prompt_keywords", [1], ["drafting"], "replace"),
        ("record_prompt_usage", "prompt-1"),
        ("create_prompt_collection", {"name": "Pack", "prompt_ids": [1]}),
        ("list_prompt_collections", {"limit": 25}),
        ("get_prompt_collection", 7),
        ("update_prompt_collection", 7, {"name": "Updated"}),
    ]
    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list][-19:] == [
        "prompts.health.detail.server",
        "prompts.sync_log.list.server",
        "prompts.search.list.server",
        "prompts.keywords.create.server",
        "prompts.keywords.list.server",
        "prompts.keywords.delete.server",
        "prompts.transfer.export.server",
        "prompts.keywords.export.server",
        "prompts.transfer.import.server",
        "prompts.templates.process.server",
        "prompts.templates.process.server",
        "prompts.templates.process.server",
        "prompts.bulk.delete.server",
        "prompts.bulk.update.server",
        "prompts.usage.update.server",
        "prompts.collections.create.server",
        "prompts.collections.list.server",
        "prompts.collections.detail.server",
        "prompts.collections.update.server",
    ]


@pytest.mark.asyncio
async def test_server_prompt_service_hard_stops_denied_ui_policy_decision():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="server_unreachable",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakePromptClient()
    service = ServerPromptService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_prompts()

    assert exc.value.reason_code == "server_unreachable"
    assert client.calls == []
