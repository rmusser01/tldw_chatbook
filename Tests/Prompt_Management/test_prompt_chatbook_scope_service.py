import pytest

from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    ServerPromptService as ScopedServerPromptService,
    build_prompt_scope_service,
)
from tldw_chatbook.Prompt_Management.prompt_chatbook_scope_service import PromptChatbookScopeService
from tldw_chatbook.Prompt_Management.server_prompt_service import ServerPromptService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakePromptBackend:
    def __init__(self, source):
        self.source = source
        self.calls = []

    async def list_prompts(self, *, include_deleted=False, limit=100, offset=0):
        self.calls.append(("list_prompts", include_deleted, limit, offset))
        return [{"id": f"{self.source}-prompt-1", "name": "Prompt", "prompt_format": "legacy"}]

    async def create_prompt(self, **kwargs):
        self.calls.append(("create_prompt", kwargs))
        return {"id": f"{self.source}-prompt-1", "name": kwargs["name"], "prompt_format": kwargs["prompt_format"]}

    async def preview_prompt(self, **kwargs):
        self.calls.append(("preview_prompt", kwargs))
        return {"name": kwargs["name"], "preview": "rendered"}

    async def update_prompt(self, prompt_id, **kwargs):
        self.calls.append(("update_prompt", prompt_id, kwargs))
        return {"id": prompt_id, "name": kwargs["name"], "prompt_format": "legacy"}

    async def delete_prompt(self, prompt_id):
        self.calls.append(("delete_prompt", prompt_id))
        return True

    async def list_prompt_versions(self, prompt_id):
        self.calls.append(("list_prompt_versions", prompt_id))
        return [{"version": 2, "prompt_uuid": f"{self.source}-prompt-1"}]

    async def restore_prompt_version(self, prompt_id, version):
        self.calls.append(("restore_prompt_version", prompt_id, version))
        return {"id": prompt_id, "name": "Restored", "version": version}

    async def get_prompts_health(self):
        self.calls.append(("get_prompts_health",))
        return {"status": "healthy"}

    async def get_prompt_sync_log(self, **kwargs):
        self.calls.append(("get_prompt_sync_log", kwargs))
        return {"changes": []}

    async def search_prompts(self, **kwargs):
        self.calls.append(("search_prompts", kwargs))
        return {"items": [{"id": f"{self.source}-prompt-1", "name": "Prompt"}]}

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
        return {"collection_id": f"{self.source}-collection-1", **kwargs}

    async def list_prompt_collections(self, **kwargs):
        self.calls.append(("list_prompt_collections", kwargs))
        return {"collections": [{"collection_id": f"{self.source}-collection-1", "name": "Pack"}]}

    async def get_prompt_collection(self, collection_id):
        self.calls.append(("get_prompt_collection", collection_id))
        return {"collection_id": collection_id, "name": "Pack"}

    async def update_prompt_collection(self, collection_id, **kwargs):
        self.calls.append(("update_prompt_collection", collection_id, kwargs))
        return {"collection_id": collection_id, **kwargs}


class FakeChatbookBackend:
    def __init__(self, source):
        self.source = source
        self.calls = []

    async def preview_chatbook(self, chatbook_file_path):
        self.calls.append(("preview_chatbook", chatbook_file_path))
        return {"success": True, "manifest": {"name": "Pack"}}

    async def export_chatbook(self, request_data):
        self.calls.append(("export_chatbook", request_data))
        return {"job_id": f"{self.source}-export-1", "status": "queued"}

    async def continue_chatbook_export(self, request_data):
        self.calls.append(("continue_chatbook_export", request_data))
        return {"job_id": f"{self.source}-continue-1", "status": "queued"}

    async def import_chatbook(self, chatbook_file_path, request_data):
        self.calls.append(("import_chatbook", chatbook_file_path, request_data))
        return {"job_id": f"{self.source}-import-1", "status": "queued"}

    async def get_export_job(self, job_id):
        self.calls.append(("get_export_job", job_id))
        return {"job_id": job_id, "status": "completed"}

    async def download_export(self, job_id, **kwargs):
        self.calls.append(("download_export", job_id, kwargs))
        return {
            "job_id": job_id,
            "content": b"chatbook-bytes",
            "content_type": "application/zip",
            "filename": "pack.chatbook.zip",
        }

    async def get_import_job(self, job_id):
        self.calls.append(("get_import_job", job_id))
        return {"job_id": job_id, "status": "completed"}

    async def list_export_jobs(self, **kwargs):
        self.calls.append(("list_export_jobs", kwargs))
        return {"items": [{"job_id": f"{self.source}-export-1", "status": "completed"}]}

    async def list_import_jobs(self, **kwargs):
        self.calls.append(("list_import_jobs", kwargs))
        return {"items": [{"job_id": f"{self.source}-import-1", "status": "completed"}]}

    async def cancel_export_job(self, job_id):
        self.calls.append(("cancel_export_job", job_id))
        return {"job_id": job_id, "cancelled": True}

    async def cancel_import_job(self, job_id):
        self.calls.append(("cancel_import_job", job_id))
        return {"job_id": job_id, "cancelled": True}

    async def remove_export_job(self, job_id):
        self.calls.append(("remove_export_job", job_id))
        return {"job_id": job_id, "removed": True}

    async def remove_import_job(self, job_id):
        self.calls.append(("remove_import_job", job_id))
        return {"job_id": job_id, "removed": True}


class FakeChatbookCrudBackend(FakeChatbookBackend):
    async def list_chatbooks(self, **kwargs):
        self.calls.append(("list_chatbooks", kwargs))
        return [{"id": f"{self.source}-chatbook-1", "name": "Pack"}]

    async def get_chatbook(self, chatbook_id):
        self.calls.append(("get_chatbook", chatbook_id))
        return {"id": chatbook_id, "name": "Pack"}

    async def create_chatbook(self, **kwargs):
        self.calls.append(("create_chatbook", kwargs))
        return {"id": f"{self.source}-chatbook-2", **kwargs}

    async def update_chatbook(self, chatbook_id, **kwargs):
        self.calls.append(("update_chatbook", chatbook_id, kwargs))
        return {"id": chatbook_id, **kwargs}

    async def delete_chatbook(self, chatbook_id):
        self.calls.append(("delete_chatbook", chatbook_id))
        return True


class FakePolicyEnforcer:
    def __init__(self, denied_reason=None):
        self.denied_reason = denied_reason
        self.calls = []

    def require_allowed(self, *, action_id):
        self.calls.append(action_id)
        if self.denied_reason:
            raise PolicyDeniedError(
                action_id=action_id,
                reason_code=self.denied_reason,
                user_message=f"{action_id} denied",
                effective_source="server",
                authority_owner="server",
            )


class FakeClientProvider:
    def __init__(self, client):
        self.client = client
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        return self.client


@pytest.mark.asyncio
async def test_server_prompt_service_uses_provider_client_when_no_direct_client():
    class FakeClient:
        async def get_prompts_health(self):
            return {"status": "healthy"}

    provider = FakeClientProvider(FakeClient())
    service = ServerPromptService.from_server_context_provider(provider)

    result = await service.get_prompts_health()

    assert result == {"status": "healthy"}
    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 1


@pytest.mark.asyncio
async def test_server_prompt_service_prefers_direct_client_over_provider():
    class FakeClient:
        def __init__(self, status):
            self.status = status

        async def get_prompts_health(self):
            return {"status": self.status}

    provider = FakeClientProvider(FakeClient("provider"))
    service = ServerPromptService(client=FakeClient("direct"), client_provider=provider)

    result = await service.get_prompts_health()

    assert result == {"status": "direct"}
    assert provider.build_calls == 0


@pytest.mark.asyncio
async def test_build_prompt_scope_service_uses_injected_server_service_without_from_config(monkeypatch):
    class FakeScopedServerPrompts:
        def __init__(self):
            self.calls = []

        async def list_prompts(self, **kwargs):
            self.calls.append(("list_prompts", kwargs))
            return {
                "items": [{"id": "injected-prompt-1", "name": "Injected"}],
                "page": kwargs["page"],
                "per_page": kwargs["per_page"],
                "total_items": 1,
            }

    server_prompts = FakeScopedServerPrompts()

    def fail_from_config(_app_config):
        raise AssertionError("from_config should not be used for injected server services")

    monkeypatch.setattr(ScopedServerPromptService, "from_config", fail_from_config)
    scope = build_prompt_scope_service(
        prompt_db=None,
        app_config={"tldw_api": {"base_url": "https://unused.invalid"}},
        server_service=server_prompts,
    )

    result = await scope.list_prompts(mode="server")

    assert result["items"][0]["id"] == "server:prompt:injected-prompt-1"
    assert server_prompts.calls == [
        (
            "list_prompts",
            {
                "page": 1,
                "per_page": 10,
                "include_deleted": False,
                "sort_by": "last_modified",
                "sort_order": "desc",
            },
        ),
    ]


@pytest.mark.asyncio
async def test_build_prompt_scope_service_uses_provider_backed_server_service_without_from_config(monkeypatch):
    class FakeClient:
        async def list_prompts(self, **kwargs):
            return {
                "items": [{"id": "provider-prompt-1", "name": "Provider"}],
                "page": kwargs["page"],
                "per_page": kwargs["per_page"],
                "total": 1,
            }

    def fail_from_config(_app_config):
        raise AssertionError("from_config should not be used when a client provider is supplied")

    provider = FakeClientProvider(FakeClient())
    monkeypatch.setattr(ScopedServerPromptService, "from_config", fail_from_config)
    scope = build_prompt_scope_service(prompt_db=None, client_provider=provider)

    result = await scope.list_prompts(mode="server", page=2, per_page=3)

    assert result["items"][0]["id"] == "server:prompt:provider-prompt-1"
    assert provider.build_calls == 1


@pytest.mark.asyncio
async def test_prompt_chatbook_scope_service_routes_prompts_and_policy_actions():
    local_prompts = FakePromptBackend("local")
    server_prompts = FakePromptBackend("server")
    policy = FakePolicyEnforcer()
    scope = PromptChatbookScopeService(
        local_prompt_service=local_prompts,
        server_prompt_service=server_prompts,
        local_chatbook_service=FakeChatbookBackend("local"),
        server_chatbook_service=FakeChatbookBackend("server"),
        policy_enforcer=policy,
    )

    local_list = await scope.list_prompts(mode="local", include_deleted=True, limit=25)
    created = await scope.create_prompt(
        mode="server",
        name="Structured",
        prompt_format="structured",
        prompt_definition={"blocks": []},
    )
    preview = await scope.preview_prompt(mode="server", name="Structured")
    updated = await scope.update_prompt(mode="server", prompt_id="server-prompt-1", name="Updated")
    deleted = await scope.delete_prompt(mode="local", prompt_id="local-prompt-1")

    assert local_list[0]["record_id"] == "local:prompt:local-prompt-1"
    assert created["record_id"] == "server:prompt:server-prompt-1"
    assert preview["preview"] == "rendered"
    assert updated["name"] == "Updated"
    assert deleted is True
    assert policy.calls == [
        "prompts.list.local",
        "prompts.create.server",
        "prompts.preview.server",
        "prompts.update.server",
        "prompts.delete.local",
    ]


@pytest.mark.asyncio
async def test_prompt_chatbook_scope_service_routes_chatbooks_and_policy_actions():
    server_chatbooks = FakeChatbookBackend("server")
    policy = FakePolicyEnforcer()
    scope = PromptChatbookScopeService(
        local_prompt_service=FakePromptBackend("local"),
        server_prompt_service=FakePromptBackend("server"),
        local_chatbook_service=FakeChatbookBackend("local"),
        server_chatbook_service=server_chatbooks,
        policy_enforcer=policy,
    )

    preview = await scope.preview_chatbook(mode="server", chatbook_file_path="/tmp/demo.chatbook.zip")
    export = await scope.export_chatbook(mode="server", request_data={"name": "Pack"})
    continued = await scope.continue_chatbook_export(
        mode="server",
        request_data={
            "export_id": "exp-1",
            "continuations": [{"type": "evaluation"}],
        },
    )
    imported = await scope.import_chatbook(
        mode="server",
        chatbook_file_path="/tmp/demo.chatbook.zip",
        request_data={"content_selections": {"conversation": ["1"]}},
    )
    export_job = await scope.get_export_job(mode="server", job_id=export["job_id"])
    import_job = await scope.get_import_job(mode="server", job_id=imported["job_id"])

    assert preview["record_id"] == "server:chatbook:Pack"
    assert export["record_id"] == "server:chatbook_job:server-export-1"
    assert continued["record_id"] == "server:chatbook_job:server-continue-1"
    assert imported["record_id"] == "server:chatbook_job:server-import-1"
    assert export_job["status"] == "completed"
    assert import_job["status"] == "completed"
    assert policy.calls == [
        "chatbooks.detail.server",
        "chatbooks.export.server",
        "chatbooks.export.server",
        "chatbooks.import.server",
        "chatbooks.detail.server",
        "chatbooks.detail.server",
    ]


@pytest.mark.asyncio
async def test_prompt_chatbook_scope_service_routes_chatbook_job_management():
    server_chatbooks = FakeChatbookBackend("server")
    policy = FakePolicyEnforcer()
    scope = PromptChatbookScopeService(
        local_prompt_service=FakePromptBackend("local"),
        server_prompt_service=FakePromptBackend("server"),
        local_chatbook_service=FakeChatbookBackend("local"),
        server_chatbook_service=server_chatbooks,
        policy_enforcer=policy,
    )

    export_jobs = await scope.list_export_jobs(mode="server", limit=25, offset=5)
    import_jobs = await scope.list_import_jobs(mode="server", limit=10, offset=2)
    downloaded = await scope.download_export(mode="server", job_id="server-export-1", token="signed", exp=12345)
    cancelled_export = await scope.cancel_export_job(mode="server", job_id="server-export-1")
    cancelled_import = await scope.cancel_import_job(mode="server", job_id="server-import-1")
    removed_export = await scope.remove_export_job(mode="server", job_id="server-export-1")
    removed_import = await scope.remove_import_job(mode="server", job_id="server-import-1")

    assert export_jobs["items"][0]["record_id"] == "server:chatbook_job:server-export-1"
    assert import_jobs["items"][0]["record_id"] == "server:chatbook_job:server-import-1"
    assert downloaded["record_id"] == "server:chatbook_export:server-export-1"
    assert downloaded["content"] == b"chatbook-bytes"
    assert downloaded["filename"] == "pack.chatbook.zip"
    assert cancelled_export["record_id"] == "server:chatbook_job:server-export-1"
    assert cancelled_import["record_id"] == "server:chatbook_job:server-import-1"
    assert removed_export["removed"] is True
    assert removed_import["removed"] is True
    assert policy.calls == [
        "chatbooks.list.server",
        "chatbooks.list.server",
        "chatbooks.export.server",
        "chatbooks.update.server",
        "chatbooks.update.server",
        "chatbooks.delete.server",
        "chatbooks.delete.server",
    ]


@pytest.mark.asyncio
async def test_prompt_chatbook_scope_service_routes_server_prompt_version_controls():
    server_prompts = FakePromptBackend("server")
    policy = FakePolicyEnforcer()
    scope = PromptChatbookScopeService(
        local_prompt_service=FakePromptBackend("local"),
        server_prompt_service=server_prompts,
        local_chatbook_service=FakeChatbookBackend("local"),
        server_chatbook_service=FakeChatbookBackend("server"),
        policy_enforcer=policy,
    )

    versions = await scope.list_prompt_versions(mode="server", prompt_id="server-prompt-1")
    restored = await scope.restore_prompt_version(mode="server", prompt_id="server-prompt-1", version=2)

    assert versions[0]["record_type"] == "prompt_version"
    assert versions[0]["record_id"] == "server:prompt_version:2"
    assert restored["record_id"] == "server:prompt:server-prompt-1"
    assert server_prompts.calls[-2:] == [
        ("list_prompt_versions", "server-prompt-1"),
        ("restore_prompt_version", "server-prompt-1", 2),
    ]
    assert policy.calls == [
        "prompts.versions.list.server",
        "prompts.versions.restore.server",
    ]


@pytest.mark.asyncio
async def test_prompt_chatbook_scope_service_routes_local_prompt_version_controls():
    local_prompts = FakePromptBackend("local")
    policy = FakePolicyEnforcer()
    scope = PromptChatbookScopeService(
        local_prompt_service=local_prompts,
        server_prompt_service=FakePromptBackend("server"),
        local_chatbook_service=FakeChatbookBackend("local"),
        server_chatbook_service=FakeChatbookBackend("server"),
        policy_enforcer=policy,
    )

    versions = await scope.list_prompt_versions(mode="local", prompt_id="local-prompt-1")
    restored = await scope.restore_prompt_version(mode="local", prompt_id="local-prompt-1", version=2)

    assert versions[0]["record_id"] == "local:prompt_version:2"
    assert restored["record_id"] == "local:prompt:local-prompt-1"
    assert local_prompts.calls[-2:] == [
        ("list_prompt_versions", "local-prompt-1"),
        ("restore_prompt_version", "local-prompt-1", 2),
    ]
    assert policy.calls == [
        "prompts.versions.list.local",
        "prompts.versions.restore.local",
    ]


@pytest.mark.asyncio
async def test_prompt_chatbook_scope_service_routes_server_prompt_utility_surfaces():
    server_prompts = FakePromptBackend("server")
    policy = FakePolicyEnforcer()
    scope = PromptChatbookScopeService(
        local_prompt_service=FakePromptBackend("local"),
        server_prompt_service=server_prompts,
        local_chatbook_service=FakeChatbookBackend("local"),
        server_chatbook_service=FakeChatbookBackend("server"),
        policy_enforcer=policy,
    )

    health = await scope.get_prompts_health(mode="server")
    sync_log = await scope.get_prompt_sync_log(mode="server", since_change_id=5, limit=25)
    searched = await scope.search_prompts(mode="server", search_query="rag")
    keyword = await scope.create_prompt_keyword(mode="server", keyword_text="Drafting")
    keywords = await scope.list_prompt_keywords(mode="server")
    deleted_keyword = await scope.delete_prompt_keyword(mode="server", keyword_text="Drafting")
    exported = await scope.export_prompts(mode="server", export_format="markdown")
    keyword_export = await scope.export_prompt_keywords(mode="server")
    imported = await scope.import_prompts(mode="server", payload={"prompts": [{"name": "Draft", "content": "Body"}]})
    variables = await scope.extract_prompt_template_variables(mode="server", template="Hello {{name}}")
    rendered = await scope.render_prompt_template(mode="server", template="Hello {{name}}", variables={"name": "Ada"})
    converted = await scope.convert_prompt(mode="server", payload={"system_prompt": "S", "user_prompt": "U"})
    bulk_deleted = await scope.bulk_delete_prompts(mode="server", prompt_ids=[1])
    bulk_keywords = await scope.bulk_update_prompt_keywords(mode="server", prompt_ids=[1], keywords=["drafting"])
    usage = await scope.record_prompt_usage(mode="server", prompt_identifier="prompt-1")
    collection = await scope.create_prompt_collection(mode="server", name="Pack", prompt_ids=[1])
    collections = await scope.list_prompt_collections(mode="server", limit=25)
    fetched_collection = await scope.get_prompt_collection(mode="server", collection_id=7)
    updated_collection = await scope.update_prompt_collection(mode="server", collection_id=7, name="Updated")

    assert health["status"] == "healthy"
    assert sync_log == {"changes": []}
    assert searched["items"][0]["id"] == "server-prompt-1"
    assert keyword["keyword_text"] == "Drafting"
    assert keywords == ["drafting"]
    assert deleted_keyword is None
    assert exported["message"] == "exported"
    assert keyword_export["message"] == "exported"
    assert imported["imported"] == 1
    assert variables["variables"] == ["name"]
    assert rendered["rendered"] == "Hello Ada"
    assert converted["prompt_definition"] == {"blocks": []}
    assert bulk_deleted["deleted"] == 1
    assert bulk_keywords["updated"] == 1
    assert usage["usage_count"] == 1
    assert collection["collection_id"] == "server-collection-1"
    assert collections["collections"][0]["collection_id"] == "server-collection-1"
    assert fetched_collection["collection_id"] == 7
    assert updated_collection["name"] == "Updated"
    assert server_prompts.calls[-19:] == [
        ("get_prompts_health",),
        ("get_prompt_sync_log", {"since_change_id": 5, "limit": 25}),
        ("search_prompts", {"search_query": "rag"}),
        ("create_prompt_keyword", "Drafting"),
        ("list_prompt_keywords",),
        ("delete_prompt_keyword", "Drafting"),
        ("export_prompts", {"export_format": "markdown"}),
        ("export_prompt_keywords",),
        ("import_prompts", {"prompts": [{"name": "Draft", "content": "Body"}]}),
        ("extract_prompt_template_variables", "Hello {{name}}"),
        ("render_prompt_template", "Hello {{name}}", {"name": "Ada"}),
        ("convert_prompt", {"system_prompt": "S", "user_prompt": "U"}),
        ("bulk_delete_prompts", [1]),
        ("bulk_update_prompt_keywords", [1], ["drafting"], "add"),
        ("record_prompt_usage", "prompt-1"),
        ("create_prompt_collection", {"name": "Pack", "prompt_ids": [1]}),
        ("list_prompt_collections", {"limit": 25}),
        ("get_prompt_collection", 7),
        ("update_prompt_collection", 7, {"name": "Updated"}),
    ]
    assert policy.calls[-19:] == [
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
async def test_prompt_chatbook_scope_service_rejects_local_prompt_utility_surfaces_before_dispatch():
    local_prompts = FakePromptBackend("local")
    scope = PromptChatbookScopeService(
        local_prompt_service=local_prompts,
        server_prompt_service=FakePromptBackend("server"),
        local_chatbook_service=FakeChatbookBackend("local"),
        server_chatbook_service=FakeChatbookBackend("server"),
    )

    with pytest.raises(NotImplementedError, match="server-only"):
        await scope.get_prompts_health(mode="local")

    assert local_prompts.calls == []


@pytest.mark.asyncio
async def test_prompt_chatbook_scope_service_denies_before_dispatch():
    server_prompts = FakePromptBackend("server")
    scope = PromptChatbookScopeService(
        local_prompt_service=FakePromptBackend("local"),
        server_prompt_service=server_prompts,
        local_chatbook_service=FakeChatbookBackend("local"),
        server_chatbook_service=FakeChatbookBackend("server"),
        policy_enforcer=FakePolicyEnforcer("wrong_source"),
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.create_prompt(mode="server", name="Blocked")

    assert exc.value.reason_code == "wrong_source"
    assert server_prompts.calls == []


@pytest.mark.asyncio
async def test_prompt_chatbook_scope_service_exposes_unsupported_chatbook_crud_after_policy():
    policy = FakePolicyEnforcer()
    scope = PromptChatbookScopeService(
        local_prompt_service=FakePromptBackend("local"),
        server_prompt_service=FakePromptBackend("server"),
        local_chatbook_service=FakeChatbookBackend("local"),
        server_chatbook_service=FakeChatbookBackend("server"),
        policy_enforcer=policy,
    )

    with pytest.raises(NotImplementedError):
        await scope.list_chatbooks(mode="server")

    assert policy.calls == ["chatbooks.list.server"]


@pytest.mark.asyncio
async def test_prompt_chatbook_scope_service_routes_local_chatbook_record_crud_when_available():
    local_chatbooks = FakeChatbookCrudBackend("local")
    policy = FakePolicyEnforcer()
    scope = PromptChatbookScopeService(
        local_prompt_service=FakePromptBackend("local"),
        server_prompt_service=FakePromptBackend("server"),
        local_chatbook_service=local_chatbooks,
        server_chatbook_service=FakeChatbookBackend("server"),
        policy_enforcer=policy,
    )

    listed = await scope.list_chatbooks(mode="local", q="pack")
    fetched = await scope.get_chatbook(mode="local", chatbook_id="local-chatbook-1")
    created = await scope.create_chatbook(mode="local", name="Created")
    updated = await scope.update_chatbook(mode="local", chatbook_id="local-chatbook-2", name="Updated")
    deleted = await scope.delete_chatbook(mode="local", chatbook_id="local-chatbook-2")

    assert listed[0]["record_id"] == "local:chatbook:local-chatbook-1"
    assert fetched["record_id"] == "local:chatbook:local-chatbook-1"
    assert created["record_id"] == "local:chatbook:local-chatbook-2"
    assert updated["name"] == "Updated"
    assert deleted is True
    assert policy.calls == [
        "chatbooks.list.local",
        "chatbooks.detail.local",
        "chatbooks.create.local",
        "chatbooks.update.local",
        "chatbooks.delete.local",
    ]


def test_prompt_chatbook_scope_service_reports_known_unsupported_capabilities():
    scope = PromptChatbookScopeService(
        local_prompt_service=FakePromptBackend("local"),
        server_prompt_service=FakePromptBackend("server"),
        local_chatbook_service=FakeChatbookBackend("local"),
        server_chatbook_service=FakeChatbookBackend("server"),
    )

    local_report = scope.list_unsupported_capabilities(mode="local")
    server_report = scope.list_unsupported_capabilities(mode="server")

    assert [item["operation_id"] for item in local_report] == [
        "chatbooks.records.local",
        "prompts.server_utilities.local",
    ]
    assert [item["operation_id"] for item in server_report] == [
        "chatbooks.records.server",
        "chatbooks.import_content_types.server",
    ]
    assert local_report[0]["affected_action_ids"] == [
        "chatbooks.list.local",
        "chatbooks.detail.local",
        "chatbooks.create.local",
        "chatbooks.update.local",
        "chatbooks.delete.local",
    ]
    assert server_report[1]["unsupported_content_types"] == [
        "embedding",
        "evaluation",
        "media",
        "prompt",
    ]


def test_prompt_chatbook_scope_service_omits_local_record_gap_when_backend_supports_crud():
    scope = PromptChatbookScopeService(
        local_prompt_service=FakePromptBackend("local"),
        server_prompt_service=FakePromptBackend("server"),
        local_chatbook_service=FakeChatbookCrudBackend("local"),
        server_chatbook_service=FakeChatbookBackend("server"),
    )

    local_report = scope.list_unsupported_capabilities(mode="local")

    assert [item["operation_id"] for item in local_report] == [
        "prompts.server_utilities.local",
    ]
