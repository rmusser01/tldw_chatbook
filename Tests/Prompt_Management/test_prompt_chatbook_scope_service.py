import pytest

from tldw_chatbook.Prompt_Management.prompt_chatbook_scope_service import PromptChatbookScopeService
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

    assert [item["operation_id"] for item in local_report] == []
