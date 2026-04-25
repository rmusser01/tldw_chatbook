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

    async def import_chatbook(self, chatbook_file_path, request_data):
        self.calls.append(("import_chatbook", chatbook_file_path, request_data))
        return {"job_id": f"{self.source}-import-1", "status": "queued"}

    async def get_export_job(self, job_id):
        self.calls.append(("get_export_job", job_id))
        return {"job_id": job_id, "status": "completed"}

    async def get_import_job(self, job_id):
        self.calls.append(("get_import_job", job_id))
        return {"job_id": job_id, "status": "completed"}


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
    imported = await scope.import_chatbook(
        mode="server",
        chatbook_file_path="/tmp/demo.chatbook.zip",
        request_data={"content_selections": {"conversation": ["1"]}},
    )
    export_job = await scope.get_export_job(mode="server", job_id=export["job_id"])
    import_job = await scope.get_import_job(mode="server", job_id=imported["job_id"])

    assert preview["record_id"] == "server:chatbook:Pack"
    assert export["record_id"] == "server:chatbook_job:server-export-1"
    assert imported["record_id"] == "server:chatbook_job:server-import-1"
    assert export_job["status"] == "completed"
    assert import_job["status"] == "completed"
    assert policy.calls == [
        "chatbooks.detail.server",
        "chatbooks.export.server",
        "chatbooks.import.server",
        "chatbooks.detail.server",
        "chatbooks.detail.server",
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
