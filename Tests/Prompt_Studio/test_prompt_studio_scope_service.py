import pytest

from tldw_chatbook.Prompt_Studio_Interop.server_prompt_studio_service import ServerPromptStudioService
from tldw_chatbook.Prompt_Studio_Interop.prompt_studio_scope_service import PromptStudioScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeServerPromptStudioService:
    def __init__(self):
        self.calls = []

    async def list_projects(self, **kwargs):
        self.calls.append(("list_projects", kwargs))
        return {"success": True, "data": [{"id": 1, "name": "Smoke", "status": "draft"}]}

    async def create_prompt(self, request_data, idempotency_key=None):
        self.calls.append(("create_prompt", request_data, idempotency_key))
        return {"id": 11, "project_id": request_data["project_id"], "name": request_data["name"]}

    async def export_test_cases(self, project_id, request_data):
        self.calls.append(("export_test_cases", project_id, request_data))
        return {"success": True, "data": "[]"}

    async def create_evaluation(self, request_data):
        self.calls.append(("create_evaluation", request_data))
        return {"id": 31, "project_id": request_data["project_id"], "prompt_id": request_data["prompt_id"], "status": "pending"}

    async def cancel_optimization(self, optimization_id, reason=None):
        self.calls.append(("cancel_optimization", optimization_id, reason))
        return {"success": True, "data": {"id": optimization_id, "status": "cancelled"}}

    async def get_status(self, warn_seconds=30):
        self.calls.append(("get_status", warn_seconds))
        return {"success": True, "data": {"queue_depth": 0}}

    async def stream_events(self, client_id=None, project_id=None):
        self.calls.append(("stream_events", client_id, project_id))
        yield {"event": "optimization.started", "project_id": project_id}


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
async def test_server_prompt_studio_service_uses_provider_client_when_no_direct_client():
    class FakeClient:
        async def list_prompt_studio_projects(self, **kwargs):
            return {"success": True, "data": [{"id": 7, "name": "Provider", **kwargs}]}

    provider = FakeClientProvider(FakeClient())
    service = ServerPromptStudioService.from_server_context_provider(provider)

    result = await service.list_projects(search="provider")

    assert result["data"][0]["record_id"] == "server:prompt_studio_project:7"
    assert result["data"][0]["search"] == "provider"
    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 1


@pytest.mark.asyncio
async def test_server_prompt_studio_service_prefers_direct_client_over_provider():
    class FakeClient:
        def __init__(self, project_id):
            self.project_id = project_id

        async def list_prompt_studio_projects(self, **_kwargs):
            return {"success": True, "data": [{"id": self.project_id, "name": "Project"}]}

    provider = FakeClientProvider(FakeClient(8))
    service = ServerPromptStudioService(client=FakeClient(9), client_provider=provider)

    result = await service.list_projects()

    assert result["data"][0]["record_id"] == "server:prompt_studio_project:9"
    assert provider.build_calls == 0


@pytest.mark.asyncio
async def test_prompt_studio_scope_service_routes_server_and_normalizes_records():
    server = FakeServerPromptStudioService()
    policy = FakePolicyEnforcer()
    scope = PromptStudioScopeService(server_service=server, policy_enforcer=policy)

    projects = await scope.list_projects(mode="server", search="Smoke")
    prompt = await scope.create_prompt(
        mode="server",
        request_data={"project_id": 1, "name": "Prompt"},
        idempotency_key="create-prompt-1",
    )
    exported = await scope.export_test_cases(1, {"format": "json"}, mode="server")
    evaluation = await scope.create_evaluation({"project_id": 1, "prompt_id": 11}, mode="server")
    cancelled = await scope.cancel_optimization(41, reason="stop", mode="server")
    status = await scope.get_status(mode="server", warn_seconds=60)
    events = [event async for event in scope.stream_events(mode="server", client_id="chatbook-1", project_id=1)]

    assert projects["data"][0]["record_id"] == "server:prompt_studio_project:1"
    assert prompt["record_id"] == "server:prompt_studio_prompt:11"
    assert exported["record_id"] == "server:prompt_studio_test_case_export:1"
    assert evaluation["record_id"] == "server:prompt_studio_evaluation:31"
    assert cancelled["record_id"] == "server:prompt_studio_optimization:41"
    assert status["record_id"] == "server:prompt_studio_status:queue"
    assert events[0]["backend"] == "server"
    assert server.calls == [
        ("list_projects", {"search": "Smoke"}),
        ("create_prompt", {"project_id": 1, "name": "Prompt"}, "create-prompt-1"),
        ("export_test_cases", 1, {"format": "json"}),
        ("create_evaluation", {"project_id": 1, "prompt_id": 11}),
        ("cancel_optimization", 41, "stop"),
        ("get_status", 60),
        ("stream_events", "chatbook-1", 1),
    ]
    assert policy.calls == [
        "prompt_studio.projects.list.server",
        "prompt_studio.prompts.create.server",
        "prompt_studio.test_cases.export.server",
        "prompt_studio.evaluations.create.server",
        "prompt_studio.optimizations.cancel.server",
        "prompt_studio.status.detail.server",
        "prompt_studio.events.observe.server",
    ]


@pytest.mark.asyncio
async def test_prompt_studio_scope_service_rejects_local_mode_without_dispatch():
    server = FakeServerPromptStudioService()
    scope = PromptStudioScopeService(server_service=server)

    with pytest.raises(ValueError, match="server-only"):
        await scope.list_projects(mode="local")

    assert server.calls == []


@pytest.mark.asyncio
async def test_prompt_studio_scope_service_blocks_denied_action_before_dispatch():
    server = FakeServerPromptStudioService()
    scope = PromptStudioScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer("authority_denied"))

    with pytest.raises(PolicyDeniedError):
        await scope.list_projects(mode="server")

    assert server.calls == []


def test_prompt_studio_scope_service_reports_local_and_server_contract_gaps():
    scope = PromptStudioScopeService(server_service=None)

    local_report = scope.list_unsupported_capabilities(mode="local")
    server_report = scope.list_unsupported_capabilities(mode="server")

    assert local_report == [
        {
            "operation_id": "prompt_studio.remote_only.local",
            "source": "local",
            "supported": False,
            "reason_code": "remote_only_surface",
            "user_message": "Server Prompt Studio projects, prompts, test cases, evaluations, optimizations, status, and events are unavailable in local/offline mode.",
            "affected_action_ids": [],
        }
    ]
    assert server_report == [
        {
            "operation_id": "prompt_studio.websocket_realtime.server",
            "source": "server",
            "supported": False,
            "reason_code": "server_contract_followup",
            "user_message": "REST Prompt Studio operations and SSE observation are available; websocket realtime transport, background ping diagnostics, and local project mirrors remain follow-on.",
            "affected_action_ids": [],
        }
    ]
