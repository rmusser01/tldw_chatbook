import pytest

from tldw_chatbook.runtime_policy import PolicyDeniedError
from tldw_chatbook.Voice_Assistant_Interop import VoiceAssistantScopeService


class FakeVoiceAssistantService:
    def __init__(self):
        self.calls = []

    async def process_command(self, request_data):
        self.calls.append(("process_command", request_data))
        return {"backend": "server", "record_id": "server:voice_command:session-1"}

    async def list_commands(self, **kwargs):
        self.calls.append(("list_commands", kwargs))
        return {"backend": "server", "record_id": "server:voice_commands", "commands": [], "total": 0}

    async def create_command(self, request_data):
        self.calls.append(("create_command", request_data))
        return {"backend": "server", "record_id": "server:voice_command:cmd-1", "id": "cmd-1"}

    async def get_command(self, command_id, **kwargs):
        self.calls.append(("get_command", command_id, kwargs))
        return {"backend": "server", "record_id": f"server:voice_command:{command_id}"}

    async def update_command(self, command_id, request_data, **kwargs):
        self.calls.append(("update_command", command_id, request_data, kwargs))
        return {"backend": "server", "record_id": f"server:voice_command:{command_id}"}

    async def toggle_command(self, command_id, request_data, **kwargs):
        self.calls.append(("toggle_command", command_id, request_data, kwargs))
        return {"backend": "server", "record_id": f"server:voice_command:{command_id}"}

    async def validate_command(self, command_id, **kwargs):
        self.calls.append(("validate_command", command_id, kwargs))
        return {"backend": "server", "record_id": f"server:voice_command_validation:{command_id}"}

    async def get_command_usage(self, command_id, **kwargs):
        self.calls.append(("get_command_usage", command_id, kwargs))
        return {"backend": "server", "record_id": f"server:voice_command_usage:{command_id}"}

    async def delete_command(self, command_id, **kwargs):
        self.calls.append(("delete_command", command_id, kwargs))
        return {"backend": "server", "record_id": f"server:voice_command:{command_id}", "deleted": True}

    async def list_sessions(self, **kwargs):
        self.calls.append(("list_sessions", kwargs))
        return {"backend": "server", "record_id": "server:voice_sessions", "sessions": [], "total": 0}

    async def get_session(self, session_id):
        self.calls.append(("get_session", session_id))
        return {"backend": "server", "record_id": f"server:voice_session:{session_id}"}

    async def delete_session(self, session_id):
        self.calls.append(("delete_session", session_id))
        return {"backend": "server", "record_id": f"server:voice_session:{session_id}", "deleted": True}

    async def get_analytics(self, **kwargs):
        self.calls.append(("get_analytics", kwargs))
        return {"backend": "server", "record_id": "server:voice_analytics"}

    async def dry_run_command(self, request_data):
        self.calls.append(("dry_run_command", request_data))
        return {"backend": "server", "record_id": "server:voice_command_dry_run:test"}


class WebSocketCapableVoiceAssistantService(FakeVoiceAssistantService):
    supports_websocket_sessions = True


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


COMMAND_DEFINITION = {
    "name": "Summarize",
    "phrases": ["summarize this"],
    "action_type": "custom",
}


@pytest.mark.asyncio
async def test_voice_assistant_scope_service_routes_server_rest_surface():
    server = FakeVoiceAssistantService()
    policy = FakePolicyEnforcer()
    scope = VoiceAssistantScopeService(server_service=server, policy_enforcer=policy)

    await scope.process_command({"text": "summarize this"}, mode="server")
    await scope.list_commands(mode="server", include_system=False)
    await scope.create_command(COMMAND_DEFINITION, mode="server")
    await scope.get_command("cmd-1", mode="server", persona_id="persona-1")
    await scope.update_command("cmd-1", COMMAND_DEFINITION, mode="server", persona_id="persona-1")
    await scope.toggle_command("cmd-1", {"enabled": False}, mode="server", persona_id="persona-1")
    await scope.validate_command("cmd-1", mode="server", persona_id="persona-1")
    await scope.get_command_usage("cmd-1", mode="server", days=7)
    await scope.delete_command("cmd-1", mode="server", persona_id="persona-1")
    await scope.list_sessions(mode="server", active_only=False)
    await scope.get_session("session-1", mode="server")
    await scope.delete_session("session-1", mode="server")
    await scope.get_analytics(mode="server", days=30)
    await scope.dry_run_command({"phrase": "summarize this"}, mode="server")

    assert server.calls[0] == ("process_command", {"text": "summarize this"})
    assert server.calls[-1] == ("dry_run_command", {"phrase": "summarize this"})
    assert policy.calls == [
        "voice_assistant.commands.launch.server",
        "voice_assistant.commands.list.server",
        "voice_assistant.commands.create.server",
        "voice_assistant.commands.detail.server",
        "voice_assistant.commands.update.server",
        "voice_assistant.commands.update.server",
        "voice_assistant.commands.preview.server",
        "voice_assistant.commands.observe.server",
        "voice_assistant.commands.delete.server",
        "voice_assistant.sessions.list.server",
        "voice_assistant.sessions.detail.server",
        "voice_assistant.sessions.delete.server",
        "voice_assistant.analytics.observe.server",
        "voice_assistant.commands.preview.server",
    ]


@pytest.mark.asyncio
async def test_voice_assistant_scope_service_honestly_rejects_local_mode():
    server = FakeVoiceAssistantService()
    scope = VoiceAssistantScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer())

    with pytest.raises(ValueError, match="Voice Assistant server REST operations are server-only"):
        await scope.list_commands(mode="local")

    assert server.calls == []


@pytest.mark.asyncio
async def test_voice_assistant_scope_service_blocks_denied_server_action_before_dispatch():
    server = FakeVoiceAssistantService()
    scope = VoiceAssistantScopeService(
        server_service=server,
        policy_enforcer=FakePolicyEnforcer("server_unreachable"),
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.process_command({"text": "summarize this"}, mode="server")

    assert exc.value.reason_code == "server_unreachable"
    assert server.calls == []


def test_voice_assistant_scope_service_reports_known_unsupported_capabilities():
    scope = VoiceAssistantScopeService(server_service=FakeVoiceAssistantService())

    local_report = scope.list_unsupported_capabilities(mode="local")
    server_report = scope.list_unsupported_capabilities(mode="server")

    assert local_report == [
        {
            "operation_id": "voice_assistant.local_rest.local",
            "source": "local",
            "supported": False,
            "reason_code": "remote_only_surface",
            "user_message": "Voice Assistant server REST commands, sessions, and analytics are unavailable in local/offline mode.",
            "affected_action_ids": [
                "voice_assistant.commands.launch.server",
                "voice_assistant.commands.list.server",
                "voice_assistant.commands.create.server",
                "voice_assistant.commands.detail.server",
                "voice_assistant.commands.update.server",
                "voice_assistant.commands.delete.server",
                "voice_assistant.commands.preview.server",
                "voice_assistant.commands.observe.server",
                "voice_assistant.sessions.list.server",
                "voice_assistant.sessions.detail.server",
                "voice_assistant.sessions.delete.server",
                "voice_assistant.analytics.observe.server",
            ],
        }
    ]
    assert server_report == [
        {
            "operation_id": "voice_assistant.workflows.server",
            "source": "server",
            "supported": False,
            "reason_code": "deferred_scope",
            "user_message": "Voice Assistant workflow template/status/cancel routes are deferred with broader workflow parity.",
            "affected_action_ids": [],
        },
        {
            "operation_id": "voice_assistant.websocket.server",
            "source": "server",
            "supported": False,
            "reason_code": "client_adapter_missing",
            "user_message": "The server exposes Voice Assistant websocket sessions; this Chatbook adapter currently exposes REST commands, sessions, and analytics only.",
            "affected_action_ids": [],
        },
    ]


def test_voice_assistant_scope_service_omits_websocket_gap_for_capable_adapter():
    scope = VoiceAssistantScopeService(server_service=WebSocketCapableVoiceAssistantService())

    assert scope.list_unsupported_capabilities(mode="server") == [
        {
            "operation_id": "voice_assistant.workflows.server",
            "source": "server",
            "supported": False,
            "reason_code": "deferred_scope",
            "user_message": "Voice Assistant workflow template/status/cancel routes are deferred with broader workflow parity.",
            "affected_action_ids": [],
        }
    ]
