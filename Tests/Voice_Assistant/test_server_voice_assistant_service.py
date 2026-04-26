import pytest

from tldw_chatbook.runtime_policy import PolicyDeniedError
from tldw_chatbook.Voice_Assistant_Interop import ServerVoiceAssistantService


class FakeVoiceAssistantClient:
    def __init__(self):
        self.calls = []

    async def process_voice_command(self, request_data):
        self.calls.append(("process_voice_command", request_data.model_dump(exclude_none=True, mode="json")))
        return {"session_id": "session-1", "success": True}

    async def list_voice_commands(self, **kwargs):
        self.calls.append(("list_voice_commands", kwargs))
        return {"commands": [], "total": 0}

    async def create_voice_command(self, request_data):
        self.calls.append(("create_voice_command", request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": "cmd-1", "name": request_data.name}

    async def get_voice_command(self, command_id, **kwargs):
        self.calls.append(("get_voice_command", command_id, kwargs))
        return {"id": command_id}

    async def update_voice_command(self, command_id, request_data, **kwargs):
        self.calls.append(("update_voice_command", command_id, request_data.model_dump(exclude_none=True, mode="json"), kwargs))
        return {"id": command_id, "name": request_data.name}

    async def toggle_voice_command(self, command_id, request_data, **kwargs):
        self.calls.append(("toggle_voice_command", command_id, request_data.model_dump(exclude_none=True, mode="json"), kwargs))
        return {"id": command_id, "enabled": request_data.enabled}

    async def validate_voice_command(self, command_id, **kwargs):
        self.calls.append(("validate_voice_command", command_id, kwargs))
        return {"command_id": command_id, "valid": True}

    async def get_voice_command_usage(self, command_id, **kwargs):
        self.calls.append(("get_voice_command_usage", command_id, kwargs))
        return {"command_id": command_id, "total_invocations": 1}

    async def delete_voice_command(self, command_id, **kwargs):
        self.calls.append(("delete_voice_command", command_id, kwargs))
        return {}

    async def list_voice_sessions(self, **kwargs):
        self.calls.append(("list_voice_sessions", kwargs))
        return {"sessions": [], "total": 0}

    async def get_voice_session(self, session_id):
        self.calls.append(("get_voice_session", session_id))
        return {"session_id": session_id}

    async def delete_voice_session(self, session_id):
        self.calls.append(("delete_voice_session", session_id))
        return {}

    async def get_voice_analytics(self, **kwargs):
        self.calls.append(("get_voice_analytics", kwargs))
        return {"total_commands_processed": 1}

    async def dry_run_voice_command(self, request_data):
        self.calls.append(("dry_run_voice_command", request_data.model_dump(exclude_none=True, mode="json")))
        return {"dry_run": True, "phrase": request_data.phrase}


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
    "action_config": {"handler": "summarize"},
}


@pytest.mark.asyncio
async def test_server_voice_assistant_service_routes_rest_calls_with_policy_actions():
    client = FakeVoiceAssistantClient()
    policy = FakePolicyEnforcer()
    service = ServerVoiceAssistantService(client, policy_enforcer=policy)

    processed = await service.process_command({"text": "summarize this", "include_tts": False})
    listed = await service.list_commands(include_system=False, include_disabled=True, persona_id="persona-1")
    created = await service.create_command(COMMAND_DEFINITION)
    detail = await service.get_command("cmd-1", persona_id="persona-1")
    updated = await service.update_command("cmd-1", COMMAND_DEFINITION, persona_id="persona-1")
    toggled = await service.toggle_command("cmd-1", {"enabled": False}, persona_id="persona-1")
    validation = await service.validate_command("cmd-1", persona_id="persona-1")
    usage = await service.get_command_usage("cmd-1", days=14)
    deleted_command = await service.delete_command("cmd-1", persona_id="persona-1")
    sessions = await service.list_sessions(active_only=False, limit=25)
    session = await service.get_session("session-1")
    deleted_session = await service.delete_session("session-1")
    analytics = await service.get_analytics(days=30)
    dry_run = await service.dry_run_command({"phrase": "summarize this", "command_id": "cmd-1"})

    assert processed["record_id"] == "server:voice_command:session-1"
    assert listed["backend"] == "server"
    assert created["record_id"] == "server:voice_command:cmd-1"
    assert detail["record_id"] == "server:voice_command:cmd-1"
    assert updated["name"] == "Summarize"
    assert toggled["enabled"] is False
    assert validation["record_id"] == "server:voice_command_validation:cmd-1"
    assert usage["record_id"] == "server:voice_command_usage:cmd-1"
    assert deleted_command == {"backend": "server", "record_id": "server:voice_command:cmd-1", "deleted": True}
    assert sessions["record_id"] == "server:voice_sessions"
    assert session["record_id"] == "server:voice_session:session-1"
    assert deleted_session == {"backend": "server", "record_id": "server:voice_session:session-1", "deleted": True}
    assert analytics["record_id"] == "server:voice_analytics"
    assert dry_run["record_id"] == "server:voice_command_dry_run:summarize this"
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
    assert client.calls[0] == (
        "process_voice_command",
        {"text": "summarize this", "include_tts": False, "tts_format": "mp3"},
    )


@pytest.mark.asyncio
async def test_server_voice_assistant_service_denies_before_dispatch():
    client = FakeVoiceAssistantClient()
    service = ServerVoiceAssistantService(
        client,
        policy_enforcer=FakePolicyEnforcer("authority_denied"),
    )

    with pytest.raises(PolicyDeniedError):
        await service.list_commands()

    assert client.calls == []
