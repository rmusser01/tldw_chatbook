from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    TLDWAPIClient,
    VoiceActionType,
    VoiceAssistantState,
    VoiceCommandDefinition,
    VoiceCommandDryRunRequest,
    VoiceCommandRequest,
    VoiceCommandToggleRequest,
)


COMMAND_INFO = {
    "id": "cmd-1",
    "user_id": 7,
    "persona_id": "persona-1",
    "connection_id": "conn-1",
    "name": "Summarize",
    "phrases": ["summarize this"],
    "action_type": "custom",
    "action_config": {"handler": "summarize"},
    "priority": 10,
    "enabled": True,
    "requires_confirmation": False,
    "description": "Summarize current context",
    "created_at": "2026-04-22T12:00:00Z",
}


SESSION_INFO = {
    "session_id": "voice-session-1",
    "user_id": 7,
    "state": "idle",
    "created_at": "2026-04-22T12:00:00Z",
    "last_activity": "2026-04-22T12:01:00Z",
    "turn_count": 2,
}


@pytest.mark.asyncio
async def test_voice_assistant_client_wraps_rest_routes(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "session_id": "voice-session-1",
                "success": True,
                "transcription": "summarize this",
                "intent": {
                    "type": "intent",
                    "action_type": "custom",
                    "entities": {},
                    "confidence": 1.0,
                    "requires_confirmation": False,
                },
                "action_result": {
                    "type": "action_result",
                    "success": True,
                    "action_type": "custom",
                    "response_text": "Summary ready",
                    "execution_time_ms": 12.5,
                },
                "processing_time_ms": 15.0,
            },
            {"commands": [COMMAND_INFO], "total": 1},
            COMMAND_INFO,
            COMMAND_INFO,
            COMMAND_INFO,
            COMMAND_INFO | {"enabled": False},
            {
                "command_id": "cmd-1",
                "command_name": "Summarize",
                "action_type": "custom",
                "valid": True,
                "steps": [{"name": "config_schema", "passed": True, "message": "ok"}],
            },
            {
                "command_id": "cmd-1",
                "command_name": "Summarize",
                "total_invocations": 5,
                "success_count": 5,
                "error_count": 0,
                "avg_response_time_ms": 42.0,
            },
            {},
            {"sessions": [SESSION_INFO], "total": 1},
            SESSION_INFO,
            {},
            {
                "total_commands_processed": 5,
                "active_sessions": 1,
                "total_voice_commands": 3,
                "enabled_commands": 2,
                "success_rate": 1.0,
                "avg_response_time_ms": 42.0,
                "top_commands": [],
                "usage_by_day": [],
            },
            {
                "dry_run": True,
                "phrase": "summarize this",
                "matched": True,
                "match_method": "phrase",
                "matched_phrase": "summarize this",
                "confidence": 0.98,
                "action_type": "custom",
                "action_config": {"handler": "summarize"},
                "processing_time_ms": 4.5,
                "alternatives": [],
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    command_request = VoiceCommandRequest(text="summarize this", include_tts=False)
    definition = VoiceCommandDefinition(
        persona_id="persona-1",
        connection_id="conn-1",
        name="Summarize",
        phrases=["summarize this"],
        action_type=VoiceActionType.CUSTOM,
        action_config={"handler": "summarize"},
        priority=10,
        requires_confirmation=False,
    )

    processed = await client.process_voice_command(command_request)
    listed = await client.list_voice_commands(
        include_system=False,
        include_disabled=True,
        persona_id="persona-1",
    )
    created = await client.create_voice_command(definition)
    detail = await client.get_voice_command("cmd-1", persona_id="persona-1")
    updated = await client.update_voice_command("cmd-1", definition, persona_id="persona-1")
    toggled = await client.toggle_voice_command(
        "cmd-1",
        VoiceCommandToggleRequest(enabled=False),
        persona_id="persona-1",
    )
    validation = await client.validate_voice_command("cmd-1", persona_id="persona-1")
    usage = await client.get_voice_command_usage("cmd-1", days=14)
    deleted_command = await client.delete_voice_command("cmd-1", persona_id="persona-1")
    sessions = await client.list_voice_sessions(active_only=False, limit=25)
    session = await client.get_voice_session("voice-session-1")
    deleted_session = await client.delete_voice_session("voice-session-1")
    analytics = await client.get_voice_analytics(days=30)
    dry_run = await client.dry_run_voice_command(
        VoiceCommandDryRunRequest(phrase="summarize this", command_id="cmd-1")
    )

    assert processed.success is True
    assert listed.total == 1
    assert created.name == "Summarize"
    assert detail.id == "cmd-1"
    assert updated.action_type == VoiceActionType.CUSTOM
    assert toggled.enabled is False
    assert validation.valid is True
    assert usage.total_invocations == 5
    assert deleted_command == {}
    assert sessions.total == 1
    assert session.state == VoiceAssistantState.IDLE
    assert deleted_session == {}
    assert analytics.total_commands_processed == 5
    assert dry_run.matched is True

    assert [call.args[:2] for call in mocked.await_args_list] == [
        ("POST", "/api/v1/voice/command"),
        ("GET", "/api/v1/voice/commands"),
        ("POST", "/api/v1/voice/commands"),
        ("GET", "/api/v1/voice/commands/cmd-1"),
        ("PUT", "/api/v1/voice/commands/cmd-1"),
        ("POST", "/api/v1/voice/commands/cmd-1/toggle"),
        ("POST", "/api/v1/voice/commands/cmd-1/validate"),
        ("GET", "/api/v1/voice/commands/cmd-1/usage"),
        ("DELETE", "/api/v1/voice/commands/cmd-1"),
        ("GET", "/api/v1/voice/sessions"),
        ("GET", "/api/v1/voice/sessions/voice-session-1"),
        ("DELETE", "/api/v1/voice/sessions/voice-session-1"),
        ("GET", "/api/v1/voice/analytics"),
        ("POST", "/api/v1/voice/voice/commands/dry-run"),
    ]
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "text": "summarize this",
        "include_tts": False,
        "tts_format": "mp3",
    }
    assert mocked.await_args_list[1].kwargs["params"] == {
        "include_system": "false",
        "include_disabled": "true",
        "persona_id": "persona-1",
    }
    assert mocked.await_args_list[8].kwargs["params"] == {"persona_id": "persona-1"}
    assert mocked.await_args_list[9].kwargs["params"] == {
        "active_only": "false",
        "limit": 25,
    }
