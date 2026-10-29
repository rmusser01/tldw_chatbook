"""TASK-34100.5.3 review round 1 (F3/F4): speculative voice keeps no tools.

``prepare_speculative_voice_attempt`` asks the agent bridge for
``preview_tool_schemas`` and puts the rows on the speculative request. The
bridge never had that method, so every speculative voice attempt went out
with ``tools=[]``. Implementing it for the context meter would have changed
voice too: the light tools, without the light plan's "Answer directly" line
(gpt-4.1-nano called a tool 15 times in 15 without it), and every tool call
aborts the attempt and hands off to a second, authoritative request. The
speculative attempt keeps carrying no tools, as it always has.
"""

from __future__ import annotations

import pytest

from Tests.Chat.test_console_voice_preflight import controller_for

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
async def test_a_speculative_voice_attempt_carries_no_tools(tmp_path):
    import httpx

    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    cfg = {
        "api_settings": {
            "deepseek": {
                "api_key": "sk-hermetic-test-credential",
                "api_url": "https://api.deepseek.com",
            }
        },
        "providers": {"deepseek": ["deepseek-chat"]},
    }

    def forbid_network(request):
        raise AssertionError("Preparation must not send a generation request")

    preview_calls: list[object] = []

    class _Bridge:
        """A bridge whose meter preview offers the light set."""

        def preview_tool_schemas(self, resolution=None, **_kwargs):
            preview_calls.append(resolution)
            return [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": name,
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
                for name in ("get_current_datetime", "calculator", "ask_user")
            ]

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(forbid_network)
    ) as client:
        gateway = ConsoleProviderGateway(
            http_client=client, config_provider=lambda: cfg, environ={}
        )
        db = CharactersRAGDB(tmp_path / "voice-light.db", "voice-light")
        controller = controller_for(gateway, persistence=ChatPersistenceService(db))
        controller._agent_runtime_enabled = True
        controller._agent_bridge = _Bridge()
        session = controller.store.sessions()[0]
        session.persisted_conversation_id = db.add_conversation(
            {"title": "voice light surface"}
        )
        controller._provider_config = lambda: cfg
        controller._staged_evidence_provider = lambda _: False
        prepared = await controller.prepare_speculative_voice_attempt(
            attempt_epoch=1, transcript="hi", turn_id="turn"
        )
        try:
            assert tuple(prepared.request.prepared.tools) == ()
            assert preview_calls == []
            # Nothing about the (absent) preview forces the authoritative path.
            assert prepared.requires_pre_dispatch_authority is False
        finally:
            if prepared.request.provisional_trace_attempt is not None:
                gateway.abandon_provisional_voice_trace(
                    prepared.request.provisional_trace_attempt
                )
            controller._scratch_spaces.dispose()
            await gateway.aclose()
            db.close_connection()
