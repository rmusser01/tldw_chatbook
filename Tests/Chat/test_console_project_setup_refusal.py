"""A first-send folder decision must release the run before a retry."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["disable", "cancel", "unavailable"])
async def test_project_setup_refusal_ends_run_and_excludes_unsent_echo(decision):
    store = ConsoleChatStore()
    session = store.create_session(
        project_instruction_state=ProjectInstructionControlState.new_session(),
    )
    resolution = SimpleNamespace(
        ready=True,
        provider="deepseek",
        execution_key="deepseek",
        model="deepseek-chat",
        max_tokens=32,
        streaming=True,
    )
    bridge = Mock()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=SimpleNamespace(
            resolve_for_send=AsyncMock(return_value=resolution)
        ),
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    controller._select_project_instruction_binding = (
        None if decision == "unavailable" else AsyncMock(return_value=(decision, None))
    )

    bridge.reset_mock()
    result = await controller.submit_draft("Synthetic setup refusal check.")

    assert not result.accepted
    assert not result.should_clear_draft
    assert controller.run_state_for(session.id).status.value == "blocked"
    assert not controller.activity_for(session.id).occupies_slot
    messages = store.messages_for_session(session.id)
    assert (
        next(m for m in messages if m.role is ConsoleMessageRole.USER).status
        == "failed"
    )
    assert (
        next(m for m in messages if m.role is ConsoleMessageRole.ASSISTANT).status
        == "failed"
    )
    assert all(
        m["content"] != "Synthetic setup refusal check."
        for m in controller._provider_messages_for_session(session.id)
    )
    assert not bridge.mock_calls
    if decision == "disable":
        assert not session.project_instruction_state.project_instructions_enabled
