"""Closed destinations are distinct from live, uncertain draft custody."""

import asyncio

import pytest

from Tests.Chat.conftest import StreamingGateway
from Tests.Chat.console_close_helpers import close_controller_session
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state", ["closed_before_send", "closed_after_withdrawal", "review_required"]
)
async def test_manual_send_distinguishes_closed_target_from_uncertain_custody(
    monkeypatch, state
):
    class ObservedGateway(StreamingGateway):
        resolution_calls = 0

        async def resolve_for_send(self, selection):
            self.resolution_calls += 1
            return await super().resolve_for_send(selection)

    gateway = ObservedGateway()
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    target = store.ensure_session(title="Target")
    target.agent_handoff_state = "review_required"
    store.set_session_draft(target.id, "retain uncertain draft")
    sibling = controller.new_session(title="Unrelated")
    store.append_message(sibling.id, role=ConsoleMessageRole.USER, content="untouched")
    sibling_messages = store.messages_for_session(sibling.id)
    sibling_state = controller.run_state_for(sibling.id)
    if state == "closed_before_send":
        close_controller_session(controller, target.id)
    elif state == "closed_after_withdrawal":
        withdraw = controller._chat_start.withdraw_for_manual

        async def withdraw_then_close(session_id):
            await withdraw(session_id)
            await asyncio.sleep(0)
            close_controller_session(controller, session_id)

        monkeypatch.setattr(
            controller._chat_start, "withdraw_for_manual", withdraw_then_close
        )
    try:
        result = await controller.submit_draft("hello", session_id=target.id)
        if state == "review_required":
            assert result.accepted is False
            assert result.session_closed is False
            assert result.visible_copy == "Draft custody is unconfirmed."
            assert store.session_draft(target.id) == "retain uncertain draft"
            assert store.messages_for_session(target.id) == []
        else:
            assert result.accepted is True
            assert result.session_closed is True
            assert (
                result.visible_copy
                == "Console session closed before your message could send."
            )
            assert all(session.id != target.id for session in store.sessions())
        assert gateway.resolution_calls == 0
        assert store.active_session_id == sibling.id
        assert store.messages_for_session(sibling.id) == sibling_messages
        assert controller.run_state_for(sibling.id) == sibling_state
    finally:
        await controller.shutdown()
