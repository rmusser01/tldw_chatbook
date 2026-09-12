"""A suspended Console cannot count its selected conversation's results as seen."""

import pytest

from Tests.Chat.test_console_chat_controller import StreamingGateway
from tldw_chatbook.Chat.console_activity_receipts import ConsoleActivityReceiptService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


@pytest.mark.parametrize("visible", [True, False])
@pytest.mark.parametrize("queued", [False, True])
def test_selected_conversation_results_are_unseen_only_when_console_hidden(
    tmp_path, visible, queued
):
    store = ConsoleChatStore()
    session = store.ensure_session()
    db = AgentRunsDB(tmp_path / "activity.db")
    receipts = ConsoleActivityReceiptService(db, None)
    controller = ConsoleChatController(
        store=store, provider_gateway=StreamingGateway(), activity_receipts=receipts
    )
    controller._interrupt_host.set_view_visible(visible)
    if queued:
        # Chain publication occurs after its final run has settled and relinquished
        # its queue slot; idle is not a terminal outcome.
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.COMPLETED), session_id=session.id
        )
    controller._ordinary_outcome_ids[session.id] = "accepted-turn"
    controller._ordinary_outcome_assistant_ids[session.id] = "assistant"
    if queued:
        controller._publish_queue_chain_terminal(
            session.id, ConsoleRunStatus.COMPLETED, "queue-chain:accepted"
        )
    else:
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.COMPLETED), session_id=session.id
        )
    result = receipts.unseen_snapshot()
    assert len(result) == (0 if visible else 1)
    if result:
        assert result[0].session_id == session.id
        assert result[0].assistant_message_id == "assistant"
