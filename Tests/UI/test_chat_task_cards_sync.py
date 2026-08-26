"""TASK-294 (P5 minor): direct coverage for `ChatTaskCards.sync_state`.

The batch branch was exercised only through the full ChatScreen integration
path; nothing drove the wrapper directly, so a regression in its own
routing (approval payload -> card, display gating) could hide behind the
bigger harness's noise.
"""
from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp

from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards


class _CardsHarness(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield ChatTaskCards(id="chat-task-cards")


@pytest.mark.asyncio
async def test_sync_state_routes_an_approval_batch_to_the_card():
    """A pending_approval payload mounts visible rows; clearing hides all."""
    app = _CardsHarness()
    async with app.run_test() as pilot:
        await pilot.pause()
        cards = app.query_one(ChatTaskCards)
        assert cards.display is False, "precondition: hidden until pending"

        cards.sync_state(
            TaskResumeState(
                pending_approval={
                    "calls": [
                        {
                            "llm_name": "read_file",
                            "server_label": "Built-in",
                            "tool_name": "read_file",
                            "arguments": {"path": "a.md"},
                            "call_id": "c1",
                        },
                        {
                            "llm_name": "read_file",
                            "server_label": "Built-in",
                            "tool_name": "read_file",
                            "arguments": {"path": "b.md"},
                            "call_id": "c2",
                        },
                    ],
                    "timeout_seconds": 45.0,
                    "round_id": "round-x",
                }
            )
        )
        await pilot.pause()
        await pilot.pause()

        assert cards.display is True
        card = app.query_one(ChatApprovalCard)
        rows = list(card.query(".approval-row"))
        assert len(rows) == 2, "one row per call id (TASK-1861 keying)"
        assert card._batch_round_id == "round-x", (
            "the round id must ride through, or a decision resolves the "
            "wrong round"
        )

        cards.sync_state(TaskResumeState())
        await pilot.pause()
        assert cards.display is False, "cleared state must hide the surface"
