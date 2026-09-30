"""Deny reasons travel with the exact approval row and round."""

from __future__ import annotations

from typing import ClassVar

import pytest
from textual import on
from textual.app import ComposeResult
from textual.widgets import Collapsible, Input

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard


class ReasonApp(ConsolidatedCSSApp):
    CSS_PATH: ClassVar = [str(path) for path in APP_STYLESHEETS]

    def __init__(self):
        super().__init__()
        self.answers = []

    def compose(self) -> ComposeResult:
        yield ChatApprovalCard()

    @on(ChatApprovalCard.ApprovalDecided)
    def capture(self, event):
        self.answers.append((event.round_id, event.decisions))


def calls(count=1):
    return [
        {
            "llm_name": "fs_read",
            "call_id": f"call-{i}",
            "server_label": "Local",
            "tool_name": "fs_read",
            "arguments": {"path": f"file-{i}.txt"},
            "reason": "ask",
        }
        for i in range(count)
    ]


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["row", "fast", "bulk"])
async def test_denial_paths_deliver_each_rows_reason(route):
    """A missing field or a reason keyed by tool name loses distinct answers."""
    app = ReasonApp()
    async with app.run_test(size=(100, 40)) as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(
            calls(2 if route == "bulk" else 1), round_id="round-a", timeout_seconds=0
        )
        await pilot.pause()
        for details in card.query(".deny-reason").results(Collapsible):
            details.collapsed = False
        await pilot.pause()
        fields = list(card.query(".approval-row-denial-reason").results(Input))
        fields[0].value = "Use the public file instead."
        if route == "bulk":
            fields[1].value = "Leave this file alone."
            assert await pilot.click("#approval-deny-all")
            await pilot.pause()
            assert await pilot.click("#approval-submit")
        elif route == "fast":
            assert await pilot.click(card._batch_fast_buttons[1])
        else:
            card._batch_selects[0].value = "deny"
            await pilot.pause()
            assert await pilot.click("#approval-submit")
        await pilot.pause()
        round_id, answers = app.answers[0]
        assert round_id == "round-a"
        assert answers["call-0"] == "deny"
        assert answers.denial_reasons["call-0"] == "Use the public file instead."
        if route == "bulk":
            assert answers.denial_reasons["call-1"] == "Leave this file alone."


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_reason_draft_survives_sync_and_is_cleared_for_a_new_round():
    """Reusing one mounted row must not attach the previous round's reason."""
    app = ReasonApp()
    async with app.run_test(size=(80, 24)) as pilot:
        card = app.query_one(ChatApprovalCard)
        pending = calls()
        card.set_batch(pending, round_id="round-a", timeout_seconds=0)
        await pilot.pause()
        field = card.query_one(".approval-row-denial-reason", Input)
        field.value = "Private file."
        card.set_batch(pending, round_id="round-a", timeout_seconds=0)
        assert card.query_one(".approval-row-denial-reason", Input) is field
        assert field.value == "Private file."
        card.set_batch(pending, round_id="round-b", timeout_seconds=0)
        await pilot.pause()
        assert card.query_one(".approval-row-denial-reason", Input).value == ""
        card._submit_fast_decision("deny")
        await pilot.pause()
        round_id, answers = app.answers[0]
        assert round_id == "round-b"
        assert answers == {"call-0": "deny"}
        assert answers.denial_reasons == {}


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_compact_reason_disclosure_geometry():
    app = ReasonApp()
    async with app.run_test(size=(80, 24)) as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(calls(), round_id="compact", timeout_seconds=0)
        await pilot.pause()
        details = card.query_one(Collapsible)
        title = details.children[0]
        assert details.size.height <= 1, (
            details.size,
            details.styles.padding,
            details.styles.border,
            details.styles.margin,
            details.styles.min_height,
            title.size,
            title.styles.height,
            title.styles.padding,
            title.styles.border,
        )

        # Global focus borders must not consume this one-line disclosure.
        for collapsed in (True, False):
            details.collapsed = collapsed
            title.focus()
            await pilot.pause()
            assert title.content_region.height == 1
            assert title.styles.border_bottom[0] == ""
