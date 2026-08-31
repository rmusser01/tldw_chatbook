"""ADR-090: advisory context/summary lines on the approval card."""

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Chat.approval_display import (
    CONTEXT_LABEL,
    RATIONALE_DISPLAY_CAP,
    SUMMARY_LABEL,
    format_context_line,
)
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard


class _CardApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ChatApprovalCard(id="chat-approval-card")


_ROW = {
    "llm_name": "fs_write",
    "tool_name": "fs_write",
    "server_label": "Local",
    "arguments": {"path": "a.txt"},
    "reason": "ask",
    "rationale": "Saving the edited config",
}


def _texts(card: ChatApprovalCard) -> list[str]:
    return [str(s.content) for s in card.query("Static")]


def test_format_context_line_caps_tail_biased():
    out = format_context_line("A" * 300 + "B" * 300)
    assert len(out) == RATIONALE_DISPLAY_CAP
    assert out.startswith("\N{HORIZONTAL ELLIPSIS}") and out.endswith("B")


async def test_row_renders_model_context_line():
    app = _CardApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch([dict(_ROW)], timeout_seconds=0, round_id="r1")
        # set_batch mounts rows via mount(); the DOM only attaches them on
        # the next event-loop tick (same idiom as every other card suite).
        await pilot.pause()
        assert any(CONTEXT_LABEL in t for t in _texts(card))


async def test_row_without_rationale_renders_no_context_line():
    app = _CardApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        row = dict(_ROW, rationale="")
        card.set_batch([row], timeout_seconds=0, round_id="r1")
        await pilot.pause()
        assert not any(CONTEXT_LABEL in t for t in _texts(card))


async def test_set_summary_patches_only_matching_round():
    app = _CardApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch([dict(_ROW)], timeout_seconds=0, round_id="r1")
        await pilot.pause()
        card.set_summary("other-round", "stale text")  # wrong round: dropped
        assert not any(SUMMARY_LABEL in t for t in _texts(card))
        card.set_summary("r1", "Agent is saving your config file")
        assert any(SUMMARY_LABEL in t for t in _texts(card))


async def test_set_summary_never_clobbers_row_decisions():
    app = _CardApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch([dict(_ROW)], timeout_seconds=0, round_id="r1")
        await pilot.pause()
        from textual.widgets import Select

        select = card.query_one(Select)
        select.value = "deny"
        card.set_summary("r1", "late arriving summary")
        assert select.value == "deny"


async def test_payload_carried_summary_renders_on_set_batch():
    app = _CardApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(
            [dict(_ROW)], timeout_seconds=0, round_id="r1", summary="batch summary"
        )
        await pilot.pause()
        assert any(SUMMARY_LABEL in t for t in _texts(card))
