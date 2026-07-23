"""Unit tests for the extracted Console status-chips strip."""

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.Widgets.Console.console_status_chips import (
    ConsoleApprovalsChip,
    ConsoleStatusChips,
)


def _state(**overrides) -> ConsoleControlState:
    base = dict(
        provider_label="Provider: Anthropic",
        model_label="Model: claude-3-haiku",
        persona_label="Assistant: General",
        rag_label="RAG: off",
        sources_label="Sources: 0 staged",
        tools_label="Tools: 0 ready",
        approvals_label="Approvals: 0 pending",
        sources_active=False,
        tools_active=False,
        approvals_active=False,
    )
    base.update(overrides)
    return ConsoleControlState(**base)


class _ChipsApp(App):
    def __init__(self, state: ConsoleControlState) -> None:
        super().__init__()
        self._state = state

    def compose(self) -> ComposeResult:
        yield ConsoleStatusChips(self._state, id="console-status-chips")


@pytest.mark.asyncio
async def test_status_chips_render_all_seven_labels():
    app = _ChipsApp(_state())
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        for selector, expected in (
            ("#console-provider-chip", "Provider:"),
            ("#console-model-chip", "Model:"),
            ("#console-persona-chip", "Assistant:"),
            ("#console-rag-chip", "RAG:"),
            ("#console-sources-chip", "Sources:"),
            ("#console-tools-chip", "Tools:"),
            ("#console-approvals-chip", "Approvals:"),
        ):
            chip = app.query_one(selector)
            assert expected in str(chip.render())


@pytest.mark.asyncio
async def test_status_chips_sync_updates_labels_and_emphasis():
    app = _ChipsApp(_state())
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        chips = app.query_one("#console-status-chips", ConsoleStatusChips)
        chips.sync_state(
            _state(
                model_label="Model: gpt-4o",
                sources_label="Sources: 3 staged",
                sources_active=True,
            )
        )
        await pilot.pause()
        assert "gpt-4o" in str(app.query_one("#console-model-chip").render())
        sources = app.query_one("#console-sources-chip")
        assert sources.has_class("console-chip-alert")
        assert not sources.has_class("console-chip-dim")
        # A zero counter stays dim.
        assert app.query_one("#console-tools-chip").has_class("console-chip-dim")


@pytest.mark.asyncio
async def test_approvals_chip_posts_review_requested():
    app = _ChipsApp(_state())
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        chip = app.query_one("#console-approvals-chip", ConsoleApprovalsChip)
        posted: list[object] = []
        original_post_message = chip.post_message
        chip.post_message = lambda message: posted.append(message)  # type: ignore[assignment]
        try:
            chip.action_review_approval()
        finally:
            # Restore before teardown — Textual's prune cascade calls
            # post_message(Prune()) on exit and a swallowing stub hangs it.
            chip.post_message = original_post_message
        assert any(
            isinstance(m, ConsoleApprovalsChip.ReviewRequested) for m in posted
        )
