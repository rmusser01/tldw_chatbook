"""Unit tests for the extracted Console status-chips strip."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Tooltip

from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.Widgets.Console.console_status_chips import (
    ConsoleApprovalsChip,
    ConsoleStatusChips,
)


def _state(**overrides) -> ConsoleControlState:
    base = dict(
        provider_label="Provider: Anthropic",
        model_label="Model: claude-3-haiku",
        assistant_label="Assistant: General",
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
    CSS = ".console-control-chip { width: auto; }"

    def __init__(self, state: ConsoleControlState) -> None:
        super().__init__()
        self._state = state

    def compose(self) -> ComposeResult:
        yield ConsoleStatusChips(self._state, id="console-status-chips")


@pytest.mark.asyncio
async def test_status_chips_render_one_assistant_identity_label():
    app = _ChipsApp(_state())
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        for selector, expected in (
            ("#console-provider-chip", "Provider:"),
            ("#console-model-chip", "Model:"),
            ("#console-assistant-chip", "Assistant: General"),
            ("#console-rag-chip", "RAG:"),
            ("#console-sources-chip", "Sources:"),
            ("#console-tools-chip", "Tools:"),
            ("#console-approvals-chip", "Approvals:"),
        ):
            chip = app.query_one(selector)
            assert expected in str(chip.render())
        assert not app.query("#console-character-chip")
        assert not app.query("#console-persona-chip")


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
        assert any(isinstance(m, ConsoleApprovalsChip.ReviewRequested) for m in posted)


@pytest.mark.asyncio
async def test_status_chips_sync_updates_assistant_identity_chip():
    """The assistant chip must refresh on sync, not stay at its compose value.

    Live repro: starting a chat from a character rendered "Character: none"
    forever, because the chip was painted once at compose (before any character
    existed) and `sync_state` never touched it again.
    """
    app = _ChipsApp(_state())
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        chips = app.query_one("#console-status-chips", ConsoleStatusChips)
        chips.sync_state(
            ConsoleControlState.from_values(
                provider="llama_cpp", model="m", character="Seraphina"
            )
        )
        await pilot.pause()

        chip = app.query_one("#console-assistant-chip")
        assert "Seraphina" in str(chip.render())


@pytest.mark.asyncio
async def test_status_chips_do_not_parse_markup_in_assistant_names():
    """A character or Persona name must never be parsed as Rich markup.

    Names are user data (imported cards included). Rendering them through a
    markup-enabled Static lets `[red]...[/]` restyle the chip strip, or raise
    MarkupError on an unbalanced tag, from nothing more than a character name.
    """
    app = _ChipsApp(
        ConsoleControlState.from_values(
            provider="llama_cpp",
            model="m",
            assistant_kind="persona",
            assistant_name="[bold]Guide[/]",
            assistant_id="persona-7",
        )
    )
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()

        assistant_chip = app.query_one("#console-assistant-chip")

        assert "[bold]Guide[/]" in str(assistant_chip.render())
        assert "Persona:" in str(assistant_chip.render())


@pytest.mark.parametrize("after_sync", [False, True], ids=["compose", "sync"])
@pytest.mark.asyncio
async def test_assistant_tooltip_renders_malformed_markup_literally(after_sync):
    """Textual's separate Tooltip widget must not parse assistant names."""
    malformed_state = ConsoleControlState.from_values(
        provider="llama_cpp",
        model="m",
        assistant_kind="persona",
        assistant_name="[bold]Guide[/red]",
        assistant_id="persona-7",
    )
    app = _ChipsApp(_state() if after_sync else malformed_state)
    app.TOOLTIP_DELAY = 0.01

    async with app.run_test(size=(200, 6), tooltips=True) as pilot:
        await pilot.pause()
        if after_sync:
            app.query_one("#console-status-chips", ConsoleStatusChips).sync_state(
                malformed_state
            )
            await pilot.pause()

        assert await pilot.hover("#console-assistant-chip")
        await pilot.pause(0.05)

        tooltip = app.screen.get_child_by_type(Tooltip)
        assert tooltip.display is True
        assert "Persona: [bold]Guide[/red]" in str(tooltip.render())
