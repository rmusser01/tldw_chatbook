"""Tests for the Console system-prompt chip in the control bar.

The chip sits between the Model and Persona chips and opens the existing
system-prompt editor modal via ``ConsoleSystemPromptChip.EditRequested``.
"""

import pytest
from textual import on
from textual.app import App, ComposeResult

from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.Widgets.Console.console_control_bar import (
    ConsoleControlBar,
    ConsoleSystemPromptChip,
)


class _AppInstanceStub:
    """Minimal stand-in for TldwCli as consumed by CompactModelBar."""

    app_config = {"chat_defaults": {}}


class ControlBarHarness(App):
    # Mirror the app stylesheet's chip sizing (tldw_cli_modular.tcss) so the
    # whole chip row fits on screen without the full app chrome.
    CSS = """
    .console-control-chip {
        width: auto;
        min-width: 7;
        max-width: 22;
        height: 1;
        margin: 0 1 0 0;
        padding: 0 1;
    }
    """

    def __init__(self, state: ConsoleControlState):
        super().__init__()
        self._state = state
        self.edit_requests = 0

    def compose(self) -> ComposeResult:
        yield ConsoleControlBar(
            self._state, _AppInstanceStub(), id="console-control-bar"
        )

    @on(ConsoleSystemPromptChip.EditRequested)
    def _record_edit_request(self, event: ConsoleSystemPromptChip.EditRequested) -> None:
        self.edit_requests += 1


@pytest.mark.asyncio
async def test_system_prompt_chip_mounted_between_model_and_persona_chips():
    app = ControlBarHarness(ConsoleControlState.from_values())
    async with app.run_test():
        row = app.query_one("#console-control-chip-row")
        chip_ids = [child.id for child in row.children]

        model_index = chip_ids.index("console-model-chip")
        persona_index = chip_ids.index("console-persona-chip")
        assert chip_ids.index("console-system-prompt-chip") == model_index + 1
        assert persona_index == model_index + 2


@pytest.mark.asyncio
async def test_system_prompt_chip_click_posts_edit_requested():
    app = ControlBarHarness(ConsoleControlState.from_values())
    async with app.run_test(size=(160, 40)) as pilot:
        chip = app.query_one("#console-system-prompt-chip")
        chip.scroll_visible(animate=False)
        await pilot.pause()
        await pilot.click(chip)

        assert app.edit_requests == 1


@pytest.mark.asyncio
async def test_system_prompt_chip_keyboard_activation_posts_edit_requested():
    app = ControlBarHarness(ConsoleControlState.from_values())
    async with app.run_test() as pilot:
        app.query_one("#console-system-prompt-chip").focus()
        await pilot.press("enter")

        assert app.edit_requests == 1


@pytest.mark.asyncio
async def test_sync_state_updates_system_prompt_chip_label():
    app = ControlBarHarness(ConsoleControlState.from_values())
    async with app.run_test():
        bar = app.query_one(ConsoleControlBar)
        chip = app.query_one("#console-system-prompt-chip")
        assert str(chip.renderable) == "System Prompt"

        bar.sync_state(ConsoleControlState.from_values(system_prompt_set=True))

        assert str(chip.renderable) == "System Prompt: set"
        assert chip.tooltip == "System Prompt: set"
