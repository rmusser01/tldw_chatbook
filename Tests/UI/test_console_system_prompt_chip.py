"""Tests for the Console system-prompt chip in the status-chips strip.

The chip sits between the Model and Assistant (Character) chips and opens the
existing system-prompt editor modal via ``ConsoleSystemPromptChip.OpenRequested``.
"""

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.containers import HorizontalScroll

from tldw_chatbook.Chat.console_display_state import (
    CONSOLE_SYSTEM_PROMPT_LABEL_SET,
    CONSOLE_SYSTEM_PROMPT_LABEL_UNSET,
    ConsoleControlState,
)
from tldw_chatbook.Widgets.Console.console_status_chips import (
    ConsoleStatusChips,
    ConsoleSystemPromptChip,
)


class StatusChipsHarness(App):
    """Minimal app harness mounting only the ConsoleStatusChips strip."""

    # Mirror the app stylesheet's chip sizing (tldw_cli_modular.tcss) so the
    # whole chip strip fits on screen without the full app chrome.
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
        self.open_requests = 0

    def compose(self) -> ComposeResult:
        yield ConsoleStatusChips(self._state, id="console-status-chips")

    @on(ConsoleSystemPromptChip.OpenRequested)
    def _record_open_request(
        self, event: ConsoleSystemPromptChip.OpenRequested
    ) -> None:
        self.open_requests += 1


@pytest.mark.asyncio
async def test_system_prompt_chip_mounted_between_model_and_assistant_chips():
    app = StatusChipsHarness(ConsoleControlState.from_values())
    async with app.run_test():
        scroller = app.query_one("#console-status-chip-scroll", HorizontalScroll)
        chip_ids = [child.id for child in scroller.children]

        model_index = chip_ids.index("console-model-chip")
        assistant_index = chip_ids.index("console-assistant-chip")
        assert chip_ids.index("console-system-prompt-chip") == model_index + 1
        assert assistant_index == model_index + 2


@pytest.mark.asyncio
async def test_system_prompt_chip_click_posts_open_requested():
    app = StatusChipsHarness(ConsoleControlState.from_values())
    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.click("#console-system-prompt-chip")

        assert app.open_requests == 1


@pytest.mark.asyncio
async def test_system_prompt_chip_keyboard_activation_posts_open_requested():
    app = StatusChipsHarness(ConsoleControlState.from_values())
    async with app.run_test() as pilot:
        app.query_one("#console-system-prompt-chip").focus()
        await pilot.press("enter")

        assert app.open_requests == 1


@pytest.mark.asyncio
async def test_sync_state_updates_system_prompt_chip_label():
    app = StatusChipsHarness(ConsoleControlState.from_values())
    async with app.run_test():
        strip = app.query_one(ConsoleStatusChips)
        chip = app.query_one("#console-system-prompt-chip")
        assert str(chip.renderable) == CONSOLE_SYSTEM_PROMPT_LABEL_UNSET

        strip.sync_state(ConsoleControlState.from_values(system_prompt_set=True))

        assert str(chip.renderable) == CONSOLE_SYSTEM_PROMPT_LABEL_SET
        assert str(chip.tooltip) == CONSOLE_SYSTEM_PROMPT_LABEL_SET
