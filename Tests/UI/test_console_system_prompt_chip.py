"""Tests for the Console system-prompt segment in the status strip.

TASK-35160: the strip is one clickable text line. The system-prompt segment
sits between the Model and Assistant segments and opens the existing
system-prompt editor modal via ``ConsoleSystemPromptChip.OpenRequested``.
"""

import pytest
from textual import on
from textual.app import App, ComposeResult

from tldw_chatbook.Chat.console_display_state import (
    CONSOLE_SYSTEM_PROMPT_LABEL_SET,
    CONSOLE_SYSTEM_PROMPT_LABEL_UNSET,
    ConsoleControlState,
)
from tldw_chatbook.Widgets.Console.console_status_chips import (
    ConsoleStatusChips,
    ConsoleStatusText,
    ConsoleSystemPromptChip,
)

# The strip mount imports real config getters (consolidated_css -> app
# import chain); per-test env redirect trips the config-participant
# admission. Same opt-in as the cost-chip screen tests.
pytestmark = pytest.mark.bootstrap_profile


class StatusChipsHarness(App):
    """Minimal app harness mounting only the ConsoleStatusChips strip."""

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


def _line(app: StatusChipsHarness) -> ConsoleStatusText:
    return app.query_one("#console-status-text", ConsoleStatusText)


def _line_text(app: StatusChipsHarness) -> str:
    renderable = _line(app).renderable
    return renderable.plain if hasattr(renderable, "plain") else str(_line(app).render())


def _segment_start(app: StatusChipsHarness, name: str) -> int:
    for segment_name, start, _end in _line(app).segment_spans:
        if segment_name == name:
            return start
    raise AssertionError(f"segment {name!r} not on the line")


@pytest.mark.asyncio
async def test_system_prompt_segment_mounted_between_model_and_assistant():
    app = StatusChipsHarness(ConsoleControlState.from_values())
    async with app.run_test():
        names = [name for name, _s, _e in _line(app).segment_spans]
        model_index = names.index("model")
        assistant_index = names.index("assistant")
        assert names.index("system-prompt") == model_index + 1
        assert assistant_index == model_index + 2


@pytest.mark.asyncio
async def test_system_prompt_segment_click_posts_open_requested():
    app = StatusChipsHarness(ConsoleControlState.from_values())
    async with app.run_test(size=(220, 40)) as pilot:
        await pilot.pause()
        await pilot.click(
            "#console-status-text",
            offset=(_segment_start(app, "system-prompt") + 2, 0),
        )
        await pilot.pause()
        assert app.open_requests == 1


@pytest.mark.asyncio
async def test_system_prompt_segment_keyboard_activation_posts_open_requested():
    app = StatusChipsHarness(ConsoleControlState.from_values())
    async with app.run_test() as pilot:
        await pilot.pause()
        # Keyboard parity: focus the line, walk the cursor right until it
        # reaches the system-prompt segment, activate.
        _line(app).focus()
        await pilot.pause()
        for _ in range(len(_line(app).segment_spans)):
            if _line(app).segment_spans[_line(app)._cursor][0] == "system-prompt":
                break
            await pilot.press("right")
        assert _line(app).segment_spans[_line(app)._cursor][0] == "system-prompt"
        await pilot.press("enter")
        await pilot.pause()
        assert app.open_requests == 1


@pytest.mark.asyncio
async def test_sync_state_updates_system_prompt_segment_label():
    app = StatusChipsHarness(ConsoleControlState.from_values())
    async with app.run_test():
        strip = app.query_one(ConsoleStatusChips)
        assert CONSOLE_SYSTEM_PROMPT_LABEL_UNSET in _line_text(app)

        strip.sync_state(ConsoleControlState.from_values(system_prompt_set=True))

        assert CONSOLE_SYSTEM_PROMPT_LABEL_SET in _line_text(app)
