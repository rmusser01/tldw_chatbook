"""Widget tests for the Console status strip's rate segment (TASK-35100).

TASK-35160 collapsed the strip into one clickable text line; these tests pin
the rate segment's contract ON that line: settled-only (the live rate rides
the run segment), compact label, absent when hidden, and segment order
(run, rate, provider). Formatting lives in ``Chat/console_token_rate.py``
(pinned in Tests/Chat/test_console_token_rate.py).
"""

import pytest
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET
from textual.app import App, ComposeResult

from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.Widgets.Console.console_status_chips import (
    ConsoleStatusChips,
    ConsoleStatusText,
)

# The strip mount imports real config getters (consolidated_css -> app
# import chain); per-test env redirect trips the config-participant
# admission. Same opt-in as the cost-chip screen tests.
pytestmark = pytest.mark.bootstrap_profile


def _state(**overrides) -> ConsoleControlState:
    base = dict(
        provider_label="Provider: Anthropic",
        model_label="Model: claude-3-haiku",
        assistant_label="Assistant: General",
        rag_label="Library · Auto off · Agent access off",
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
    CSS_PATH = str(BUNDLED_STYLESHEET)
    CSS = ".console-status-text { width: auto; }"

    def __init__(self, state: ConsoleControlState) -> None:
        super().__init__()
        self._state = state

    def compose(self) -> ComposeResult:
        yield ConsoleStatusChips(self._state, id="console-status-chips")


def _line_text(strip: ConsoleStatusChips) -> str:
    """The strip's plain line text (what a user reads)."""
    line = strip.query_one("#console-status-text", ConsoleStatusText)
    return line.renderable.plain if hasattr(line.renderable, "plain") else str(line.render())


def _segment_names(strip: ConsoleStatusChips) -> list[str]:
    line = strip.query_one("#console-status-text", ConsoleStatusText)
    return [name for name, _start, _end in line.segment_spans]


def _segment_span(strip: ConsoleStatusChips, name: str) -> tuple[int, int]:
    line = strip.query_one("#console-status-text", ConsoleStatusText)
    for segment_name, start, end in line.segment_spans:
        if segment_name == name:
            return start, end
    raise AssertionError(f"segment {name!r} not on the line")


@pytest.mark.asyncio
async def test_rate_segment_absent_before_any_settled_reading() -> None:
    app = _ChipsApp(_state())
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        text = _line_text(app.query_one("#console-status-chips", ConsoleStatusChips))
        assert "tok/s" not in text


@pytest.mark.asyncio
async def test_sync_rate_chip_shows_settled_label_on_the_line() -> None:
    app = _ChipsApp(_state())
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        strip = app.query_one("#console-status-chips", ConsoleStatusChips)
        strip.sync_rate_chip(True, "80.0 tok/s avg", "800 tok · 10.0s")
        await pilot.pause()
        assert "80.0 tok/s avg" in _line_text(strip)
        # Detail lives in the destinations now; the line stays compact.
        assert "800 tok" not in _line_text(strip)


@pytest.mark.asyncio
async def test_sync_rate_chip_hides_again_after_reset() -> None:
    app = _ChipsApp(_state())
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        strip = app.query_one("#console-status-chips", ConsoleStatusChips)
        strip.sync_rate_chip(True, "80.0 tok/s avg", "800 tok · 10.0s")
        strip.sync_rate_chip(False, "", "")
        await pilot.pause()
        assert "tok/s" not in _line_text(strip)


@pytest.mark.asyncio
async def test_rate_segment_sits_between_run_and_provider() -> None:
    # TASK-35100 placement contract, carried over to the text line: the
    # transient cluster (run, then the settled rate badge) reads as one
    # group ahead of the stable readiness segments.
    app = _ChipsApp(_state())
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        strip = app.query_one("#console-status-chips", ConsoleStatusChips)
        strip.sync_rate_chip(True, "80.0 tok/s avg", "800 tok · 10.0s")
        strip.sync_run_chip(True, "Streaming response.")
        await pilot.pause()
        names = _segment_names(strip)
        assert names.index("run") < names.index("rate") < names.index("provider")
        # The live rate fragment contract is screen-side; on the widget the
        # run segment simply carries whatever copy the screen pushes.
        assert "Streaming response." in _line_text(strip)
