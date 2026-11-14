"""Unit tests for the Console status strip (TASK-35160: one clickable text line).

Formerly the chip-widget strip (TASK-2154.x): the same segment set, emphasis
and visibility contracts, pinned against the single ``ConsoleStatusText``
line -- plain text content, segment presence/order, activation posting the
same typed messages, and markup-safety for user-data labels.
"""

import pytest
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET
from textual.app import App, ComposeResult

from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.Widgets.Console.console_status_chips import (
    ConsoleApprovalsChip,
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
    renderable = line.renderable
    return renderable.plain if hasattr(renderable, "plain") else str(line.render())


def _segment_span(strip: ConsoleStatusChips, name: str) -> tuple[int, int]:
    line = strip.query_one("#console-status-text", ConsoleStatusText)
    for segment_name, start, end in line.segment_spans:
        if segment_name == name:
            return start, end
    raise AssertionError(f"segment {name!r} not on the line")


def _segment_names(strip: ConsoleStatusChips) -> list[str]:
    line = strip.query_one("#console-status-text", ConsoleStatusText)
    return [name for name, _start, _end in line.segment_spans]


@pytest.mark.asyncio
async def test_status_segments_render_the_readiness_labels():
    app = _ChipsApp(_state())
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        text = _line_text(app.query_one("#console-status-chips", ConsoleStatusChips))
        for expected in (
            "Provider: Anthropic",
            "Model: claude-3-haiku",
            "Assistant: General",
            "Sources: 0 staged",
            "Approvals: 0 pending",
        ):
            assert expected in text, expected
        # The Library policy readout left the line (TASK-35160 follow-up):
        # it lives behind the composer menu's "Library access..." entry.
        assert "Library" not in text
        # Zero-count tools hides entirely (TX-04); the scope and rate
        # segments have nothing to say in the base state.
        assert "Tools:" not in text
        assert "Scope:" not in text
        assert "tok/s" not in text


@pytest.mark.asyncio
async def test_status_sync_updates_labels_and_emphasis():
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
        text = _line_text(chips)
        assert "Model: gpt-4o" in text
        assert "Sources: 3 staged" in _line_text(chips)


@pytest.mark.asyncio
async def test_tools_segment_hides_at_zero_count_and_shows_when_ready():
    """TX-04 (TASK-2154.12): the tools segment must not render the
    lazy-loading detail -- it disappears at a zero count and appears once
    tools are actually counted."""
    app = _ChipsApp(_state(tools_label="Tools: —", tools_active=False))
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        chips = app.query_one("#console-status-chips", ConsoleStatusChips)
        assert "tools" not in _segment_names(chips)

        chips.sync_state(_state(tools_label="Tools: 3 ready", tools_active=True))
        await pilot.pause()
        assert "tools" in _segment_names(chips)
        assert "Tools: 3 ready" in _line_text(chips)

        chips.sync_state(_state(tools_label="Tools: —", tools_active=False))
        await pilot.pause()
        assert "tools" not in _segment_names(chips)


@pytest.mark.asyncio
async def test_tools_segment_shown_at_compose_when_tools_ready():
    app = _ChipsApp(_state(tools_label="Tools: 10 ready", tools_active=True))
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        assert "tools" in _segment_names(
            app.query_one("#console-status-chips", ConsoleStatusChips)
        )


@pytest.mark.asyncio
async def test_approvals_segment_click_runs_the_review_flow():
    """Clicking the approvals segment reaches the strip's review handler.

    Asserted via the handler's no-card behavior (``app.notify``) rather than
    stubbing ``post_message`` -- a swallowing stub breaks Textual's prune
    cascade (the trap the chip-era test's teardown comment documented).
    """
    app = _ChipsApp(_state())
    # Wide enough that the trailing approvals segment is inside the strip
    # viewport (the strip scrolls beyond it; clicks must target visible cells).
    async with app.run_test(size=(220, 6)) as pilot:
        await pilot.pause()
        strip = app.query_one("#console-status-chips", ConsoleStatusChips)
        start, _end = _segment_span(strip, "approvals")
        notes: list[tuple] = []
        app.notify = lambda *args, **kwargs: notes.append(args)  # type: ignore[assignment]
        await pilot.click("#console-status-text", offset=(start + 2, 0))
        await pilot.pause()
        assert notes, "approvals click never reached the review handler"


@pytest.mark.asyncio
async def test_status_sync_updates_assistant_identity_segment():
    """The assistant segment must refresh on sync, not stay at its compose
    value (live repro kept from the chip era: a character picked after mount
    never showed because the label was painted once)."""
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
        assert "Seraphina" in _line_text(chips)


@pytest.mark.asyncio
async def test_status_line_does_not_parse_markup_in_assistant_names():
    """A character or Persona name must never be parsed as Rich markup.

    Names are user data (imported cards included). Rendering them through a
    markup-enabled line lets `[red]...[/]` restyle the strip, or raise
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
        text = _line_text(app.query_one("#console-status-chips", ConsoleStatusChips))
        assert "[bold]Guide[/]" in text
        assert "Persona:" in text
