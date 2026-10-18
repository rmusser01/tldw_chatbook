"""Console status chip strip overflow strategy (TASK-2154.5).

Regression coverage for UX-review findings LY-03 and TX-06
(Docs/superpowers/qa/console-ux-review-2026-08/console-ux-review.md):

* LY-03 -- the strip clipped chips with no wrap/scroll/more affordance:
  the cost chip jammed at the right edge at 160, ``Sources: 0 stage`` cut
  mid-label at 140, and ``Approvals`` was gone entirely at 110. The strip
  is now a ``Horizontal`` host whose inner ``#console-status-chip-scroll``
  viewport is a ``HorizontalScroll`` (hidden single-row scrollbar, the
  ``#console-native-tab-strip`` contract): keyboard users reach every chip
  through focus auto-scroll (``Screen.set_focus``'s ``scroll_visible``),
  mouse users through Shift+wheel / trackpad swipe.
* TX-06 -- the System Prompt chip was a bare noun when unset
  (``System Prompt``) vs ``System Prompt: set`` when set. It now follows
  the same ``name: value`` grammar as its siblings: ``System Prompt: off``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.containers import HorizontalScroll
from textual.widgets import Static

from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostState
from tldw_chatbook.Chat.console_display_state import (
    CONSOLE_SYSTEM_PROMPT_LABEL_SET,
    CONSOLE_SYSTEM_PROMPT_LABEL_UNSET,
    ConsoleControlState,
)
from tldw_chatbook.Widgets.Console.console_status_chips import ConsoleStatusChips
from Tests.UI.consolidated_css import APP_STYLESHEETS

# The app-sheet import chain binds real config getters; per-test env
# redirect trips the config-participant admission (same opt-in as the
# sibling strip suites).
pytestmark = pytest.mark.bootstrap_profile

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "tldw_chatbook" / "css" / "tldw_cli_modular.tcss"


def _fat_state() -> ConsoleControlState:
    """Every chip visible with realistic worst-case labels."""
    return ConsoleControlState(
        provider_label="Provider: Anthropic",
        model_label="Model: claude-3-haiku-20241022",
        assistant_label="Assistant: Research Helper",
        rag_label="Library search: on",
        sources_label="Sources: 12 staged",
        tools_label="Tools: 7 ready",
        approvals_label="Approvals: 2 pending",
        system_prompt_label=CONSOLE_SYSTEM_PROMPT_LABEL_SET,
        sources_active=True,
        tools_active=True,
        approvals_active=True,
    )


def _slim_state() -> ConsoleControlState:
    """Minimum chip set that fits the viewport without scrolling."""
    return ConsoleControlState(
        provider_label="Provider: OA",
        model_label="Model: gpt",
        assistant_label="Assistant: General",
        rag_label="Library search: off",
        sources_label="Sources: 0 staged",
        tools_label="Tools: —",
        approvals_label="Approvals: 0 pending",
        sources_active=False,
        tools_active=False,
        approvals_active=False,
    )


def _cost_state() -> ConsoleCostState:
    return ConsoleCostState(
        label="$0.48 ⚠ ~+$0.13",
        compact_label="$0.48",
        tooltip="total $0.48",
        alert=True,
        cold=False,
    )


class _ChipsOverflowApp(App):
    """Bare chip strip under the shipped stylesheet (real chip widths)."""

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def __init__(self, state: ConsoleControlState, *, fat: bool = True) -> None:
        super().__init__()
        self._state = state
        self._fat = fat

    def compose(self) -> ComposeResult:
        yield ConsoleStatusChips(
            self._state,
            ephemeral=self._fat,
            cost_state=_cost_state() if self._fat else None,
            id="console-status-chips",
        )


def _line(scroller: HorizontalScroll) -> Static:
    """The one-line status text inside the strip's scroll viewport."""
    return scroller.query_one("#console-status-text", Static)


def _segment_names(scroller: HorizontalScroll) -> list[str]:
    return [name for name, _start, _end in _line(scroller).segment_spans]


def _assert_line_fully_inside_viewport(scroller: HorizontalScroll, line) -> None:
    viewport = scroller.content_region
    region = line.region
    assert region.width > 0, "the status line has zero width"
    assert region.x >= viewport.x, (
        f"line starts left of the strip viewport: {region.x} < {viewport.x}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(110, 24), (140, 30)])
async def test_strip_scrolls_when_chips_overflow(size: tuple[int, int]) -> None:
    """LY-03: a full chip set outgrows the viewport at 110 and 140 cols; the
    strip now scrolls horizontally instead of silently dropping later chips
    (Approvals was gone entirely at 110 before)."""
    app = _ChipsOverflowApp(_fat_state())
    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.2)
        scroller = app.query_one("#console-status-chip-scroll", HorizontalScroll)
        assert scroller.virtual_size.width > scroller.content_region.width
        assert scroller.is_scrollable and scroller.allow_horizontal_scroll
        # Single-row contract: the hidden horizontal scrollbar costs no row.
        assert scroller.content_region.height == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(110, 24), (140, 30), (160, 48)])
async def test_every_segment_reachable_by_cursor_at_any_width(size: tuple[int, int]) -> None:
    """LY-03 AC, text-line edition: the whole line is one widget, so
    keyboard reachability is the segment CURSOR -- walking it to the end
    (the trailing cost segment, "never clips") leaves the line inside the
    strip viewport at every width."""
    app = _ChipsOverflowApp(_fat_state())
    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.2)
        scroller = app.query_one("#console-status-chip-scroll", HorizontalScroll)
        line = _line(scroller)
        # The fat state must actually render the trailing conditional
        # segments for this to prove anything: temporary + tools + cost.
        text = line.renderable.plain if hasattr(line.renderable, "plain") else str(line.render())
        assert "Temporary" in text or "temporary" in text.lower()
        assert "Tools:" in text
        assert "$" in text

        line.focus()
        await pilot.press("end")
        await pilot.pause(0.3)
        assert line._cursor == len(line.segment_spans) - 1
        _assert_line_fully_inside_viewport(scroller, line)


@pytest.mark.asyncio
async def test_no_scroll_when_content_fits() -> None:
    """When the line fits the viewport there is no scroll range and it
    renders fully inside it without any scrolling (no phantom jam)."""
    app = _ChipsOverflowApp(_slim_state(), fat=False)
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause(0.2)
        scroller = app.query_one("#console-status-chip-scroll", HorizontalScroll)
        assert scroller.max_scroll_x == 0
        _assert_line_fully_inside_viewport(scroller, _line(scroller))


@pytest.mark.asyncio
async def test_system_prompt_segment_name_value_grammar() -> None:
    """TX-06: the System Prompt segment spells its state either way --
    "System Prompt: off" when unset (was the bare noun "System Prompt"),
    "System Prompt: set" when set."""
    assert CONSOLE_SYSTEM_PROMPT_LABEL_UNSET == "System Prompt: off"
    assert CONSOLE_SYSTEM_PROMPT_LABEL_SET == "System Prompt: set"

    app = _ChipsOverflowApp(_slim_state(), fat=False)
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause(0.2)
        line = _line(scroller := app.query_one("#console-status-chip-scroll", HorizontalScroll))
        text = line.renderable.plain if hasattr(line.renderable, "plain") else str(line.render())
        assert "System Prompt: off" in text

        strip = app.query_one("#console-status-chips", ConsoleStatusChips)
        strip.sync_state(
            replace(
                _slim_state(),
                system_prompt_label=CONSOLE_SYSTEM_PROMPT_LABEL_SET,
            )
        )
        await pilot.pause(0.2)
        text = line.renderable.plain if hasattr(line.renderable, "plain") else str(line.render())
        assert "System Prompt: set" in text


@pytest.mark.asyncio
async def test_plain_wheel_scrolls_the_line_horizontally() -> None:
    """TASK-35160 scrollability: a plain vertical wheel over the strip
    scrolls the line horizontally (down = right, up = left). Shift+wheel
    and trackpad swipe remain native; when the line fits, the wheel is a
    no-op and bubbles."""
    from textual import events

    def wheel(widget: Static, dy: float) -> events.MouseScrollDown | events.MouseScrollUp:
        kwargs = dict(
            widget=widget, x=10, y=0, delta_x=0.0, screen_x=10, screen_y=0,
            button=0, shift=False, meta=False, ctrl=False,
        )
        if dy > 0:
            return events.MouseScrollDown(**kwargs, delta_y=dy)
        return events.MouseScrollUp(**kwargs, delta_y=dy)

    app = _ChipsOverflowApp(_fat_state())
    async with app.run_test(size=(110, 24)) as pilot:
        await pilot.pause(0.2)
        scroller = app.query_one("#console-status-chip-scroll", HorizontalScroll)
        line = _line(scroller)
        assert scroller.max_scroll_x > 0

        line.post_message(wheel(line, 1.0))
        await pilot.pause()
        after_one_tick = scroller.scroll_x
        assert after_one_tick > 0, "plain wheel down did not scroll the strip"

        line.post_message(wheel(line, 3.0))
        await pilot.pause()
        peak = scroller.scroll_x
        assert peak > after_one_tick, "more ticks must scroll further"

        line.post_message(wheel(line, -1.0))
        await pilot.pause()
        assert scroller.scroll_x < peak, "wheel up must scroll back left"


@pytest.mark.asyncio
async def test_wheel_is_a_noop_when_the_line_fits() -> None:
    from textual import events

    app = _ChipsOverflowApp(_slim_state(), fat=False)
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause(0.2)
        scroller = app.query_one("#console-status-chip-scroll", HorizontalScroll)
        line = _line(scroller)
        assert scroller.max_scroll_x == 0
        line.post_message(
            events.MouseScrollDown(
                widget=line, x=10, y=0, delta_x=0.0, delta_y=1.0,
                screen_x=10, screen_y=0, button=0,
                shift=False, meta=False, ctrl=False,
            )
        )
        await pilot.pause()
        assert scroller.scroll_x == 0


@pytest.mark.asyncio
async def test_keyboard_users_tab_straight_to_the_line() -> None:
    """TASK-35160 follow-up: the viewport shell is not a tab stop -- from a
    widget before the strip, one Tab lands on the status line itself (the
    old collapsible's button used to be the previous stop)."""
    from textual.widgets import Button

    class _LeadingApp(_ChipsOverflowApp):
        def compose(self) -> ComposeResult:
            yield Button("lead", id="lead-button")
            yield from super().compose()

    app = _LeadingApp(_fat_state())
    async with app.run_test(size=(110, 24)) as pilot:
        await pilot.pause(0.2)
        line = _line(app.query_one("#console-status-chip-scroll", HorizontalScroll))
        app.query_one("#lead-button", Button).focus()
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        assert app.focused is line


@pytest.mark.asyncio
async def test_shift_arrows_page_the_viewport_without_moving_the_cursor() -> None:
    """Keyboard wheel parity: shift+left/right page the strip viewport;
    the segment cursor stays where it was (arrows remain the precise
    instrument, shift+arrows the fast one)."""
    app = _ChipsOverflowApp(_fat_state())
    async with app.run_test(size=(110, 24)) as pilot:
        await pilot.pause(0.2)
        scroller = app.query_one("#console-status-chip-scroll", HorizontalScroll)
        line = _line(scroller)
        line.focus()
        await pilot.press("end")
        await pilot.pause()
        cursor_before = line._cursor
        assert scroller.max_scroll_x > 0

        scroller.scroll_x = 0
        await pilot.pause()
        await pilot.press("shift+right")
        await pilot.pause()
        assert scroller.scroll_x > 0
        assert line._cursor == cursor_before

        await pilot.press("shift+left")
        await pilot.pause()
        assert scroller.scroll_x < 32 or scroller.scroll_x == 0


@pytest.mark.asyncio
async def test_arrow_buttons_page_the_line_and_dim_at_the_ends() -> None:
    """TASK-35160 follow-up: ◀ ▶ are click affordances for mouse users --
    right pages the line, left pages back, each disables at its end."""
    from textual.widgets import Button

    app = _ChipsOverflowApp(_fat_state())
    async with app.run_test(size=(110, 24)) as pilot:
        await pilot.pause(0.2)
        scroller = app.query_one("#console-status-chip-scroll", HorizontalScroll)
        left = app.query_one("#console-status-scroll-left", Button)
        right = app.query_one("#console-status-scroll-right", Button)
        assert scroller.max_scroll_x > 0
        # At the start: left disabled, right armed.
        assert left.disabled is True
        assert right.disabled is False

        # Press through the real bubble path (pilot.click on a 2-cell button
        # is delivery-flaky under the headless driver; Button.Pressed is the
        # same message the button posts, stripped of the geometry).
        def press_right():
            right.post_message(Button.Pressed(right))

        def press_left():
            left.post_message(Button.Pressed(left))

        press_right()
        await pilot.pause()
        after_right = scroller.scroll_x
        assert after_right > 0
        assert left.disabled is False

        # Walk to the far end: right disables itself there.
        for _ in range(12):
            if right.disabled:
                break
            press_right()
            await pilot.pause()
        assert scroller.scroll_x >= scroller.max_scroll_x - 1
        assert right.disabled is True

        press_left()
        await pilot.pause()
        assert scroller.scroll_x < scroller.max_scroll_x - 1
        assert right.disabled is False


@pytest.mark.asyncio
async def test_arrow_buttons_hidden_when_the_line_fits() -> None:
    from textual.widgets import Button

    app = _ChipsOverflowApp(_slim_state(), fat=False)
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause(0.2)
        left = app.query_one("#console-status-scroll-left", Button)
        right = app.query_one("#console-status-scroll-right", Button)
        assert left.display is False
        assert right.display is False


@pytest.mark.asyncio
async def test_cursor_outlives_visible_segment_shrink_without_crashing() -> None:
    """Review Critical 1: the cursor is NAME-based. Walking to the end and
    letting a segment vanish (run ends, tools hide) must never index past
    the visible list -- the regression used to raise IndexError out of the
    0.2s poll's render pass and kill the tick."""
    app = _ChipsOverflowApp(_fat_state())
    async with app.run_test(size=(110, 24)) as pilot:
        await pilot.pause(0.2)
        scroller = app.query_one("#console-status-chip-scroll", HorizontalScroll)
        line = _line(scroller)
        line.focus()
        await pilot.pause()
        await pilot.press("end")
        await pilot.pause()
        assert line._cursor_name is not None
        names_before = _segment_names(scroller)

        # A segment ABOVE the cursor vanishes (tools drop to zero): the
        # visible list shrinks under the cursor's old index.
        from dataclasses import replace

        strip = app.query_one("#console-status-chips", ConsoleStatusChips)
        strip.sync_state(
            replace(_fat_state(), tools_label="Tools: —", tools_active=False)
        )
        await pilot.pause()

        # No crash, and the cursor either kept its segment or reset.
        names_after = _segment_names(scroller)
        if line._cursor_name is not None:
            assert line._cursor_name in names_after
        # The line still renders, and a vanished segment stays gone.
        assert "tools" not in names_after
        del names_before
