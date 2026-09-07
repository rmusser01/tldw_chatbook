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

from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostState
from tldw_chatbook.Chat.console_display_state import (
    CONSOLE_SYSTEM_PROMPT_LABEL_SET,
    CONSOLE_SYSTEM_PROMPT_LABEL_UNSET,
    ConsoleControlState,
)
from tldw_chatbook.Widgets.Console.console_status_chips import ConsoleStatusChips

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

    CSS_PATH = str(BUNDLE)

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


def _visible_chips(scroller: HorizontalScroll):
    return [
        chip
        for chip in scroller.query(".console-control-chip")
        if chip.display is True
    ]


def _assert_chip_fully_inside_viewport(scroller: HorizontalScroll, chip) -> None:
    viewport = scroller.content_region
    region = chip.region
    assert region.width > 0, f"#{chip.id} has zero width"
    assert region.x >= viewport.x, (
        f"#{chip.id} starts left of the strip viewport: {region.x} < {viewport.x}"
    )
    assert region.right <= viewport.right, (
        f"#{chip.id} is clipped by the strip viewport: right {region.right} "
        f"> viewport right {viewport.right}"
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
async def test_every_chip_fully_reachable_by_focus(size: tuple[int, int]) -> None:
    """LY-03 AC: every chip -- including the trailing cost chip at 160
    ("never clips") -- is reachable: focusing it scrolls it fully inside the
    strip viewport, whole, not jammed mid-glyph at the edge."""
    app = _ChipsOverflowApp(_fat_state())
    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.2)
        scroller = app.query_one("#console-status-chip-scroll", HorizontalScroll)
        chips = _visible_chips(scroller)
        # The fat state must actually render the trailing conditional chips
        # for this to prove anything: temporary + tools + cost are present.
        ids = {chip.id for chip in chips}
        assert "console-temporary-chip" in ids
        assert "console-tools-chip" in ids
        assert "console-cost-chip" in ids

        for chip in chips:
            chip.focus()
            await pilot.pause(0.3)
            _assert_chip_fully_inside_viewport(scroller, chip)


@pytest.mark.asyncio
async def test_no_scroll_when_content_fits() -> None:
    """When the chips fit the viewport there is no scroll range and every
    chip renders fully inside it without any scrolling (no phantom jam)."""
    app = _ChipsOverflowApp(_slim_state(), fat=False)
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause(0.2)
        scroller = app.query_one("#console-status-chip-scroll", HorizontalScroll)
        assert scroller.max_scroll_x == 0
        for chip in _visible_chips(scroller):
            _assert_chip_fully_inside_viewport(scroller, chip)


@pytest.mark.asyncio
async def test_system_prompt_chip_name_value_grammar() -> None:
    """TX-06: the System Prompt chip spells its state either way --
    "System Prompt: off" when unset (was the bare noun "System Prompt"),
    "System Prompt: set" when set."""
    assert CONSOLE_SYSTEM_PROMPT_LABEL_UNSET == "System Prompt: off"
    assert CONSOLE_SYSTEM_PROMPT_LABEL_SET == "System Prompt: set"

    app = _ChipsOverflowApp(_slim_state(), fat=False)
    async with app.run_test(size=(160, 48)) as pilot:
        await pilot.pause(0.2)
        chip = app.query_one("#console-system-prompt-chip")
        assert str(chip.render()) == "System Prompt: off"

        strip = app.query_one("#console-status-chips", ConsoleStatusChips)
        strip.sync_state(
            replace(
                _slim_state(),
                system_prompt_label=CONSOLE_SYSTEM_PROMPT_LABEL_SET,
            )
        )
        await pilot.pause(0.2)
        assert str(chip.render()) == "System Prompt: set"
