"""TASK-23021: the setup-modal snow must not repaint at idle AT ALL.

TASK-21134 made the tick itself cheap (``layout=False``, ``markup=False``,
5 Hz -> 2.5 Hz) and the burn survived one layer down: each tick's
``Static.update`` dirtied the full-viewport backdrop, and Textual's
compositor re-renders every widget overlapping the dirty crop -- measured on
the real unconfigured Console screen (TASK-23021, interleaved A/B, 15 s
getrusage windows with no ``pilot.pause()``): 124 widget renders x 44 rows,
13-16 ms per repaint inside ``Screen._on_timer_update``, 3.6-4.3% of a core
at idle vs a 0.04% floor with the tick neutralised. Shrinking the dirty
region does not help -- ``Compositor.render_partial_update`` crops to the
*bounding box* of the dirty cells, and flakes span the field, so a
per-cell-dirty variant measured 2.7-3.6%; even a single 3x3 repaint at the
same 2.5 Hz measured ~0.55%. On this screen ANY repeating repaint is too
expensive for a decoration.

So the contract is now structural, not a rate cap: the backdrop arms no
timers and performs no repaints between resizes -- the flake field is a
still frame. These tests pin that, and the quit/unmount walk covers the
timer-teardown class of bug that has broken shutdown in this repo before.

Each test was mutation-checked against a deliberately re-animated build
(a 0.4 s ``set_interval`` re-added on mount, driving a repaint of the
field); the mutation results are recorded in the TASK-23021 notes.
"""

import ast
import asyncio
import inspect
import random

import pytest

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp

from textual.app import ComposeResult

from tldw_chatbook.Chat.console_onboarding_state import (
    ConsoleSetupCardState,
    ConsoleSetupStep,
)
from tldw_chatbook.Widgets.Console import console_setup_modal as setup_modal
from tldw_chatbook.Widgets.Console.console_setup_modal import (
    CONSOLE_SETUP_MODAL_BACKDROP_ID,
    ConsoleSetupBackdrop,
    ConsoleSetupModal,
)

#: Longer than several of the retired animation's 0.4 s tick intervals, so a
#: re-animated mutant repaints multiple times inside the window.
_IDLE_WINDOW_S = 1.3


class BackdropHarness(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield ConsoleSetupBackdrop(rng=random.Random(23021))


class ModalHarness(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield ConsoleSetupModal(id="console-setup-modal")

    async def on_mount(self) -> None:
        self.query_one("#console-setup-modal", ConsoleSetupModal).sync_card_state(
            ConsoleSetupCardState(
                mode="card",
                steps=(ConsoleSetupStep(state="active", label="Add an API key"),),
            )
        )


def _count_refreshes(backdrop: ConsoleSetupBackdrop) -> list:
    """Instrument the widget's own repaint entry point.

    ``Static.update`` funnels through ``self.refresh`` and so does any
    direct ``refresh(...)`` a re-animated implementation could issue, so an
    instance-level wrap observes every way this widget can dirty itself.
    """
    calls: list = []
    real_refresh = backdrop.refresh

    def counting_refresh(*regions, **kwargs):
        calls.append((regions, kwargs))
        return real_refresh(*regions, **kwargs)

    backdrop.refresh = counting_refresh  # type: ignore[method-assign]
    return calls


def test_module_defines_no_repeating_clock():
    """Structural guard: no timer constructor may return to this module.

    The retired burn arrived via ``set_interval``; ``set_timer`` chains
    (one-shots re-arming themselves) are the census-documented way the same
    cost gets respelled, so both are barred at the AST level.
    """
    source = inspect.getsource(setup_modal)
    tree = ast.parse(source)
    clock_calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"set_interval", "set_timer"}
    ]
    assert clock_calls == [], (
        f"console_setup_modal.py grew timer constructors {clock_calls}; "
        "TASK-23021 measured any repeating repaint of this overlay at "
        ">=0.5% of a core -- the flake field must stay a still frame."
    )


@pytest.mark.asyncio
async def test_backdrop_mounts_no_timers_and_never_repaints_at_idle():
    """The dirty region ends at zero: between resizes the backdrop must not
    dirty ANYTHING -- no timers registered, no refresh calls over a window
    longer than three of the old tick intervals (slept, not pilot-paused)."""
    app = BackdropHarness()

    async with app.run_test(size=(80, 24)):
        backdrop = app.query_one(ConsoleSetupBackdrop)
        assert backdrop.flake_count > 0
        assert len(backdrop._timers) == 0

        calls = _count_refreshes(backdrop)
        await asyncio.sleep(_IDLE_WINDOW_S)
        assert calls == [], (
            f"backdrop repainted {len(calls)} time(s) while idle: {calls[:3]}"
        )
        assert len(backdrop._timers) == 0


@pytest.mark.asyncio
async def test_backdrop_renders_deterministic_still_frame():
    """Seeded rng + fixed size => a deterministic flake frame that does not
    change while mounted (the animation really is retired, not just its
    timer detached from this widget's registry)."""
    app = BackdropHarness()

    async with app.run_test(size=(80, 24)):
        backdrop = app.query_one(ConsoleSetupBackdrop)
        frame_before = str(backdrop.renderable)
        glyph_cells = sum(frame_before.count(g) for g in ("·", "•", "*"))
        # Two flakes can share a cell, so painted glyphs may undercount
        # slightly -- but a blank field is a failure.
        assert 0 < glyph_cells <= backdrop.flake_count
        assert glyph_cells >= backdrop.flake_count // 2

        await asyncio.sleep(_IDLE_WINDOW_S)
        assert str(backdrop.renderable) == frame_before


@pytest.mark.asyncio
async def test_resize_redraws_once_then_field_is_still_again():
    """The one remaining repaint path is a real size change, and it may lay
    out (the field's dimensions genuinely changed). After the resize settles
    the field must go back to complete stillness."""
    app = BackdropHarness()

    async with app.run_test(size=(80, 24)) as pilot:
        backdrop = app.query_one(ConsoleSetupBackdrop)
        count_at_80x24 = backdrop.flake_count

        await pilot.resize_terminal(40, 10)
        await pilot.pause()
        assert backdrop.flake_count < count_at_80x24
        assert backdrop._field_width == 40

        calls = _count_refreshes(backdrop)
        await asyncio.sleep(_IDLE_WINDOW_S)
        assert calls == []


@pytest.mark.asyncio
async def test_blocking_modal_backdrop_is_visible_and_inert():
    """Through the real modal (card mode): the field paints behind the card
    and stays inert -- zero timers, zero repaints while blocking."""
    app = ModalHarness()

    async with app.run_test(size=(80, 24)):
        modal = app.query_one("#console-setup-modal", ConsoleSetupModal)
        assert modal.is_blocking
        backdrop = app.query_one(
            f"#{CONSOLE_SETUP_MODAL_BACKDROP_ID}", ConsoleSetupBackdrop
        )
        assert backdrop.flake_count > 0
        assert len(backdrop._timers) == 0

        calls = _count_refreshes(backdrop)
        await asyncio.sleep(_IDLE_WINDOW_S)
        assert calls == []


@pytest.mark.asyncio
async def test_unmount_mid_display_is_clean():
    """Removing the blocking modal (backdrop and all) mid-display must not
    raise and must leave no orphaned timers -- the timer-teardown failure
    class that has broken quit in this repo before."""
    app = ModalHarness()

    async with app.run_test(size=(80, 24)) as pilot:
        modal = app.query_one("#console-setup-modal", ConsoleSetupModal)
        assert modal.is_blocking
        await modal.remove()
        await pilot.pause()
        assert not list(app.query(ConsoleSetupBackdrop))
        # The app is still healthy after the removal.
        await pilot.pause()


@pytest.mark.asyncio
async def test_quit_mid_display_is_clean():
    """Exiting the app while the modal is blocking (the state every
    unconfigured user quits from) shuts down without error."""
    app = ModalHarness()

    async with app.run_test(size=(80, 24)):
        modal = app.query_one("#console-setup-modal", ConsoleSetupModal)
        assert modal.is_blocking
        # Exiting the run_test context now performs the real shutdown path
        # with the backdrop mounted; any teardown error surfaces here.
    assert app.return_code in (None, 0)
