"""TASK-21134 item 1: the setup-modal snow must not burn a core while idle.

The backdrop covers the whole Console screen, so every repaint re-composites
every line of it. Measured on dev ``68c061984``: 15.8 ms of CPU per tick at
5 Hz -- ~6-15% of a core (load-dependent), burnt continuously by every
not-yet-configured user, against a 0.09% floor with the field frozen. Density
and glyph count are irrelevant to that cost; only the repaint RATE and whether
the repaint also arms a layout pass are.

These tests pin the three mechanics of the fix. Each was mutation-checked
against the pre-fix code.
"""

import random

import pytest
from textual.app import ComposeResult
from textual.widgets import Static

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp

from tldw_chatbook.Widgets.Console import console_setup_modal as setup_modal
from tldw_chatbook.Widgets.Console.console_setup_modal import ConsoleSetupBackdrop


class BackdropHarness(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield ConsoleSetupBackdrop(rng=random.Random(21134))


@pytest.mark.asyncio
async def test_snow_tick_repaints_without_arming_a_layout_pass(monkeypatch):
    """A tick cannot change the field's size, so it must not request layout."""
    app = BackdropHarness()

    async with app.run_test(size=(80, 24)):
        backdrop = app.query_one(ConsoleSetupBackdrop)
        assert backdrop.flake_count > 0

        seen: list[bool] = []
        real_update = Static.update

        def recording_update(self, content="", *, layout: bool = True) -> None:
            if self is backdrop:
                seen.append(layout)
            real_update(self, content, layout=layout)

        monkeypatch.setattr(Static, "update", recording_update)

        backdrop._tick()
        assert seen == [False], f"tick armed a layout pass: {seen}"

        # The resize path really does change the field's dimensions, so it
        # keeps asking for layout.
        seen.clear()
        backdrop._resize_flake_field()
        assert seen == [True], f"resize skipped its layout pass: {seen}"


@pytest.mark.asyncio
async def test_snow_field_is_not_parsed_as_console_markup():
    """The field is spaces plus three glyphs -- markup parsing is pure waste."""
    app = BackdropHarness()

    async with app.run_test(size=(80, 24)):
        backdrop = app.query_one(ConsoleSetupBackdrop)
        assert backdrop._render_markup is False


def test_snow_repaint_rate_is_capped_and_drift_speed_is_preserved():
    """Interval and displacement are a matched pair, not independent knobs.

    Halving the repaint rate only helps if the per-tick displacement grows to
    match; otherwise the fix silently slows the snow down instead of making it
    cheaper. Both halves are asserted so neither can be changed alone.
    """
    interval = setup_modal._SNOW_TICK_INTERVAL

    # Repaint-rate ceiling: the whole-screen repaint may not run faster than
    # 2.5 Hz. The pre-fix 0.2 s (5 Hz) fails here.
    assert interval >= 0.4, f"snow repaints at {1 / interval:.1f} Hz"

    # On-screen drift is unchanged from the original 5 Hz field: flakes still
    # fall at 2.0-7.0 rows/s and wobble up to 2.0 columns/s.
    assert setup_modal._SNOW_MIN_SPEED / interval == pytest.approx(2.0)
    assert setup_modal._SNOW_MAX_SPEED / interval == pytest.approx(7.0)
    assert setup_modal._SNOW_MAX_WOBBLE / interval == pytest.approx(2.0)
