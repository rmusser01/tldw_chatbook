"""Console Inspector access below the 150-col compact collapse (TASK-2154.2).

Regression coverage for UX-review findings LY-11 and DS-06
(Docs/superpowers/qa/console-ux-review-2026-08/console-ux-review.md):

- LY-11: below 150 cols the Inspector was force-collapsed and clicking its
  handle silently persisted ``right_open=True`` with zero visual change,
  leaving staged Sources/scope/run inspector unreachable. The force rule is
  now the responsive DEFAULT: an explicit toggle is honored at any width,
  with the main column's min-width waived so the grid always resolves.
- DS-06: the Sources/Tools chips were focusable but inert. Both now open
  the Inspector rail -- at narrow widths they are the ONLY route to it
  (handles hide in single-pane mode).

Both rails' manual toggles must always produce visible feedback.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)


def _stored_rail_preferences(app) -> dict:
    """Return the single stored rail-preference payload for the test workspace."""
    rail_state_config = app.app_config.get("console", {}).get("rail_state", {})
    assert len(rail_state_config) == 1
    return next(iter(rail_state_config.values()))


@pytest.mark.asyncio
async def test_standard_width_inspector_priority_preserves_context_preference():
    """At 120 columns Inspector wins without rewriting Context preference."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-grid")
        await pilot.pause(0.2)

        rail_state = console._current_console_rail_state(available_columns=120)

        assert rail_state.left_open is False
        assert rail_state.right_open is True
        assert rail_state.right_compact_override is True
        assert rail_state.compact_override is True
        assert console.query_one("#console-left-rail").display is False
        assert console.query_one("#console-right-rail").display is True
        assert console.query_one("#console-context-rail-handle").display is True
        assert console.query_one("#console-main-column").styles.min_width.value == 0
        assert not app.app_config.get("console", {}).get("rail_state")


@pytest.mark.asyncio
async def test_context_reveal_switches_from_inspector_at_120_columns():
    """The Context handle persists an exact Inspector-to-Context switch."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-context-rail-open")
        await pilot.pause(0.2)
        console._set_console_rail_preference(right_open=True)
        await pilot.pause(0.2)

        assert console.query_one("#console-left-rail").display is False
        assert console.query_one("#console-right-rail").display is True

        assert await pilot.click("#console-context-rail-open")
        await pilot.pause(0.2)

        assert console.query_one("#console-left-rail").display is True
        assert console.query_one("#console-right-rail").display is False
        stored = _stored_rail_preferences(app)
        assert stored["left_open"] is True
        assert stored["right_open"] is False


@pytest.mark.asyncio
async def test_visible_attach_context_action_switches_rails_without_file_picker():
    """Only the visible Workbench action reveals Context at 120 columns."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-control-attach-context")
        await pilot.pause(0.2)
        file_picker = AsyncMock()
        console._handle_console_attach_context = file_picker
        console._set_console_rail_preference(right_open=True)
        await pilot.pause(0.2)

        assert not list(console.query("#console-attach-context"))
        assert not list(console.query("#console-staged-context-attach"))
        assert console.query_one("#console-control-attach-context").display is True

        await pilot.click("#console-control-attach-context")
        await pilot.pause(0.2)

        assert console.query_one("#console-left-rail").display is True
        assert console.query_one("#console-right-rail").display is False
        stored = _stored_rail_preferences(app)
        assert stored["left_open"] is True
        assert stored["right_open"] is False
        file_picker.assert_not_awaited()


@pytest.mark.asyncio
async def test_inspector_handle_opens_and_collapses_rail_at_140_cols():
    """LY-11: at 140 cols the Inspector handle visibly opens the rail (and
    the in-rail collapse button visibly closes it) -- never a silent
    preference-only change."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-inspector-rail-handle")
        await pilot.pause(0.2)

        right_rail = console.query_one("#console-right-rail")
        handle = console.query_one("#console-inspector-rail-handle")
        assert right_rail.display is False
        assert handle.display is True

        await pilot.click("#console-inspector-rail-open")
        await pilot.pause(0.3)

        assert right_rail.display is True
        assert handle.display is False
        assert _stored_rail_preferences(app)["right_open"] is True
        # Compact override: the honored rail waives the main column's
        # min-width so the grid always resolves below 150 cols.
        main_column = console.query_one("#console-main-column")
        assert main_column.styles.min_width.value == 0
        # AC1: Inspector content is actually reachable.
        assert console.query_one("#console-staged-context-tray").display is True

        await pilot.click("#console-inspector-rail-collapse")
        await pilot.pause(0.3)

        assert right_rail.display is False
        assert handle.display is True
        assert _stored_rail_preferences(app)["right_open"] is False


@pytest.mark.asyncio
async def test_sources_chip_opens_inspector_at_140_cols():
    """DS-06: clicking the Sources chip opens the Inspector rail (the
    staged-sources tray's only surface below 150 cols) and focuses it, so
    activation is visible even when the rail was already open."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-sources-chip")
        await pilot.pause(0.2)

        right_rail = console.query_one("#console-right-rail")
        assert right_rail.display is False

        # The ConsoleHarness does not load the app's chip-width CSS, so the
        # chip can lay out beyond the clickable viewport here; keyboard
        # activation (focus + Enter) exercises the same BINDINGS path, and
        # the real-app click path is covered by the UAT (p7c).
        sources_chip = console.query_one("#console-sources-chip")
        sources_chip.focus()
        await pilot.pause(0.1)
        await pilot.press("enter")
        await pilot.pause(0.3)

        assert right_rail.display is True
        assert console.focused is right_rail
        assert _stored_rail_preferences(app)["right_open"] is True

        # Already open: activation must still give feedback (focus frame).
        sources_chip.focus()
        await pilot.pause(0.1)
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert right_rail.display is True
        assert console.focused is right_rail


@pytest.mark.asyncio
async def test_tools_chip_opens_inspector_at_80x24_single_pane():
    """AC1 at 80x24: single-pane mode hides both handles, so the chips are
    the route to the Inspector; keyboard activation (Enter) opens it
    in-grid while the transcript keeps a usable share."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(80, 24)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await pilot.pause(0.2)

        # Single-pane defaults: both handles hidden, both rails closed.
        assert console.query_one("#console-context-rail-handle").display is False
        assert console.query_one("#console-inspector-rail-handle").display is False
        right_rail = console.query_one("#console-right-rail")
        assert right_rail.display is False

        # Chips clip visually at 80 cols (LY-03), so activate from the
        # keyboard: focus + Enter exercises the chip's BINDINGS path.
        tools_chip = console.query_one("#console-tools-chip")
        tools_chip.focus()
        await pilot.pause(0.1)
        await pilot.press("enter")
        await pilot.pause(0.3)

        assert right_rail.display is True
        assert console.focused is right_rail
        main_column = console.query_one("#console-main-column")
        assert main_column.styles.min_width.value == 0
        transcript_region = console.query_one("#console-transcript-region")
        assert transcript_region.outer_size.width > 0
        # Regression guard (2026-08-05 UAT): toggling the RIGHT rail
        # serializes the full payload including left_open -- without the
        # explicit-toggle marker the left rail must stay force-collapsed
        # here, or the LY-08 single-pane protection is silently lost.
        assert console.query_one("#console-left-rail").display is False
        assert "left_open_explicit" not in _stored_rail_preferences(app)

        # The rail's own collapse button is the way back; activate it from
        # the keyboard because the stripped harness clips narrow controls.
        collapse_button = console.query_one("#console-inspector-rail-collapse", Button)
        collapse_button.focus()
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert right_rail.display is False


@pytest.mark.asyncio
async def test_left_handle_opens_left_rail_at_90_cols():
    """AC2 for the LEFT rail: at 90 cols (below the 100-col compact
    collapse) the Context handle visibly opens the rail -- the write-through
    persists left_open=True plus the explicit-toggle marker even though the
    coerced default is already True, which is what the marker-based
    force-collapse rule honors (ADR-043)."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(90, 24)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-context-rail-handle")
        await pilot.pause(0.2)

        left_rail = console.query_one("#console-left-rail")
        left_handle = console.query_one("#console-context-rail-handle")
        context_button = console.query_one("#console-context-rail-open", Button)
        assert left_rail.display is False
        assert left_handle.display is True
        assert context_button.label == "Context->"
        assert context_button.tooltip == "Open Context rail"

        context_button.focus()
        await pilot.press("enter")
        await pilot.pause(0.3)

        assert left_rail.display is True
        assert left_handle.display is False
        assert _stored_rail_preferences(app)["left_open"] is True
        # The marker (not the left_open value, which matches the default)
        # is what the narrow force-collapse rule honors (ADR-043).
        assert _stored_rail_preferences(app)["left_open_explicit"] is True
        main_column = console.query_one("#console-main-column")
        assert main_column.styles.min_width.value == 0
        transcript_region = console.query_one("#console-transcript-region")
        assert transcript_region.outer_size.width > 0

        collapse_button = console.query_one("#console-context-rail-collapse", Button)
        collapse_button.focus()
        await pilot.press("enter")
        await pilot.pause(0.3)

        assert left_rail.display is False
        assert left_handle.display is True
        assert _stored_rail_preferences(app)["left_open"] is False


@pytest.mark.asyncio
async def test_section_toggle_preserves_left_open_explicit_marker():
    """A later write that did not touch the left rail (a section toggle
    re-serializing the full payload) must not erase the explicit-toggle
    marker, or the honored left rail would silently re-collapse below 100
    cols on the next state build."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(90, 24)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-context-rail-handle")
        await pilot.pause(0.2)

        context_button = console.query_one("#console-context-rail-open", Button)
        context_button.focus()
        await pilot.press("enter")
        await pilot.pause(0.3)
        assert _stored_rail_preferences(app)["left_open_explicit"] is True

        # The Details toggle is scrolled out of view at 90x24; calling the
        # handler directly exercises the same section-only write path
        # (_set_console_rail_preference(section_updates=...)).
        console._toggle_console_rail_section("details")
        await pilot.pause(0.3)

        stored = _stored_rail_preferences(app)
        assert stored["details_open"] is True
        assert stored["left_open_explicit"] is True
        assert console.query_one("#console-left-rail").display is True


@pytest.mark.asyncio
async def test_default_layout_unchanged_at_160_cols():
    """AC4: with no explicit toggles stored, the 160-col default layout is
    exactly what it was -- left rail open, Inspector closed, standard
    main-column min-width."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-left-rail")
        await pilot.pause(0.2)

        assert console.query_one("#console-left-rail").display is True
        assert console.query_one("#console-context-rail-handle").display is False
        assert console.query_one("#console-right-rail").display is False
        assert console.query_one("#console-inspector-rail-handle").display is True
        main_column = console.query_one("#console-main-column")
        assert main_column.styles.min_width.value == 56
        assert "console" not in app.app_config or not app.app_config["console"].get(
            "rail_state"
        )
