"""Console narrow-terminal responsive fallback contracts (TASK-2154.1).

Regression coverage for UX-review findings LY-08/LY-09/LY-10
(Docs/superpowers/qa/console-ux-review-2026-08/console-ux-review.md):

- LY-08: at 80x24 the left rail ate the whole workspace grid and the
  transcript vanished. Below 100 cols the left rail now force-collapses
  (rendering override); below 84 cols the grid drops to a single pane.
- LY-09: at 60x18 the screen was an empty frame; the ready empty-state line
  never rendered. Single-pane mode waives the main column's min-width so
  the transcript always renders.
- LY-10: compact-height mode (<35 rows) hides the header -- and with it the
  Ready/Running/Blocked badge. A control-bar marker now mirrors that badge
  for exactly as long as the header is hidden.
"""

from __future__ import annotations

import time

import pytest
from textual.widgets import Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_internals_decomposition import (
    _configure_native_ready_console,
)
from Tests.UI.test_destination_shells import _visible_text, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings


async def _wait_for_condition(pilot, predicate, *, timeout: float = 4.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError("Timed out waiting for condition.")


def _static_text(widget: Static) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


@pytest.mark.asyncio
async def test_console_narrow_80x24_keeps_transcript_visible():
    """LY-08: at 80x24 the transcript is the single pane; rails and handles
    no longer eat the grid."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(80, 24)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await pilot.pause(0.2)

        main_column = console.query_one("#console-main-column")
        transcript_region = console.query_one("#console-transcript-region")
        # Single-pane: 80 < CONSOLE_SINGLE_PANE_COLUMNS (84).
        assert main_column.display is True
        assert main_column.styles.min_width.value == 0
        assert transcript_region.outer_size.width >= 40
        assert console.query_one("#console-left-rail").display is False
        assert console.query_one("#console-context-rail-handle").display is False
        assert console.query_one("#console-inspector-rail-handle").display is False


@pytest.mark.asyncio
async def test_console_narrow_90x24_collapses_left_rail_with_handles():
    """LY-08: the narrow band (84-99 cols) keeps the collapse handles; only
    the left rail itself is forced closed."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(90, 24)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        await pilot.pause(0.2)

        main_column = console.query_one("#console-main-column")
        assert console.query_one("#console-left-rail").display is False
        assert console.query_one("#console-context-rail-handle").display is True
        assert console.query_one("#console-inspector-rail-handle").display is True
        assert main_column.display is True
        assert main_column.styles.min_width.value == 56
        assert main_column.outer_size.width >= 56


@pytest.mark.asyncio
async def test_console_narrow_60x18_renders_ready_empty_state():
    """LY-09: at 60x18 the ready empty-state line renders inside the
    transcript instead of the screen being an empty frame."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(60, 18)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")

        # Same readiness recipe as the setup-modal dismissal contract test.
        _configure_native_ready_console(app)
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(provider="llama_cpp", model="local-model"),
        )
        console._sync_console_transcript_guidance()
        await pilot.pause(0.2)

        transcript_region = console.query_one("#console-transcript-region")
        assert transcript_region.outer_size.width > 0
        assert console.query_one("#console-context-rail-handle").display is False
        assert console.query_one("#console-inspector-rail-handle").display is False
        assert "Ready — type a message to begin." in _visible_text(console)


@pytest.mark.asyncio
async def test_console_compact_height_status_marker_mirrors_header_badge():
    """LY-10: below 35 rows the header badge hides; the control-bar marker
    shows the same status label instead."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 24)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-compact-status-marker")
        shell = console.query_one("#console-shell")
        marker = console.query_one("#console-compact-status-marker", Static)
        header = console.query_one("#console-workbench-header")
        badge = header.query_one("#workbench-header-status", Static)

        await _wait_for_condition(
            pilot, lambda: shell.has_class("-console-compact") and marker.display
        )

        badge_text = _static_text(badge)
        assert badge_text
        assert _static_text(marker) == badge_text


@pytest.mark.asyncio
async def test_console_status_marker_hidden_at_normal_height():
    """The marker is a compact-height stand-in only; at normal heights the
    header badge is visible and the marker stays hidden."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-compact-status-marker")
        await pilot.pause(0.3)

        shell = console.query_one("#console-shell")
        marker = console.query_one("#console-compact-status-marker", Static)
        assert not shell.has_class("-console-compact")
        assert marker.display is False


@pytest.mark.asyncio
async def test_console_compact_marker_keeps_control_actions_on_screen():
    """Regression: the marker must hug its content width. A bare Static
    defaults to 1fr and claimed the whole action row at 80x24 during UAT,
    pushing every control button off the right edge of the screen."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(90, 24)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-compact-status-marker")
        shell = console.query_one("#console-shell")
        marker = console.query_one("#console-compact-status-marker", Static)
        await _wait_for_condition(
            pilot, lambda: shell.has_class("-console-compact") and marker.display
        )

        row = console.query_one("#console-control-action-row")
        buttons = [
            child
            for child in row.children
            if getattr(child, "_workbench_action_id", "")
        ]
        assert buttons, "control actions must stay mounted next to the marker"
        for button in buttons:
            assert button.display is True
            assert button.region.y == marker.region.y
            assert button.region.x + button.region.width <= 90


@pytest.mark.asyncio
async def test_console_live_resize_narrowing_collapses_left_rail():
    """A live resize across the 100-col threshold re-evaluates the rail
    rules even though no console sync tick fires in between."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-left-rail")
        left_rail = console.query_one("#console-left-rail")
        left_handle = console.query_one("#console-context-rail-handle")
        await pilot.pause(0.2)
        assert left_rail.display is True
        assert left_handle.display is False

        await pilot.resize_terminal(90, 42)
        await _wait_for_condition(pilot, lambda: left_rail.display is False)
        assert left_handle.display is True

        await pilot.resize_terminal(140, 42)
        await _wait_for_condition(pilot, lambda: left_rail.display is True)
        assert left_handle.display is False
