"""Console narrow-terminal responsive fallback contracts (TASK-2154.1).

Regression coverage for UX-review findings LY-08/LY-09/LY-10
(Docs/superpowers/qa/console-ux-review-2026-08/console-ux-review.md):

- LY-08: at 80x24 the left rail ate the whole workspace grid and the
  transcript vanished. Below 100 cols the left rail now force-collapses
  (rendering override); below 84 cols the grid drops to a single pane.
- LY-09: at 60x18 the screen was an empty frame; the ready empty-state line
  never rendered. Single-pane mode waives the main column's min-width so
  the transcript always renders.
- TASK-21201: compact-height mode keeps the header and its speech controls,
  while the normal control bar gives back the row those controls used to own.
"""

from __future__ import annotations

from html import unescape
import re
import time
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from textual.widgets import Button, Static, Switch

from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET
from Tests.UI.test_console_internals_decomposition import (
    _configure_native_ready_console,
)
from Tests.UI.test_destination_shells import _visible_text, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings


class ConsoleLayoutHarness(ConsoleHarness):
    """Mount the real Console with the same app bundle production loads."""

    CSS_PATH = str(BUNDLED_STYLESHEET)


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


def _compositor_text(svg: str) -> str:
    """Return only glyphs painted into an exported Textual frame."""
    joined = "".join(re.findall(r"<text[^>]*>([^<]*)</text>", svg))
    return unescape(joined).replace("\xa0", " ")


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
@pytest.mark.parametrize("size", [(60, 18), (80, 24)])
async def test_console_compact_height_keeps_header_controls_and_row_budget(
    size: tuple[int, int],
) -> None:
    """TASK-21201 keeps the header while reclaiming the old speech row."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleLayoutHarness(app)

    async with host.run_test(size=size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-auto-speak")
        shell = console.query_one("#console-shell")
        header = console.query_one("#console-workbench-header")
        bar = console.query_one("#console-control-bar")
        badge = header.query_one("#workbench-header-status", Static)
        auto_speak = console.query_one("#console-auto-speak", Switch)
        hands_free = console.query_one("#console-hands-free-switch", Switch)

        await pilot.pause(0.3)
        assert shell.has_class("-console-compact")
        assert header.display is True
        assert header.region.height == 1, {
            "header": header.region,
            "header_height": header.styles.height,
            "speech_controls": console.query_one("#console-speech-controls").region,
            "speech_height": console.query_one(
                "#console-speech-controls"
            ).styles.height,
            "auto_control": console.query_one("#console-auto-speak-control").region,
            "hands_control": console.query_one("#console-hands-free-control").region,
            "badge": badge.region,
        }
        assert bar.region.height == 1
        assert header.region.height + bar.region.height == 2
        assert auto_speak.region.y == badge.region.y == hands_free.region.y
        assert badge.region.x + badge.region.width == size[0] - 1

        painted = _compositor_text(host.export_screenshot(simplify=True))
        for copy in ("Console", "Speak replies", "Hands-free", _static_text(badge)):
            assert copy in painted


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [60, 90, 140, 235])
@pytest.mark.parametrize("status", ["ready", "running", "blocked"])
async def test_console_header_speech_controls_stay_left_of_status(
    width: int,
    status: str,
) -> None:
    """The subtitle yields first while fixed header controls stay on one row."""
    app = _build_test_app()
    host = ConsoleLayoutHarness(app)

    async with host.run_test(size=(width, 30)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-auto-speak")
        await pilot.pause(0.2)

        header = console.query_one("#console-workbench-header")
        header.sync_state(replace(header.state, status=status, status_label=""))
        await pilot.pause()

        bar = console.query_one("#console-control-bar")
        title = console.query_one("#workbench-header-title", Static)
        subtitle = console.query_one("#workbench-header-subtitle", Static)
        auto_speak_label = console.query_one("#console-auto-speak-label", Static)
        auto_speak = console.query_one("#console-auto-speak", Switch)
        hands_free_label = console.query_one("#console-hands-free-label", Static)
        hands_free = console.query_one("#console-hands-free-switch", Switch)
        badge = console.query_one("#workbench-header-status", Static)
        retry = console.query_one("#console-auto-speak-retry", Button)
        resume = console.query_one("#console-auto-speak-resume", Button)

        assert bar.region.height == 1
        row_y = badge.region.y
        assert all(
            widget.region.y == row_y
            for widget in (
                title,
                subtitle,
                auto_speak_label,
                auto_speak,
                hands_free_label,
                hands_free,
            )
        )
        assert title.region.width > 0
        assert auto_speak.region.width >= 4
        assert hands_free.region.width >= 4
        assert subtitle.region.x + subtitle.region.width <= auto_speak_label.region.x
        assert (
            auto_speak.region.x + auto_speak.region.width <= hands_free_label.region.x
        )
        assert hands_free.region.x + hands_free.region.width + 2 <= badge.region.x
        assert badge.region.x + badge.region.width == width - 1
        assert retry.display is False
        assert resume.display is False


@pytest.mark.asyncio
async def test_console_header_subtitle_yields_width_before_fixed_controls() -> None:
    """Only the descriptive subtitle shrinks as the terminal narrows."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleLayoutHarness(app)

    async with host.run_test(size=(140, 30)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-auto-speak")
        subtitle = console.query_one("#workbench-header-subtitle", Static)
        fixed = tuple(
            console.query_one(selector)
            for selector in (
                "#workbench-header-title",
                "#console-auto-speak-label",
                "#console-auto-speak",
                "#console-hands-free-label",
                "#console-hands-free-switch",
                "#workbench-header-status",
            )
        )
        await pilot.pause()
        wide_subtitle_width = subtitle.region.width
        wide_fixed_widths = tuple(widget.region.width for widget in fixed)

        await pilot.resize_terminal(60, 30)
        await pilot.pause(0.2)

        assert subtitle.region.width < wide_subtitle_width
        assert tuple(widget.region.width for widget in fixed) == wide_fixed_widths
        assert str(subtitle.styles.text_overflow) == "ellipsis"
        painted = _compositor_text(host.export_screenshot(simplify=True))
        assert "Speak replies" in painted
        assert "Hands-free" in painted


@pytest.mark.asyncio
async def test_console_recovery_controls_resize_bar_one_two_one() -> None:
    """Recovery visibility and bar height change atomically without a blank row."""
    app = _build_test_app()
    host = ConsoleLayoutHarness(app)

    async with host.run_test(size=(90, 30)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-auto-speak-retry")
        bar = console.query_one("#console-control-bar")
        retry = console.query_one("#console-auto-speak-retry", Button)
        resume = console.query_one("#console-auto-speak-resume", Button)

        assert bar.region.height == 1
        bar.sync_auto_speak(
            enabled=True,
            paused=True,
            retry_available=True,
        )
        await pilot.pause()

        assert bar.region.height == 2
        assert retry.display is True
        assert resume.display is True
        assert retry.region.width > 0
        assert resume.region.width > 0

        bar.sync_auto_speak(enabled=True, paused=False)
        await pilot.pause()

        assert bar.region.height == 1
        assert retry.display is False
        assert resume.display is False


@pytest.mark.asyncio
async def test_console_retry_speech_button_routes_without_resuming() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(90, 30)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-auto-speak-retry")
        retry = MagicMock()
        resume = MagicMock()
        console._console_auto_speak.request_retry = retry
        console._console_auto_speak.request_resume = resume
        bar = console.query_one("#console-control-bar")
        bar.sync_auto_speak(
            enabled=True,
            paused=True,
            retry_available=True,
        )
        await pilot.pause()

        await pilot.click("#console-auto-speak-retry")
        await pilot.pause()

        retry.assert_called_once_with()
        resume.assert_not_called()


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


# --- TASK-24607 -------------------------------------------------------------


async def _open_inspector_narrow(console, pilot) -> None:
    """Open the Inspector rail and settle layout (narrow-width variant)."""
    right_rail = console.query_one("#console-right-rail")
    if getattr(right_rail, "display", False) and right_rail.region.width > 0:
        return
    await pilot.click("#console-inspector-rail-open")
    await _wait_for_condition(
        pilot,
        lambda: (
            getattr(console.query_one("#console-right-rail"), "display", False)
            and console.query_one("#console-right-rail").region.width > 0
        ),
    )


@pytest.mark.parametrize("size", [(120, 35), (100, 30)])
@pytest.mark.asyncio
async def test_scope_row_never_paints_a_bare_label(size):
    """TASK-24607: the Scope row shows its value or an ellipsis, never bare.

    Live capture at 120 columns rendered ``Scope:`` with nothing after it,
    two rows below the pinned authority block's ``Scope: Everything
    available`` -- the same fact, one rendered and one blank, on screen at
    once. Cause: the row is capped at ``max-height: 1`` while the label
    declared neither ``text-wrap: nowrap`` nor ``text-overflow: ellipsis``,
    so the label wrapped and its second line was clipped away.
    """
    app = _build_test_app()
    # Must run BEFORE mount: the first-run setup modal otherwise blocks the
    # composer and the rail never opens (the harness in test_console_right_
    # rail.py configures readiness the same way, for the same reason).
    _configure_native_ready_console(app)
    host = ConsoleLayoutHarness(app)
    async with host.run_test(size=size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _open_inspector_narrow(console, pilot)

        row = console.query_one("#console-retrieval-scope-row")
        body = console.query_one("#console-inspector-rail-body")
        body.scroll_to_widget(row, animate=False, immediate=True)
        await pilot.pause()

        label = console.query_one("#console-retrieval-scope-label", Static)
        assert _static_text(label) == "Scope: everything"

        # Assert on the LABEL'S OWN painted strip, not the whole frame.
        # Searching the frame is what made the first version of this test pass
        # vacuously: the adjacent "Narrow…" button supplies a "…" anywhere on
        # screen, so a global ellipsis check can never fail.
        assert label.region.height == 1, (
            f"scope label is {label.region.height} rows; the row is capped at "
            "max-height 1, so anything past row 0 is clipped away unseen"
        )
        painted_label = label.render_line(0).text.rstrip()

        assert painted_label.startswith("Scope:"), (
            f"scope label not painted; got {painted_label!r} "
            f"(region={label.region})"
        )
        value = painted_label[len("Scope:") :].strip()
        assert value, (
            "Scope row painted a bare label with no value and no ellipsis at "
            f"width {size[0]}: {painted_label!r} (region={label.region}). The "
            "pinned authority block shows this same fact in full two rows up."
        )
