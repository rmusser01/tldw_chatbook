"""Regression tests for the Logs rebuild (UX-032..UX-038).

Covers the diagnose-and-share loop: level/text filtering, pause/resume,
copy-visible vs copy-all, empty state, status line honesty, and the
structured record intake from the app's persistent handler.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Input, RichLog

from tldw_chatbook.UI.Logs_Window import (
    FILTER_DEBOUNCE_SECONDS,
    MAX_RENDERED_LINES,
    LogsWindow,
    LogRecord,
    _passes_filter,
)


def _records():
    return deque(
        [
            ("INFO", "app.boot", "2026-08-03 09:00:00 - app.boot - INFO - boot ok"),
            (
                "WARNING",
                "app.sync",
                "2026-08-03 09:01:00 - app.sync - WARNING - slow pull",
            ),
            (
                "ERROR",
                "app.tts",
                "2026-08-03 09:02:00 - app.tts - ERROR - voice failed",
            ),
        ],
        maxlen=10000,
    )


def _fake_app_instance():
    return SimpleNamespace(_log_records=_records())


class _LogsHarness(ConsolidatedCSSApp):
    def __init__(self):
        super().__init__()
        self.logs_window = LogsWindow(_fake_app_instance(), id="logs-window")
        self._copied = ""

    def compose(self) -> ComposeResult:
        yield self.logs_window

    def copy_to_clipboard(self, text: str) -> None:
        self._copied = text


def test_passes_filter_level_and_text() -> None:
    error = LogRecord("ERROR", "mod", "boom")
    info = LogRecord("INFO", "mod", "fine")
    assert _passes_filter(error, "all", "")
    assert _passes_filter(error, "error", "")
    assert not _passes_filter(info, "error", "")
    assert _passes_filter(info, "info", "")
    assert _passes_filter(info, "all", "FIN")
    assert not _passes_filter(info, "all", "missing")


@pytest.mark.asyncio
async def test_loads_records_and_renders_all_by_default() -> None:
    app = _LogsHarness()
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        app.logs_window.load_from_app()
        await pilot.pause()
        lines = app.query_one("#app-log-display", RichLog).lines
        assert len(lines) == 3
        status = str(app.query_one("#logs-status-line").render())
        assert "Showing 3 of 3 lines" in status
        # Chip counts reflect the buffer.
        assert "1" in str(app.query_one("#logs-filter-error").label)


@pytest.mark.asyncio
async def test_level_chip_filters_view_and_copy_visible() -> None:
    app = _LogsHarness()
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        window = app.logs_window
        window.load_from_app()
        window._level_chip = "error"
        window._render_view()
        await pilot.pause()
        lines = app.query_one("#app-log-display", RichLog).lines
        assert len(lines) == 1
        assert "voice failed" in lines[0].text
        window._on_copy_visible()
        assert "voice failed" in app._copied
        assert "boot ok" not in app._copied


@pytest.mark.asyncio
async def test_text_filter_narrows_view() -> None:
    app = _LogsHarness()
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        window = app.logs_window
        window.load_from_app()
        app.query_one("#logs-filter-text", Input).value = "sync"
        # Debounced (task-15476): the view only re-renders once the filter
        # settles, not on every keystroke.
        await pilot.pause(FILTER_DEBOUNCE_SECONDS + 0.1)
        lines = app.query_one("#app-log-display", RichLog).lines
        assert len(lines) == 1
        assert "slow pull" in lines[0].text


@pytest.mark.asyncio
async def test_pause_buffers_new_records_and_resume_flushes() -> None:
    app = _LogsHarness()
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        window = app.logs_window
        window.load_from_app()
        window._paused = True
        window.append_record("INFO", "app.late", "late line while paused")
        await pilot.pause()
        lines = app.query_one("#app-log-display", RichLog).lines
        assert len(lines) == 3  # unchanged while paused
        status = str(app.query_one("#logs-status-line").render())
        assert "paused" in status and "1 new" in status
        window._paused = False
        window._pending_while_paused = 0
        window._render_view()
        await pilot.pause()
        lines = app.query_one("#app-log-display", RichLog).lines
        assert len(lines) == 4


@pytest.mark.asyncio
async def test_empty_state_shows_guidance() -> None:
    class EmptyWindow(LogsWindow):
        pass

    class EmptyHarness(ConsolidatedCSSApp):
        def compose(self) -> ComposeResult:
            yield LogsWindow(SimpleNamespace(_log_records=deque(maxlen=10000)))

    app = EmptyHarness()
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        app.query_one(LogsWindow).load_from_app()
        await pilot.pause()
        empty = app.query_one("#logs-empty-state")
        assert empty.display is True
        text = str(empty.render())
        assert "No log entries yet" in text
        # TASK-19555: the guidance must describe the artifact it is inviting
        # the user to share. It used to say "copy the logs and share them"
        # with no mention that the payload carried their file names, note
        # titles and search terms -- and, before the sink-side redaction, any
        # API key that had been logged.
        assert "Copy visible logs" in text
        assert "file names" in text
        assert "read before you share" in text


# ---------------------------------------------------------------------------
# task-15476 AC #2: capped rendered slice + truncation disclosure.
# ---------------------------------------------------------------------------


def _many_matching_records(count: int) -> deque:
    """``count`` INFO records, oldest (lowest index) first -- mirrors the
    buffer's own append order, so ``line {count - 1}`` is the most recent."""
    return deque(
        (
            (
                "INFO",
                "app.bulk",
                f"2026-08-11 00:00:00 - app.bulk - INFO - line {i}",
            )
            for i in range(count)
        ),
        maxlen=10000,
    )


class _ManyRecordsHarness(ConsolidatedCSSApp):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.logs_window = LogsWindow(
            SimpleNamespace(_log_records=_many_matching_records(count)),
            id="logs-window",
        )

    def compose(self) -> ComposeResult:
        yield self.logs_window


@pytest.mark.asyncio
async def test_a_filter_matching_more_than_the_cap_renders_only_the_most_recent_slice() -> (
    None
):
    """A render pass must clear+rewrite the RichLog with at most
    `MAX_RENDERED_LINES` lines, even when far more records match -- the
    defect this task fixes rewrote it with ALL matches, unbounded."""
    total = MAX_RENDERED_LINES + 50
    app = _ManyRecordsHarness(total)
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        app.logs_window.load_from_app()
        await pilot.pause()

        lines = app.query_one("#app-log-display", RichLog).lines
        assert len(lines) == MAX_RENDERED_LINES
        # The MOST RECENT matches are kept (highest indices -- the last
        # ones appended), not the oldest: the earliest 50 records (0..49)
        # are trimmed, leaving 50..1049 as the exact rendered window.
        first_kept_index = total - MAX_RENDERED_LINES
        assert lines[0].text.endswith(f"line {first_kept_index}")
        assert lines[-1].text.endswith(f"line {total - 1}")


@pytest.mark.asyncio
async def test_the_status_line_discloses_the_render_cap_truncation() -> None:
    """Honest accounting (task-15476 AC #2): when the cap trims output, the
    status line must say so, not just silently under-report."""
    total = MAX_RENDERED_LINES + 50
    app = _ManyRecordsHarness(total)
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        app.logs_window.load_from_app()
        await pilot.pause()

        status = str(app.query_one("#logs-status-line").render())
        assert f"Showing {MAX_RENDERED_LINES} of {total} lines" in status
        assert f"filter matched {total}" in status
        assert f"showing most recent {MAX_RENDERED_LINES}" in status


@pytest.mark.asyncio
async def test_a_filter_within_the_cap_discloses_no_truncation() -> None:
    """Below the cap, the status line must not claim a truncation that
    didn't happen."""
    app = _ManyRecordsHarness(5)
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        app.logs_window.load_from_app()
        await pilot.pause()

        lines = app.query_one("#app-log-display", RichLog).lines
        assert len(lines) == 5
        status = str(app.query_one("#logs-status-line").render())
        assert "Showing 5 of 5 lines" in status
        assert "filter matched" not in status


@pytest.mark.asyncio
async def test_next_error_jump_targets_only_rendered_rows_when_capped() -> None:
    """n/N error-jump indices must be computed against what's actually in
    the RichLog (the capped slice), not the full (larger) match set --
    otherwise a jump could target a row the widget never rendered."""
    total = MAX_RENDERED_LINES + 20
    app = _ManyRecordsHarness(total)
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        app.logs_window.load_from_app()
        await pilot.pause()

        # No ERROR/CRITICAL records were seeded -- the jump must find none
        # rather than raising or indexing past the rendered slice.
        assert app.logs_window._error_row_indices() == []
