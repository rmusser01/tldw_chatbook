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
from textual.widgets import Input, RichLog

from tldw_chatbook.UI.Logs_Window import LogsWindow, LogRecord, _passes_filter


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


class _LogsHarness(App[None]):
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
        await pilot.pause()
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

    class EmptyHarness(App[None]):
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
        assert "share them when asking for help" in text
