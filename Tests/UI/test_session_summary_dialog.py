"""SessionSummaryDialog pilot tests (issue #365).

Render evidence: export_screenshot SVG text assertions (lessons-testing-
evidence). Timer evidence: the close callback must actually run.
"""

import asyncio
import time

import pytest

from tldw_chatbook.Chat.session_usage import SessionUsageSnapshot
from tldw_chatbook.Widgets.session_summary_dialog import SessionSummaryDialog

from textual.app import App, ComposeResult
from textual.widgets import Static

# The dialog import chain pulls the guarded config bootstrap, which under
# the per-test sandbox fails closed with RecoveryRequired (the keep-list
# signature in Tests/conftest.py). Keep the bootstrap profile.
pytestmark = pytest.mark.bootstrap_profile


def _visible_text(svg: str) -> str:
    """export_screenshot emits styled words separated by non-breaking
    spaces (``&#160;``); normalize so multi-word copy assertions work."""
    return svg.replace("&#160;", " ").replace("\xa0", " ")


class _DialogApp(App[None]):
    def __init__(self, dialog: SessionSummaryDialog) -> None:
        super().__init__()
        self._dialog = dialog

    def compose(self) -> ComposeResult:
        yield Static("behind")

    def on_mount(self) -> None:
        self.push_screen(self._dialog)


def _dialog(duration: float, snapshot: SessionUsageSnapshot | None = None) -> SessionSummaryDialog:
    return SessionSummaryDialog(
        snapshot or SessionUsageSnapshot(exact_tokens=42318, estimated_tokens=0, calls=3),
        started_at=time.perf_counter() - 4320.0,  # 1h 12m ago
        duration_seconds=duration,
    )


async def test_dialog_renders_summary_copy():
    app = _DialogApp(_dialog(duration=30))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        svg = _visible_text(app.export_screenshot())
        assert "Session summary" in svg
        assert "42,318 tokens" in svg
        assert "session" in svg
        assert "press any key to exit" in svg


async def test_dialog_renders_no_usage_and_estimate_marker():
    empty = _DialogApp(_dialog(duration=30, snapshot=SessionUsageSnapshot()))
    async with empty.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        svg = _visible_text(empty.export_screenshot())
        assert "No usage recorded this session" in svg

    mixed = _DialogApp(
        _dialog(
            duration=30,
            snapshot=SessionUsageSnapshot(exact_tokens=100, estimated_tokens=40, calls=2),
        )
    )
    async with mixed.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        svg = _visible_text(mixed.export_screenshot())
        assert "140 tokens" in svg
        assert "includes estimates" in svg


async def test_any_key_skips_dialog():
    dialog = _dialog(duration=30)
    app = _DialogApp(dialog)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen is not dialog


async def test_dialog_auto_dismisses_after_duration():
    dialog = _dialog(duration=0.05)
    app = _DialogApp(dialog)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        dismissed = False
        for _ in range(60):  # poll, never a fixed pause (lessons)
            if app.screen is not dialog:
                dismissed = True
                break
            await asyncio.sleep(0.05)
        assert dismissed, "auto-close timer never fired"


def _elapsed_dialog(seconds_ago: float, snapshot=None) -> SessionSummaryDialog:
    return SessionSummaryDialog(
        snapshot or SessionUsageSnapshot(exact_tokens=10, estimated_tokens=0, calls=1),
        started_at=time.perf_counter() - seconds_ago,
        duration_seconds=30,
    )


def test_format_elapsed_uses_day_unit_past_24h():
    from tldw_chatbook.Widgets.session_summary_dialog import _format_elapsed

    assert _format_elapsed(25 * 3600) == "1d 1h session"
    assert _format_elapsed(3 * 86400 + 2 * 3600) == "3d 2h session"
    assert _format_elapsed(4320) == "1h 12m session"
    assert _format_elapsed(90) == "1m session"


async def test_dialog_renders_embedding_line():
    app = _DialogApp(
        _dialog(
            duration=30,
            snapshot=SessionUsageSnapshot(
                exact_tokens=100, estimated_tokens=0, calls=2, embeddings_tokens=512
            ),
        )
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        svg = _visible_text(app.export_screenshot())
        assert "100 tokens" in svg
        assert "512 embedding tokens" in svg
