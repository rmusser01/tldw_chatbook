# test_recompose_capture_guard.py
# Description: Regression coverage for RecomposeCaptureGuard's exception logging.
"""
PR #905 review (finding 3): ``RecomposeCaptureGuard`` used to log capture
release/sweep failures with ``logger.debug(..., exc_info=True)``. Loguru does
not honor the stdlib ``exc_info`` kwarg -- it is bound as an opaque "extra"
field instead of triggering traceback formatting -- so the traceback was
silently dropped, defeating the whole point of logging it. The fix uses
loguru's own mechanism, ``logger.opt(exception=True)``.
"""

from __future__ import annotations

import asyncio
import io

import pytest
from loguru import logger
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard


@pytest.fixture
def loguru_sink():
    """Capture loguru output into an in-memory buffer for one test."""
    sink = io.StringIO()
    handler_id = logger.add(sink, level="DEBUG", format="{message}")
    try:
        yield sink
    finally:
        logger.remove(handler_id)


class _FailingApp:
    """Minimal stand-in whose capture_mouse always raises, to hit the log line."""

    def capture_mouse(self, widget) -> None:
        raise RuntimeError("capture_mouse boom")


class _GuardHost(RecomposeCaptureGuard):
    """Bare host exercising the mixin's logging without any real Textual app."""

    def __init__(self) -> None:
        self.app = _FailingApp()


def test_release_own_capture_failure_logs_traceback_via_loguru_opt(loguru_sink) -> None:
    """``_release_own_capture_if_any`` must attach a real traceback on failure.

    Regression for the exc_info=True bug: before the fix this assertion is
    RED because loguru's stdlib-style ``exc_info=True`` kwarg is bound as an
    inert "extra" field rather than formatting a traceback, so the emitted
    line never contains ``RuntimeError`` or ``capture_mouse boom`` -- the
    failure is logged with its cause silently discarded.
    """
    host = _GuardHost()

    # _capture_is_within_self must see a captured widget that is "within"
    # self for the guarded capture_mouse(None) call (and its except branch)
    # to run at all.
    host._capture_is_within_self = lambda captured: True  # type: ignore[method-assign]

    host._release_own_capture_if_any(context="before recompose")

    output = loguru_sink.getvalue()
    assert "mouse-capture release before recompose skipped" in output
    assert "RuntimeError" in output
    assert "capture_mouse boom" in output
    assert "Traceback" in output


# ---- Keeping focus across a guarded widget's own rebuild (TASK-33621.12) ----
#
# Live, closing Console's Save .md… prompt re-synced the Conversations tray,
# whose recompose removed the focused row control. Textual's automatic
# `_reset_focus` then moved focus to whatever preceded it in the focus chain --
# the section toggle, "New conversation", even the Console header's Settings
# control, where a following Enter opened Settings. An opted-in guard puts
# focus back on the same-id replacement once its rebuild has mounted.


class _SlowTeardown(Static):
    """Holds the rebuild's teardown open, as a large tray's rebuild does."""

    async def on_unmount(self) -> None:
        await asyncio.sleep(0.2)


class _KeepsFocusRows(RecomposeCaptureGuard, Vertical):
    RECOMPOSE_KEEPS_FOCUS = True
    DEFAULT_CSS = "_KeepsFocusRows { height: auto; }"

    def compose(self) -> ComposeResult:
        yield _SlowTeardown("slow teardown")
        yield Button("Row", id="rebuild-row")


class _DefaultRows(_KeepsFocusRows):
    RECOMPOSE_KEEPS_FOCUS = False


class _RebuildHost(App[None]):
    def __init__(self, rows_class: type[Vertical]) -> None:
        super().__init__()
        self._rows_class = rows_class

    def compose(self) -> ComposeResult:
        # Precedes the rows in the focus chain, so it is where Textual's
        # reset sends focus when the focused row is torn down.
        yield Button("Outside", id="outside")
        yield self._rows_class()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rows_class", "expected_id"),
    [(_KeepsFocusRows, "rebuild-row"), (_DefaultRows, "outside")],
    ids=["opted-in", "default"],
)
async def test_rebuild_returns_focus_to_the_same_id_replacement_when_opted_in(
    rows_class, expected_id
) -> None:
    app = _RebuildHost(rows_class)
    async with app.run_test() as pilot:
        rows = app.query_one(rows_class)
        original = app.query_one("#rebuild-row", Button)
        original.focus()
        await pilot.pause()
        assert app.focused is original

        rows.refresh(recompose=True)
        await pilot.pause(0.6)

        focused = app.focused
        assert focused is not original, "the rebuild did not replace the row"
        assert focused is not None and focused.id == expected_id, focused
        assert rows.recompose_in_flight is False


@pytest.mark.asyncio
async def test_rebuild_never_pulls_focus_that_was_elsewhere_before_it() -> None:
    """Only a rebuild that removed the focused control restores anything."""
    app = _RebuildHost(_KeepsFocusRows)
    async with app.run_test() as pilot:
        rows = app.query_one(_KeepsFocusRows)
        outside = app.query_one("#outside", Button)
        outside.focus()
        await pilot.pause()

        rows.refresh(recompose=True)
        await pilot.pause(0.6)

        assert app.focused is outside
