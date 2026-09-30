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
    def __init__(self, rows_class: type[Vertical], *, rows_first: bool = False) -> None:
        super().__init__()
        self._rows_class = rows_class
        self._rows_first = rows_first

    def compose(self) -> ComposeResult:
        if self._rows_first:
            # The row is first in the focus chain, so Textual's reset wraps
            # around to the LAST focusable control ("elsewhere").
            yield self._rows_class()
            yield Button("Outside", id="outside")
        else:
            # Precedes the rows in the focus chain, so it is where Textual's
            # reset sends focus when the focused row is torn down.
            yield Button("Outside", id="outside")
            yield self._rows_class()
        # Follows the rows: never the reset target in the default order --
        # the stand-in for the Console composer a row selection focuses.
        yield Button("Elsewhere", id="elsewhere")


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


async def _wait_for_teardown(rows) -> None:
    # Plain sleeps: `pilot.pause()` first waits for the screen to go idle,
    # which only happens once the whole rebuild has finished.
    for _ in range(100):
        if rows.recompose_in_flight:
            return
        await asyncio.sleep(0.005)
    raise AssertionError("the rebuild never started its teardown")


@pytest.mark.asyncio
async def test_rebuild_leaves_focus_that_moved_on_purpose_during_it() -> None:
    """Focus moved deliberately mid-rebuild is not pulled back to the row.

    Selecting a Console conversation focuses the composer while the tray is
    still rebuilding. The first opt-in refocused the row whenever focus sat
    outside the rebuilt widget, so the composer lost focus and the next keys
    went to the row (2026-09-30 review: 3 of 3 Pilot clicks, 4 of 14 live).
    Only focus still where Textual's reset put it may be taken back.
    """
    app = _RebuildHost(_KeepsFocusRows)
    async with app.run_test() as pilot:
        rows = app.query_one(_KeepsFocusRows)
        elsewhere = app.query_one("#elsewhere", Button)
        app.query_one("#rebuild-row", Button).focus()
        await pilot.pause()

        rows.refresh(recompose=True)
        await _wait_for_teardown(rows)
        elsewhere.focus()
        await pilot.pause(0.6)

        assert rows.recompose_in_flight is False
        assert app.focused is elsewhere, app.focused


@pytest.mark.asyncio
async def test_rebuild_refocuses_after_a_reset_that_wrapped_around() -> None:
    """The row is first in the chain, so the reset wraps to the last control.

    The guard predicts where Textual's reset will land; this pins that the
    prediction follows the wrap-around, or the row would never be restored.
    """
    app = _RebuildHost(_KeepsFocusRows, rows_first=True)
    async with app.run_test() as pilot:
        rows = app.query_one(_KeepsFocusRows)
        original = app.query_one("#rebuild-row", Button)
        original.focus()
        await pilot.pause()

        rows.refresh(recompose=True)
        await _wait_for_teardown(rows)
        assert app.focused is app.query_one("#elsewhere", Button), (
            "Textual's reset no longer wraps to the last control"
        )
        await pilot.pause(0.6)

        focused = app.focused
        assert focused is not original and focused is not None
        assert focused.id == "rebuild-row", focused


@pytest.mark.asyncio
async def test_a_move_queued_behind_a_busy_app_still_wins() -> None:
    """The restore happens as the rebuild ends -- it is never queued.

    ``Widget.focus`` only queues the change on the app (``call_later``), so a
    move requested during the rebuild can be applied after it has finished
    when the app is busy -- traced in the Console, where the composer's
    focus arrived 340 ms into the tray's rebuild. The restore must not join
    that queue behind the move, or it lands last and overrides it.
    """
    app = _RebuildHost(_KeepsFocusRows)
    async with app.run_test() as pilot:
        rows = app.query_one(_KeepsFocusRows)
        elsewhere = app.query_one("#elsewhere", Button)
        app.query_one("#rebuild-row", Button).focus()
        await pilot.pause()

        async def keep_the_app_busy_until_rebuilt() -> None:
            for _ in range(200):
                if not rows.recompose_in_flight:
                    return
                await asyncio.sleep(0.005)

        rows.refresh(recompose=True)
        await _wait_for_teardown(rows)
        app.call_later(keep_the_app_busy_until_rebuilt)
        elsewhere.focus()
        await pilot.pause(0.6)

        assert rows.recompose_in_flight is False
        assert app.focused is elsewhere, app.focused
