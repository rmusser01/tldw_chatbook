# test_text_selection_crash_guard.py
# Description: task-14903 -- Textual's text-selection MouseDown crash during
# relayout, and the app-level guard that keeps it from killing the app.
"""
Observed live (task-4023 verification): a click on the Library Search/RAG
canvas's ``#library-rag-query-quiet-line`` Static ~1s after a 50->24-row
terminal resize died inside Textual 8.2.8's ``Screen._forward_event`` at the
text-selection begin block::

    event.screen_offset - container.region.offset
    AttributeError: 'NoneType' object has no attribute 'region'   (container=None)

and the AttributeError propagated out of ``App.on_event`` and terminated the
whole application.

Root cause (attributed by reading Textual 8.2.8): the MouseDown selection
begin resolves the clicked widget via the COMPOSITOR's cached map, which is
only rebuilt at the next reflow -- a widget pruned mid-recompose
(``parent is None``) stays resolvable in the stale map. A detached widget's
``Widget.region`` swallows ``NoScreen``/``NoWidget`` into ``NULL_REGION``,
which forces ``get_widget_and_offset_at`` into its coordinate-clamp branch
and returns a NON-``None`` offset, so ``_forward_event`` takes the content
path and dereferences ``content_widget.parent`` (``None``).

These tests reproduce that state deterministically: prune the widget
(``await widget.remove()`` -- prune complete, reflow still pending, i.e. the
exact mid-recompose window) and drive a ``MouseDown`` through
``App.on_event``, the same dispatcher call the live crash traversed.
"""

from __future__ import annotations

import io

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.widgets import Static

from loguru import logger

from tldw_chatbook.Utils.text_selection_crash_guard import (
    TextSelectionCrashGuard,
    match_selection_begin_container_crash,
)

CRASH_SIGNATURE = "'NoneType' object has no attribute 'region'"


class _VanillaApp(App[None]):
    """Plain Textual app: exhibits the upstream crash unguarded."""

    def compose(self) -> ComposeResult:
        yield Static("The quick brown fox jumps over the lazy dog", id="quiet-line")


class _GuardedApp(TextSelectionCrashGuard, App[None]):
    """Same app with the shipped guard mixin, mirroring TldwCli's base list."""

    def compose(self) -> ComposeResult:
        yield Static("The quick brown fox jumps over the lazy dog", id="quiet-line")


def _mouse_down(x: int = 3, y: int = 0) -> events.MouseDown:
    return events.MouseDown(
        widget=None,
        x=x,
        y=y,
        delta_x=0,
        delta_y=0,
        button=1,
        shift=False,
        meta=False,
        ctrl=False,
    )


async def _enter_mid_recompose_crash_state(app: App[None], pilot) -> Static:
    """Prune the Static but stop before the reflow -- the live crash window.

    Precondition asserts pin the attribution: after the prune the widget is
    detached (``parent is None``) yet the compositor's STALE map still
    resolves it at the click position with a non-``None`` offset -- exactly
    the state that sends ``_forward_event`` down the content path into
    ``container = content_widget.parent`` (``None``).
    """
    await pilot.pause()
    static = app.query_one("#quiet-line", Static)

    # Healthy state: the compositor resolves the widget with a real offset.
    widget, offset = app.screen.get_widget_and_offset_at(3, 0)
    assert widget is static and offset is not None

    # Prune (what recompose does to every child) WITHOUT a pause afterward:
    # the removal has completed but the compositor reflow has not yet run.
    await static.remove()
    assert static.parent is None

    # The stale compositor map still resolves the detached widget, and the
    # NULL_REGION clamp hands back a non-None offset: crash state reached.
    stale_widget, stale_offset = app.screen.get_widget_and_offset_at(3, 0)
    assert stale_widget is static
    assert stale_offset is not None
    return static


@pytest.fixture
def loguru_sink():
    """Capture loguru output into an in-memory buffer for one test."""
    sink = io.StringIO()
    handler_id = logger.add(sink, level="DEBUG", format="{message}")
    try:
        yield sink
    finally:
        logger.remove(handler_id)


# --------------------------------------------------------------------------
# Reproduction pin: the upstream bug, unguarded.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mousedown_on_mid_recompose_widget_crashes_vanilla_textual():
    """Pin the upstream crash signature on the installed Textual.

    This is the task-14903 reproduction: without the guard, the MouseDown
    raises the exact live-crash AttributeError out of ``App.on_event`` --
    which in a real run reaches ``_handle_exception`` and kills the app.
    If a future Textual bump makes this test FAIL (no crash), the upstream
    bug is fixed and ``TextSelectionCrashGuard`` can be retired.
    """
    app = _VanillaApp()
    async with app.run_test(size=(60, 10)) as pilot:
        await _enter_mid_recompose_crash_state(app, pilot)

        with pytest.raises(AttributeError, match=CRASH_SIGNATURE) as exc_info:
            await app.on_event(_mouse_down())

        # The raising frame is Textual's selection-begin block with the
        # detached parent -- the exact signature the guard keys on.
        assert (
            match_selection_begin_container_crash(exc_info.value, _mouse_down())
            is not None
        )


# --------------------------------------------------------------------------
# The guard: the app survives the click, logs the drop, and stays usable.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guard_drops_the_crashing_click_and_app_survives(loguru_sink):
    app = _GuardedApp()
    async with app.run_test(size=(60, 10)) as pilot:
        await _enter_mid_recompose_crash_state(app, pilot)

        # Without the guard this call raises (previous test); with it, the
        # click is dropped and nothing escapes toward _handle_exception.
        await app.on_event(_mouse_down())

        # The guard completed Textual's own "not selectable" branch.
        assert app.screen._select_state is None

        # The drop is visible in the log. TASK-15103 (ADR-029): the event is
        # fixed text — a widget repr can embed rendered user content and the
        # coordinates are input telemetry, so neither is echoed any more.
        output = loguru_sink.getvalue()
        assert "task-14903" in output
        assert "quiet-line" not in output

        # The app is still alive and interactive: after the pending reflow
        # runs, a freshly mounted widget accepts a normal selection click.
        await pilot.pause()
        await app.screen.mount(Static("hello again", id="fresh-line"))
        await pilot.pause()
        fresh = app.query_one("#fresh-line", Static)
        widget, offset = app.screen.get_widget_and_offset_at(2, 0)
        assert widget is fresh and offset is not None
        await app.on_event(_mouse_down(x=2, y=0))
        assert app.screen._select_state is not None


@pytest.mark.asyncio
async def test_guard_does_not_break_normal_text_selection():
    """A NORMAL MouseDown must still start text selection through the guard."""
    app = _GuardedApp()
    async with app.run_test(size=(60, 10)) as pilot:
        await pilot.pause()
        static = app.query_one("#quiet-line", Static)
        assert app.screen._select_state is None

        await app.on_event(_mouse_down())

        select_state = app.screen._select_state
        assert select_state is not None
        assert select_state.start.content_widget is static
        assert select_state.start.container is static.parent


# --------------------------------------------------------------------------
# Narrowness: anything that is not EXACTLY the signature must re-raise.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guard_reraises_lookalike_errors_from_other_frames():
    """Same message, same event type -- but raised outside Textual's
    ``screen.py::_forward_event`` frame: the guard must NOT eat it."""
    app = _GuardedApp()
    async with app.run_test(size=(60, 10)) as pilot:
        await pilot.pause()

        def _explode(event: events.Event) -> None:
            raise AttributeError(CRASH_SIGNATURE)

        app.screen._forward_event = _explode  # type: ignore[method-assign]
        with pytest.raises(AttributeError, match=CRASH_SIGNATURE):
            await app.on_event(_mouse_down())


@pytest.mark.asyncio
async def test_matcher_rejects_non_mousedown_events():
    """The real crash exception matched against a non-MouseDown event must
    not match: only the observed MouseDown signature is guarded."""
    app = _VanillaApp()
    async with app.run_test(size=(60, 10)) as pilot:
        await _enter_mid_recompose_crash_state(app, pilot)
        with pytest.raises(AttributeError, match=CRASH_SIGNATURE) as exc_info:
            await app.on_event(_mouse_down())

    mouse_up = events.MouseUp(
        widget=None,
        x=3,
        y=0,
        delta_x=0,
        delta_y=0,
        button=1,
        shift=False,
        meta=False,
        ctrl=False,
    )
    assert match_selection_begin_container_crash(exc_info.value, mouse_up) is None


def test_tldwcli_carries_the_guard_before_app():
    """Wiring pin: TldwCli must inherit the guard, ahead of App in the MRO,
    or the live app is unprotected no matter what the mixin tests prove."""
    from textual.app import App as TextualApp

    from tldw_chatbook.app import TldwCli

    mro = TldwCli.__mro__
    assert TextSelectionCrashGuard in mro
    assert mro.index(TextSelectionCrashGuard) < mro.index(TextualApp)


def test_matcher_rejects_non_attribute_errors():
    assert (
        match_selection_begin_container_crash(RuntimeError("boom"), _mouse_down())
        is None
    )


def test_matcher_rejects_errors_without_traceback():
    assert (
        match_selection_begin_container_crash(
            AttributeError(CRASH_SIGNATURE), _mouse_down()
        )
        is None
    )
