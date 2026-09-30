"""The handler-error keep-alive never leaves a dead screen in charge (TASK-33621.13).

GAP4-01 (Console UX review 2026-09-29). TASK-32533's keep-alive in
``TldwCli._handle_exception`` swallows an exception raised inside a pump's own
handler so one panel's bug does not exit the app. Textual still ends that
pump's message loop and detaches it. When the pump was a SCREEN -- the
Conversation Inspector, whose handler awaited ``push_screen_wait`` outside a
worker -- the dead screen stayed on the stack (with the half-opened picker on
top). After the picker closed, every key, click and resize went to the dead
screen, and even Ctrl+Q was ignored because the binding chain was built from
the dead screen's detached focused widget. The same focus trap exists for a
plain widget: when the widget whose pump died held focus, the live screen's
``focused`` still pointed at the detached widget and its binding chain no
longer reached the App.

These tests run the REAL ``TldwCli`` (shared factory) with the keep-alive
switched on exactly as a non-headless session has it, and drive real keys
through the real driver. Nothing on the failure path is stubbed.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest
from textual import on
from textual.app import ComposeResult
from textual.pilot import WaitForScreenTimeout
from textual.screen import ModalScreen, Screen
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app

_CANARY = "zq-dead-screen-canary"
_DIAGNOSTICS_LOGGER = "tldw_chatbook.diagnostics.app"


class _PickerModal(ModalScreen[str | None]):
    """Stands in for the folder picker the Inspector half-opened."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        yield Button("Pick", id="zq-picker-pick")

    def action_cancel(self) -> None:
        self.dismiss(None)


class _WaitingModal(ModalScreen[None]):
    """The GAP4-01 shape: a screen-owned ``@on`` handler awaits a dismissal
    outside any worker. Textual pushes the picker, then raises
    ``NoActiveWorker`` inside this screen's own dispatch."""

    def compose(self) -> ComposeResult:
        yield Button("Choose", id="zq-screen-trigger")

    @on(Button.Pressed, "#zq-screen-trigger")
    async def _choose(self, event: Button.Pressed) -> None:
        event.stop()
        await self.app.push_screen_wait(_PickerModal())


class _RaisingModal(ModalScreen[None]):
    """A screen-owned handler that simply raises while the modal is up."""

    def compose(self) -> ComposeResult:
        yield Button("Boom", id="zq-screen-trigger")

    @on(Button.Pressed, "#zq-screen-trigger")
    def _boom(self, event: Button.Pressed) -> None:
        event.stop()
        raise RuntimeError(_CANARY)


class _RaisingButton(Button):
    """A focused widget whose own handler raises inside a live screen."""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        raise RuntimeError(_CANARY)


class _RecordSink(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def crash_records():
    logger = logging.getLogger(_DIAGNOSTICS_LOGGER)
    sink = _RecordSink()
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(sink)
    try:
        yield sink.messages
    finally:
        logger.removeHandler(sink)
        logger.setLevel(previous)


def _crashes(messages: list[str]) -> list[str]:
    return [m for m in messages if "event=unhandled_exception" in m]


async def _poll(predicate, *, timeout: float = 6.0) -> bool:
    """Poll the clock, not ``pilot.pause()``: a pause queues a callback on
    every widget of ``app.screen`` and waits for all of them, which never
    finishes while a dead pump is on top -- the very state under test."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return bool(predicate())


async def _press(pilot, key: str) -> None:
    """Send one key through the real driver; a dead screen cannot stall us."""
    try:
        await asyncio.wait_for(pilot.press(key), timeout=5)
    except (TimeoutError, WaitForScreenTimeout):
        pass


async def _ready(app, pilot) -> Screen:
    assert await _poll(lambda: getattr(app, "_ui_ready", False), timeout=15), (
        "production TldwCli never reached _ui_ready"
    )
    await pilot.pause()
    return app.screen


def _dead_screens(app) -> list[str]:
    """Screens on the stack whose message loop no longer runs."""
    return [
        type(screen).__name__
        for screen in app.screen_stack
        if not screen.is_running or screen._closing or screen._closed
    ]


def _stack(app) -> list[str]:
    return [type(screen).__name__ for screen in app.screen_stack]


async def _trigger_screen_handler_error(app, pilot, crash_records, modal_cls):
    """Push the modal, press its button with the keyboard, wait for the crash."""
    content = await _ready(app, pilot)
    modal = modal_cls()
    await app.push_screen(modal)
    assert await _poll(
        lambda: app.screen is modal and bool(modal.query("#zq-screen-trigger"))
    )
    await pilot.pause()
    modal.query_one("#zq-screen-trigger", Button).focus()
    await pilot.pause()
    await _press(pilot, "enter")
    assert await _poll(lambda: bool(_crashes(crash_records))), (
        "the handler never raised"
    )
    # Let the keep-alive and Textual's loop-exit settle.
    await asyncio.sleep(0.3)
    return content, modal


async def _assert_ctrl_q_quits(app, pilot) -> None:
    before = (_stack(app), _dead_screens(app), type(app.focused).__name__)
    await _press(pilot, "ctrl+q")
    assert await _poll(lambda: bool(app._exit), timeout=10), (
        "Ctrl+Q did not quit the app after a handler error "
        f"(stack={before[0]}, dead={before[1]}, focused={before[2]})"
    )


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "modal_cls", [_WaitingModal, _RaisingModal], ids=["push-screen-wait", "raise"]
)
async def test_ctrl_q_quits_after_a_screen_handler_error_while_a_modal_is_up(
    request, crash_records, modal_cls
):
    """AC#4. Pre-fix: the dead modal stays on the stack and Ctrl+Q is ignored."""
    app = _build_test_app()
    app._keep_screen_alive_on_handler_error = True
    async with app.run_test(size=(140, 44)) as pilot:
        await _trigger_screen_handler_error(app, pilot, crash_records, modal_cls)
        assert app.is_running and app._exception is None
        # The user's next move in GAP4-01: Esc at whatever is on top.
        await _press(pilot, "escape")
        await asyncio.sleep(0.2)
        await _assert_ctrl_q_quits(app, pilot)


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "modal_cls", [_WaitingModal, _RaisingModal], ids=["push-screen-wait", "raise"]
)
async def test_a_screen_handler_error_leaves_only_live_screens_on_the_stack(
    request, crash_records, modal_cls
):
    """AC#3: the dead screen (and anything it pushed) is gone, the screen
    beneath is live and focused sanely, keys and resize reach it."""
    app = _build_test_app()
    app._keep_screen_alive_on_handler_error = True
    async with app.run_test(size=(140, 44)) as pilot:
        content, modal = await _trigger_screen_handler_error(
            app, pilot, crash_records, modal_cls
        )
        assert await _poll(lambda: app.screen is content), _stack(app)
        assert modal not in app.screen_stack
        assert not any(isinstance(s, _PickerModal) for s in app.screen_stack)
        assert _dead_screens(app) == []
        assert app.is_running and app._exception is None
        # The user is told, without the exception's message.
        messages = [n.message for n in app._notifications]
        assert any("Something went wrong" in m for m in messages), messages
        assert all(_CANARY not in m for m in messages)
        # Keys: the binding chain from the live screen reaches the App.
        focused = content.focused
        assert focused is None or focused.is_attached
        assert any(node is app for node, _ in content._binding_chain)
        # Resize reaches the live screen.
        await pilot.resize_terminal(120, 40)
        assert await _poll(lambda: content.size.width == 120), content.size
        await _assert_ctrl_q_quits(app, pilot)


@private_profile_test
@pytest.mark.asyncio
async def test_a_focused_widget_handler_error_keeps_ctrl_q_reachable(
    request, crash_records
):
    """AC#3 for a plain widget: focus must not stay on the detached widget."""
    app = _build_test_app()
    app._keep_screen_alive_on_handler_error = True
    async with app.run_test(size=(140, 44)) as pilot:
        content = await _ready(app, pilot)
        button = _RaisingButton("boom", id="zq-raising-button")
        await content.mount(button)
        await pilot.pause()
        button.focus()
        await pilot.pause()
        assert app.focused is button
        await _press(pilot, "enter")
        assert await _poll(lambda: bool(_crashes(crash_records)))
        assert await _poll(lambda: button.parent is None)
        assert app.screen is content and _dead_screens(app) == []
        focused = content.focused
        assert focused is not button
        assert focused is None or focused.is_attached
        await _assert_ctrl_q_quits(app, pilot)


# --------------------------------------------------------------------------
# `retire_dead_pump` on a plain Textual app: the two decisions the production
# tests above cannot reach cheaply.
# --------------------------------------------------------------------------


class _Plain(Screen):
    def compose(self) -> ComposeResult:
        yield Button("plain")


@pytest.mark.asyncio
async def test_a_dead_content_screen_over_only_the_placeholder_takes_the_loud_exit():
    """Nothing live would be left in charge, so the keep-alive must not keep
    the app alive: `None` sends ``_handle_exception`` to Textual's exit."""
    from textual.app import App

    from tldw_chatbook.app_keep_alive import retire_dead_pump

    app = App()
    async with app.run_test() as pilot:
        content = _Plain()
        await app.push_screen(content)
        await pilot.pause()
        assert retire_dead_pump(app, content) is None
        assert app.screen is content


@pytest.mark.asyncio
async def test_retiring_a_dead_screen_resumes_a_worker_awaiting_a_screen_above_it():
    """Screens above the dead one are popped with their pending result
    resolved to ``None`` -- a ``push_screen_wait`` in a worker resumes with
    the ordinary cancel value instead of hanging forever."""
    from textual.app import App

    from tldw_chatbook.app_keep_alive import retire_dead_pump

    app = App()
    results: list[object] = []
    async with app.run_test() as pilot:
        content = _Plain()
        await app.push_screen(content)
        dead = _Plain()
        await app.push_screen(dead)
        await pilot.pause()

        async def ask() -> None:
            results.append(await app.push_screen_wait(_PickerModal()))

        worker = app.run_worker(ask())
        assert await _poll(lambda: isinstance(app.screen, _PickerModal))
        assert retire_dead_pump(app, dead) == "screen"
        await asyncio.wait_for(worker.wait(), timeout=5)
        await pilot.pause()
        assert results == [None]
        assert app.screen is content
        assert dead not in app.screen_stack
