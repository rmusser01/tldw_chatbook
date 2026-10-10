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
from textual.app import App, ComposeResult
from textual.pilot import WaitForScreenTimeout
from textual.screen import ModalScreen, Screen
from textual.widget import Widget
from textual.widgets import Button
from textual.worker import WorkerState

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


class _PlainModal(ModalScreen[None]):
    """A live modal with nothing of its own to fail."""

    BINDINGS = [("escape", "close", "Close")]

    def compose(self) -> ComposeResult:
        yield Button("ok", id="zq-plain-ok")

    def action_close(self) -> None:
        self.dismiss(None)


def _boom() -> None:
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


async def _poll(predicate, *, timeout: float = 20.0) -> bool:
    """Poll the clock, not ``pilot.pause()``: a pause queues a callback on
    every widget of ``app.screen`` and waits for all of them, which never
    finishes while a dead pump is on top -- the very state under test. The
    timeouts are generous on purpose: they bound a failure, never a pass, and
    a run on a heavily loaded machine took half again as long as usual."""
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
    assert await _poll(
        lambda: getattr(app, "_ui_ready", False), timeout=60
    ), "production TldwCli never reached _ui_ready"
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
    assert await _poll(
        lambda: bool(_crashes(crash_records))
    ), "the handler never raised"
    # Let the keep-alive and Textual's loop-exit settle.
    await asyncio.sleep(0.3)
    return content, modal


async def _assert_ctrl_q_quits(app, pilot) -> None:
    before = (_stack(app), _dead_screens(app), type(app.focused).__name__)
    await _press(pilot, "ctrl+q")
    assert await _poll(lambda: bool(app._exit), timeout=20), (
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
@pytest.mark.parametrize(
    ("where", "how"),
    [("content", "call_after_refresh"), ("modal", "call_next")],
    ids=["content-call-after-refresh", "modal-call-next"],
)
async def test_a_callback_error_on_a_live_screen_keeps_that_screen(
    request, crash_records, where, how
):
    """Only a failed DISPATCH (or mount) ends a pump's message loop. A failed
    ``call_after_refresh``/``call_next`` callback leaves an inner loop and the
    screen keeps running, so the keep-alive must neither pop it nor exit the
    app (the TASK-32533 contract; Console alone has dozens of such callbacks).
    """
    app = _build_test_app()
    app._keep_screen_alive_on_handler_error = True
    async with app.run_test(size=(140, 44)) as pilot:
        content = await _ready(app, pilot)
        target = content
        if where == "modal":
            target = _PlainModal()
            await app.push_screen(target)
            assert await _poll(lambda: app.screen is target)
            await pilot.pause()
        stack_before = list(app.screen_stack)
        getattr(target, how)(_boom)
        assert await _poll(
            lambda: bool(_crashes(crash_records))
        ), "the callback never raised"
        await asyncio.sleep(0.3)
        assert app.is_running and not app._exit and app._exception is None
        assert list(app.screen_stack) == stack_before, _stack(app)
        assert target.is_running and _dead_screens(app) == []
        # notify() posts asynchronously; a live screen does not imply delivery.
        assert await _poll(
            lambda: any("kept running" in n.message for n in app._notifications)
        ), "the keep-alive notification was not delivered"
        messages = [n.message for n in app._notifications]
        assert any("kept running" in m for m in messages), messages
        assert not any("was closed" in m for m in messages), messages
        assert all(_CANARY not in m for m in messages)
        if where == "modal":
            # Still live: it handles its own key. (A live modal also keeps
            # the App's non-priority Ctrl+Q out of the chain, as it always has.)
            await _press(pilot, "escape")
            assert await _poll(lambda: app.screen is content), _stack(app)
        await _assert_ctrl_q_quits(app, pilot)


@private_profile_test
@pytest.mark.asyncio
async def test_a_dead_content_screen_takes_textuals_loud_exit(request, crash_records):
    """The content screen's own dispatch failed and only Textual's blank
    placeholder is beneath it: nothing live could be left in charge, so the
    app exits the way Textual always did instead of freezing."""
    app = _build_test_app()
    app._keep_screen_alive_on_handler_error = True
    with pytest.raises(RuntimeError, match=_CANARY):
        async with app.run_test(size=(140, 44)) as pilot:
            content = await _ready(app, pilot)
            content.call_later(_boom)
            assert await _poll(lambda: app._exception is not None, timeout=20)
    assert _crashes(crash_records)
    assert app.return_code == 1


@private_profile_test
@pytest.mark.asyncio
async def test_a_suspended_reusable_screen_that_dies_is_rebuilt_on_the_next_visit(
    request, crash_records
):
    """Console is a reusable route: its installed instance survives navigation
    and keeps processing messages while suspended. If it dies there, the
    cache must not hand the dead instance back -- that is the GAP4-01 wedge
    again (focus on a detached widget, Ctrl+Q ignored)."""
    app = _build_test_app()
    app._keep_screen_alive_on_handler_error = True
    async with app.run_test(size=(170, 48)) as pilot:
        console = await _ready(app, pilot)
        assert type(console).__name__ == "ChatScreen"
        await _press(pilot, "ctrl+1")
        assert await _poll(
            lambda: app.screen is not console and app.screen.is_running, timeout=20
        ), _stack(app)
        await asyncio.sleep(0.5)
        assert console.is_running and app.is_screen_installed(console)

        console.call_later(_boom)
        assert await _poll(lambda: bool(_crashes(crash_records)))
        assert await _poll(lambda: not console.is_running)
        assert app.is_running and app._exception is None
        assert not app.is_screen_installed(console)
        cache = getattr(app, "_reusable_screen_instances", {})
        assert all(screen is not console for _identity, screen in cache.values())

        await _press(pilot, "ctrl+2")
        assert await _poll(
            lambda: (
                type(app.screen).__name__ == "ChatScreen" and app.screen is not console
            ),
            timeout=20,
        ), _stack(app)
        fresh = app.screen
        assert await _poll(
            lambda: fresh.is_running and bool(fresh.query("#console-native-composer")),
            timeout=20,
        )
        assert _dead_screens(app) == []
        focused = fresh.focused
        assert focused is None or focused.is_attached
        assert any(node is app for node, _ in fresh._binding_chain)
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


#: The traceback head ``_handle_exception`` sees when a pump's own message
#: dispatch raised: the loop that caught it, then the dispatch it called.
_DISPATCH_DEATH = [
    ("textual.message_pump", "_process_messages_loop", 1),
    ("textual.message_pump", "_dispatch_message", 2),
]


@pytest.mark.asyncio
async def test_a_dead_content_screen_over_only_the_placeholder_takes_the_loud_exit():
    """Nothing live would be left in charge, so the keep-alive must not keep
    the app alive: `None` sends ``_handle_exception`` to Textual's exit."""
    from tldw_chatbook.app_keep_alive import retire_dead_pump

    app = App()
    async with app.run_test() as pilot:
        content = _Plain()
        await app.push_screen(content)
        await pilot.pause()
        assert retire_dead_pump(app, content, _DISPATCH_DEATH) is None
        assert app.screen is content


@pytest.mark.asyncio
async def test_a_recovery_that_raises_is_logged_before_the_loud_exit(monkeypatch):
    """PR #2945 review: ``except Exception: return None`` left no record, so a
    recovery that crashed and a stack with nothing live left took the same
    exit and could not be told apart in the log. The crash is logged as a
    warning naming its class and where it was raised -- identifiers only,
    never the message; the honest ``None`` logs nothing."""
    from loguru import logger

    from tldw_chatbook import app_keep_alive

    def broken_discard(_app, _screen):
        raise RuntimeError(_CANARY)

    raise_line = broken_discard.__code__.co_firstlineno + 1
    lines: list[str] = []
    sink_id = logger.add(
        lambda message: lines.append(message.record["message"]),
        level="WARNING",
        format="{message}",
        diagnose=False,
    )
    try:
        app = App()
        async with app.run_test() as pilot:
            content = _Plain()
            await app.push_screen(content)
            await pilot.pause()
            # Nothing live left: the loud exit, and nothing to report.
            assert (
                app_keep_alive.retire_dead_pump(app, content, _DISPATCH_DEATH) is None
            )
            assert lines == []
            # The recovery itself raises: the same exit, but on the record.
            monkeypatch.setattr(app_keep_alive, "_discard_dead_screen", broken_discard)
            assert (
                app_keep_alive.retire_dead_pump(app, content, _DISPATCH_DEATH) is None
            )
    finally:
        logger.remove(sink_id)
    assert len(lines) == 1, lines
    assert lines[0].startswith("Dead pump recovery failed: RuntimeError at "), lines
    assert f"broken_discard:{raise_line}" in lines[0], lines[0]
    assert "tldw_chatbook.app_keep_alive.retire_dead_pump:" in lines[0], lines[0]
    assert _CANARY not in lines[0]


@pytest.mark.asyncio
async def test_retiring_a_dead_screen_resumes_a_worker_awaiting_a_screen_above_it():
    """Screens above the dead one are popped with their pending result
    resolved to ``None`` -- a ``push_screen_wait`` in a worker resumes with
    the ordinary cancel value instead of hanging forever."""
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
        assert retire_dead_pump(app, dead, _DISPATCH_DEATH) == "screen"
        await asyncio.wait_for(worker.wait(), timeout=5)
        await pilot.pause()
        assert results == [None]
        assert app.screen is content
        assert dead not in app.screen_stack


class _MountRaiser(Button):
    def on_mount(self) -> None:
        raise RuntimeError(_CANARY)


class _RecordingApp(App):
    """Records every error's pump and frames, and keeps running."""

    def __init__(self) -> None:
        super().__init__()
        self.errors: list[tuple[object, list]] = []

    def _handle_exception(self, error: Exception) -> None:
        from textual.message_pump import active_message_pump

        from tldw_chatbook.app_lifecycle import _exception_frames

        self.errors.append((active_message_pump.get(None), _exception_frames(error)))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("how", "ends"),
    [
        ("call_later", True),
        ("mount", True),
        ("call_next", False),
        ("call_after_refresh", False),
    ],
)
async def test_pump_loop_ended_matches_what_textual_does_to_the_pump(how, ends):
    """``pump_loop_ended`` reads the traceback; this pins it to what Textual
    actually does. A handler error in dispatch (``call_later`` runs through
    ``_dispatch_message``) or in mount ends the pump; a ``call_next`` or
    ``call_after_refresh`` callback error leaves it running. If a Textual
    upgrade moves where the loop catches, this is the test that says so."""
    from tldw_chatbook.app_keep_alive import pump_loop_ended

    app = _RecordingApp()
    async with app.run_test() as pilot:
        screen = _Plain()
        await app.push_screen(screen)
        await pilot.pause()
        if how == "mount":
            await screen.mount(_MountRaiser("mount"))
        else:
            getattr(screen.query_one(Button), how)(_boom)
        assert await _poll(lambda: bool(app.errors))
        await asyncio.sleep(0.2)
        pump, frames = app.errors[0]
        assert pump_loop_ended(frames) is ends, frames
        assert pump.is_running is (not ends)


# --------------------------------------------------------------------------
# A retired screen is torn down whichever loop-ending path killed it.
# --------------------------------------------------------------------------


class _Ticker(Widget):
    """A child whose own interval ticks, and whose own worker runs, for as
    long as its pump lives."""

    def __init__(self, *, unmount_raises: bool = False) -> None:
        super().__init__()
        self.unmount_raises = unmount_raises
        self.ticks = 0
        self.worker = None

    def on_mount(self) -> None:
        self.set_interval(0.01, self._tick)
        self.worker = self.run_worker(asyncio.sleep(3600))

    def on_unmount(self) -> None:
        if self.unmount_raises:
            raise RuntimeError(_CANARY)

    def _tick(self) -> None:
        self.ticks += 1


class _DyingScreen(Screen):
    """Mounts a ticking child, ticks itself and starts a worker, then dies:
    in ``on_mount`` (Textual's ``_pre_process``) or, once mounted, in a
    dispatched handler (``_process_messages_loop``). ``unmount_raises`` names
    whose ``on_unmount`` raises during the teardown: ``"screen"`` or
    ``"child"``."""

    def __init__(self, death: str, *, unmount_raises: str | None = None) -> None:
        super().__init__()
        self.death = death
        self.unmount_raises = unmount_raises
        self.ticks = 0
        self.mounted = False
        self.ticker: _Ticker | None = None
        self.worker = None

    def compose(self) -> ComposeResult:
        yield _Ticker(unmount_raises=self.unmount_raises == "child")

    async def on_mount(self) -> None:
        self.ticker = self.query_one(_Ticker)
        self.set_interval(0.01, self._tick)
        self.worker = self.run_worker(asyncio.sleep(3600))
        # Everything this screen owns is running before it dies.
        for _ in range(500):
            if self.ticker.ticks and self.ticks:
                break
            await asyncio.sleep(0.01)
        if self.death == "mount":
            raise RuntimeError(_CANARY)
        self.mounted = True

    def on_unmount(self) -> None:
        if self.unmount_raises == "screen":
            raise RuntimeError(_CANARY)

    def _tick(self) -> None:
        self.ticks += 1


class _KeepAliveApp(App):
    """Hands every pump error to ``retire_dead_pump`` the way
    ``TldwCli._handle_exception`` does, taking Textual's exit on ``None``."""

    def __init__(self) -> None:
        super().__init__()
        self.kinds: list[str | None] = []

    def _handle_exception(self, error: Exception) -> None:
        from textual.message_pump import active_message_pump

        from tldw_chatbook.app_keep_alive import retire_dead_pump
        from tldw_chatbook.app_lifecycle import _exception_frames

        pump = active_message_pump.get(None)
        kind = None
        if pump is not None and pump is not self:
            kind = retire_dead_pump(self, pump, _exception_frames(error))
        self.kinds.append(kind)
        if kind is None:
            super()._handle_exception(error)


async def _kill_and_retire(app, pilot, *, death: str, installed: bool, **screen_kw):
    content = _Plain()
    await app.push_screen(content)
    await pilot.pause()
    dying = _DyingScreen(death, **screen_kw)
    if installed:
        app.install_screen(dying, name="zq-dying")
    app.push_screen(dying)
    if death == "dispatch":
        assert await _poll(lambda: dying.mounted, timeout=5)
        dying.call_later(_boom)
    assert await _poll(lambda: bool(app.kinds), timeout=5), "the screen never died"
    assert app.kinds == ["screen"], app.kinds
    return content, dying


def _torn_down(app, node) -> bool:
    return node not in app._registry and node._parent is None and not node.is_running


def _cancelled(app, worker) -> bool:
    return worker.state == WorkerState.CANCELLED and worker not in app.workers


@pytest.mark.asyncio
@pytest.mark.parametrize("installed", [True, False], ids=["installed", "pushed"])
@pytest.mark.parametrize("death", ["mount", "dispatch"])
async def test_a_retired_screen_is_torn_down_however_its_loop_ended(death, installed):
    """PR #2945 review (round 3). A screen whose mount raises (here in
    ``on_mount``) ends in ``_pre_process``, and Textual returns from
    ``_process_messages`` without the ``_message_loop_exit`` that tears a
    dispatch-killed screen down. (An ordinary exception inside ``compose()``
    is not this path: Textual's ``Widget._compose`` catches it and the screen
    keeps running.) Once ``retire_dead_pump`` popped and forgot it, it stayed
    attached and in ``app._registry`` with its child's interval still ticking
    (30 ticks in 0.3 s), and ``remove()`` could not reach it: its ``Prune``
    goes to a queue no loop reads. Both deaths must end in the same state:
    the screen and its child unregistered and detached, no timer of theirs
    ticking, their workers cancelled, the screen beneath in charge."""
    app = _KeepAliveApp()
    async with app.run_test() as pilot:
        content, dying = await _kill_and_retire(
            app, pilot, death=death, installed=installed
        )
        ticker = dying.ticker
        assert ticker is not None
        assert ticker.ticks and dying.ticks, "nothing ticked before the death"
        assert await _poll(
            lambda: _torn_down(app, dying), timeout=5
        ), f"registered={dying in app._registry} parent={dying._parent!r}"
        assert await _poll(lambda: _torn_down(app, ticker), timeout=5), (
            f"registered={ticker in app._registry} parent={ticker._parent!r} "
            f"running={ticker.is_running}"
        )
        ticks = (ticker.ticks, dying.ticks)
        await asyncio.sleep(0.3)
        assert (ticker.ticks, dying.ticks) == ticks, "a dead screen's timer ticked"
        for worker in (dying.worker, ticker.worker):
            assert await _poll(
                lambda w=worker: _cancelled(app, w), timeout=5
            ), f"{type(worker.node).__name__}'s worker is {worker.state.name}"
        assert app.screen is content and content.is_running
        assert dying not in app.screen_stack
        assert not app.is_screen_installed(dying)
        assert app.kinds == ["screen"], app.kinds


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raiser", "site"),
    [("screen", "_DyingScreen.on_unmount:"), ("child", "_Ticker.on_unmount:")],
    ids=["screen", "child"],
)
async def test_a_dead_screen_whose_unmount_raises_is_still_dropped(raiser, site):
    """The mount-death teardown dispatches Unmount, as Textual's own loop exit
    does. A handler that raises there stops ``_message_loop_exit`` before its
    last steps, so the screen is dropped from the DOM and the registry anyway,
    and the failure is logged by class and site -- never its message.

    PR #2945 review (round 4): its workers must still be cancelled. Textual
    dispatches subclass-first, so a node's own ``on_unmount`` runs before
    ``Widget._on_unmount`` (which cancels that node's workers) and, by
    raising, stops the dispatch first. When the screen's raised, the screen's
    worker kept running after the drop (and could later exit the app); when
    the child's raised, its task failed, the screen's ``gather`` of its
    children raised before the screen's Unmount, and both workers kept
    running."""
    from loguru import logger

    lines: list[str] = []
    sink_id = logger.add(
        lambda message: lines.append(message.record["message"]),
        level="WARNING",
        format="{message}",
        diagnose=False,
    )
    try:
        app = _KeepAliveApp()
        async with app.run_test() as pilot:
            _content, dying = await _kill_and_retire(
                app, pilot, death="mount", installed=False, unmount_raises=raiser
            )
            ticker = dying.ticker
            assert ticker is not None
            assert await _poll(lambda: _torn_down(app, dying), timeout=5)
            if raiser == "screen":
                # The child unmounted cleanly before the screen's own raised.
                assert await _poll(lambda: _torn_down(app, ticker), timeout=5)
            assert await _poll(lambda: not ticker.is_running, timeout=5)
            for worker in (dying.worker, ticker.worker):
                assert await _poll(
                    lambda w=worker: _cancelled(app, w), timeout=5
                ), f"{type(worker.node).__name__}'s worker is {worker.state.name}"
            assert app.kinds == ["screen"], app.kinds
    finally:
        logger.remove(sink_id)
    teardown = [line for line in lines if line.startswith("Dead screen teardown")]
    assert len(teardown) == 1, lines
    assert "RuntimeError at " in teardown[0], teardown
    assert site in teardown[0], teardown
    assert _CANARY not in teardown[0]
