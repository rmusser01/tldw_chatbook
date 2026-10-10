"""A Send that needs hook review never freezes the pump that dispatched it.

TASK-33621.28. Found while triaging three W003 roots after TASK-33621.13 was
rebased onto dev: ``ChatScreen.on_button_pressed``,
``on_console_workbench_action_requested`` and
``on_key->_send_console_message_from_visible_action``. All three reach PR
#2922's ``ConsoleHooksController.dispatch``, which, when a hook still needed
consent, awaited ``request_hook_review``. That helper pushed the review with
``callback=`` and awaited a future the callback resolved -- but Textual runs a
result callback through ``requester.call_next``, and the requester was the
pump awaiting the future, so the callback could never run:

* Send button / Workbench "send": the requester was the ``ChatScreen``.
  Escape dismissed the modal, but the Send never settled and the Console's
  message pump never ran again.
* Enter: ``on_key`` schedules the send with ``app.call_later``, so the
  requester was the APP, whose pump reads every key: Escape never reached the
  modal and Ctrl+Q was ignored -- a GAP4-01-class total freeze.

The fix has two halves, and these tests need both. The review's answer is a
future the modal settles in its own ``dismiss`` (no pump has to flush it), and
a Send made anywhere but a worker's own task hands its review-then-dispatch
continuation to a worker, so no pump -- the app's least of all -- waits for
the user's answer. "A worker's own task" is decided by task identity: a
Console reached by tab navigation has a pump that inherited the navigation
worker's contextvar (``NavigatedConsoleHarness`` below).

Input after the review opens is delivered the way the terminal driver does
(``_key``), not through Pilot, whose idle wait never returns while a pump is
blocked: pre-fix, these tests fail on a bounded poll instead of hanging.

Every test releases the review in a ``finally`` so a red run still tears down
instead of hanging the suite, and ``_open_review_from`` releases a review it
already pushed when its own checks fail, before that ``finally`` exists.

These tests reach into Textual internals on purpose, pinned to Textual 8.x:
``app._driver.send_message`` (driver-style input), and, only to tear down a
red run, ``modal._result_callbacks[-1].requester`` and its
``_flush_next_callbacks()``.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys

import pytest
from textual import events
from textual.widgets import Button
from textual.worker import NoActiveWorker, get_current_worker

from Tests.Agents.test_hook_permissions import hook_file as _hook_file
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_console_workbench_contract import (
    ConsoleHarness,
    _configure_native_ready_console,
)
from Tests.UI.test_destination_shells import _build_test_app
from tldw_chatbook.UI.Console_Modules.dictation import _VOICE_ACK_NOT_SENT
from tldw_chatbook.UI.Console_Modules.prompt_queue import (
    ConsolePromptDispatchResult,
    ConsolePromptDispatchStatus,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Workbench.workbench_widgets import WorkbenchActionRequested
from tldw_chatbook.Widgets.Console.console_hooks_review_modal import (
    ConsoleHooksReviewModal,
)

pytestmark = pytest.mark.bootstrap_profile

hook_file = _hook_file

DRAFT = "send me after review"
CANCELLED = "Send cancelled; draft kept."
ROUTES = ["send-button", "workbench-send", "enter"]


class NavigatedConsoleHarness(ConsolidatedCSSApp):
    """Push the Console from a worker, as clicking its tab does.

    ``TldwCli._dispatch_screen_navigation`` runs navigation in the app's
    ``screen-navigation`` worker, and Textual starts the pushed screen's pump
    with ``create_task``, which copies that worker's contextvars -- so every
    handler on the Console's pump still finds it through
    ``get_current_worker()``. ``ConsoleHarness`` pushes from ``on_mount`` on
    the app pump and cannot show this. (Not a ``ConsoleHarness`` subclass:
    Textual runs every class's ``on_mount`` along the MRO, so its push would
    run as well.)
    """

    def __init__(self, app_instance):
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        async def navigate() -> None:
            await self.push_screen(ChatScreen(self.app_instance))

        self.run_worker(navigate(), group="screen-navigation")

    async def _shutdown(self) -> None:
        from tldw_chatbook.UI.Console_Modules.view_workers import (
            capture_console_view_workers,
            drain_console_view_workers,
        )

        self._exit = True
        drain_error = None
        try:
            # The actual host owns current and detached nodes in these groups.
            captured = capture_console_view_workers(self)
            view = self.screen
            view._console_chat_tearing_down = True
            await drain_console_view_workers(captured)
        except BaseException as error:
            drain_error = error
        try:
            await super()._shutdown()
        except BaseException as error:
            if drain_error is not None:
                error.add_note(
                    "Captured Console view drain also failed before host shutdown"
                )
            raise
        if drain_error is not None:
            raise drain_error


def _key(app, key: str, char: str | None = None) -> None:
    """Deliver a key the way the terminal driver does, through the app pump,
    without Pilot's idle wait (which itself never returns while a pump is
    blocked)."""
    event = events.Key(key, char)
    event.set_sender(app)
    app._driver.send_message(event)


@contextlib.contextmanager
def _task_factory(factory: str):
    """``eager`` is how the real app runs: Textual's ``App.run_async``
    installs ``asyncio.eager_task_factory``, and ``run_test`` does not."""
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    if factory == "eager":
        loop.set_task_factory(asyncio.eager_task_factory)
    try:
        yield
    finally:
        loop.set_task_factory(previous)


async def _until(predicate, seconds: float) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


async def _pump_runs(pump, seconds: float = 2.0) -> bool:
    ran = asyncio.Event()
    pump.call_later(ran.set)
    try:
        await asyncio.wait_for(ran.wait(), seconds)
    except TimeoutError:
        return False
    return True


def _record_dispatch(console) -> list[str]:
    calls: list[str] = []

    async def dispatch(draft, **_kwargs):
        calls.append(draft)
        return ConsolePromptDispatchResult(
            ConsolePromptDispatchStatus.SENT,
            console._console_visible_send_session_id(),
            "",
        )

    console._prompt_queue.dispatch = dispatch
    return calls


def _record_notices(app) -> list[str]:
    notices: list[str] = []
    app.notify = lambda message, *_args, **_kwargs: notices.append(str(message))
    return notices


async def _open_review_from(route: str, host, pilot):
    console = host.screen
    composer = console._console_composer_or_none()
    composer.load_draft(DRAFT)
    await pilot.pause()
    if route == "send-button":
        console.query_one("#console-send-message", Button).press()
    elif route == "workbench-send":
        composer.post_message(WorkbenchActionRequested("send"))
    else:
        composer.focus()
        await pilot.pause()
        _key(host, "enter", "\r")
    try:
        # Mounted, not merely pushed: answering it before its on_mount ran
        # raised NoMatches there and buried the real failure under it.
        assert await _until(
            lambda: (
                isinstance(host.screen, ConsoleHooksReviewModal)
                and host.screen.is_mounted
            ),
            10,
        ), f"{route}: the hook review never opened"
        await asyncio.sleep(0.1)
        modal = host.screen
        return console, composer, modal, modal._result_callbacks[-1].requester
    except BaseException:
        # The caller's `finally` has not started yet, so release a review
        # this already pushed here: its result callback is stranded on the
        # blocked pump that pushed it, and run_test's teardown awaits that
        # pump's task -- the run would hang instead of failing. BaseException,
        # not Exception: `pytest.fail` (a timeout plugin's too),
        # KeyboardInterrupt and CancelledError skipped an `except Exception`.
        for screen in list(host.screen_stack):
            if isinstance(screen, ConsoleHooksReviewModal) and screen._result_callbacks:
                await _until(lambda screen=screen: screen.is_mounted, 5)
                await _release(host, screen, screen._result_callbacks[-1].requester)
        raise


async def _assert_pumps_run(host, console, when: str) -> None:
    assert await _pump_runs(host), (
        f"{when}: the app pump -- the one that reads every key, Ctrl+Q "
        "included -- is blocked"
    )
    assert await _pump_runs(console), f"{when}: the Console message pump is blocked"


def _answer(host, modal, answer: str) -> None:
    if answer == "escape":
        _key(host, "escape")
        return
    button_id = {
        "allow-all": "console-hooks-allow-all",
        "not-now": "console-hooks-cancel",
    }
    modal.query_one(f"#{button_id[answer]}", Button).focus()
    _key(host, "enter", "\r")


async def _release(host, modal, requester) -> None:
    """A red run must still tear down instead of hanging the suite: dismiss a
    review nothing answered and, if the pump that pushed it is blocked (the
    pre-fix freeze), run the result callback that pump is holding."""
    if host._exit:
        return
    if host.screen is modal:
        # Cancel the resident request, not only its disposable presentation.
        await modal.request_safe_cancel(source="test-cleanup")
    await asyncio.sleep(0.2)
    if not await _pump_runs(requester, 0.5):
        await requester._flush_next_callbacks()
        await asyncio.sleep(0.2)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("answer", ["escape", "not-now"])
async def test_a_declined_send_review_settles_and_keeps_the_draft(
    route, answer, hook_file
):
    """AC#1/#2: the review answers to the keyboard on every route, and a
    declined review refuses the Send with visible copy and keeps the draft."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    notices = _record_notices(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen
        calls = _record_dispatch(console)
        console, composer, modal, requester = await _open_review_from(
            route, host, pilot
        )
        try:
            await _assert_pumps_run(host, console, f"{route}, review open")
            _answer(host, modal, answer)
            assert await _until(lambda: host.screen is console, 5), (
                f"{route}: {answer} never reached the hook review"
            )
            assert await _until(lambda: not console._hooks._busy, 5), (
                f"{route}: the declined Send never settled"
            )
            assert CANCELLED in notices, f"{route}: no refusal copy: {notices!r}"
            assert calls == []
            assert composer.draft_text() == DRAFT
            await _assert_pumps_run(host, console, f"{route}, after the review")
        finally:
            await _release(host, modal, requester)


#: How the open check fails once the review is up: its assertion, or a
#: BaseException -- what ``pytest.fail`` raises (a plugin's timeout among its
#: callers), like ``KeyboardInterrupt`` and ``CancelledError``.
_OPEN_FAILURES = {
    "assertion": (AssertionError, "the hook review never opened"),
    "base-exception": (pytest.fail.Exception, "Timeout"),
}


@pytest.mark.parametrize("failure", sorted(_OPEN_FAILURES))
async def test_a_failed_open_check_still_releases_the_review_it_opened(
    failure, hook_file, monkeypatch
):
    """PR #2944 review: each test opened the review BEFORE its ``try``, so a
    failure inside ``_open_review_from`` once the review was pushed -- its
    mounted check timing out -- skipped the ``finally`` and left the result
    callback stranded on the pump that pushed it. Textual's teardown awaits
    that blocked pump's task (``MessagePump._close_messages``), so the run
    would hang to the suite timeout instead of failing. The helper must
    release what it opened before its error propagates, whatever the error:
    catching only ``Exception`` let a BaseException skip the release."""
    real_until = _until
    expected, message = _OPEN_FAILURES[failure]
    calls = 0

    async def wait_then_fail(predicate, seconds):
        # Wait for real, then fail the open check: the review IS open,
        # exactly what a mount that finishes past the deadline leaves
        # behind. Only the open check fails; the release's own waits run.
        nonlocal calls
        calls += 1
        ready = await real_until(predicate, seconds)
        if calls > 1:
            return ready
        if failure == "base-exception":
            pytest.fail("Timeout while the hook review mounted")
        return False

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen
        module = sys.modules[__name__]
        monkeypatch.setattr(module, "_until", wait_then_fail)
        with pytest.raises(expected, match=message):
            await _open_review_from("send-button", host, pilot)
        monkeypatch.setattr(module, "_until", real_until)
        stranded = [
            screen
            for screen in host.screen_stack
            if isinstance(screen, ConsoleHooksReviewModal)
        ]
        try:
            assert not stranded, "the failed open left its review on the stack"
            assert await _until(lambda: not console._hooks._busy, 5), (
                "the Send behind the failed open never settled"
            )
            assert await _pump_runs(console), "the Console pump is still blocked"
        finally:
            # A red run must still tear down.
            for modal in stranded:
                if modal._result_callbacks:
                    await _release(host, modal, modal._result_callbacks[-1].requester)


@pytest.mark.parametrize("route", ROUTES)
async def test_allow_all_resumes_the_captured_send_exactly_once(route, hook_file):
    """AC#2/#3: Allow all approves, closes the review and dispatches the
    captured Send once -- #2922's 'Allow resumes once' -- from every route."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen
        calls = _record_dispatch(console)
        console, _composer, modal, requester = await _open_review_from(
            route, host, pilot
        )
        try:
            await _assert_pumps_run(host, console, f"{route}, review open")
            _answer(host, modal, "allow-all")
            assert await _until(lambda: calls == [DRAFT], 5), (
                f"{route}: the approved Send never dispatched: {calls!r}"
            )
            assert await _until(
                lambda: host.screen is console and not console._hooks._busy, 5
            ), f"{route}: the approved Send never settled"
            await _assert_pumps_run(host, console, f"{route}, after the review")
            assert calls == [DRAFT]
            assert console._console_runtime().ensure_hook_permissions().snapshot().ready
        finally:
            await _release(host, modal, requester)


@pytest.mark.parametrize("route", ["send-button", "workbench-send"])
@pytest.mark.parametrize(
    ("answer", "factory"),
    [("escape", "lazy"), ("allow-all", "lazy"), ("allow-all", "eager")],
)
async def test_a_console_reached_by_navigation_never_awaits_review_on_its_pump(
    route, answer, factory, hook_file
):
    """TASK-33621.28 review finding: on a Console reached by tab navigation,
    the Send button and Workbench send awaited the whole review on the
    Console pump (live: ``ui_dispatch status=refused duration_ms=16328``),
    because the pump inherited the navigation worker as ``active_worker``.
    No deadlock -- the modal settles its own answer -- but the Console was
    blocked for the whole review. ``eager`` runs it under the task factory
    the real app runs with (see ``_task_factory``)."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    notices = _record_notices(app)
    host = NavigatedConsoleHarness(app)
    with _task_factory(factory):
        await _run_navigated_review(host, route=route, answer=answer, notices=notices)


async def _run_navigated_review(host, *, route, answer, notices):
    async with host.run_test(size=(120, 40)) as pilot:
        assert await _until(
            lambda: isinstance(host.screen, ChatScreen) and host.screen.is_mounted,
            10,
        ), "the navigation worker never mounted the Console"
        console = host.screen
        await pilot.pause()
        inherited: list = []

        def note_worker() -> None:
            try:
                inherited.append(get_current_worker())
            except NoActiveWorker:
                inherited.append(None)

        console.call_later(note_worker)
        assert await _until(lambda: inherited, 5)
        # The negative control: this harness reproduces navigation only if
        # the Console's pump really did inherit the navigation worker.
        assert inherited[0] is not None and inherited[0].group == "screen-navigation"
        calls = _record_dispatch(console)
        console, composer, modal, requester = await _open_review_from(
            route, host, pilot
        )
        try:
            await _assert_pumps_run(host, console, f"{route}, review open")
            _answer(host, modal, answer)
            if answer == "allow-all":
                assert await _until(lambda: calls == [DRAFT], 5), (
                    f"{route}: the approved Send never dispatched: {calls!r}"
                )
            else:
                assert await _until(lambda: CANCELLED in notices, 5), (
                    f"{route}: no refusal copy: {notices!r}"
                )
                assert calls == [] and composer.draft_text() == DRAFT
            assert await _until(
                lambda: host.screen is console and not console._hooks._busy, 5
            ), f"{route}: the Send never settled"
            await _assert_pumps_run(host, console, f"{route}, after the review")
        finally:
            await _release(host, modal, requester)


async def test_a_handed_off_send_records_its_reviewed_outcome(hook_file, monkeypatch):
    """TASK-33621.28 review finding: a handed-off Send was logged
    ``ui_action not_dispatched`` even when Allow all then sent it. The UI
    scopes now close as ``awaiting_review``, and the worker records
    ``hook_review_continuation`` with the real status, all under one attempt."""
    from tldw_chatbook.Chat.console_send_diagnostics import SendDiagnostic

    stages: list[tuple[str, str, str]] = []
    record = SendDiagnostic.record

    def recording(self, phase, status="entered", **fields):
        stages.append((self.attempt_id, phase, status))
        return record(self, phase, status, **fields)

    monkeypatch.setattr(SendDiagnostic, "record", recording)
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen
        calls = _record_dispatch(console)
        console, _composer, modal, requester = await _open_review_from(
            "send-button", host, pilot
        )
        try:
            _answer(host, modal, "allow-all")
            assert await _until(
                lambda: (
                    ("hook_review_continuation", "sent")
                    in {(phase, status) for _, phase, status in stages}
                ),
                5,
            ), stages
            assert calls == [DRAFT]
        finally:
            await _release(host, modal, requester)
    settled = [(phase, status) for _, phase, status in stages if status != "entered"]
    assert settled == [
        ("ui_dispatch", "awaiting_review"),
        ("ui_action", "awaiting_review"),
        ("hook_review_continuation", "sent"),
    ]
    assert len({attempt for attempt, _, _ in stages}) == 1


async def test_the_app_pump_answers_ctrl_q_while_the_enter_review_is_open(hook_file):
    """Pre-fix, Enter's review blocked the app pump, so no binding could run
    and nothing but killing the process ended the app.

    What this proves is the pump, not the real app's quit: the harness is a
    stock Textual ``App``, whose Ctrl+Q is a PRIORITY binding and so is
    checked before any modal. ``TldwCli`` replaces it with a non-priority
    ``ctrl+q``, and Textual's ``_modal_binding_chain`` stops at the topmost
    ``ModalScreen`` -- so in the real app Ctrl+Q is ignored under EVERY modal
    (the Ctrl+K session switcher too), with or without this fix. That is a
    separate, app-wide gap, reported with TASK-33621.28 rather than changed
    here."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen
        _record_dispatch(console)
        _console, _composer, modal, requester = await _open_review_from(
            "enter", host, pilot
        )
        try:
            _key(host, "ctrl+q")
            assert await _until(lambda: bool(host._exit), 10), (
                "Ctrl+Q did not quit while the Enter Send's hook review was open"
            )
        finally:
            await _release(host, modal, requester)


@pytest.mark.parametrize(
    ("answer", "spoken", "sent"),
    [("allow-all", "Sent.", [DRAFT]), ("escape", _VOICE_ACK_NOT_SENT, [])],
)
async def test_spoken_send_acknowledges_the_reviewed_outcome(
    answer, spoken, sent, hook_file
):
    """AC#3: a spoken "Console, send." already runs in a worker, so it still
    waits for the review and speaks the Send's real outcome -- never "Sent."
    for a Send the review cancelled."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen
        calls = _record_dispatch(console)
        composer = console._console_composer_or_none()
        composer.load_draft(DRAFT)
        await pilot.pause()
        said: list[str] = []
        console._speak_status = said.append
        console._console_pending_voice_action = "send"
        session_id = console._console_visible_send_session_id()
        worker = console.run_worker(
            console._run_pending_console_voice_action(session_id),
            group="test-spoken-send",
        )
        assert await _until(
            lambda: (
                isinstance(host.screen, ConsoleHooksReviewModal)
                and host.screen.is_mounted
            ),
            10,
        ), "the spoken Send never opened the hook review"
        modal = host.screen
        requester = modal._result_callbacks[-1].requester
        await asyncio.sleep(0.1)
        try:
            assert said == [], "acknowledged before the review was answered"
            _answer(host, modal, answer)
            await asyncio.wait_for(worker.wait(), 10)
            assert said == [spoken]
            assert calls == sent
        finally:
            await _release(host, modal, requester)
