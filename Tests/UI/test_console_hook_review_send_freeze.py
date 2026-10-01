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
a Send outside a worker hands its review-then-dispatch continuation to a
worker, so no pump -- the app's least of all -- waits for the user's answer.

Input after the review opens is delivered the way the terminal driver does
(``_key``), not through Pilot, whose idle wait never returns while a pump is
blocked: pre-fix, these tests fail on a bounded poll instead of hanging.

Every test releases the review in a ``finally`` so a red run still tears down
instead of hanging the suite, and ``_open_review_from`` releases a review it
already pushed when its own checks fail, before that ``finally`` exists.
"""

from __future__ import annotations

import asyncio
import sys

import pytest
from textual import events
from textual.widgets import Button

from Tests.Agents.test_hook_permissions import hook_file as _hook_file
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
from tldw_chatbook.UI.Workbench.workbench_widgets import WorkbenchActionRequested
from tldw_chatbook.Widgets.Console.console_hooks_review_modal import (
    ConsoleHooksReviewModal,
)

pytestmark = pytest.mark.bootstrap_profile

hook_file = _hook_file

DRAFT = "send me after review"
CANCELLED = "Send cancelled; draft kept."
ROUTES = ["send-button", "workbench-send", "enter"]


def _key(app, key: str, char: str | None = None) -> None:
    """Deliver a key the way the terminal driver does, through the app pump,
    without Pilot's idle wait (which itself never returns while a pump is
    blocked)."""
    event = events.Key(key, char)
    event.set_sender(app)
    app._driver.send_message(event)


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
    button_id = {"allow-all": "console-hooks-allow-all", "not-now": "console-hooks-cancel"}
    modal.query_one(f"#{button_id[answer]}", Button).focus()
    _key(host, "enter", "\r")


async def _release(host, modal, requester) -> None:
    """A red run must still tear down instead of hanging the suite: dismiss a
    review nothing answered and, if the pump that pushed it is blocked (the
    pre-fix freeze), run the result callback that pump is holding."""
    if host._exit:
        return
    if host.screen is modal:
        modal.dismiss(None)
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
