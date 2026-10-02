"""A Send that needs hook review freezes the pump that dispatched it (U3w).

Found while triaging three new W003 roots after TASK-33621.13 was rebased onto
dev: ``ChatScreen.on_button_pressed``, ``on_console_workbench_action_requested``
and ``on_key->_send_console_message_from_visible_action``. All three reach
PR #2922's ``ConsoleHooksController.dispatch``, which, when a hook still needs
consent, awaits ``request_hook_review``. That helper deliberately avoids
``push_screen_wait`` (so no ``NoActiveWorker``) by pushing with ``callback=``
and awaiting a future the callback resolves -- but Textual runs a result
callback through ``requester.call_next``, and the requester is the pump that
pushed, which is the very pump now awaiting the future. The callback is queued
on a pump that can never flush it:

* Send button / Workbench "send": the requester is the ``ChatScreen``. Escape
  dismisses the modal, but the Send never settles and the Console's message
  pump never runs again.
* Enter: ``on_key`` schedules the send with ``app.call_later``, so the
  requester is the APP. Its pump is the one that reads keys, so Escape never
  reaches the modal and Ctrl+Q is ignored -- the GAP4-01 total freeze.

W003 now models this "hand-rolled wait" and censuses all three roots at
their real push site, ``request_hook_review``. These tests are the runtime
evidence, and they are strict xfails (TASK-33621.28): the fix makes them
XPASS, which fails until the marker comes off.

Only the freeze itself may satisfy the xfail. The freeze assertions raise
:class:`FreezeObserved` and the marker accepts nothing else, so a broken
precondition -- the review never opening, the wrong requester, Escape not
dismissing it where the app pump is alive -- FAILS instead of passing as the
expected XFAIL (``raises=AssertionError`` let exactly that go green).

Every test releases the stranded result callback in a ``finally`` so a red
run still tears down instead of hanging the suite, and ``_open_review_from``
releases a review it already pushed when its own checks fail, before that
``finally`` exists.
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
from tldw_chatbook.UI.Console_Modules.prompt_queue import (
    ConsolePromptDispatchResult,
    ConsolePromptDispatchStatus,
)
from tldw_chatbook.UI.Workbench.workbench_widgets import WorkbenchActionRequested
from tldw_chatbook.Widgets.Console.console_hooks_review_modal import (
    ConsoleHooksReviewModal,
    HookReviewResult,
)

pytestmark = pytest.mark.bootstrap_profile

hook_file = _hook_file


class FreezeObserved(Exception):
    """The known hook-review freeze, observed. Deliberately NOT an
    ``AssertionError``: the xfail accepts only this, so every other failure
    in these tests is a real failure."""


def _freeze_unless(condition: bool, message: str) -> None:
    if not condition:
        raise FreezeObserved(message)


_FREEZE = pytest.mark.xfail(
    strict=True,
    raises=FreezeObserved,
    reason=(
        "TASK-33621.28: request_hook_review awaits a push_screen callback that "
        "Textual queues on the awaiting pump, so Send with an unconsented hook "
        "freezes the app (Enter) or deadlocks the Console."
    ),
)


def _key(app, key: str, char: str | None = None) -> None:
    """Deliver a key the way the terminal driver does, without Pilot's idle wait
    (which itself never returns while a pump is blocked)."""
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


async def _open_review_from(route: str, host, pilot):
    console = host.screen
    composer = console._console_composer_or_none()
    composer.load_draft("send me after review")
    await pilot.pause()
    calls: list[str] = []

    async def dispatch(draft, **_kwargs):
        calls.append(draft)
        return ConsolePromptDispatchResult(
            ConsolePromptDispatchStatus.SENT,
            console._console_visible_send_session_id(),
            "",
        )

    console._prompt_queue.dispatch = dispatch
    if route == "send-button":
        console.query_one("#console-send-message", Button).press()
    elif route == "workbench-send":
        composer.post_message(WorkbenchActionRequested("send"))
    else:
        composer.focus()
        await pilot.pause()
        _key(host, "enter", "\r")
    try:
        # Mounted, not merely pushed: dismissing it before its on_mount ran
        # raised NoMatches there and buried the real failure under it.
        assert await _until(
            lambda: (
                isinstance(host.screen, ConsoleHooksReviewModal)
                and host.screen.is_mounted
            ),
            10,
        ), f"{route}: the hook review never opened"
        modal = host.screen
        return console, modal, modal._result_callbacks[-1].requester
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


async def _release(host, modal, requester) -> None:
    """Resolve the review by hand: dismiss it if keys never reached it, then
    run the result callback its requester's blocked pump is holding."""
    if host.screen is modal:
        modal.dismiss(HookReviewResult("cancel"))
    await asyncio.sleep(0.2)
    await requester._flush_next_callbacks()
    await asyncio.sleep(0.2)


@_FREEZE
@pytest.mark.parametrize("route", ["send-button", "workbench-send"])
async def test_console_pump_runs_again_after_the_send_review_is_dismissed(
    route, hook_file
):
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(120, 40)) as pilot:
        console, modal, requester = await _open_review_from(route, host, pilot)
        try:
            assert requester is console, f"{route}: requester is {requester!r}"
            _key(host, "escape")
            # The app pump is alive on these routes, so Escape reaching the
            # modal is a precondition, not the freeze.
            assert await _until(lambda: host.screen is console, 5), (
                f"{route}: Escape did not dismiss the review"
            )
            _freeze_unless(
                await _until(lambda: not console._hooks._busy, 3),
                f"{route}: the cancelled Send never settled -- its review "
                "callback is stranded on the ChatScreen pump that awaits it",
            )
            _freeze_unless(
                await _pump_runs(console),
                f"{route}: the Console message pump is dead after the review",
            )
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


@_FREEZE
async def test_enter_send_review_still_receives_keys(hook_file):
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(120, 40)) as pilot:
        console, modal, requester = await _open_review_from("enter", host, pilot)
        try:
            assert requester is host, f"enter: requester is {requester!r}"
            _freeze_unless(
                await _pump_runs(host),
                "Enter: the app pump -- the one that reads every key, Ctrl+Q "
                "included -- is blocked awaiting the review it just opened",
            )
            _key(host, "escape")
            _freeze_unless(
                await _until(lambda: host.screen is console, 3),
                "Enter: Escape never reached the hook review",
            )
        finally:
            await _release(host, modal, requester)
