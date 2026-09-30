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

W003 reports these roots only through an unrelated name collision
(``self._review`` resolves by name to ``BuddyManagementModal._review``); its
model has no "callback-resolved future" shape. These tests are the real
evidence, and they are strict xfails: the follow-up that fixes the freeze
makes them XPASS, which fails until the marker comes off.

Every test releases the stranded result callback in a ``finally`` so a red
run still tears down instead of hanging the suite.
"""

from __future__ import annotations

import asyncio

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

_FREEZE = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "Known freeze: request_hook_review awaits a push_screen callback that "
        "Textual queues on the awaiting pump (follow-up: Console Send hook "
        "review deadlocks its dispatching pump)."
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
    assert await _until(lambda: isinstance(host.screen, ConsoleHooksReviewModal), 10), (
        f"{route}: the hook review never opened"
    )
    modal = host.screen
    return console, modal, modal._result_callbacks[-1].requester


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
        assert requester is console
        try:
            _key(host, "escape")
            assert await _until(lambda: host.screen is console, 5), (
                f"{route}: Escape did not dismiss the review"
            )
            assert await _until(lambda: not console._hooks._busy, 3), (
                f"{route}: the cancelled Send never settled -- its review "
                "callback is stranded on the ChatScreen pump that awaits it"
            )
            assert await _pump_runs(console), (
                f"{route}: the Console message pump is dead after the review"
            )
        finally:
            await _release(host, modal, requester)


@_FREEZE
async def test_enter_send_review_still_receives_keys(hook_file):
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(120, 40)) as pilot:
        console, modal, requester = await _open_review_from("enter", host, pilot)
        assert requester is host
        try:
            assert await _pump_runs(host), (
                "Enter: the app pump -- the one that reads every key, Ctrl+Q "
                "included -- is blocked awaiting the review it just opened"
            )
            _key(host, "escape")
            assert await _until(lambda: host.screen is console, 3), (
                "Enter: Escape never reached the hook review"
            )
        finally:
            await _release(host, modal, requester)
