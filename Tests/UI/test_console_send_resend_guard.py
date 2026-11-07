"""A Console send never re-sends a captured draft (TASK-33620.15.2).

Lead ruling (TASK-33620.15): what is sent is what was in the composer at the
press that sent it; keys typed afterwards stay as the next draft; nothing
typed is ever lost or silently sent. Keys flow while a send is admitted, so a
second send path can act on the same draft before the first one commits it:

* a press held behind a running send, whose draft a spoken "Console, send."
  sends meanwhile, was replayed and sent the same text again;
* an Enter whose capture a spoken send committed while the Enter's
  acknowledgement was painting went on to send it again;
* Enter, a tab round-trip and Enter during admission sent the draft twice:
  the reload gave the draft a new generation, so the second press looked new
  and the first send's commit failed closed.

A sent draft that cannot be committed out of the composer must say so (a
non-append edit during admission), or come off the draft it was sent from
(a tab left during admission). A press refused because of a send in another
tab names that tab, and a bare image-only repeat press is deduplicated like
a text repeat.

The mounted tests use the app's eager task factory, store and runtime. The
busy-tab case invokes the original hook dispatch in workers and changes tabs
with a driver-delivered key; the other send journeys use their named inputs.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from concurrent.futures.thread import _WorkItem
from functools import partial
from types import CodeType

import pytest

from Tests.Agents.test_hook_permissions import hook_file as _hook_file

from tldw_chatbook.Chat.console_display_state import ConsoleBlockedReason
from Tests.UI.test_console_send_acknowledgement import (
    DRAFT,
    HeldAdmission,
    build,
    eager_tasks,
    press,
    ready_console,
    until,
)
from Tests.UI.test_console_send_admission_off_pump import (
    B_DRAFT,
    ENTRY_SECONDS,
    HOLD_SECONDS,
    HeldMcpRead,
    _allow_second_turns,
    _first_turn_done,
    _hold_first_send_tail,
    _record_notices,
    _second_tab,
    _sent_or_queued,
    _settled,
    _users,
)

hook_file = _hook_file
pytestmark = pytest.mark.bootstrap_profile


async def _idle(host, console, pilot, session_id: str) -> None:
    """Wait until no visible send, held press or turn is still running."""
    await _settled(console, pilot)
    controller = console._ensure_console_chat_controller()

    def quiet() -> bool:
        activity = controller.activity_for(session_id)
        return not (activity.occupies_slot or activity.accepted_live_turn)

    await until(quiet)
    await _settled(console, pilot)


@pytest.mark.asyncio
async def test_a_draft_sent_after_a_non_append_edit_says_it_was_sent(record_property):
    """AC#2: a sent draft the composer still shows is announced as sent.

    Enter, then Home and "x" while the send is admitted. The captured draft
    is sent (lead ruling), but the composer now reads "x" + the draft, so the
    commit fails closed and leaves it, as it must: the "x" is the user's.
    Nothing said that the draft in it had already gone, which invites an
    accidental resend.
    """
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            session_id = console._console_chat_store.active_session_id
            notices = _record_notices(host)
            async with HeldMcpRead(host.app_instance.unified_mcp_service) as hold:
                gateway.validation_release.set()
                try:
                    press(host, "enter", "\r")
                    await until(hold.entered.is_set, timeout=ENTRY_SECONDS)
                    assert not hold.timed_out and not hold.future.done()
                    assert not hold.body_returned
                    press(host, "home")
                    press(host, "x", "x")
                    await until(
                        lambda: composer.draft_text() == "x" + DRAFT, timeout=10
                    )
                finally:
                    hold.release.set()
                await until(lambda: gateway.stream_calls == 1)
                await _idle(host, console, pilot, session_id)
                assert _sent_or_queued(console, session_id) == [DRAFT]
                assert composer.draft_text() == "x" + DRAFT
                assert [n for n in notices if "was sent" in n], notices

    record_property(
        "original_native_permission_return",
        {
            "body": hold.body_code.co_qualname,
            "callback": hold.callback_code.co_qualname,
            "worker": hold.invoke_code.co_qualname,
            "thread": hold.thread,
            "entered": hold.entered.is_set(),
            "released": hold.release.is_set(),
            "timed_out": hold.timed_out,
            "body_returned": hold.body_returned,
            "callback_returned": hold.callback_returned,
            "invoke_returned": hold.invoke_returned,
            "physical_future_done": hold.future.done(),
            "read_retired": hold.read.retired.done(),
            "observer_retired": hold.observer_retired,
            "host_context_exited": True,
        },
    )


@pytest.mark.asyncio
async def test_a_press_held_on_a_non_append_edit_of_a_sent_draft_is_not_sent():
    """AC#2: a press captured before the commit retired its draft sends nothing.

    Enter, a tab round trip, then Home, "x" and Enter while the first send is
    admitted. The round trip reloaded the draft with a new generation, so the
    second press is not taken for the running capture: it is held, with
    "x" + the sent draft. The first send's commit cannot take its text out of
    that edit, so it retires every capture of the draft
    (``retire_captured_drafts``), and the held press is dropped instead of
    sending the text a second time inside the "x" (TASK-38718 review).
    """
    host, gateway, _timeline = build()
    _allow_second_turns(host)
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            session_a, session_b = await _second_tab(console, pilot, ready=True)
            notices = _record_notices(host)
            tail = _hold_first_send_tail(console)
            async with HeldMcpRead(host.app_instance.unified_mcp_service) as hold:
                gateway.validation_release.set()
                try:
                    press(host, "enter", "\r")
                    await until(hold.entered.is_set, timeout=ENTRY_SECONDS)
                    press(host, "alt+2")
                    await until(
                        lambda: console._console_visible_draft_session_id == session_b
                    )
                    press(host, "alt+1")
                    await until(
                        lambda: console._console_visible_draft_session_id == session_a
                    )
                    await until(lambda: composer.draft_text() == DRAFT)
                    press(host, "home")
                    press(host, "x", "x")
                    await until(
                        lambda: composer.draft_text() == "x" + DRAFT, timeout=10
                    )
                    press(host, "enter", "\r")
                    await until(lambda: _held(console), timeout=10)
                finally:
                    hold.release.set()
                try:
                    await until(lambda: _first_turn_done(host, console))
                finally:
                    tail.set()
                await _idle(host, console, pilot, session_a)
                await pilot.pause(0.5)
                assert _sent_or_queued(console, session_a) == [DRAFT]
                assert gateway.stream_calls == 1
                assert composer.draft_text() == "x" + DRAFT
                assert [n for n in notices if "was sent" in n], notices


@pytest.mark.asyncio
async def test_a_draft_sent_from_a_tab_left_during_admission_does_not_come_back():
    """AC#2: the sent draft comes off the chat it was sent from.

    Enter, "x", then Alt+2 while the send is admitted: the send still goes
    out in tab A, but A's draft had been saved for the switch, and the
    commit, made while A was hidden, touched only the composer. (The runtime
    clears a saved draft only when it IS the sent text.) Back in A, the
    composer showed the sent text again, with the "x" after it.
    """
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            session_a, session_b = await _second_tab(console, pilot, ready=True)
            store = console._ensure_console_chat_store()
            async with HeldMcpRead(host.app_instance.unified_mcp_service) as hold:
                gateway.validation_release.set()
                try:
                    press(host, "enter", "\r")
                    await until(hold.entered.is_set, timeout=ENTRY_SECONDS)
                    press(host, "x", "x")
                    await until(
                        lambda: composer.draft_text() == DRAFT + "x", timeout=10
                    )
                    press(host, "alt+2")
                    await until(
                        lambda: console._console_visible_draft_session_id == session_b
                    )
                finally:
                    hold.release.set()
                await until(lambda: gateway.stream_calls == 1)
                await _idle(host, console, pilot, session_a)
                assert _sent_or_queued(console, session_a) == [DRAFT]
                assert store.session_draft(session_a) == "x"
                press(host, "alt+1")
                await until(
                    lambda: console._console_visible_draft_session_id == session_a
                )
                await pilot.pause(0.3)
                assert composer.draft_text() == "x"
                assert store.session_draft(session_b) == B_DRAFT


def _flight(console):
    return getattr(console, "_console_send_flight", None)


def _held(console) -> bool:
    flight = _flight(console)
    return flight is not None and flight.deferred is not None


def _count_scheduled_sends(monkeypatch) -> list[str]:
    """Record each visible send the flight schedules (and so acknowledges)."""
    from tldw_chatbook.UI.Console_Modules import send_acknowledgement

    scheduled: list[str] = []
    real = send_acknowledgement.schedule_acknowledged_send

    def counting(screen, pending_send):
        scheduled.append(pending_send.stash.text if pending_send.stash else "")
        return real(screen, pending_send)

    monkeypatch.setattr(send_acknowledgement, "schedule_acknowledged_send", counting)
    return scheduled


def _speak_send(console, session_id: str):
    """Run "Console, send." the way the dictation tail does: in a worker."""
    console._console_pending_voice_action = "send"
    return console.run_worker(
        console._run_pending_console_voice_action(session_id),
        group="console-voice-send-test",
        exit_on_error=False,
    )


def _turns_done(console, session_id: str, count: int) -> bool:
    """Whether ``count`` turns have their replies and nothing else runs."""
    controller = console._ensure_console_chat_controller()
    activity = controller.activity_for(session_id)
    replies = [
        message
        for message in console._ensure_console_chat_store().messages_for_session(
            session_id
        )
        if message.role.value == "assistant" and message.content
    ]
    busy = activity.occupies_slot or activity.accepted_live_turn
    return len(replies) >= count and not busy


@pytest.mark.asyncio
async def test_a_held_enter_never_resends_a_draft_a_spoken_send_took(monkeypatch):
    """AC#1: the held press is dropped once another send took its draft.

    The first send's draft has left the composer and its send is still
    running, so "y" plus Enter is held. A spoken "Console, send." then sends
    the composer's "y" and is answered. The held press was replayed when the
    first send settled and sent "y" a second time. It is not even scheduled
    now: a scheduled press paints "Sending…" for a send that never happens.
    """
    host, gateway, _timeline = build()
    _allow_second_turns(host)
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            session_id = console._console_chat_store.active_session_id
            scheduled = _count_scheduled_sends(monkeypatch)
            tail = _hold_first_send_tail(console)
            gateway.validation_release.set()
            try:
                press(host, "enter", "\r")
                await until(lambda: composer.draft_text() == "", timeout=ENTRY_SECONDS)
                press(host, "y", "y")
                press(host, "enter", "\r")
                await until(lambda: _held(console), timeout=10)
                await until(lambda: _first_turn_done(host, console))
                _speak_send(console, session_id)
                await until(lambda: _turns_done(console, session_id, 2))
                assert composer.draft_text() == ""
            finally:
                tail.set()
            await _idle(host, console, pilot, session_id)
            await pilot.pause(0.5)
            assert _sent_or_queued(console, session_id) == [DRAFT, "y"]
            assert gateway.stream_calls == 2
            assert composer.draft_text() == ""
            assert len(scheduled) == 1, scheduled


@pytest.mark.asyncio
async def test_an_enter_never_resends_a_draft_a_spoken_send_committed_meanwhile():
    """AC#1: an Enter's capture is spent once another send commits it.

    A spoken send of "y" is admitted when Enter captures the same "y". The
    spoken send commits it before Enter's captured send starts; that delayed
    send once sent the same "y" again.
    """
    from tldw_chatbook.UI.Console_Modules import send_acknowledgement

    host, gateway, _timeline = build()
    _allow_second_turns(host)
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            composer.load_draft("y")
            await pilot.pause()
            session_id = console._console_chat_store.active_session_id
            async with HeldMcpRead(host.app_instance.unified_mcp_service) as hold:
                gateway.validation_release.set()
                reached, release = asyncio.Event(), asyncio.Event()
                real_start = send_acknowledgement._start_send

                def held_start(screen, send):
                    async def start_when_released():
                        reached.set()
                        await release.wait()
                        real_start(screen, send)

                    asyncio.get_running_loop().create_task(start_when_released())

                send_acknowledgement._start_send = held_start
                try:
                    spoken = _speak_send(console, session_id)
                    await until(hold.entered.is_set, timeout=ENTRY_SECONDS)
                    press(host, "enter", "\r")
                    await until(lambda: console._console_pending_send is not None)
                    await until(reached.is_set, timeout=10)
                    pending = console._console_pending_send
                    assert pending.session_id == session_id
                    assert pending.stash.text == "y"
                    assert not release.is_set()
                    hold.release.set()
                    await until(lambda: spoken.is_finished, timeout=ENTRY_SECONDS)
                    await until(lambda: _turns_done(console, session_id, 1))
                    await until(lambda: _sent_or_queued(console, session_id) == ["y"])
                    await until(lambda: composer.draft_text() == "")
                    assert console._console_pending_send is pending
                finally:
                    hold.release.set()
                    release.set()
                    send_acknowledgement._start_send = real_start
                await until(lambda: console._console_pending_send is None)
                await _idle(host, console, pilot, session_id)
                assert _sent_or_queued(console, session_id) == ["y"]
                assert gateway.stream_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("typed_in_b", [False, True], ids=["round_trip", "typed_in_b"])
async def test_enter_after_a_tab_round_trip_during_admission_sends_the_draft_once(
    typed_in_b,
):
    """AC#1/#2: Enter, Alt+2, Alt+1, Enter during admission sends one message.

    The round-trip reloads the draft with a new generation, so the second
    Enter was held as a new press, the first send's commit failed closed,
    and the replay sent the draft a second time.

    Either way the reloaded draft is the first capture, so it leaves tab A's
    composer and the held press is not sent. A key typed in tab B stays in
    B's draft. TASK-38718: this once expected the sent text to stay in A
    with a "was sent" notice when a key was typed in B, because the
    composer's widget-wide edit serial could not tell the reload from a new
    draft. Since 077e5bfa09 the store's per-chat draft revision decides
    (``_project_console_saved_input``), and A's draft did not change.
    """
    host, gateway, _timeline = build()
    _allow_second_turns(host)
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            session_a, session_b = await _second_tab(console, pilot, ready=True)
            store = console._ensure_console_chat_store()
            notices = _record_notices(host)
            tail = _hold_first_send_tail(console)
            async with HeldMcpRead(host.app_instance.unified_mcp_service) as hold:
                gateway.validation_release.set()
                try:
                    press(host, "enter", "\r")
                    await until(hold.entered.is_set, timeout=ENTRY_SECONDS)
                    press(host, "alt+2")
                    await until(
                        lambda: console._console_visible_draft_session_id == session_b
                    )
                    if typed_in_b:
                        await until(lambda: composer.draft_text() == B_DRAFT)
                        press(host, "q", "q")
                        await until(lambda: composer.draft_text() == B_DRAFT + "q")
                    press(host, "alt+1")
                    await until(
                        lambda: console._console_visible_draft_session_id == session_a
                    )
                    await until(lambda: composer.draft_text() == DRAFT)
                    press(host, "enter", "\r")
                    await until(lambda: _held(console), timeout=10)
                finally:
                    hold.release.set()
                try:
                    await until(lambda: _first_turn_done(host, console))
                finally:
                    tail.set()
                await _idle(host, console, pilot, session_a)
                await pilot.pause(0.5)
                assert _sent_or_queued(console, session_a) == [DRAFT]
                assert gateway.stream_calls == 1
                assert composer.draft_text() == ""
                assert store.session_draft(session_a) == ""
                assert not [n for n in notices if "was sent" in n], notices
                assert store.session_draft(session_b) == B_DRAFT + ("q" if typed_in_b else "")


@pytest.mark.asyncio
async def test_a_press_refused_for_a_send_held_in_another_tab_names_that_tab():
    """AC#3: the refusal says which tab's message is still being sent.

    A press held in tab A (its first send still running), then Enter in tab
    B: B's press is refused with "your previous message is still being sent"
    although B sent nothing before. The notice now names tab A.
    """
    host, gateway, _timeline = build()
    _allow_second_turns(host)
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            session_a, session_b = await _second_tab(console, pilot, ready=True)
            store = console._ensure_console_chat_store()
            store.rename_session(session_a, "Alpha plans")
            notices = _record_notices(host)
            tail = _hold_first_send_tail(console)
            gateway.validation_release.set()
            try:
                press(host, "enter", "\r")
                await until(lambda: composer.draft_text() == "", timeout=ENTRY_SECONDS)
                press(host, "y", "y")
                press(host, "enter", "\r")
                await until(lambda: _held(console), timeout=10)
                press(host, "alt+2")
                await until(
                    lambda: console._console_visible_draft_session_id == session_b
                )
                await until(lambda: composer.draft_text() == B_DRAFT)
                press(host, "enter", "\r")
                await until(lambda: any("still being sent" in n for n in notices))
                await until(lambda: _turns_done(console, session_a, 1))
            finally:
                tail.set()
            await _idle(host, console, pilot, session_a)
            refusals = [n for n in notices if "still being sent" in n]
            assert refusals and all("Alpha plans" in n for n in refusals), refusals
            assert _sent_or_queued(console, session_b) == []
            assert (
                store.session_draft(session_b) == B_DRAFT
                or composer.draft_text() == B_DRAFT
            )


class _HeldProfileRead(HeldMcpRead):
    """Hold A's original post-hook profile read under a private observer lease."""

    def __init__(self, console, controller, runtime, store, session):
        from Tests.Performance.console_storage_unit_observer import (
            OriginalStorageUnitObserver,
        )
        from tldw_chatbook.Chat import (
            console_configuration_preparation as configuration,
        )
        from tldw_chatbook.Chat import console_preparation_reads as preparation
        from tldw_chatbook.Chat import console_turn_context as context

        OriginalStorageUnitObserver.__init__(
            self, {}, lambda: False, lambda _unit: None
        )
        self.console, self.controller, self.runtime = console, controller, runtime
        self.store, self.session, self.app = store, session, controller.app
        self.workspace = session.workspace_id
        self.preparation, self.configuration = preparation, configuration
        self.entered, self.release = threading.Event(), threading.Event()
        self.timed_out = False
        self.thread = None
        self.armed = True
        self.read = self.future = self.callback = None
        self.body_returned = self.callback_returned = self.invoke_returned = False
        self.tool_name = "held-native-profile-read"
        self.enabled_codes = set()
        self.observer_retired = self.foreign_callback = False
        self.profile = context.resolve_turn_tool_policy_profile_id
        assert configuration.resolve_turn_tool_policy_profile_id is self.profile
        self.body_code = self._pin(self.profile)
        owner_code = self._pin(configuration.capture_console_turn_configuration_owned)
        callbacks = [
            value
            for value in owner_code.co_consts
            if type(value) is CodeType
            and value.co_name == "<lambda>"
            and {
                "resolve_turn_tool_policy_profile_id",
                "resolve_turn_persona_policy_rules",
            }
            <= set(value.co_names)
        ]
        assert len(callbacks) == 1
        self.callback_code = callbacks[0]
        preparation_code = self._pin(preparation.run_preparation_read)
        invokes = [
            value
            for value in preparation_code.co_consts
            if type(value) is CodeType and value.co_name == "invoke"
        ]
        assert len(invokes) == 1
        self.invoke_code = invokes[0]
        self.stack = (
            (self.body_code, vars(context)),
            (self.callback_code, vars(configuration)),
            (self.invoke_code, vars(preparation)),
            (self._pin(_WorkItem.run), _WorkItem.run.__globals__),
        )
        self.codes = {self.body_code, self.callback_code, self.invoke_code}

    def _start(self, observed_code, _offset):
        if observed_code is not self.body_code or not self.armed:
            return
        frame = sys._getframe(1)
        frames = []
        try:
            for code, namespace in self.stack:
                if (
                    frame is None
                    or frame.f_code is not code
                    or frame.f_globals is not namespace
                ):
                    return
                frames.append(frame)
                frame = frame.f_back
            callback = frames[2].f_locals["callback"]
            read = frames[2].f_locals["read"]
            with self.preparation._READS_LOCK:
                selected = (
                    read in self.controller._preparation_reads
                    and read in self.runtime._preparation_reads
                )
            if not (
                frames[0].f_locals["app"] is self.app
                and frames[0].f_locals["workspace_id"] == self.workspace
                and frames[1].f_locals["app"] is self.app
                and frames[1].f_locals["session"] is self.session
                and read.callback is callback
                and callback.__code__ is self.callback_code
                and callback.__globals__ is vars(self.configuration)
                and read.creator is self.controller
                and read.session_id == self.session.id
                and read.source is None
                and selected
                and self.controller.app is self.app
                and self.controller.store is self.store
                and self.controller._hooks_v2_runtime is self.runtime
                and self.runtime._chat_store is self.store
                and self.runtime._chat_controller is self.controller
                and self.console._console_runtime_ref is self.runtime
                and self.console.app_instance is self.app
                and self.app.console_runtime is self.runtime
                and self.store._sessions.get(self.session.id) is self.session
                and self.session.workspace_id == self.workspace
                and self.configuration.resolve_turn_tool_policy_profile_id
                is self.profile
                and self.profile.__code__ is self.body_code
            ):
                return
            future = frames[3].f_locals["self"].future
            assert future.running() and not future.done()
            self.read, self.future, self.callback = read, future, callback
            self.armed = False
            self.worker_thread = threading.current_thread()
            self.thread = self.worker_thread.name
        finally:
            frames.clear()
            del frame
        if not self.armed:
            self.entered.set()
            self.timed_out = not self.release.wait(timeout=HOLD_SECONDS)

    def _return(self, observed_code, _offset, _result):
        if self.armed or threading.current_thread() is not self.worker_thread:
            return
        frame = sys._getframe(1)
        try:
            if (
                observed_code is self.body_code
                and frame.f_locals.get("app") is self.app
            ):
                self.body_returned = True
            elif (
                observed_code is self.callback_code
                and frame.f_locals.get("session") is self.session
            ):
                self.callback_returned = True
            elif (
                observed_code is self.invoke_code
                and frame.f_locals.get("read") is self.read
            ):
                self.invoke_returned = True
        finally:
            del frame

    async def __aexit__(self, _kind, primary, _traceback):
        self.release.set()
        errors = []
        try:
            if self.entered.is_set():
                await until(
                    lambda: self.future.done() and self.read.retired.done(),
                    timeout=ENTRY_SECONDS,
                )
                self.future.result()
                assert self.read._producer.done() and not self.read.retired.cancelled()
                assert (
                    self.body_returned
                    and self.callback_returned
                    and self.invoke_returned
                )
                assert not self.timed_out, "original native profile hold timed out"
        except BaseException as error:  # noqa: BLE001 - retain cleanup and original test failures together.
            errors.append(error)
        finally:
            ownership_errors = (
                [RuntimeError("monitoring_callback_not_unowned")]
                if self.foreign_callback
                else []
            )
            mask = self.monitor.events.PY_START | self.monitor.events.PY_RETURN
            if self.entered.is_set() and not self.future.done():
                ownership_errors.append(
                    RuntimeError("original_native_worker_not_returned")
                )
            elif self.monitor.get_tool(self.tool) != self.tool_name:
                ownership_errors.append(RuntimeError("monitoring_tool_owner_changed"))
            else:
                for code in self.enabled_codes:
                    if self.monitor.get_local_events(self.tool, code) == mask:
                        self.monitor.set_local_events(self.tool, code, 0)
                    else:
                        ownership_errors.append(
                            RuntimeError("monitoring_local_events_changed")
                        )
                for event, expected in self.registered.items():
                    current = self.monitor.register_callback(self.tool, event, None)
                    if current is not expected:
                        self.monitor.register_callback(self.tool, event, current)
                        ownership_errors.append(
                            RuntimeError("monitoring_callback_owner_changed")
                        )
                if self.monitor.get_events(self.tool):
                    ownership_errors.append(
                        RuntimeError("monitoring_global_events_changed")
                    )
                if not ownership_errors:
                    self.monitor.free_tool_id(self.tool)
                    self.observer_retired = True
            errors.extend(ownership_errors)
        if errors:
            raise BaseExceptionGroup(
                "profile hold cleanup failed",
                ([primary] if primary is not None else []) + errors,
            )


async def _retire_busy_tab_workers(runtime, hooks, workers, hold, primary):
    """Bound waits on exact legacy workers and the actual runtime disposal."""
    from textual.worker import WorkerCancelled

    errors = []
    if hold is not None:
        hold.release.set()
    tasks = {}
    cancelled = set()
    for worker in workers:
        task = worker._task
        if worker._thread_worker or not isinstance(task, asyncio.Task):
            errors.append(RuntimeError("busy-tab worker task is not an async task"))
            continue
        tasks[task] = worker
        if not task.done():
            cancelled.add(worker)
            worker.cancel()
    if tasks:
        try:
            done, pending = await asyncio.wait(tasks, timeout=ENTRY_SECONDS)
            if pending:
                errors.append(RuntimeError("busy-tab worker physical return unknown"))
            for task in done:
                worker = tasks[task]
                try:
                    task.result()
                except asyncio.CancelledError:
                    pass
                except BaseException as error:  # noqa: BLE001 - preserve worker and primary failures.
                    errors.append(error)
                try:
                    await worker.wait()
                except WorkerCancelled as error:
                    if worker not in cancelled:
                        errors.append(error)
                except BaseException as error:  # noqa: BLE001 - preserve worker and primary failures.
                    errors.append(error)
        except BaseException as error:  # noqa: BLE001 - continue actual runtime disposal.
            errors.append(error)
    try:
        disposal = asyncio.create_task(runtime.dispose())
        done, _ = await asyncio.wait({disposal}, timeout=ENTRY_SECONDS)
        if not done:
            errors.append(RuntimeError("runtime disposal physical return unknown"))
        else:
            disposal.result()
            controller = runtime._chat_controller
            assert runtime._disposed and not runtime._preparation_reads
            assert controller is None or not controller._preparation_reads
            assert not hooks._native_reads
    except BaseException as error:  # noqa: BLE001 - preserve disposal and primary failures.
        errors.append(error)
    if errors:
        raise BaseExceptionGroup(
            "busy-tab cleanup failed",
            ([primary] if primary is not None else []) + errors,
        )


@pytest.mark.asyncio
async def test_a_send_refused_while_another_tab_sends_names_that_tab(record_property):
    """AC#3: original worker hook dispatch names the other busy tab.

    A and B call the wired hook API in Textual workers; Alt+2 is physical.
    This case does not claim a native voice send or a physical Enter in B.
    """
    from tldw_chatbook.UI.Console_Modules.prompt_queue import (
        ConsolePromptDispatchResult,
        ConsolePromptDispatchStatus,
    )

    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            runtime = console._console_runtime()
            workers = []
            hold = None
            try:
                session_a, session_b = await _second_tab(console, pilot, ready=True)
                store = console._ensure_console_chat_store()
                controller = console._ensure_console_chat_controller()
                session = next(
                    item for item in store.sessions() if item.id == session_a
                )
                store.rename_session(session_a, "Alpha plans")
                notices = _record_notices(host)
                hooks, queue = console._hooks, console._prompt_queue
                stash_a = composer.capture_draft_for_send()
                assert stash_a.text == DRAFT
                assert hooks._session() == session_a
                assert queue._chat_controller_accessor() is controller
                assert controller._hooks_v2_runtime is runtime
                gateway.validation_release.set()
                async with _HeldProfileRead(
                    console, controller, runtime, store, session
                ) as hold:
                    try:
                        worker_a = console.run_worker(
                            hooks.dispatch(
                                DRAFT,
                                session_id=session_a,
                                stash=stash_a,
                                dispatch=partial(
                                    queue.dispatch,
                                    DRAFT,
                                    session_id=session_a,
                                    stash=stash_a,
                                ),
                            ),
                            group="busy-tab-a",
                            exit_on_error=False,
                        )
                        workers.append(worker_a)
                        await until(hold.entered.is_set, timeout=ENTRY_SECONDS)
                        assert hooks._busy and hooks._busy_session == session_a
                        assert not worker_a.is_finished and gateway.stream_calls == 0
                        assert hold.future.running() and not hold.future.done()
                        assert (
                            not hold.read.retired.done() and not hold.release.is_set()
                        )
                        assert not hold.timed_out and not hold.body_returned
                        press(host, "alt+2")
                        await until(
                            lambda: (
                                console._console_visible_draft_session_id == session_b
                            )
                        )
                        await until(lambda: composer.draft_text() == B_DRAFT)
                        assert store.session_draft(session_b) == B_DRAFT
                        stash_b = composer.capture_draft_for_send()
                        calls_b = 0
                        continuation_b = partial(
                            queue.dispatch, B_DRAFT, session_id=session_b, stash=stash_b
                        )

                        async def dispatch_b():
                            nonlocal calls_b
                            calls_b += 1
                            return await continuation_b()

                        worker_b = console.run_worker(
                            hooks.dispatch(
                                B_DRAFT,
                                session_id=session_b,
                                stash=stash_b,
                                dispatch=dispatch_b,
                            ),
                            group="busy-tab-b",
                            exit_on_error=False,
                        )
                        workers.append(worker_b)
                        result_b = await worker_b.wait()
                        assert isinstance(result_b, ConsolePromptDispatchResult)
                        assert result_b.status is ConsolePromptDispatchStatus.REFUSED
                        assert (
                            result_b.session_id == session_b
                            and "Alpha plans" in result_b.detail
                        )
                        refusals = [n for n in notices if "already in progress" in n]
                        assert refusals and all("Alpha plans" in n for n in refusals), (
                            refusals
                        )
                        assert hooks._busy and hooks._busy_session == session_a
                        assert not worker_a.is_finished and not hold.future.done()
                        assert not hold.release.is_set() and not hold.timed_out
                        assert gateway.stream_calls == 0 and calls_b == 0
                        assert _users(console, session_b) == []
                        assert (
                            controller.prompt_queue_registry.snapshot(session_b).entries
                            == ()
                        )
                        assert store.session_draft(session_b) == B_DRAFT
                        assert composer.draft_text() == B_DRAFT
                    finally:
                        hold.release.set()
                    result_a = await worker_a.wait()
                    assert isinstance(result_a, ConsolePromptDispatchResult)
                    assert result_a.status is ConsolePromptDispatchStatus.SENT
                    assert result_a.session_id == session_a
                    await until(
                        lambda: _users(console, session_a) == [DRAFT],
                        timeout=ENTRY_SECONDS,
                    )
                    await until(
                        lambda: gateway.stream_calls == 1, timeout=ENTRY_SECONDS
                    )
                    await _idle(host, console, pilot, session_a)
                    assert _users(console, session_a) == [DRAFT]
                    assert _sent_or_queued(console, session_a) == [DRAFT]
                    assert _users(console, session_b) == []
                    assert (
                        controller.prompt_queue_registry.snapshot(session_b).entries
                        == ()
                    )
                    assert _sent_or_queued(console, session_b) == []
                    assert calls_b == 0 and gateway.stream_calls == 1
                    assert store.session_draft(session_b) == B_DRAFT
                    assert composer.draft_text() == B_DRAFT
                assert hold.observer_retired
                hold.read._producer.result()
                hold.read.retired.result()
                record_property(
                    "original_native_profile_return",
                    {
                        "body": hold.body_code.co_qualname,
                        "callback": hold.callback_code.co_qualname,
                        "invoke": hold.invoke_code.co_qualname,
                        "thread": hold.thread,
                        "entered": hold.entered.is_set(),
                        "released": hold.release.is_set(),
                        "timed_out": hold.timed_out,
                        "body_returned": hold.body_returned,
                        "callback_returned": hold.callback_returned,
                        "invoke_returned": hold.invoke_returned,
                        "physical_future_done": hold.future.done(),
                        "producer_done": hold.read._producer.done(),
                        "read_retired": hold.read.retired.done(),
                        "observer_retired": hold.observer_retired,
                    },
                )
            finally:
                await _retire_busy_tab_workers(
                    runtime, console._hooks, workers, hold, sys.exception()
                )


@pytest.mark.asyncio
async def test_cancelling_profile_entry_wait_retires_preentry_review(
    hook_file, record_property
):
    """Caller cancellation retires an actual Send waiting on hook consent."""
    from textual.worker import WorkerState

    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_chat_models import CONSOLE_PENDING_HOOK_REVIEW_KIND
    from tldw_chatbook.Widgets.Console.console_hooks_review_modal import (
        ConsoleHooksReviewModal,
    )

    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            runtime = console._console_runtime()
            store = console._ensure_console_chat_store()
            controller = console._ensure_console_chat_controller()
            hooks, queue = console._hooks, console._prompt_queue
            session_id = store.active_session_id
            session = next(item for item in store.sessions() if item.id == session_id)
            stash = composer.capture_draft_for_send()
            snapshot = runtime.ensure_hook_permissions().snapshot()
            assert snapshot.config.config_path.resolve() == hook_file.resolve()
            assert snapshot.config.profile_data_dir == config.get_user_data_dir()
            assert (
                not snapshot.ready
                and snapshot.blocked_reason == "Review enabled hooks before sending."
            )
            assert any(
                row.entry is not None and row.state == "pending"
                for row in snapshot.rows
            )
            hold = _HeldProfileRead(console, controller, runtime, store, session)
            workers = []

            async def subject():
                try:
                    async with hold:
                        worker = console.run_worker(
                            hooks.dispatch(
                                DRAFT,
                                session_id=session_id,
                                stash=stash,
                                dispatch=partial(
                                    queue.dispatch,
                                    DRAFT,
                                    session_id=session_id,
                                    stash=stash,
                                ),
                            ),
                            group="busy-tab-cancel",
                            exit_on_error=False,
                        )
                        workers.append(worker)
                        await until(hold.entered.is_set, timeout=ENTRY_SECONDS)
                finally:
                    await _retire_busy_tab_workers(
                        runtime, hooks, workers, hold, sys.exception()
                    )

            task = asyncio.create_task(subject())
            try:
                await until(
                    lambda: (
                        hooks.review_open
                        and isinstance(host.screen, ConsoleHooksReviewModal)
                        and host.screen.is_mounted
                    ),
                    timeout=ENTRY_SECONDS,
                )
                waiting = task.get_coro().cr_await
                assert waiting is not None and waiting.cr_code is until.__code__
                assert waiting.cr_frame.f_locals["timeout"] == ENTRY_SECONDS
                assert len(workers) == 1 and not workers[0]._task.done()
                assert hooks._busy and hooks._busy_session == session_id
                assert not hold.entered.is_set() and gateway.stream_calls == 0
                review_host = controller._interrupt_host
                with review_host.lock:
                    (review,) = tuple(
                        review_host.registries[
                            CONSOLE_PENDING_HOOK_REVIEW_KIND
                        ].values()
                    )
                    answer = review["answer"]
                    assert not review["settled"] and not answer.done()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert (
                    workers[0]._task.done()
                    and workers[0].state is WorkerState.CANCELLED
                )
                assert not hooks._busy and not hooks.review_open
                assert (
                    hold.release.is_set()
                    and hold.observer_retired
                    and not hold.entered.is_set()
                )
                assert runtime._disposed and not runtime._preparation_reads
                assert not controller._preparation_reads and not hooks._native_reads
                with review_host.lock:
                    assert not review_host.registries[CONSOLE_PENDING_HOOK_REVIEW_KIND]
                await until(answer.done, timeout=ENTRY_SECONDS)
                assert answer.result().kind == "cancel"
                await until(lambda: host.screen is console, timeout=ENTRY_SECONDS)
                assert gateway.stream_calls == 0 and _users(console, session_id) == []
                assert (
                    controller.prompt_queue_registry.snapshot(session_id).entries == ()
                )
                assert store.session_draft(session_id) == DRAFT
                assert composer.draft_text() == DRAFT
            finally:
                if not task.done():
                    task.cancel()
                    done, _ = await asyncio.wait({task}, timeout=ENTRY_SECONDS)
                    if done:
                        try:
                            task.result()
                        except asyncio.CancelledError:
                            pass
            assert task.done() and task.cancelled()
    record_property("preentry_review_caller_cancel_retired", True)


BLOCKED = "Console send blocked: refused for this test."


def _rows(console, session_id: str, text: str) -> int:
    store = console._ensure_console_chat_store()
    return sum(
        text in (message.content or "")
        for message in store.messages_for_session(session_id)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("draft", ["", DRAFT], ids=["image_only", "text"])
async def test_a_bare_second_enter_during_a_refused_send_is_not_refused_again(draft):
    """AC#4: a bare repeat press is one press, image-only or text.

    The first send is held at its hook read while Enter is pressed again with
    nothing new typed; the send is then refused. A bare text repeat is held
    and replayed only while its capture still shows; an image-only press was
    never treated as a repeat, so its replay was refused a second time and
    wrote the refusal twice.
    """
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            composer.load_draft(draft)
            await pilot.pause()
            session_id = console._console_chat_store.active_session_id
            if not draft:
                image = object()
                console._console_pending_image_attachment = lambda: image
            # The send gate the prompt queue reads (TASK-33621.11: coded).
            console._console_send_gate = lambda: ConsoleBlockedReason(BLOCKED)
            gate = HeldAdmission(console)
            try:
                press(host, "enter", "\r")
                await until(gate.entered.is_set, timeout=ENTRY_SECONDS)
                press(host, "enter", "\r")
                await pilot.pause(0.2)
            finally:
                gate.release.set()
            await until(lambda: _rows(console, session_id, BLOCKED) >= 1)
            await _settled(console, pilot)
            await pilot.pause(0.5)
            assert _rows(console, session_id, BLOCKED) == 1
            assert gateway.stream_calls == 0
            assert composer.draft_text() == draft


# Checkpoint-review fixes. A clear or a replaced draft is not a send: it
# never spends a press made before it, and never draws the "was sent"
# notice. These pass on dev, which did both right, and failed on the first
# cut of this task, which keyed both on any move of the draft generation.

ESCAPED = r"\! hello there"


async def _sends(console, session_id: str, count: int) -> list[str]:
    """What the chat sent or queued, once ``count`` arrived or time ran out."""
    try:
        await until(lambda: len(_sent_or_queued(console, session_id)) >= count)
    except AssertionError:
        pass
    return _sent_or_queued(console, session_id)


def _clear(host, how: str) -> None:
    if how == "ctrl+u":
        press(host, "ctrl+u")
        return
    for _ in DRAFT:
        press(host, "backspace")


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["ctrl+u", "backspace"], ids=["ctrl_u", "backspaces"])
async def test_a_draft_cleared_while_it_is_sent_draws_no_was_sent_notice(how):
    """The notice is only for sent text the composer still shows.

    Enter, then the draft cleared (Ctrl+U, or Backspace over all of it) and
    "n" typed while the send is admitted. The commit cannot take the sent
    text out because it is gone; the first cut still said "Your message was
    sent, but the composer still shows it" over a composer showing "n".
    The "n" is a new message: Enter on it later sends it (Backspace leaves
    the draft's generation as it was, and a spent generation must not
    swallow it).
    """
    host, gateway, _timeline = build()
    _allow_second_turns(host)
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            session_id = console._console_chat_store.active_session_id
            notices = _record_notices(host)
            async with HeldMcpRead(host.app_instance.unified_mcp_service) as hold:
                gateway.validation_release.set()
                try:
                    press(host, "enter", "\r")
                    await until(hold.entered.is_set, timeout=ENTRY_SECONDS)
                    _clear(host, how)
                    await until(lambda: composer.draft_text() == "", timeout=10)
                    press(host, "n", "n")
                    await until(lambda: composer.draft_text() == "n", timeout=10)
                finally:
                    hold.release.set()
                await until(lambda: gateway.stream_calls == 1)
                await _idle(host, console, pilot, session_id)
                assert _sent_or_queued(console, session_id) == [DRAFT]
                assert composer.draft_text() == "n"
                assert not [n for n in notices if "was sent" in n], notices
                await until(lambda: _turns_done(console, session_id, 1))
                press(host, "enter", "\r")
                assert await _sends(console, session_id, 2) == [DRAFT, "n"]
                await _idle(host, console, pilot, session_id)
                assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_a_hidden_chat_whose_draft_was_replaced_draws_no_was_sent_notice():
    """The hidden-chat notice is only for a saved draft that still has it.

    Enter, Ctrl+U, "n", then Alt+2 while the send is admitted: tab A saves
    "n". The first cut said "Your message in “…” was sent, but that chat's
    draft still shows it" for any saved draft not starting with the sent
    text, including this one.
    """
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            session_a, session_b = await _second_tab(console, pilot, ready=True)
            store = console._ensure_console_chat_store()
            notices = _record_notices(host)
            async with HeldMcpRead(host.app_instance.unified_mcp_service) as hold:
                gateway.validation_release.set()
                try:
                    press(host, "enter", "\r")
                    await until(hold.entered.is_set, timeout=ENTRY_SECONDS)
                    press(host, "ctrl+u")
                    await until(lambda: composer.draft_text() == "", timeout=10)
                    press(host, "n", "n")
                    await until(lambda: composer.draft_text() == "n", timeout=10)
                    press(host, "alt+2")
                    await until(
                        lambda: console._console_visible_draft_session_id == session_b
                    )
                finally:
                    hold.release.set()
                await until(lambda: gateway.stream_calls == 1)
                await _idle(host, console, pilot, session_a)
                assert _sent_or_queued(console, session_a) == [DRAFT]
                assert store.session_draft(session_a) == "n"
                assert not [n for n in notices if "was sent" in n], notices


@pytest.mark.asyncio
async def test_an_escaped_bang_chat_send_leaves_the_composer():
    """A "\\! " chat send goes out as "! …" and leaves the composer.

    The send dispatches the draft with its escape removed, and the commit
    looked for that unescaped text, which the composer never shows. On dev
    the sent draft stayed in the composer with no cue; the first cut of this
    task said "the draft changed while sending" when nothing had changed.
    """
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            composer.load_draft(ESCAPED)
            await pilot.pause()
            session_id = console._console_chat_store.active_session_id
            store = console._ensure_console_chat_store()
            notices = _record_notices(host)
            gateway.validation_release.set()
            press(host, "enter", "\r")
            await until(lambda: gateway.stream_calls == 1, timeout=ENTRY_SECONDS)
            await _idle(host, console, pilot, session_id)
            assert _sent_or_queued(console, session_id) == [ESCAPED[1:]]
            assert composer.draft_text() == ""
            assert store.session_draft(session_id) == ""
            assert not [n for n in notices if "was sent" in n], notices


@pytest.mark.asyncio
async def test_a_held_enter_is_still_sent_when_the_composer_is_cleared_after_it():
    """Clearing the composer after a press does not cancel that press.

    The first send's draft has left the composer and its send is still
    running, so "y" plus Enter is held. Then Ctrl+U and "z". Lead ruling:
    what is sent is what the composer held at the press, so "y" is sent and
    "z" stays. The first cut took any move of the draft generation (a clear
    included) for "another send took this draft" and dropped "y" silently:
    neither sent nor left in the composer.
    """
    host, gateway, _timeline = build()
    _allow_second_turns(host)
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            session_id = console._console_chat_store.active_session_id
            notices = _record_notices(host)
            tail = _hold_first_send_tail(console)
            gateway.validation_release.set()
            try:
                press(host, "enter", "\r")
                await until(lambda: composer.draft_text() == "", timeout=ENTRY_SECONDS)
                press(host, "y", "y")
                press(host, "enter", "\r")
                await until(lambda: _held(console), timeout=10)
                press(host, "ctrl+u")
                await until(lambda: composer.draft_text() == "", timeout=10)
                press(host, "z", "z")
                await until(lambda: composer.draft_text() == "z", timeout=10)
                await until(lambda: _first_turn_done(host, console))
            finally:
                tail.set()
            assert await _sends(console, session_id, 2) == [DRAFT, "y"]
            await _idle(host, console, pilot, session_id)
            assert gateway.stream_calls == 2
            assert composer.draft_text() == "z"
            assert not [n for n in notices if "was sent" in n], notices


@pytest.mark.asyncio
async def test_an_enter_is_still_sent_when_the_composer_is_cleared_before_it_starts(
    monkeypatch,
):
    """A clear before the Enter's send starts does not cancel it either.

    Enter on "y"; its "Sending…" row is painted, and the send is started
    once that frame is out. Ctrl+U and "z" land in between. The first cut's
    check at the start of the send took the clear for another send and
    dropped "y" silently.
    """
    from tldw_chatbook.UI.Console_Modules import send_acknowledgement

    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            composer.load_draft("y")
            await pilot.pause()
            session_id = console._console_chat_store.active_session_id
            notices = _record_notices(host)
            gateway.validation_release.set()
            reached, release = asyncio.Event(), asyncio.Event()
            real_start = send_acknowledgement._start_send

            def held_start(screen, send):
                async def start_when_released():
                    reached.set()
                    await release.wait()
                    real_start(screen, send)

                asyncio.get_running_loop().create_task(start_when_released())

            monkeypatch.setattr(send_acknowledgement, "_start_send", held_start)
            try:
                press(host, "enter", "\r")
                await until(reached.is_set, timeout=ENTRY_SECONDS)
                press(host, "ctrl+u")
                await until(lambda: composer.draft_text() == "", timeout=10)
                press(host, "z", "z")
                await until(lambda: composer.draft_text() == "z", timeout=10)
            finally:
                release.set()
            assert await _sends(console, session_id, 1) == ["y"]
            await _idle(host, console, pilot, session_id)
            assert gateway.stream_calls == 1
            assert composer.draft_text() == "z"
            assert not [n for n in notices if "was sent" in n], notices


@pytest.mark.asyncio
async def test_an_enter_never_resends_a_draft_a_spoken_send_took_before_a_clear(
    monkeypatch,
):
    """A sent capture is spent even when its commit cannot take it out.

    A spoken send of "y" is admitted when Enter captures the same "y"; the
    Enter's send is about to start. Ctrl+U and "z" land, so the spoken send
    sends "y" and its commit finds nothing to take out. The Enter's capture
    is still the text that went: it is dropped, and "z" stays. (On dev the
    Enter sent "y" a second time.)
    """
    from tldw_chatbook.UI.Console_Modules import send_acknowledgement

    host, gateway, _timeline = build()
    _allow_second_turns(host)
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            composer.load_draft("y")
            await pilot.pause()
            session_id = console._console_chat_store.active_session_id
            notices = _record_notices(host)
            async with HeldMcpRead(host.app_instance.unified_mcp_service) as hold:
                gateway.validation_release.set()
                reached, release = asyncio.Event(), asyncio.Event()
                real_start = send_acknowledgement._start_send

                def held_start(screen, send):
                    async def start_when_released():
                        reached.set()
                        await release.wait()
                        real_start(screen, send)

                    asyncio.get_running_loop().create_task(start_when_released())

                monkeypatch.setattr(send_acknowledgement, "_start_send", held_start)
                try:
                    spoken = _speak_send(console, session_id)
                    await until(hold.entered.is_set, timeout=ENTRY_SECONDS)
                    press(host, "enter", "\r")
                    await until(reached.is_set, timeout=10)
                    press(host, "ctrl+u")
                    await until(lambda: composer.draft_text() == "", timeout=10)
                    press(host, "z", "z")
                    await until(lambda: composer.draft_text() == "z", timeout=10)
                    hold.release.set()
                    await until(lambda: spoken.is_finished, timeout=ENTRY_SECONDS)
                    # Started now, the Enter's send would be a second turn.
                    await until(lambda: _turns_done(console, session_id, 1))
                finally:
                    hold.release.set()
                    release.set()
                await _idle(host, console, pilot, session_id)
                await pilot.pause(0.5)
                assert _sent_or_queued(console, session_id) == ["y"]
                assert gateway.stream_calls == 1
                assert composer.draft_text() == "z"
                assert not [n for n in notices if "was sent" in n], notices
