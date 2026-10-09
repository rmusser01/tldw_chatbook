"""A Console send's admission never holds key delivery (TASK-33620.15).

Measured before the fix (mounted harness, one cold send): the turn
configuration builder ran 531 ms on the UI pump, reading the MCP definition
maximum, the skill catalog, project authority, RAG profile depth and the
scratch space; live (Anthropic haiku, 160x45) a first send held keys for
613 ms. Two causes, both pinned here:

* Those reads ran on the UI thread.
* The send itself ran inside an app-pump callback, which Textual awaits: any
  await in the send -- even a read moved to a worker thread -- kept the app
  pump from delivering the next key.

The admission test holds the MCP read that admission makes (the real
service, wrapped where ``capture_mcp_definition_maximum`` calls it) and types
a key while it is held. On the parent build the read ran on the loop thread,
so no key could be handled until the hold timed out. Negative controls (run
by hand, TASK-33620.15 notes): sending from inside the app-pump callback, or
reading the authority on the loop thread, turns it red.

The Send button and the Workbench send now take Enter's path, so the frame on
screen when their send starts is the same "Sending…" acknowledgement.
"""

from __future__ import annotations

import asyncio
import sys
import threading

import pytest

from Tests.UI.test_console_send_acknowledgement import (
    DRAFT,
    REPLY,
    _painted_lines,
    build,
    eager_tasks,
    mark_send_start,
    press,
    ready_console,
    until,
)
from tldw_chatbook.UI.Workbench.workbench_widgets import WorkbenchActionRequested

pytestmark = pytest.mark.bootstrap_profile

#: How long the held read waits for the test before giving up. Long enough
#: for a free UI pump to deliver a key many times over, even on a loaded host.
HOLD_SECONDS = 10.0
#: How long the send may take to reach that read (a loaded host is slow to
#: open the worker's first database connections).
ENTRY_SECONDS = 30.0


class HeldMcpRead:
    """Hold the MCP read admission makes, until the test releases it.

    Wraps the real ``unified_mcp_service.get_kill_switch``; only the first
    call made by ``capture_mcp_definition_maximum`` (admission's capture, on
    whichever thread admission reads it) is held. Every other caller, such as
    the run's own catalog composition, passes straight through.
    """

    def __init__(self, service) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.timed_out = False
        self.thread: str | None = None
        self.armed = True
        real = service.get_kill_switch

        def get_kill_switch():
            caller = sys._getframe(1).f_code.co_name
            if self.armed and caller == "capture_mcp_definition_maximum":
                self.armed = False
                self.thread = threading.current_thread().name
                self.entered.set()
                self.timed_out = not self.release.wait(timeout=HOLD_SECONDS)
            return real()

        service.get_kill_switch = get_kill_switch


@pytest.mark.asyncio
async def test_a_key_typed_while_admission_reads_mcp_is_handled_before_the_read_returns():
    """AC#2/#6: keys keep flowing while the send's admission reads its authority.

    The key is typed into the composer while admission's MCP read is held;
    it must reach the composer before the read is released. It then belongs
    to the next draft: the provider gets the captured draft and the composer
    keeps the typed character once the turn is accepted.
    """
    host, gateway, timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            hold = HeldMcpRead(host.app_instance.unified_mcp_service)
            gateway.validation_release.set()
            try:
                press(host, "enter", "\r")
                await until(hold.entered.is_set, timeout=ENTRY_SECONDS)
                press(host, "x", "x")
                await until(lambda: composer.draft_text().endswith("x"), timeout=10)
                typed_while_held = not hold.release.is_set() and not hold.timed_out
            finally:
                hold.release.set()
            assert typed_while_held, (
                "admission held the UI loop: the typed key was handled only "
                f"after its MCP read (thread {hold.thread!r}) gave up waiting"
            )
            assert hold.thread != threading.main_thread().name, hold.thread
            await until(lambda: gateway.stream_calls == 1)
            await until(lambda: REPLY in "\n".join(_painted_lines(host)))
            assert composer.draft_text() == "x"
            store = console._ensure_console_chat_store()
            users = [
                message.content
                for message in store.messages_for_session(store.active_session_id)
                if message.role.value == "user"
            ]
            assert users == [DRAFT]


async def _assert_acknowledged_when_send_starts(host, gateway, timeline, trigger):
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            at_send = mark_send_start(console, timeline)
            timeline.install()
            try:
                trigger(console)
                await until(gateway.validation_started.is_set)
                assert len(at_send) == 1
                shown = at_send[0]
                assert shown is not None, "nothing was written before the send"
                assert shown.user_row and shown.sending, shown
                assert shown.tab_running and shown.run_chip, shown
                assert not shown.header_idle, shown
                assert not shown.empty_draft_reason, shown
                assert gateway.stream_calls == 0
            finally:
                gateway.validation_release.set()
            await until(lambda: gateway.stream_calls == 1)
            await until(lambda: REPLY in "\n".join(_painted_lines(host)))


@pytest.mark.asyncio
async def test_the_send_button_paints_sending_before_its_send_starts():
    """AC#4: a mouse Send shows Enter's acknowledgement before its send runs."""
    from textual.widgets import Button

    host, gateway, timeline = build()
    await _assert_acknowledged_when_send_starts(
        host,
        gateway,
        timeline,
        lambda console: console.query_one("#console-send-message", Button).press(),
    )


@pytest.mark.asyncio
async def test_the_workbench_send_paints_sending_before_its_send_starts():
    """AC#4: the Workbench's send shows Enter's acknowledgement too."""
    host, gateway, timeline = build()
    await _assert_acknowledged_when_send_starts(
        host,
        gateway,
        timeline,
        lambda console: console.post_message(WorkbenchActionRequested("send")),
    )


@pytest.mark.asyncio
async def test_enter_during_admission_is_replayed_after_it_as_the_held_pump_did():
    """The repeated-Enter guard (dev parity): no duplicate, no lost text.

    Before the fix the app pump was busy for the whole admission, so a second
    Enter -- and anything typed -- was handled once the first send had been
    accepted and its draft committed: a bare double press found an empty
    draft, and typed text plus Enter met the next turn's own gate. The
    admission no longer holds the pump, so a send request made during it is
    replayed once it settles. Without that, the second Enter re-sent the
    still-captured draft into the busy hook gate ("already in progress").
    """
    host, gateway, timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            notices: list[str] = []
            real_notify = host.app_instance.notify
            host.app_instance.notify = lambda message, **kwargs: (
                notices.append(str(message)),
                real_notify(message, **kwargs),
            )
            hold = HeldMcpRead(host.app_instance.unified_mcp_service)
            gateway.validation_release.set()
            try:
                press(host, "enter", "\r")
                await until(hold.entered.is_set, timeout=ENTRY_SECONDS)
                press(host, "enter", "\r")
                press(host, "y", "y")
                press(host, "enter", "\r")
                await until(lambda: composer.draft_text().endswith("y"), timeout=10)
            finally:
                hold.release.set()
            await until(lambda: gateway.stream_calls >= 1)
            await until(lambda: REPLY in "\n".join(_painted_lines(host)))
            flight = getattr(console, "_console_send_flight", None)
            await until(lambda: flight is None or not (flight.tasks or flight.deferred))
            await pilot.pause(0.3)
            controller = console._ensure_console_chat_controller()
            store = console._ensure_console_chat_store()
            session_id = store.active_session_id
            users = [
                message.content
                for message in store.messages_for_session(session_id)
                if message.role.value == "user"
            ]
            queued = [
                entry.preview
                for entry in controller.prompt_queue_registry.snapshot(
                    session_id
                ).entries
            ]
            assert users.count(DRAFT) == 1, users
            kept = users + queued + [composer.draft_text()]
            assert kept.count("y") == 1, kept
            assert not [n for n in notices if "already in progress" in n], notices


@pytest.mark.asyncio
async def test_a_key_typed_while_the_hook_read_is_held_joins_the_next_draft():
    """Type-ahead never cancels the send it follows (TASK-340's contract).

    The send's first await is its hook-permission read. Keys now flow during
    it, and the hook gate compared the composer with the captured draft, so
    the typed key refused the send ("Draft, chat or hooks changed; Send
    again."). A key typed straight after Enter did the same on dev (it lands
    before the send starts). The captured draft is what Enter sends; the key
    belongs to the next draft. Only a review, which takes the user's time,
    re-checks the draft.
    """
    from Tests.UI.test_console_send_acknowledgement import HeldAdmission

    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            gate = HeldAdmission(console)
            gateway.validation_release.set()
            try:
                press(host, "enter", "\r")
                await until(gate.entered.is_set, timeout=ENTRY_SECONDS)
                press(host, "x", "x")
                await until(lambda: composer.draft_text().endswith("x"), timeout=10)
            finally:
                gate.release.set()
            await until(lambda: gateway.stream_calls == 1)
            await until(lambda: REPLY in "\n".join(_painted_lines(host)))
            assert composer.draft_text() == "x"
            assert _users(console, console._console_chat_store.active_session_id) == [
                DRAFT
            ]


B_DRAFT = "private draft for tab b"


def _users(console, session_id: str) -> list[str]:
    store = console._ensure_console_chat_store()
    return [
        message.content
        for message in store.messages_for_session(session_id)
        if message.role.value == "user"
    ]


async def _second_tab(console, pilot, *, ready: bool) -> tuple[str, str]:
    """Open tab B holding its own unsent draft; return to tab A's draft."""
    from Tests.UI.test_console_native_chat_flow import _select_llamacpp_console

    store = console._ensure_console_chat_store()
    session_a = store.active_session_id
    session_b = store.create_session(title="Session B").id
    if ready:
        await console._session._activate_native_console_session(session_b)
        await pilot.pause(0.2)
        _select_llamacpp_console(console)
        await pilot.pause(0.2)
    store.set_session_draft(session_b, B_DRAFT)
    await console._session._activate_native_console_session(session_a)
    await pilot.pause(0.3)
    composer = console._console_composer_or_none()
    composer.load_draft(DRAFT)
    composer.focus()
    await pilot.pause()
    return session_a, session_b


async def _settled(console, pilot) -> None:
    flight = getattr(console, "_console_send_flight", None)
    await until(lambda: flight is None or not (flight.tasks or flight.deferred))
    await pilot.pause(1.0)


@pytest.mark.asyncio
async def test_an_enter_deferred_behind_a_send_never_sends_another_tab_draft():
    """A deferred send request replays only in the chat it was made in.

    A second Enter during A's admission is deferred until A's send settles.
    It was replayed in whatever chat was visible then: Enter, Enter, then
    Alt+2 sent tab B's unsent draft to the provider (stream_calls 2). The
    parent build's busy pump ran that Enter in tab A, before the switch.
    """
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            session_a, session_b = await _second_tab(console, pilot, ready=True)
            store = console._ensure_console_chat_store()
            hold = HeldMcpRead(host.app_instance.unified_mcp_service)
            gateway.validation_release.set()
            try:
                press(host, "enter", "\r")
                await until(hold.entered.is_set, timeout=ENTRY_SECONDS)
                press(host, "enter", "\r")
                press(host, "alt+2")
                await until(lambda: store.active_session_id == session_b, timeout=10)
            finally:
                hold.release.set()
            await until(lambda: gateway.stream_calls >= 1)
            await _settled(console, pilot)
            assert _users(console, session_a) == [DRAFT]
            assert _users(console, session_b) == []
            assert store.session_draft(session_b) == B_DRAFT
            assert gateway.stream_calls == 1


@pytest.mark.asyncio
async def test_leaving_a_tab_while_its_send_is_admitted_still_sends_it_there():
    """The send's gate is read for its own chat, never the one switched to.

    The screen's send gate (provider readiness, archive, evidence, vision)
    describes the visible chat. It was read after admission's authority read,
    while keys flow, so Alt+2 to an unconfigured tab refused tab A's send
    with tab B's reason ("Add API key ..."). The parent build sent it.
    """
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            session_a, session_b = await _second_tab(console, pilot, ready=False)
            store = console._ensure_console_chat_store()
            notices: list[str] = []
            real_notify = host.app_instance.notify
            host.app_instance.notify = lambda message, **kwargs: (
                notices.append(str(message)),
                real_notify(message, **kwargs),
            )
            hold = HeldMcpRead(host.app_instance.unified_mcp_service)
            gateway.validation_release.set()
            try:
                press(host, "enter", "\r")
                await until(hold.entered.is_set, timeout=ENTRY_SECONDS)
                press(host, "alt+2")
                await until(lambda: store.active_session_id == session_b, timeout=10)
            finally:
                hold.release.set()
            await until(lambda: gateway.stream_calls == 1)
            await _settled(console, pilot)
            assert _users(console, session_a) == [DRAFT]
            assert _users(console, session_b) == []
            assert store.session_draft(session_b) == B_DRAFT
            assert not [n for n in notices if "API key" in n], notices


def _sent_or_queued(console, session_id: str) -> list[str]:
    controller = console._ensure_console_chat_controller()
    queued = controller.prompt_queue_registry.snapshot(session_id).entries
    return _users(console, session_id) + [entry.preview for entry in queued]


def _record_notices(host) -> list[str]:
    notices: list[str] = []
    real_notify = host.app_instance.notify
    host.app_instance.notify = lambda message, **kwargs: (
        notices.append(str(message)),
        real_notify(message, **kwargs),
    )
    return notices


def _hold_first_send_tail(console) -> asyncio.Event:
    """Keep the first visible send running after its own dispatch returns.

    A send request made meanwhile is deferred until the running send
    settles. Released once the first turn has finished (``_first_turn_done``),
    the deferred request is an ordinary second turn, so what it carries is
    what reaches the provider. (Released earlier it meets "Preparing the
    current turn" and is refused, whatever it captured.)
    """
    tail = asyncio.Event()
    real_send = console._send_console_message_from_visible_action
    calls: list[object] = []

    async def send_then_hold(**kwargs):
        calls.append(kwargs)
        sent = await real_send(**kwargs)
        if len(calls) == 1:
            await tail.wait()
        return sent

    console._send_console_message_from_visible_action = send_then_hold
    return tail


def _allow_second_turns(host) -> None:
    """A second turn reads its persisted chat's archive state; none archived."""
    from types import SimpleNamespace

    host.app_instance.local_chat_conversation_service = SimpleNamespace(
        db=SimpleNamespace(is_memory_db=True),
        get_conversation_archive_states=lambda _ids: {},
    )


def _first_turn_done(host, console) -> bool:
    if REPLY not in "\n".join(_painted_lines(host)):
        return False
    controller = console._ensure_console_chat_controller()
    activity = controller.activity_for(console._console_chat_store.active_session_id)
    return not (activity.occupies_slot or activity.accepted_live_turn)


@pytest.mark.asyncio
async def test_an_enter_made_while_the_sent_draft_still_shows_never_takes_later_keys():
    """Lead ruling: a deferred Enter sends only what the composer held then.

    Enter, then "y" and Enter while the first send is admitted, then "z". The
    deferred Enter was replayed once the first send settled by capturing the
    composer at that moment, so "z" -- typed after it -- was sent with it
    ("yz"). At that press the composer still showed the first send's
    uncommitted draft, so "y" alone cannot be captured safely: the press is
    refused with a notice, and everything typed stays in the composer.
    """
    host, gateway, _timeline = build()
    _allow_second_turns(host)
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            notices = _record_notices(host)
            tail = _hold_first_send_tail(console)
            hold = HeldMcpRead(host.app_instance.unified_mcp_service)
            gateway.validation_release.set()
            try:
                press(host, "enter", "\r")
                await until(hold.entered.is_set, timeout=ENTRY_SECONDS)
                press(host, "y", "y")
                press(host, "enter", "\r")
                press(host, "z", "z")
                await until(lambda: composer.draft_text().endswith("z"), timeout=10)
            finally:
                hold.release.set()
            try:
                await until(lambda: _first_turn_done(host, console))
            finally:
                tail.set()
            await _settled(console, pilot)
            await until(lambda: _first_turn_done(host, console))
            await _settled(console, pilot)
            session_id = console._console_chat_store.active_session_id
            assert _sent_or_queued(console, session_id) == [DRAFT]
            assert composer.draft_text() == "yz"
            assert [n for n in notices if "still being sent" in n], notices


@pytest.mark.asyncio
async def test_an_enter_deferred_behind_a_send_sends_its_own_capture_only():
    """Lead ruling: the deferred Enter carries the draft captured at its press.

    The first send's draft has left the composer, but its send is still
    running, so a second Enter is deferred. It replayed by capturing the
    composer when it ran, so "z", typed after that Enter, was sent with it
    ("yz"). It now sends "y", and "z" stays as the next draft.
    """
    host, gateway, _timeline = build()
    _allow_second_turns(host)
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            tail = _hold_first_send_tail(console)
            gateway.validation_release.set()
            try:
                press(host, "enter", "\r")
                await until(lambda: composer.draft_text() == "", timeout=ENTRY_SECONDS)
                press(host, "y", "y")
                press(host, "enter", "\r")
                press(host, "z", "z")
                await until(lambda: composer.draft_text() == "yz", timeout=10)
                await until(lambda: _first_turn_done(host, console))
            finally:
                tail.set()
            await _settled(console, pilot)
            session_id = console._console_chat_store.active_session_id
            await until(lambda: len(_sent_or_queued(console, session_id)) >= 2)
            assert _sent_or_queued(console, session_id) == [DRAFT, "y"]
            assert composer.draft_text() == "z"


UNKNOWN = "/nope x"


@pytest.mark.asyncio
async def test_a_second_enter_after_an_unknown_command_hint_sends_it_as_text():
    """The unknown-command "Press Enter again" works while its send settles.

    The first Enter shows the unknown-command hint and leaves the draft in
    the composer, but its send can still be settling when the second Enter
    lands (a loaded host widens that window). A press repeating the running
    send's capture was dropped, on the assumption that the running send
    would commit it: nothing was sent and the draft stayed. The press is
    held now, and sent because its draft never left the composer.
    """
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            composer.load_draft(UNKNOWN)
            await pilot.pause()
            tail = _hold_first_send_tail(console)
            gateway.validation_release.set()
            try:
                press(host, "enter", "\r")
                await until(lambda: console._console_unknown_send_armed == UNKNOWN)
                press(host, "enter", "\r")
                await pilot.pause(0.2)
            finally:
                tail.set()
            await _settled(console, pilot)
            session_id = console._console_chat_store.active_session_id
            assert _sent_or_queued(console, session_id) == [UNKNOWN]
            assert composer.draft_text() == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("then_y", [False, True], ids=["alone", "then_y"])
async def test_a_bare_second_enter_never_sends_a_committed_draft_again(then_y):
    """A held repeat press is dropped once its draft has left the composer.

    The second Enter lands while the first send's draft is still in the
    composer, so it is held; the first send commits that draft and is
    answered before the held press is looked at. Sent then, it would be a
    second turn carrying the same text. Nor does it keep the next press out:
    "y" and Enter typed once the draft has left are sent, as they were when
    a repeat press was dropped at once, and "z" typed after them stays.
    """
    host, gateway, _timeline = build()
    _allow_second_turns(host)
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            notices = _record_notices(host)
            tail = _hold_first_send_tail(console)
            hold = HeldMcpRead(host.app_instance.unified_mcp_service)
            gateway.validation_release.set()
            try:
                press(host, "enter", "\r")
                await until(hold.entered.is_set, timeout=ENTRY_SECONDS)
                press(host, "enter", "\r")
                await pilot.pause(0.2)
            finally:
                hold.release.set()
            try:
                if then_y:
                    await until(lambda: composer.draft_text() == "", timeout=10)
                    press(host, "y", "y")
                    press(host, "enter", "\r")
                    press(host, "z", "z")
                    await until(lambda: composer.draft_text() == "yz", timeout=10)
                await until(lambda: _first_turn_done(host, console))
            finally:
                tail.set()
            await _settled(console, pilot)
            session_id = console._console_chat_store.active_session_id
            if then_y:
                await until(lambda: len(_sent_or_queued(console, session_id)) >= 2)
            sent = [DRAFT, "y"] if then_y else [DRAFT]
            assert _sent_or_queued(console, session_id) == sent
            assert composer.draft_text() == ("z" if then_y else "")
            assert not [n for n in notices if "still being sent" in n], notices


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", ["hi", "! pwd"], ids=["chat", "raw"])
async def test_enter_on_an_empty_composer_never_sends_text_typed_after_it(typed):
    """An empty capture has nothing to send (no image staged); also on dev.

    Enter on an empty composer captures nothing, and the send then read the
    live composer instead: text typed straight after that Enter was sent, and
    a ``! `` command typed after it was started.
    """
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            composer.load_draft("")
            await pilot.pause()
            started: list[object] = []
            console._raw_cli.start_user_command = lambda stash: started.append(stash)
            gateway.validation_release.set()
            press(host, "enter", "\r")
            for char in typed:
                key = {"!": "exclamation_mark", " ": "space"}.get(char, char)
                press(host, key, char)
            await until(lambda: composer.draft_text() == typed or bool(started))
            await _settled(console, pilot)
            session_id = console._console_chat_store.active_session_id
            assert [stash.text for stash in started] == []
            assert _sent_or_queued(console, session_id) == []
            assert gateway.stream_calls == 0
            assert composer.draft_text() == typed


@pytest.mark.asyncio
async def test_an_image_only_enter_never_takes_text_typed_after_it():
    """An empty capture with an image staged sends the image and no text.

    The send read the live composer for its draft, so text typed after the
    Enter rode along with the image. Read as an image-only draft, it was
    still captured from the live composer for the send to commit: accepted,
    that would take the later text out of the composer unsent.
    """
    from tldw_chatbook.UI.Console_Modules.prompt_queue import (
        ConsolePromptDispatchResult,
        ConsolePromptDispatchStatus,
    )

    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            composer.load_draft("")
            await pilot.pause()
            console._console_pending_image_attachment = lambda: object()
            dispatched: list[tuple[str, object]] = []

            async def hook_dispatch(draft, *, session_id, stash, dispatch):
                dispatched.append((draft, stash.text if stash else None))
                return ConsolePromptDispatchResult(
                    ConsolePromptDispatchStatus.SENT, session_id
                )

            console._hooks.dispatch = hook_dispatch
            press(host, "enter", "\r")
            press(host, "h", "h")
            press(host, "i", "i")
            await until(lambda: bool(dispatched) and composer.draft_text() == "hi")
            await _settled(console, pilot)
            assert dispatched == [("", None)]
            assert composer.draft_text() == "hi"


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["enter", "button", "workbench"])
@pytest.mark.parametrize("accepted", [True, False], ids=["accepted", "refused"])
async def test_a_raw_command_leaves_the_composer_once_whichever_send_starts_it(
    trigger, accepted
):
    """A trusted ``! `` command is consumed by every send, and restored once.

    Before TASK-33620.15 the Send button and the Workbench's send took the
    draft out of the composer (a destructive stash) before starting the
    command; Enter only captured it, so an accepted command stayed in the
    composer and a refused one was restored beside it ("! pwd! pwd", dev
    dce291d0fa). All three now capture first, so the captured draft is
    committed before the command starts.
    """
    from unittest.mock import AsyncMock, Mock

    from textual.widgets import Button

    from Tests.UI.test_console_command_composer import _type_raw_cli_prefix

    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer = await ready_console(host, pilot, gateway)
        composer.load_draft("")
        _type_raw_cli_prefix(composer)
        composer.insert_pasted_text("pwd")
        started = []
        real_start = console._raw_cli.start_user_command
        if accepted:
            console._raw_cli.start_user_command = Mock(
                side_effect=lambda stash: started.append(stash) or True
            )
        else:  # the real controller: raw CLI is not armed in this profile

            def start(stash):
                started.append(stash)
                return real_start(stash)

            console._raw_cli.start_user_command = start
        console._dispatch_console_draft_send = AsyncMock(return_value=True)
        if trigger == "enter":
            press(host, "enter", "\r")
        elif trigger == "button":
            console.query_one("#console-send-message", Button).press()
        else:
            console.post_message(WorkbenchActionRequested("send"))
        await until(lambda: bool(started))
        await pilot.pause(0.2)
        assert [stash.text for stash in started] == ["! pwd"]
        console._dispatch_console_draft_send.assert_not_awaited()
        assert composer.draft_text() == ("" if accepted else "! pwd")


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["enter", "button", "workbench"])
async def test_rewind_takes_its_command_out_of_the_composer_whichever_send_opens_it(
    trigger,
):
    """``/rewind`` leaves the composer once its menu opens, from every send.

    On dev only the Send button and the Workbench cleared it (they had no
    captured draft); Enter left "/rewind" in the composer behind the menu.
    All three now capture first, so the captured command is committed once
    the menu opens -- text typed after the capture stays. A click on Send
    leaves the slash-command popup open until then (Enter would accept its
    entry instead, so Enter's send dismisses it first); the commit must close
    it, as dev's clear did (live: the "/rewind" popup row stayed drawn under
    the menu).
    """
    from textual.widgets import Button

    from Tests.UI.app_factory import attach_chachanotes_db
    from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
    from Tests.UI.test_console_rewind_restore import _seed_u1_a1_u2_a2
    from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )
    from tldw_chatbook.Widgets.Console.console_rewind_modal import ConsoleRewindModal

    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _seed_u1_a1_u2_a2(console)
        composer = console.query_one("#console-native-composer")
        composer.focus()
        composer.load_draft("/rewind")
        console._sync_console_workbench_actions_from_draft()
        console._sync_console_command_popup()
        await pilot.pause()
        popup = console._console_command_popup_or_none()
        assert popup is not None and popup.is_open
        if trigger == "enter":
            console._dismiss_console_command_popup()
            await pilot.pause()
            press(host, "enter", "\r")
        elif trigger == "button":
            console.query_one("#console-send-message", Button).press()
        else:
            console.post_message(WorkbenchActionRequested("send"))
        await until(lambda: isinstance(host.screen_stack[-1], ConsoleRewindModal))
        await pilot.pause()
        assert composer.draft_text() == ""
        assert not popup.is_open
