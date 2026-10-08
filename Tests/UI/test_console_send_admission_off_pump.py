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
