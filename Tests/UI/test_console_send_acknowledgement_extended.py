"""Enter's "Sending…" acknowledgement: the mounted variants (TASK-33620.5).

The lean core in ``test_console_send_acknowledgement.py`` is in the UI PR
gate: the ordering test at 80x24 (the size where live polling caught the
tab dot missing) and the acknowledgement's own rules. These variants stay
out of the gate, as the lane's 60 s per-file rule asks: together with the
core they ran about 58-110 s serially on a local machine. Each mounted test
boots the real Console, so each one adds about 4-6 s.

They cover the ordering at the two wider sizes, the three ways a send can
fail after its row is painted (AC#3), a repeated Enter during admission
(AC#4), and the two PR #3022 review cases: a tab switch before the paint, and
a USER row that is not the send's own echo.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from Tests.UI.test_console_native_chat_flow import (
    _select_llamacpp_console,
    _wait_for_selector,
)
from Tests.UI.test_console_send_acknowledgement import (
    DRAFT,
    REFUSAL,
    REPLY,
    HeldAdmission,
    _painted_lines,
    build,
    check_sending_row_before_admission,
    eager_tasks,
    paint_state,
    press,
    ready_console,
    until,
)
from tldw_chatbook.UI.Console_Modules.prompt_queue import turn_recovery_label
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 45), (235, 52)])
async def test_enter_paints_a_sending_row_before_admission_at_wider_sizes(size):
    """AC#1/#5/#6 at the review's other sizes; the gated core runs 80x24."""
    await check_sending_row_before_admission(size)


@pytest.mark.asyncio
async def test_refused_admission_rolls_the_sending_row_back_and_keeps_the_draft():
    """AC#3: a runtime refusal removes the pending row and leaves the draft."""
    host, gateway, _timeline = build()
    notices: list[str] = []
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            host.app_instance.notify = lambda message, **_k: notices.append(
                str(message)
            )
            gate = HeldAdmission(console)
            runtime = console._console_runtime()

            def refuse(*_args, **_kwargs):
                raise RuntimeError("Console session is closed.")

            runtime.accept_turn = refuse
            try:
                press(host, "enter", "\r")
                await until(gate.entered.is_set)
                held = paint_state(host)
                assert held.user_row and held.sending, held
            finally:
                gate.release.set()
            await until(lambda: "Console session is closed." in notices)
            await until(lambda: not paint_state(host).sending)
            await pilot.pause()
            settled = paint_state(host)
            assert not settled.user_row and not settled.sending, settled
            assert settled.draft_in_composer, settled
            assert settled.header_idle and not settled.run_chip, settled
            assert composer.draft_text() == DRAFT
            assert gateway.stream_calls == 0


@pytest.mark.asyncio
async def test_turn_refused_before_its_echo_releases_the_row_and_offers_recovery():
    """AC#3: custody that ends before the store echo drops the pending row."""
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            controller = console._ensure_console_chat_controller()
            controller.hook_admission_reason = AsyncMock(return_value=REFUSAL)
            gate = HeldAdmission(console)
            try:
                press(host, "enter", "\r")
                await until(gate.entered.is_set)
                held = paint_state(host)
                assert held.user_row and held.sending, held
            finally:
                gate.release.set()
            await until(lambda: not console._console_runtime().has_custodied_turns())
            await until(lambda: not paint_state(host).sending)
            await pilot.pause()
            settled = paint_state(host)
            assert not settled.user_row and not settled.sending, settled
            assert not settled.run_chip and settled.header_idle, settled
            # Dev's TASK-33621.2 shelf states the refusal: "Not sent: <reason>".
            shelf = turn_recovery_label(REFUSAL)
            assert shelf.startswith("Not sent: Hook admission refused"), shelf
            assert shelf in "\n".join(_painted_lines(host))
            assert gateway.stream_calls == 0


@pytest.mark.asyncio
async def test_failed_durable_commit_releases_the_row_and_restores_send():
    """AC#3: a durable-commit failure leaves no "Sending…" and offers the turn."""
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            store = console._ensure_console_chat_store()

            def refuse_commit(*_args, **_kwargs):
                raise RuntimeError("durable commit refused for this test")

            store.commit_durable_turn = refuse_commit
            gate = HeldAdmission(console)
            try:
                press(host, "enter", "\r")
                await until(gate.entered.is_set)
                held = paint_state(host)
                assert held.user_row and held.sending, held
            finally:
                gate.release.set()
                # The durable commit comes after provider validation.
                gateway.validation_release.set()
            await until(lambda: not console._console_runtime().has_custodied_turns())
            await until(lambda: paint_state(host).header_idle)
            await pilot.pause()
            settled = paint_state(host)
            painted = "\n".join(_painted_lines(host))
            assert not settled.sending and not settled.run_chip, settled
            assert settled.header_idle, settled
            # Send is Send again, and the shelf offers the refused turn back
            # (measured: "Not sent: Couldn't save the prepared turn. Retry
            # or…" with Restore and Discard).
            assert "Sending" not in painted
            assert "Not sent: " in painted and "Restore" in painted, painted
            assert not console._console_send_ack.active_for(store.active_session_id)
            assert gateway.stream_calls == 0


@pytest.mark.asyncio
async def test_second_enter_during_admission_never_starts_a_second_provider_call():
    """AC#4: the first send is visibly in flight; a repeat is refused, not sent."""
    host, gateway, _timeline = build()
    notices: list[str] = []
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            host.app_instance.notify = lambda message, **_k: notices.append(
                str(message)
            )
            gate = HeldAdmission(console)
            try:
                press(host, "enter", "\r")
                await until(gate.entered.is_set)
                assert paint_state(host).user_row, paint_state(host)
                assert paint_state(host).sending, paint_state(host)
                # Queued behind the held send, like a key typed during the
                # real synchronous admission.
                press(host, "enter", "\r")
                await asyncio.sleep(0.2)
                gate.release.set()
                await until(gateway.validation_started.is_set)
                await until(lambda: composer.draft_text() == "")
                composer.load_draft(DRAFT)
                await pilot.pause()
                press(host, "enter", "\r")
                await until(lambda: len(notices) >= 1)
                await pilot.pause()
                in_flight = paint_state(host)
                assert in_flight.user_row and not in_flight.header_idle, in_flight
                assert in_flight.draft_in_composer, in_flight
            finally:
                gate.release.set()
                gateway.validation_release.set()
            await until(lambda: REPLY in "\n".join(_painted_lines(host)))
            await pilot.pause(0.3)
            assert gateway.stream_calls == 1
            assert composer.draft_text() == DRAFT


@pytest.mark.asyncio
async def test_a_tab_switch_before_the_paint_never_acknowledges_the_shown_tab(
    monkeypatch,
):
    """The paint is for the Enter's own tab, never whichever tab shows now.

    A tab press activates its session after an awaited read, so the switch
    can land after Enter captured its tab and before the deferred paint
    runs. That send is refused ("Console chat changed before send"); the
    tab now on screen must not read "Sending…" for it meanwhile (PR #3022
    review). Red before the fix: the paint pushed "Sending…" to the shared
    composer while the other tab was active.
    """
    from tldw_chatbook.UI.Console_Modules import send_acknowledgement as module

    host, gateway, _timeline = build()
    notices: list[str] = []
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-composer")
            _select_llamacpp_console(console)
            store = console._ensure_console_chat_store()
            sent_from = store.active_session_id
            await pilot.click("#console-new-chat-tab")
            await pilot.pause()
            shown = store.active_session_id
            assert shown not in (None, sent_from)
            await console._session._activate_native_console_session(sent_from)
            await pilot.pause(0.3)
            composer = console.query_one("#console-native-composer", ConsoleComposerBar)
            composer.focus()
            composer.load_draft(DRAFT)
            await pilot.pause()
            host.app_instance.notify = lambda message, **_k: notices.append(
                str(message)
            )
            acknowledged_on: list[str | None] = []
            real_show = ConsoleComposerBar.show_send_acknowledged

            def show(self, label: str, reason: str) -> None:
                acknowledged_on.append(store.active_session_id)
                real_show(self, label, reason)

            monkeypatch.setattr(ConsoleComposerBar, "show_send_acknowledged", show)
            real_call_later = console.call_later

            def call_later(callback, *args, **kwargs):
                if callback is not module._paint_then:
                    return real_call_later(callback, *args, **kwargs)

                async def switch_then_paint() -> None:
                    # The tab press's activation lands first.
                    await console._session._activate_native_console_session(shown)
                    await callback(*args, **kwargs)

                return real_call_later(switch_then_paint)

            monkeypatch.setattr(console, "call_later", call_later)
            press(host, "enter", "\r")
            await until(lambda: any("changed before send" in n for n in notices))
            await until(lambda: not console._console_send_ack.active_for(sent_from))
            await pilot.pause(0.3)
            assert store.active_session_id == shown
            assert acknowledged_on == [], acknowledged_on
            settled = paint_state(host)
            assert not settled.sending and not settled.user_row, settled
            assert gateway.stream_calls == 0


@pytest.mark.asyncio
async def test_a_user_row_that_is_not_the_echo_keeps_the_sending_row():
    """Only the Enter's own echo replaces its row (PR #3022 review).

    An Edit & resend lands a USER sibling in the same transcript. While the
    Enter still awaits admission that row cannot be its echo -- the echo is
    written only by the turn the runtime admits -- so "Sending…" stays
    until the Enter's own turn writes its echo. Red before the fix: any new
    USER id released the row, and the draft vanished from the transcript.
    """
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole as Role

    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            store = console._ensure_console_chat_store()
            session_id = store.active_session_id
            earlier = store.append_message(
                session_id, role=Role.USER, content="an earlier question"
            )
            store.append_message(
                session_id, role=Role.ASSISTANT, content="an earlier answer"
            )
            await console._sync_native_console_chat_ui()
            gate = HeldAdmission(console)
            try:
                press(host, "enter", "\r")
                await until(gate.entered.is_set)
                held = paint_state(host)
                assert held.user_row and held.sending, held
                sibling = store.create_sibling(
                    earlier.id, role=Role.USER, content="an edited question"
                )
                assert [
                    message.id
                    for message in store.messages_for_session(session_id)
                    if message.role is Role.USER
                ] == [sibling.id]
                # The whole sync projects the sibling (red: it released here).
                await console._sync_native_console_chat_ui()
                assert console._console_send_ack.active_for(session_id)
                # The held send keeps the app busy, so Pilot never sees idle.
                await asyncio.sleep(0.2)
                held = paint_state(host)
                assert held.user_row and held.sending, held
            finally:
                gate.release.set()
                gateway.validation_release.set()
            await until(lambda: not console._console_send_ack.active_for(session_id))
            await until(lambda: not paint_state(host).sending)
            await pilot.pause()
            settled = paint_state(host)
            assert settled.user_row and not settled.sending, settled
