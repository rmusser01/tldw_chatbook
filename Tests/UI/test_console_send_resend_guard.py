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

Every test drives the real ChatScreen send path (driver-delivered keys, the
app's eager task factory, a real in-memory store and runtime).
"""

from __future__ import annotations

import asyncio

import pytest

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
    HeldMcpRead,
    _allow_second_turns,
    _first_turn_done,
    _hold_first_send_tail,
    _record_notices,
    _second_tab,
    _sent_or_queued,
    _settled,
)

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
async def test_a_draft_sent_after_a_non_append_edit_says_it_was_sent():
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
            hold = HeldMcpRead(host.app_instance.unified_mcp_service)
            gateway.validation_release.set()
            try:
                press(host, "enter", "\r")
                await until(hold.entered.is_set, timeout=ENTRY_SECONDS)
                press(host, "home")
                press(host, "x", "x")
                await until(lambda: composer.draft_text() == "x" + DRAFT, timeout=10)
            finally:
                hold.release.set()
            await until(lambda: gateway.stream_calls == 1)
            await _idle(host, console, pilot, session_id)
            assert _sent_or_queued(console, session_id) == [DRAFT]
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
            hold = HeldMcpRead(host.app_instance.unified_mcp_service)
            gateway.validation_release.set()
            try:
                press(host, "enter", "\r")
                await until(hold.entered.is_set, timeout=ENTRY_SECONDS)
                press(host, "x", "x")
                await until(lambda: composer.draft_text() == DRAFT + "x", timeout=10)
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
            await until(lambda: console._console_visible_draft_session_id == session_a)
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
    spoken send commits it while Enter's "Sending…" is being painted; the
    Enter's send then started with its capture and sent "y" again.
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
            hold = HeldMcpRead(host.app_instance.unified_mcp_service)
            gateway.validation_release.set()
            paint_release = asyncio.Event()
            real_paint = send_acknowledgement.paint_acknowledgement

            async def held_paint(screen, painted_session_id):
                await paint_release.wait()
                await real_paint(screen, painted_session_id)

            send_acknowledgement.paint_acknowledgement = held_paint
            try:
                _speak_send(console, session_id)
                await until(hold.entered.is_set, timeout=ENTRY_SECONDS)
                press(host, "enter", "\r")
                await until(lambda: console._console_pending_send is not None)
                hold.release.set()
                await until(lambda: _sent_or_queued(console, session_id) == ["y"])
                await until(lambda: composer.draft_text() == "")
            finally:
                hold.release.set()
                paint_release.set()
                send_acknowledgement.paint_acknowledgement = real_paint
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

    With nothing typed since the first capture the reloaded draft is that
    capture, and it leaves the composer. With a key typed in tab B the
    composer cannot tell the reload from a new draft: the sent text stays,
    the user is told it was sent, and the held press, captured before that,
    is not sent.
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
            hold = HeldMcpRead(host.app_instance.unified_mcp_service)
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
            told = [n for n in notices if "was sent" in n]
            if typed_in_b:
                assert composer.draft_text() == DRAFT
                assert told, notices
                assert store.session_draft(session_b) == B_DRAFT + "q"
            else:
                assert composer.draft_text() == ""
                assert not told, notices
                assert store.session_draft(session_b) == B_DRAFT


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


@pytest.mark.asyncio
async def test_a_send_refused_while_another_tab_sends_names_that_tab():
    """AC#3: the Send gate's "already in progress" names the busy tab.

    A spoken send in tab A is being admitted (it holds the Send gate), then
    Alt+2 and Enter in tab B: B's send was refused with "Hook review or Send
    is already in progress." although nothing was in progress in B.
    """
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            session_a, session_b = await _second_tab(console, pilot, ready=True)
            store = console._ensure_console_chat_store()
            store.rename_session(session_a, "Alpha plans")
            notices = _record_notices(host)
            hold = HeldMcpRead(host.app_instance.unified_mcp_service)
            gateway.validation_release.set()
            try:
                _speak_send(console, session_a)
                await until(hold.entered.is_set, timeout=ENTRY_SECONDS)
                press(host, "alt+2")
                await until(
                    lambda: console._console_visible_draft_session_id == session_b
                )
                await until(lambda: composer.draft_text() == B_DRAFT)
                press(host, "enter", "\r")
                await until(lambda: any("already in progress" in n for n in notices))
            finally:
                hold.release.set()
            await until(lambda: gateway.stream_calls == 1)
            await _idle(host, console, pilot, session_a)
            refusals = [n for n in notices if "already in progress" in n]
            assert refusals and all("Alpha plans" in n for n in refusals), refusals
            assert _sent_or_queued(console, session_a) == [DRAFT]
            assert _sent_or_queued(console, session_b) == []
            assert composer.draft_text() == B_DRAFT


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
            console._console_send_blocked_reason = lambda: BLOCKED
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
