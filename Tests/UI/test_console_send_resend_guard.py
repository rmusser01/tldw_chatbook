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

import pytest

from Tests.UI.test_console_send_acknowledgement import (
    DRAFT,
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
