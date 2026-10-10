"""A prompt typed right after a session-tab click goes to the clicked chat.

TASK-33622.7 (review finding GAP1-01). Live on dev 64579cce2c (235x52, slow
proxy): with a run live in tab B, a click on tab A, typing and Enter 1.5-2.2 s
later queued the prompt into B, 3 runs of 3. Once the screen already showed A
when Enter landed, and B's leftover draft was merged with the new text and sent
to B as one turn. Idle, text typed 0.4 s after a tab click was lost from both
composers, and its "y" opened Trace.

The cause: ``_activate_native_console_session`` left the composer to the
coalescable console-sync pass. While a pass is already running (routine during
a live run) the activation only sets ``_console_sync_requested``. The running
pass had bound the draft to the old chat before its first await, and goes on to
paint the new chat's tabs and transcript. Enter then pins the send to the stale
visible-draft session id.

These tests drive the real ChatScreen and session store. They hold a REAL sync
pass in flight at its first await (the effective-scope warm, which comes after
the pass's draft sync), or hold the activation itself at the scope refresh that
follows the switch, and deliver the mouse click and keys through the driver.
The send is recorded at ``_dispatch_console_draft_send``: the session it is
handed there is the chat the turn would be sent or queued to.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field

import pytest
from textual import events
from textual.pilot import _get_mouse_message_arguments
from textual.widgets import Button

from Tests.UI.test_console_native_chat_flow import _select_llamacpp_console
from Tests.UI.test_console_send_acknowledgement import press, until
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

pytestmark = pytest.mark.bootstrap_profile

B_LEFTOVER = "Reply with the single word ECHO."
A_OWN = "a-own:"
ACTIVE_TAB = "console-session-tab-active"


@dataclass
class Gate:
    """One held await: ``entered`` when reached, waits for ``release``."""

    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)


_GATES: list[Gate] = []


@contextlib.asynccontextmanager
async def _running(host):
    """Run the Console; release every held await before the app shuts down.

    A red assertion inside a held activation would otherwise leave the
    screen's pump awaiting forever and hang the app's teardown.
    """
    async with host.run_test(size=(160, 48)) as pilot:
        try:
            yield pilot
        finally:
            for gate in _GATES:
                gate.release.set()
            _GATES.clear()


@dataclass
class Chats:
    """The mounted Console with two chats: B active, A in the background."""

    console: object
    composer: ConsoleComposerBar
    store: object
    a: object
    b: object


def _host() -> ConsoleHarness:
    app = _build_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    return ConsoleHarness(app)


async def _two_chats(host, pilot) -> Chats:
    """Mount the Console with B active (holding a leftover draft) and A stored."""
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-native-composer")
    _select_llamacpp_console(console)
    store = console._ensure_console_chat_store()
    b = next(item for item in store.sessions() if item.id == store.active_session_id)
    a = store.create_session(title="Chat A", settings=b.settings, activate=False)
    store.set_session_draft(a.id, A_OWN)
    composer = console.query_one("#console-native-composer", ConsoleComposerBar)
    composer.focus()
    composer.load_draft(B_LEFTOVER)
    await console._sync_native_console_chat_ui()
    await _wait_for_selector(console, pilot, f"#console-session-tab-{a.id}")
    await pilot.pause()
    assert console._console_visible_draft_session_id == b.id
    assert store.session_draft(b.id) == B_LEFTOVER
    assert _active_tabs(console) == [b.id]
    return Chats(console, composer, store, a, b)


def _active_tabs(console) -> list[str]:
    """Session ids whose tab carries the active highlight."""
    return [
        str(tab.id).removeprefix("console-session-tab-")
        for tab in console.query(f".{ACTIVE_TAB}")
        if str(tab.id).startswith("console-session-tab-")
    ]


def _click(host, widget) -> None:
    """Left-click ``widget`` the way the terminal driver does (no Pilot wait)."""
    arguments = _get_mouse_message_arguments(widget, (2, 0), button=1)
    for event_type in (events.MouseDown, events.MouseUp):
        event = event_type(**arguments)
        event.set_sender(host)
        host._driver.send_message(event)


def _record_dispatch(console) -> list[tuple[str | None, str]]:
    """Record the chat each send is dispatched to, instead of sending it."""
    dispatched: list[tuple[str | None, str]] = []

    async def dispatch(draft, stash=None, *, session_id=None):
        dispatched.append((session_id, draft))
        return True

    console._dispatch_console_draft_send = dispatch
    return dispatched


def _record_notices(console) -> list[str]:
    notices: list[str] = []
    real = console.app_instance.notify

    def notify(message, *args, **kwargs):
        notices.append(str(message))
        return real(message, *args, **kwargs)

    console.app_instance.notify = notify
    return notices


def _refused(notices: list[str]) -> bool:
    """Whether the send guard refused a send (TASK-33622.7 AC#3)."""
    return any("nothing was sent" in notice.lower() for notice in notices)


def _record_trace_opens(console) -> list[str]:
    opened: list[str] = []
    console._review_selection.open_trajectory_view = lambda: opened.append("trace")
    return opened


def _hold_sync_pass(console) -> Gate:
    """Hold the next console-sync pass at its first await, after its draft sync."""
    gate = Gate()
    _GATES.append(gate)
    retrieval = console._retrieval
    real = retrieval._warm_console_effective_scope_cache_if_stale

    async def held() -> None:
        if not gate.entered.is_set():
            gate.entered.set()
            await gate.release.wait()
        await real()

    retrieval._warm_console_effective_scope_cache_if_stale = held
    return gate


def _hold_activation(console) -> Gate:
    """Hold the next session activation at its scope refresh, after the switch."""
    gate = Gate()
    _GATES.append(gate)
    session = console._session
    real = session._refresh_effective_scope_and_sync_fn

    async def held(*args, **kwargs) -> None:
        if not gate.entered.is_set():
            gate.entered.set()
            await gate.release.wait()
        await real(*args, **kwargs)

    session._refresh_effective_scope_and_sync_fn = held
    return gate


def _named(pairs, a, b) -> list[tuple[str, str]]:
    names = {a.id: "A", b.id: "B"}
    return [
        (names.get(session_id, str(session_id)), text) for session_id, text in pairs
    ]


def _facts(console, composer, a, b) -> dict[str, object]:
    """What the user sees and what Enter would be pinned to, by chat name."""
    names = {a.id: "A", b.id: "B"}
    owner = console._console_visible_draft_session_id
    return {
        "draft owner": names.get(owner, owner),
        "highlighted tabs": [names.get(item, item) for item in _active_tabs(console)],
        "composer": composer.draft_text(),
    }


async def _sync_idle(console) -> None:
    await until(
        lambda: (
            not console._console_sync_in_progress
            and not console._console_sync_requested
        )
    )


@pytest.mark.asyncio
async def test_enter_after_a_tab_click_during_a_sync_pass_sends_to_the_clicked_chat():
    """AC#2/#4/#6: the reviewer's sequence with a sync pass held in flight."""
    host = _host()
    async with _running(host) as pilot:
        chats = await _two_chats(host, pilot)
        console, composer, store, a, b = (
            chats.console,
            chats.composer,
            chats.store,
            chats.a,
            chats.b,
        )
        dispatched = _record_dispatch(console)
        notices = _record_notices(console)
        held_pass = _hold_sync_pass(console)
        console.run_worker(
            console._sync_native_console_chat_ui(),
            exclusive=True,
            group="console-sync",
        )
        await until(held_pass.entered.is_set)
        assert console._console_sync_in_progress

        _click(host, console.query_one(f"#console-session-tab-{a.id}"))
        await until(lambda: store.active_session_id == a.id)
        await pilot.pause()
        assert console._console_sync_in_progress, "the held pass must still run"

        # The click is answered while the pass is still in flight.
        seen = _facts(console, composer, a, b)
        await pilot.press("x", "enter")
        await until(lambda: bool(dispatched) or _refused(notices))
        seen["dispatched"] = _named(dispatched, a, b)
        seen["refused"] = _refused(notices)
        assert seen == {
            "draft owner": "A",
            "highlighted tabs": ["A"],
            "composer": A_OWN,
            "dispatched": [("A", A_OWN + "x")],
            "refused": False,
        }

        held_pass.release.set()
        await _sync_idle(console)
        await pilot.pause()
        assert store.session_draft(b.id) == B_LEFTOVER
        assert console._console_visible_draft_session_id == a.id
        assert _active_tabs(console) == [a.id]


@pytest.mark.asyncio
async def test_tab_click_moves_highlight_draft_and_focus_before_its_first_await():
    """AC#1/#4: the clicked tab and its own draft show in one step."""
    host = _host()
    async with _running(host) as pilot:
        chats = await _two_chats(host, pilot)
        console, composer, store, a, b = (
            chats.console,
            chats.composer,
            chats.store,
            chats.a,
            chats.b,
        )
        held = _hold_activation(console)
        _click(host, console.query_one(f"#console-session-tab-{a.id}"))
        await until(held.entered.is_set)

        assert store.active_session_id == a.id
        seen = _facts(console, composer, a, b)
        seen["B stored draft"] = store.session_draft(b.id)
        seen["composer focused"] = console.app.focused is composer
        assert seen == {
            "draft owner": "A",
            "highlighted tabs": ["A"],
            "composer": A_OWN,
            "B stored draft": B_LEFTOVER,
            "composer focused": True,
        }

        held.release.set()
        await _sync_idle(console)
        await pilot.pause()
        assert composer.draft_text() == A_OWN
        assert store.session_draft(b.id) == B_LEFTOVER


@pytest.mark.asyncio
async def test_keys_typed_during_a_held_tab_activation_reach_the_new_chat_composer():
    """AC#5: a "y" typed straight after the click is text, not Trace."""
    host = _host()
    async with _running(host) as pilot:
        chats = await _two_chats(host, pilot)
        console, composer, store, a, b = (
            chats.console,
            chats.composer,
            chats.store,
            chats.a,
            chats.b,
        )
        traces = _record_trace_opens(console)
        held = _hold_activation(console)
        _click(host, console.query_one(f"#console-session-tab-{a.id}"))
        await until(held.entered.is_set)
        for key in ("y", "e", "s"):
            press(host, key, key)
        await asyncio.sleep(0.2)

        held.release.set()
        await _sync_idle(console)
        await pilot.pause()
        assert store.active_session_id == a.id
        assert {
            "trace opened": traces,
            "composer": composer.draft_text(),
            "B stored draft": store.session_draft(b.id),
        } == {
            "trace opened": [],
            "composer": A_OWN + "yes",
            "B stored draft": B_LEFTOVER,
        }


@pytest.mark.asyncio
async def test_pressing_a_session_tab_leaves_the_keyboard_in_the_composer():
    """AC#5: the mouse-down on a tab, before any activation runs, keeps focus.

    Live, a busy screen can take a moment to handle the click. On dev the
    mouse-down alone moved focus to the tab button, so a "y" typed then
    opened Trace and the other letters were lost.
    """
    host = _host()
    async with _running(host) as pilot:
        chats = await _two_chats(host, pilot)
        console, composer, store, a, b = (
            chats.console,
            chats.composer,
            chats.store,
            chats.a,
            chats.b,
        )
        traces = _record_trace_opens(console)
        tab = console.query_one(f"#console-session-tab-{a.id}")
        event = events.MouseDown(**_get_mouse_message_arguments(tab, (2, 0), button=1))
        event.set_sender(host)
        host._driver.send_message(event)
        await pilot.pause()
        focused = console.app.focused
        press(host, "y", "y")
        await pilot.pause()
        assert {
            "focused": getattr(focused, "id", focused),
            "trace opened": traces,
            "composer": composer.draft_text(),
            "active": store.active_session_id == b.id,
        } == {
            "focused": "console-native-composer",
            "trace opened": [],
            "composer": B_LEFTOVER + "y",
            "active": True,
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("send", ["enter", "send_button"])
async def test_send_refuses_and_keeps_the_draft_when_the_composer_is_bound_elsewhere(
    send,
):
    """AC#3: a composer still bound to another chat never sends there.

    The reviewer's screen: a switch path that leaves the rebind to the sync
    pass moves the store to A while a pass is in flight. That pass synced the
    draft before the switch, then paints A's tab and transcript, so A is on
    screen while the composer still holds B's draft.

    ``send_button`` is the call the Send button and the spoken "Console,
    send." both make (``handle_console_send_message``).
    """
    host = _host()
    async with _running(host) as pilot:
        chats = await _two_chats(host, pilot)
        console, composer, store, a, b = (
            chats.console,
            chats.composer,
            chats.store,
            chats.a,
            chats.b,
        )
        dispatched = _record_dispatch(console)
        notices = _record_notices(console)
        held_pass = _hold_sync_pass(console)
        console.run_worker(
            console._sync_native_console_chat_ui(),
            exclusive=True,
            group="console-sync",
        )
        await until(held_pass.entered.is_set)
        store.switch_session(a.id)
        held_pass.release.set()
        await _sync_idle(console)
        await pilot.pause()
        assert {
            "draft owner": console._console_visible_draft_session_id == b.id,
            "highlighted tabs": _active_tabs(console),
            "transcript": console._last_native_transcript_session_id == a.id,
            "composer": composer.draft_text(),
        } == {
            "draft owner": True,
            "highlighted tabs": [a.id],
            "transcript": True,
            "composer": B_LEFTOVER,
        }

        if send == "enter":
            await pilot.press("enter")
        else:
            button = console.query_one("#console-send-message", Button)
            await console.handle_console_send_message(Button.Pressed(button))
        await until(lambda: bool(dispatched) or _refused(notices))
        assert {
            "dispatched": _named(dispatched, a, b),
            "composer": composer.draft_text(),
            "refused": _refused(notices),
        } == {"dispatched": [], "composer": B_LEFTOVER, "refused": True}

        # The next pass binds the composer to A; B keeps its own draft.
        await console._sync_native_console_chat_ui()
        await _sync_idle(console)
        assert {
            "dispatched": dispatched,
            "composer": composer.draft_text(),
            "B stored draft": store.session_draft(b.id),
        } == {"dispatched": [], "composer": A_OWN, "B stored draft": B_LEFTOVER}
