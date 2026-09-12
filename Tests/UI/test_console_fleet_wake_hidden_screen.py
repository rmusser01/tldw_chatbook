"""Wake-integrity arc (tasks 15970 + 15971): the mounted-but-hidden Console.

The PR3a-2 residue arc's live pass found two failures with instrumented
evidence (ledger ``residue-frames/dbg.log``): a wake fired straight
through a draft the user was HOLDING in the visible composer (probe read
``draft=''`` while the pane showed text, twice), and a wake DELIVERED
while the Library screen was displayed, with no surviving unseen mark.

This arc's diagnosis, reproduced here through the real production paths
(no ``load_draft``, no hand-built screens): the app's navigation builds a
FRESH screen per route and ``switch_screen`` pops the TOP of the screen
stack -- so a navigation issued while any pushed screen (the nav overflow
menu, a picker, a rename modal) sits above the Chat screen pops the MODAL
and leaves the Chat screen alive in the stack: mounted, pump running,
controller un-shut-down, wake coordinator armed. That resident hidden
screen is what the residue arc's ``mounted=True`` instrumentation saw.
Both live failures follow:

**Correction, 2026-08-14 (task-16300).** That navigation leak is FIXED:
`_handle_screen_navigation_locked` now reduces the stack to its content
screen before switching, so the outgoing Console screen is the thing
``switch_screen`` replaces and unmounts, and its controller shuts down
(pinned by ``Tests/UI/test_screen_residency.py``). The behaviours below
are unchanged and still required -- Console is mounted-but-not-DISPLAYED
whenever any pushed screen covers it, which is the state the 15971 live
pass actually verified -- but their setups no longer come from the leak:
the covered-Console cases push a real modal over Console, and the
two-Console case (a hidden Console while a DIFFERENT Console is
displayed, which only the leak used to produce) is built directly and
kept as defence in depth for the probe's cross-screen resolution.

- **task-15970**: the hidden screen's user-wins-ties probe read ITS OWN
  (empty) composer while the user typed into the DISPLAYED screen's --
  the "blindness" was a stale screen, not segment plumbing (the suspected
  PR #1554-era composer path verified correct under real key input here).
- **task-15971**: the hidden screen's controller never shuts down, so its
  coordinator delivers off-screen -- ruled INTENDED by the coordinator's
  design ruling (the supervisor acts immediately; that is the auto-wake
  invariant) -- but the user must still LEARN of it: the hidden screen's
  own 1s sync tick was "view"-clearing the FLEET_UNSEEN mark while the
  user was on Library, which is how the live run ended with no badge.

Pinned here: the probe reads the composer the user can actually type
into; a mounted-but-undisplayed screen's sync never view-clears the mark;
the displayed screen still clears it (Task 4 semantics preserved); and
the screen wires the coordinator's conversation-in-view probe.
"""

from __future__ import annotations

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from tldw_chatbook.Chat.console_fleet_attention import (
    bump_fleet_unseen_revision,
)
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.UI.Navigation.nav_overflow_menu import NavOverflowMenu
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


async def _wait_for_composer(screen, pilot) -> None:
    from Tests.UI.test_destination_shells import _wait_for_selector

    await _wait_for_selector(screen, pilot, "#console-native-composer")


async def _mount_chat(app, pilot) -> ChatScreen:
    """Push the initial Chat screen the way startup does."""
    await app.push_screen(ChatScreen(app))
    app._initial_screen_pushed = True
    await pilot.pause()
    chat = app.screen
    assert isinstance(chat, ChatScreen)
    await _wait_for_composer(chat, pilot)
    return chat


async def _hide_chat_under_a_modal(app, pilot) -> ChatScreen:
    """Mount Console and cover it with a real pushed screen.

    Correction 2026-08-14 (task-16300): this used to be
    ``_leak_resident_chat``, which built the hidden state by NAVIGATING
    with the nav overflow menu on top and asserting -- as a harness
    precondition -- that Chat stayed resident in the stack afterwards.
    That leak is fixed: navigation now reduces the stack to its content
    screen first, so the outgoing Console screen is the thing
    ``switch_screen`` replaces and unmounts
    (``Tests/UI/test_screen_residency.py``).

    Mounted-but-not-DISPLAYED Console is still an ordinary state, and it
    is the one the 15971 live pass actually verified: any pushed screen
    (the command palette, the nav overflow menu, a picker) covers Console
    while Console keeps running underneath. That is what this builds now
    -- no navigation, no leak.
    """
    chat = await _mount_chat(app, pilot)
    app.push_screen(NavOverflowMenu())
    await pilot.pause()
    assert app.screen is not chat, "the modal must be the displayed screen"
    assert chat in app.screen_stack
    assert chat.is_running and chat._console_chat_controller is not None
    assert not chat._console_chat_controller._shutdown_requested.is_set()
    return chat


async def _second_console_over_chat(app, pilot) -> tuple[ChatScreen, ChatScreen]:
    """Build the two-Console geometry the 15970 probe fix guards against.

    Honest note (task-16300, 2026-08-14): this state used to arise from
    production navigation -- navigating away under a pushed screen left
    the old ChatScreen resident, and navigating back built a SECOND live
    one on top of it. That is exactly the live 15970 failure, and it is
    now unreachable through navigation. The probe's cross-screen
    resolution (``wiring._displayed_console_composer_draft``) survives as
    defence in depth for any Console screen that is mounted while a
    DIFFERENT Console screen is displayed, so the geometry is constructed
    here directly through ``push_screen`` rather than through a bug.

    Returns:
        ``(hidden, displayed)`` -- both live Console screens.
    """
    hidden = await _mount_chat(app, pilot)
    displayed = ChatScreen(app)
    await app.push_screen(displayed)
    await pilot.pause()
    await _wait_for_composer(displayed, pilot)
    assert app.screen is displayed and displayed is not hidden
    assert hidden in app.screen_stack and hidden.is_running
    return hidden, displayed


def _build_app(tmp_path):
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    return app


@pytest.mark.asyncio
async def test_probe_sees_a_draft_typed_with_real_keys(tmp_path):
    """task-15970 AC#2 groundwork: real key input through the production
    ``on_key`` path (NOT ``load_draft`` -- the existing wiring test's
    harness shortcut is exactly what let the live blindness pass) reaches
    the probe on the DISPLAYED screen."""
    app = _build_app(tmp_path)
    async with app.run_test(size=(160, 48)) as pilot:
        chat = await _mount_chat(app, pilot)
        session_id = chat._ensure_console_chat_store().ensure_session().id
        controller = chat._ensure_console_chat_controller()
        probe = controller.wake_user_priority_probe
        assert probe(session_id) is False, "an empty composer holds no claim"
        await pilot.press(*"drafting")
        await pilot.pause()
        assert probe(session_id) is True, (
            "a draft typed with real keys is the user's sending claim"
        )


@pytest.mark.asyncio
async def test_hidden_screens_probe_sees_the_displayed_screens_typed_draft(
    tmp_path,
):
    """task-15970, the live shape: a mounted Chat screen's coordinator
    consults the probe while the user types into the DISPLAYED Chat
    screen's composer. RED when written: the probe read its own (empty)
    composer -- ``probe: composer=True draft=''`` while the pane visibly
    held the text -- and the wake fired through the held draft.

    task-16300 note: the two-Console geometry is now built directly
    (see ``_second_console_over_chat``) instead of through the navigation
    leak that used to produce it live."""
    app = _build_app(tmp_path)
    async with app.run_test(size=(160, 48)) as pilot:
        hidden, displayed = await _second_console_over_chat(app, pilot)

        await pilot.press(*"drafting")
        await pilot.pause()
        displayed_composer = displayed._console_composer_or_none()
        assert displayed_composer is not None
        assert displayed_composer.draft_text() == "drafting", (
            "harness precondition: the typed keys must land in the "
            "displayed composer through the production key path"
        )
        hidden_session_id = hidden._ensure_console_chat_store().ensure_session().id
        hidden_probe = hidden._console_chat_controller.wake_user_priority_probe
        assert hidden_probe(hidden_session_id) is True, (
            "the user-wins-ties probe must see the draft the user is "
            "actually holding -- the hidden screen's coordinator firing "
            "through the displayed composer's text is the live 15970 bug"
        )


@pytest.mark.asyncio
async def test_typed_draft_defers_the_hidden_coordinators_due_wake(tmp_path):
    """task-15970 AC#1, outcome level: a due wake on a hidden Console
    screen's coordinator DEFERS (no delivery scheduled, pending intact)
    while a real-keys draft is held in the displayed composer."""
    app = _build_app(tmp_path)
    async with app.run_test(size=(160, 48)) as pilot:
        hidden, displayed = await _second_console_over_chat(app, pilot)
        hidden_controller = hidden._console_chat_controller
        wake = hidden_controller.fleet_wake
        hidden_session = hidden._ensure_console_chat_store().ensure_session()

        await pilot.press(*"drafting my next thought".replace(" ", "_"))
        await pilot.pause()

        with wake._registry_lock:
            wake._pending[hidden_session.id] = {"r-held": "done"}
        wake._attempt(hidden_session.id)
        assert wake.delivering_conversation_id() is None, (
            "a wake must defer while the user holds a typed draft -- "
            "delivering here is the live 'wake fired straight through a "
            "held draft' failure"
        )
        assert not wake._delivery_tasks
        assert wake.has_pending(hidden_session.id), (
            "deferral must leave the pending bit untouched"
        )


@pytest.mark.asyncio
async def test_hidden_screen_sync_never_view_clears_the_unseen_mark(tmp_path):
    """task-15971 AC#2: 'viewing IS the clear' requires VIEWING. A
    Console screen covered by a pushed screen keeps its 1s sync tick
    running (the dbg log's continuous sync-run beats) and used to consume
    the FLEET_UNSEEN mark -- which is why the live off-screen delivery
    left the user nothing. A mounted-but-undisplayed Console must not
    count as 'in Console'."""
    app = _build_app(tmp_path)
    async with app.run_test(size=(160, 48)) as pilot:
        hidden = await _hide_chat_under_a_modal(app, pilot)
        marks = app.conversation_local_marks_service
        session = hidden._ensure_console_chat_store().ensure_session()
        marks.set_mark(session.id, ConversationLocalMarksService.FLEET_UNSEEN)
        bump_fleet_unseen_revision(app)

        await hidden._sync_console_native_session_tabs()
        await pilot.pause()
        assert marks.has_mark(session.id, ConversationLocalMarksService.FLEET_UNSEEN), (
            "a hidden resident screen's sync tick must not view-clear the "
            "unseen mark while the user is on another screen -- this is "
            "how the live run ended with no badge after the off-screen "
            "delivery"
        )


@pytest.mark.asyncio
async def test_displayed_screen_sync_still_view_clears_the_mark(tmp_path):
    """task-15971 AC#3, the preserved side: on the DISPLAYED Console the
    active conversation's mark still clears on the sync tick (Task 4's
    viewing-is-the-clear, unchanged for a delivered wake)."""
    app = _build_app(tmp_path)
    async with app.run_test(size=(160, 48)) as pilot:
        chat = await _mount_chat(app, pilot)
        marks = app.conversation_local_marks_service
        session = chat._ensure_console_chat_store().ensure_session()
        chat._ensure_console_chat_controller()
        marks.set_mark(session.id, ConversationLocalMarksService.FLEET_UNSEEN)
        bump_fleet_unseen_revision(app)

        await chat._sync_console_native_session_tabs()
        await pilot.pause()
        assert not marks.has_mark(
            session.id, ConversationLocalMarksService.FLEET_UNSEEN
        ), "viewing the conversation on the displayed screen IS the clear"


@pytest.mark.asyncio
async def test_screen_wires_the_conversation_in_view_probe(tmp_path):
    """task-15971 AC#1 wiring: the coordinator's delivery commit consults
    a screen-wired conversation-in-view probe; True only when this screen
    is the DISPLAYED one and the session is the active one."""
    app = _build_app(tmp_path)
    async with app.run_test(size=(160, 48)) as pilot:
        chat = await _mount_chat(app, pilot)
        controller = chat._ensure_console_chat_controller()
        session = chat._ensure_console_chat_store().ensure_session()
        probe = getattr(controller, "wake_conversation_in_view", None)
        assert callable(probe), (
            "the screen must wire the conversation-in-view probe the "
            "delivery commit consults"
        )
        assert probe(session.id, session.id) is True
        assert probe(session.id, "some-other-session") is False

        # Cover Console with a real pushed screen: mounted, not displayed.
        app.push_screen(NavOverflowMenu())
        await pilot.pause()
        assert chat in app.screen_stack and app.screen is not chat
        assert probe(session.id, session.id) is False, (
            "a mounted-but-undisplayed screen's conversation is not in view"
        )
