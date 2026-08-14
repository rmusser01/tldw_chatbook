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
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
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


async def _leak_resident_chat(app, pilot) -> ChatScreen:
    """Drive the real navigation path that leaves Chat resident-but-hidden.

    The nav overflow menu (a real pushed screen, reachable via its "More"
    affordance) is above Chat when the navigation runs; ``switch_screen``
    pops the MENU, and the Chat screen stays alive in the stack below the
    incoming Library screen.
    """
    chat = await _mount_chat(app, pilot)
    app.push_screen(NavOverflowMenu())
    await pilot.pause()
    await app.handle_screen_navigation(NavigateToScreen("library"))
    await pilot.pause()
    assert type(app.screen).__name__ == "LibraryScreen"
    assert chat in app.screen_stack, (
        "harness precondition: the nav-under-a-pushed-screen path must "
        "leave the Chat screen resident in the stack (the residue arc's "
        "live mounted=True state); if this stops holding, the hidden-"
        "screen scenario below needs a new construction"
    )
    assert chat.is_running and chat._console_chat_controller is not None
    assert not chat._console_chat_controller._shutdown_requested.is_set()
    return chat


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
async def test_hidden_screen_sync_never_view_clears_the_unseen_mark(tmp_path):
    """task-15971 AC#2: 'viewing IS the clear' requires VIEWING. The
    resident hidden screen's own sync tick kept running during Library
    display (the dbg log's continuous sync-run beats) and consumed the
    FLEET_UNSEEN mark -- which is why the live off-screen delivery left
    the user nothing. A mounted-but-undisplayed Console must not count as
    'in Console'."""
    app = _build_app(tmp_path)
    async with app.run_test(size=(160, 48)) as pilot:
        hidden = await _leak_resident_chat(app, pilot)
        marks = app.conversation_local_marks_service
        session = hidden._ensure_console_chat_store().ensure_session()
        marks.set_mark(session.id, ConversationLocalMarksService.FLEET_UNSEEN)
        bump_fleet_unseen_revision(app)

        await hidden._sync_console_native_session_tabs()
        await pilot.pause()
        assert marks.has_mark(
            session.id, ConversationLocalMarksService.FLEET_UNSEEN
        ), (
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

        # Hide the screen through the real leak path: not displayed any more.
        app.push_screen(NavOverflowMenu())
        await pilot.pause()
        await app.handle_screen_navigation(NavigateToScreen("library"))
        await pilot.pause()
        assert chat in app.screen_stack
        assert probe(session.id, session.id) is False, (
            "a mounted-but-undisplayed screen's conversation is not in view"
        )
