"""TASK-34100.5 AC#8 (entry-exit-handoff-05, task-33620.6).

'A Console turn completed while hidden. Return to Console to review.' fired
while the user was looking at Console: on the first reply after Start
chatting, after Explore Home -> Console, and on back-to-back turns in one tab.
The notice and the nav '!' badge must be decided when the turn completes,
from (Console is the current screen) AND (the turn's conversation is the
active tab). These tests reach Console through both first-run exits, run real
turns through the runtime's production admission path, and keep a hidden
control that must still notify exactly once.
"""

from __future__ import annotations

import asyncio

import pytest

from Tests.UI.test_console_store_continuity import _navigate
from Tests.UI.test_console_turn_navigation_continuity import _build_navigation_app
from Tests.UI.test_destination_shells import _wait_for_selector
from textual.screen import ModalScreen
from textual.widgets import Static

from tldw_chatbook.Constants import TAB_CHAT, TAB_HOME
from tldw_chatbook.UI.Console_Modules.wiring import _admit_console_turn_to_runtime
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    ConsoleFirstChatIntent,
    HandoffChannel,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import SetupWizardContainer

_HIDDEN_COPY = "completed while hidden"

# Mounted-app tests that build app config in their body keep the
# collection-time profile (Tests/conftest.py ``bootstrap_profile``; see
# lessons-testing-evidence "Tests/UI RecoveryRequired at setup").
pytestmark = pytest.mark.bootstrap_profile


def _console_app(tmp_path):
    app, gateway = _build_navigation_app(tmp_path)
    # Console composes its context estimate on first mount; the shared
    # navigation double predates that seam (it is only ever mounted after a
    # seeding push), so give it the same "window unknown" answer a real
    # gateway gives before any probe.
    gateway.cached_context_window = lambda _settings: None
    app._initial_tab_value = "home"
    return app, gateway


def _record_attention(monkeypatch, app) -> tuple[list[str], list[bool]]:
    notices: list[str] = []
    projections: list[bool] = []
    original_projection = app.set_console_attention_projection

    def record_projection(value: bool) -> None:
        projections.append(bool(value))
        original_projection(value)

    monkeypatch.setattr(
        app,
        "notify",
        lambda message, *args, **kwargs: notices.append(str(message)),
    )
    monkeypatch.setattr(app, "set_console_attention_projection", record_projection)
    return notices, projections


async def _reach_console(app, pilot, route: str) -> ChatScreen:
    for _ in range(300):
        if (
            type(app.screen).__name__ == "HomeScreen"
            and app.screen.is_mounted
            and getattr(app, "_initial_screen_pushed", False)
        ):
            break
        await pilot.pause(0.05)
    await pilot.pause(0.2)
    intent: ConsoleFirstChatIntent | None = None
    if route == "start_chatting":
        # Review round 2 (R2-F3): stage the handoff exactly as the wizard's
        # Start chatting does, so Console's real first mount consumes it --
        # the reserved session, the retired first-mount chat and the swapped
        # active tab are all part of what this route has to survive.
        assert SetupWizardContainer(app)._stage_console_first_chat_handoff()
        claim = app.pending_handoffs.claim(HandoffChannel.CONSOLE_FIRST_CHAT)
        assert claim is not None and isinstance(claim.value, ConsoleFirstChatIntent)
        intent = claim.value
        assert app.pending_handoffs.release(claim)
        app.handle_first_run_wizard_result({"completed": True, "exit_route": TAB_CHAT})
    else:
        app.handle_first_run_wizard_result({"completed": True, "exit_route": TAB_HOME})
        for _ in range(200):
            if type(app.screen).__name__ == "HomeScreen":
                break
            await pilot.pause(0.05)
        await _navigate(app, pilot, "chat", expect="ChatScreen")
    for _ in range(300):
        if isinstance(app.screen, ChatScreen) and app.screen.is_mounted:
            break
        await pilot.pause(0.05)
    chat = app.screen
    assert isinstance(chat, ChatScreen)
    await _wait_for_selector(chat, pilot, "#console-native-composer")
    if intent is not None:
        revision = claim.revision
        for _ in range(400):
            status = app.pending_handoffs.exact_revision_status(
                HandoffChannel.CONSOLE_FIRST_CHAT, revision
            )
            if status == "settled":
                break
            await pilot.pause(0.05)
        assert status == "settled", status
        store = chat._console_chat_store
        for _ in range(100):
            if store.active_session_id == intent.session_id:
                break
            await pilot.pause(0.05)
        # The handoff really applied: the reserved first chat is the tab.
        assert store.active_session_id == intent.session_id
    await pilot.pause(0.3)
    return chat


async def _run_visible_turn(chat, pilot, gateway, marker: str, session_id: str) -> None:
    runtime = chat._console_runtime()
    gateway.reply = marker
    turn_id = _admit_console_turn_to_runtime(chat, f"say {marker}", session_id)
    task = runtime._turn_custody[turn_id].task
    assert task is not None
    outcome = await asyncio.wait_for(task, timeout=10)
    assert outcome.accepted
    for _ in range(40):
        await pilot.pause(0.05)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["start_chatting", "explore_home"])
async def test_turns_finishing_in_the_visible_tab_raise_no_hidden_notice(
    tmp_path, monkeypatch, route
):
    app, gateway = _console_app(tmp_path)
    notices, projections = _record_attention(monkeypatch, app)

    async with app.run_test(size=(160, 48)) as pilot:
        chat = await _reach_console(app, pilot, route)
        store = chat._console_chat_store
        session_id = store.active_session_id
        assert session_id

        # The first reply and back-to-back follow-ups, all in the visible tab.
        marks = chat._console_runtime()._console_local_marks_service()
        for marker in ("VISIBLE-ONE", "VISIBLE-TWO", "VISIBLE-THREE"):
            await _run_visible_turn(chat, pilot, gateway, marker, session_id)
            # The visible view rendered the reply, so it acknowledged the
            # receipt: nothing is left to turn "hidden" on a later navigation.
            assert marks.list_console_unseen_marks() == ()

        assert app.screen is chat
        assert not any(_HIDDEN_COPY in text for text in notices), notices
        assert True not in projections, projections


@pytest.mark.asyncio
async def test_a_turn_finishing_while_console_is_off_screen_still_notifies_once(
    tmp_path, monkeypatch
):
    app, gateway = _console_app(tmp_path)
    notices, projections = _record_attention(monkeypatch, app)

    async with app.run_test(size=(160, 48)) as pilot:
        chat = await _reach_console(app, pilot, "explore_home")
        runtime = chat._console_runtime()
        store = chat._console_chat_store
        session_id = store.active_session_id
        await _run_visible_turn(chat, pilot, gateway, "VISIBLE-FIRST", session_id)
        assert not any(_HIDDEN_COPY in text for text in notices)

        gateway.arm_two_chunks("HIDDEN")
        turn_id = _admit_console_turn_to_runtime(chat, "finish while away", session_id)
        task = runtime._turn_custody[turn_id].task
        assert task is not None
        await asyncio.wait_for(gateway.first_chunk.wait(), timeout=5)
        try:
            await _navigate(
                app, pilot, "library", expect="LibraryScreen", allow_confirmation=False
            )
        finally:
            gateway.release_second.set()
            gateway.release_terminal.set()
        outcome = await asyncio.wait_for(task, timeout=10)
        assert outcome.accepted
        for _ in range(40):
            await pilot.pause(0.05)

        assert [text for text in notices if _HIDDEN_COPY in text] == [
            "A Console turn completed while hidden. Return to Console to review."
        ]
        assert projections and projections[-1] is True


@pytest.mark.asyncio
async def test_a_turn_finishing_in_a_background_tab_notifies_once_while_console_shows(
    tmp_path, monkeypatch
):
    """Review round 1 (F11): AC#8's second half. Console stays on screen, but
    the user opened another tab before the turn finished -- that turn did
    finish out of sight, so it notifies exactly once and raises the '!'."""
    app, gateway = _console_app(tmp_path)
    notices, projections = _record_attention(monkeypatch, app)

    async with app.run_test(size=(160, 48)) as pilot:
        chat = await _reach_console(app, pilot, "explore_home")
        runtime = chat._console_runtime()
        store = chat._console_chat_store
        controller = chat._ensure_console_chat_controller()
        session_id = store.active_session_id
        await _run_visible_turn(chat, pilot, gateway, "VISIBLE-FIRST", session_id)
        assert not any(_HIDDEN_COPY in text for text in notices)

        gateway.arm_two_chunks("BACKGROUND")
        turn_id = _admit_console_turn_to_runtime(chat, "finish in the background", session_id)
        task = runtime._turn_custody[turn_id].task
        assert task is not None
        await asyncio.wait_for(gateway.first_chunk.wait(), timeout=5)
        try:
            other = controller.new_session(title="Second tab")
            controller.switch_session(other.id)
            assert store.active_session_id == other.id
        finally:
            gateway.release_second.set()
            gateway.release_terminal.set()
        outcome = await asyncio.wait_for(task, timeout=10)
        assert outcome.accepted
        for _ in range(40):
            await pilot.pause(0.05)

        assert app.screen is chat, "Console never left the screen"
        assert [text for text in notices if _HIDDEN_COPY in text] == [
            "A Console turn completed while hidden. Return to Console to review."
        ]
        assert projections and projections[-1] is True


class _OverConsoleModal(ModalScreen[None]):
    """Stands in for Alt+M's model picker, Rename Chat Tab or the palette."""

    def compose(self):
        yield Static("modal over Console")


@pytest.mark.asyncio
async def test_a_turn_finishing_under_a_modal_over_console_raises_no_hidden_notice(
    tmp_path, monkeypatch
):
    """Review round 2 (V2-F1), live g5-v2-anthome 08/09, g5-v2-antsc 03: a
    turn that finished in the active tab while Alt+M's picker, the Rename
    Chat Tab dialog or the command palette was open over Console raised
    'completed while hidden' and the nav '!'. Console was still on screen
    behind the modal."""
    app, gateway = _console_app(tmp_path)
    notices, projections = _record_attention(monkeypatch, app)

    async with app.run_test(size=(160, 48)) as pilot:
        chat = await _reach_console(app, pilot, "explore_home")
        runtime = chat._console_runtime()
        session_id = chat._console_chat_store.active_session_id
        await _run_visible_turn(chat, pilot, gateway, "VISIBLE-FIRST", session_id)

        gateway.arm_two_chunks("UNDER-MODAL")
        turn_id = _admit_console_turn_to_runtime(chat, "finish under a modal", session_id)
        task = runtime._turn_custody[turn_id].task
        assert task is not None
        await asyncio.wait_for(gateway.first_chunk.wait(), timeout=5)
        modal = _OverConsoleModal()
        try:
            await app.push_screen(modal)
            await pilot.pause(0.1)
            assert app.screen is modal
        finally:
            gateway.release_second.set()
            gateway.release_terminal.set()
        outcome = await asyncio.wait_for(task, timeout=10)
        assert outcome.accepted
        for _ in range(40):
            await pilot.pause(0.05)

        assert app.screen is modal, "the modal stayed open while the turn ended"
        assert not any(_HIDDEN_COPY in text for text in notices), notices
        assert True not in projections, projections

        # Closing the modal shows the reply, which acknowledges it: no notice
        # or '!' arrives late, and nothing is left to turn "hidden" later.
        modal.dismiss()
        marks = runtime._console_local_marks_service()
        for _ in range(100):
            if app.screen is chat and marks.list_console_unseen_marks() == ():
                break
            await pilot.pause(0.05)
        assert app.screen is chat
        assert marks.list_console_unseen_marks() == ()
        assert not any(_HIDDEN_COPY in text for text in notices), notices
        assert True not in projections, projections
