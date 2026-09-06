"""Characterisation of `ChatScreen.on_button_pressed`'s branch behaviour.

Written BEFORE wave 4 task 2 routed the dispatcher's branch bodies to the
controllers that already own their state. Every test here presses the REAL
button on a mounted Console and asserts the **persisted** result -- a row in
the local-marks DB, a preference in `app_config`, a session in the chat
store, or controller state that outlives the repaint -- never the widget
tree the press happened to touch. That is what makes the file survive the
move byte-for-byte: it pins what a press DOES, not where the doing lives.

Coverage is deliberately weighted. `on_button_pressed` had 19 top-level
branches; the five largest (`console-workspace-conversation-` at 81 lines,
`console-conversation-star-` at 65, `console-close-session-tab-` at 39,
`console-workspace-conversations-toggle` at 35, `console-dictation` at 31)
account for 251 of its 381 lines and are all pinned below, whether or not
the extraction moved them -- two of those five stay on the screen as
coordination, and a characterisation test is exactly how a "stays" verdict
is defended later. The three smaller branches that also moved
(`console-conversation-browser-section-toggle-`,
`...-group-toggle-`, `console-session-tab-`) are pinned too. The remaining
branches are one- or two-line delegations that already read as routing-table
rows and already have coverage in their owning feature's test file.
"""

from __future__ import annotations

import asyncio
from types import MappingProxyType, SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_conversation_actions import (
    ACTION_FAVORITE,
    ACTION_UNFAVORITE,
    ConversationMenuTarget,
)
from tldw_chatbook.Widgets.Console.console_conversation_action_menu import (
    ConversationActionChosen,
)
from textual.css.query import NoMatches
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_workspace_context_rail import (
    _base_grouped_workspace_state,
    _browser_row,
)
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_prompt_queue import PromptQueuePauseReason
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Widgets.Console import ConsoleWorkspaceContextTray
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

#: Terminal size every routing test mounts at. Deliberately file-local
#: rather than shared: the Console shell is RESPONSIVE -- regions hide and
#: show by width, and the geometry baseline pins OBSERVED truth at specific
#: sizes -- so a repo-wide "standard console size" would let one edit
#: silently change what a dozen unrelated tests exercise. 160x44 is wide
#: enough that every button these tests press is mounted and hit-testable.
_ROUTING_SIZE = (160, 44)


def _install_real_marks_service(app, tmp_path) -> ConversationLocalMarksService:
    """Give the test app a marks service backed by a real sqlite file.

    `_build_test_app` leaves `chachanotes_db` unset, so the production
    wiring resolves `conversation_local_marks_service` to None. A star
    assertion is only worth making against a real row, so build one.
    """
    db = CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="button-routing")
    service = ConversationLocalMarksService(db)
    app.conversation_local_marks_service = service
    return service


async def _mounted_console(host, pilot, selector: str = "#console-workspace-context"):
    """Return the mounted Console screen once `selector` exists."""
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, selector)
    return console


async def _wait_for_confirmation(
    host, *, previous: ConfirmationDialog | None = None
) -> ConfirmationDialog:
    """Wait for a close worker to mount its confirmation without fixed sleeps."""

    for _ in range(200):
        candidate = host.screen_stack[-1]
        if isinstance(candidate, ConfirmationDialog) and candidate is not previous:
            try:
                candidate.query_one("#confirm-button", Button)
            except NoMatches:
                pass
            else:
                return candidate
        await asyncio.sleep(0.01)
    raise AssertionError("Console close confirmation did not mount")


async def _sync_tray(console, pilot, state) -> ConsoleWorkspaceContextTray:
    tray = console.query_one("#console-workspace-context", ConsoleWorkspaceContextTray)
    tray.sync_state(state)
    await pilot.pause()
    return tray


def _section_collapsed(console, group_id: str) -> bool:
    """Whether a browser SECTION is collapsed in the state the toggle reads.

    Deliberately `_build_console_workspace_context_state()` and not the tray
    the test synced: that is the exact source
    `_toggle_console_conversation_browser_section` consults to decide which
    way to flip.
    """
    section_id = group_id.removeprefix("section:")
    browser = console._workspace._build_console_workspace_context_state()
    assert browser.conversation_browser is not None
    section = next(
        candidate
        for candidate in browser.conversation_browser.sections
        if candidate.section_id == section_id
    )
    return bool(section.collapsed)


def _browser_config(app) -> dict:
    """The persisted grouped-browser preference dict, or an empty dict."""
    console_config = (app.app_config or {}).get("console")
    if not isinstance(console_config, dict):
        return {}
    browser_config = console_config.get("conversation_browser")
    if not isinstance(browser_config, dict):
        return {}
    collapsed = browser_config.get("collapsed_groups")
    return collapsed if isinstance(collapsed, dict) else {}


# --------------------------------------------------------------------------
# console-conversation-star-  (65 lines -- the second-largest branch)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_star_button_writes_a_durable_local_mark_and_toggles_it_back(tmp_path, monkeypatch):
    """Pressing the star persists through `conversation_local_marks_service`.

    The mark is the whole point of the branch: the row repaints from the
    service on the next sync, so asserting the glyph would only prove the
    repaint ran. Assert the service.
    """
    app = _build_test_app()
    marks = _install_real_marks_service(app, tmp_path)
    rows = (
        _browser_row(
            "conv-star-1",
            "Planning notes",
            scope_type="global",
            workspace_id=None,
            workspace_label="Chats",
        ),
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=_ROUTING_SIZE) as pilot:
        console = await _mounted_console(host, pilot)
        state = _base_grouped_workspace_state(rows=rows)
        # Keep the fixture row at the actual paint-input boundary. A queued
        # boot refresh otherwise replaces the directly painted fixture with
        # the unrelated blank native-session row during pilot.pause().
        monkeypatch.setattr(
            console._workspace, "_build_console_workspace_context_state", lambda: state
        )
        await _sync_tray(console, pilot, state)

        # TASK-23200: the per-row star button became an asterisk that opens
        # the row action menu. The durable write this test guards is now
        # reached through the menu's Favourite entry, which routes into the
        # same `_toggle_console_conversation_star` branch.
        opener = console.query_one("#console-conversation-actions-0", Button)
        assert opener.disabled is False
        assert opener.conversation_id == "conv-star-1"

        console.on_conversation_action_chosen(
            ConversationActionChosen(
                ACTION_FAVORITE,
                ConversationMenuTarget(
                    conversation_id="conv-star-1", title="Planning notes"
                ),
            )
        )
        await pilot.pause()
        # task-15471: the durable write runs on a worker now -- wait for it
        # rather than racing the pool thread with the assertion.
        await console.workers.wait_for_complete()
        assert marks.is_starred("conv-star-1") is True

        # Choosing it again unstars: the branch reads current truth from the
        # service, not from whatever the row was painted with.
        await _sync_tray(console, pilot, _base_grouped_workspace_state(rows=rows))
        console.on_conversation_action_chosen(
            ConversationActionChosen(
                ACTION_UNFAVORITE,
                ConversationMenuTarget(
                    conversation_id="conv-star-1",
                    title="Planning notes",
                    starred=True,
                ),
            )
        )
        await pilot.pause()
        await console.workers.wait_for_complete()
        assert marks.is_starred("conv-star-1") is False


@pytest.mark.asyncio
async def test_star_button_writes_nothing_when_the_marks_service_is_missing():
    """No service is a warning, not a crash and not a half-write."""
    app = _build_test_app()
    app.conversation_local_marks_service = None
    rows = (
        _browser_row(
            "conv-star-2",
            "Unbacked row",
            scope_type="global",
            workspace_id=None,
            workspace_label="Chats",
        ),
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=_ROUTING_SIZE) as pilot:
        console = await _mounted_console(host, pilot)
        await _sync_tray(console, pilot, _base_grouped_workspace_state(rows=rows))

        # TASK-23200: reached through the row action menu's Favourite entry
        # rather than a dedicated star button.
        console.on_conversation_action_chosen(
            ConversationActionChosen(
                ACTION_FAVORITE,
                ConversationMenuTarget(
                    conversation_id="conv-star-2", title="Unbacked row"
                ),
            )
        )
        await pilot.pause()

        # Survived the choice with the branch's own guard, not an exception.
        assert app.conversation_local_marks_service is None


# --------------------------------------------------------------------------
# console-conversation-browser-section-toggle-  (22 lines)
# console-conversation-browser-group-toggle-    (24 lines)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_browser_section_toggle_persists_its_collapse_preference():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=_ROUTING_SIZE) as pilot:
        console = await _mounted_console(host, pilot)
        await _sync_tray(console, pilot, _base_grouped_workspace_state())

        def _section_toggles() -> list[Button]:
            # Re-queried before EVERY press: `_toggle_console_conversation_
            # browser_section` ends in `_sync_console_workspace_context()`,
            # which rebuilds the tray with NEW Button instances. The old
            # object still answers `is_mounted` True but is no longer in the
            # DOM, and pressing it is a silent no-op.
            return [
                button
                for button in console.query(Button)
                if str(getattr(button, "id", "") or "").startswith(
                    "console-conversation-browser-section-toggle-"
                )
            ]

        assert _section_toggles(), "the grouped browser must render section toggles"
        group_id = _section_toggles()[0].group_id
        assert group_id.startswith("section:")

        def _press_section_toggle() -> None:
            # By group_id, not by position: a rebuild is free to reorder.
            next(
                button for button in _section_toggles() if button.group_id == group_id
            ).press()

        # The polarity is NOT a constant. The handler flips the section's
        # CURRENT collapsed state, taken from the screen's own rebuilt
        # context -- not from the fabricated tray state `_sync_tray` painted
        # above. Starred default-collapses while it holds no rows
        # (TASK-2154.3 LY-04, commit 7dbbc401b), which is the screen's real
        # situation here, so the first press EXPANDS and persists False.
        # `is True` pinned the pre-2154.3 default; what this test claims is
        # that a press persists a preference under the toggle's own key and
        # that the preference is a genuine flip, so assert that instead.
        before = _section_collapsed(console, group_id)

        _press_section_toggle()
        await pilot.pause()
        assert _browser_config(app).get(group_id) is (not before)

        _press_section_toggle()
        await pilot.pause()
        assert _browser_config(app).get(group_id) is before


@pytest.mark.asyncio
async def test_flat_browser_has_no_retired_workspace_group_toggles():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=_ROUTING_SIZE) as pilot:
        console = await _mounted_console(host, pilot)
        await _sync_tray(console, pilot, _base_grouped_workspace_state())

        toggles = [
            button
            for button in console.query(Button)
            if str(getattr(button, "id", "") or "").startswith(
                "console-conversation-browser-group-toggle-"
            )
        ]
        assert toggles == []


@pytest.mark.asyncio
async def test_workspace_files_controls_carry_stable_workspace_ids() -> None:
    """The tree menu target addresses a workspace without parsing its label."""
    app = _build_test_app()
    app.workspace_registry_service.create_workspace(
        workspace_id="ws-a", name="Workspace [label only]"
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=_ROUTING_SIZE) as pilot:
        console = await _mounted_console(host, pilot)
        console._workspace._workspace_files_availability_by_id = MappingProxyType(
            {"ws-a": True}
        )
        target = console._row_actions._workspace_menu_target("ws-a")

        assert target.workspace_id == "ws-a"
        assert target.name == "Workspace [label only]"
        assert target.files_available is True


# --------------------------------------------------------------------------
# console-workspace-conversations-toggle  (35 lines -- fourth-largest)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_widget_carries_the_workspace_conversations_toggle_id():
    """The 35-line `console-workspace-conversations-toggle` branch is DEAD.

    Nothing in the app constructs a widget with that id. It was the legacy
    ungrouped conversation list's collapse control, and commit `3b0374479`
    ("retire the unreachable legacy conversation list [TASK-1190]") removed
    the button while leaving the dispatcher branch behind --
    `console-workspace-conversations-toggle` survives only as a CSS *class*
    on the grouped browser's section/group toggles, whose ids are
    `console-conversation-browser-{section,group}-toggle-*` and which take
    entirely different branches.

    So this branch cannot be characterised by pressing anything, and it has
    no owner to be routed to. Pin the deadness instead: if someone gives a
    widget that id again, this fails and the branch is live code that needs
    a real test and a real owner.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=_ROUTING_SIZE) as pilot:
        console = await _mounted_console(host, pilot)
        await _sync_tray(console, pilot, _base_grouped_workspace_state())

        assert len(console.query("#console-workspace-conversations-toggle")) == 0
        # ...and the class-bearing toggles that DO exist route elsewhere.
        classed = list(console.query(".console-workspace-conversations-toggle"))
        assert classed
        assert all(
            str(button.id or "").startswith(
                "console-conversation-browser-section-toggle-"
            )
            or str(button.id or "").startswith(
                "console-conversation-browser-group-toggle-"
            )
            for button in classed
        )


# --------------------------------------------------------------------------
# console-workspace-conversation-  (81 lines -- the largest branch)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workspace_conversation_row_switches_to_its_already_open_session():
    """Pressing a row with an open native tab switches the store, not resumes.

    The persisted result is `store.active_session_id`; every visible effect
    (tab strip, transcript, temporary chip) is downstream of it.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=_ROUTING_SIZE) as pilot:
        console = await _mounted_console(host, pilot)
        store = console._ensure_console_chat_store()
        first_session_id = store.active_session_id
        second = store.create_session()
        store.switch_session(first_session_id)
        assert store.active_session_id == first_session_id

        rows = (
            _browser_row(
                f"native:{second.id}",
                "Second tab",
                conversation_id=f"native:{second.id}",
                native_session_id=second.id,
                source_kind="native",
                scope_type="global",
                workspace_id=None,
                workspace_label="Chats",
            ),
        )
        await _sync_tray(console, pilot, _base_grouped_workspace_state(rows=rows))

        console.query_one("#console-workspace-conversation-0", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert store.active_session_id == second.id


# --------------------------------------------------------------------------
# console-close-session-tab-  (39 lines -- third-largest)
# console-session-tab-       (9 lines)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_tab_button_drops_an_empty_session_without_confirmation():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=_ROUTING_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        keeper_id = store.active_session_id
        doomed = store.create_session()
        # Close a BACKGROUND tab (the × on a non-active tab). Closing the
        # active one leaves an undo entry behind, because the draft sync
        # that follows the close re-seeds `_console_undo_histories` for the
        # still-visible session id -- pre-existing, and not what this
        # branch is being characterised for.
        store.switch_session(keeper_id)
        # A real `ConsoleComposerUndoHistory` (an (undo, redo) stack pair),
        # so the close path's own consumers see the shape they expect.
        console._console_undo_histories[doomed.id] = ([], [])
        await console._sync_native_console_chat_ui()
        await pilot.pause()

        close = console.query_one(f"#console-close-session-tab-{doomed.id}", Button)
        close.press()
        await pilot.pause()
        await pilot.pause()

        assert doomed.id not in {session.id for session in store.sessions()}
        assert doomed.id not in console._console_undo_histories
        assert not isinstance(host.screen_stack[-1], ConfirmationDialog)


@pytest.mark.asyncio
async def test_close_tab_button_drops_an_idle_saved_session_without_confirmation():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=_ROUTING_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        keeper_id = store.active_session_id
        saved = store.restore_persisted_session(
            title="Saved chat",
            workspace_id=None,
            persisted_conversation_id="saved-conversation",
            all_nodes=(
                ConsoleChatMessage(
                    role=ConsoleMessageRole.USER,
                    content="already durable",
                    persisted_message_id="saved-message",
                ),
            ),
        )
        store.switch_session(keeper_id)
        await console._sync_native_console_chat_ui()
        await pilot.pause()

        console.query_one(f"#console-close-session-tab-{saved.id}", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert saved.id not in {session.id for session in store.sessions()}
        assert not isinstance(host.screen_stack[-1], ConfirmationDialog)


@pytest.mark.asyncio
async def test_close_tab_button_confirms_for_unsaved_message_on_hidden_branch():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=_ROUTING_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        keeper_id = store.active_session_id
        root = ConsoleChatMessage(
            id="saved-root",
            role=ConsoleMessageRole.USER,
            content="saved root",
            persisted_message_id="saved-root",
        )
        saved_leaf = ConsoleChatMessage(
            id="saved-leaf",
            role=ConsoleMessageRole.ASSISTANT,
            content="saved branch",
            persisted_message_id="saved-leaf",
            parent_message_id="saved-root",
        )
        saved = store.restore_persisted_session(
            title="Saved chat with hidden work",
            workspace_id=None,
            persisted_conversation_id="saved-conversation",
            all_nodes=(root, saved_leaf),
            active_leaf_persisted_id="saved-leaf",
        )
        hidden_unsaved = store.create_sibling(
            saved_leaf.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="unsaved hidden branch",
        )
        store.set_active_leaf(saved.id, saved_leaf.id)
        assert hidden_unsaved.id not in store.active_path_message_ids(saved.id)
        assert all(
            message.persisted_message_id is not None
            for message in store.messages_for_session(saved.id)
        )
        store.switch_session(keeper_id)
        await console._sync_native_console_chat_ui()
        await pilot.pause()

        console.query_one(f"#console-close-session-tab-{saved.id}", Button).press()
        dialog = await _wait_for_confirmation(host)

        assert "Transcript messages: 1" in dialog.message
        assert saved.id in {session.id for session in store.sessions()}


@pytest.mark.asyncio
async def test_close_tab_button_confirms_before_dropping_a_session_with_messages():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=_ROUTING_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        keeper_id = store.active_session_id
        doomed = store.create_session()
        store.switch_session(keeper_id)
        store.append_message(
            doomed.id, role=ConsoleMessageRole.USER, content="do not lose me"
        )
        await console._sync_native_console_chat_ui()
        await pilot.pause()

        close = console.query_one(f"#console-close-session-tab-{doomed.id}", Button)
        close.press()
        dialog = await _wait_for_confirmation(host)

        assert dialog.message.startswith("Closing this session will discard or cancel:")
        assert "Transcript messages: 1" in dialog.message
        assert "Live agent turns: 0" in dialog.message
        assert "Unsent queued prompts: 0" in dialog.message
        # Still open: the confirmation is a gate, not a notification.
        assert doomed.id in {session.id for session in store.sessions()}

        dialog.query_one("#confirm-button", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert doomed.id not in {session.id for session in store.sessions()}


@pytest.mark.asyncio
async def test_close_empty_session_with_queue_warns_without_exposing_prompt_text():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=_ROUTING_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        controller = console._ensure_console_chat_controller()
        store = controller.store
        keeper_id = store.active_session_id
        doomed = store.create_session()
        store.switch_session(keeper_id)

        registry = controller.prompt_queue_registry
        snapshot = registry.snapshot(doomed.id)
        started = registry.begin_chain(
            doomed.id,
            context_epoch=store.conversation_context_epoch(doomed.id),
            expected_revision=snapshot.revision,
        )
        assert started.applied
        controller.prompt_queue_coordinator._changed(doomed.id)
        queued = controller.queue_prompt(
            doomed.id,
            text="secret queued close text",
            expected_revision=started.snapshot.revision,
        )
        assert queued.applied
        paused = registry.pause(
            doomed.id,
            reason=PromptQueuePauseReason.MANUAL,
            expected_revision=queued.snapshot.revision,
        )
        assert paused.applied
        controller.prompt_queue_coordinator._changed(doomed.id)

        await console._sync_native_console_chat_ui()
        await pilot.pause()
        console.query_one(f"#console-close-session-tab-{doomed.id}", Button).press()
        dialog = await _wait_for_confirmation(host)

        assert "Transcript messages: 0" in dialog.message
        assert "Live agent turns: 0" in dialog.message
        assert "Unsent queued prompts: 1" in dialog.message
        assert "secret queued close text" not in dialog.message

        dialog.query_one("#cancel-button", Button).press()
        await pilot.pause()
        assert doomed.id in {session.id for session in store.sessions()}
        assert registry.snapshot(doomed.id).total_count == 1


@pytest.mark.asyncio
async def test_close_revalidates_changed_impact_and_presents_updated_dialog():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=_ROUTING_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        keeper_id = store.active_session_id
        doomed = store.create_session()
        store.switch_session(keeper_id)
        store.append_message(doomed.id, role=ConsoleMessageRole.USER, content="one")
        await console._sync_native_console_chat_ui()
        await pilot.pause()

        console.query_one(f"#console-close-session-tab-{doomed.id}", Button).press()
        first = await _wait_for_confirmation(host)
        assert "Transcript messages: 1" in first.message

        store.append_message(doomed.id, role=ConsoleMessageRole.USER, content="two")
        first.query_one("#confirm-button", Button).press()
        second = await _wait_for_confirmation(host, previous=first)

        assert "Transcript messages: 2" in second.message
        assert doomed.id in {session.id for session in store.sessions()}
        second.query_one("#cancel-button", Button).press()
        await pilot.pause()
        assert doomed.id in {session.id for session in store.sessions()}


@pytest.mark.asyncio
async def test_session_tab_button_activates_an_inactive_session():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=_ROUTING_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        first_session_id = store.active_session_id
        second = store.create_session()
        store.switch_session(first_session_id)
        await console._sync_native_console_chat_ui()
        await pilot.pause()

        console.query_one(f"#console-session-tab-{second.id}", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert store.active_session_id == second.id


# --------------------------------------------------------------------------
# console-dictation  (31 lines -- fifth-largest)
# --------------------------------------------------------------------------


class _ExitRecorder:
    """A stand-in voice session whose controller records `on_exit_request`.

    `tick_timer`/`assistant_row_id` are here for the screen's UNMOUNT
    teardown, which runs against whatever session is still installed when
    the harness tears down; both defaults make that teardown a no-op.
    """

    def __init__(self) -> None:
        self.exits = 0
        self.tick_timer = None
        self.assistant_row_id = None
        self.controller = SimpleNamespace(on_exit_request=self._on_exit_request)

    def _on_exit_request(self) -> None:
        self.exits += 1


@pytest.mark.asyncio
async def test_mic_button_opens_a_capture_when_idle(monkeypatch):
    """Idle press arms a capture: state and origin session both persist."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=_ROUTING_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")

        async def _no_op_start() -> None:
            return None

        monkeypatch.setattr(
            console._dictation, "_start_console_dictation", _no_op_start
        )
        store = console._ensure_console_chat_store()
        assert console._console_dictation_state == "idle"

        console.query_one("#console-dictation", Button).press()
        await pilot.pause()

        assert console._console_dictation_state == "starting"
        assert (
            console._dictation._console_dictation_origin_session_id
            == store.active_session_id
        )


@pytest.mark.asyncio
async def test_mic_button_exits_the_hands_free_loop_instead_of_toggling():
    """A running hands-free loop supersedes the one-shot toggle entirely."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=_ROUTING_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        recorder = _ExitRecorder()
        console._console_hands_free = recorder

        console.query_one("#console-dictation", Button).press()
        await pilot.pause()

        assert recorder.exits == 1
        # No fall-through: the one-shot capture never armed.
        assert console._console_dictation_state == "idle"
        console._console_hands_free = None


@pytest.mark.asyncio
async def test_mic_button_exits_the_realtime_loop_instead_of_toggling():
    """Same rule for the V4 realtime engine (it would otherwise double-open
    the microphone and keep billing)."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=_ROUTING_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        recorder = _ExitRecorder()
        console._console_realtime = recorder

        console.query_one("#console-dictation", Button).press()
        await pilot.pause()

        assert recorder.exits == 1
        assert console._console_dictation_state == "idle"
        console._console_realtime = None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "expected"),
    [("starting", "cancel"), ("recording", "stop")],
)
async def test_mic_button_routes_a_live_capture_to_cancel_or_stop(
    state: str, expected: str, monkeypatch
):
    """`starting` cancels (the only way out of a model download); `recording`
    stops (the capture is worth finishing)."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=_ROUTING_SIZE) as pilot:
        console = await _mounted_console(host, pilot, "#console-native-composer")
        calls: list[str] = []
        for name in ("start", "cancel", "stop"):
            monkeypatch.setattr(
                console._dictation,
                f"_request_console_dictation_{name}",
                (lambda captured=name: calls.append(captured)),
            )
        console._console_dictation_state = state

        console.query_one("#console-dictation", Button).press()
        await pilot.pause()

        assert calls == [expected]
