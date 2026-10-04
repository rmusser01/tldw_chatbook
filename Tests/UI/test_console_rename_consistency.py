"""Explicit rename must publish one durable title across live Console surfaces."""

from __future__ import annotations

import asyncio
import threading

import pytest
from textual.widgets import Button, Input, Static

from Tests.UI.test_console_character_activation_presentation import (
    _until,
    activation_library,  # noqa: F401
)
from Tests.UI.test_console_native_chat_flow import _static_plain_text
from Tests.UI.test_console_workbench_contract import (
    ConsoleHarness,
    _configure_native_ready_console,
)
from Tests.UI.test_library_inspection_admission import library  # noqa: F401
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
    ConsoleSessionSwitcherModal,
)
from tldw_chatbook.Widgets.Console.console_workspace_tree import ConsoleWorkspaceTree

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
async def test_title_receipt_retires_when_the_tray_is_replaced(
    activation_library,  # noqa: F811
    monkeypatch,
) -> None:
    """The screen, not a retiring tray, must own the refresh callback.

    Args:
        activation_library: Isolated app and SQLite owners.
        monkeypatch: Hold the receipt until the actual tray has been removed.
    """
    owner, _, _ = activation_library
    host = ConsoleHarness(owner)
    async with host.run_test(size=(120, 50)) as pilot:
        chat = host.screen
        await pilot.pause()
        tray = chat.query_one("#console-workspace-context")
        pending = []

        def hold_receipt(node, original):
            def schedule(callback, *args, **kwargs):
                if getattr(callback, "__name__", "") == "complete_refresh":
                    pending.append((node, callback))
                    return True
                return original(callback, *args, **kwargs)

            return schedule

        monkeypatch.setattr(
            chat, "call_after_refresh", hold_receipt(chat, chat.call_after_refresh)
        )
        monkeypatch.setattr(
            tray, "call_after_refresh", hold_receipt(tray, tray.call_after_refresh)
        )
        receipt = asyncio.create_task(tray.wait_for_publication())
        try:
            await _until(lambda: bool(pending))
            await tray.remove()
            # A retired message pump drops its queued callbacks. Only the
            # still-running screen can deliver this deferred refresh.
            for node, callback in pending:
                if node is chat:
                    callback()
            assert await asyncio.wait_for(receipt, 1) is False
        finally:
            if not receipt.done():
                receipt.cancel()
                await asyncio.gather(receipt, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("departure", ["cancel", "profile"])
async def test_cancelled_rename_settles_committed_title_before_releasing_worker(
    activation_library,  # noqa: F811
    monkeypatch,
    tmp_path,
    departure: str,
) -> None:
    """Cancelling an await cannot cancel the SQLite write already in its thread.

    Args:
        activation_library: Isolated app and real SQLite lifetime owners.
        monkeypatch: Hold only the durable write to establish cancellation order.
        tmp_path: Owns the independent database for a profile departure.
        departure: Cancellation or profile change while the write is running.
    """
    owner, _, db = activation_library
    _configure_native_ready_console(owner)
    entered, release, completed = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    original_update = db.update_conversation
    second_db = None
    if departure == "profile":
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

        second_db = CharactersRAGDB(
            tmp_path / "departed-profile.sqlite", client_id="rename-departure"
        )
        second_db.add_conversation({"id": "exact", "title": "Independent profile"})

    def held_update(*args, **kwargs):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("rename test did not release its write")
        try:
            return original_update(*args, **kwargs)
        finally:
            completed.set()

    host = ConsoleHarness(owner)
    async with host.run_test(size=(120, 50)):
        chat = host.screen
        assert await chat._workspace._resume_console_workspace_conversation("exact")
        target = chat._ensure_console_chat_store().ensure_session()
        notes = []
        owner.notify = lambda message, **kwargs: notes.append(str(message))
        monkeypatch.setattr(db, "update_conversation", held_update)
        chat._workspace._rename_console_conversation(
            "exact", "Committed while cancelling"
        )
        try:
            await _until(entered.is_set)
            worker = next(
                worker
                for worker in chat.workers
                if worker.group == "console-conversation-rename"
            )
            if departure == "cancel":
                worker.cancel()
            else:
                owner.chachanotes_db = second_db
        finally:
            release.set()
        try:
            await _until(completed.is_set)
            await _until(lambda: worker._task.done())
            assert (
                db.get_conversation_by_id("exact")["title"]
                == "Committed while cancelling"
            )
            assert target.title == "Committed while cancelling"
            assert not any(note.startswith("Renamed to") for note in notes)
            if second_db is not None:
                assert (
                    second_db.get_conversation_by_id("exact")["title"]
                    == "Independent profile"
                )
        finally:
            owner.chachanotes_db = db
            if second_db is not None:
                from Tests.conftest import _close_database_instance

                _close_database_instance(second_db)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_point", ["rail", "tab"])
async def test_rename_modal_rejects_a_changed_profile_before_dispatch(
    activation_library,  # noqa: F811
    tmp_path,
    monkeypatch,
    entry_point: str,
) -> None:
    """An old modal's conversation ID must not be applied in another profile.

    Args:
        activation_library: Original app and SQLite owners.
        tmp_path: Owns the independent second profile database.
        monkeypatch: Route the owner's modal push to the mounted harness.
        entry_point: Existing rail or tab title prompt.
    """
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    owner, _, db = activation_library
    _configure_native_ready_console(owner)
    second_db = CharactersRAGDB(
        tmp_path / "second-profile.sqlite", client_id="rename-profile"
    )
    second_db.add_conversation({"id": "exact", "title": "Another profile"})
    try:
        host = ConsoleHarness(owner)
        async with host.run_test(size=(120, 50)) as pilot:
            monkeypatch.setattr(owner, "push_screen", host.push_screen)
            chat = host.screen
            assert await chat._workspace._resume_console_workspace_conversation("exact")
            target = chat._ensure_console_chat_store().ensure_session()
            old_title = target.title
            notes = []
            owner.notify = lambda message, **kwargs: notes.append(str(message))
            if entry_point == "rail":
                chat._workspace.open_console_conversation_rename("exact", old_title)
            else:
                chat._session._open_console_session_rename_modal(target.id)
            await _until(
                lambda: (
                    host.screen is not chat
                    and bool(host.screen.query("#console-rename-session-title"))
                )
            )
            await pilot.pause()
            owner.chachanotes_db = second_db
            host.screen.query_one(
                "#console-rename-session-title", Input
            ).value = "Wrong profile rename"
            await pilot.press("enter")
            await pilot.pause()
            await chat.workers.wait_for_complete(
                [
                    worker
                    for worker in chat.workers
                    if worker.group == "console-conversation-rename"
                ]
            )
            assert db.get_conversation_by_id("exact")["title"] == old_title
            assert (
                second_db.get_conversation_by_id("exact")["title"] == "Another profile"
            )
            assert target.title == old_title
            assert any("profile" in note.lower() for note in notes)
    finally:
        owner.chachanotes_db = db
        from Tests.conftest import _close_database_instance

        _close_database_instance(second_db)


@pytest.mark.asyncio
@pytest.mark.parametrize("active", [True, False], ids=["active", "inactive"])
@pytest.mark.parametrize("entry_point", ["rail", "tab"])
async def test_rail_rename_publishes_saved_title_to_open_tab_before_confirmation(
    activation_library,  # noqa: F811
    monkeypatch,
    active: bool,
    entry_point: str,
) -> None:
    """A durable rename cannot leave the bound runtime or switcher stale.

    Args:
        activation_library: Isolated app and real local SQLite lifetime owners.
        monkeypatch: Route the owner's modal push to the mounted harness.
        active: Whether the renamed conversation is the currently selected tab.
        entry_point: Mounted rail menu or bound tab/F2 modal callback.
    """
    owner, _, db = activation_library
    _configure_native_ready_console(owner)
    owner.local_chat_conversation_service = ChatConversationService(db)
    db.add_conversation({"id": "rename-other", "title": "Other chat"})
    host = ConsoleHarness(owner)
    async with host.run_test(size=(120, 50)) as pilot:
        monkeypatch.setattr(owner, "push_screen", host.push_screen)
        chat = host.screen
        await _until(lambda: chat.query("#console-native-composer"))
        assert await chat._workspace._resume_console_workspace_conversation("exact")
        store = chat._ensure_console_chat_store()
        target = next(
            session
            for session in store.sessions()
            if session.id == store.active_session_id
        )
        assert await chat._workspace._resume_console_workspace_conversation(
            "rename-other"
        )
        other = next(
            session
            for session in store.sessions()
            if session.id == store.active_session_id
        )
        if active:
            await chat._session._activate_native_console_session(target.id)
        selected = store.active_session_id
        notes: list[str] = []
        confirmed: list[tuple[str, str, tuple[str, ...], str]] = []

        def observe_notification(message: str, **kwargs) -> None:
            if str(message).startswith("Renamed to"):
                button = chat.query_one(f"#console-session-tab-{target.id}", Button)
                rail_labels = tuple(
                    str(row.label)
                    for row in chat.query_one("#console-workspace-context").query(
                        Button
                    )
                    if (
                        getattr(row, "native_session_id", None) == target.id
                        or getattr(row, "conversation_id", None) == "exact"
                    )
                    and row.has_class("console-workspace-conversation-row")
                )
                header = _static_plain_text(
                    chat.query_one("#console-transcript-title", Static)
                )
                confirmed.append((target.title, str(button.label), rail_labels, header))
            notes.append(str(message))

        owner.notify = observe_notification
        if entry_point == "rail":
            await pilot.pause()
            opener = next(
                button
                for button in chat.query_one("#console-workspace-context").query(Button)
                if getattr(button, "native_session_id", None) == target.id
                and (button.id or "").startswith("console-conversation-actions-")
            )
            opener.press()
            await _until(
                lambda: bool(chat.query("#console-conversation-action-rename"))
            )
            chat.query_one("#console-conversation-action-rename", Button).press()
        else:
            chat._session._open_console_session_rename_modal(target.id)
        await _until(
            lambda: (
                host.screen is not chat
                and bool(host.screen.query("#console-rename-session-title"))
            )
        )
        await pilot.pause()
        host.screen.query_one(
            "#console-rename-session-title", Input
        ).value = "Renamed Alpha"
        await pilot.press("enter")
        await pilot.pause()
        await chat.workers.wait_for_complete(
            [
                worker
                for worker in chat.workers
                if worker.group == "console-conversation-rename"
            ]
        )
        await pilot.pause()

        assert db.get_conversation_by_id("exact")["title"] == "Renamed Alpha"
        assert target.title == "Renamed Alpha"
        assert other.title == "Other chat"
        assert store.active_session_id == selected
        assert any(note.startswith("Renamed to") for note in notes)
        assert len(confirmed) == 1
        assert confirmed[0][0] == "Renamed Alpha"
        assert "Renamed Alpha" in confirmed[0][1]
        assert confirmed[0][2], [
            (row.id, str(row.label), getattr(row, "conversation_id", None))
            for row in chat.query_one("#console-workspace-context").query(Button)
        ]
        assert all("Renamed Alpha" in label for label in confirmed[0][2]), confirmed[0][
            2
        ]
        if active:
            assert "Renamed Alpha" in confirmed[0][3]
            assert "Renamed Alpha" in _static_plain_text(
                chat.query_one("#console-transcript-title", Static)
            )
        tray = chat.query_one("#console-workspace-context")
        matching_rows = [
            button
            for button in tray.query(Button)
            if getattr(button, "native_session_id", None) == target.id
            or getattr(button, "conversation_id", None) == "exact"
        ]
        assert matching_rows
        assert any("Renamed Alpha" in str(button.label) for button in matching_rows)
        await chat.action_open_console_session_switcher()
        await _until(lambda: isinstance(host.screen, ConsoleSessionSwitcherModal))
        modal = host.screen
        await _until(lambda: bool(modal._entries))
        entries = [
            entry for entry in modal._entries if entry.native_session_id == target.id
        ]
        assert len(entries) == 1
        assert entries[0].title == "Renamed Alpha"
        await pilot.press("escape")
        if not active:
            await chat._session._activate_native_console_session(target.id)
            assert "Renamed Alpha" in _static_plain_text(
                chat.query_one("#console-transcript-title", Static)
            )


@pytest.mark.asyncio
async def test_filtered_workspace_tree_publishes_the_committed_title(
    activation_library,  # noqa: F811
) -> None:
    """A filtered named-workspace lane must not retain the old title copy.

    Args:
        activation_library: Isolated app and real SQLite owners.
    """
    owner, _, db = activation_library
    _configure_native_ready_console(owner)
    owner.workspace_registry_service.create_workspace(
        workspace_id="rename-workspace", name="Rename Workspace"
    )
    record = db.get_conversation_by_id("exact")
    assert db.update_conversation(
        "exact",
        {"scope_type": "workspace", "workspace_id": "rename-workspace"},
        record["version"],
    )
    host = ConsoleHarness(owner)
    async with host.run_test(size=(160, 50)) as pilot:
        chat = host.screen
        assert await chat._workspace._resume_console_workspace_conversation("exact")
        await chat._workspace.refresh_workspace_tree_search("inspection")
        await pilot.pause()
        lane = chat._workspace._workspace_tree_search
        assert any(
            row.conversation_id == "exact" and row.title == "Exact local inspection"
            for row in lane.rows
        )
        confirmed = []

        def observe(message, **kwargs):
            if str(message).startswith("Renamed to"):
                tree = chat.query_one(ConsoleWorkspaceTree)
                labels = (
                    (str(tree.conversation_nodes["exact"].label),)
                    if "exact" in tree.conversation_nodes
                    else ()
                )
                confirmed.append(
                    (
                        tuple(
                            row.title
                            for row in lane.rows
                            if row.conversation_id == "exact"
                        ),
                        labels,
                    )
                )

        owner.notify = observe
        chat._workspace._rename_console_conversation("exact", "Inspection Renamed")
        await chat.workers.wait_for_complete(
            [
                worker
                for worker in chat.workers
                if worker.group == "console-conversation-rename"
            ]
        )
        assert confirmed
        assert confirmed[0][0] == ("Inspection Renamed",)
        assert confirmed[0][1] and all(
            "Inspection Renamed" in label for label in confirmed[0][1]
        ), confirmed


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_point", ["rail", "tab"])
@pytest.mark.parametrize("failure", ["refused", "exception"])
async def test_failed_rename_preserves_both_saved_and_live_title(
    activation_library,  # noqa: F811
    monkeypatch,
    entry_point: str,
    failure: str,
) -> None:
    """A refused optimistic write must never publish a speculative live title.

    Args:
        activation_library: Isolated app and real SQLite lifetime owners.
        monkeypatch: Inject a failure only at the durable update seam.
        entry_point: Rail action or bound tab/F2 modal callback.
        failure: Falsy update result or storage exception.
    """
    owner, _, db = activation_library
    _configure_native_ready_console(owner)
    host = ConsoleHarness(owner)
    async with host.run_test(size=(120, 50)) as pilot:
        chat = host.screen
        assert await chat._workspace._resume_console_workspace_conversation("exact")
        store = chat._ensure_console_chat_store()
        target = store.ensure_session()
        old_title = target.title
        notes: list[str] = []
        owner.notify = lambda message, **kwargs: notes.append(str(message))

        def refuse_update(*args, **kwargs):
            if failure == "exception":
                raise RuntimeError("qualification write refusal")
            return False

        monkeypatch.setattr(db, "update_conversation", refuse_update)
        if entry_point == "rail":
            chat._workspace._rename_console_conversation("exact", "Not saved")
        else:
            chat._session._open_console_session_rename_modal(target.id)
            await _until(
                lambda: (
                    host.screen is not chat
                    and bool(host.screen.query("#console-rename-session-title"))
                )
            )
            await pilot.pause()
            host.screen.query_one(
                "#console-rename-session-title", Input
            ).value = "Not saved"
            await pilot.press("enter")
            await pilot.pause()
        await chat.workers.wait_for_complete(
            [
                worker
                for worker in chat.workers
                if worker.group == "console-conversation-rename"
            ]
        )
        assert db.get_conversation_by_id("exact")["title"] == old_title
        assert target.title == old_title
        assert not any(note.startswith("Renamed to") for note in notes)
        assert any("Could not rename" in note for note in notes)
