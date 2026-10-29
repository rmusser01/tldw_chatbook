"""Uncapped note capture, pinned source, and one-write navigation lifecycle."""

import asyncio
from types import SimpleNamespace
import threading

import pytest

from Tests.UI.test_console_trace_copy import TraceStore, trace_controller
from Tests.UI.test_library_notes_files_sync_journey import (
    _isolate_actor_pack_recovery,  # noqa: F401 - shared autouse startup isolation
)

pytestmark = pytest.mark.bootstrap_profile


class NoteStore(TraceStore):
    def __init__(self, *, ephemeral=False):
        super().__init__([])
        self._sessions = {
            "source": SimpleNamespace(
                id="source", title="Source chat", ephemeral=ephemeral
            ),
            "other": SimpleNamespace(id="other", title="Other chat", ephemeral=False),
        }

    def sessions(self):
        return list(self._sessions.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("length", [3999, 4000, 4001, 10_000, 100_000])
@pytest.mark.parametrize("open_note", [False, True])
async def test_both_note_actions_preserve_complete_text(length, open_note):
    prefix, suffix = "  漢字\n", "\n  "
    selected = prefix + "x" * (length - len(prefix) - len(suffix)) + suffix
    writes, opened = [], []

    async def save(title, content):
        writes.append((title, content))
        return "note-123"

    controller, jobs, _, _ = trace_controller(
        NoteStore(),
        save_selection_note=save,
        open_selection_note=lambda note_id: opened.append(note_id) or True,
    )
    controller.request_selection_note(
        selected, source_session_id="source", open_note=open_note
    )
    await asyncio.gather(*jobs)
    assert len(writes) == 1
    assert writes[0][1].startswith(selected + "\n\n— Console selection, Source chat, ")
    assert len(writes[0][0]) <= 48
    assert opened == (["note-123"] if open_note else [])


@pytest.mark.asyncio
async def test_note_double_activation_and_cancel_do_not_repeat_issued_write():
    entered, release = asyncio.Event(), asyncio.Event()
    writes = []

    async def save(title, content):
        writes.append(content)
        entered.set()
        await release.wait()
        return "created"

    controller, jobs, _, _ = trace_controller(NoteStore(), save_selection_note=save)
    controller.request_selection_note("selected", source_session_id="source")
    await entered.wait()
    jobs[0].cancel()
    await asyncio.sleep(0)
    controller.request_selection_note("selected", source_session_id="source")
    release.set()
    await asyncio.gather(*jobs, return_exceptions=True)
    assert len(writes) == 1


@pytest.mark.asyncio
async def test_failed_save_never_opens_and_temporary_chat_never_writes():
    opened, writes = [], []

    async def fail(title, content):
        writes.append(content)
        raise RuntimeError("failed write")

    for ephemeral in (False, True):
        controller, jobs, _, notices = trace_controller(
            NoteStore(ephemeral=ephemeral),
            save_selection_note=fail,
            open_selection_note=lambda note_id: opened.append(note_id) or True,
        )
        controller.request_selection_note(
            "selected", source_session_id="source", open_note=True
        )
        await asyncio.gather(*jobs)
        assert notices
    assert opened == []
    assert len(writes) == 1


@pytest.mark.asyncio
async def test_saved_note_handoff_failure_offers_open_without_creating_again():
    writes, receipts = [], []

    async def save(title, content):
        writes.append(content)
        return "saved-note"

    async def offer(note_id):
        receipts.append(note_id)

    controller, jobs, _, _ = trace_controller(
        NoteStore(),
        save_selection_note=save,
        open_selection_note=lambda note_id: False,
        offer_selection_note_open=offer,
    )
    controller.request_selection_note(
        "selected", source_session_id="source", open_note=True
    )
    await asyncio.gather(*jobs)
    assert len(writes) == 1 and receipts == ["saved-note"]


@pytest.mark.asyncio
async def test_failed_handoff_receipt_does_not_report_failed_creation():
    async def save(title, content):
        return "saved-note"

    async def fail_receipt(note_id):
        raise RuntimeError("view was closed")

    controller, jobs, _, notices = trace_controller(
        NoteStore(),
        save_selection_note=save,
        open_selection_note=lambda note_id: False,
        offer_selection_note_open=fail_receipt,
    )
    controller.request_selection_note(
        "selected", source_session_id="source", open_note=True
    )
    await asyncio.gather(*jobs)
    assert not any("Could not create" in notice for notice in notices)
    assert "saved-note" in notices[-1]


@pytest.mark.asyncio
async def test_navigation_completion_failure_retries_only_the_saved_note():
    from tldw_chatbook.Constants import LIBRARY_NAV_CONTEXT_NOTE_ID, TAB_LIBRARY
    from tldw_chatbook.UI.Console_Modules.message import ConsoleMessageController

    events, offered, jobs, notices = [], [], [], []
    store = SimpleNamespace(active_session_id="source")

    async def offer(note_id):
        offered.append(note_id)

    host = SimpleNamespace(
        app_instance=SimpleNamespace(
            post_message=lambda event: events.append(event) or True,
            notify=lambda message, **_kwargs: notices.append(message),
        ),
        _ensure_console_chat_store=lambda: store,
        _offer_captured_note_handoff=offer,
        run_worker=lambda work, **_kwargs: jobs.append(asyncio.create_task(work)),
    )
    assert ConsoleMessageController.open_console_selection_note(host, "saved-note")
    assert events[0].screen_name == TAB_LIBRARY
    assert events[0].screen_context == {LIBRARY_NAV_CONTEXT_NOTE_ID: "saved-note"}
    events[0].report_completion(False)
    await asyncio.gather(*jobs)
    assert offered == ["saved-note"]
    assert notices


@pytest.mark.asyncio
async def test_tab_switch_during_save_keeps_source_provenance_and_does_not_navigate():
    entered, release = asyncio.Event(), asyncio.Event()
    opened, writes = [], []

    async def save(title, content):
        writes.append(content)
        entered.set()
        await release.wait()
        return "saved-note"

    store = NoteStore()
    controller, jobs, _, notices = trace_controller(
        store,
        save_selection_note=save,
        open_selection_note=lambda note_id: opened.append(note_id) or True,
    )
    controller.request_selection_note(
        "selected", source_session_id="source", open_note=True
    )
    await entered.wait()
    store.active_session_id = "other"
    release.set()
    await asyncio.gather(*jobs)
    assert "Source chat" in writes[0] and "Other chat" not in writes[0]
    assert opened == []
    assert any("saved-note" in notice for notice in notices)


@pytest.mark.asyncio
async def test_real_notes_service_saves_long_selection_on_worker(tmp_path):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Notes.Notes_Library import NotesInteropService
    from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
    from tldw_chatbook.UI.Console_Modules.message import ConsoleMessageController

    database = CharactersRAGDB(str(tmp_path / "notes.sqlite"), "notes-test")
    local = NotesInteropService(tmp_path, "notes-test", global_db_to_use=database)
    service = NotesScopeService(local, None)
    host = SimpleNamespace(
        app_instance=SimpleNamespace(notes_scope_service=service),
        _console_notes_owner_id=lambda: "notes-test",
    )
    owner_thread = threading.get_ident()
    threads = []
    entered, release = threading.Event(), threading.Event()
    original = local.add_note

    def record(*args, **kwargs):
        threads.append(threading.get_ident())
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    local.add_note = record
    selected = "long retained message " * 6000
    connections_before = database.registered_connection_count()
    try:
        saving = asyncio.create_task(
            ConsoleMessageController.save_console_selection_note(
                host, "Full message", selected
            )
        )
        assert await asyncio.to_thread(entered.wait, 5)
        heartbeat = asyncio.Event()
        asyncio.get_running_loop().call_soon(heartbeat.set)
        await asyncio.wait_for(heartbeat.wait(), 1)
        assert not saving.done()
        release.set()
        note_id = await saving
        assert note_id
        assert database.get_note_by_id(note_id)["content"] == selected
        assert threads and all(thread != owner_thread for thread in threads)
        assert all(
            db.registered_connection_count() == connections_before
            for db in local._db_instances.values()
        )
    finally:
        release.set()
        local.close_all_user_connections()
        database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("go", [False, True])
async def test_note_buttons_save_full_message_selected_in_menu(go):
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
    )
    from Tests.UI.test_console_trace_copy import TraceCopyApp, open_trace_menu

    body = "full selected body\n" * 800 + "LAST CHARACTERS"

    class NoteCaptureApp(TraceCopyApp):
        def __init__(self):
            super().__init__(
                [ConsoleChatMessage(ConsoleMessageRole.USER, body, id="user")]
            )
            self.writes, self.opened = [], []
            self.controller._save_selection_note = self.save
            self.controller._open_selection_note = (
                lambda note_id: self.opened.append(note_id) or True
            )

        async def save(self, title, content):
            self.writes.append(content)
            return "exact-created-note"

        def on_console_selection_note_requested(self, event):
            self.controller.request_selection_note(
                event.quote,
                source_session_id=event.source_session_id,
                open_note=event.open_note,
            )

    app = NoteCaptureApp()
    async with app.run_test(size=(100, 40)) as pilot:
        await open_trace_menu(app, pilot, "user")
        await pilot.click("#console-selection-full-message")
        await pilot.pause()
        button = (
            "#console-selection-create-note-go"
            if go
            else "#console-selection-create-note"
        )
        await pilot.click(button)
        await pilot.pause()
        await asyncio.gather(*app.jobs)
        assert len(app.writes) == 1 and app.writes[0].startswith(body + "\n\n")
        assert app.opened == (["exact-created-note"] if go else [])


@pytest.mark.asyncio
async def test_create_and_go_arrives_at_the_created_note_in_real_library(tmp_path):
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.library_quit_guard_support import _NotesProfile, _armed_editor, _until
    from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
    from Tests.UI.test_library_shell import _seed_conversations
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
    from tldw_chatbook.Widgets.Console.console_selection_menu import (
        ConsoleSelectionNoteRequested,
    )

    class NoteNavigationHarness(ConsoleHarness):
        async def on_navigate_to_screen(self, event):
            library = LibraryScreen(self.app_instance)
            library.apply_navigation_context(event.screen_context)
            await self.switch_screen(library)
            event.report_completion(True)

    app = _build_test_app(configured_default="chat")
    _configure_native_ready_console(app)
    _seed_conversations(app, [])
    profile = _NotesProfile(tmp_path)
    app.chachanotes_db = profile.db
    app.notes_scope_service = profile.scope_service
    app.notes_service = profile.interop
    host = NoteNavigationHarness(app)
    app.post_message = host.post_message
    selected = "whole response café λ\n" * 600 + "FINAL SELECTED CHARACTERS  "
    try:
        async with host.run_test(size=(160, 45)) as pilot:
            await _until(pilot, lambda: isinstance(host.screen, ChatScreen), "Console")
            console = host.screen
            await _until(
                pilot,
                lambda: bool(console.query("#console-native-composer")),
                "interactive Console",
            )
            store = console._ensure_console_chat_store()
            session = store.ensure_session(title="Source for Notes")
            source_title = session.title
            console.post_message(
                ConsoleSelectionNoteRequested(
                    selected,
                    source_session_id=session.id,
                    open_note=True,
                )
            )
            await _until(
                pilot, lambda: isinstance(host.screen, LibraryScreen), "Library arrival"
            )
            rows = [
                row
                for row in profile.db.list_notes(limit=10)
                if row["content"].startswith(selected)
            ]
            assert len(rows) == 1
            await _armed_editor(host.screen, pilot, rows[0]["id"])
            assert rows[0]["content"].startswith(
                selected + f"\n\n— Console selection, {source_title}, "
            )
            assert (
                host.screen.query_one("#library-note-body").text == rows[0]["content"]
            )
    finally:
        profile.interop.close_all_user_connections()
        profile.db.close()
