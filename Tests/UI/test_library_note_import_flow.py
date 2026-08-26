"""Retained-shell integration for reviewed one-time Database Notes import."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _two_notes,
    _wait_for_library_shell,
    _wait_for_condition,
    _wait_for_selector,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.Library.library_note_import_state import (
    NoteImportPhase,
    initial_note_import_snapshot,
)
from tldw_chatbook.Notes.note_import_execution_models import (
    ImportExecutionReceipt,
    ImportSessionState,
)
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library import LibraryNotesCanvas
from tldw_chatbook.Widgets.Library.library_note_import_canvas import (
    LibraryNoteImportCanvas,
)


pytestmark = pytest.mark.asyncio


def _resolve_picker_immediately(screen, selected_path: Path | None) -> list[object]:
    dialogs: list[object] = []

    def push_screen(dialog, callback=None):
        dialogs.append(dialog)
        if callback is not None:
            screen.run_worker(callback(selected_path))
        return None

    screen.app.push_screen = push_screen
    return dialogs


async def _open_import_once(screen, pilot, selected_path: Path | None) -> list[object]:
    """Enter Import once through the shipped Add-from-files authority chooser."""
    await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
    dialogs = _resolve_picker_immediately(screen, selected_path)
    screen.query_one("#library-notes-add-from-files").press()
    await _wait_for_selector(screen, pilot, "#notes-add-import-once")
    screen.query_one("#notes-add-import-once").press()
    await _wait_for_selector(
        screen,
        pilot,
        "#note-import-destination"
        if selected_path is not None
        else "#library-notes-import-back",
    )
    return dialogs


async def test_picker_file_enters_destination_without_immediate_note_mutation(
    tmp_path: Path,
) -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)
    source = tmp_path / "review-me.md"
    source.write_text("# Review me\nBody", encoding="utf-8")

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
        canvas = screen.query_one("#library-notes-canvas", LibraryNotesCanvas)
        dialogs = await _open_import_once(screen, pilot, source)

        assert dialogs
        assert screen.query_one("#library-notes-canvas") is canvas
        assert screen._library_notes_view == "import"
        assert screen._library_note_import_controller.snapshot.selected_paths == (
            source,
        )
        assert app.notes_scope_service.save_calls == []


async def test_import_back_retains_canvas_and_shows_truthful_lasting_availability() -> (
    None
):
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
        canvas = screen.query_one("#library-notes-canvas", LibraryNotesCanvas)
        await _open_import_once(screen, pilot, None)
        await _wait_for_selector(screen, pilot, "#library-notes-import-back")
        screen.query_one("#library-notes-import-back").press()
        await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")

        assert screen.query_one("#library-notes-canvas") is canvas
        screen.query_one("#library-notes-add-from-files").press()
        await _wait_for_selector(screen, pilot, "#notes-add-keep-synced")
        keep_synced = screen.query_one("#notes-add-keep-synced", Button)
        keep_synced.press()
        await pilot.pause()
        assert screen._library_notes_sync_controller.snapshot.phase == "choose"
        assert "unavailable" in (
            screen._library_notes_sync_controller.snapshot.status_line.casefold()
        )
        assert screen.query_one("#notes-add-import-once", Button).disabled is False
        assert screen._library_notes_view == "lasting_add"


async def test_hidden_import_snapshot_is_retained_without_dom_sync() -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
        hidden = replace(
            screen._library_note_import_snapshot,
            phase="receipt",
            status_line="Import finished.",
            receipt_line="1 imported · 0 updated · 0 skipped · 0 failed",
            receipt_detail="All planned items settled.",
        )
        screen._library_notes_view = "list"

        with patch.object(library_screen_module, "_sync_library_canvas") as sync:
            screen._publish_library_note_import_snapshot(hidden)

        assert screen._library_note_import_snapshot is hidden
        sync.assert_not_called()


async def test_receipt_back_to_list_can_reopen_the_exact_same_session_receipt() -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)
    receipt = ImportExecutionReceipt(
        approval_id=str(uuid4()),
        state=ImportSessionState.COMPLETED,
        total=1,
        completed=1,
        imported=1,
        updated=0,
        skipped=0,
        failed=0,
        retryable=0,
    )

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
        controller = screen._library_note_import_controller
        controller._state = initial_note_import_snapshot(latest_receipt=receipt)
        controller.revisit_receipt()
        screen._library_notes_view = "import"
        library_screen_module._sync_library_canvas(screen, "notes")
        await _wait_for_selector(screen, pilot, "#library-notes-import-back")

        screen.query_one("#library-notes-import-back").press()
        await _wait_for_selector(screen, pilot, "#library-notes-import-receipt")
        screen.query_one("#library-notes-import-receipt").press()
        await _wait_for_selector(screen, pilot, "#note-import-receipt")

        assert controller.snapshot.receipt is receipt
        assert controller.snapshot.latest_receipt is receipt


async def test_back_during_import_offers_view_and_reopens_same_progress() -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
        controller = screen._library_note_import_controller
        active = replace(controller.snapshot, phase=NoteImportPhase.IMPORTING)
        controller._state = active
        controller.publish()
        screen._library_notes_view = "import"
        library_screen_module._sync_library_canvas(screen, "notes")
        await _wait_for_selector(screen, pilot, "#library-notes-import-back")

        screen.query_one("#library-notes-import-back").press()
        await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
        reopen = screen.query_one("#library-notes-add-from-files")
        assert reopen.label.plain == "View import"

        begin_selection = Mock()
        picker = Mock()
        controller.begin_selection = begin_selection
        screen._push_library_note_import_picker = picker
        reopen.press()
        await _wait_for_selector(screen, pilot, "#note-import-cancel")

        assert controller.snapshot is active
        assert screen.focused is screen.query_one("#note-import-cancel")
        begin_selection.assert_not_called()
        picker.assert_not_called()


async def test_screen_unmount_signals_import_cancel_before_owner_teardown() -> None:
    source = inspect.getsource(LibraryScreen.on_unmount)
    assert source.index("_library_note_import_controller.cancel()") < source.index(
        "super().on_unmount()"
    )


async def test_double_import_handler_activation_admits_only_one_worker() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def admitted_execution() -> None:
        started.set()
        await release.wait()

    admitted = admitted_execution()
    controller = SimpleNamespace(
        admit_execution=Mock(side_effect=(admitted, None)),
    )
    tasks: list[asyncio.Task[None]] = []

    def run_worker(coroutine, **kwargs):
        tasks.append(asyncio.create_task(coroutine))

    screen = SimpleNamespace(
        _library_note_import_controller=controller,
        _run_library_note_import_execution=(
            lambda execution: LibraryScreen._run_library_note_import_execution(
                screen, execution
            )
        ),
        _notify_library_note_import_failure=Mock(),
        run_worker=run_worker,
    )
    first = LibraryNoteImportCanvas.ImportRequested()
    second = LibraryNoteImportCanvas.ImportRequested()

    LibraryScreen.handle_library_note_import_execute(screen, first)
    LibraryScreen.handle_library_note_import_execute(screen, second)
    await started.wait()

    assert controller.admit_execution.call_count == 2
    assert len(tasks) == 1
    release.set()
    await tasks[0]


async def test_last_receipt_is_hidden_while_a_new_selection_is_active() -> None:
    receipt = ImportExecutionReceipt(
        approval_id=str(uuid4()),
        state=ImportSessionState.COMPLETED,
        total=1,
        completed=1,
        imported=1,
        updated=0,
        skipped=0,
        failed=0,
        retryable=0,
    )
    idle = initial_note_import_snapshot(latest_receipt=receipt)
    selecting = replace(
        idle,
        phase=NoteImportPhase.DESTINATION,
        selected_paths=(Path("next.md"),),
    )

    assert idle.can_revisit_receipt is True
    assert selecting.can_revisit_receipt is False


async def test_real_file_backed_screen_check_is_read_only_then_import_refreshes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notes_path = tmp_path / "notes.sqlite"
    receipt_path = tmp_path / "import-receipts.sqlite"
    source = tmp_path / "review-me.md"
    source.write_text("# Review me\n\nBody", encoding="utf-8")
    database = CharactersRAGDB(notes_path, client_id="library-import-e2e")
    folders = LocalNoteFolderRepository(database)
    interop = NotesInteropService(
        base_db_directory=tmp_path,
        api_client_id="library-import-e2e",
        global_db_to_use=database,
    )
    scope_service = NotesScopeService(
        local_notes_service=interop,
        server_service=None,
        folder_repository=folders,
    )
    monkeypatch.setattr(
        library_screen_module,
        "get_notes_sync_state_db_path",
        lambda: receipt_path,
    )
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=[])
    app.chachanotes_db = database
    app.notes_scope_service = scope_service
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            screen.query_one("#library-row-browse-notes").press()
            await _open_import_once(screen, pilot, source)
            screen.query_one("#note-import-destination").value = "Inbox"
            await pilot.pause()

            schema_before = tuple(
                database.get_connection()
                .execute(
                    "SELECT type, name, tbl_name, sql FROM sqlite_master "
                    "ORDER BY type, name"
                )
                .fetchall()
            )
            version_before = (
                database.get_connection().execute("PRAGMA user_version").fetchone()[0]
            )
            assert database.count_notes() == 0
            assert folders.get_folder_by_path(("Inbox",)) is None
            assert not receipt_path.exists()

            screen.query_one("#note-import-check").press()
            await _wait_for_selector(screen, pilot, "#note-import-import")

            schema_after_check = tuple(
                database.get_connection()
                .execute(
                    "SELECT type, name, tbl_name, sql FROM sqlite_master "
                    "ORDER BY type, name"
                )
                .fetchall()
            )
            assert schema_after_check == schema_before
            assert (
                database.get_connection().execute("PRAGMA user_version").fetchone()[0]
                == version_before
            )
            assert database.count_notes() == 0
            assert folders.get_folder_by_path(("Inbox",)) is None
            assert not receipt_path.exists()

            screen.query_one("#note-import-import").press()
            await _wait_for_selector(screen, pilot, "#note-import-receipt")
            await _wait_for_condition(
                pilot,
                lambda: screen._local_source_counts.get("notes") == 1,
                message="Imported note never refreshed the local Notes count.",
            )
            await _wait_for_condition(
                pilot,
                lambda: (
                    (projection := screen._build_library_notes_tree_projection())
                    is not None
                    and any(row.label == "Inbox" for row in projection.rows)
                ),
                message="Imported folder never refreshed the Notes tree.",
            )

            assert database.count_notes() == 1
            assert folders.get_folder_by_path(("Inbox",)) is not None
            assert receipt_path.exists()
            assert screen._library_note_import_controller.snapshot.receipt.imported == 1
    finally:
        interop.close_all_user_connections()
        database.close_connection()


async def test_hidden_import_fences_notes_mutations_until_receipt(
    tmp_path: Path,
) -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)
    receipt = ImportExecutionReceipt(
        approval_id=str(uuid4()),
        state=ImportSessionState.COMPLETED,
        total=1,
        completed=1,
        imported=1,
        updated=0,
        skipped=0,
        failed=0,
        retryable=0,
    )

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
        controller = screen._library_note_import_controller
        active = replace(controller.snapshot, phase=NoteImportPhase.IMPORTING)
        controller._state = active
        controller.publish()
        screen._library_notes_view = "import"
        library_screen_module._sync_library_canvas(screen, "notes")
        await _wait_for_selector(screen, pilot, "#library-notes-import-back")
        screen.query_one("#library-notes-import-back").press()
        await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")

        for selector in (
            "#library-notes-new",
            "#library-notes-sort",
            "#library-notes-select-toggle",
            "#library-notes-export",
        ):
            assert screen.query_one(selector, Button).disabled is True
        assert all(button.disabled for button in screen.query(".library-notes-row"))
        view_import = screen.query_one("#library-notes-add-from-files", Button)
        assert view_import.disabled is False
        assert view_import.label.plain == "View import"
        assert screen._begin_library_notes_operation("export") is None
        with patch.object(screen, "run_worker") as run_worker:
            screen._schedule_library_notes_tree_mutation(
                "create_folder", name="Blocked", parent_id=None
            )
        run_worker.assert_not_called()
        assert screen._begin_library_note_create() is None

        view_import.press()
        await _wait_for_selector(screen, pilot, "#note-import-cancel")
        assert controller.snapshot is active
        screen.query_one("#library-notes-import-back").press()
        await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")

        settled = replace(
            active,
            phase=NoteImportPhase.RECEIPT,
            receipt=receipt,
            latest_receipt=receipt,
            cancel_requested=False,
        )
        controller._state = settled
        controller.publish()
        library_screen_module._sync_library_canvas(screen, "notes")
        await pilot.pause()

        for selector in (
            "#library-notes-new",
            "#library-notes-sort",
            "#library-notes-select-toggle",
            "#library-notes-export",
        ):
            assert screen.query_one(selector, Button).disabled is False
        assert all(not button.disabled for button in screen.query(".library-notes-row"))
        screen.query_one("#library-notes-add-from-files").press()
        await _wait_for_selector(screen, pilot, "#notes-add-import-once")
        assert screen._library_notes_view == "lasting_add"
