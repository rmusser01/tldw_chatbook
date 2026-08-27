"""Mounted Database Notes journeys through the shared adaptive reader."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import fields
from typing import get_args

import pytest
from textual.widgets import Button, Static, TextArea

from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _open_note_editor,
    _press_note_back,
    _seed_conversations,
    _two_conversations,
    _two_notes,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
import tldw_chatbook.UI.Screens.library_screen as library_screen_module
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.UI.Library_Modules.library_notes_work_session import (
    NotesWorkSessionEvent,
    NotesWorkSessionPhase,
)
from tldw_chatbook.Widgets.Library import (
    LibraryAdaptiveReaderShell,
    LibraryNoteWorkPane,
    LibraryNotesCanvas,
)


def test_folder_files_reader_authority_scaffold_is_distinct() -> None:
    app = _build_test_app()
    library = app.app_config.setdefault("library", {})
    library["reader"] = {
        "library_open": False,
        "custom_widths_enabled": True,
        "library_width": 31,
    }
    library["notes_reader"] = {
        "items_open": False,
        "items_width": 33,
        "files_tree_open": True,
        "files_tree_width": 44,
    }

    screen = LibraryScreen(app)

    assert set(get_args(library_screen_module.LibraryReaderDestination)) == {
        "media",
        "conversations",
        "notes",
        "notes_files",
    }
    assert tuple(
        field.name
        for field in fields(library_screen_module._LibraryReaderPersistenceTarget)
    ) == ("section", "config_key", "authority", "preferences_attribute")
    assert library_screen_module._LIBRARY_READER_PERSISTENCE_TARGETS[
        ("notes_files", "items")
    ] == library_screen_module._LibraryReaderPersistenceTarget(
        section="library.notes_reader",
        config_key="files_tree_open",
        authority="notes_file_items",
        preferences_attribute="_library_file_notes_reader_preferences",
    )

    database = screen._library_notes_reader_preferences
    folder = screen._library_file_notes_reader_preferences
    assert database is not folder
    assert database.library_open is folder.library_open is False
    assert database.library_width == folder.library_width == 31
    assert (database.items_open, database.items_width) == (False, 33)
    assert (folder.items_open, folder.items_width) == (True, 44)
    assert screen._library_notes_reader_layout is not (
        screen._library_file_notes_reader_layout
    )
    assert screen._library_reader_durable_preferences["notes_items"] is False
    assert screen._library_reader_durable_preferences["notes_file_items"] is True
    assert "notes_file_items" in screen._library_reader_persistence_generations
    assert "notes_file_items" in screen._library_reader_durable_generations
    assert (
        screen._library_notes_reader_persistence_locks["library"]
        is screen._library_file_notes_reader_persistence_locks["library"]
    )
    assert (
        screen._library_notes_reader_persistence_locks["items"]
        is not screen._library_file_notes_reader_persistence_locks["items"]
    )


@pytest.mark.asyncio
async def test_notes_reader_persistence_authorities_reconcile_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Database and Folder Items writes cannot supersede or roll back each other."""
    app = _build_test_app()
    app.app_config.setdefault("library", {}).setdefault("notes_reader", {}).update(
        {
            "items_open": True,
            "items_width": 33,
            "files_tree_open": True,
            "files_tree_width": 44,
        }
    )
    notifications: list[tuple[str, dict[str, object]]] = []
    app.notify = lambda message, **kwargs: notifications.append((message, kwargs))
    screen = LibraryScreen(app)
    writes: list[tuple[str, str, bool]] = []

    def save(section: str, key: str, value: bool) -> bool:
        writes.append((section, key, value))
        return key != "items_open"

    monkeypatch.setattr(library_screen_module, "save_setting_to_cli_config", save)

    database_generation = screen._claim_library_reader_persistence("notes", "items")
    screen._replace_library_reader_preference("notes", "items_open", False)
    screen._mirror_library_notes_reader_preference("items_open", False)
    assert screen._library_notes_reader_preferences.items_open is False
    assert app.app_config["library"]["notes_reader"]["items_open"] is False

    folder_generation = screen._claim_library_reader_persistence("notes_files", "items")
    screen._replace_library_reader_preference("notes_files", "items_open", False)
    screen._mirror_library_file_notes_reader_preference("items_open", False)
    assert screen._library_file_notes_reader_preferences.items_open is False
    assert app.app_config["library"]["notes_reader"]["files_tree_open"] is False

    await screen._persist_library_reader_preference(
        "notes_files", "items", False, folder_generation
    )
    await screen._persist_library_reader_preference(
        "notes", "items", False, database_generation
    )

    assert writes == [
        ("library.notes_reader", "files_tree_open", False),
        ("library.notes_reader", "items_open", False),
    ]
    assert screen._library_file_notes_reader_preferences.items_open is False
    assert screen._library_notes_reader_preferences.items_open is True
    assert app.app_config["library"]["notes_reader"] == {
        "items_open": True,
        "items_width": 33,
        "files_tree_open": False,
        "files_tree_width": 44,
    }
    assert all(key not in {"items_width", "files_tree_width"} for _, key, _ in writes)
    assert notifications == [
        (
            "Library reader layout could not be saved; the previous pane choice was restored.",
            {"severity": "warning"},
        )
    ]

    notifications.clear()
    monkeypatch.setattr(
        screen,
        "_read_library_reader_persisted_preference",
        lambda *_args, **_kwargs: asyncio.sleep(0, result=None),
    )
    verify_generation = screen._claim_library_reader_persistence("notes", "items")
    screen._replace_library_reader_preference("notes", "items_open", False)
    screen._mirror_library_notes_reader_preference("items_open", False)
    await screen._persist_library_reader_preference(
        "notes",
        "items",
        False,
        verify_generation,
        verify_failure_from_config=True,
    )

    assert screen._library_notes_reader_preferences.items_open is False
    assert screen._library_file_notes_reader_preferences.items_open is False
    assert notifications == [
        (
            "Library reader layout could not be verified from configuration; the current pane choice was kept.",
            {"severity": "warning"},
        )
    ]


@pytest.mark.asyncio
async def test_notes_reader_opposite_generation_races_stay_authority_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opposite completion order converges each Items authority independently."""
    app = _build_test_app()
    app.app_config.setdefault("library", {}).setdefault("notes_reader", {}).update(
        {"items_open": True, "files_tree_open": True}
    )
    screen = LibraryScreen(app)
    notes_reader = app.app_config["library"]["notes_reader"]
    initial_items_width = notes_reader["items_width"]
    initial_files_tree_width = notes_reader["files_tree_width"]
    database_started = threading.Event()
    folder_started = threading.Event()
    release_database = threading.Event()
    release_folder = threading.Event()
    writes: list[tuple[str, bool]] = []

    def save(_section: str, key: str, value: bool) -> bool:
        writes.append((key, value))
        if value is False and key == "items_open":
            database_started.set()
            assert release_database.wait(5)
        elif value is False and key == "files_tree_open":
            folder_started.set()
            assert release_folder.wait(5)
        return True

    monkeypatch.setattr(library_screen_module, "save_setting_to_cli_config", save)

    database_old = screen._claim_library_reader_persistence("notes", "items")
    screen._replace_library_reader_preference("notes", "items_open", False)
    screen._mirror_library_notes_reader_preference("items_open", False)
    folder_old = screen._claim_library_reader_persistence("notes_files", "items")
    screen._replace_library_reader_preference("notes_files", "items_open", False)
    screen._mirror_library_file_notes_reader_preference("items_open", False)
    database_task = asyncio.create_task(
        screen._persist_library_reader_preference("notes", "items", False, database_old)
    )
    folder_task = asyncio.create_task(
        screen._persist_library_reader_preference(
            "notes_files", "items", False, folder_old
        )
    )
    assert await asyncio.to_thread(database_started.wait, 5)
    assert await asyncio.to_thread(folder_started.wait, 5)

    database_new = screen._claim_library_reader_persistence("notes", "items")
    screen._replace_library_reader_preference("notes", "items_open", True)
    screen._mirror_library_notes_reader_preference("items_open", True)
    folder_new = screen._claim_library_reader_persistence("notes_files", "items")
    screen._replace_library_reader_preference("notes_files", "items_open", True)
    screen._mirror_library_file_notes_reader_preference("items_open", True)
    database_new_task = asyncio.create_task(
        screen._persist_library_reader_preference("notes", "items", True, database_new)
    )
    folder_new_task = asyncio.create_task(
        screen._persist_library_reader_preference(
            "notes_files", "items", True, folder_new
        )
    )

    release_folder.set()
    await folder_task
    await folder_new_task
    assert screen._library_reader_durable_preferences["notes_file_items"] is True
    assert screen._library_reader_durable_preferences["notes_items"] is True

    release_database.set()
    await database_task
    await database_new_task

    assert set(writes[:2]) == {
        ("items_open", False),
        ("files_tree_open", False),
    }
    assert writes[2:] == [("files_tree_open", True), ("items_open", True)]
    assert screen._library_notes_reader_preferences.items_open is True
    assert screen._library_file_notes_reader_preferences.items_open is True
    assert screen._library_reader_durable_preferences["notes_items"] is True
    assert screen._library_reader_durable_preferences["notes_file_items"] is True
    assert notes_reader["items_open"] is True
    assert notes_reader["files_tree_open"] is True
    assert notes_reader["items_width"] == initial_items_width
    assert notes_reader["files_tree_width"] == initial_files_tree_width


@pytest.mark.asyncio
async def test_database_notes_work_session_activates_once_and_resets_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_test_app()
    app.app_config.setdefault("library", {}).setdefault("reader", {})[
        "library_open"
    ] = True
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    writes: list[tuple[str, str, bool]] = []
    monkeypatch.setattr(
        library_screen_module,
        "save_setting_to_cli_config",
        lambda section, key, value: (writes.append((section, key, value)), True)[1],
    )
    host = LibraryHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        shell = screen.query_one(
            "#library-notes-reader-shell", LibraryAdaptiveReaderShell
        )

        assert (
            screen._library_notes_work_session_phase is NotesWorkSessionPhase.INACTIVE
        )
        assert shell.effective_layout.library_open is True
        assert writes == []

        screen.query_one("#library-notes-row-0", Button).press()
        await _wait_for_selector(screen, pilot, "#library-note-body")
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_notes_work_session_phase is NotesWorkSessionPhase.ACTIVE
                and not shell.effective_layout.library_open
            ),
            message="Database editable open did not activate work-first layout",
        )
        assert screen._library_notes_reader_preferences.library_open is True
        assert writes == []

        shell.library_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_notes_work_session_phase
                is NotesWorkSessionPhase.MANUALLY_CANCELLED
                and shell.effective_layout.library_open
            ),
            message="Saved-open manual expansion did not cancel work-first",
        )
        assert writes == []

        for event in (
            NotesWorkSessionEvent.SELECTION_CHANGED,
            NotesWorkSessionEvent.ITEM_CHANGED,
            NotesWorkSessionEvent.EDIT_MODE_CHANGED,
            NotesWorkSessionEvent.PREVIEW_MODE_CHANGED,
            NotesWorkSessionEvent.INFO_MODE_CHANGED,
            NotesWorkSessionEvent.MANAGE_MODE_CHANGED,
            NotesWorkSessionEvent.SAVE,
            NotesWorkSessionEvent.CONFLICT,
            NotesWorkSessionEvent.RECOVERY,
            NotesWorkSessionEvent.RESIZE,
        ):
            screen._dispatch_library_notes_work_session(event)
        assert (
            screen._library_notes_work_session_phase
            is NotesWorkSessionPhase.MANUALLY_CANCELLED
        )
        assert shell.effective_layout.library_open is True
        assert writes == []

        _press_note_back(screen)
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_notes_work_session_phase
                is NotesWorkSessionPhase.INACTIVE
            ),
            message="Database identity clear did not reset work session",
        )
        shell.library_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: writes == [("library.reader", "library_open", False)],
            message="Explicit saved-closed Library request did not persist",
        )
        assert screen._library_notes_reader_preferences.library_open is False

        writes.clear()
        screen.query_one("#library-notes-row-1", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_notes_work_session_phase is NotesWorkSessionPhase.ACTIVE
            ),
            message="A new Database work session did not activate",
        )
        assert writes == []
        shell.library_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: writes == [("library.reader", "library_open", True)],
            message="Saved-closed manual expansion did not persist exactly once",
        )
        assert (
            screen._library_notes_work_session_phase
            is NotesWorkSessionPhase.MANUALLY_CANCELLED
        )
        assert screen._library_notes_reader_preferences.library_open is True

        await screen._select_library_rail_row("browse-media")
        assert (
            screen._library_notes_work_session_phase is NotesWorkSessionPhase.INACTIVE
        )


@pytest.mark.asyncio
async def test_database_notes_mount_three_retained_roles_once() -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-reader-shell")

        shell = screen.query_one(
            "#library-notes-reader-shell", LibraryAdaptiveReaderShell
        )
        rail = shell.query_one("#library-rail")
        items = shell.query_one("#library-notes-canvas", LibraryNotesCanvas)
        work = shell.query_one("#library-note-work-pane", LibraryNoteWorkPane)
        identities = (id(shell), id(rail), id(items), id(work))

        shell.library_grip.press()
        await pilot.pause()
        shell.library_grip.press()
        await pilot.pause()
        shell.items_grip.press()
        await pilot.pause()
        shell.items_grip.press()
        await pilot.pause()

        assert (id(shell), id(rail), id(items), id(work)) == identities
        assert shell.work is work and work.is_mounted and work.display
        assert len(shell.query(".library-adaptive-reader-pane-grip")) == 2


@pytest.mark.asyncio
async def test_list_and_work_identity_survive_open_preview_info_and_edit() -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        notes_list = screen.query_one("#library-notes-canvas", LibraryNotesCanvas)
        work = screen.query_one("#library-note-work-pane", LibraryNoteWorkPane)

        screen.query_one("#library-notes-row-0", Button).press()
        await _wait_for_selector(screen, pilot, "#library-note-title")
        body = screen.query_one("#library-note-body", TextArea)
        body.text = "current unsaved preview body"
        await pilot.pause()
        screen.query_one("#library-note-preview", Button).press()
        await pilot.pause()
        screen.query_one("#library-note-context", Button).press()
        await pilot.pause()
        screen.query_one("#library-note-context-back", Button).press()
        await pilot.pause()

        assert screen.query_one("#library-notes-canvas") is notes_list
        assert screen.query_one("#library-note-work-pane") is work
        assert screen.query_one("#library-note-body") is body
        assert (
            "current unsaved preview body"
            in screen.query_one("#library-note-preview-body").source
        )


@pytest.mark.asyncio
async def test_work_pane_focus_is_classified_as_notes_stage() -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        await _open_note_editor(screen, pilot)
        body = screen.query_one("#library-note-body", TextArea)
        screen._library_notes_stage = "rail"
        body.focus()
        await pilot.pause()

        identity = screen._capture_library_notes_focus_identity(stage_from_focus=True)

        assert identity.stage == "notes"
        assert identity.semantic_role == "body"


@pytest.mark.asyncio
async def test_create_replaces_only_work_content_and_keeps_list_mounted() -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-new")
        notes_list = screen.query_one("#library-notes-canvas", LibraryNotesCanvas)

        screen.query_one("#library-notes-new", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")

        assert screen.query_one("#library-notes-canvas") is notes_list
        assert notes_list.is_mounted and notes_list.display
        assert screen.query_one("#library-notes-create-blank").is_mounted


@pytest.mark.asyncio
async def test_editor_back_preserves_shell_list_and_work_owners() -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        shell = screen.query_one("#library-notes-reader-shell")
        notes_list = screen.query_one("#library-notes-canvas")
        work = screen.query_one("#library-note-work-pane")
        await _open_note_editor(screen, pilot)
        assert screen.query_one("#library-notes-canvas") is notes_list

        screen.query_one("#library-note-back", Button).press()
        await _wait_for_selector(screen, pilot, "#library-note-work-empty")

        assert screen.query_one("#library-notes-reader-shell") is shell
        assert screen.query_one("#library-notes-canvas") is notes_list
        assert screen.query_one("#library-note-work-pane") is work


@pytest.mark.asyncio
async def test_create_back_preserves_shell_list_and_work_owners() -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-new")
        shell = screen.query_one("#library-notes-reader-shell")
        notes_list = screen.query_one("#library-notes-canvas")
        work = screen.query_one("#library-note-work-pane")
        screen.query_one("#library-notes-new", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-create-back")

        screen.query_one("#library-notes-create-back", Button).press()
        await _wait_for_selector(screen, pilot, "#library-note-work-empty")

        assert screen.query_one("#library-notes-reader-shell") is shell
        assert screen.query_one("#library-notes-canvas") is notes_list
        assert screen.query_one("#library-note-work-pane") is work


@pytest.mark.asyncio
async def test_create_success_preserves_shell_list_and_work_owners() -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-new")
        shell = screen.query_one("#library-notes-reader-shell")
        notes_list = screen.query_one("#library-notes-canvas")
        work = screen.query_one("#library-note-work-pane")

        screen.query_one("#library-notes-new", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")
        screen.query_one("#library-notes-create-blank", Button).press()
        await _wait_for_selector(screen, pilot, "#library-note-title")

        assert screen.query_one("#library-notes-reader-shell") is shell
        assert screen.query_one("#library-notes-canvas") is notes_list
        assert screen.query_one("#library-note-work-pane") is work


@pytest.mark.asyncio
async def test_delete_and_receipt_preserve_shell_list_and_work_owners() -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        shell = screen.query_one("#library-notes-reader-shell")
        notes_list = screen.query_one("#library-notes-canvas")
        work = screen.query_one("#library-note-work-pane")
        screen.query_one("#library-notes-row-0", Button).press()
        await _wait_for_selector(screen, pilot, "#library-note-delete")

        screen.query_one("#library-note-delete", Button).press()
        await _wait_for_selector(screen, pilot, "#library-note-delete-confirm")
        screen.query_one("#library-note-delete-confirm", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-delete-receipt-copy")

        assert screen.query_one("#library-notes-reader-shell") is shell
        assert screen.query_one("#library-notes-canvas") is notes_list
        assert screen.query_one("#library-note-work-pane") is work


@pytest.mark.asyncio
async def test_eighty_columns_protect_editor_and_keep_both_restore_grips() -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=(80, 24)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-reader-shell")
        await _open_note_editor(screen, pilot)
        shell = screen.query_one(
            "#library-notes-reader-shell", LibraryAdaptiveReaderShell
        )
        await pilot.pause()

        assert shell.work.region.width >= 48
        assert shell.library_grip.region.width == 5
        assert shell.items_grip.region.width == 5
        assert shell.library_grip.region.x + shell.library_grip.region.width <= 80
        assert shell.items_grip.region.x + shell.items_grip.region.width <= 80


@pytest.mark.asyncio
async def test_wide_editor_deep_link_keeps_reader_navigation_and_local_back() -> None:
    """A first-paint editor uses the adaptive shell, never legacy task mode."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    screen = LibraryScreen(app)
    screen.apply_navigation_context({"note_id": "n-1"})
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-note-body")
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_notes_work_session_phase is NotesWorkSessionPhase.ACTIVE
            ),
            message=lambda: (
                "Deep-linked Database editor did not settle work-first: "
                f"phase={screen._library_notes_work_session_phase!r}, "
                f"pending={screen._library_notes_work_session_activation_pending!r}, "
                f"selected={screen._selected_note_id!r}, "
                f"view={screen._library_notes_view!r}, "
                f"source={screen._library_notes_source!r}, "
                f"snapshot={screen._library_note_session.snapshot is not None!r}, "
                f"reader_width={screen._library_notes_work_session_reader_width()!r}"
            ),
        )

        assert screen.query_one("#library-rail").display is False
        assert screen.query_one("#library-canvas").display is True
        assert screen.query_one("#library-notes-task-return", Button).display is False
        assert screen.query_one("#library-note-back", Button).display is True


@pytest.mark.asyncio
async def test_bulk_mode_keeps_last_note_as_labelled_read_only_preview() -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-notes-select-toggle", Button).press()
        await pilot.pause()

        bulk_status = screen.query_one("#library-note-bulk-status", Static)
        assert bulk_status.display is True
        assert "Read-only preview" in str(bulk_status.renderable)
        assert "Not included" in str(bulk_status.renderable)
        assert screen.query_one("#library-note-preview-region").display is True
        assert screen.query_one("#library-note-editor-region").display is False
        assert screen.query_one("#library-note-back", Button).display is False
        assert screen.check_action("library_notes_save", ()) is False
        assert screen.check_action("library_note_editor_back", ()) is False
        for selector in (
            "#library-note-save",
            "#library-note-context",
            "#library-note-use-in-console",
            "#library-note-export-md",
            "#library-note-copy",
            "#library-note-delete",
        ):
            assert screen.query_one(selector, Button).disabled is True

        loaded_note_id = screen._selected_note_id
        loaded_row = next(
            row
            for row in screen.query(".library-notes-row")
            if getattr(row, "note_id", "") == loaded_note_id
        )
        loaded_row.press()
        await pilot.pause()

        assert str(bulk_status.renderable).endswith("Included in bulk selection")
        assert "Not included" not in str(bulk_status.renderable)

        save_calls = 0

        async def save_note(*, explicit: bool) -> None:
            nonlocal save_calls
            save_calls += 1

        screen._save_library_note = save_note
        await screen.action_library_notes_save()
        assert save_calls == 0
        assert await screen._exit_library_note_editor_guarded() is False
        assert screen._library_notes_select_mode is True
        assert screen._library_notes_view == "editor"

        await screen.action_library_notes_escape()
        await pilot.pause()

        assert screen._library_notes_select_mode is False
        assert screen._library_notes_view == "editor"
