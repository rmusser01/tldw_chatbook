"""Mounted Database Notes journeys through the shared adaptive reader."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import fields, replace
from types import SimpleNamespace
from typing import get_args
from unittest.mock import Mock, patch
import pytest
from textual.widgets import Button, Input, Static, TextArea

from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryGlobalKeyProductionCSSHarness,
    LibraryHarness,
    LibraryProductionCSSHarness,
    _active_library_screen,
    _build_test_app,
    _open_note_editor,
    _press_note_back,
    _seed_conversations,
    _two_conversations,
    _two_notes,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_condition,
    _wait_for_selector,
)
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
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


@pytest.mark.asyncio
async def test_database_notes_capability_inventory_and_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Database Notes keeps its full inventory behind three explicit modes."""
    app = _build_test_app(configured_default="library")
    _seed_conversations(app, _two_conversations(), notes=_two_notes())

    def settings_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return default

    with patch(
        "tldw_chatbook.app.get_cli_setting",
        side_effect=settings_without_splash,
    ):
        async with app.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            await _wait_for_condition(
                pilot,
                lambda: isinstance(app.screen, LibraryScreen),
                message="production app did not mount Library",
            )
            screen = app.screen
            assert isinstance(screen, LibraryScreen)
            await _wait_for_library_shell(screen, pilot)
            screen.query_one("#library-row-browse-notes", Button).press()
            await _wait_for_selector(screen, pilot, "#library-notes-row-0")

            # The navigator capability inventory remains on the incumbent controls.
            for selector in (
                "#library-notes-filter",
                "#library-notes-sort",
                "#library-notes-select-toggle",
                "#library-notes-new",
                "#library-notes-add-from-files",
                "#library-notes-export",
            ):
                assert screen.query_one(selector)
            assert not screen.query("#library-notes-delete-selected")
            filter_input = screen.query_one("#library-notes-filter", Input)
            filter_input.value = "alpha"
            filter_input.focus()
            await pilot.press("enter")
            await _wait_for_selector(screen, pilot, "#library-notes-filter-clear")
            screen.query_one("#library-notes-filter-clear", Button).press()
            await _wait_for_selector(screen, pilot, "#library-notes-row-0")
            screen.query_one("#library-notes-select-toggle", Button).press()
            await _wait_for_selector(screen, pilot, "#library-notes-export-selected")
            export_selected = screen.query_one("#library-notes-export-selected", Button)
            assert export_selected.disabled and export_selected.tooltip
            screen.query_one("#library-notes-select-toggle", Button).press()
            await _wait_for_selector(screen, pilot, "#library-notes-row-0")

            screen.query_one("#library-notes-row-0", Button).press()
            await _wait_for_selector(screen, pilot, "#library-note-body")
            body = screen.query_one("#library-note-body", TextArea)
            body.text = "retained mode draft"
            await pilot.pause()
            selected_title = screen._library_note_session.snapshot.title
            edit_title = screen.query_one("#library-note-editor-title", Static)
            assert edit_title.display
            assert str(edit_title.renderable) == selected_title

            assert tuple(
                str(button.label)
                for button in screen.query_one("#library-note-mode-controls").query(
                    Button
                )
            ) == ("Edit", "Preview", "Info")
            assert screen.query_one("#library-note-edit", Button).has_class("is-active")
            save = screen.query_one("#library-note-save", Button)
            use = screen.query_one("#library-note-use-in-console", Button)
            assert save.display and use.display
            assert use.parent is screen.query_one("#library-note-task-actions")
            title_input = screen.query_one("#library-note-title", Input)
            title_input.focus()
            await pilot.pause()
            assert title_input.has_focus
            await pilot.press("f6")
            await pilot.pause()
            assert save.has_focus
            for _ in range(len(screen.focus_chain) + 1):
                if save.has_focus:
                    break
                await pilot.press("tab")
            assert save.has_focus

            screen.query_one("#library-note-preview", Button).press()
            await pilot.pause()
            assert screen.query_one("#library-note-preview-region").display
            assert screen.query_one("#library-note-preview", Button).has_class(
                "is-active"
            )
            screen.query_one("#library-note-context", Button).press()
            await pilot.pause()
            info = screen.query_one("#library-note-context-region")
            assert info.display
            assert screen.query_one("#library-note-context", Button).has_class(
                "is-active"
            )
            assert tuple(
                str(section.renderable)
                for section in info.query(".destination-section")
            ) == ("Properties", "Reuse & Export", "Danger")
            for selector in (
                "#library-note-context-keywords",
                "#library-note-context-meta",
                "#library-note-context-copy",
                "#library-note-context-export-md",
                "#library-note-context-export-txt",
                "#library-note-context-delete",
            ):
                assert info.query_one(selector)
            screen.query_one("#library-note-edit", Button).press()
            await pilot.pause()
            assert screen.query_one("#library-note-editor-region").display
            assert screen.query_one("#library-note-body", TextArea) is body
            assert body.text == "retained mode draft"

    # Exercise the conditional inventory through the incumbent handlers and
    # assert their service/guard arguments. A method-existence check would let
    # broken event dispatch or a silently changed mutation contract pass.
    with monkeypatch.context() as patcher:
        sync_calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        patcher.setattr(
            library_screen_module,
            "_sync_library_canvas",
            lambda *args, **kwargs: sync_calls.append(("sync", args, kwargs)),
        )

        stopped = Mock()
        event = SimpleNamespace(stop=stopped)
        permitted = SimpleNamespace(
            kind=library_screen_module.NoteFlushOutcomeKind.PERMITTED
        )

        async def flush() -> object:
            return permitted

        refresh_roots = Mock()
        revisit_receipt = Mock()
        task_screen = SimpleNamespace(
            _library_notes_mutation_fenced=lambda: False,
            _flush_library_note_save=flush,
            _library_notes_sync_controller=SimpleNamespace(refresh_roots=refresh_roots),
            _library_note_import_controller=SimpleNamespace(
                revisit_receipt=revisit_receipt
            ),
            _library_notes_lasting_origin="setup",
            _library_notes_view="list",
            _apply_library_notes_footer_context=Mock(),
        )
        await LibraryScreen.handle_library_notes_manage_sync_folders(task_screen, event)
        refresh_roots.assert_called_once_with()
        assert task_screen._library_notes_view == "lasting_roots"
        assert len(sync_calls) == 1
        await LibraryScreen.handle_library_notes_import_receipt(task_screen, event)
        revisit_receipt.assert_called_once_with()
        assert task_screen._library_notes_view == "import"
        assert len(sync_calls) == 2

        mutations: list[tuple[str, dict[str, object]]] = []
        pushed: list[tuple[object, object]] = []
        folder = SimpleNamespace(
            kind="folder",
            protected=False,
            folder_id="folder-1",
            version=7,
            label="Projects",
        )
        note = SimpleNamespace(
            kind="note",
            protected=False,
            folder_id="folder-1",
            note_id="note-1",
            membership_id="member-1",
            version=11,
        )
        tree_screen = SimpleNamespace(
            app=SimpleNamespace(
                push_screen=lambda modal, callback: pushed.append((modal, callback))
            ),
            _selected_library_notes_tree_row=lambda: folder,
            _library_notes_folder_target_options=lambda **kwargs: (),
            _schedule_library_notes_tree_mutation=lambda action, **kwargs: (
                mutations.append((action, kwargs))
            ),
            _library_notes_deleted_folder_receipt=SimpleNamespace(
                folder_id="folder-deleted", expected_version=13
            ),
        )
        LibraryScreen.handle_library_notes_folder_new(tree_screen, event)
        pushed.pop()[1]("Inbox")
        assert mutations.pop() == (
            "create_folder",
            {"name": "Inbox", "parent_id": "folder-1"},
        )
        LibraryScreen.handle_library_notes_folder_rename(tree_screen, event)
        pushed.pop()[1]("Renamed")
        assert mutations.pop() == (
            "rename_folder",
            {
                "folder_id": "folder-1",
                "name": "Renamed",
                "expected_version": 7,
                "protected": False,
            },
        )
        LibraryScreen.handle_library_notes_folder_move(tree_screen, event)
        pushed.pop()[1]("folder-2")
        assert mutations.pop() == (
            "move_folder",
            {
                "folder_id": "folder-1",
                "parent_id": "folder-2",
                "expected_version": 7,
                "protected": False,
            },
        )
        LibraryScreen.handle_library_notes_folder_remove(tree_screen, event)
        pushed.pop()[1](True)
        assert mutations.pop() == (
            "delete_folder",
            {
                "folder_id": "folder-1",
                "expected_version": 7,
                "protected": False,
            },
        )
        LibraryScreen.handle_library_notes_folder_restore(tree_screen, event)
        assert mutations.pop() == (
            "restore_folder",
            {"folder_id": "folder-deleted", "expected_version": 13},
        )

        tree_screen._selected_library_notes_tree_row = lambda: note
        tree_screen._choose_library_notes_placement_target = lambda *, move: (
            LibraryScreen._choose_library_notes_placement_target(tree_screen, move=move)
        )
        LibraryScreen.handle_library_notes_placement_add(tree_screen, event)
        pushed.pop()[1]("folder-2")
        assert mutations.pop() == (
            "add_placement",
            {"folder_id": "folder-2", "note_id": "note-1"},
        )
        LibraryScreen.handle_library_notes_placement_move(tree_screen, event)
        pushed.pop()[1]("folder-2")
        assert mutations.pop() == (
            "move_placement",
            {
                "note_id": "note-1",
                "destination_folder_id": "folder-2",
                "source_folder_id": "folder-1",
                "membership_version": 11,
                "protected": False,
            },
        )
        LibraryScreen.handle_library_notes_placement_remove(tree_screen, event)
        assert mutations.pop() == (
            "detach_placement",
            {
                "folder_id": "folder-1",
                "note_id": "note-1",
                "expected_version": 11,
                "protected": False,
            },
        )

        export_scope = object()
        opened_exports: list[object] = []

        async def open_export(scope: object) -> None:
            opened_exports.append(scope)

        export_screen = SimpleNamespace(
            _library_notes_mutation_fenced=lambda: False,
            _library_notes_row_selection=SimpleNamespace(
                count=1, export_scope=lambda: export_scope
            ),
            _open_library_export_canvas=open_export,
        )
        await LibraryScreen.handle_library_notes_export_selected(export_screen, event)
        assert opened_exports == [export_scope]

        receipt = object()
        undo_receipts: list[object] = []
        workers: list[tuple[object, bool, str]] = []

        def undo(selected_receipt: object) -> object:
            undo_receipts.append(selected_receipt)

            async def settle() -> None:
                return None

            return settle()

        undo_screen = SimpleNamespace(
            _library_notes_mutation_fenced=lambda: False,
            _library_note_delete_receipt=receipt,
            _library_notes_mutation_in_flight=False,
            is_mounted=False,
            _undo_library_note_delete=undo,
            run_worker=lambda coroutine, *, exclusive, group: workers.append(
                (coroutine, exclusive, group)
            ),
        )
        LibraryScreen.handle_library_note_delete_undo(undo_screen, event)
        assert undo_screen._library_notes_mutation_in_flight
        assert undo_receipts == [receipt]
        assert [(exclusive, group) for _, exclusive, group in workers] == [
            (True, "library_note_mutation")
        ]
        workers[0][0].close()
        assert stopped.call_count == 12


@pytest.mark.parametrize(
    ("authority", "state", "expected_content", "expected_authority", "safe"),
    (
        ("database", "conflict", "Conflict", "Database Notes", "Review recovery"),
        ("database", "read_only", "Read-only", "Database Notes", "Keep the draft"),
        ("database", "failed", "Save failed", "Database Notes", "Retry Save"),
        ("database", "saving", "Saving", "Database Notes", None),
        ("database", "dirty", "Unsaved changes", "Database Notes", None),
        ("database", "clean", "Saved", "Database Notes", None),
        ("folder", "conflict", "Conflict", "Folder Files", "Save Copy"),
        ("folder", "read_only", "Read-only", "Folder Files", "Open Manage"),
        ("folder", "failed", "Save failed", "Folder Files", "Save Copy"),
        ("folder", "saving", "Saving", "Folder Files", None),
        ("folder", "dirty", "Unsaved changes", "Folder Files", None),
        ("folder", "clean", "Saved", "Folder Files", None),
    ),
)
def test_notes_header_status_channels_follow_approved_precedence(
    authority: str,
    state: str,
    expected_content: str,
    expected_authority: str,
    safe: str | None,
) -> None:
    """Pure status channels keep content recovery ahead of Git detail."""
    if authority == "database":
        from tldw_chatbook.Widgets.Library.library_notes_canvas import (
            resolve_database_note_status_channels,
        )

        channels = resolve_database_note_status_channels(
            conflict=state == "conflict",
            read_only=state == "read_only",
            save_failed=state == "failed",
            saving=state == "saving",
            dirty=state == "dirty",
        )
    else:
        from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
            resolve_file_note_status_channels,
        )

        channels = resolve_file_note_status_channels(
            root="/notes/project",
            conflict=state == "conflict",
            read_only=state == "read_only",
            save_failed=state == "failed",
            saving=state == "saving",
            dirty=state == "dirty",
            git_failure="Push failed",
            git_running="Pushing",
            git_changes=3,
        )

    assert channels.content_recovery.startswith(expected_content)
    assert channels.authority_git.startswith(expected_authority)
    assert (channels.safe_next_action or None) == safe
    if authority == "folder":
        # Git failure wins only inside its own channel; it cannot replace the
        # content/recovery decision or its safe next action.
        assert "Push failed" in channels.authority_git
        assert "Pushing" not in channels.authority_git
        assert "3 changes" not in channels.authority_git
    assert "\n" not in channels.content_recovery
    assert "\n" not in channels.authority_git


@pytest.mark.asyncio
async def test_database_note_status_header_paints_actionable_detail() -> None:
    """Detailed failure/recovery copy survives the pure status projection."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=(240, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        await _open_note_editor(screen, pilot)
        snapshot = screen._library_note_session.snapshot
        assert snapshot is not None
        cases = (
            (
                "validation",
                False,
                False,
                "",
                "Title is required before saving.",
                "Next: Retry Save.",
            ),
            (
                "error",
                False,
                False,
                "",
                "Save failed — database busy. Edits kept. Press Save to retry.",
                "Next: Retry Save.",
            ),
            (
                "conflict",
                True,
                False,
                "",
                "Draft changed — Reload not applied. Choose again.",
                "Next: Review recovery.",
            ),
            (
                "idle",
                False,
                True,
                "Save unavailable — finish bulk selection first.",
                "Saved",
                "",
            ),
        )
        missing: list[str] = []

        for (
            autosave_state,
            in_conflict,
            read_only,
            shortcut_status,
            status_message,
            expected_next,
        ) in cases:
            screen._library_note_autosave_state = autosave_state
            screen._library_note_shortcut_status = shortcut_status
            screen._library_notes_select_mode = read_only
            screen._library_note_session._snapshot = replace(
                snapshot,
                dirty=autosave_state != "idle",
                saving=False,
                in_conflict=in_conflict,
                status_message=status_message,
            )
            screen._apply_library_note_presentation_state()
            await pilot.pause()
            expected = shortcut_status or status_message
            status = screen.query_one("#library-note-status", Static)
            rendered = " ".join(
                status.render_line(row).text.strip()
                for row in range(status.region.height)
            )
            painted = "\n".join(
                strip.text for strip in screen._compositor.render_strips()
            )
            next_is_honest = (
                expected_next in rendered if expected_next else "Next:" not in rendered
            )
            if (
                expected not in rendered
                or expected not in painted
                or not next_is_honest
            ):
                missing.append(
                    f"{autosave_state}: expected {expected!r}, rendered {rendered!r}, "
                    f"status_region={status.region!r}, "
                    f"actions_region={screen.query_one('#library-note-primary-actions').region!r}"
                )

        assert missing == []


def test_folder_status_projects_reachable_read_only_recovery_and_bounds_git() -> None:
    """Folder header copy stays truthful, semantic, and bounded."""
    from rich.cells import cell_len

    from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
        resolve_file_note_status_channels,
    )

    ordinary = resolve_file_note_status_channels(
        root="/notes/project",
        read_only=True,
    )
    excerpt = resolve_file_note_status_channels(
        root="/notes/project",
        read_only=True,
        exact_export_available=True,
    )
    git_states = (
        resolve_file_note_status_channels(
            root="/notes/project",
            git_failure="FAILED — Git action failed: " + "private detail " * 30,
        ),
        resolve_file_note_status_channels(
            root="/notes/project",
            git_uncertain="Outcome uncertain: " + "private detail " * 30,
        ),
        resolve_file_note_status_channels(
            root="/notes/project",
            git_running="Checking: " + "private detail " * 30,
        ),
        resolve_file_note_status_channels(
            root=None,
            git_running="Checking: " + "private detail " * 30,
        ),
    )

    assert ordinary.safe_next_action == "Open Manage"
    assert excerpt.safe_next_action == "Export exact copy"
    assert all(cell_len(channels.authority_git) <= 60 for channels in git_states)
    assert all(
        "private detail" not in channels.authority_git for channels in git_states
    )
    assert "Git action failed" in git_states[0].authority_git


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
async def test_notes_global_f6_cycles_only_visible_regions_when_library_collapsed() -> None:
    """At 120 columns the hidden Library region is skipped by the F6 cycle."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryGlobalKeyProductionCSSHarness(app)

    async with host.run_test(size=(120, 35)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        await _open_note_editor(screen, pilot)
        shell = screen.query_one(
            "#library-notes-reader-shell", LibraryAdaptiveReaderShell
        )
        assert shell.effective_layout.library_open is False
        screen.query_one("#library-note-title", Input).focus()
        await pilot.pause()

        await pilot.press("f6")
        await pilot.pause()

        assert screen.query_one("#library-notes-filter", Input).has_focus

        await pilot.press("f6")
        await pilot.pause()

        assert screen.query_one("#library-note-title", Input).has_focus


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
async def test_reader_route_parks_dirty_note_selection_and_preview_without_saving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reader-to-reader routing retains the Notes-owned working session."""
    monkeypatch.setattr(
        library_screen_module, "LIBRARY_NOTES_AUTOSAVE_SECONDS", 3600
    )
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-1")
        screen.query_one("#library-notes-row-1", Button).press()
        body = await _wait_for_selector(screen, pilot, "#library-note-body")
        body.text = "parked reader-route draft"
        await pilot.pause()
        assert screen._library_notes_autosave_timer is not None
        screen.query_one("#library-note-preview", Button).press()
        await pilot.pause()

        screen.query_one("#library-row-browse-media", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-reader-shell")

        screen.query_one("#library-row-browse-media", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-reader-shell")

        snapshot = screen._library_note_session.snapshot
        assert snapshot is not None
        assert (snapshot.note_id, snapshot.body, snapshot.dirty) == (
            "n-2",
            "parked reader-route draft",
            True,
        )
        assert screen._selected_note_id == "n-2"
        assert screen._library_note_preview is True
        assert screen._library_notes_autosave_timer is None
        assert app.notes_scope_service.save_calls == []

        monkeypatch.setattr(
            library_screen_module, "LIBRARY_NOTES_AUTOSAVE_SECONDS", 0.5
        )
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-note-preview-body")

        assert screen._selected_note_id == "n-2"
        assert screen._library_note_preview is True
        assert screen.query_one("#library-note-body", TextArea).text == (
            "parked reader-route draft"
        )
        assert screen._library_notes_autosave_timer is not None
        assert app.notes_scope_service.save_calls == []

        await _wait_for_condition(
            pilot,
            lambda: len(app.notes_scope_service.save_calls) == 1
            and screen._library_note_session.snapshot is not None
            and not screen._library_note_session.snapshot.dirty,
            message="Revisited dirty Notes draft did not resume autosave.",
        )

        snapshot = screen._library_note_session.snapshot
        assert snapshot is not None
        assert snapshot.dirty is False
        assert app.notes_scope_service.save_calls[0]["content"] == (
            "parked reader-route draft"
        )


@pytest.mark.asyncio
async def test_reader_route_invalidates_autosave_queued_before_park(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timer-fired autosave cannot begin persistence after Notes is hidden."""
    monkeypatch.setattr(
        library_screen_module, "LIBRARY_NOTES_AUTOSAVE_SECONDS", 3600
    )
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
        body.text = "queued autosave draft"
        await pilot.pause()

        timer = screen._library_notes_autosave_timer
        assert timer is not None
        callback = getattr(timer._callback, "args", (None,))[0]
        assert callable(callback)
        timer.stop()
        queued_autosaves = []
        original_run_worker = screen.run_worker

        def queue_without_start(awaitable, **_kwargs):
            queued_autosaves.append(awaitable)

        monkeypatch.setattr(screen, "run_worker", queue_without_start)
        callback()
        monkeypatch.setattr(screen, "run_worker", original_run_worker)

        assert len(queued_autosaves) == 1
        assert screen._library_notes_autosave_timer is None
        assert app.notes_scope_service.save_calls == []

        screen.query_one("#library-row-browse-media", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-reader-shell")
        await queued_autosaves[0]

        snapshot = screen._library_note_session.snapshot
        assert snapshot is not None
        assert snapshot.dirty is True
        assert app.notes_scope_service.save_calls == []

        monkeypatch.setattr(
            library_screen_module, "LIBRARY_NOTES_AUTOSAVE_SECONDS", 0.5
        )
        rearm_calls = 0
        original_schedule = screen._schedule_library_note_autosave

        def count_rearm() -> None:
            nonlocal rearm_calls
            rearm_calls += 1
            original_schedule()

        monkeypatch.setattr(screen, "_schedule_library_note_autosave", count_rearm)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-note-body")

        assert rearm_calls == 1
        assert screen._library_notes_autosave_timer is not None
        assert screen.query_one("#library-note-body", TextArea).text == (
            "queued autosave draft"
        )

        await _wait_for_condition(
            pilot,
            lambda: len(app.notes_scope_service.save_calls) == 1
            and screen._library_note_session.snapshot is not None
            and not screen._library_note_session.snapshot.dirty,
            message="Rearmed autosave did not settle exactly once.",
        )

        assert len(app.notes_scope_service.save_calls) == 1
        assert app.notes_scope_service.save_calls[0]["content"] == (
            "queued autosave draft"
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
async def test_emergency_width_preserves_manual_collapse_and_notes_adaptive_owner() -> (
    None
):
    """Ordinary emergency geometry never mutates requested/adaptive state."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(80, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-hub-action-import").focus()
        screen._library_rail_collapsed = True
        screen._apply_library_notes_stage_visibility()

        await pilot.resize_terminal(63, 30)
        await pilot.pause()
        assert screen._library_emergency_stage == "canvas-only"
        assert screen._library_rail_collapsed is True

        await pilot.resize_terminal(64, 30)
        await pilot.pause()
        assert screen._library_emergency_stage is None
        assert screen._library_rail_collapsed is True
        assert screen.query_one("#library-rail").display is False
        assert screen.query_one("#library-canvas").display is True

        await pilot.resize_terminal(63, 30)
        await screen._select_library_rail_row("browse-notes")
        shell = await _wait_for_selector(screen, pilot, "#library-notes-reader-shell")
        await _wait_for_condition(
            pilot,
            lambda: screen._library_emergency_stage is None,
            message="Adaptive Notes did not release ordinary emergency ownership.",
        )

        assert screen._library_notes_stage == "notes"
        assert screen._library_rail_collapsed is True
        assert isinstance(shell, LibraryAdaptiveReaderShell)
        assert shell.work.display is True


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

        heading = screen.query_one("#library-note-heading")
        second_row = screen.query_one("#library-note-header-second-row")
        editor = screen.query_one("#library-note-editor-region")
        # Textual's wide controls use a three-cell physical button height;
        # each wrapper must still remain one logical header row.
        assert heading.region.height <= 3
        assert second_row.region.height <= 3, (
            second_row.styles.height,
            second_row.styles.min_height,
            second_row.styles.max_height,
            second_row.styles.layout,
        )
        assert editor.region.y <= second_row.region.y + 3

        await pilot.resize_terminal(60, 20)
        await pilot.pause()
        canvas = screen.query_one("#library-note-work-pane", LibraryNoteWorkPane)
        from tldw_chatbook.Widgets.Library.library_notes_canvas import (
            resolve_database_note_status_channels,
        )

        canvas.apply_session_state(
            replace(
                screen._library_note_presentation_state(),
                compact=True,
                status_channels=resolve_database_note_status_channels(conflict=True),
            )
        )
        await pilot.pause()
        status = screen.query_one("#library-note-status", Static)
        rendered = " ".join(
            status.render_line(row).text.strip() for row in range(status.region.height)
        )
        assert "Next: Review recovery." in rendered
        selected_title = screen._library_note_session.snapshot.title
        primary = screen.query_one("#library-note-primary-actions")
        compact_controls = (
            screen.query_one("#library-note-editor-title", Static),
            screen.query_one("#library-note-save", Button),
            screen.query_one("#library-note-use-in-console", Button),
        )
        assert all(
            control in screen._compositor.visible_widgets
            for control in compact_controls
        )
        assert all(
            primary.region.contains_region(control.region)
            for control in compact_controls[1:]
        )
        painted = "\n".join(strip.text for strip in screen._compositor.render_strips())
        assert selected_title in painted
        assert "Save" in painted
        assert "Use in Console" in painted


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
