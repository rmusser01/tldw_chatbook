"""Mounted Database Notes journeys through the shared adaptive reader."""

from __future__ import annotations

import pytest
from textual.widgets import Button, TextArea

from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _open_note_editor,
    _seed_conversations,
    _two_conversations,
    _two_notes,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Widgets.Library import (
    LibraryAdaptiveReaderShell,
    LibraryNoteWorkPane,
    LibraryNotesCanvas,
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
