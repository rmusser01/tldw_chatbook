"""Mounted Database Notes journeys through the shared adaptive reader."""

from __future__ import annotations

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
    _seed_conversations,
    _two_conversations,
    _two_notes,
    _wait_for_library_shell,
    _wait_for_condition,
    _wait_for_selector,
)
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library import (
    LibraryAdaptiveReaderShell,
    LibraryNoteWorkPane,
    LibraryNotesCanvas,
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

        assert screen.query_one("#library-rail").display is True
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
