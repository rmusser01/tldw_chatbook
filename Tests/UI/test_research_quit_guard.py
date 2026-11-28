"""Ctrl+Q protects a Quick Note in the real Research destination."""

from __future__ import annotations

import sqlite3
from unittest.mock import AsyncMock

import pytest
from textual.widgets import Button, Input, TextArea

from Tests.app_module_patches import patch_app_global
from Tests.UI.app_factory import (
    _build_test_app,
    attach_chachanotes_db,
    swap_app_aliases,
)
from Tests.UI.library_quit_guard_support import (
    _ctrl_q,
    _keep_editing,
    _settings_without_splash,
    _until,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.UI.Research_Workspace_Modules.quick_notes_section import (
    ResearchQuickNotesSection,
)
from tldw_chatbook.UI.Screens.research_workspace_screen import ResearchWorkspaceScreen
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


@pytest.mark.parametrize("scenario", ["save", "refused", "edited-during-save"])
async def test_ctrl_q_protects_dirty_research_quick_note(
    tmp_path, monkeypatch, scenario
):
    db_path = tmp_path / "notes.sqlite3"
    app = _build_test_app(configured_default="home")
    db = attach_chachanotes_db(app, db_path=db_path)
    interop = NotesInteropService(
        base_db_directory=tmp_path, api_client_id="test-client", global_db_to_use=db
    )
    notes = NotesScopeService(local_notes_service=interop, server_service=None)
    if scenario == "refused":
        monkeypatch.setattr(
            notes, "save_note", AsyncMock(side_effect=sqlite3.OperationalError("busy"))
        )
    registry_db = WorkspaceDB(tmp_path / "workspaces.sqlite3")
    registry = LocalWorkspaceRegistryService(registry_db)
    registry.create_workspace(workspace_id="research", name="Reading group")
    swap_app_aliases(
        app, workspace_registry_service=registry, notes_scope_service=notes
    )
    quits = []

    async def record_quit():
        # Open a separate owner to see exactly what would survive a relaunch.
        reopened = CharactersRAGDB(db_path, client_id="reopened")
        try:
            quits.append(
                [(row["title"], row["content"]) for row in reopened.list_notes()]
            )
        finally:
            reopened.close_connection()

    monkeypatch.setattr(app, "_run_approved_quit_cleanup", record_quit)
    try:
        with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
            async with app.run_test(size=(160, 45)) as pilot:
                await _until(pilot, lambda: app.current_tab == "home", "Home to mount")
                await pilot.press("f5")
                await _until(
                    pilot,
                    lambda: isinstance(app.screen, ResearchWorkspaceScreen),
                    "Research",
                    timeout=10.0,
                )
                screen = app.screen
                editor = screen.query_one(ResearchQuickNotesSection)
                title = editor.query_one("#research-quick-note-title", Input)
                body = editor.query_one("#research-quick-note-body", TextArea)
                save = editor.query_one("#research-quick-note-save", Button)
                await _until(
                    pilot,
                    lambda: editor.editor_ref is not None and not save.disabled,
                    "Quick Note editor",
                    timeout=10.0,
                )
                title.value = "Reading notes"
                body.load_text("Check the original study.")
                body.focus()
                assert editor.has_nonempty_dirty_draft

                if scenario == "edited-during-save":
                    save_note = notes.save_note

                    async def save_with_later_edit(**kwargs):
                        body.load_text("Check the updated study as well.")
                        return await save_note(**kwargs)

                    monkeypatch.setattr(notes, "save_note", save_with_later_edit)

                prompt = await _ctrl_q(pilot, app, quits)
                if scenario == "save":
                    assert prompt is None
                    assert quits == [[("Reading notes", "Check the original study.")]]
                    return

                assert prompt is not None, "Ctrl+Q lost the draft without asking"
                await _keep_editing(pilot, app, screen, prompt)
                assert editor.has_nonempty_dirty_draft
                assert title.value == "Reading notes"
                assert quits == []
                if scenario == "edited-during-save":
                    assert body.text == "Check the updated study as well."
                    monkeypatch.setattr(notes, "save_note", save_note)
                    assert await _ctrl_q(pilot, app, quits) is None
                    assert quits == [
                        [("Reading notes", "Check the updated study as well.")]
                    ]
                    return

                assert body.text == "Check the original study."

                prompt = await _ctrl_q(pilot, app, quits)
                assert prompt is not None
                await app.pop_screen()
                await _until(
                    pilot,
                    lambda: not app._quit_in_progress,
                    "quit veto after the prompt closed",
                )
                assert editor.has_nonempty_dirty_draft
                assert quits == []

                prompt = await _ctrl_q(pilot, app, quits)
                assert prompt is not None
                discard = prompt.query_one("#confirm-button", Button)
                assert str(discard.label) == "Discard and quit"
                discard.press()
                await _until(pilot, lambda: bool(quits), "the explicitly approved quit")
                assert quits == [[]]
    finally:
        registry_db.close()
