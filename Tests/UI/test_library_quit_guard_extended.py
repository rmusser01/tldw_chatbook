"""Ctrl+Q over unsaved Library work: the variants outside the PR gate.

TASK-34000.1 (review N-01). ``test_library_quit_guard.py`` is the lean
PR-gated core. These cover every other variant and run with the full
``Tests/UI`` suite, not the UI Fast Lane, which is near its 20-minute cap
(TASK-34000.47):
- AC#3: an untouched new note, and a whitespace-only-title new note, leave
  no row and do not ask.
- AC#2: a failed write and a conflicting save ask; Discard keeps the other
  version.
- AC#4: steady typing is saved within the max wait, and keys typed during an
  in-flight save persist.
- AC#1: a dirty Prompt or Skill draft asks, with no veto toast behind the
  prompt, and a Folder files edit is written to disk.

Same harness as the core (``library_quit_guard_support.py``): the real
``TldwCli``, a real database or file, and the quit recorded at
``_run_approved_quit_cleanup``.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Button, Input, Label, TextArea

import tldw_chatbook.UI.Screens.library_screen as library_screen_module
from Tests.UI.library_quit_guard_support import (
    _BODY,
    _DEBOUNCE,
    _OTHER_BODY,
    _OTHER_TITLE,
    _TITLE,
    SIZE,
    _ctrl_q,
    _keep_editing,
    _library,
    _library_app,
    _new_blank_note,
    _no_autosave,
    _open_note,
    _prompt_title,
    _record_toasts,
    _scaled_autosave,
    _settings_without_splash,
    _type,
    _type_at_end,
    _type_steadily,
    _until,
)
from Tests.UI.test_library_shell import _wait_for_selector
from Tests.app_module_patches import patch_app_global
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    LocalPromptService,
    PromptScopeService,
)
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    LibraryFileNotesWorkspace,
)

#: Same reason as the core file: full-app boots need the bootstrap profile
#: locally (``raw_source_selection_changed`` under the per-test redirect).
pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


async def test_ctrl_q_on_an_untouched_new_note_leaves_no_untitled_row(
    tmp_path, monkeypatch
):
    _no_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(
        tmp_path, monkeypatch, events, lambda notes: notes.rows()
    )
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            await _new_blank_note(screen, pilot)
            assert len(profile.rows()) == 3, "the blank note's row exists while open"

            assert await _ctrl_q(pilot, app, events) is None
            [(kind, rows)] = events
            assert kind == "quit"
            assert sorted(rows) == sorted([(_TITLE, _BODY), (_OTHER_TITLE, _OTHER_BODY)]), (
                f"the untouched new note survived the quit: {rows}"
            )
    profile.db.close_connection()



async def test_ctrl_q_on_a_new_note_with_only_a_blank_title_discards_it(
    tmp_path, monkeypatch
):
    """Review finding #7: quit treats a whitespace-only title like navigation.

    Leaving such a note discards it ("Empty note discarded"). Quitting must
    not first try to save the spaces, hit the whitespace veto and ask.
    """
    _no_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(
        tmp_path, monkeypatch, events, lambda notes: notes.rows()
    )
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            await _new_blank_note(screen, pilot)
            screen.query_one("#library-note-title", Input).focus()
            await pilot.pause()
            await _type(pilot, "  ")
            assert screen._library_note_session.snapshot.dirty

            assert await _ctrl_q(pilot, app, events) is None, (
                "a whitespace-only new note asked instead of being discarded"
            )
            [(kind, rows)] = events
            assert kind == "quit"
            assert sorted(rows) == sorted(
                [(_TITLE, _BODY), (_OTHER_TITLE, _OTHER_BODY)]
            ), rows
    profile.db.close_connection()



async def test_a_failed_write_asks_instead_of_exiting(tmp_path, monkeypatch):
    """Write failure: the note's text is never silently dropped."""
    _no_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(
        tmp_path, monkeypatch, events, lambda notes: notes.note(notes.note_id)
    )

    def _disk_full(**_kwargs):
        raise OSError("disk full")

    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            body = await _open_note(screen, pilot, _TITLE, profile.note_id)
            await _type_at_end(pilot, body, " unsaved")
            monkeypatch.setattr(profile.scope_service, "save_note", _disk_full)

            prompt = await _ctrl_q(pilot, app, events)

            assert prompt is not None, "a failed save quit silently"
            assert _prompt_title(prompt) == (
                f'Quit and discard unsaved changes to "{_TITLE}"?'
            )
            await _keep_editing(pilot, app, screen, prompt)
            assert body.text == _BODY + " unsaved"
            assert profile.note(profile.note_id)["content"] == _BODY
    profile.db.close_connection()



async def test_a_conflicting_save_asks_and_keeps_the_other_version(
    tmp_path, monkeypatch
):
    """AC#2 conflict variant: the note changed elsewhere since it opened."""
    _no_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(
        tmp_path, monkeypatch, events, lambda notes: notes.note(notes.note_id)
    )
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            body = await _open_note(screen, pilot, _TITLE, profile.note_id)
            await _type_at_end(pilot, body, " mine")
            version = profile.note(profile.note_id)["version"]
            assert profile.db.update_note(
                profile.note_id, {"content": "edited elsewhere"}, version
            )

            prompt = await _ctrl_q(pilot, app, events)

            assert prompt is not None, "a conflicting save quit silently"
            assert _prompt_title(prompt) == (
                f'Quit and discard unsaved changes to "{_TITLE}"?'
            )
            message = str(prompt.query_one(".dialog-message", Label).renderable)
            assert "changed elsewhere" in message, message
            await _keep_editing(pilot, app, screen, prompt)
            assert body.text == _BODY + " mine"
            assert screen._library_note_session.snapshot.in_conflict

            prompt = await _ctrl_q(pilot, app, events)
            assert prompt is not None
            await pilot.click("#confirm-button")
            await _until(pilot, lambda: bool(events), "Discard and quit to quit")
            [(kind, row)] = events
            assert kind == "quit"
            assert row["content"] == "edited elsewhere", (
                "the other version was overwritten"
            )
    profile.db.close_connection()



async def test_continuous_typing_is_saved_within_the_max_wait(tmp_path, monkeypatch):
    """One key every 0.25 s never lets a 1 s debounce fire; the max wait does.

    The production pair is 2 s / 10 s; scaled down here so the burst fits a
    test. Every keystroke lands well inside the debounce, so without a
    maximum wait nothing reaches the database until typing stops.
    """
    _scaled_autosave(monkeypatch)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda _notes: None)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            body = await _open_note(screen, pilot, _TITLE, profile.note_id)
            body.focus()
            body.move_cursor(body.document.end)
            await pilot.pause()

            gaps, persisted = await _type_steadily(
                pilot,
                "abcdefghijklmnop",  # 16 keys over ~4 s
                lambda: profile.note(profile.note_id)["content"],
            )

            first_save = next(
                (index for index, content in enumerate(persisted) if content != _BODY),
                None,
            )
            assert first_save is not None, (
                "four seconds of continuous typing never reached the database"
            )
            assert persisted[first_save].startswith(_BODY + "a"), persisted[first_save]
            # Only the max wait can have saved it: no gap before that save
            # was long enough for the debounce to fire on its own.
            assert max(gaps[: first_save + 1]) < _DEBOUNCE, (
                f"a {max(gaps[: first_save + 1]):.2f}s pause let the debounce "
                "fire; this run cannot tell the max wait from the debounce"
            )
    profile.db.close_connection()



async def test_keys_typed_while_a_save_is_in_flight_are_saved(tmp_path, monkeypatch):
    """Review finding #5: keys that land during a slow save still persist.

    ``_schedule_library_note_autosave`` arms nothing while the session is
    saving; the coordinator's save loop picks those keys up instead.
    """
    monkeypatch.setattr(library_screen_module, "LIBRARY_NOTES_AUTOSAVE_SECONDS", 0.2)
    events: list = []
    app, profile = _library_app(tmp_path, monkeypatch, events, lambda _notes: None)
    original_save = profile.scope_service.save_note
    in_flight: list[bool] = []

    async def _slow_save(**kwargs):
        in_flight.append(True)
        await asyncio.sleep(0.8)
        return await original_save(**kwargs)

    monkeypatch.setattr(profile.scope_service, "save_note", _slow_save)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            body = await _open_note(screen, pilot, _TITLE, profile.note_id)
            await _type_at_end(pilot, body, "a")
            await _until(
                pilot,
                lambda: bool(screen._library_note_session.snapshot.saving),
                "the autosave to start its slow save",
                timeout=10.0,
            )
            await _type(pilot, "bcd")  # typed while that save is in flight
            assert screen._library_note_session.snapshot.saving

            # The DB row lands inside the port call, a moment before the
            # coordinator settles its snapshot; wait for both.
            await _until(
                pilot,
                lambda: (
                    profile.note(profile.note_id)["content"] == _BODY + "abcd"
                    and not screen._library_note_session.snapshot.saving
                ),
                "the keys typed during the save to reach the database",
                timeout=15.0,
            )
            assert not screen._library_note_session.snapshot.dirty
    profile.db.close_connection()



async def test_ctrl_q_with_a_dirty_prompt_draft_asks_and_keeps_it(
    tmp_path, monkeypatch
):
    """Prompts are explicit-Save only: the quit asks instead of dropping them."""
    events: list = []
    app, _ = _library_app(
        tmp_path, monkeypatch, events, lambda _notes: None, notes=False
    )
    prompts_db = PromptsDatabase(tmp_path / "prompts.db", client_id="task-34000-1")
    prompt_id, _uuid, _message = prompts_db.add_prompt(
        name="Summarize",
        author="Alice",
        details="A summarizer",
        system_prompt="You are concise.",
        user_prompt="Summarize: {text}",
    )
    app.prompt_scope_service = PromptScopeService(
        local_service=LocalPromptService(prompts_db), server_service=None
    )
    toasts = _record_toasts(monkeypatch, app)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            screen.query_one("#library-row-browse-prompts", Button).press()
            row = await _wait_for_selector(
                screen, pilot, f"#library-prompt-row-{prompt_id}"
            )
            row.press()
            name = await _wait_for_selector(screen, pilot, "#library-prompt-name")
            await _until(
                pilot, lambda: screen._prompts_state.editor_armed, "the prompt editor"
            )
            name.focus()
            await pilot.press("!")
            await _until(pilot, lambda: screen._prompts_state.dirty, "a dirty prompt")
            typed = name.value
            toasts.clear()

            prompt = await _ctrl_q(pilot, app, events)

            assert prompt is not None, "Ctrl+Q quit past a dirty prompt draft"
            # Review finding #6: the prompt says it; no veto toast behind it.
            assert not [t for t in toasts if "Unsaved Prompt" in t], toasts
            assert _prompt_title(prompt) == (
                'Quit and discard unsaved changes to "Summarize"?'
            )
            await _keep_editing(pilot, app, screen, prompt)
            assert screen._prompts_state.dirty
            assert name.value == typed
    prompts_db.close_connection()



async def test_ctrl_q_with_a_dirty_skill_draft_asks(tmp_path, monkeypatch):
    events: list = []
    app, _ = _library_app(
        tmp_path, monkeypatch, events, lambda _notes: None, notes=False
    )
    toasts = _record_toasts(monkeypatch, app)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            screen.query_one("#library-row-create-skill", Button).press()
            name = await _wait_for_selector(screen, pilot, "#library-skill-name")
            await _until(
                pilot, lambda: screen._skills_state.editor_armed, "the skill editor"
            )
            name.focus()
            await _type(pilot, "dirty-demo")
            await _until(pilot, lambda: screen._skills_state.dirty, "a dirty skill")
            toasts.clear()

            prompt = await _ctrl_q(pilot, app, events)

            assert prompt is not None, "Ctrl+Q quit past a dirty skill draft"
            assert not [t for t in toasts if "Unsaved skill" in t], toasts
            assert _prompt_title(prompt) == (
                'Quit and discard unsaved changes to "new skill"?'
            )
            await _keep_editing(pilot, app, screen, prompt)
            assert screen._skills_state.dirty
            assert name.value == "dirty-demo"



async def test_ctrl_q_flushes_a_folder_files_edit_to_disk(tmp_path, monkeypatch):
    """B-static: ``shutdown`` stopped the Folder files autosave without saving."""
    root = tmp_path / "folder"
    root.mkdir()
    source = root / "source.md"
    source.write_text("# Source\n", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root, replica=replica, autosave_delay=3600, poll_interval=3600
    )
    events: list = []
    app, _ = _library_app(
        tmp_path,
        monkeypatch,
        events,
        lambda _notes: source.read_text(encoding="utf-8"),
        notes=False,
    )
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            screen._notes_state.file_notes_workspace_factory = lambda: workspace
            screen.query_one("#library-row-browse-notes", Button).press()
            files = await _wait_for_selector(
                screen, pilot, "#library-notes-source-files"
            )
            files.press()
            await _until(
                pilot,
                lambda: workspace.initialized and workspace.is_mounted,
                "Folder files to mount",
            )
            assert await workspace.open_path("source.md")
            editor = workspace.query_one("#file-notes-editor", TextArea)
            await _until(pilot, lambda: not editor.read_only, "an editable file")
            await _type_at_end(pilot, editor, "tail")
            await _until(
                pilot, lambda: workspace.save_state == "dirty", "a dirty file edit"
            )

            assert await _ctrl_q(pilot, app, events) is None
            [(kind, text)] = events
            assert kind == "quit"
            # The workspace keeps the file's final newline on save.
            assert text == "# Source\ntail\n", "Ctrl+Q exited before the file was saved"
    replica.close()

