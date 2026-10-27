"""Ctrl+S in the prompt editor and in Folder files (TASK-34000.40 N-33).

The sibling of ``test_library_ctrl_s_saves_in_every_editor.py`` (the notes
arms), split for the 30 s census budget. Production ``TldwCli`` booted into
Library: a real Prompts database (its service gated to make a REAL held
Save), and a real vault on disk under the production Folder files workspace.
Real keys and clicks; painted text; data read back from the stores.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from textual.widgets import Button, Input, TextArea

# Stubs first: they register the optional modules the app imports below probe.
import Tests.UI._optional_module_stubs  # noqa: F401
from Tests.app_module_patches import patch_app_global
from Tests.UI.library_quit_guard_support import (
    SIZE,
    _library,
    _library_app,
    _record_toasts,
    _settings_without_splash,
    _type,
    _until,
)
from Tests.UI.test_library_ctrl_s_saves_in_every_editor import (
    NO_CHANGES,
    _f1_keys,
    _focused_id,
    _footer,
    _painted,
    _until_selector,
)
from Tests.UI.test_library_file_notes_workspace import _production_workspace_context
from Tests.UI.test_library_prompts_canvas import (
    _open_prompt_editor,
    _real_prompt_scope_service,
)
from Tests.UI.test_library_shell import _wait_for_condition
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    LibraryFileNotesWorkspace,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: Review R1 N-1: a held write waits at most this long -- far past every wait
#: in the test -- and reaching it FAILS the test; it never releases silently.
HOLD_BOUND = 120.0


async def test_ctrl_s_saves_a_prompt_and_refuses_when_the_button_would(
    tmp_path, monkeypatch
):
    """The prompt editor's Ctrl+S is its Save button: a dirty prompt is written
    to the Prompts database; a clean editor says "Saved — no changes." with no
    chip in the footer (ruling T9-j); and a Save held by a REAL write in
    flight (the Prompts service is gated) refuses at the editor with the
    reason, writes nothing more (review M-5), and F1 does not list the key
    there (ruling T9-h)."""
    db, service = _real_prompt_scope_service(tmp_path)
    gate = threading.Event()
    gate.set()
    save_calls: list[object] = []
    hold_timeouts: list[object] = []
    real_save_prompt = service.save_prompt

    async def gated_save_prompt(*args, **kwargs):
        save_calls.append(kwargs.get("name", args[:1]))
        if not await asyncio.to_thread(gate.wait, HOLD_BOUND):
            hold_timeouts.append(save_calls[-1])
            raise AssertionError(f"the held save outlived its {HOLD_BOUND} s bound")
        return await real_save_prompt(*args, **kwargs)

    service.save_prompt = gated_save_prompt
    prompt_id, _uuid, _message = db.add_prompt(
        name="Iota", author="Original", details="d", user_prompt="x"
    )
    app, _profile = _library_app(tmp_path, monkeypatch, [], lambda _p: None, notes=False)
    app.prompt_scope_service = service
    toasts = _record_toasts(monkeypatch, app)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            await _open_prompt_editor(screen, pilot, prompt_id)
            name = await _until_selector(pilot, screen, "#library-prompt-name")
            assert isinstance(name, Input)
            await _until(pilot, lambda: name.region.height > 0, "the Name field laid out")
            await pilot.click("#library-prompt-name")
            await _until(
                pilot,
                lambda: _focused_id(screen) == "library-prompt-name",
                "focus in Name",
            )
            await pilot.press("end")
            await _type(pilot, " two")
            await _until(pilot, lambda: name.value == "Iota two", "the typed name")
            await _until(
                pilot,
                lambda: "ctrl+s save changes" in _footer(app),
                "a dirty prompt to chip Ctrl+S",
            )

            await pilot.press("ctrl+s")
            await _until(
                pilot,
                lambda: db.fetch_prompt_details(prompt_id)["name"] == "Iota two",
                "Ctrl+S to write the prompt",
            )
            assert db.fetch_prompt_details(prompt_id)["version"] == 2

            def _status() -> str:
                return _painted(app, screen.query_one("#library-prompt-save-status"))

            await _wait_for_condition(
                pilot,
                lambda: "Saved." in _status(),
                message=lambda: f"no painted Saved receipt; the status line paints {_status()!r}",
            )
            # Review R1 N-1: by id -- a rebuild may replace the Input, not it.
            assert _focused_id(screen) == "library-prompt-name"
            await _until(
                pilot,
                lambda: "ctrl+s" not in _footer(app),
                "the saved, clean prompt to drop the Ctrl+S chip",
            )
            await pilot.press("ctrl+s")
            await _until(
                pilot,
                lambda: NO_CHANGES in _status(),
                "a clean prompt's Ctrl+S to say there were no changes",
            )
            assert db.fetch_prompt_details(prompt_id)["version"] == 2
            assert len(save_calls) == 1

            # A REAL hold: the next save blocks in the service, so Save stays
            # shown but disabled while it writes; the key refuses with the
            # reason and starts no second write. The gate opens in ``finally``
            # so a failing test never leaves the write blocked.
            gate.clear()
            try:
                await _type(pilot, "x")
                await _until(pilot, lambda: screen._prompts_state.dirty, "a new edit")
                await pilot.press("ctrl+s")
                await _until(
                    pilot, lambda: len(save_calls) == 2, "the held save to start"
                )
                await _until(
                    pilot,
                    lambda: screen.query_one("#library-prompt-save", Button).disabled,
                    "the in-flight write to hold Save",
                )
                await pilot.press("ctrl+s")
                await _until(
                    pilot,
                    lambda: "Save unavailable" in _status(),
                    "a held Save to refuse Ctrl+S with its reason",
                )
                await _until(
                    pilot,
                    lambda: "ctrl+s" not in _footer(app),
                    "a held Save to withdraw the Ctrl+S chip",
                )
                assert len(save_calls) == 2, save_calls
                assert _focused_id(screen) == "library-prompt-name"
                # Ruling T9-h: F1 mirrors the footer -- a refusing editor's
                # key still answers, but F1 does not list it.
                assert "ctrl+s" not in await _f1_keys(pilot, app, screen)
            finally:
                gate.set()
            await _until(
                pilot,
                lambda: db.fetch_prompt_details(prompt_id)["name"] == "Iota twox",
                "the held save to land",
            )
            assert len(save_calls) == 2, save_calls
            assert not [toast for toast in toasts if "Save" in toast], toasts
    db.close_connection()
    assert not hold_timeouts, f"a held save timed out: {hold_timeouts}"


async def test_ctrl_s_in_folder_files_flushes_the_draft_to_disk(tmp_path):
    """Folder files autosaves after 2 s; Ctrl+S writes the file now. The chip
    shows only while the draft is dirty (ruling T9-j). A file changed on disk
    under a dirty draft refuses at the editor, keeps the caret, never
    overwrites the other writer, and drops the chip."""
    root = tmp_path / "vault"
    root.mkdir()
    note = root / "today.md"
    note.write_text("first line\n", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=root, replica=replica, autosave_delay=3600)
    async with _production_workspace_context(workspace, size=SIZE) as pilot:
        screen = pilot.app.screen
        assert await workspace.open_path("today.md")
        await _until(
            pilot, lambda: workspace.current_path == "today.md", "today.md to open"
        )
        editor = workspace.query_one("#file-notes-editor", TextArea)
        await _until(pilot, lambda: editor.region.height > 0, "the editor laid out")
        await pilot.click("#file-notes-editor")
        await _until(pilot, lambda: screen.focused is editor, "focus in the editor")

        # Review R1 N-1: re-queried on every read -- no widget is held
        # across a save.
        def _editor_text() -> str:
            return workspace.query_one("#file-notes-editor", TextArea).text

        def _action() -> str:
            return _painted(pilot.app, workspace.query_one("#file-notes-action-status"))

        await _until(
            pilot, lambda: "esc files" in _footer(pilot.app), "the editor's footer"
        )
        assert "ctrl+s" not in _footer(pilot.app), _footer(pilot.app)
        await _type(pilot, "typed ")
        await _until(pilot, lambda: workspace.save_state == "dirty", "a dirty draft")
        assert note.read_text(encoding="utf-8") == "first line\n"
        await _until(
            pilot,
            lambda: "ctrl+s save file" in _footer(pilot.app),
            "a dirty draft to chip Ctrl+S",
        )

        await pilot.press("ctrl+s")
        await _until(
            pilot,
            lambda: note.read_text(encoding="utf-8") == _editor_text(),
            "Ctrl+S to write the draft to disk",
        )
        await _until(
            pilot,
            lambda: "ctrl+s" not in _footer(pilot.app),
            "the saved file to drop the chip",
        )
        await _until(pilot, lambda: "Saved." in _action(), "the Saved receipt")
        assert _focused_id(screen) == "file-notes-editor"

        await pilot.press("ctrl+s")
        await _until(
            pilot,
            lambda: NO_CHANGES in _action(),
            "a clean file's Ctrl+S to say there were no changes",
        )

        await _type(pilot, "mine ")
        await _until(pilot, lambda: workspace.save_state == "dirty", "a second draft")
        note.write_text("someone else wrote this\n", encoding="utf-8")
        await pilot.press("ctrl+s")
        await _until(
            pilot,
            lambda: "Save refused" in _action(),
            "a changed file to refuse Ctrl+S at the editor",
        )
        assert note.read_text(encoding="utf-8") == "someone else wrote this\n"
        assert _focused_id(screen) == "file-notes-editor"
        # The key refuses here, so the footer stops promising "save file".
        await _until(
            pilot,
            lambda: "ctrl+s" not in _footer(pilot.app),
            "the refused save to withdraw the footer's ctrl+s chip",
        )
    await workspace.shutdown()
    replica.close()
