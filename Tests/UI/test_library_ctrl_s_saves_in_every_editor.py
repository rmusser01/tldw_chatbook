"""Ctrl+S saves in every Library editor, and nowhere else (TASK-34000.40 N-33).

The 2026-10-02 review found four save models on one screen: Ctrl+S saved a
skill, and did nothing in a note (the status sat on "Unsaved changes" until the
2 s autosave), a prompt or a Folder files document. The owner ruled
(ADR-031 refinement 2026-10-03; settled 2026-10-27): Ctrl+S performs the
focused editor's visible commit, flushes an autosaving editor at once, says
why at the editor when it refuses, and is unbound outside an editor.

Every test here drives the production ``TldwCli`` booted into Library and
presses the real keys. The data claims read real stores: a ChaChaNotes
database, a Prompts database and a Folder files vault on disk. The two Media
forms run against the production stylesheet harness in
``test_library_ctrl_s_saves_in_every_editor_media.py``.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, TextArea

# Stubs first: they register the optional modules the app imports below probe.
import Tests.UI._optional_module_stubs  # noqa: F401
from Tests.app_module_patches import patch_app_global
from Tests.UI.library_quit_guard_support import (
    _BODY,
    _TITLE,
    SIZE,
    _library,
    _library_app,
    _no_autosave,
    _open_note,
    _record_toasts,
    _settings_without_splash,
    _type,
    _until,
)
from Tests.UI.test_library_file_notes_workspace import _production_workspace_context
from Tests.UI.test_library_shell import _wait_for_condition
from Tests.UI.test_library_prompts_canvas import (
    _open_prompt_editor,
    _real_prompt_scope_service,
)
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    LibraryFileNotesWorkspace,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: Owner ruling 2026-10-27 (TASK-34000.40 AC#3): a clean editor's Ctrl+S.
NOTHING_TO_FLUSH = "Saved — nothing to flush."


def _painted(app, widget) -> str:
    """The text the compositor paints inside ``widget``'s region."""
    region = widget.region
    strips = list(app.screen._compositor.render_strips())
    return "\n".join(
        strips[y].crop(region.x, region.right).text.rstrip()
        for y in range(region.y, min(region.bottom, len(strips)))
    )


def _footer(app) -> str:
    return _painted(app, app.screen.query_one(AppFooterStatus))


def _within(widget, selector_id: str) -> bool:
    return widget is not None and any(
        node.id == selector_id for node in widget.ancestors_with_self
    )


async def _f1_keys(pilot, app, screen) -> set[str]:
    """Open F1, read the keys it lists, close it again."""
    await pilot.press("f1")
    await _until(
        pilot, lambda: isinstance(app.screen, WorkbenchHelpPanel), "F1 to open help"
    )
    keys = {key for key, _label in app.screen.state.shortcuts}
    await pilot.press("escape")
    await _until(pilot, lambda: app.screen is screen, "help to close")
    return keys


async def _click_into_body(pilot, screen) -> TextArea:
    body = screen.query_one("#library-note-body", TextArea)
    await pilot.click("#library-note-body")
    await _until(pilot, lambda: screen.focused is body, "a click to focus the body")
    return body


async def test_ctrl_s_in_the_note_body_flushes_the_autosave_now(tmp_path, monkeypatch):
    """Rule (b): the key writes the draft now, not when the debounce fires.

    Autosave is pushed out to an hour, so the only thing that can write the
    row is the key. Then a second Ctrl+S on the clean note says so on the
    editor's own status line -- no toast -- and the caret never moves.
    """
    _no_autosave(monkeypatch)
    app, profile = _library_app(tmp_path, monkeypatch, [], lambda notes: None)
    toasts = _record_toasts(monkeypatch, app)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            await _open_note(screen, pilot, _TITLE, profile.note_id)
            body = await _click_into_body(pilot, screen)
            await _type(pilot, "ctrl-s test ")
            await _until(
                pilot,
                lambda: screen._library_note_session.snapshot.dirty,
                "typing to dirty the note",
            )
            assert profile.note(profile.note_id)["content"] == _BODY
            # Ctrl+S is real input: it reaches ``on_key`` like any key, so a
            # pending programmatic landing is withdrawn (standing owner rule).
            screen._library_notes_filter_row_handoff = ("pending", True)

            await pilot.press("ctrl+s")
            await _until(
                pilot,
                lambda: profile.note(profile.note_id)["content"] == body.text,
                "Ctrl+S to write the typed body to the database",
                timeout=3.0,
            )
            assert "ctrl-s test" in profile.note(profile.note_id)["content"]
            assert screen._library_notes_filter_row_handoff == ("pending", False)
            status = screen.query_one("#library-note-status")
            await _until(
                pilot,
                lambda: _painted(app, status).startswith("Saved"),
                "the status line to say Saved",
            )
            assert screen.focused is body, f"Ctrl+S moved focus to {screen.focused!r}"
            assert "ctrl+s save now" in _footer(app), _footer(app)

            await pilot.press("ctrl+s")
            await _until(
                pilot,
                lambda: NOTHING_TO_FLUSH in _painted(app, status),
                "a clean note's Ctrl+S to say there was nothing to flush",
            )
            assert screen.focused is body
            assert not [toast for toast in toasts if "Saved" in toast], toasts

            # A real key still types after both saves (lessons-textual :680).
            await _type(pilot, "z")
            await _until(
                pilot,
                lambda: screen._library_note_session.snapshot.dirty,
                "a key typed after Ctrl+S to reach the body",
            )
    profile.db.close_connection()


async def test_ctrl_s_outside_an_editor_does_nothing_and_is_not_advertised(
    tmp_path, monkeypatch
):
    """Rules (c), (d), (e). The Delete confirm refuses at the editor without
    moving focus; the Notes list does nothing even with a dirty note open
    beside it (ruling T9-d), and neither the footer nor F1 offers the key
    there, while the body's footer and F1 do."""
    _no_autosave(monkeypatch)
    app, profile = _library_app(tmp_path, monkeypatch, [], lambda notes: None)
    toasts = _record_toasts(monkeypatch, app)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _library(app, pilot)
            await _open_note(screen, pilot, _TITLE, profile.note_id)

            # The Delete confirm: refused, focus kept on Cancel, nothing written.
            await pilot.click("#library-note-context")
            delete = await _until_selector(pilot, screen, "#library-note-delete")
            # Setup, not the subject: Delete sits below Info's fold at 160x45.
            delete.press()
            cancel = await _until_selector(pilot, screen, "#library-note-delete-cancel")
            await _until(pilot, lambda: screen.focused is cancel, "focus on Cancel")
            await pilot.press("ctrl+s")
            status = screen.query_one("#library-note-status")
            await _until(
                pilot,
                lambda: "Save locked" in _painted(app, status),
                "the Delete confirm to refuse Ctrl+S at the editor",
            )
            assert screen.focused is cancel
            assert profile.note(profile.note_id)["content"] == _BODY
            await pilot.press("enter")
            await _until(
                pilot,
                lambda: not screen._notes_state.confirming_delete,
                "Enter on Cancel to close the confirm",
            )
            assert profile.note(profile.note_id) is not None

            # A dirty note open in the pane; focus moves to the Notes list.
            await pilot.click("#library-note-edit")
            await _until(
                pilot,
                lambda: bool(screen.query("#library-note-body"))
                and screen.query_one("#library-note-body").display,
                "the editor to return",
            )
            await _click_into_body(pilot, screen)
            await _type(pilot, "unsaved ")
            await _until(
                pilot,
                lambda: screen._library_note_session.snapshot.dirty,
                "typing to dirty the note",
            )
            assert "ctrl+s save now" in _footer(app), _footer(app)
            assert "ctrl+s" in await _f1_keys(pilot, app, screen)

            # Shift+F6 lands in the Notes filter; Down from its empty box
            # lands on the folder tree's first row (TASK-34000.28 N-21).
            toasts_before = list(toasts)
            await pilot.press("shift+f6")
            await _until(
                pilot,
                lambda: getattr(screen.focused, "id", None) == "library-notes-filter",
                "Shift+F6 to move focus to the Notes filter",
            )
            for step in ("filter", "tree"):
                if step == "tree":
                    await pilot.press("down")
                    await _until(
                        pilot,
                        lambda: isinstance(screen.focused, Button)
                        and not _within(screen.focused, "library-note-work-pane"),
                        "Down to move focus onto the folder tree",
                    )
                moved_to = screen.focused
                await pilot.press("ctrl+s")
                await pilot.pause(0.5)
                assert profile.note(profile.note_id)["content"] == _BODY, (
                    f"Ctrl+S on {moved_to!r} wrote the note open beside it"
                )
                assert screen._library_note_session.snapshot.dirty
                assert screen.focused is moved_to
                assert toasts == toasts_before, toasts
                footer = _footer(app)
                assert "ctrl+s" not in footer, (step, footer)
            assert "ctrl+s" not in await _f1_keys(pilot, app, screen)
    profile.db.close_connection()


async def _until_selector(pilot, screen, selector: str):
    await _until(
        pilot,
        lambda: bool(screen.query(selector)) and screen.query_one(selector).display,
        f"{selector} to mount",
    )
    return screen.query_one(selector)


async def test_ctrl_s_saves_a_prompt_and_refuses_when_the_button_would(
    tmp_path, monkeypatch
):
    """The prompt editor's Ctrl+S is its Save button: a dirty prompt is written
    to the Prompts database; a disabled Save (an update in flight) refuses at
    the editor with the reason; a clean editor says nothing was pending."""
    db, service = _real_prompt_scope_service(tmp_path)
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
            await _until(pilot, lambda: screen.focused is name, "focus in Name")
            await pilot.press("end")
            await _type(pilot, " two")
            await _until(pilot, lambda: name.value == "Iota two", "the typed name")

            await pilot.press("ctrl+s")
            await _until(
                pilot,
                lambda: db.fetch_prompt_details(prompt_id)["name"] == "Iota two",
                "Ctrl+S to write the prompt",
                timeout=5.0,
            )
            assert db.fetch_prompt_details(prompt_id)["version"] == 2
            def _status() -> str:
                return _painted(app, screen.query_one("#library-prompt-save-status"))

            await _wait_for_condition(
                pilot,
                lambda: "Saved." in _status(),
                message=lambda: f"no painted Saved receipt; the status line paints {_status()!r}",
            )
            assert screen.focused is name

            await pilot.press("ctrl+s")
            await _until(
                pilot,
                lambda: NOTHING_TO_FLUSH in _status(),
                "a clean prompt's Ctrl+S to say nothing was pending",
            )
            assert db.fetch_prompt_details(prompt_id)["version"] == 2

            # A disabled Save: the key refuses with the reason, writes nothing.
            await _type(pilot, "x")
            await _until(pilot, lambda: screen._prompts_state.dirty, "a new edit")
            screen._prompts_state.mutation_in_flight = True
            screen._prompts_controller._sync_library_prompt_save_action_widgets()
            await pilot.pause()
            assert screen.query_one("#library-prompt-save", Button).disabled
            await pilot.press("ctrl+s")
            await _until(
                pilot,
                lambda: "Save unavailable" in _status(),
                "a disabled Save to refuse Ctrl+S with its reason",
            )
            assert db.fetch_prompt_details(prompt_id)["name"] == "Iota two"
            assert screen.focused is name
            assert not [toast for toast in toasts if "Save" in toast], toasts
            screen._prompts_state.mutation_in_flight = False
            screen._prompts_controller._sync_library_prompt_save_action_widgets()
    db.close_connection()


async def test_ctrl_s_in_folder_files_flushes_the_draft_to_disk(tmp_path):
    """Folder files autosaves after 2 s; Ctrl+S writes the file now. A file
    changed on disk under a dirty draft refuses at the editor, keeps the
    caret, and never overwrites the other writer."""
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
        await _type(pilot, "typed ")
        await _until(pilot, lambda: workspace.save_state == "dirty", "a dirty draft")
        assert note.read_text(encoding="utf-8") == "first line\n"

        await pilot.press("ctrl+s")
        await _until(
            pilot,
            lambda: note.read_text(encoding="utf-8") == editor.text,
            "Ctrl+S to write the draft to disk",
            timeout=5.0,
        )
        assert "ctrl+s save file" in _footer(pilot.app), _footer(pilot.app)
        action = workspace.query_one("#file-notes-action-status")
        await _until(
            pilot, lambda: "Saved." in _painted(pilot.app, action), "the Saved receipt"
        )
        assert screen.focused is editor

        await pilot.press("ctrl+s")
        await _until(
            pilot,
            lambda: NOTHING_TO_FLUSH in _painted(pilot.app, action),
            "a clean file's Ctrl+S to say nothing was pending",
        )

        await _type(pilot, "mine ")
        await _until(pilot, lambda: workspace.save_state == "dirty", "a second draft")
        note.write_text("someone else wrote this\n", encoding="utf-8")
        await pilot.press("ctrl+s")
        await _until(
            pilot,
            lambda: "Save refused" in _painted(pilot.app, action),
            "a changed file to refuse Ctrl+S at the editor",
            timeout=5.0,
        )
        assert note.read_text(encoding="utf-8") == "someone else wrote this\n"
        assert screen.focused is editor
        # The key refuses here, so the footer stops promising "save file".
        await _until(
            pilot,
            lambda: "ctrl+s" not in _footer(pilot.app),
            "the refused save to withdraw the footer's ctrl+s chip",
        )
    await workspace.shutdown()
    replica.close()
