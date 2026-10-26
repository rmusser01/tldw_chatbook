"""Notes focus capture inside a recompose, and focus after a failed or kept
export (TASK-34000.29, review R0 of polish-1b Task 6).

- Ruling T6-a: while a Notes canvas recompose prunes the focused control,
  Textual parks focus on the control its ``_reset_focus`` picks until the
  replay runs. A focus capture taken then must record the pruned control,
  but only while focus is still on that landing (or on nothing). A move the
  user made elsewhere inside the window is captured as made.
- Ruling T6-b: keeping the existing file at the replace prompt is the user
  cancelling. It leaves no failure-phase line, shows one toast, and returns
  focus to the button that started the export.
- AC#2's "fails" arm: after a failed write, focus is back on that button too.

Two boots of the real ``LibraryScreen`` with the production stylesheet over a
real ChaChaNotes database (``Tests/UI/test_library_note_editor_feedback_is_
honest.py``'s profile). The capture window is entered deterministically: the
work pane's ``compose`` runs after Textual has pruned the old children and
moved focus, and before the new ones mount. The capture is taken there.
"""

from __future__ import annotations

import hashlib

import pytest
from textual.widgets import Button

from Tests.UI.test_library_note_editor_feedback_is_honest import (
    EXPORT_MD,
    IDEAS,
    SIZE,
    _back_on_screen,
    _export_dialog,
    _focus_settles_on,
    _focused_id,
    _info,
    _library,
    _mounted,
    _painted,
    _painted_of,
    _wait,
)
from Tests.UI.library_quit_guard_support import _open_note
from tldw_chatbook.Library.library_notes_state import LibraryNotesFocusIdentity
from tldw_chatbook.Third_Party.textual_fspicker.file_dialog import FileNameInput
from tldw_chatbook.UI.Library_Modules.canvas_sync import _sync_library_canvas
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

EXPORT_ROLE = f"context-action:{EXPORT_MD}"


async def _tab_to(screen, pilot, control: str) -> None:
    """Walk focus to ``control`` with Tab, as the keyboard user does."""
    for _ in range(12):
        if _focused_id(screen) == control:
            return
        await pilot.press("tab")
        await pilot.pause()
    assert _focused_id(screen) == control, _focused_id(screen)


async def _capture_inside_the_prune(screen, pilot, monkeypatch, *, user_moves_to=None):
    """Resync the Notes canvases; capture focus while the work pane is between
    pruning its children and mounting their replacements.

    Returns ``(focus Textual left, predicted landing, focus at the capture,
    captured identity)``. ``user_moves_to`` is focused first (public
    ``set_focus``, as a click does) to stand for the user's own move inside
    the window.
    """
    pane = screen.query_one("#library-note-work-pane")
    real_compose = pane.compose
    seen: dict = {}

    def compose_after_the_prune():
        seen["landing"] = screen.focused
        pruned = pane.focus_before_prune
        seen["predicted"] = pruned[1] if isinstance(pruned, tuple) else None
        if user_moves_to is not None:
            screen.set_focus(user_moves_to)
        seen["captured_on"] = screen.focused
        seen["identity"] = screen._capture_library_notes_focus_identity()
        monkeypatch.setattr(pane, "compose", real_compose)
        yield from real_compose()

    monkeypatch.setattr(pane, "compose", compose_after_the_prune)
    _sync_library_canvas(screen, "notes")
    await _wait(pilot, lambda: "identity" in seen, "the work pane to recompose")
    return seen["landing"], seen["predicted"], seen["captured_on"], seen["identity"]


async def test_a_capture_in_a_prune_keeps_the_pruned_control_only_on_textuals_landing(
    tmp_path, monkeypatch
) -> None:
    """Ruling T6-a, both directions, plus the ``use-in-console`` role.

    1. Focus on Info's Export Markdown; a Notes resync recomposes the work
       pane. Inside the window, focus is on Textual's landing and the capture
       names Export Markdown, so the replay puts focus back there and
       Shift+Tab (a real key) walks on to Copy.
    2. The same window, but the user's own move lands on a control with no
       portable role first: the capture does not claim Export Markdown.
    """
    host, seeded, _toasts, _copies = _mounted(tmp_path, monkeypatch)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _library(host, pilot)
        await _open_note(screen, pilot, IDEAS, seeded.ideas_id)
        await _info(host, screen, pilot)
        await _tab_to(screen, pilot, EXPORT_MD)

        landing, predicted, captured_on, identity = await _capture_inside_the_prune(
            screen, pilot, monkeypatch
        )
        assert getattr(landing, "id", None) != EXPORT_MD
        assert captured_on is landing
        assert identity.semantic_role == EXPORT_ROLE, identity
        await _focus_settles_on(
            host, screen, pilot, EXPORT_MD, "export Markdown", after="the resync"
        )
        await pilot.press("shift+tab")
        await _focus_settles_on(
            host, screen, pilot, "library-note-context-copy", "copy note",
            after="Shift+Tab",
        )
        await pilot.press("tab")
        await _focus_settles_on(
            host, screen, pilot, EXPORT_MD, "export Markdown", after="Tab"
        )

        # The rail's search box is collapsed away while a note is open at
        # this size, so the user's role-less landing is the "Library notes"
        # source toggle above the panes (outside both recomposing canvases).
        toggle = screen.query_one("#library-notes-source-database", Button)
        assert toggle.focusable
        assert screen._library_notes_semantic_role(toggle) == ""
        _landing, _predicted, captured_on, identity = await _capture_inside_the_prune(
            screen, pilot, monkeypatch, user_moves_to=toggle
        )
        assert captured_on is toggle
        assert identity.semantic_role != EXPORT_ROLE, identity
        # The window was the one the rule names: Textual's own reset put
        # focus on exactly the landing the canvas predicted.
        assert landing is None or landing is predicted, (landing, predicted)

        # The header Use in Console button's portable role, both halves.
        console = screen.query_one("#library-note-use-in-console", Button)
        assert screen._library_notes_semantic_role(console) == "use-in-console"
        role_target = screen._notes_controller._library_notes_role_target(
            LibraryNotesFocusIdentity(
                stage="notes",
                region="context",
                note_id=seeded.ideas_id,
                semantic_role="use-in-console",
            )
        )
        assert role_target is console


async def test_a_failed_or_kept_export_returns_focus_and_keep_is_a_cancel(
    tmp_path, monkeypatch
) -> None:
    """AC#2's failure arm and ruling T6-b.

    A write into a folder that does not exist fails honestly ("Export
    failed") and focus is back on Export Markdown, where Enter (a real key)
    re-opens the picker. Then keeping an existing file at the replace prompt
    is a cancel: one toast, no failure or cancel line in the status
    surfaces, the file's bytes unchanged, focus on Export Markdown.
    """
    host, seeded, toasts, _copies = _mounted(tmp_path, monkeypatch)
    folder = tmp_path / "exports"
    folder.mkdir()
    kept = folder / "ideas.md"
    kept.write_bytes(b"PRECIOUS USER FILE\n")
    before = hashlib.md5(kept.read_bytes()).hexdigest()
    async with host.run_test(size=SIZE) as pilot:
        screen = await _library(host, pilot)
        await _open_note(screen, pilot, IDEAS, seeded.ideas_id)
        await _info(host, screen, pilot)
        await _tab_to(screen, pilot, EXPORT_MD)

        # Failed: the folder does not exist.
        await pilot.press("enter")
        dialog = await _export_dialog(host, pilot)
        dialog.query_one(FileNameInput).value = str(tmp_path / "missing" / "ideas.md")
        await pilot.press("enter")
        await _back_on_screen(host, screen, pilot)
        await _wait(
            pilot,
            lambda: "Export failed" in _painted(host),
            lambda: f"the failed export to report (toasts: {toasts!r})",
        )
        assert any("Error exporting note" in toast for toast in toasts), toasts
        await _focus_settles_on(
            host, screen, pilot, EXPORT_MD, "export Markdown",
            after="the failed export",
        )

        # Kept: Enter re-opens the picker; the replace prompt's Cancel keeps.
        toasts.clear()
        await pilot.press("enter")
        dialog = await _export_dialog(host, pilot)
        dialog.query_one(FileNameInput).value = str(kept)
        await pilot.press("enter")
        await _wait(
            pilot,
            lambda: isinstance(host.screen, ConfirmationDialog)
            and host.screen.focused is not None,
            "the replace prompt",
        )
        assert host.screen.focused.id == "cancel-button"
        await pilot.press("escape")
        await _back_on_screen(host, screen, pilot)
        await _wait(
            pilot,
            lambda: bool(toasts) and "Export…" not in _painted(host),
            lambda: f"the kept export to settle (toasts: {toasts!r})",
        )
        painted = _painted(host)
        assert "Export cancelled" not in painted, painted
        assert "Export failed" not in painted, painted
        assert _painted_of(host, screen, "#library-note-context-transfer-status") == ""
        assert toasts == ["Note export cancelled. ideas.md was left unchanged."]
        assert hashlib.md5(kept.read_bytes()).hexdigest() == before
        await _focus_settles_on(
            host, screen, pilot, EXPORT_MD, "export Markdown", after="Keep"
        )
        assert sorted(path.name for path in folder.iterdir()) == ["ideas.md"]
