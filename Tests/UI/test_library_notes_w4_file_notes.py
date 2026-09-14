"""Folder files header, keys, hidden folders and Escape (task-32543, task-32552).

Critique #3 of Library ▸ Notes, Folder files workflow, both assessors:

* the canvas authority line said "Git · 1 change" on a folder that is not a
  Git repository (task-32543);
* Ctrl+End was inert in the Folder files editor, ``.trash`` was listed while
  ``.obsidian`` was hidden, an embed wrapped mid-token at 100x30, and Escape
  from the editor dropped the whole mode in one press (task-32552).

The header pins here run the REAL Git service against a real throwaway
repository (and a plain folder beside it) so the "confirmed repository" fact
is ``git rev-parse`` truth, pinned both ways.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.cells import cell_len
from textual.widgets import Input, TextArea, Tree

# Stubs first in the local group: it registers the optional MLX modules the
# application imports below would otherwise probe.
import Tests.UI._optional_module_stubs  # noqa: F401
from Tests.Notes.test_file_notes_git_integration import _disposable_repository
from Tests.UI.test_library_file_notes_workspace import (
    _production_workspace_context,
    _static_text,
    _wait_until,
    _WorkspaceHarness,
)
from tldw_chatbook.Notes.file_notes_git_service import FileNotesGitService
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Notes.file_notes_session_owner import (
    FileNotesSessionOwner,
    SessionChange,
)
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    LibraryFileNotesWorkspace,
)

WIDE = (235, 52)
CRITIQUE_COMPACT = (100, 30)
EMBED = "![[attachments/diagram.png]]"


def _long_note(lines: int = 400) -> str:
    return "\n".join(f"line {index} of a long daily note" for index in range(lines)) + "\n"


async def _open(pilot, workspace: LibraryFileNotesWorkspace, relative_path: str) -> TextArea:
    assert await workspace.open_path(relative_path)
    await _wait_until(
        pilot,
        lambda: workspace.current_path == relative_path,
        f"{relative_path} did not open",
    )
    editor = workspace.query_one("#file-notes-editor", TextArea)
    editor.focus()
    await pilot.pause()
    assert pilot.app.focused is editor
    return editor


# --- task-32552 AC#1 ---------------------------------------------------------


@pytest.mark.asyncio
async def test_ctrl_end_and_ctrl_home_move_the_folder_files_caret_and_the_footer_advertises_them(
    tmp_path: Path,
) -> None:
    """Critique B D9: typed text landed at the click; the key had no binding.

    Same defect task-32247 fixed one editor over: Textual's ``TextArea`` binds
    ``end`` to the LINE end and defines no ``ctrl+end`` at all. The Folder
    files editor was a plain ``TextArea``.
    """
    root = tmp_path / "vault"
    root.mkdir()
    (root / "long.md").write_text(_long_note(), encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=root, replica=replica)
    async with _production_workspace_context(workspace, size=WIDE) as pilot:
        screen = pilot.app.screen
        editor = await _open(pilot, workspace, "long.md")
        editor.move_cursor((0, 0))
        await pilot.pause()

        await pilot.press("ctrl+end")
        await pilot.pause()
        assert editor.cursor_location == editor.document.end, (
            f"Ctrl+End left the caret at {editor.cursor_location} of "
            f"{editor.document.end}"
        )

        await pilot.press("ctrl+home")
        await pilot.pause()
        assert editor.cursor_location == (0, 0), (
            f"Ctrl+Home left the caret at {editor.cursor_location}"
        )

        shortcuts = dict(screen._library_footer_shortcuts_for_current_state())
        assert shortcuts.get("ctrl+end") == "end of file", shortcuts
    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_the_folder_files_chips_follow_editor_focus_in_the_registered_footer(
    tmp_path: Path,
) -> None:
    """task-32552 AC#1/#3: both chips are editor-focus facts, and the footer
    that is actually REGISTERED has to follow them.

    Two halves, both found live rather than in the resolver:

    * the Ctrl+End chip was gated on an open file, so it stood after a
      folder change had closed one -- the gate is the editor's focus;
    * the registered set is re-applied only when
      ``_refresh_footer_typing_context``'s context tuple flips, and a move
      from the editor to the "File contents…" box flips neither the typing
      flag (both are text widgets) nor the Enter label, so the stale
      ``esc files`` chip survived a move out of the editor.

    This reads ``_footer_shortcut_registration`` -- what the footer was
    handed -- not the freshly computed set, so the flip gate is in scope.
    """
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("a note\n", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=root, replica=replica)

    def registered() -> dict[str, str]:
        registration = pilot.app.screen._footer_shortcut_registration
        assert registration is not None and registration[0] == "library"
        return dict(registration[1])

    async with _production_workspace_context(workspace, size=WIDE) as pilot:
        await _open(pilot, workspace, "note.md")
        assert registered().get("ctrl+end") == "end of file", registered()
        assert registered().get("esc") == "files", registered()

        workspace.query_one("#file-notes-search", Input).focus()
        await _wait_until(
            pilot,
            lambda: registered().get("esc") == "notes",
            f"the registered footer kept the editor's Escape chip: {registered()}",
        )
        assert "ctrl+end" not in registered(), registered()
    await workspace.shutdown()
    replica.close()


# --- task-32552 AC#3 ---------------------------------------------------------


@pytest.mark.asyncio
async def test_escape_in_the_folder_files_editor_returns_to_the_tree_then_leaves_the_mode(
    tmp_path: Path,
) -> None:
    """Critique A 58: one Escape from the editor dropped the whole mode.

    The screen's ``library_notes_files_back`` binding was the only Escape
    on this surface. The editor now steps back to the files tree first, and
    the footer names each step: ``esc files`` in the editor, ``esc notes``
    on the tree.
    """
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("a note\n", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=root, replica=replica)
    async with _production_workspace_context(workspace, size=WIDE) as pilot:
        screen = pilot.app.screen
        await _open(pilot, workspace, "note.md")
        assert ("esc", "files") in screen._library_footer_shortcuts_for_current_state()

        await pilot.press("escape")
        await pilot.pause()
        tree = workspace.query_one("#file-notes-tree", Tree)
        assert screen._notes_state.source == "files", "one Escape left Folder files"
        assert pilot.app.focused is tree, pilot.app.focused
        assert ("esc", "notes") in screen._library_footer_shortcuts_for_current_state()

        await pilot.press("escape")
        await _wait_until(
            pilot,
            lambda: screen._notes_state.source == "database",
            "the second Escape did not leave Folder files",
        )
    await workspace.shutdown()
    replica.close()


# --- task-32552 AC#4 (qualified: lines no longer than the editor pane) ------


@pytest.mark.asyncio
async def test_no_line_the_editor_pane_can_hold_wraps_at_100x30(
    tmp_path: Path,
) -> None:
    """Critique A 57: ``![[attachments/diagram.png]]`` wrapped mid-token.

    Measured live on this branch (capture ``fn-19-embed-wrap-100x30.txt``):
    at 100x30 the Folder files editor frame is 32 cells, and a document
    taller than the pane paints a vertical scrollbar inside it, leaving a
    27-cell wrap width. The embed is 28 cells, so NO row can hold it and it
    splits as ``![[attachments/diagram.png]`` / ``]`` -- which is why the
    controller qualified AC#4 to lines the pane can hold. (In a short
    document there is no scrollbar, the wrap width is 31, and the same embed
    renders on one row: capture ``fn-03-embed-wrap-100x30.txt``.)

    So this pins the qualified property at the REAL width, scrollbar and
    all: every line the pane can hold renders on one row. It does not pin
    the pane's width -- ``LIBRARY_FILE_NOTES_READER_PROFILE.work_min_width``
    (30) is what leaves the reader at its floor while the files tree holds
    ~59 columns, and widening that is a layout decision, not this task's.
    """
    root = tmp_path / "vault"
    (root / "Daily").mkdir(parents=True)
    body = "\n".join(f"line {index} of a daily note" for index in range(60))
    (root / "Daily" / "2026-09-07.md").write_text(
        f"# Daily\n\nA short line.\n\n{EMBED}\n{body}\n", encoding="utf-8"
    )
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=root, replica=replica)
    async with _production_workspace_context(workspace, size=CRITIQUE_COMPACT) as pilot:
        editor = await _open(pilot, workspace, "Daily/2026-09-07.md")
        await pilot.pause()
        wrap_width = editor.wrap_width
        assert wrap_width > 0, "the editor pane never laid out at 100x30"
        assert editor.wrapped_document.height > editor.content_region.height, (
            "this document fits the pane, so no scrollbar narrows it -- the "
            "pin would not measure the width the critique saw"
        )
        for index, line in enumerate(editor.text.splitlines()):
            if cell_len(line) > wrap_width:
                continue
            assert editor.wrapped_document.get_offsets(index) == [], (
                f"line {index} is {cell_len(line)} cells and the pane holds "
                f"{wrap_width}, yet it wrapped: {line!r}"
            )
    await workspace.shutdown()
    replica.close()


# --- task-32543: "Git · N change(s)" only on a confirmed repository ---------


@pytest.mark.parametrize("repository", (False, True))
@pytest.mark.asyncio
async def test_the_header_says_git_only_when_the_folder_is_a_repository(
    tmp_path: Path,
    repository: bool,
) -> None:
    """Both ways, by ``git rev-parse``: plain folder → "N session change(s)";
    repository → "Git · N change(s)" (task-32543 AC#1/#2)."""
    repo = _disposable_repository(tmp_path)
    if repository:
        root = repo.path
    else:
        root = tmp_path / "plain"
        root.mkdir()
    (root / "note.md").write_text("a note\n", encoding="utf-8")
    owner = FileNotesSessionOwner()
    owner.attach_git_service(
        FileNotesGitService(
            owner,
            git_executable=repo.git,
            environment=repo.service_environment,
        )
    )
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        session_owner=owner,
        poll_interval=10,
    )
    async with _WorkspaceHarness(workspace).run_test(size=WIDE) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        binding = workspace._session_binding
        assert binding is not None
        assert owner.record_change(binding, SessionChange("modified", "note.md"))
        workspace._render_session_git_label()
        expected = "Git · 1 change" if repository else "1 session change"
        await _wait_until(
            pilot,
            lambda: expected in _static_text(workspace, "#file-notes-authority"),
            f"authority never read {expected!r}: "
            f"{_static_text(workspace, '#file-notes-authority')!r}",
        )
        authority = _static_text(workspace, "#file-notes-authority")
        assert authority.startswith("Folder files · Folder: ")
        if not repository:
            assert "Git" not in authority
    await workspace.shutdown()
    replica.close()
