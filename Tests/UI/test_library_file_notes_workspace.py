"""Focused mounted tests for Library File Notes."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static, TextArea, Tree

# Avoid importing the unrelated optional MLX stack during focused UI tests.
sys.modules.setdefault("parakeet_mlx", types.ModuleType("parakeet_mlx"))

import tldw_chatbook.Widgets.Library.library_file_notes_workspace as workspace_module  # noqa: E402
from tldw_chatbook.Library.library_shell_state import (  # noqa: E402
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_NOTES,
)
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica  # noqa: E402
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen  # noqa: E402
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (  # noqa: E402
    LibraryFileNotesWorkspace,
)
from Tests.UI.test_library_shell import (  # noqa: E402
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _seed_conversations,
    _two_conversations,
    _wait_for_library_shell,
)
from Tests.UI.test_screen_navigation import _build_test_app  # noqa: E402


class _WorkspaceHarness(App[None]):
    """Mount one retained workspace without the rest of Library."""

    def __init__(self, workspace: LibraryFileNotesWorkspace) -> None:
        super().__init__()
        self.workspace = workspace

    def compose(self) -> ComposeResult:
        yield self.workspace


async def _wait_until(
    pilot,
    predicate: Callable[[], bool],
    message: str,
    *,
    attempts: int = 150,
) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause(0.02)
    raise AssertionError(message)


def _static_text(workspace: LibraryFileNotesWorkspace, selector: str) -> str:
    renderable = workspace.query_one(selector, Static).renderable
    return getattr(renderable, "plain", str(renderable))


def _tree_labels(tree: Tree) -> list[str]:
    labels: list[str] = []

    def visit(node) -> None:
        label = getattr(node.label, "plain", str(node.label))
        labels.append(label)
        for child in node.children:
            visit(child)

    visit(tree.root)
    return labels


def _replace_editor_text(editor: TextArea, text: str) -> None:
    editor.select_all()
    editor.replace(text, editor.selection.start, editor.selection.end)


@pytest.mark.asyncio
async def test_empty_offline_and_persisted_root_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replica = FileNotesReplica(":memory:")
    empty = LibraryFileNotesWorkspace(root=None, replica=replica)
    async with _WorkspaceHarness(empty).run_test() as pilot:
        await pilot.pause()
        assert empty.query_one("#file-notes-choose-root", Button).display
        assert (
            _static_text(empty, "#file-notes-root-status") == "Choose a notes folder."
        )

        root = tmp_path / "chosen"
        root.mkdir()
        saved: list[tuple[str, str, str]] = []
        monkeypatch.setattr(
            workspace_module,
            "save_setting_to_cli_config",
            lambda section, key, value: saved.append((section, key, value)) or True,
        )
        assert await empty.set_root(root)
        assert saved == [("file_notes", "root", str(root.resolve()))]
    replica.close()

    missing_root = tmp_path / "missing"
    offline_replica = FileNotesReplica(":memory:")
    offline = LibraryFileNotesWorkspace(root=missing_root, replica=offline_replica)
    async with _WorkspaceHarness(offline).run_test() as pilot:
        await _wait_until(
            pilot,
            lambda: offline.initialized,
            "offline root scan did not finish",
        )
        assert not missing_root.exists()
        assert "Offline" in _static_text(offline, "#file-notes-root-status")
    offline_replica.close()

    persisted_root = tmp_path / "persisted"
    persisted_root.mkdir()
    (persisted_root / "kept.md").write_text("persisted body", encoding="utf-8")
    monkeypatch.setattr(
        workspace_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            str(persisted_root) if (section, key) == ("file_notes", "root") else default
        ),
    )
    persisted_replica = FileNotesReplica(":memory:")
    persisted = LibraryFileNotesWorkspace(replica=persisted_replica)
    async with _WorkspaceHarness(persisted).run_test() as pilot:
        await _wait_until(
            pilot,
            lambda: persisted.initialized and "kept.md" in persisted.entries,
            "persisted root was not scanned",
        )
        assert persisted.root == persisted_root.resolve()
        assert "kept.md" in _tree_labels(persisted.query_one("#file-notes-tree", Tree))
    persisted_replica.close()


@pytest.mark.asyncio
async def test_tree_search_open_dirty_and_autosave_keep_one_editor(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    (root / "folder").mkdir(parents=True)
    (root / "folder" / "alpha.md").write_text(
        "needle in this body\n",
        encoding="utf-8",
    )
    (root / "beta.txt").write_text("other body", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        autosave_delay=0.08,
        poll_interval=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=(110, 36)) as pilot:
        await _wait_until(
            pilot,
            lambda: workspace.initialized and len(workspace.entries) == 2,
            "initial tree did not load",
        )
        tree = workspace.query_one("#file-notes-tree", Tree)
        assert {"folder", "alpha.md", "beta.txt"}.issubset(_tree_labels(tree))

        search = workspace.query_one("#file-notes-search", Input)
        search.value = "needle"
        await _wait_until(
            pilot,
            lambda: workspace.query_one("#file-notes-search-results", Tree).display,
            "search results did not replace the tree",
        )
        assert not tree.display
        results = workspace.query_one("#file-notes-search-results", Tree)
        assert "folder/alpha.md" in _tree_labels(results)
        editor = workspace.query_one("#file-notes-editor", TextArea)
        match = next(
            node
            for node in results.root.children
            if node.data == ("file", "folder/alpha.md")
        )
        results.select_node(match)
        await _wait_until(
            pilot,
            lambda: workspace.current_path == "folder/alpha.md",
            "selecting the visible search result did not open its file",
        )
        assert editor.text == "needle in this body\n"

        search.value = ""
        await _wait_until(pilot, lambda: tree.display, "tree did not return")

        assert workspace.query_one("#file-notes-editor", TextArea) is editor
        assert workspace.save_state == "saved"

        _replace_editor_text(editor, "changed body")
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "dirty",
            "body edit did not become dirty",
        )
        assert not workspace.leave_allowed
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "saved",
            "debounced autosave did not complete",
        )
        assert (root / "folder" / "alpha.md").read_text(encoding="utf-8") == (
            "changed body\n"
        )
        assert workspace.query_one("#file-notes-editor", TextArea) is editor
    replica.close()


@pytest.mark.asyncio
async def test_create_move_delete_protect_and_restore_use_real_service(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "start.md").write_text("start", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        poll_interval=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("start.md")

        path_input = workspace.query_one("#file-notes-path", Input)
        path_input.value = "created.md"
        workspace.query_one("#file-notes-new", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.current_path == "created.md",
            "new file did not open",
        )
        assert (root / "created.md").exists()

        workspace.query_one("#file-notes-protect", Button).press()
        await _wait_until(
            pilot,
            lambda: (
                workspace.current_document is not None
                and workspace.current_document.protected
            ),
            "protect did not apply",
        )
        assert str(workspace.query_one("#file-notes-protect", Button).label) == (
            "Unprotect"
        )
        workspace.query_one("#file-notes-protect", Button).press()
        await _wait_until(
            pilot,
            lambda: (
                workspace.current_document is not None
                and not workspace.current_document.protected
            ),
            "unprotect did not apply",
        )

        path_input.value = "moved.md"
        workspace.query_one("#file-notes-move", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.current_path == "moved.md",
            "move did not open destination",
        )
        assert not (root / "created.md").exists()
        assert (root / "moved.md").exists()

        workspace.query_one("#file-notes-delete", Button).press()
        await _wait_until(
            pilot,
            lambda: (
                str(workspace.query_one("#file-notes-delete", Button).label)
                == "Confirm delete"
            ),
            "delete confirmation did not arm",
        )
        assert (root / "moved.md").exists()
        assert str(workspace.query_one("#file-notes-delete", Button).label) == (
            "Confirm delete"
        )
        workspace.query_one("#file-notes-delete", Button).press()
        await _wait_until(
            pilot,
            lambda: not (root / "moved.md").exists(),
            "confirmed delete did not remove the file",
        )
        assert "Recently deleted" in _tree_labels(
            workspace.query_one("#file-notes-tree", Tree)
        )

        workspace.query_one("#file-notes-restore", Button).press()
        await _wait_until(
            pilot,
            lambda: (root / "moved.md").exists(),
            "restore did not recreate exact file",
        )
        changes = _static_text(workspace, "#file-notes-session-changes")
        assert all(
            action in changes for action in ("created", "moved", "deleted", "restored")
        )
    replica.close()


@pytest.mark.asyncio
async def test_conflict_reload_save_copy_and_leave_guards_preserve_draft(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    source = root / "source.md"
    source.write_bytes(b"\xef\xbb\xbf---\r\ntitle: Exact\r\n---\r\nold\r\nbody\r\n")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        autosave_delay=10,
        poll_interval=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("source.md")
        first_session = workspace.session_key
        editor = workspace.query_one("#file-notes-editor", TextArea)
        _replace_editor_text(editor, "dirty reload guard")
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "dirty",
            "dirty Reload setup did not arm",
        )
        workspace.query_one("#file-notes-reload", Button).press()
        await _wait_until(
            pilot,
            lambda: (
                workspace.save_state == "saved"
                and editor.text == "dirty reload guard\n"
            ),
            "Reload discarded a merely dirty draft instead of flushing it",
        )
        assert source.read_bytes().endswith(b"dirty reload guard\r\n")

        _replace_editor_text(editor, "kept\ndraft")
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "dirty",
            "draft did not become dirty",
        )

        source.write_bytes(b"\xef\xbb\xbf---\r\ntitle: External\r\n---\r\nexternal\r\n")
        await workspace.refresh_files()
        assert workspace.save_state == "conflict"
        assert editor.text == "kept\ndraft"
        assert not await workspace.flush_pending_work()

        workspace.query_one("#file-notes-path", Input).value = "copy.md"
        workspace.query_one("#file-notes-save-copy", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.current_path == "copy.md",
            "save copy did not open the safe copy",
        )
        assert (root / "copy.md").read_bytes() == (
            b"\xef\xbb\xbf---\r\ntitle: Exact\r\n---\r\nkept\r\ndraft\r\n"
        )

        assert await workspace.open_path("source.md")
        _replace_editor_text(editor, "another draft")
        await pilot.pause()
        source.write_bytes(
            b"\xef\xbb\xbf---\r\ntitle: External 2\r\n---\r\nreload me\r\n"
        )
        await workspace.refresh_files()
        assert workspace.save_state == "conflict"
        workspace.query_one("#file-notes-reload", Button).press()
        await _wait_until(
            pilot,
            lambda: (
                workspace.save_state == "saved"
                and workspace.query_one("#file-notes-editor", TextArea).text
                == "reload me\n"
            ),
            "reload did not resolve the conflict",
        )
        assert workspace.session_key != first_session
        assert workspace.query_one("#file-notes-editor", TextArea) is editor

        _replace_editor_text(editor, "flush me")
        await pilot.pause()
        assert await workspace.flush_pending_work()
        assert source.read_bytes().endswith(b"flush me\r\n")

        workspace.query_one("#file-notes-protect", Button).press()
        await _wait_until(
            pilot,
            lambda: (
                workspace.current_document is not None
                and workspace.current_document.protected
            ),
            "protected error setup did not finish",
        )
        replica.close()
        _replace_editor_text(editor, "surviving error draft")
        await pilot.pause()
        assert not await workspace.flush_pending_work()
        assert workspace.save_state == "error"
        assert editor.text == "surviving error draft"
    replica.close()


@pytest.mark.asyncio
async def test_recently_deleted_survives_a_second_workspace(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    original = b"recover me exactly\r\n"
    (root / "recover.md").write_bytes(original)
    replica_path = tmp_path / "file_notes.sqlite"

    first_replica = FileNotesReplica(replica_path)
    first = LibraryFileNotesWorkspace(
        root=root,
        replica=first_replica,
        poll_interval=10,
    )
    async with _WorkspaceHarness(first).run_test() as pilot:
        await _wait_until(pilot, lambda: first.initialized, "first scan did not finish")
        assert await first.open_path("recover.md")
        first.query_one("#file-notes-delete", Button).press()
        await _wait_until(
            pilot,
            lambda: (
                str(first.query_one("#file-notes-delete", Button).label)
                == "Confirm delete"
            ),
            "delete confirmation did not arm",
        )
        first.query_one("#file-notes-delete", Button).press()
        await _wait_until(
            pilot,
            lambda: not (root / "recover.md").exists(),
            "delete did not finish",
        )
    first_replica.close()

    second_replica = FileNotesReplica(replica_path)
    second = LibraryFileNotesWorkspace(
        root=root,
        replica=second_replica,
        poll_interval=10,
    )
    async with _WorkspaceHarness(second).run_test() as pilot:
        await _wait_until(
            pilot, lambda: second.initialized, "second scan did not finish"
        )
        assert "Recently deleted" in _tree_labels(
            second.query_one("#file-notes-tree", Tree)
        )
        assert second.select_deleted("recover.md")
        second.query_one("#file-notes-restore", Button).press()
        await _wait_until(
            pilot,
            lambda: (root / "recover.md").exists(),
            "second workspace could not restore tombstone",
        )
        assert (root / "recover.md").read_bytes() == original
    second_replica.close()


@pytest.mark.asyncio
async def test_poll_and_narrow_navigation_retain_the_text_area(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "open.md").write_text("first", encoding="utf-8")
    (root / "delete.md").write_text("gone soon", encoding="utf-8")
    (root / "folder").mkdir()
    (root / "folder" / "nested.md").write_text("nested", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        poll_interval=0.05,
        autosave_delay=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=(64, 28)) as pilot:
        await _wait_until(
            pilot,
            lambda: workspace.initialized and workspace.narrow,
            "workspace did not choose narrow mode from its width",
        )
        editor = workspace.query_one("#file-notes-editor", TextArea)
        assert workspace.navigator_visible
        assert not workspace.editor_visible
        tree = workspace.query_one("#file-notes-tree", Tree)
        folder = next(
            node
            for node in tree.root.children
            if getattr(node.label, "plain", str(node.label)) == "folder"
        )
        folder.expand()

        assert await workspace.open_path("open.md")
        assert workspace.editor_visible
        assert not workspace.navigator_visible
        assert workspace.query_one("#file-notes-editor", TextArea) is editor

        (root / "open.md").write_text("external", encoding="utf-8")
        (root / "created.md").write_text("new", encoding="utf-8")
        (root / "delete.md").unlink()
        await _wait_until(
            pilot,
            lambda: (
                set(workspace.entries) == {"created.md", "folder/nested.md", "open.md"}
                and editor.text == "external"
            ),
            "poll did not reconcile external create/modify/delete",
        )
        assert workspace.query_one("#file-notes-editor", TextArea) is editor
        refreshed_folder = next(
            node
            for node in workspace.query_one("#file-notes-tree", Tree).root.children
            if getattr(node.label, "plain", str(node.label)) == "folder"
        )
        assert refreshed_folder.is_expanded
        await pilot.pause(0.15)
        assert len(workspace._workers) <= 1

        workspace.query_one("#file-notes-back", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.navigator_visible and not workspace.editor_visible,
            "Back did not return to the retained navigator",
        )
    replica.close()


@pytest.mark.asyncio
async def test_library_database_files_switch_retains_workspace_and_database_canvas(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "library.md").write_text("library file", encoding="utf-8")
    (root / "other.md").write_text("other file", encoding="utf-8")
    replacement_root = tmp_path / "replacement"
    replacement_root.mkdir()
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        poll_interval=10,
        autosave_delay=10,
    )
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[{"title": "Database note", "id": "db-note-1"}],
    )
    screen = LibraryScreen(
        app,
        file_notes_workspace_factory=lambda: workspace,
    )
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#library-notes-canvas")),
            "Database Notes canvas did not compose",
        )
        assert screen.query_one("#library-rail")
        assert screen.query_one("#library-notes-source-strip")
        assert screen._library_file_notes_workspace is None

        screen.query_one("#library-notes-source-files", Button).press()
        await _wait_until(
            pilot,
            lambda: screen._library_notes_source == "files",
            "Files source handler did not run",
        )
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#library-file-notes-workspace")),
            "Files workspace did not replace the Database rail/canvas",
        )
        retained = screen.query_one(
            "#library-file-notes-workspace",
            LibraryFileNotesWorkspace,
        )
        editor = retained.query_one("#file-notes-editor", TextArea)
        assert retained is workspace
        assert not screen.query("#library-rail")

        screen._apply_local_source_snapshot(
            {
                "notes": ({"title": "Updated DB note", "id": "db-note-2"},),
                "media": (),
                "conversations": (),
            },
            {"notes": 1, "media": 0, "conversations": 0},
            {"notes": True, "media": True, "conversations": True},
        )
        await pilot.pause()
        assert screen.query_one("#library-file-notes-workspace") is retained
        assert retained.query_one("#file-notes-editor", TextArea) is editor

        await retained.open_path("library.md")
        _replace_editor_text(editor, "draft")
        await pilot.pause()
        (root / "library.md").write_text("external", encoding="utf-8")
        await retained.refresh_files()
        assert retained.save_state == "conflict"

        assert not await retained.open_path("other.md")
        assert retained.current_path == "library.md"
        assert editor.text == "draft"

        assert not await retained.set_root(replacement_root, persist=False)
        assert retained.root == root.resolve()
        assert editor.text == "draft"

        assert not await screen.flush_pending_work()
        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES
        assert screen.query_one("#library-file-notes-workspace") is retained

        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_MEDIA)
        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES
        assert screen.query_one("#library-file-notes-workspace") is retained
        assert editor.text == "draft"

        screen.query_one("#library-notes-source-database", Button).press()
        await pilot.pause()
        assert screen._library_notes_source == "files"
        assert screen.query_one("#library-file-notes-workspace") is retained

        retained.query_one("#file-notes-reload", Button).press()
        await _wait_until(
            pilot,
            lambda: retained.save_state == "saved",
            "reload did not clear the source-switch veto",
        )
        _replace_editor_text(editor, "saved before hiding")
        await _wait_until(
            pilot,
            lambda: retained.save_state == "dirty",
            "pre-remount edit did not become dirty",
        )
        assert await retained.flush_pending_work()
        assert "modified library.md" in _static_text(
            retained,
            "#file-notes-session-changes",
        )
        screen.query_one("#library-notes-source-database", Button).press()
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#library-notes-canvas")),
            "Database Notes did not return",
        )
        assert screen.query_one("#library-rail")
        assert screen._local_source_records["notes"][0]["title"] == "Updated DB note"

        (root / "library.md").write_text("changed while hidden", encoding="utf-8")
        screen.query_one("#library-notes-source-files", Button).press()
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#library-file-notes-workspace")),
            "retained Files workspace did not remount",
        )
        assert screen.query_one("#library-file-notes-workspace") is retained
        assert retained.query_one("#file-notes-editor", TextArea) is editor
        await _wait_until(
            pilot,
            lambda: editor.text == "changed while hidden",
            "remount did not reconcile the retained open file",
        )
        assert "modified library.md" in _static_text(
            retained,
            "#file-notes-session-changes",
        )
    replica.close()
