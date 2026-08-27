"""Focused mounted tests for Library File Notes."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from itertools import product
from pathlib import Path
from time import perf_counter
from unittest.mock import AsyncMock, patch

import pytest
from rich.cells import cell_len
from textual.app import App, ComposeResult
from textual.screen import ModalScreen, Screen

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.color import Color
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Input, Static, TextArea, Tree

import Tests.UI._optional_module_stubs  # noqa: F401
import tldw_chatbook.Widgets.Library.library_file_notes_workspace as workspace_module  # noqa: E402
from tldw_chatbook.config import ConfigMutationResult  # noqa: E402
from tldw_chatbook.css.Themes.themes import ALL_THEMES  # noqa: E402
from tldw_chatbook.Library.library_shell_state import (  # noqa: E402
    LIBRARY_DISABLED_ACTION_MARKER,
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_NOTES,
)
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica  # noqa: E402
from tldw_chatbook.Notes.file_notes_session_owner import (  # noqa: E402
    FileNotesSessionOwner,
    SessionChange,
)
from tldw_chatbook.Notes.file_notes_service import (  # noqa: E402
    INTERACTIVE_FILE_CHARS,
    LARGE_FILE_EXCERPT_CHARS,
    FileNoteEntry,
    FileNotesService,
    OperationResult,
    ReconcileResult,
    ScanResult,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen  # noqa: E402
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (  # noqa: E402
    FileNotesConflictCompareDialog,
    LibraryFileNotesWorkspace,
)
from tldw_chatbook.Widgets.Library.library_file_notes_git_panel import (  # noqa: E402
    LibraryFileNotesGitPanel,
    PushPanelResultProjection,
)
from tldw_chatbook.Widgets.Library.library_adaptive_reader_shell import (  # noqa: E402
    LibraryAdaptiveReaderShell,
)
from Tests.UI.test_library_shell import (  # noqa: E402
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _seed_conversations,
    _two_conversations,
    _wait_for_library_shell,
)
from Tests.UI.app_factory import _build_test_app as _build_tldw_test_app  # noqa: E402


def _build_test_app(*args, **kwargs):
    """Force the legacy (graduated) Library profile this suite assumes.

    TASK-19602: under the pytest sandbox the factory builds a NEW profile
    (lifecycle UNKNOWN -> landing surfaces), which strands the
    Database-Notes switch contract; mirrors test_library_shell.py's
    wrapper. Pass ``preserve_profile_admission=True`` for new-profile
    tests.
    """
    app = _build_tldw_test_app(*args, **kwargs)
    if not kwargs.pop("preserve_profile_admission", False):
        app.library_new_profile_admission = False
    return app


class _WorkspaceHarness(ConsolidatedCSSApp):
    """Mount one retained workspace without the rest of Library."""

    def __init__(self, workspace: LibraryFileNotesWorkspace) -> None:
        super().__init__()
        self.workspace = workspace

    def compose(self) -> ComposeResult:
        yield self.workspace


class _CssTrueWorkspaceHarness(_WorkspaceHarness):
    """Mount File Notes with the production bundle and shipped themes."""

    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )

    def on_mount(self) -> None:
        for theme in ALL_THEMES:
            self.register_theme(theme)


class _TwoWorkspaceHarness(ConsolidatedCSSApp):
    """Mount two workspaces that share one process owner."""

    def __init__(
        self,
        first: LibraryFileNotesWorkspace,
        second: LibraryFileNotesWorkspace,
    ) -> None:
        super().__init__()
        self.first = first
        self.second = second

    def compose(self) -> ComposeResult:
        with Vertical(id="first-workspace-host"):
            yield self.first
        with Vertical(id="second-workspace-host"):
            yield self.second


class _DynamicWorkspaceHarness(ConsolidatedCSSApp):
    """Mount a second workspace after the first is already running."""

    def __init__(self, workspace: LibraryFileNotesWorkspace) -> None:
        super().__init__()
        self.workspace = workspace

    def compose(self) -> ComposeResult:
        with Vertical(id="primary-workspace-host"):
            yield self.workspace
        yield Vertical(id="dynamic-workspace-host")


def test_workspace_transition_admission_is_exact_binding_and_idempotent(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    workspace = LibraryFileNotesWorkspace(
        root=tmp_path / "notes",
        replica=None,
        session_owner=owner,
    )
    workspace._session_binding = binding

    release = workspace.acquire_transition("source")
    assert callable(release)
    assert owner.try_acquire_mutation(binding) is None
    release()
    release()

    mutation = owner.try_acquire_mutation(binding)
    assert mutation is not None
    assert workspace.acquire_transition("screen") is False
    mutation.release()


def test_reconcile_tolerates_projection_disappearing_during_unmount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = LibraryFileNotesWorkspace(root=None, replica=None)
    workspace._active = True
    projection_attempts: list[bool | None] = []

    def missing_root_surface(*, offline: bool | None = None) -> None:
        projection_attempts.append(offline)
        raise NoMatches("root surface was removed during unmount")

    # Reproduce the teardown window itself: the initial lifecycle guard has
    # passed, but descendants disappear before the first projection query.
    monkeypatch.setattr(
        LibraryFileNotesWorkspace,
        "is_mounted",
        property(lambda _workspace: True),
    )
    monkeypatch.setattr(
        LibraryFileNotesWorkspace,
        "children",
        property(lambda _workspace: (object(),)),
    )
    monkeypatch.setattr(workspace, "_update_root_surface", missing_root_surface)

    applied = workspace._apply_reconcile(
        ReconcileResult(status="ok"),
        ("deleted.md",),
    )

    assert applied is False
    assert projection_attempts == [False]
    assert workspace.entries == {}
    assert workspace._deleted_paths == ("deleted.md",)


def test_scan_and_reconcile_clear_stale_replica_warning() -> None:
    """A clean service result must clear a warning from the prior result."""
    workspace = LibraryFileNotesWorkspace(root=None, replica=None)

    workspace._runtime_warning = "stale scan warning"
    workspace._adopt_scan_state(ScanResult(status="ok"), ())
    assert workspace._runtime_warning == ""

    workspace._runtime_warning = "stale reconcile warning"
    workspace._apply_reconcile(ReconcileResult(status="ok"), ())
    assert workspace._runtime_warning == ""


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


@asynccontextmanager
async def _production_workspace_context(
    workspace: LibraryFileNotesWorkspace,
    *,
    size: tuple[int, int],
):
    """Mount File Notes through the production TldwCli and Library screen."""
    app = _build_test_app(configured_default="library")

    def settings_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return default

    with patch(
        "tldw_chatbook.app.get_cli_setting",
        side_effect=settings_without_splash,
    ):
        async with app.run_test(size=size) as pilot:
            await _wait_until(
                pilot,
                lambda: isinstance(app.screen, LibraryScreen),
                "production app did not mount Library",
            )
            screen = app.screen
            assert isinstance(screen, LibraryScreen)
            screen._library_file_notes_workspace_factory = lambda: workspace
            await _wait_for_library_shell(screen, pilot)
            await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
            await _wait_until(
                pilot,
                lambda: bool(screen.query("#library-notes-source-files")),
                "Library Notes source selector did not mount",
            )
            screen.query_one("#library-notes-source-files", Button).press()
            await _wait_until(
                pilot,
                lambda: (
                    workspace.initialized
                    and workspace.is_mounted
                    and screen._library_file_notes_workspace is workspace
                ),
                "production Library did not mount File Notes",
            )
            yield pilot


def _static_text(workspace: LibraryFileNotesWorkspace, selector: str) -> str:
    widget = workspace.query_one(selector)
    renderable = widget.label if isinstance(widget, Button) else widget.renderable
    return getattr(renderable, "plain", str(renderable))


def _relative_luminance(color) -> float:
    """Return WCAG relative luminance for a Rich color."""
    triplet = color.get_truecolor()

    def channel(value: int) -> float:
        srgb = value / 255
        return srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel(triplet.red)
        + 0.7152 * channel(triplet.green)
        + 0.0722 * channel(triplet.blue)
    )


def _contrast_ratio(first, second) -> float:
    """Return WCAG contrast for two Rich colors."""
    lighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def _painted_style_of_text(app: App, region, needle: str):
    """Return the compositor style that actually paints ``needle``."""
    strips = list(app.screen._compositor.render_strips())
    for y in range(region.y, region.y + region.height):
        if y >= len(strips):
            break
        segments = list(strips[y]._segments)
        row_text = "".join(segment.text for segment in segments)
        index = row_text.find(needle)
        if index == -1:
            continue
        x = 0
        for segment in segments:
            if x + len(segment.text) > index:
                return segment.style
            x += len(segment.text)
    return None


def _painted_text_in_region(app: App, region) -> str:
    """Return only compositor cells painted inside ``region``."""
    strips = list(app.screen._compositor.render_strips())
    return "\n".join(
        strips[y].crop(region.x, region.right).text.rstrip()
        for y in range(region.y, region.bottom)
    )


def _assert_legible_painted_text(
    app: App,
    widget,
    needle: str,
    *,
    theme_name: str,
    minimum_ratio: float = 3.0,
) -> None:
    style = _painted_style_of_text(app, widget.region, needle)
    assert style is not None and style.color is not None
    assert style.bgcolor is not None
    ratio = _contrast_ratio(style.color, style.bgcolor)
    assert ratio >= minimum_ratio, (
        f"{theme_name}: {needle!r} paints at {ratio:.2f}:1, below {minimum_ratio}:1"
    )


def _show_disabled_git_result(
    workspace: LibraryFileNotesWorkspace,
) -> tuple[LibraryFileNotesGitPanel, Button]:
    """Project a visible unavailable Git action with an explicit recovery."""
    workspace._navigator_mode = "git"
    workspace._narrow_view = "navigator"
    workspace._apply_responsive_layout(workspace.size.width)
    panel = workspace.query_one(
        "#file-notes-git-panel",
        LibraryFileNotesGitPanel,
    )
    panel.render_push_result(
        PushPanelResultProjection(
            title="Remote state could not be confirmed",
            message="The reviewed commit may or may not have reached the remote.",
            action="check_remote_again",
            action_enabled=False,
            disabled_reason=(
                "Restore network access, then activate Check remote again."
            ),
        ),
        operation_id=1,
    )
    return panel, panel.query_one("#file-notes-git-push-check-remote", Button)


def _tree_labels(tree: Tree) -> list[str]:
    labels: list[str] = []

    def visit(node) -> None:
        label = getattr(node.label, "plain", str(node.label))
        labels.append(label)
        for child in node.children:
            visit(child)

    visit(tree.root)
    return labels


def _tree_node_count(tree: Tree) -> int:
    """Return the number of currently materialized nodes, including root."""
    return len(_tree_labels(tree))


@pytest.mark.asyncio
async def test_folder_files_builds_shared_adaptive_reader_roles(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "draft.md").write_text("draft", encoding="utf-8")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica_path=tmp_path / "replica.sqlite",
        poll_interval=10,
    )
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[{"id": "db-note", "title": "Database note", "content": "body"}],
    )
    screen = LibraryScreen(app, file_notes_workspace_factory=lambda: workspace)

    async with LibraryHarness(app, screen=screen).run_test(size=(160, 45)) as pilot:
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#library-notes-source-files")),
            "Notes source chooser did not mount",
        )
        screen.query_one("#library-notes-source-files", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.initialized and workspace.is_mounted,
            "Folder Files did not mount",
        )

        shells = workspace.query(LibraryAdaptiveReaderShell)
        assert len(shells) == 1
        shell = shells.first()
        assert shell.id == "library-file-notes-reader-shell"
        assert shell.library is workspace.query_one("#library-rail")
        assert shell.items is workspace.query_one("#file-notes-navigator")
        assert shell.work is workspace.query_one("#file-notes-work")
        assert shell.items.query_one("#file-notes-tree", Tree).is_mounted
        assert shell.work.query_one("#file-notes-editor", TextArea).is_mounted
        assert shell.items.query_one(
            "#file-notes-git-panel", LibraryFileNotesGitPanel
        ).is_mounted
        assert shell.work.query_one("#file-notes-resolution-actions").is_mounted
        assert shell.library_grip.region.width == 5
        assert shell.items_grip.region.width == 5

        identities = tuple(
            map(
                id,
                (
                    shell,
                    shell.library,
                    shell.items,
                    shell.work,
                    shell.items.query_one("#file-notes-tree"),
                    shell.work.query_one("#file-notes-editor"),
                    shell.items.query_one("#file-notes-git-panel"),
                    shell.work.query_one("#file-notes-resolution-actions"),
                ),
            )
        )
        shell.items_grip.press()
        await pilot.pause()
        shell.items_grip.press()
        await pilot.pause()
        assert (
            tuple(
                map(
                    id,
                    (
                        shell,
                        shell.library,
                        shell.items,
                        shell.work,
                        shell.items.query_one("#file-notes-tree"),
                        shell.work.query_one("#file-notes-editor"),
                        shell.items.query_one("#file-notes-git-panel"),
                        shell.work.query_one("#file-notes-resolution-actions"),
                    ),
                )
            )
            == identities
        )

    await workspace.shutdown()


@pytest.mark.asyncio
async def test_folder_files_shared_shell_retains_state_across_breakpoints(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "folder").mkdir()
    (root / "folder" / "draft.md").write_text(
        "first line\n" + "scroll line\n" * 80,
        encoding="utf-8",
    )
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica_path=tmp_path / "replica.sqlite",
        poll_interval=10,
        autosave_delay=30,
    )
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=[])
    screen = LibraryScreen(app, file_notes_workspace_factory=lambda: workspace)

    async with LibraryHarness(app, screen=screen).run_test(size=(160, 45)) as pilot:
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#library-notes-source-files")),
            "Notes source chooser did not mount",
        )
        screen.query_one("#library-notes-source-files", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.initialized and workspace.is_mounted,
            "Folder Files did not mount",
        )
        assert await workspace.open_path("folder/draft.md")

        shell = workspace.query_one(
            "#library-file-notes-reader-shell", LibraryAdaptiveReaderShell
        )
        tree = workspace.query_one("#file-notes-tree", Tree)
        folder = next(
            node
            for node in tree.root.children
            if getattr(node.data, "relative_path", None) == "folder"
        )
        folder.expand()
        tree.move_cursor(folder)
        search = workspace.query_one("#file-notes-search", Input)
        search.value = "scroll"
        editor = workspace.query_one("#file-notes-editor", TextArea)
        _replace_editor_text(editor, "retained draft\n" + "body\n" * 80)
        editor.selection = editor.selection.__class__((0, 1), (0, 7))
        editor.scroll_to(y=12, animate=False, force=True, immediate=True)
        workspace._set_save_state("conflict", "disk changed")
        workspace._conflict_resolution_active = True
        workspace._update_controls()
        workspace._push_phase = "needs_attention"
        workspace._render_session_git_label(3)
        await pilot.pause()

        git_panel = workspace.query_one(
            "#file-notes-git-panel", LibraryFileNotesGitPanel
        )
        recovery = workspace.query_one("#file-notes-resolution-actions")
        autosave_timer = workspace._autosave_timer
        identities = tuple(
            map(
                id,
                (
                    shell,
                    shell.library,
                    shell.items,
                    shell.work,
                    tree,
                    search,
                    editor,
                    editor.history,
                    git_panel,
                    recovery,
                ),
            )
        )
        editor_state = (
            editor.text,
            editor.cursor_location,
            editor.selection,
            int(editor.scroll_y),
        )

        shell.items_grip.press()
        await pilot.pause()
        shell.items_grip.press()
        await pilot.pause()
        for width in (160, 120, 119, 160, 100, 80, 79, 60):
            await pilot.resize_terminal(width, 32)
            await _wait_until(
                pilot,
                lambda: shell.region.width > 0 and shell.region.width <= width,
                f"Folder shell did not settle at {width} columns",
            )
            assert shell.library_grip.region.width == 5
            assert shell.items_grip.region.width == 5
            assert (
                tuple(
                    map(
                        id,
                        (
                            shell,
                            shell.library,
                            shell.items,
                            shell.work,
                            tree,
                            search,
                            editor,
                            editor.history,
                            git_panel,
                            recovery,
                        ),
                    )
                )
                == identities
            )
            assert (
                editor.text,
                editor.cursor_location,
                editor.selection,
                int(editor.scroll_y),
            ) == editor_state
            assert folder.is_expanded
            assert tree.cursor_node is folder
            assert search.value == "scroll"
            assert workspace.save_state == "conflict"
            assert workspace.conflict_resolution_active
            assert workspace._push_phase == "needs_attention"
            assert workspace._autosave_timer is autosave_timer

    await workspace.shutdown()


@pytest.mark.asyncio
async def test_notes_authority_round_trip_retains_both_workspaces(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "folder").mkdir()
    (root / "folder" / "file.md").write_text("folder body", encoding="utf-8")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica_path=tmp_path / "replica.sqlite",
        poll_interval=10,
        autosave_delay=30,
    )
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[
            {
                "id": "db-note",
                "title": "Database note",
                "content": "database body",
                "version": 1,
                "keywords": [],
            }
        ],
    )
    screen = LibraryScreen(app, file_notes_workspace_factory=lambda: workspace)

    async with LibraryHarness(app, screen=screen).run_test(size=(160, 45)) as pilot:
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#library-notes-row-0")),
            "Database Notes did not mount",
        )
        database_list = screen.query_one("#library-notes-canvas")
        database_list.scroll_to(y=3, animate=False, force=True, immediate=True)
        database_scroll = int(database_list.scroll_y)
        screen.query_one("#library-notes-row-0", Button).press()
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#library-note-body")),
            "Database note did not open",
        )
        database_id = screen._selected_note_id
        database_editor = screen.query_one("#library-note-body", TextArea)
        _replace_editor_text(database_editor, "database draft")
        await pilot.pause()
        database_draft = database_editor.text
        database_receipt = screen._library_notes_browse_return_receipt

        screen.query_one("#library-notes-source-files", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.initialized and workspace.is_mounted,
            "Folder Files did not mount",
        )
        assert await workspace.open_path("folder/file.md")
        folder_editor = workspace.query_one("#file-notes-editor", TextArea)
        _replace_editor_text(folder_editor, "folder draft")
        await pilot.pause()
        assert await workspace.flush_pending_work()
        folder_editor.selection = folder_editor.selection.__class__((0, 1), (0, 6))
        folder_search = workspace.query_one("#file-notes-search", Input)
        folder_search.value = "folder"
        folder_tree = workspace.query_one("#file-notes-tree", Tree)
        folder_node = next(
            node
            for node in folder_tree.root.children
            if getattr(node.data, "relative_path", None) == "folder"
        )
        folder_node.expand()
        workspace._push_phase = "needs_attention"
        workspace._render_session_git_label(2)
        await pilot.pause()
        folder_identities = tuple(
            map(
                id,
                (
                    workspace.query_one("#library-file-notes-reader-shell"),
                    folder_editor,
                    folder_editor.history,
                    folder_search,
                    folder_tree,
                    workspace._git_panel_widget,
                    workspace.query_one("#file-notes-resolution-actions"),
                ),
            )
        )
        folder_state = (
            folder_editor.text,
            folder_editor.cursor_location,
            folder_editor.selection,
            folder_search.value,
        )

        assert await workspace.flush_pending_work()
        await screen._return_to_library_database_notes()
        await _wait_until(
            pilot,
            lambda: screen._library_notes_source == "database",
            "Database Notes did not return",
        )
        assert screen._selected_note_id == database_id
        assert screen.query_one("#library-note-body", TextArea).text == database_draft
        assert screen._library_notes_browse_return_receipt is database_receipt
        assert int(screen.query_one("#library-notes-list").scroll_y) == database_scroll

        screen.query_one("#library-notes-source-files", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.is_mounted and screen._library_notes_source == "files",
            "Folder Files did not return",
        )
        await _wait_until(
            pilot,
            lambda: bool(workspace.query("#file-notes-editor")),
            "Folder Files roles did not remount",
        )
        assert (
            tuple(
                map(
                    id,
                    (
                        workspace.query_one("#library-file-notes-reader-shell"),
                        workspace.query_one("#file-notes-editor"),
                        workspace.query_one("#file-notes-editor").history,
                        workspace.query_one("#file-notes-search"),
                        workspace.query_one("#file-notes-tree"),
                        workspace._git_panel_widget,
                        workspace.query_one("#file-notes-resolution-actions"),
                    ),
                )
            )
            == folder_identities
        )
        returned_editor = workspace.query_one("#file-notes-editor", TextArea)
        assert (
            returned_editor.text,
            returned_editor.cursor_location,
            returned_editor.selection,
            workspace.query_one("#file-notes-search", Input).value,
        ) == folder_state
        assert folder_node.is_expanded
        assert workspace.save_state == "saved"
        assert workspace._push_phase == "needs_attention"

    await workspace.shutdown()


@pytest.mark.asyncio
async def test_file_notes_field_labels_persist_across_path_context(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "open.md").write_text("body", encoding="utf-8")
    workspace = LibraryFileNotesWorkspace(root=root, replica=None, poll_interval=10)

    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(
            pilot,
            lambda: workspace.initialized,
            "File Notes did not finish its initial scan",
        )
        search_label = workspace.query_one("#file-notes-search-label", Static)
        path_label = workspace.query_one("#file-notes-path-label", Static)
        search = workspace.query_one("#file-notes-search", Input)
        path = workspace.query_one("#file-notes-path", Input)

        assert str(search_label.renderable) == "Search"
        expected_label = "Target path · New / Move / Save copy"
        assert str(path_label.renderable) == expected_label
        workspace._navigator_mode = "git"
        workspace._sync_navigator_mode()
        assert not workspace.query_one("#file-notes-search-row").display
        workspace._navigator_mode = "files"
        workspace._sync_navigator_mode()
        assert workspace.query_one("#file-notes-search-row").display
        search.value = "body"
        path.value = "created.md"
        await pilot.pause()
        assert str(search_label.renderable) == "Search"
        assert str(path_label.renderable) == expected_label

        assert await workspace.open_path("open.md")
        assert str(path_label.renderable) == expected_label
        path.value = "moved.md"
        await pilot.pause()
        assert str(path_label.renderable) == expected_label

        for save_state in ("dirty", "conflict", "error"):
            workspace._set_save_state(save_state)
            assert str(path_label.renderable) == expected_label

        workspace._opened = replace(workspace._opened, protected=True)
        workspace._update_controls()
        assert str(path_label.renderable) == expected_label

        workspace._opened = replace(
            workspace._opened,
            protected=False,
            editable=False,
            is_excerpt=True,
            read_only_reason="large_file",
        )
        workspace._update_controls()
        assert str(path_label.renderable) == expected_label

        workspace._deleted_paths = ("deleted.md",)
        assert workspace.select_deleted("deleted.md")
        assert str(path_label.renderable) == expected_label
        assert path.value == "deleted.md"

    await workspace.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((120, 40), (40, 20)))
async def test_file_notes_field_labels_preserve_input_geometry_and_tab_access(
    tmp_path: Path,
    size: tuple[int, int],
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "open.md").write_text("body", encoding="utf-8")
    workspace = LibraryFileNotesWorkspace(root=root, replica=None, poll_interval=10)

    async with _WorkspaceHarness(workspace).run_test(size=size) as pilot:
        await _wait_until(
            pilot,
            lambda: workspace.initialized,
            "File Notes did not finish its initial scan",
        )
        search_label = workspace.query_one("#file-notes-search-label", Static)
        search = workspace.query_one("#file-notes-search", Input)
        for _ in range(80):
            if search.has_focus:
                break
            await pilot.press("tab")
        assert search.has_focus
        assert search_label.region.width > 0
        assert search.region.width >= 10
        assert search_label.region.right <= search.region.x
        assert search.region.right <= workspace.region.right

        assert await workspace.open_path("open.md")
        await pilot.pause()
        path_label = workspace.query_one("#file-notes-path-label", Static)
        path = workspace.query_one("#file-notes-path", Input)
        for _ in range(80):
            if path.has_focus:
                break
            await pilot.press("tab")
        assert path.has_focus
        assert path_label.region.width > 0
        assert path.region.width >= 10
        painted_label = path_label.render_line(0).text.strip()
        if size[0] >= 80:
            assert painted_label == "Target path · New / Move / Save copy"
        else:
            assert painted_label.startswith("Target path · New / Move /")
        assert path_label.region.bottom <= path.region.y
        assert path.region.right <= workspace.region.right

    await workspace.shutdown()


def _replace_editor_text(editor: TextArea, text: str) -> None:
    editor.select_all()
    editor.replace(text, editor.selection.start, editor.selection.end)


def _visible_editor_action_ids(
    workspace: LibraryFileNotesWorkspace,
) -> set[str]:
    """Return the currently disclosed editor action ids."""
    return {
        button.id
        for toolbar in workspace.query(".file-notes-toolbar")
        for button in toolbar.query(Button)
        if button.display and button.id is not None
    }


def _visible_primary_action_labels(
    workspace: LibraryFileNotesWorkspace,
) -> tuple[str, ...]:
    """Return visible primary action labels in keyboard/DOM order."""
    return tuple(
        str(button.label)
        for button in workspace.query_one("#file-notes-file-actions").query(Button)
        if button.display
    )


def _assert_visible_editor_actions_fit(
    workspace: LibraryFileNotesWorkspace,
) -> None:
    """Assert disclosed actions keep complete labels inside the editor pane."""
    pane = workspace.query_one("#file-notes-editor-pane")
    visible = tuple(
        (toolbar, button)
        for toolbar in pane.query(".file-notes-toolbar")
        if toolbar.display
        for button in toolbar.query(Button)
        if button.display
    )
    assert visible
    for toolbar, button in visible:
        label = str(button.label)
        assert button.render_line(0).text.strip() == label, (
            button.id,
            label,
            button.region,
            button.content_region,
            pane.region,
            workspace.classes,
        )
        assert button.render().plain == label
        assert cell_len(label) <= button.content_region.width
        assert toolbar.content_region.contains_region(button.region), (
            toolbar.id,
            toolbar.content_region,
            button.id,
            button.region,
        )
        assert pane.content_region.contains_region(button.region)


async def _show_maintenance_actions(
    workspace: LibraryFileNotesWorkspace,
    pilot,
) -> None:
    """Open the retained secondary-action disclosure through its real control."""
    toolbar = workspace.query_one("#file-notes-maintenance-actions")
    if toolbar.display:
        return
    workspace.query_one("#file-notes-maintenance-toggle", Button).press()
    await _wait_until(
        pilot,
        lambda: toolbar.display,
        "Maintenance actions did not open",
    )


def _delayed_call(call):
    started = threading.Event()
    release = threading.Event()

    def delayed(*args, **kwargs):
        started.set()
        release.wait(5)
        return call(*args, **kwargs)

    return delayed, started, release


def _event_loop_heartbeat(
    event_loop: asyncio.AbstractEventLoop,
    blocked: threading.Event,
    *release_on_failure: threading.Event,
) -> tuple[threading.Thread, threading.Event, list[bool]]:
    checked = threading.Event()
    observations: list[bool] = []

    def check() -> None:
        heartbeat_ran = threading.Event()
        if blocked.wait(timeout=5):
            event_loop.call_soon_threadsafe(heartbeat_ran.set)
            observations.append(heartbeat_ran.wait(timeout=1))
        else:
            observations.append(False)
        checked.set()
        if not observations[-1]:
            for release in release_on_failure:
                release.set()

    return threading.Thread(target=check, daemon=True), checked, observations


def _root_transition_workspace(tmp_path: Path):
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    old_root.mkdir()
    new_root.mkdir()
    owner = FileNotesSessionOwner()
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=old_root,
        replica=replica,
        session_owner=owner,
        poll_interval=10,
    )
    return old_root, new_root, owner, replica, workspace


@pytest.mark.asyncio
async def test_folder_files_authority_row_tracks_root_save_and_session_git(
    tmp_path: Path,
) -> None:
    """One pinned row projects facts already owned by each update choke point."""
    root = tmp_path / "notes"
    root.mkdir()
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=None, replica=replica)
    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await pilot.pause()
        authority = workspace.query_one("#file-notes-authority", Static)
        assert workspace.children[0] is authority
        assert authority._render_markup is False
        assert _static_text(workspace, "#file-notes-authority") == (
            "Folder files · No folder selected · Next: Choose folder."
        )

        workspace._root = root
        workspace._root_offline = False
        workspace._update_root_surface()
        assert "Folder: notes" in _static_text(
            workspace,
            "#file-notes-authority",
        )

        workspace._set_save_state("error", "permission denied")
        save_failure = _static_text(workspace, "#file-notes-authority")
        assert "Save failed" in save_failure
        assert "permission denied" not in save_failure
        assert "permission denied" in _static_text(
            workspace,
            "#file-notes-save-status",
        )
        assert "Next: Retry/copy." in save_failure

        workspace._set_save_state("saved")
        workspace._push_phase = "needs_attention"
        workspace._render_session_git_label(2)
        git_attention = _static_text(workspace, "#file-notes-authority")
        assert "Saved" in git_attention
        assert "Session Git: 2 · Push attention" in git_attention
        assert "Next: Review changes." in git_attention
    replica.close()


@pytest.mark.asyncio
async def test_folder_files_authority_status_survives_in_surface_navigation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=root, replica=replica, poll_interval=10)

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        workspace._push_phase = "needs_attention"
        workspace._render_session_git_label(3)
        expected = _static_text(workspace, "#file-notes-authority")

        workspace._navigator_mode = "git"
        workspace._sync_navigator_mode()
        workspace._navigator_mode = "files"
        workspace._sync_navigator_mode()
        await pilot.pause()

        assert _static_text(workspace, "#file-notes-authority") == expected
        assert "Push attention" in expected
        assert "Next: Review changes." in expected

    await workspace.shutdown()
    replica.close()


@pytest.mark.parametrize(
    ("save_state", "save_detail"),
    (("error", "permission denied"), ("conflict", "disk changed")),
)
@pytest.mark.parametrize("git_first", (True, False))
@pytest.mark.asyncio
async def test_folder_files_authority_merges_save_and_git_in_either_update_order(
    tmp_path: Path,
    save_state: str,
    save_detail: str,
    git_first: bool,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    owner = FileNotesSessionOwner()
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        session_owner=owner,
        poll_interval=10,
    )

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        binding = workspace._session_binding
        assert binding is not None
        assert owner.record_change(binding, SessionChange("modified", "draft.md"))
        if git_first:
            workspace._render_session_git_label()
            workspace._set_save_state(save_state, save_detail)
        else:
            workspace._set_save_state(save_state, save_detail)
            workspace._render_session_git_label()

        expected = _static_text(workspace, "#file-notes-authority")
        assert "Folder: notes" in expected
        assert ("Conflict" if save_state == "conflict" else "Save failed") in expected
        assert save_detail not in expected
        assert save_detail in _static_text(workspace, "#file-notes-save-status")
        assert "Session Git: 1 change" in expected
        assert "Next:" in expected

        workspace._navigator_mode = "git"
        workspace._sync_navigator_mode()
        workspace._navigator_mode = "files"
        workspace._sync_navigator_mode()
        await pilot.pause()
        assert _static_text(workspace, "#file-notes-authority") == expected

    await workspace.shutdown()
    replica.close()


@pytest.mark.parametrize("size", ((160, 45), (120, 40), (60, 20)))
@pytest.mark.asyncio
async def test_file_notes_authority_copy_is_complete_and_bounded(
    tmp_path: Path,
    size: tuple[int, int],
) -> None:
    """Verify authority copy uses bounded rows at supported terminal sizes.

    Args:
        tmp_path: Temporary directory used as the linked File Notes root.
        size: Terminal dimensions used to mount the workspace.
    """
    root = tmp_path / "notes"
    root.mkdir()
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        poll_interval=10,
    )
    expected = (
        "Folder files · Folder: notes · Ready "
        "Session Git: 0 changes · Next: Choose/new file."
    )

    async with _WorkspaceHarness(workspace).run_test(size=size) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        purpose = workspace.query_one("#file-notes-authority")
        rendered = " ".join(
            purpose.render_line(row).text.strip()
            for row in range(purpose.region.height)
        )
        assert " ".join(rendered.split()) == expected
        assert purpose.region.height == 2
        assert workspace.query_one("#file-notes-body").region.height >= 8

    replica.close()


def test_configured_root_authority_state_table_is_two_line_and_bounded(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Research notes with a very long private directory name"
    root.mkdir()
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=root, replica=replica)
    root_states = (
        (False, "", ""),
        (None, "", "Checking"),
        (True, "", "Offline"),
        (True, "Replica warning", "Offline+Warning"),
        (False, "Replica warning", "Warning"),
    )
    transitions = (
        (False, False, ""),
        (True, False, "Changing folder"),
        (False, True, "File operation"),
    )
    save_states = (
        ("idle", ""),
        ("dirty", "Unsaved"),
        ("saving", "Saving"),
        ("saved", "Saved"),
        ("conflict", "Conflict"),
        ("error", "Save failed"),
    )
    push_states = (
        ("idle", ""),
        ("checking", "Check push"),
        ("pushing", "Pushing"),
        ("needs_attention", "Push attention"),
    )

    for root_state, transition, save_state, push_state, git_count in product(
        root_states,
        transitions,
        save_states,
        push_states,
        (0, 1, 125),
    ):
        offline, warning, root_copy = root_state
        root_transitioning, path_transitioning, transition_copy = transition
        save_value, save_copy = save_state
        push_value, push_copy = push_state
        workspace._root_offline = offline
        workspace._runtime_warning = warning
        workspace._root_transitioning = root_transitioning
        workspace._path_transitioning = path_transitioning
        workspace._save_state = save_value
        workspace._push_phase = push_value
        authority = workspace._authority_copy(git_count)
        lines = authority.splitlines()
        context = (root_state, transition, save_state, push_state, git_count)

        assert len(lines) == 2, context
        assert all(cell_len(line) <= 60 for line in lines), (context, lines)
        assert "Folder files" in lines[0], context
        assert "Folder:" in lines[0], context
        state_copy = transition_copy or root_copy
        if state_copy:
            assert state_copy in lines[0], context
        if save_copy:
            assert save_copy in lines[0], context
        elif not state_copy:
            assert "Ready" in lines[0], context
        expected_count = "99+" if git_count > 99 else str(git_count)
        assert f"Session Git: {expected_count}" in lines[1], context
        if push_copy:
            assert push_copy in lines[1], context
        assert "Next:" in lines[1], context

        if root_transitioning:
            next_copy = "Wait for change."
        elif path_transitioning:
            next_copy = "Wait for file."
        elif offline is None:
            next_copy = "Wait for check."
        elif offline is True:
            next_copy = "Reconnect/change."
        elif warning:
            next_copy = "Open Details."
        elif save_value == "conflict":
            next_copy = "Resolve/copy."
        elif save_value == "error":
            next_copy = "Retry/copy."
        elif save_value == "saving":
            next_copy = "Wait for save."
        elif save_value == "dirty":
            next_copy = "Keep editing."
        elif push_value != "idle" or git_count:
            next_copy = "Review changes."
        elif save_value == "saved":
            next_copy = "Keep editing."
        else:
            next_copy = "Choose/new file."
        assert f"Next: {next_copy}" in lines[1], context

    replica.close()


@pytest.mark.parametrize("size", ((160, 45), (120, 40), (60, 20)))
@pytest.mark.asyncio
async def test_file_notes_linked_root_leads_with_friendly_folder_identity(
    tmp_path: Path,
    size: tuple[int, int],
) -> None:
    """Verify the persistent root row favors a friendly folder identity.

    Args:
        tmp_path: Temporary directory used to construct the linked root.
        size: Terminal dimensions used to mount the workspace.
    """
    root = tmp_path / "deep" / "nested" / "Research Notes"
    root.mkdir(parents=True)
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        poll_interval=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=size) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        await pilot.pause()
        summary = _static_text(workspace, "#file-notes-root-status")
        assert summary == "Linked · Local folder: Research Notes"
        assert str(root.resolve()) not in summary
        assert str(root.resolve()) in workspace._root_status_detail
        assert workspace.query_one("#file-notes-root-details", Button).display

    replica.close()


@pytest.mark.asyncio
async def test_file_notes_root_details_preserve_exact_path_and_warning(
    tmp_path: Path,
) -> None:
    """Verify root details retain exact telemetry and recovery warnings.

    Args:
        tmp_path: Temporary directory used to construct the linked root.
    """
    root = tmp_path / "deep" / "nested" / "Research Notes"
    root.mkdir(parents=True)
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        poll_interval=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        workspace._apply_reconcile(
            ReconcileResult(
                status="ok",
                replica_warning="Recovery unavailable: replica locked",
            ),
            (),
        )
        await pilot.pause()

        assert (
            _static_text(workspace, "#file-notes-root-status")
            == "Warning · Local folder: Research Notes"
        )
        root_status = workspace.query_one("#file-notes-root-status")
        assert root_status.has_class("-warning")
        assert "#file-notes-root-status.-warning," in workspace.DEFAULT_CSS
        assert "#file-notes-root-status.-offline," in workspace.DEFAULT_CSS
        assert "background: $warning 14%;" in workspace.DEFAULT_CSS
        tooltip = workspace.query_one("#file-notes-root-status").tooltip
        assert tooltip is not None
        assert str(root.resolve()) in str(tooltip)
        assert "Recovery unavailable: replica locked" in str(tooltip)
        details = workspace.query_one("#file-notes-root-details", Button)
        details.press()
        await pilot.pause()
        exact = workspace.app.screen.query_one(
            "#file-notes-root-details-text",
            TextArea,
        ).text
        assert str(root.resolve()) in exact
        assert "Recovery unavailable: replica locked" in exact

        workspace._apply_reconcile(ReconcileResult(status="ok"), ())
        await pilot.pause()
        assert workspace._runtime_warning == ""
        assert not root_status.has_class("-warning")

    replica.close()


@pytest.mark.asyncio
async def test_file_notes_navigation_and_key_guidance_use_one_phrase() -> None:
    """Keep return navigation and key guidance consistent across File Notes."""
    workspace = LibraryFileNotesWorkspace(root=None)

    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        editor_back = workspace.query_one("#file-notes-back", Button)
        git_back = workspace.query_one("#file-notes-git-back", Button)
        guide = _static_text(workspace, "#file-notes-git-guide")

        assert str(editor_back.label) == "Back to navigator"
        assert str(git_back.label) == "Back to navigator"
        assert guide == "Up/Down select · Tab actions · Enter run · Esc back"
        assert "|" not in guide

    await workspace.shutdown()


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

        def save_root_mutation(
            section_values: dict[str, dict[str, str]],
        ) -> ConfigMutationResult:
            saved.append(
                (
                    "file_notes",
                    "root",
                    section_values["file_notes"]["root"],
                )
            )
            return ConfigMutationResult(True, True, None)

        monkeypatch.setattr(
            workspace_module,
            "apply_settings_mutation_to_cli_config",
            save_root_mutation,
            raising=False,
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
async def test_empty_root_prompt_and_choose_button_render_adjacent() -> None:
    """task-2850 AC2: the "no root chosen" empty state is a prompt +
    adjacent action, not a status toolbar with the button pinned ~150
    columns away. Mounted at a wide (170-col) size -- the width the UAT
    finding reproduced at -- so a regression to the old ``width: 1fr``
    status (which pushes the button to the far right of whatever it is
    mounted in) is caught even though this harness mounts the workspace
    alone, narrower than the reported full-screen gap.
    """
    replica = FileNotesReplica(":memory:")
    empty = LibraryFileNotesWorkspace(root=None, replica=replica)
    async with _WorkspaceHarness(empty).run_test(size=(170, 50)) as pilot:
        await pilot.pause()
        status = empty.query_one("#file-notes-root-status")
        choose = empty.query_one("#file-notes-choose-root", Button)
        assert status.has_class("-empty-root")
        gap = choose.region.x - status.region.right
        assert gap <= 2, (
            f"'{_static_text(empty, '#file-notes-root-status')}' and "
            f"'Choose folder…' are {gap} columns apart -- not adjacent"
        )
    replica.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 50), (100, 30), (40, 20)])
async def test_files_mode_uses_focused_canvas_and_keeps_shell_mounted(
    size: tuple[int, int],
) -> None:
    """Files keeps the shell mounted but owns the focused workbench width."""
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=None, replica=replica)
    async with _production_workspace_context(workspace, size=size) as pilot:
        screen = pilot.app.screen
        rail = screen.query_one("#library-rail")
        shell_grid = screen.query_one("#library-shell-grid")
        shell = workspace.query_one("#library-file-notes-reader-shell")
        assert rail.display is False, (
            f"size={screen.size!r}, grid={screen.query_one('#library-shell-grid').region!r}, "
            f"compact={screen._library_notes_compact!r}, "
            f"stage={screen._library_notes_stage!r}"
        )
        if size[0] >= 120:
            task_return = screen.query_one("#library-notes-task-return", Button)
            assert task_return.display
            assert str(task_return.label) == "‹ Library / Notes"
        else:
            task_returns = screen.query("#library-notes-task-return")
            assert not task_returns or task_returns.first().display is False
        # Folder Files is now a retained sibling authority whose own shared
        # shell supplies Library / Items / Work roles.
        assert workspace.parent is shell_grid
        assert shell.work is workspace._reader_work_widget
        assert workspace.region.x >= 0
        assert workspace.region.right <= screen.size.width
        assert workspace.region.bottom <= screen.size.height
    replica.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 50), (40, 20)])
async def test_escape_in_files_mode_returns_to_database_notes(
    size: tuple[int, int],
) -> None:
    """task-2850 AC3: Escape is a real, working way out of Files mode --
    not just the small "Database" strip link, which was the only exit
    before this fix (Escape was previously dead on this surface).
    """
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=None, replica=replica)
    async with _production_workspace_context(workspace, size=size) as pilot:
        screen = pilot.app.screen
        assert screen._library_notes_source == "files"

        await pilot.press("escape")
        await _wait_until(
            pilot,
            lambda: screen._library_notes_source == "database",
            "Escape did not return Files mode to Database Notes",
        )
        assert screen.query_one("#library-notes-source-database", Button)
        assert screen.query_one("#library-file-notes-workspace") is workspace
        assert workspace.display is False
    replica.close()


@pytest.mark.asyncio
async def test_wide_files_task_return_reuses_the_existing_leave_guard() -> None:
    """The wide cue cannot bypass the Files flush/conflict admission seam."""
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=None, replica=replica)
    async with _production_workspace_context(workspace, size=(170, 48)) as pilot:
        screen = pilot.app.screen
        blocked_flush = AsyncMock(return_value=False)
        screen._flush_active_file_notes = blocked_flush

        screen.query_one("#library-notes-task-return", Button).press()
        await pilot.pause()

        blocked_flush.assert_awaited_once_with()
        assert screen._library_notes_source == "files"
        assert screen.query_one("#library-file-notes-workspace") is workspace

        blocked_flush.reset_mock()
        blocked_flush.return_value = True
        screen.query_one("#library-notes-task-return", Button).press()
        await _wait_until(
            pilot,
            lambda: screen._library_notes_source == "database",
            "Wide task return did not reopen Database Notes after admission.",
        )
        for _ in range(10):
            await pilot.pause()

        blocked_flush.assert_awaited_once_with()
        assert screen.query_one("#library-notes-source-database", Button)
        assert screen.query_one("#library-rail").display is True


@pytest.mark.asyncio
async def test_initial_root_scan_projects_checking_authority_while_actions_are_gated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    scan_started = threading.Event()
    release_scan = threading.Event()
    original_scan = FileNotesService.scan

    def delayed_scan(service: FileNotesService):
        scan_started.set()
        assert release_scan.wait(timeout=5)
        return original_scan(service)

    monkeypatch.setattr(FileNotesService, "scan", delayed_scan)
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=root, replica=replica, poll_interval=10)

    async with _WorkspaceHarness(workspace).run_test(size=(60, 20)) as pilot:
        try:
            await _wait_until(
                pilot,
                scan_started.is_set,
                "initial root scan did not start",
            )
            assert "Checking" in _static_text(
                workspace,
                "#file-notes-root-status",
            )
            authority = _static_text(workspace, "#file-notes-authority")
            assert "Checking" in authority
            assert "Ready" not in authority
            assert "Next: Wait for check." in authority
            assert workspace.query_one("#file-notes-new", Button).disabled
        finally:
            release_scan.set()

        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")

    replica.close()


@pytest.mark.asyncio
async def test_wide_files_task_return_restores_database_browse_receipt() -> None:
    """Files returns to the prior Database row and both independent scroll owners."""
    notes = [
        {
            "id": f"note-{index:02d}",
            "title": f"Browse note {index:02d}",
            "content": f"body {index}",
            "last_modified": f"2026-07-{(index % 28) + 1:02d}T12:00:00+00:00",
            "version": 1,
            "keywords": [],
        }
        for index in range(32)
    ]
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=None, replica=replica)
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=notes)
    screen = LibraryScreen(app, file_notes_workspace_factory=lambda: workspace)

    async with LibraryHarness(app, screen=screen).run_test(size=(170, 24)) as pilot:
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        await _wait_until(
            pilot,
            lambda: len(screen.query(".library-notes-row")) >= 20,
            "Database Notes did not render the browse rows.",
        )
        screen.query_one("#library-notes-sort", Button).press()
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#library-notes-sort-title")),
            "Notes sort choices did not open.",
        )
        screen.query_one("#library-notes-sort-title", Button).press()
        await pilot.pause()

        row = list(screen.query(".library-notes-row"))[18]
        note_id = str(row.note_id)
        notes_list = screen.query_one("#library-notes-list")
        rail = screen.query_one("#library-rail")
        notes_list.scroll_to(y=7, animate=False, force=True, immediate=True)
        rail.scroll_to(y=2, animate=False, force=True, immediate=True)
        screen._mark_library_notes_user_interaction()
        row.focus(scroll_visible=False)
        await pilot.pause()
        before_list_scroll = int(notes_list.scroll_y)
        before_rail_scroll = int(rail.scroll_y)
        row.press()
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#library-note-body")),
            "Database note editor did not open.",
        )
        browse_receipt = screen._library_notes_browse_return_receipt
        assert browse_receipt is not None
        await pilot.resize_terminal(100, 30)
        await _wait_until(
            pilot,
            lambda: screen._library_notes_compact,
            "Notes did not enter the compact presentation.",
        )

        screen.query_one("#library-notes-source-files", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.is_mounted,
            "Files workspace did not mount.",
        )
        assert screen._library_notes_browse_return_receipt is browse_receipt
        await pilot.resize_terminal(170, 24)
        await _wait_until(
            pilot,
            lambda: not screen._library_notes_compact,
            "Notes did not return to the wide presentation.",
        )
        screen.query_one("#library-notes-task-return", Button).press()
        await _wait_until(
            pilot,
            lambda: (
                screen._library_notes_source == "database"
                and getattr(screen.focused, "note_id", None) == note_id
            ),
            "Files return did not restore the prior Database note row.",
        )
        await pilot.pause()

        assert screen._library_notes_sort == "title"
        assert (
            int(screen.query_one("#library-notes-list").scroll_y) == before_list_scroll
        )
        assert int(screen.query_one("#library-rail").scroll_y) == before_rail_scroll
        assert screen.query_one("#library-rail").display is True

    await workspace.shutdown()


@pytest.mark.parametrize(
    ("save_state", "save_copy"),
    (("error", "Save failed"), ("conflict", "Conflict")),
)
@pytest.mark.parametrize(
    ("push_phase", "push_copy", "git_count"),
    (("idle", "", 0), ("needs_attention", "Push attention", 1)),
)
@pytest.mark.asyncio
async def test_path_transition_authority_names_file_operation_and_settles(
    tmp_path: Path,
    save_state: str,
    save_copy: str,
    push_phase: str,
    push_copy: str,
    git_count: int,
) -> None:
    root = tmp_path / "Research notes with a very long private directory name"
    root.mkdir()
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=root, replica=replica, poll_interval=10)

    async with _CssTrueWorkspaceHarness(workspace).run_test(size=(60, 20)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        workspace._set_save_state(save_state, "a long private filesystem detail")
        workspace._push_phase = push_phase

        with workspace._hold_path_transition() as transition:
            assert transition is not None
            workspace._render_session_git_label(git_count)
            await pilot.pause()
            authority = workspace.query_one("#file-notes-authority", Static)
            authority_copy = _static_text(workspace, "#file-notes-authority")
            painted = _painted_text_in_region(pilot.app, authority.region)
            assert authority.region.height == 2
            assert len(authority_copy.splitlines()) == 2
            assert all(
                cell_len(row) <= authority.region.width
                for row in authority_copy.splitlines()
            )
            assert "Folder: Rese…" in painted
            assert "File operation" in painted
            assert save_copy in painted
            assert "Changing folder" not in painted
            assert f"Session Git: {git_count}" in painted
            if push_copy:
                assert push_copy in painted
            assert "Next: Wait for file." in painted

        await pilot.pause()
        authority = _static_text(workspace, "#file-notes-authority")
        assert "File operation" not in authority
        assert save_copy in authority

    replica.close()


@pytest.mark.parametrize(
    ("push_phase", "push_copy"),
    (("idle", ""), ("needs_attention", "Push attention")),
)
@pytest.mark.asyncio
async def test_saved_authority_with_session_git_paints_at_60x20(
    tmp_path: Path,
    push_phase: str,
    push_copy: str,
) -> None:
    root = tmp_path / "Research notes with a very long private directory name"
    root.mkdir()
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=root, replica=replica, poll_interval=10)

    async with _CssTrueWorkspaceHarness(workspace).run_test(size=(60, 20)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        workspace._set_save_state("saved")
        workspace._push_phase = push_phase
        workspace._render_session_git_label(1)
        await pilot.pause()

        authority = workspace.query_one("#file-notes-authority", Static)
        authority_copy = _static_text(workspace, "#file-notes-authority")
        painted = _painted_text_in_region(pilot.app, authority.region)
        assert authority.region.height == 2
        assert len(authority_copy.splitlines()) == 2
        assert all(
            cell_len(row) <= authority.region.width
            for row in authority_copy.splitlines()
        )
        assert "Folder files" in painted
        assert "Folder: Rese…" in painted
        assert "Saved" in painted
        assert "Session Git: 1" in painted
        if push_copy:
            assert push_copy in painted
        assert "Next: Review changes." in painted

    replica.close()


@pytest.mark.asyncio
async def test_root_transition_retains_and_freezes_old_document_until_scan_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_root = tmp_path / "old"
    old_root.mkdir()
    (old_root / "old.md").write_text("old body", encoding="utf-8")
    new_root = tmp_path / "new"
    new_root.mkdir()
    (new_root / "new.md").write_text("new body", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=old_root,
        replica=replica,
        autosave_delay=10,
        poll_interval=10,
    )
    original_scan = FileNotesService.scan
    scan_started = threading.Event()
    release_scan = threading.Event()

    def delayed_scan(service):
        if service.root == new_root.resolve():
            scan_started.set()
            release_scan.wait(5)
        return original_scan(service)

    monkeypatch.setattr(FileNotesService, "scan", delayed_scan)
    async with _WorkspaceHarness(workspace).run_test(size=(110, 36)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("old.md")
        editor = workspace.query_one("#file-notes-editor", TextArea)
        _replace_editor_text(editor, "saved before root change")
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "dirty",
            "root-change draft did not become dirty",
        )
        transition = asyncio.create_task(workspace.set_root(new_root, persist=False))
        await _wait_until(
            pilot,
            scan_started.is_set,
            "candidate root scan did not start",
        )
        assert workspace.root == old_root.resolve()
        assert workspace.current_path == "old.md"
        assert editor.text == "saved before root change"
        assert editor.read_only
        assert workspace.query_one("#file-notes-new", Button).disabled
        assert workspace.query_one("#file-notes-search", Input).disabled
        authority = _static_text(workspace, "#file-notes-authority")
        assert "Changing folder" in authority
        assert "Next: Wait for change." in authority
        assert (old_root / "old.md").read_text(encoding="utf-8") == (
            "saved before root change"
        )

        release_scan.set()
        assert await transition
        assert workspace.root == new_root.resolve()
        assert workspace.current_path == ""
        assert editor.text == ""
        assert "new.md" in workspace.entries
    release_scan.set()
    replica.close()


@pytest.mark.asyncio
async def test_root_transition_rebinds_after_owned_replica_reopens(
    tmp_path: Path,
) -> None:
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    old_root.mkdir()
    new_root.mkdir()
    workspace = LibraryFileNotesWorkspace(
        root=old_root,
        replica_path=tmp_path / "owned.sqlite",
        poll_interval=10,
    )

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _wait_until(
            pilot,
            lambda: workspace.initialized,
            "initial scan did not finish",
        )
        old_service = workspace._service
        assert old_service is not None
        old_service.close()
        workspace._replica = None
        workspace._service = None

        assert await workspace.set_root(new_root, persist=False)
        service = workspace._service
        assert service is not None
        assert service.create_file("new.md", "new").status == "ok"
        workspace._refresh_session_changes()
        assert (
            _static_text(workspace, "#file-notes-session-changes")
            == "Review session changes (1)"
        )

    await workspace.shutdown()


@pytest.mark.asyncio
async def test_delayed_old_workspace_cannot_replace_current_workspace_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_root = (tmp_path / "old").resolve()
    new_root = (tmp_path / "new").resolve()
    old_root.mkdir()
    new_root.mkdir()
    owner = FileNotesSessionOwner()
    old_replica = FileNotesReplica(":memory:")
    new_replica = FileNotesReplica(":memory:")
    old_canonical_started = threading.Event()
    release_old_canonical = threading.Event()
    real_canonical_root = LibraryFileNotesWorkspace._canonical_root

    def delayed_canonical_root(value: object) -> Path | None:
        canonical = real_canonical_root(value)
        if canonical == old_root and not old_canonical_started.is_set():
            old_canonical_started.set()
            assert release_old_canonical.wait(timeout=5)
        return canonical

    monkeypatch.setattr(
        LibraryFileNotesWorkspace,
        "_canonical_root",
        staticmethod(delayed_canonical_root),
    )
    old_workspace = LibraryFileNotesWorkspace(
        root=old_root,
        replica=old_replica,
        session_owner=owner,
        poll_interval=10,
    )
    new_workspace = LibraryFileNotesWorkspace(
        root=new_root,
        replica=new_replica,
        session_owner=owner,
        poll_interval=10,
    )
    old_workspace._active = True
    old_initialization = asyncio.create_task(old_workspace._initialize())

    try:
        assert await asyncio.to_thread(old_canonical_started.wait, 1)
        async with _WorkspaceHarness(new_workspace).run_test() as pilot:
            await _wait_until(
                pilot,
                lambda: new_workspace.initialized,
                "current workspace did not initialize",
            )
            current_service = new_workspace._service
            current_binding = new_workspace._session_binding
            assert current_service is not None
            assert current_binding is not None
            assert current_service.create_file("before.md", "before").status == "ok"

            release_old_canonical.set()
            await old_initialization

            assert current_service.create_file("after.md", "after").status == "ok"
            assert [
                item.change.relative_path
                for item in owner.snapshot(current_binding).changes
            ] == ["before.md", "after.md"]
            assert current_service.session_changes == tuple(
                item.change for item in owner.snapshot(current_binding).changes
            )
    finally:
        release_old_canonical.set()
        if not old_initialization.done():
            await old_initialization
        old_workspace._active = False
        await old_workspace.shutdown()
        await new_workspace.shutdown()
        owner.shutdown()
        old_replica.close()
        new_replica.close()


@pytest.mark.asyncio
async def test_overlapping_root_persistence_only_winner_updates_config_and_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_root = (tmp_path / "old").resolve()
    slow_root = (tmp_path / "slow").resolve()
    winner_root = (tmp_path / "winner").resolve()
    old_root.mkdir()
    slow_root.mkdir()
    winner_root.mkdir()
    owner = FileNotesSessionOwner()
    slow_replica = FileNotesReplica(":memory:")
    winner_replica = FileNotesReplica(":memory:")
    slow_workspace = LibraryFileNotesWorkspace(
        root=old_root,
        replica=slow_replica,
        session_owner=owner,
        poll_interval=10,
    )
    winner_workspace = LibraryFileNotesWorkspace(
        root=old_root,
        replica=winner_replica,
        session_owner=owner,
        poll_interval=10,
    )
    slow_scan_started = threading.Event()
    release_slow_scan = threading.Event()
    real_scan = FileNotesService.scan
    persisted_roots: list[str] = []

    def delayed_scan(service: FileNotesService):
        if service.root == slow_root:
            slow_scan_started.set()
            assert release_slow_scan.wait(timeout=5)
        return real_scan(service)

    def persist_mutation(
        section_values: dict[str, dict[str, str]],
    ) -> ConfigMutationResult:
        persisted_roots.append(section_values["file_notes"]["root"])
        return ConfigMutationResult(True, True, None)

    monkeypatch.setattr(FileNotesService, "scan", delayed_scan)
    monkeypatch.setattr(
        workspace_module,
        "apply_settings_mutation_to_cli_config",
        persist_mutation,
        raising=False,
    )
    slow_transition: asyncio.Task[bool] | None = None
    try:
        async with _TwoWorkspaceHarness(
            slow_workspace,
            winner_workspace,
        ).run_test() as pilot:
            await _wait_until(
                pilot,
                lambda: slow_workspace.initialized and winner_workspace.initialized,
                "shared-root workspaces did not initialize",
            )
            old_service = slow_workspace._service
            old_binding = slow_workspace._session_binding
            assert old_service is not None
            assert old_binding is not None
            assert winner_workspace._session_binding == old_binding

            slow_transition = asyncio.create_task(slow_workspace.set_root(slow_root))
            await _wait_until(
                pilot,
                slow_scan_started.is_set,
                "slow candidate scan did not start",
            )
            assert await winner_workspace.set_root(winner_root)
            winner_binding = winner_workspace._session_binding
            assert winner_binding is not None

            release_slow_scan.set()
            assert not await slow_transition

            assert persisted_roots == [str(winner_root)]
            assert owner.current_binding() == winner_binding
            assert winner_workspace.root == winner_root
            assert slow_workspace.root == old_root
            assert slow_workspace._service is old_service
            assert slow_workspace._session_binding == old_binding
    finally:
        release_slow_scan.set()
        if slow_transition is not None and not slow_transition.done():
            await slow_transition
        await slow_workspace.shutdown()
        await winner_workspace.shutdown()
        owner.shutdown()
        slow_replica.close()
        winner_replica.close()


@pytest.mark.parametrize("selection_timing", ("during", "after"))
@pytest.mark.asyncio
async def test_fresh_shared_workspace_follows_committed_owner_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection_timing: str,
) -> None:
    old_root = (tmp_path / "old").resolve()
    winner_root = (tmp_path / "winner").resolve()
    old_root.mkdir()
    winner_root.mkdir()
    (winner_root / "winner.md").write_text("winner", encoding="utf-8")
    owner = FileNotesSessionOwner()
    winner_replica = FileNotesReplica(":memory:")
    fresh_replica = FileNotesReplica(":memory:")
    winner = LibraryFileNotesWorkspace(
        root=old_root,
        replica=winner_replica,
        session_owner=owner,
        poll_interval=10,
    )
    persisted_root = [str(old_root)]
    persistence_started = threading.Event()
    release_persistence = threading.Event()
    config_read = threading.Event()
    allow_config_return = threading.Event()
    event_loop = asyncio.get_running_loop()
    fresh: LibraryFileNotesWorkspace | None = None

    def get_setting(
        section: str,
        key: str | None = None,
        default: object = None,
    ) -> object:
        if (section, key) == ("file_notes", "root"):
            configured = persisted_root[0]
            config_read.set()
            if selection_timing == "after":
                assert allow_config_return.wait(timeout=5)
            return configured
        return default

    def persist(
        section_values: dict[str, dict[str, str]],
    ) -> ConfigMutationResult:
        persistence_started.set()
        assert release_persistence.wait(timeout=5)
        persisted_root[0] = section_values["file_notes"]["root"]
        return ConfigMutationResult(True, True, None)

    monkeypatch.setattr(workspace_module, "get_cli_setting", get_setting)
    monkeypatch.setattr(
        workspace_module,
        "apply_settings_mutation_to_cli_config",
        persist,
        raising=False,
    )
    heartbeat_thread, heartbeat_checked, heartbeat_while_waiting = (
        _event_loop_heartbeat(
            event_loop,
            config_read,
            release_persistence,
            allow_config_return,
        )
    )
    transition: asyncio.Task[bool] | None = None
    heartbeat_started = False
    try:
        harness = _DynamicWorkspaceHarness(winner)
        async with harness.run_test() as pilot:
            await _wait_until(
                pilot,
                lambda: winner.initialized,
                "winner workspace did not initialize",
            )
            old_binding = owner.current_binding()
            assert old_binding is not None
            fresh = LibraryFileNotesWorkspace(
                replica=fresh_replica,
                session_owner=owner,
                poll_interval=10,
            )
            assert fresh._initial_session_binding == old_binding
            transition = asyncio.create_task(winner.set_root(winner_root))
            await _wait_until(
                pilot,
                persistence_started.is_set,
                "winner persistence did not start",
            )

            heartbeat_thread.start()
            heartbeat_started = True
            await harness.query_one(
                "#dynamic-workspace-host",
                Vertical,
            ).mount(fresh)
            await _wait_until(
                pilot,
                config_read.is_set,
                "fresh workspace did not read configured root",
            )
            if selection_timing == "during":
                await pilot.pause()
                assert not fresh.initialized
            release_persistence.set()
            assert await transition
            allow_config_return.set()
            await _wait_until(
                pilot,
                heartbeat_checked.is_set,
                "event-loop heartbeat was not checked",
            )
            assert heartbeat_while_waiting == [True]
            await _wait_until(
                pilot,
                lambda: fresh.initialized and fresh._service is not None,
                "fresh workspace did not initialize after root commit",
            )

            binding = owner.current_binding()
            assert binding is not None
            assert fresh.root == winner_root
            assert fresh._session_binding == binding
            assert fresh._service is not None
            assert fresh._service.root == winner_root
            assert set(fresh.entries) == {"winner.md"}
    finally:
        release_persistence.set()
        allow_config_return.set()
        if transition is not None and not transition.done():
            assert await transition
        if heartbeat_started:
            heartbeat_thread.join(timeout=1)
        await winner.shutdown()
        if fresh is not None:
            await fresh.shutdown()
        owner.shutdown()
        winner_replica.close()
        fresh_replica.close()


@pytest.mark.asyncio
async def test_bound_injected_owner_overrides_unrelated_explicit_seed(
    tmp_path: Path,
) -> None:
    owner_root = (tmp_path / "owner").resolve()
    unrelated_root = (tmp_path / "unrelated").resolve()
    owner_root.mkdir()
    unrelated_root.mkdir()
    (owner_root / "owner.md").write_text("owner", encoding="utf-8")
    owner = FileNotesSessionOwner()
    binding = owner.select_root(owner_root)
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=unrelated_root,
        replica=replica,
        session_owner=owner,
        poll_interval=10,
    )
    try:
        async with _WorkspaceHarness(workspace).run_test() as pilot:
            await _wait_until(
                pilot,
                lambda: workspace.initialized and workspace._service is not None,
                "bound-owner workspace did not initialize",
            )
            assert owner.current_binding() == binding
            assert workspace.root == owner_root
            assert workspace._session_binding == binding
            assert set(workspace.entries) == {"owner.md"}
    finally:
        await workspace.shutdown()
        owner.shutdown()
        replica.close()


@pytest.mark.asyncio
async def test_failed_root_persistence_keeps_old_owner_log_and_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_root, new_root, owner, replica, workspace = _root_transition_workspace(tmp_path)

    def fail_persistence(*_args: object, **_kwargs: object) -> None:
        raise OSError("forced persistence failure")

    monkeypatch.setattr(
        workspace_module,
        "apply_settings_mutation_to_cli_config",
        fail_persistence,
        raising=False,
    )
    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _wait_until(
            pilot,
            lambda: workspace.initialized,
            "old workspace did not initialize",
        )
        old_service = workspace._service
        old_binding = workspace._session_binding
        assert old_service is not None
        assert old_binding is not None
        assert old_service.create_file("before.md", "before").status == "ok"

        with pytest.raises(OSError, match="forced persistence failure"):
            await workspace.set_root(new_root)

        assert workspace.root == old_root.resolve()
        assert workspace._service is old_service
        assert workspace._session_binding == old_binding
        assert old_service.create_file("after.md", "after").status == "ok"
        assert [
            item.change.relative_path for item in owner.snapshot(old_binding).changes
        ] == ["before.md", "after.md"]

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_before_replace_root_failure_keeps_old_owner_log_and_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_root, new_root, owner, replica, workspace = _root_transition_workspace(tmp_path)

    monkeypatch.setattr(
        workspace_module,
        "apply_settings_mutation_to_cli_config",
        lambda *_args, **_kwargs: ConfigMutationResult(
            False,
            False,
            "before_replace",
        ),
        raising=False,
    )
    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _wait_until(
            pilot,
            lambda: workspace.initialized,
            "old workspace did not initialize",
        )
        old_service = workspace._service
        old_binding = workspace._session_binding
        assert old_service is not None
        assert old_binding is not None
        assert old_service.create_file("before.md", "before").status == "ok"

        assert not await workspace.set_root(new_root)

        assert workspace.root == old_root.resolve()
        assert workspace._service is old_service
        assert workspace._session_binding == old_binding
        assert old_service.create_file("after.md", "after").status == "ok"
        assert [
            item.change.relative_path for item in owner.snapshot(old_binding).changes
        ] == ["before.md", "after.md"]

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_cache_reload_failure_adopts_persisted_root_with_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_root, new_root, owner, replica, workspace = _root_transition_workspace(tmp_path)

    monkeypatch.setattr(
        workspace_module,
        "apply_settings_mutation_to_cli_config",
        lambda *_args, **_kwargs: ConfigMutationResult(
            True,
            False,
            "cache_reload",
        ),
        raising=False,
    )
    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _wait_until(
            pilot,
            lambda: workspace.initialized,
            "old workspace did not initialize",
        )
        old_service = workspace._service
        old_binding = workspace._session_binding
        assert old_service is not None
        assert old_binding is not None
        assert old_service.create_file("before.md", "before").status == "ok"

        assert await workspace.set_root(new_root)

        new_binding = workspace._session_binding
        assert new_binding is not None
        assert new_binding != old_binding
        assert workspace.root == new_root.resolve()
        assert workspace._service is not old_service
        assert owner.current_binding() == new_binding
        assert "cache reload" in workspace._runtime_warning.lower()

        monkeypatch.setattr(
            workspace_module,
            "apply_settings_mutation_to_cli_config",
            lambda *_args, **_kwargs: ConfigMutationResult(True, True, None),
            raising=False,
        )
        assert await workspace.set_root(old_root)
        assert workspace._runtime_warning == ""

    await workspace.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_cancelled_root_persistence_settles_and_adopts_written_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_root, new_root, owner, replica, workspace = _root_transition_workspace(tmp_path)
    (old_root / "open.md").write_text("old root", encoding="utf-8")
    (old_root / "deleted.md").write_text("old tombstone", encoding="utf-8")
    (new_root / "new.md").write_text("new root", encoding="utf-8")
    persistence_started = threading.Event()
    release_persistence = threading.Event()
    persistence_finished = threading.Event()
    persisted_roots: list[str] = []
    event_loop = asyncio.get_running_loop()

    def persist(
        section_values: dict[str, dict[str, str]],
    ) -> ConfigMutationResult:
        persistence_started.set()
        assert release_persistence.wait(timeout=5)
        persisted_roots.append(section_values["file_notes"]["root"])
        persistence_finished.set()
        return ConfigMutationResult(True, True, None)

    monkeypatch.setattr(
        workspace_module,
        "apply_settings_mutation_to_cli_config",
        persist,
        raising=False,
    )
    heartbeat_thread, heartbeat_checked, heartbeat_during_persistence = (
        _event_loop_heartbeat(
            event_loop,
            persistence_started,
            release_persistence,
        )
    )
    transition: asyncio.Task[bool] | None = None
    try:
        async with _WorkspaceHarness(workspace).run_test() as pilot:
            await _wait_until(
                pilot,
                lambda: workspace.initialized,
                "old workspace did not initialize",
            )
            old_service = workspace._service
            assert old_service is not None
            assert old_service.delete_file("deleted.md").status == "ok"
            assert await workspace.refresh_files()
            assert set(workspace.entries) == {"open.md"}
            assert workspace._deleted_paths == ("deleted.md",)
            assert await workspace.open_path("open.md")
            assert workspace.current_document is not None

            heartbeat_thread.start()
            transition = asyncio.create_task(workspace.set_root(new_root))
            await _wait_until(
                pilot,
                heartbeat_checked.is_set,
                "event-loop heartbeat was not checked",
            )
            assert heartbeat_during_persistence == [True]

            transition.cancel()
            await pilot.pause()
            assert not transition.done()
            assert not persistence_finished.is_set()

            release_persistence.set()
            with pytest.raises(asyncio.CancelledError):
                await transition

            binding = workspace._session_binding
            assert persistence_finished.is_set()
            assert persisted_roots == [str(new_root.resolve())]
            assert binding is not None
            assert workspace.root == new_root.resolve()
            assert workspace._service is not None
            assert workspace._service.root == new_root.resolve()
            assert owner.current_binding() == binding
            assert workspace.current_document is None
            assert workspace.current_path == ""
            assert set(workspace.entries) == {"new.md"}
            assert workspace._deleted_paths == ()
            assert workspace._root_offline is False
            tree_labels = _tree_labels(workspace.query_one("#file-notes-tree", Tree))
            assert "new.md" in tree_labels
            assert "open.md" not in tree_labels
            assert "deleted.md" not in tree_labels
    finally:
        release_persistence.set()
        if transition is not None and not transition.done():
            transition.cancel()
            with pytest.raises(asyncio.CancelledError):
                await transition
        heartbeat_thread.join(timeout=1)
        await workspace.shutdown()
        owner.shutdown()
        replica.close()


@pytest.mark.asyncio
async def test_injected_owner_shutdown_waits_for_root_commit_before_replica_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_root = (tmp_path / "old").resolve()
    new_root = (tmp_path / "new").resolve()
    old_root.mkdir()
    new_root.mkdir()
    (new_root / "new.md").write_text("new root", encoding="utf-8")
    owner = FileNotesSessionOwner()
    workspace = LibraryFileNotesWorkspace(
        root=old_root,
        replica_path=tmp_path / "owned.sqlite",
        session_owner=owner,
        poll_interval=10,
    )
    persistence_started = threading.Event()
    release_persistence = threading.Event()
    persistence_finished = threading.Event()
    owner_wait_entered = threading.Event()
    replica_closed = threading.Event()
    close_observations: list[tuple[bool, Path | None, object]] = []
    owned_replica: FileNotesReplica | None = None
    real_wait = FileNotesSessionOwner.wait_for_root_commit
    real_close = FileNotesReplica.close

    def persist(
        _section_values: dict[str, dict[str, str]],
    ) -> ConfigMutationResult:
        persistence_started.set()
        assert release_persistence.wait(timeout=5)
        persistence_finished.set()
        return ConfigMutationResult(True, True, None)

    def observed_wait(session_owner: FileNotesSessionOwner) -> None:
        if session_owner is owner:
            owner_wait_entered.set()
        real_wait(session_owner)

    def observed_close(replica: FileNotesReplica) -> None:
        if replica is owned_replica:
            service = workspace._service
            close_observations.append(
                (
                    persistence_finished.is_set(),
                    None if service is None else service.root,
                    owner.current_binding(),
                )
            )
            replica_closed.set()
        real_close(replica)

    monkeypatch.setattr(
        workspace_module,
        "apply_settings_mutation_to_cli_config",
        persist,
        raising=False,
    )
    monkeypatch.setattr(
        FileNotesSessionOwner,
        "wait_for_root_commit",
        observed_wait,
    )
    monkeypatch.setattr(FileNotesReplica, "close", observed_close)
    transition: asyncio.Task[bool] | None = None
    shutdown_task: asyncio.Task[None] | None = None
    try:
        async with _WorkspaceHarness(workspace).run_test() as pilot:
            await _wait_until(
                pilot,
                lambda: workspace.initialized,
                "owned-replica workspace did not initialize",
            )
            owned_replica = workspace._replica
            assert owned_replica is not None

            transition = asyncio.create_task(workspace.set_root(new_root))
            await _wait_until(
                pilot,
                persistence_started.is_set,
                "root persistence did not start",
            )
            shutdown_task = asyncio.create_task(workspace.shutdown())
            await _wait_until(
                pilot,
                lambda: owner_wait_entered.is_set() or replica_closed.is_set(),
                "shutdown neither waited nor closed the replica",
            )

            transition.cancel()
            await pilot.pause()
            assert not transition.done()
            assert owner_wait_entered.is_set()
            assert not replica_closed.is_set()
            assert not shutdown_task.done()
            assert workspace._replica is owned_replica

            release_persistence.set()
            with pytest.raises(asyncio.CancelledError):
                await transition
            await shutdown_task

            binding = owner.current_binding()
            assert persistence_finished.is_set()
            assert replica_closed.is_set()
            assert close_observations == [(True, new_root, binding)]
            assert binding is not None
            assert binding.root_key == str(new_root)
            assert workspace._replica is None
            assert workspace._service is None
            await pilot.pause()
            assert workspace._replica is None
            assert workspace._service is None

            status = owner.try_acquire_status(binding)
            assert status is not None
            status.release()
    finally:
        release_persistence.set()
        if transition is not None and not transition.done():
            transition.cancel()
            with pytest.raises(asyncio.CancelledError):
                await transition
        if shutdown_task is not None and not shutdown_task.done():
            await shutdown_task
        await workspace.shutdown()
        owner.shutdown()


@pytest.mark.asyncio
async def test_stale_candidate_scan_keeps_old_owner_log_and_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    old_root.mkdir()
    new_root.mkdir()
    owner = FileNotesSessionOwner()
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=old_root,
        replica=replica,
        session_owner=owner,
        poll_interval=10,
    )
    candidate_scan_started = threading.Event()
    release_candidate_scan = threading.Event()
    real_scan = FileNotesService.scan

    def delayed_candidate_scan(service: FileNotesService):
        if service.root == new_root.resolve():
            candidate_scan_started.set()
            assert release_candidate_scan.wait(timeout=5)
        return real_scan(service)

    monkeypatch.setattr(FileNotesService, "scan", delayed_candidate_scan)
    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _wait_until(
            pilot,
            lambda: workspace.initialized,
            "old workspace did not initialize",
        )
        old_service = workspace._service
        old_binding = workspace._session_binding
        assert old_service is not None
        assert old_binding is not None
        assert old_service.create_file("before.md", "before").status == "ok"

        transition = asyncio.create_task(workspace.set_root(new_root, persist=False))
        await _wait_until(
            pilot,
            candidate_scan_started.is_set,
            "candidate scan did not start",
        )
        workspace.on_unmount()
        release_candidate_scan.set()
        assert not await transition

        assert workspace.root == old_root.resolve()
        assert workspace._service is old_service
        assert workspace._session_binding == old_binding
        assert old_service.create_file("after.md", "after").status == "ok"
        assert [
            item.change.relative_path for item in owner.snapshot(old_binding).changes
        ] == ["before.md", "after.md"]

    release_candidate_scan.set()
    await workspace.shutdown()
    owner.shutdown()
    replica.close()


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
        assert {"folder", "beta.txt"}.issubset(_tree_labels(tree))
        folder = next(
            node
            for node in tree.root.children
            if getattr(node.label, "plain", str(node.label)) == "folder"
        )
        folder.expand()
        await _wait_until(
            pilot,
            lambda: "alpha.md" in _tree_labels(tree),
            "expanded folder did not mount its first bounded batch",
        )

        search = workspace.query_one("#file-notes-search", Input)
        search.value = "needle"
        await _wait_until(
            pilot,
            lambda: (
                workspace.query_one("#file-notes-search-results", Tree).display
                and "folder/alpha.md"
                in _tree_labels(workspace.query_one("#file-notes-search-results", Tree))
            ),
            "search results did not replace the tree with the mounted match",
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
    await workspace.shutdown()
    assert replica.list_deleted(str(root.resolve())) == []
    replica.close()


@pytest.mark.asyncio
async def test_save_status_names_local_folder_and_preserved_draft(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "note.md").write_text("body", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        poll_interval=10,
        autosave_delay=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert _static_text(workspace, "#file-notes-save-status") == (
            "Auto-save to local folder: idle"
        )

        assert await workspace.open_path("note.md")
        assert _static_text(workspace, "#file-notes-save-status") == (
            "Saved to local folder"
        )
        workspace._set_save_state("dirty")
        assert _static_text(workspace, "#file-notes-save-status") == (
            "Auto-save pending for local folder"
        )
        workspace._set_save_state("saving")
        assert _static_text(workspace, "#file-notes-save-status") == (
            "Saving to local folder…"
        )
        workspace._set_save_state("conflict", "file changed on disk")
        assert _static_text(workspace, "#file-notes-save-status") == (
            "Conflict: draft preserved in editor; file changed on disk"
        )
        save_copy = workspace.query_one("#file-notes-save-copy", Button)
        assert str(save_copy.label) == "Save copy"
        assert save_copy.display
        resolve = workspace.query_one("#file-notes-resolve-conflict", Button)
        assert resolve.display
        assert str(resolve.label) == "Resolve conflict"
        assert workspace.query_one("#file-notes-save-status").display

    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ((120, 40), (40, 20)))
async def test_maintenance_disclosure_keeps_secondary_file_actions_reachable(
    tmp_path: Path,
    size: tuple[int, int],
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "note.md").write_text("body", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        poll_interval=10,
        autosave_delay=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=size) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("note.md")
        await pilot.pause()

        toggle = workspace.query_one("#file-notes-maintenance-toggle", Button)
        maintenance = workspace.query_one("#file-notes-maintenance-actions")
        assert str(toggle.label) == "More file actions"
        assert toggle.render_line(0).text.strip() == "More file actions"
        assert toggle.region.right <= workspace.region.right
        assert not maintenance.display
        assert workspace.query_one("#file-notes-move", Button).display
        assert not workspace.query_one("#file-notes-protect", Button).display

        toggle.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert str(toggle.label) == "Hide file actions"
        assert toggle.render_line(0).text.strip() == "Hide file actions"
        assert toggle.region.right <= workspace.region.right
        assert maintenance.display
        assert toggle.has_focus
        assert {
            button.id for button in maintenance.query(Button) if button.display
        } == {
            "file-notes-protect",
            "file-notes-refresh",
        }
        assert workspace.query_one("#file-notes-reload", Button).display
        _assert_visible_editor_actions_fit(workspace)

        protect = workspace.query_one("#file-notes-protect", Button)
        protect.focus()
        await pilot.pause()
        assert protect.has_focus
        toggle.press()
        await pilot.pause()
        assert str(toggle.label) == "More file actions"
        assert not maintenance.display
        assert toggle.has_focus

    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_create_move_delete_protect_and_restore_use_real_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    async with _production_workspace_context(
        workspace,
        size=(120, 40),
    ) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("start.md")

        editor = workspace.query_one("#file-notes-editor", TextArea)
        service = workspace._service
        assert service is not None
        delayed_create, create_started, release_create = _delayed_call(
            service.create_file
        )
        monkeypatch.setattr(service, "create_file", delayed_create)
        path_input = workspace.query_one("#file-notes-path", Input)
        path_input.value = "created.md"
        create_button = workspace.query_one("#file-notes-new", Button)
        creating = asyncio.create_task(
            workspace._new_file(Button.Pressed(create_button))
        )
        await _wait_until(pilot, create_started.is_set, "new file did not start")
        editor.focus()
        await pilot.press("x")
        state_during_create = (editor.read_only, workspace.leave_allowed, editor.text)
        release_create.set()
        await creating
        await _wait_until(
            pilot,
            lambda: workspace.current_path == "created.md",
            "new file did not open",
        )
        assert state_during_create == (True, False, "start")
        assert (root / "created.md").exists()

        await _show_maintenance_actions(workspace, pilot)
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
        assert replica.list_deleted(str(root.resolve())) == []
        assert "Recently deleted" not in _tree_labels(
            workspace.query_one("#file-notes-tree", Tree)
        )

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
            lambda: (
                not (root / "moved.md").exists()
                and "Recently deleted"
                in _tree_labels(workspace.query_one("#file-notes-tree", Tree))
            ),
            "confirmed delete did not finish updating the tree",
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
        assert (
            _static_text(workspace, "#file-notes-session-changes")
            == "Review session changes (1)"
        )
    replica.close()


@pytest.mark.asyncio
async def test_injected_owner_retains_same_root_log_across_workspaces(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    owner = FileNotesSessionOwner()
    replica = FileNotesReplica(":memory:")
    first = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        session_owner=owner,
        poll_interval=10,
    )

    async with _WorkspaceHarness(first).run_test() as pilot:
        await _wait_until(pilot, lambda: first.initialized, "first scan did not finish")
        service = first._service
        assert service is not None
        assert service.create_file("retained.md", "retained").status == "ok"
        first._refresh_session_changes()
        assert (
            _static_text(first, "#file-notes-session-changes")
            == "Review session changes (1)"
        )
    await first.shutdown()

    binding = owner.select_root(root)
    status = owner.try_acquire_status(binding)
    assert status is not None
    status.release()

    second = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        session_owner=owner,
        poll_interval=10,
    )
    async with _WorkspaceHarness(second).run_test() as pilot:
        await _wait_until(
            pilot,
            lambda: second.initialized,
            "second scan did not finish",
        )
        assert (
            _static_text(second, "#file-notes-session-changes")
            == "Review session changes (1)"
        )
    await second.shutdown()
    owner.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_direct_workspace_shuts_down_only_its_private_session_owner(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=root, replica=replica, poll_interval=10)

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _wait_until(
            pilot,
            lambda: workspace.initialized,
            "workspace scan did not finish",
        )
        owner = workspace._session_owner
        binding = owner.select_root(root)

    await workspace.shutdown()
    await workspace.shutdown()

    assert owner.try_acquire_status(binding) is None
    replica.close()


@pytest.mark.parametrize(
    ("button_id", "handler_name", "service_method", "action"),
    (
        ("#file-notes-new", "_new_file", "create_file", "Create"),
        ("#file-notes-move", "_move_file", "move_file", "Move"),
        ("#file-notes-restore", "_restore_file", "restore_file", "Restore"),
        (
            "#file-notes-save-copy",
            "_save_copy",
            "save_copy",
            "Save draft as copy",
        ),
    ),
)
@pytest.mark.asyncio
async def test_raw_path_actions_validate_input_before_service_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    button_id: str,
    handler_name: str,
    service_method: str,
    action: str,
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
        service = workspace._service
        assert service is not None
        validation_calls: list[tuple[str, int, bool]] = []
        calls: list[tuple[object, ...]] = []
        raw_path = (" " * 4097) + r"nested/double..dots\<script>note.md"
        destination = raw_path.strip()
        real_validate_text_input = workspace_module.validate_text_input

        def capture_path_validation(
            text: str,
            max_length: int = 10000,
            allow_html: bool = False,
        ) -> bool:
            validation_calls.append((text, max_length, allow_html))
            return real_validate_text_input(
                text,
                max_length=max_length,
                allow_html=allow_html,
            )

        def capture_call(*args: object) -> OperationResult:
            calls.append(args)
            return OperationResult(
                status="error",
                relative_path=destination,
                message="service should not receive invalid input",
            )

        monkeypatch.setattr(
            workspace_module,
            "validate_text_input",
            capture_path_validation,
        )
        monkeypatch.setattr(service, service_method, capture_call)
        workspace.query_one("#file-notes-path", Input).value = raw_path
        button = workspace.query_one(button_id, Button)
        await getattr(workspace, handler_name)(Button.Pressed(button))

        assert validation_calls == [(raw_path, 4096, True)]
        assert calls == []
        assert (
            _static_text(
                workspace,
                "#file-notes-action-status",
            )
            == f"{action} failed: unsupported path text."
        )
        assert workspace.current_path == "start.md"
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
        await _show_maintenance_actions(workspace, pilot)
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
        workspace.query_one("#file-notes-resolve-conflict", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.conflict_resolution_active,
            "conflict choices did not open",
        )
        workspace.query_one("#file-notes-resolution-save-new", Button).press()
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
            lambda: workspace.reload_confirmation_active,
            "reload confirmation did not open",
        )
        assert editor.text == "another draft"
        workspace.query_one("#file-notes-reload-confirm", Button).press()
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
        assert str(workspace.query_one("#file-notes-reload", Button).label) == (
            "Discard draft and reload"
        )
        workspace.query_one("#file-notes-reload", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.reload_confirmation_active,
            "error-state reload confirmation did not open",
        )
        assert workspace.save_state == "error"
        assert editor.text == "surviving error draft"
        workspace.query_one("#file-notes-reload-cancel", Button).press()
        await _wait_until(
            pilot,
            lambda: not workspace.reload_confirmation_active,
            "error-state reload confirmation did not cancel",
        )
        assert workspace.save_state == "error"
        assert editor.text == "surviving error draft"
    replica.close()


@pytest.mark.asyncio
@pytest.mark.allow_network
@pytest.mark.parametrize("size", [(40, 20), (120, 40)])
async def test_conflict_compare_preserves_draft_and_restores_opener_focus_once(
    tmp_path: Path,
    size: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare labels all sides without resolving or mutating the conflict."""
    root = tmp_path / "notes"
    root.mkdir()
    source = root / "source.md"
    source.write_text("base line\nshared\n", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        autosave_delay=10,
        poll_interval=10,
    )
    ui_thread = threading.get_ident()
    comparison_threads: list[int] = []
    compare_focus_calls = 0
    original_builder = workspace_module.build_conflict_comparison
    original_focus = Button.focus

    def observed_builder(*args):
        comparison_threads.append(threading.get_ident())
        return original_builder(*args)

    def count_compare_focus(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal compare_focus_calls
        if self.id == "file-notes-compare":
            compare_focus_calls += 1
        return original_focus(self, *args, **kwargs)

    monkeypatch.setattr(
        workspace_module,
        "build_conflict_comparison",
        observed_builder,
    )
    monkeypatch.setattr(Button, "focus", count_compare_focus)

    async with _WorkspaceHarness(workspace).run_test(size=size) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("source.md")
        compare = workspace.query_one("#file-notes-compare", Button)
        assert not compare.display

        editor = workspace.query_one("#file-notes-editor", TextArea)
        _replace_editor_text(editor, "draft line\nshared\n")
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "dirty",
            "draft did not become dirty",
        )
        source.write_text("disk line\nshared\n", encoding="utf-8")
        await workspace.refresh_files()
        assert workspace.save_state == "conflict"
        assert compare.display and not compare.disabled

        compare.focus()
        compare.press()
        await _wait_until(
            pilot,
            lambda: isinstance(pilot.app.screen, FileNotesConflictCompareDialog),
            "conflict comparison did not open",
        )
        dialog = pilot.app.screen
        assert isinstance(dialog, FileNotesConflictCompareDialog)
        assert comparison_threads
        assert all(thread_id != ui_thread for thread_id in comparison_threads)
        assert dialog.query_one("#file-notes-conflict-dialog").region.right <= size[0]
        assert dialog.query_one("#file-notes-conflict-dialog").region.bottom <= size[1]
        summary = dialog.query_one("#file-notes-conflict-summary", TextArea).text
        diff = dialog.query_one("#file-notes-conflict-diff", TextArea).text
        assert "Base · editor baseline" in summary
        assert "Draft · current editor" in summary
        assert "Disk · latest readable snapshot" in summary
        assert "Base → Draft" in diff
        assert "+draft line" in diff
        assert "Base → Disk" in diff
        assert "+disk line" in diff
        assert workspace.save_state == "conflict"
        assert editor.text == "draft line\nshared\n"
        assert source.read_text(encoding="utf-8") == "disk line\nshared\n"

        compare_focus_calls = 0
        await pilot.press("escape")
        await _wait_until(
            pilot,
            lambda: pilot.app.screen is pilot.app.screen_stack[0],
            "conflict comparison did not close",
        )
        await _wait_until(
            pilot,
            lambda: compare.has_focus,
            "focus did not return to Compare",
        )
        assert compare_focus_calls == 1
        assert workspace.save_state == "conflict"
        assert editor.text == "draft line\nshared\n"

    replica.close()


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_conflict_compare_represents_deleted_disk(
    tmp_path: Path,
) -> None:
    """A deleted disk side remains explicit while the editor draft stays live."""
    root = tmp_path / "notes"
    root.mkdir()
    source = root / "source.md"
    source.write_text("base", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        autosave_delay=10,
        poll_interval=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=(80, 24)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("source.md")
        editor = workspace.query_one("#file-notes-editor", TextArea)
        _replace_editor_text(editor, "retained draft")
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "dirty",
            "draft did not become dirty",
        )
        source.unlink()
        await workspace.refresh_files()
        assert workspace.save_state == "conflict"

        workspace.query_one("#file-notes-compare", Button).press()
        await _wait_until(
            pilot,
            lambda: isinstance(pilot.app.screen, FileNotesConflictCompareDialog),
            "deleted-side comparison did not open",
        )
        dialog = pilot.app.screen
        assert isinstance(dialog, FileNotesConflictCompareDialog)
        summary = dialog.query_one("#file-notes-conflict-summary", TextArea).text
        diff = dialog.query_one("#file-notes-conflict-diff", TextArea).text
        assert "Disk · absent" in summary
        assert "Disk is absent; no textual diff is available." in diff
        assert editor.text == "retained draft"
        assert workspace.save_state == "conflict"

    replica.close()


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_conflict_compare_rejects_a_late_editor_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disk read finishing for a stale editor cannot publish comparison."""
    root = tmp_path / "notes"
    root.mkdir()
    source = root / "source.md"
    source.write_text("base", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        autosave_delay=10,
        poll_interval=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=(80, 24)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("source.md")
        editor = workspace.query_one("#file-notes-editor", TextArea)
        _replace_editor_text(editor, "retained draft")
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "dirty",
            "draft did not become dirty",
        )
        source.write_text("disk", encoding="utf-8")
        await workspace.refresh_files()
        assert workspace.save_state == "conflict"
        service = workspace._service
        assert service is not None
        original_open = service.open_file
        read_started = threading.Event()
        release_read = threading.Event()

        def delayed_open(relative_path: str):
            read_started.set()
            assert release_read.wait(2)
            return original_open(relative_path)

        monkeypatch.setattr(service, "open_file", delayed_open)
        compare = workspace.query_one("#file-notes-compare", Button)
        comparison_task = asyncio.create_task(
            workspace._compare_conflict(Button.Pressed(compare))
        )
        assert await asyncio.to_thread(read_started.wait, 2)
        workspace._session_key = "replacement-session"
        release_read.set()
        await comparison_task

        assert not isinstance(pilot.app.screen, FileNotesConflictCompareDialog)
        assert workspace.save_state == "conflict"
        assert editor.text == "retained draft"
        status = _static_text(workspace, "#file-notes-action-status")
        assert "editing session changed" in status
        assert "Draft preserved" in status

    replica.close()


@pytest.mark.asyncio
@pytest.mark.allow_network
@pytest.mark.parametrize("size", [(40, 20), (120, 40)])
async def test_conflict_resolution_discloses_only_safe_choices(
    tmp_path: Path,
    size: tuple[int, int],
) -> None:
    """Conflict choices stay explicit, bounded, and non-resolving by default."""
    root = tmp_path / "notes"
    root.mkdir()
    source = root / "source.md"
    source.write_text("base", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        autosave_delay=10,
        poll_interval=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=size) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("source.md")
        resolve = workspace.query_one("#file-notes-resolve-conflict", Button)
        assert not resolve.display

        editor = workspace.query_one("#file-notes-editor", TextArea)
        _replace_editor_text(editor, "retained draft")
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "dirty",
            "draft did not become dirty",
        )
        source.write_text("latest disk", encoding="utf-8")
        await workspace.refresh_files()
        assert workspace.save_state == "conflict"
        assert resolve.display and not resolve.disabled
        assert workspace.query_one("#file-notes-save-copy", Button).display

        resolve.focus()
        resolve.press()
        await _wait_until(
            pilot,
            lambda: workspace.conflict_resolution_active,
            "conflict choices did not open",
        )
        keep = workspace.query_one("#file-notes-resolution-keep", Button)
        save_new = workspace.query_one(
            "#file-notes-resolution-save-new",
            Button,
        )
        assert save_new.tooltip == (
            "Write the complete draft to the Target path without replacing "
            "an existing file"
        )
        discard = workspace.query_one(
            "#file-notes-resolution-discard",
            Button,
        )
        choices = (keep, save_new, discard)
        assert [str(choice.label) for choice in choices] == [
            "Keep editing",
            "Save draft as new note",
            "Discard draft and load disk",
        ]
        assert all("overwrite" not in str(choice.label).lower() for choice in choices)
        await _wait_until(
            pilot,
            lambda: keep.has_focus,
            "resolution choices did not focus Keep editing",
        )
        assert _static_text(workspace, "#file-notes-resolution-copy") == (
            "Choose a safe next step. No option overwrites the disk file."
        )
        assert _static_text(workspace, "#file-notes-path-label") == (
            "Target path · New / Move / Save copy"
        )
        assert workspace.query_one("#file-notes-compare", Button).display
        assert not workspace.query_one("#file-notes-delete", Button).display
        for choice in choices:
            assert choice.display and not choice.disabled
            choice.focus()
            choice.scroll_visible(animate=False)
            await pilot.pause()
            assert choice.has_focus
            painted_label = choice.render_line(0).text.strip()
            if size == (40, 20):
                assert choice.render().plain == str(choice.label)
                assert painted_label
                assert str(choice.label).startswith(painted_label)
            else:
                assert painted_label == str(choice.label)
            assert choice.region.right <= workspace.region.right
            assert choice.region.bottom <= workspace.region.bottom

        keep.press()
        await _wait_until(
            pilot,
            lambda: not workspace.conflict_resolution_active,
            "Keep editing did not close the choices",
        )
        await _wait_until(
            pilot,
            lambda: resolve.has_focus,
            "Keep editing did not return focus to Resolve conflict",
        )
        assert workspace.save_state == "conflict"
        assert editor.text == "retained draft"
        assert source.read_text(encoding="utf-8") == "latest disk"

    replica.close()


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_conflict_resolution_saves_draft_as_new_note_without_clobber(
    tmp_path: Path,
) -> None:
    """Save draft as new note preserves source Disk and exact body style."""
    root = tmp_path / "notes"
    root.mkdir()
    source = root / "source.md"
    source.write_bytes(b"\xef\xbb\xbf---\r\ntitle: Base\r\n---\r\nbase\r\n")
    occupied = root / "occupied.md"
    occupied.write_text("do not replace", encoding="utf-8")
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
        editor = workspace.query_one("#file-notes-editor", TextArea)
        _replace_editor_text(editor, "retained\ndraft")
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "dirty",
            "draft did not become dirty",
        )
        source.write_bytes(b"\xef\xbb\xbf---\r\ntitle: Disk\r\n---\r\nlatest\r\n")
        disk_bytes = source.read_bytes()
        await workspace.refresh_files()
        workspace.query_one("#file-notes-resolve-conflict", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.conflict_resolution_active,
            "conflict choices did not open",
        )
        path = workspace.query_one("#file-notes-path", Input)
        save_new = workspace.query_one(
            "#file-notes-resolution-save-new",
            Button,
        )

        path.value = "occupied.md"
        save_new.press()
        await _wait_until(
            pilot,
            lambda: (
                "already exists"
                in _static_text(
                    workspace,
                    "#file-notes-action-status",
                ).lower()
            ),
            "existing destination did not fail closed",
        )
        assert occupied.read_text(encoding="utf-8") == "do not replace"
        assert source.read_bytes() == disk_bytes
        assert editor.text == "retained\ndraft"
        assert workspace.save_state == "conflict"
        assert workspace.conflict_resolution_active

        path.value = "recovered.md"
        save_new.press()
        await _wait_until(
            pilot,
            lambda: workspace.current_path == "recovered.md",
            "safe draft copy did not open as the current note",
        )
        assert (root / "recovered.md").read_bytes() == (
            b"\xef\xbb\xbf---\r\ntitle: Base\r\n---\r\nretained\r\ndraft\r\n"
        )
        assert source.read_bytes() == disk_bytes
        assert workspace.save_state == "saved"
        assert not workspace.conflict_resolution_active
        assert editor.has_focus

    replica.close()


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_conflict_resolution_discard_keeps_cancel_first_confirmation(
    tmp_path: Path,
) -> None:
    """Discard routes through the existing revalidated, safe-default decision."""
    root = tmp_path / "notes"
    root.mkdir()
    source = root / "source.md"
    source.write_text("base", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        autosave_delay=10,
        poll_interval=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=(80, 24)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("source.md")
        editor = workspace.query_one("#file-notes-editor", TextArea)
        _replace_editor_text(editor, "retained draft")
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "dirty",
            "draft did not become dirty",
        )
        source.write_text("latest disk", encoding="utf-8")
        await workspace.refresh_files()
        workspace.query_one("#file-notes-resolve-conflict", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.conflict_resolution_active,
            "conflict choices did not open",
        )
        discard = workspace.query_one(
            "#file-notes-resolution-discard",
            Button,
        )
        discard.press()
        await _wait_until(
            pilot,
            lambda: workspace.reload_confirmation_active,
            "discard did not open the destructive confirmation",
        )
        cancel = workspace.query_one("#file-notes-reload-cancel", Button)
        assert cancel.has_focus
        assert editor.text == "retained draft"
        assert workspace.save_state == "conflict"

        cancel.press()
        await _wait_until(
            pilot,
            lambda: not workspace.reload_confirmation_active,
            "Cancel did not close destructive confirmation",
        )
        await _wait_until(
            pilot,
            lambda: discard.has_focus,
            "Cancel did not return focus to the resolution choice",
        )
        assert workspace.conflict_resolution_active
        assert editor.text == "retained draft"

        discard.press()
        await _wait_until(
            pilot,
            lambda: workspace.reload_confirmation_active,
            "second discard did not reopen confirmation",
        )
        workspace.query_one("#file-notes-reload-confirm", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "saved" and editor.text == "latest disk",
            "confirmed discard did not load the captured disk state",
        )
        assert not workspace.conflict_resolution_active

    replica.close()


@pytest.mark.asyncio
@pytest.mark.allow_network
@pytest.mark.parametrize("size", [(40, 20), (120, 40)])
async def test_conflict_reload_requires_keyboard_confirmation_in_library_shell(
    tmp_path: Path,
    size: tuple[int, int],
) -> None:
    """Reload keeps the draft until a distinct, keyboard-safe confirmation.

    The Windows Proactor loop used by the full production ``TldwCli`` harness
    owns a local loopback socket pair; the test does not contact an external
    service.
    """
    root = tmp_path / "notes"
    root.mkdir()
    source = root / "source.md"
    source.write_text("disk before", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        autosave_delay=10,
        poll_interval=10,
    )

    async with _production_workspace_context(workspace, size=size) as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, LibraryScreen)
        assert await workspace.open_path("source.md")
        editor = workspace.query_one("#file-notes-editor", TextArea)
        _replace_editor_text(editor, "draft to preserve")
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "dirty",
            "draft did not become dirty",
        )
        source.write_text("disk after", encoding="utf-8")
        await workspace.refresh_files()
        assert workspace.save_state == "conflict"

        await _show_maintenance_actions(workspace, pilot)
        reload_button = workspace.query_one("#file-notes-reload", Button)
        assert str(reload_button.label) == "Reload from disk"
        assert reload_button.display
        assert not reload_button.disabled
        for _ in range(120):
            if reload_button.has_focus:
                break
            await pilot.press("tab")
        assert reload_button.has_focus
        await pilot.press("enter")
        await _wait_until(
            pilot,
            lambda: workspace.reload_confirmation_active,
            "destructive reload confirmation did not open",
        )

        cancel = workspace.query_one("#file-notes-reload-cancel", Button)
        confirm = workspace.query_one("#file-notes-reload-confirm", Button)
        copy = workspace.query_one("#file-notes-reload-confirm-copy")
        assert editor.text == "draft to preserve"
        assert workspace.save_state == "conflict"
        assert _static_text(
            workspace,
            "#file-notes-reload-confirm-copy",
        ) == (
            "Discard the draft in the editor and load the current disk version? "
            "This cannot be undone."
        )
        assert cancel.has_focus
        assert str(cancel.label) == "Cancel"
        assert str(confirm.label) == "Discard draft and load disk"
        assert copy.region.right <= screen.size.width
        assert copy.region.bottom <= screen.size.height
        assert ("esc", "cancel reload") in (
            screen._library_footer_shortcuts_for_current_state()
        )

        await pilot.press("enter")
        await _wait_until(
            pilot,
            lambda: not workspace.reload_confirmation_active,
            "safe-default Cancel did not close destructive reload",
        )
        assert editor.text == "draft to preserve"
        assert workspace.save_state == "conflict"
        assert reload_button.has_focus

        await pilot.press("enter")
        await _wait_until(
            pilot,
            lambda: workspace.reload_confirmation_active,
            "second destructive reload confirmation did not open",
        )
        await pilot.press("escape")
        await _wait_until(
            pilot,
            lambda: not workspace.reload_confirmation_active,
            "Escape did not cancel destructive reload",
        )
        assert editor.text == "draft to preserve"
        assert workspace.save_state == "conflict"
        assert reload_button.has_focus

        await pilot.press("enter")
        await _wait_until(
            pilot,
            lambda: workspace.reload_confirmation_active,
            "third destructive reload confirmation did not open",
        )
        await pilot.press("tab")
        assert confirm.has_focus
        await pilot.press("enter")
        await _wait_until(
            pilot,
            lambda: (
                not workspace.reload_confirmation_active
                and workspace.save_state == "saved"
                and editor.text == "disk after"
            ),
            "confirmed reload did not intentionally replace the draft",
        )

    replica.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_axis", ["root", "file", "session"])
async def test_reload_confirmation_rejects_stale_editor_identity(
    tmp_path: Path,
    stale_axis: str,
) -> None:
    """Confirm fails closed if root, path identity, or session changes."""
    root = tmp_path / "notes"
    root.mkdir()
    source = root / "source.md"
    source.write_text("disk before", encoding="utf-8")
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
        editor = workspace.query_one("#file-notes-editor", TextArea)
        _replace_editor_text(editor, "retained draft")
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "dirty",
            "draft did not become dirty",
        )
        source.write_text("disk after", encoding="utf-8")
        await workspace.refresh_files()
        await _show_maintenance_actions(workspace, pilot)
        workspace.query_one("#file-notes-reload", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.reload_confirmation_active,
            "reload confirmation did not open",
        )
        assert not workspace.has_class("-reload-confirming")

        if stale_axis == "root":
            workspace._session_owner.select_root(tmp_path / "other-root")
        elif stale_axis == "file":
            assert workspace._opened is not None
            workspace._opened = replace(workspace._opened)
        else:
            workspace._session_key = "replacement-session"

        workspace.query_one("#file-notes-reload-confirm", Button).press()
        await _wait_until(
            pilot,
            lambda: not workspace.reload_confirmation_active,
            f"{stale_axis} identity change did not close confirmation",
        )
        assert editor.text == "retained draft"
        assert workspace.save_state == "conflict"
        status = _static_text(workspace, "#file-notes-action-status")
        assert "active root, file, or editing session changed" in status
        assert "Draft preserved" in status

    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_high_stakes_file_notes_states_keep_explicit_labels_and_classes(
    tmp_path: Path,
) -> None:
    """State classes must reinforce complete labels and clear without residue."""
    root = tmp_path / "notes"
    root.mkdir()
    (root / "state.md").write_text("body\n", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        autosave_delay=10,
        poll_interval=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("state.md")
        root_status = workspace.query_one("#file-notes-root-status")
        save_status = workspace.query_one("#file-notes-save-status")

        workspace._root_offline = True
        workspace._runtime_warning = "Recovery unavailable: replica locked"
        workspace._update_root_surface()
        workspace._set_save_state("conflict", "file changed on disk")
        await pilot.pause()

        assert _static_text(workspace, "#file-notes-root-status") == (
            "Offline · Warning · Local folder: notes"
        )
        assert root_status.has_class("-offline")
        assert root_status.has_class("-warning")
        assert _static_text(workspace, "#file-notes-save-status") == (
            "Conflict: draft preserved in editor; file changed on disk"
        )
        assert save_status.has_class("-conflict")
        assert not save_status.has_class("-error")

        workspace._set_save_state("error", "permission denied")
        assert _static_text(workspace, "#file-notes-save-status") == (
            "Save failed: draft preserved in editor; permission denied"
        )
        assert save_status.has_class("-error")
        assert not save_status.has_class("-conflict")

        workspace._root_offline = False
        workspace._runtime_warning = ""
        workspace._update_root_surface()
        workspace._set_save_state("saved")
        await pilot.pause()
        assert _static_text(workspace, "#file-notes-root-status") == (
            "Linked · Local folder: notes"
        )
        assert not root_status.has_class("-offline")
        assert not root_status.has_class("-warning")
        assert not save_status.has_class("-conflict")
        assert not save_status.has_class("-error")

    await workspace.shutdown()
    replica.close()


@pytest.mark.parametrize("size", ((120, 40), (40, 20)))
@pytest.mark.asyncio
async def test_high_stakes_file_notes_states_are_legible_in_shipped_themes(
    tmp_path: Path,
    size: tuple[int, int],
) -> None:
    """Semantic state tints must preserve painted copy and compact reachability."""
    root = tmp_path / "notes"
    root.mkdir()
    (root / "state.md").write_text("body\n", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        autosave_delay=10,
        poll_interval=10,
    )

    async with _CssTrueWorkspaceHarness(workspace).run_test(size=size) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("state.md")
        workspace._narrow_view = "editor"
        workspace._apply_responsive_layout(workspace.size.width)
        root_status = workspace.query_one("#file-notes-root-status")
        save_status = workspace.query_one("#file-notes-save-status")

        for theme_name in (
            "textual-dark",
            "textual-light",
            "high_contrast_yellow_black",
        ):
            pilot.app.theme = theme_name
            workspace._root_offline = True
            workspace._runtime_warning = "Recovery unavailable"
            workspace._update_root_surface()
            workspace._set_save_state("conflict", "file changed on disk")
            await pilot.pause()
            await pilot.pause()

            assert "Offline" in _static_text(
                workspace,
                "#file-notes-root-status",
            )
            assert "Conflict" in _static_text(
                workspace,
                "#file-notes-save-status",
            )
            assert root_status in pilot.app.screen._compositor.visible_widgets
            assert save_status in pilot.app.screen._compositor.visible_widgets
            _assert_legible_painted_text(
                pilot.app,
                root_status,
                "Offline",
                theme_name=theme_name,
                minimum_ratio=4.5,
            )
            _assert_legible_painted_text(
                pilot.app,
                save_status,
                "Conflict",
                theme_name=theme_name,
                minimum_ratio=4.5,
            )

            workspace._set_save_state("error", "permission denied")
            await pilot.pause()
            _assert_legible_painted_text(
                pilot.app,
                save_status,
                "Save failed",
                theme_name=theme_name,
                minimum_ratio=4.5,
            )
            for status in (root_status, save_status):
                assert status.region.x >= 0
                assert status.region.right <= size[0]
                assert status.region.y >= 0
                assert status.region.bottom <= size[1]
                border = status.styles.border
                assert all(
                    edge[0] in {"", "none"}
                    for edge in (
                        border.top,
                        border.right,
                        border.bottom,
                        border.left,
                    )
                )
                assert status.styles.padding.width == 0

    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("disk_change", ["changed", "missing"])
async def test_reload_confirmation_revalidates_current_disk_state(
    tmp_path: Path,
    disk_change: str,
) -> None:
    """Confirm never applies disk bytes that changed after the warning."""
    root = tmp_path / "notes"
    root.mkdir()
    source = root / "source.md"
    source.write_text("disk before", encoding="utf-8")
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
        editor = workspace.query_one("#file-notes-editor", TextArea)
        _replace_editor_text(editor, "retained draft")
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "dirty",
            "draft did not become dirty",
        )
        source.write_text("disk at prompt", encoding="utf-8")
        await workspace.refresh_files()
        await _show_maintenance_actions(workspace, pilot)
        workspace.query_one("#file-notes-reload", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.reload_confirmation_active,
            "reload confirmation did not open",
        )

        if disk_change == "changed":
            source.write_text("disk after prompt", encoding="utf-8")
        else:
            source.unlink()
        workspace.query_one("#file-notes-reload-confirm", Button).press()
        await _wait_until(
            pilot,
            lambda: not workspace.reload_confirmation_active,
            f"{disk_change} disk target did not fail closed",
        )
        assert editor.text == "retained draft"
        assert workspace.save_state == "conflict"
        status = _static_text(workspace, "#file-notes-action-status")
        expected = (
            "changed again on disk"
            if disk_change == "changed"
            else "no longer available on disk"
        )
        assert expected in status
        assert "Draft preserved" in status

    await workspace.shutdown()
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
    monkeypatch: pytest.MonkeyPatch,
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

        service = workspace._service
        assert service is not None
        delayed_open, reload_started, release_reload = _delayed_call(service.open_file)
        monkeypatch.setattr(service, "open_file", delayed_open)
        (root / "open.md").write_text("external", encoding="utf-8")
        (root / "created.md").write_text("new", encoding="utf-8")
        (root / "delete.md").unlink()
        await _wait_until(
            pilot,
            reload_started.is_set,
            "external reload did not start",
        )
        _replace_editor_text(editor, "draft during reload")
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "dirty",
            "reload-window edit did not become dirty",
        )
        release_reload.set()
        await _wait_until(
            pilot,
            lambda: (
                set(workspace.entries) == {"created.md", "folder/nested.md", "open.md"}
                and workspace.save_state == "conflict"
            ),
            "poll did not reconcile external create/modify/delete",
        )
        assert editor.text == "draft during reload"
        assert workspace.query_one("#file-notes-editor", TextArea) is editor
        refreshed_folder = next(
            node
            for node in workspace.query_one("#file-notes-tree", Tree).root.children
            if getattr(node.label, "plain", str(node.label)) == "folder"
        )
        assert refreshed_folder.is_expanded
        await pilot.pause(0.15)
        active = [
            worker
            for worker in workspace.workers
            if worker.node is workspace and not worker.is_finished
        ]
        assert len(active) <= 1

        workspace.query_one("#file-notes-back", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.navigator_visible and not workspace.editor_visible,
            "Back did not return to the retained navigator",
        )
    replica.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(40, 20), (120, 40), (160, 45)])
async def test_library_notes_source_choices_render_and_switch_by_keyboard(
    tmp_path: Path,
    size: tuple[int, int],
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "library.md").write_text("library file", encoding="utf-8")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica_path=tmp_path / "owned.sqlite",
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

    async with LibraryHarness(app, screen=screen).run_test(size=size) as pilot:
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#library-notes-source-strip")),
            "Notes source strip did not compose",
        )

        strip = screen.query_one("#library-notes-source-strip")
        separator = screen.query_one("#library-notes-source-separator")
        database = screen.query_one("#library-notes-source-database", Button)
        files = screen.query_one("#library-notes-source-files", Button)
        await _wait_until(
            pilot,
            lambda: (
                separator.region.width == 1
                and database.region.width > 0
                and files.region.width > 0
            ),
            "Notes source choices did not receive visible geometry",
        )
        assert strip.content_region.contains_region(database.region)
        assert strip.content_region.contains_region(separator.region)
        assert strip.content_region.contains_region(files.region)
        assert separator.region.width == 1
        assert str(database.label) == "Library notes"
        assert database.has_class("-selected")
        assert not database.disabled
        assert database.can_focus
        assert str(files.label) == "Folder files"
        assert not files.disabled
        assert files.can_focus

        for _ in range(60):
            if database.has_focus:
                break
            await pilot.press("tab")
        assert database.has_focus
        assert strip.content_region.contains_region(database.region)
        await pilot.press("tab")
        await _wait_until(
            pilot,
            lambda: files.has_focus,
            "Tab did not move from Database to the visible Files source",
        )
        await pilot.press("enter")
        await _wait_until(
            pilot,
            lambda: (
                screen._library_notes_source == "files"
                and workspace.initialized
                and bool(screen.query("#library-file-notes-workspace"))
            ),
            "Files source did not open from the keyboard",
        )
        shell_grid = screen.query_one("#library-shell-grid")
        rail = screen.query_one("#library-rail")
        await _wait_until(
            pilot,
            lambda: workspace.region.width > 0 and workspace.region.height > 0,
            "File Notes workspace did not receive rendered geometry",
        )
        assert shell_grid.display is True
        assert workspace.region.x >= 0
        assert workspace.region.y >= 0
        assert workspace.region.right <= screen.size.width
        assert workspace.region.bottom <= screen.size.height
        if size == (40, 20):
            assert screen._library_notes_stage == "notes"
            assert rail.display is False
            search = workspace.query_one("#file-notes-search", Input)
            for _ in range(120):
                if search.has_focus:
                    break
                await pilot.press("tab")
            assert search.has_focus
            assert search.region.right <= screen.size.width
        else:
            # TASK-19602: from 1bda754fa, a wide focused File-Notes task
            # intentionally hides the rail (the task-return control is the
            # way back); only the compact size keeps stage "rail" logic, so
            # the wide sizes assert the canvas-owns-geometry contract.
            assert rail.display is False

        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        await _wait_until(
            pilot,
            lambda: (
                screen._library_notes_source == "files"
                and bool(screen.query("#library-file-notes-workspace"))
                and bool(screen.query("#library-notes-source-strip"))
            ),
            "Reselecting Notes did not retain the file-notes workspace",
        )

        strip = screen.query_one("#library-notes-source-strip")
        database = screen.query_one("#library-notes-source-database", Button)
        files = screen.query_one("#library-notes-source-files", Button)
        assert str(database.label) == "Library notes"
        assert not database.disabled
        assert str(files.label) == "Folder files"
        assert files.has_class("-selected")
        assert not files.disabled

        if size == (40, 20):
            assert strip.content_region.contains_region(database.region)
            assert strip.content_region.contains_region(files.region)
            for _ in range(240):
                if files.has_focus:
                    break
                await pilot.press("tab")
            assert files.has_focus
            assert strip.content_region.contains_region(files.region)
            await pilot.press("shift+tab")
            await _wait_until(
                pilot,
                lambda: database.has_focus,
                "Shift+Tab did not move from Files to the visible Database source",
            )
            await pilot.press("enter")
        else:
            # TASK-19602: in wide focused-task mode the source strip's
            # buttons are hidden by design (1bda754fa) -- the keyboard way
            # back is the task-return control.
            assert database.display is False
            assert files.display is False
            task_return = screen.query_one("#library-notes-task-return", Button)
            for _ in range(240):
                if task_return.has_focus:
                    break
                await pilot.press("tab")
            assert task_return.has_focus
            await pilot.press("enter")
        await _wait_until(
            pilot,
            lambda: (
                screen._library_notes_source == "database"
                and bool(screen.query("#library-notes-canvas"))
            ),
            "Database source did not reopen from the keyboard",
        )
        assert (
            str(
                screen.query_one(
                    "#library-notes-source-database",
                    Button,
                ).label
            )
            == "Library notes"
        )

    await workspace.shutdown()


@pytest.mark.asyncio
async def test_file_notes_production_shell_preserves_canvas_across_breakpoints(
    tmp_path: Path,
) -> None:
    """Files stays visible, focused, and retained through shell breakpoints."""
    root = tmp_path / "notes"
    root.mkdir()
    (root / "library.md").write_text("library file", encoding="utf-8")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica_path=tmp_path / "owned.sqlite",
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

    async with LibraryHarness(app, screen=screen).run_test(size=(160, 45)) as pilot:
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#library-notes-source-files")),
            "Notes source strip did not compose",
        )
        screen.query_one("#library-notes-source-files", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.initialized and workspace.is_mounted,
            "File Notes workspace did not mount",
        )
        assert await workspace.open_path("library.md")
        editor = workspace.query_one("#file-notes-editor", TextArea)
        _replace_editor_text(editor, "retained file draft")
        editor.move_cursor((0, 4))
        selection = editor.selection
        screen.set_focus(editor)
        await pilot.pause()

        for width, height in ((40, 20), (120, 40), (160, 45), (40, 20)):
            await pilot.resize_terminal(width, height)
            await _wait_until(
                pilot,
                lambda: screen._library_notes_compact is (width < 120),
                f"Library compact state did not settle at {width}x{height}",
            )
            await _wait_until(
                pilot,
                lambda: workspace.region.width > 0 and workspace.region.height > 0,
                f"File Notes lost rendered geometry at {width}x{height}",
            )

            canvas = screen.query_one("#library-canvas")
            rail = screen.query_one("#library-rail")
            assert canvas.display is True
            assert workspace is screen.query_one("#library-file-notes-workspace")
            assert workspace.query_one("#file-notes-editor", TextArea) is editor
            assert workspace.current_path == "library.md"
            assert editor.text == "retained file draft"
            assert editor.selection == selection
            assert workspace.region.x >= 0
            assert workspace.region.right <= screen.size.width
            assert workspace.region.bottom <= screen.size.height
            assert screen.focused is editor, f"editor focus lost at {width}x{height}"
            assert editor.has_focus
            assert screen.focused.visible
            assert workspace._reader_work_widget in screen.focused.ancestors_with_self
            if width < 120:
                assert screen._library_notes_stage == "notes"
                assert rail.display is False
                assert (
                    screen.query_one("#library-notes-task-return", Button).display
                    is False
                )
            else:
                assert rail.display is False
                task_return = screen.query_one("#library-notes-task-return", Button)
                assert task_return.display is True
                assert str(task_return.label) == "‹ Library / Notes"

    await workspace.shutdown()


@pytest.mark.asyncio
async def test_file_notes_authority_is_painted_and_contained_at_60x20_shell(
    tmp_path: Path,
) -> None:
    """The real Library hierarchy paints the complete pinned Files authority."""
    root = tmp_path / "notes"
    root.mkdir()
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica_path=tmp_path / "owned.sqlite",
        poll_interval=10,
    )
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[{"title": "Database note", "id": "db-note-1"}],
    )
    screen = LibraryScreen(app, file_notes_workspace_factory=lambda: workspace)

    async with LibraryHarness(app, screen=screen).run_test(size=(60, 20)) as pilot:
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#library-notes-source-files")),
            "Notes source strip did not compose",
        )
        screen.query_one("#library-notes-source-files", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.initialized and workspace.is_mounted,
            "File Notes workspace did not mount",
        )
        await pilot.pause()

        authority = workspace.query_one("#file-notes-authority", Static)
        shell_grid = screen.query_one("#library-shell-grid")
        assert authority in pilot.app.screen._compositor.visible_widgets
        assert workspace.content_region.contains_region(authority.region)
        assert shell_grid.content_region.contains_region(workspace.region)
        assert 0 < authority.region.height <= 2
        assert _painted_style_of_text(pilot.app, authority.region, "Folder files")
        assert _painted_style_of_text(pilot.app, authority.region, "Next:")

    await workspace.shutdown()


@pytest.mark.parametrize(
    ("save_state", "state_copy", "next_copy"),
    (
        ("error", "Save failed", "Next: Retry/copy."),
        ("conflict", "Conflict", "Next: Resolve/copy."),
    ),
)
@pytest.mark.asyncio
async def test_file_notes_merged_recovery_authority_paints_at_60x20_shell(
    tmp_path: Path,
    save_state: str,
    state_copy: str,
    next_copy: str,
) -> None:
    """Long save failures keep all authority facts and recovery above the fold."""
    root = tmp_path / "Research notes with a very long private directory name"
    root.mkdir()
    detail = "permission denied while writing an unusually long private filesystem path"
    owner = FileNotesSessionOwner()
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica_path=tmp_path / "owned.sqlite",
        session_owner=owner,
        poll_interval=10,
    )
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[{"title": "Database note", "id": "db-note-1"}],
    )
    screen = LibraryScreen(app, file_notes_workspace_factory=lambda: workspace)

    async with LibraryHarness(app, screen=screen).run_test(size=(60, 20)) as pilot:
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#library-notes-source-files")),
            "Notes source strip did not compose",
        )
        screen.query_one("#library-notes-source-files", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.initialized and workspace.is_mounted,
            "File Notes workspace did not mount",
        )
        binding = workspace._session_binding
        assert binding is not None
        assert owner.record_change(binding, SessionChange("modified", "draft.md"))
        workspace._set_save_state(save_state, detail)
        workspace._render_session_git_label()
        await pilot.pause()

        authority = workspace.query_one("#file-notes-authority", Static)
        shell_grid = screen.query_one("#library-shell-grid")
        painted = _painted_text_in_region(pilot.app, authority.region)
        assert authority in pilot.app.screen._compositor.visible_widgets
        assert workspace.content_region.contains_region(authority.region)
        assert shell_grid.content_region.contains_region(workspace.region)
        assert authority.region.height == 2
        assert "Folder files" in painted
        assert "Folder: Rese…" in painted
        assert state_copy in painted
        assert "Session Git: 1 change" in painted
        assert next_copy in painted
        assert detail not in _static_text(workspace, "#file-notes-authority")
        assert detail in _static_text(workspace, "#file-notes-save-status")

    await workspace.shutdown()


@pytest.mark.parametrize("root_state", ("offline", "warning"))
@pytest.mark.parametrize("save_state", ("error", "conflict"))
@pytest.mark.asyncio
async def test_file_notes_combined_non_ready_authority_matrix_paints_at_60x20(
    tmp_path: Path,
    root_state: str,
    save_state: str,
) -> None:
    """Root, save, Git, push, and recovery remain painted in two rows."""
    root = tmp_path / "Research notes with a very long private directory name"
    root.mkdir()
    owner = FileNotesSessionOwner()
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica_path=tmp_path / "owned.sqlite",
        session_owner=owner,
        poll_interval=10,
    )
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[{"title": "Database note", "id": "db-note-1"}],
    )
    screen = LibraryScreen(app, file_notes_workspace_factory=lambda: workspace)

    async with LibraryHarness(app, screen=screen).run_test(size=(60, 20)) as pilot:
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#library-notes-source-files")),
            "Notes source strip did not compose",
        )
        screen.query_one("#library-notes-source-files", Button).press()
        await _wait_until(
            pilot,
            lambda: workspace.initialized and workspace.is_mounted,
            "File Notes workspace did not mount",
        )
        binding = workspace._session_binding
        assert binding is not None
        assert owner.record_change(binding, SessionChange("modified", "draft.md"))
        workspace._root_offline = root_state == "offline"
        workspace._runtime_warning = (
            "" if root_state == "offline" else "Replica recovery unavailable"
        )
        workspace._update_root_surface()
        workspace._set_save_state(
            save_state,
            "permission denied while writing a long private filesystem path",
        )

        for push_phase, push_copy in (
            ("idle", ""),
            ("checking", "Check push"),
            ("pushing", "Pushing"),
            ("needs_attention", "Push attention"),
        ):
            workspace._push_phase = push_phase
            workspace._render_session_git_label()
            await pilot.pause()

            authority = workspace.query_one("#file-notes-authority", Static)
            authority_copy = _static_text(workspace, "#file-notes-authority")
            painted = _painted_text_in_region(pilot.app, authority.region)
            assert authority.region.height == 2, (root_state, save_state, push_phase)
            assert len(authority_copy.splitlines()) == 2
            assert all(
                cell_len(row) <= authority.region.width
                for row in authority_copy.splitlines()
            )
            assert "Folder files" in painted
            assert "Folder: Rese…" in painted
            assert ("Offline" if root_state == "offline" else "Warning") in painted
            assert (
                "Conflict" if save_state == "conflict" else "Save failed"
            ) in painted
            assert "Session Git: 1" in painted
            if push_copy:
                assert push_copy in painted
            expected_next = (
                "Next: Reconnect/change."
                if root_state == "offline"
                else "Next: Open Details."
            )
            assert expected_next in painted

    await workspace.shutdown()


@pytest.mark.asyncio
async def test_poll_completion_ignores_partially_detached_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "open.md").write_text("first", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        poll_interval=10,
    )

    async with _WorkspaceHarness(workspace).run_test() as pilot:
        await _wait_until(
            pilot,
            lambda: workspace.initialized,
            "workspace scan did not finish",
        )
        service = workspace._service
        assert service is not None
        delayed_reconcile, reconcile_started, release_reconcile = _delayed_call(
            service.reconcile
        )
        monkeypatch.setattr(service, "reconcile", delayed_reconcile)
        refresh = asyncio.create_task(workspace.refresh_files())
        await _wait_until(
            pilot,
            reconcile_started.is_set,
            "workspace refresh did not start",
        )
        (root / "created.md").write_text("new", encoding="utf-8")
        try:
            await workspace.query_one("#file-notes-root-row").remove()
        finally:
            release_reconcile.set()

        assert not await refresh
        assert set(workspace.entries) == {"open.md"}
        assert "created.md" not in _tree_labels(
            workspace.query_one("#file-notes-tree", Tree)
        )
    replica.close()


@pytest.mark.asyncio
async def test_library_database_files_switch_retains_workspace_and_database_canvas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "library.md").write_text("library file", encoding="utf-8")
    (root / "other.md").write_text("other file", encoding="utf-8")
    replacement_root = tmp_path / "replacement"
    replacement_root.mkdir()
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica_path=tmp_path / "owned.sqlite",
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
    save_started = threading.Event()
    release_save = threading.Event()
    open_started = threading.Event()
    release_open = threading.Event()
    detail_started = threading.Event()
    release_detail = threading.Event()

    def delayed_detail(**_kwargs):
        detail_started.set()
        release_detail.wait(5)
        return {
            "id": "db-note-1",
            "title": "Database note",
            "content": "body",
            "version": 1,
        }

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

        app.notes_scope_service.get_note_detail = delayed_detail
        screen._selected_note_id = "db-note-1"
        screen._library_notes_view = "editor"
        detail_task = asyncio.create_task(
            screen._refresh_library_note_detail("db-note-1")
        )
        await _wait_until(
            pilot,
            detail_started.is_set,
            "Database detail fetch did not start",
        )
        screen.query_one("#library-notes-source-files", Button).press()
        await _wait_until(
            pilot,
            lambda: screen._library_notes_source == "files",
            "Files source handler did not run",
        )
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#library-file-notes-workspace")),
            "Files workspace did not mount into the canvas pane",
        )
        await _wait_until(
            pilot,
            lambda: workspace.initialized and workspace._replica is not None,
            "Files workspace did not initialize",
        )
        retained = screen.query_one(
            "#library-file-notes-workspace",
            LibraryFileNotesWorkspace,
        )
        editor = retained.query_one("#file-notes-editor", TextArea)
        owned_replica = retained._replica
        assert owned_replica is not None
        assert retained is workspace
        # task-2850: Files mode renders INSIDE the same rail + canvas frame
        # every other notes view uses -- it must never blank the rail.
        assert screen.query_one("#library-rail")

        release_detail.set()
        await detail_task
        assert screen._library_note_detail is not None
        assert screen._library_note_detail["id"] == "db-note-1"
        screen._library_notes_view = "list"
        screen._selected_note_id = None

        assert await retained.open_path("library.md")
        service = retained._service
        assert service is not None
        original_open = service.open_file

        def delayed_open(relative_path):
            if relative_path == "other.md":
                open_started.set()
                release_open.wait(5)
            return original_open(relative_path)

        monkeypatch.setattr(service, "open_file", delayed_open)
        opening = asyncio.create_task(retained.open_path("other.md"))
        await _wait_until(pilot, open_started.is_set, "slow open did not start")
        before_open = editor.text
        editor.focus()
        await pilot.press("x")
        competing_open = await retained.open_path("library.md")
        frozen_during_open = editor.read_only
        text_during_open = editor.text
        release_open.set()
        assert await opening
        monkeypatch.setattr(service, "open_file", original_open)
        assert await retained.open_path("library.md")

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

        await _show_maintenance_actions(retained, pilot)
        retained.query_one("#file-notes-reload", Button).press()
        await _wait_until(
            pilot,
            lambda: retained.reload_confirmation_active,
            "reload confirmation did not open before clearing the veto",
        )
        assert retained.save_state == "conflict"
        assert editor.text == "draft"
        retained.query_one("#file-notes-reload-confirm", Button).press()
        await _wait_until(
            pilot,
            lambda: (
                not retained.reload_confirmation_active
                and retained.save_state == "saved"
                and editor.text == "external"
            ),
            "confirmed reload did not clear the source-switch veto",
        )
        _replace_editor_text(editor, "saved before hiding")
        await _wait_until(
            pilot,
            lambda: retained.save_state == "dirty",
            "pre-remount edit did not become dirty",
        )
        assert await retained.flush_pending_work()
        assert (
            _static_text(retained, "#file-notes-session-changes")
            == "Review session changes (1)"
        )
        # TASK-19602: in wide terminals a focused File-Notes task hides the
        # source strip's Database button in favor of the task-return control
        # (both handlers route to _return_to_library_database_notes) --
        # press whichever is live at this width.
        database_button = screen.query_one("#library-notes-source-database", Button)
        if database_button.display:
            database_button.press()
        else:
            screen.query_one("#library-notes-task-return", Button).press()
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#library-notes-canvas")),
            "Database Notes did not return",
        )
        assert screen.query_one("#library-rail")
        assert screen._local_source_records["notes"][0]["title"] == "Updated DB note"

        (root / "library.md").write_text("changed while hidden", encoding="utf-8")
        await retained.refresh_files()
        await _wait_until(
            pilot,
            lambda: editor.text == "changed while hidden",
            "retained hidden workspace did not reconcile its open file",
        )
        screen.query_one("#library-notes-source-files", Button).press()
        await _wait_until(
            pilot,
            lambda: bool(
                screen.query("#library-file-notes-workspace #file-notes-editor")
            ),
            "retained Files workspace editor did not remount",
        )
        assert screen.query_one("#library-file-notes-workspace") is retained
        assert retained.query_one("#file-notes-editor", TextArea) is editor
        assert editor.text == "changed while hidden"
        assert (
            _static_text(retained, "#file-notes-session-changes")
            == "Review session changes (1)"
        )

        original_finish = service._finish_published_file

        def delayed_finish(*args, **kwargs):
            save_started.set()
            release_save.wait(5)
            return original_finish(*args, **kwargs)

        monkeypatch.setattr(service, "_finish_published_file", delayed_finish)
        _replace_editor_text(editor, "draft across retained source")
        await _wait_until(
            pilot,
            lambda: retained.save_state == "dirty",
            "retained-source draft did not become dirty",
        )
        retained._start_autosave()
        await _wait_until(
            pilot,
            lambda: save_started.is_set() and retained.save_state == "saving",
            "retained-source save did not start",
        )
        session_key = retained._session_key
        assert retained._active
        assert screen.query_one("#library-file-notes-workspace") is retained
        release_save.set()
        await _wait_until(
            pilot,
            lambda: retained.save_state == "saved",
            "published retained-source draft was not adopted",
        )
        assert (root / "library.md").read_text(encoding="utf-8") == (
            "draft across retained source"
        )
        assert editor.text == "draft across retained source"
        assert retained._session_key == session_key
        assert retained.query_one("#file-notes-editor", TextArea) is editor
        assert not competing_open
        assert frozen_during_open
        assert text_during_open == before_open
    assert workspace._shutdown
    await workspace.shutdown()
    with pytest.raises(sqlite3.ProgrammingError):
        owned_replica.list_deleted(str(root.resolve()))


@pytest.mark.asyncio
async def test_file_notes_focus_is_content_safe_under_production_css(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "focused.md").write_text("focus stays readable\n", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        poll_interval=10,
    )

    async with _production_workspace_context(workspace, size=(120, 40)) as pilot:
        assert await workspace.open_path("focused.md")

        buttons = (
            workspace.query_one("#file-notes-root-details", Button),
            workspace.query_one("#file-notes-choose-root", Button),
            workspace.query_one("#file-notes-session-changes", Button),
            *tuple(
                button
                for toolbar in workspace.query(".file-notes-toolbar")
                for button in toolbar.query(Button)
                if button.display
            ),
        )
        for button in buttons:
            button.focus()
            await pilot.pause()
            assert button.has_focus
            assert button.render_line(0).text.strip() == str(button.label)
            assert not button.styles.outline
            assert button.styles.background == Color.parse("#51677e")
            assert button.styles.text_style.bold
            assert button.styles.text_style.underline

        for selector in ("#file-notes-tree", "#file-notes-search-results"):
            tree = workspace.query_one(selector, Tree)
            tree.display = True
            tree.focus()
            await pilot.pause()
            assert tree.has_focus
            assert not tree.styles.outline
            cursor = tree.get_component_styles("tree--cursor")
            assert cursor.background == Color.parse("#51677e")
            assert cursor.text_style.bold
            assert cursor.text_style.underline

        for selector, widget_type in (
            ("#file-notes-search", Input),
            ("#file-notes-path", Input),
            ("#file-notes-editor", TextArea),
        ):
            field = workspace.query_one(selector, widget_type)
            field.focus()
            await pilot.pause()
            assert field.has_focus
            assert not field.styles.outline
            assert field.styles.border.top[0] not in {"", "none"}

    replica.close()


@pytest.mark.parametrize(
    ("editor_state", "expected_primary"),
    (
        ("normal", ("New", "Move", "Delete", "More file actions")),
        (
            "dirty",
            ("New", "Move", "Save copy", "Delete", "More file actions"),
        ),
        (
            "conflict",
            (
                "New",
                "Move",
                "Compare",
                "Resolve conflict",
                "Reload from disk",
                "Save copy",
                "Delete",
                "More file actions",
            ),
        ),
        (
            "error",
            (
                "New",
                "Move",
                "Discard draft and reload",
                "Save copy",
                "Delete",
                "More file actions",
            ),
        ),
        ("deleted", ("New", "Restore", "More file actions")),
        ("protected", ("New", "Move", "Delete", "More file actions")),
        (
            "excerpt",
            (
                "New",
                "Move",
                "Export exact copy",
                "Delete",
                "More file actions",
            ),
        ),
    ),
)
@pytest.mark.asyncio
async def test_file_notes_primary_actions_follow_editor_state(
    tmp_path: Path,
    editor_state: str,
    expected_primary: tuple[str, ...],
) -> None:
    """State-critical recovery actions outrank disclosed maintenance actions."""
    root = tmp_path / "notes"
    root.mkdir()
    (root / "state.md").write_text("saved\n", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        autosave_delay=10,
        poll_interval=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("state.md")

        if editor_state in {"dirty", "conflict", "error"}:
            workspace._set_save_state(editor_state)
        elif editor_state == "deleted":
            workspace._deleted_paths = ("state.md",)
            assert workspace.select_deleted("state.md")
        elif editor_state == "protected":
            workspace._opened = replace(workspace._opened, protected=True)
            workspace._update_controls()
        elif editor_state == "excerpt":
            workspace._opened = replace(
                workspace._opened,
                editable=False,
                is_excerpt=True,
                read_only_reason="large_file",
            )
            workspace._update_controls()

        assert _static_text(workspace, "#file-notes-path-label") == (
            "Target path · New / Move / Save copy"
        )
        assert _visible_primary_action_labels(workspace) == expected_primary
        assert not workspace.query_one("#file-notes-maintenance-actions").display
        reload_button = workspace.query_one("#file-notes-reload", Button)
        assert reload_button.parent is workspace.query_one("#file-notes-file-actions")
        if editor_state in {"dirty", "conflict", "error"}:
            assert workspace.query_one("#file-notes-save-copy", Button).display

    await workspace.shutdown()
    replica.close()


@pytest.mark.parametrize("size", ((120, 40), (40, 20)))
@pytest.mark.parametrize("save_state", ("conflict", "error"))
@pytest.mark.asyncio
async def test_reload_confirmation_keeps_target_and_save_copy_reachable(
    tmp_path: Path,
    size: tuple[int, int],
    save_state: str,
) -> None:
    """A destructive reload decision must not hide the safe copy escape hatch."""
    root = tmp_path / "notes"
    root.mkdir()
    source = root / "state.md"
    source.write_text("disk\n", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        autosave_delay=10,
        poll_interval=10,
    )

    async with _production_workspace_context(workspace, size=size) as pilot:
        assert await workspace.open_path("state.md")
        editor = workspace.query_one("#file-notes-editor", TextArea)
        _replace_editor_text(editor, "draft to preserve")
        workspace._set_save_state(save_state)
        path = workspace.query_one("#file-notes-path", Input)
        path.value = "saved-copy.md"
        reload_button = workspace.query_one("#file-notes-reload", Button)

        await workspace._reload_file(Button.Pressed(reload_button))
        await _wait_until(
            pilot,
            lambda: workspace.reload_confirmation_active,
            "reload confirmation did not open",
        )
        cancel = workspace.query_one("#file-notes-reload-cancel", Button)
        confirm = workspace.query_one("#file-notes-reload-confirm", Button)
        await _wait_until(
            pilot,
            lambda: cancel.has_focus,
            "reload confirmation did not focus safe Cancel",
        )
        assert str(confirm.label) == "Discard draft and load disk"
        assert confirm.display and not confirm.disabled
        assert source.read_text(encoding="utf-8") == "disk\n"
        assert editor.text == "draft to preserve"

        path_label = workspace.query_one("#file-notes-path-label", Static)
        save_copy = workspace.query_one("#file-notes-save-copy", Button)
        assert path_label.display and path.display
        assert _static_text(workspace, "#file-notes-path-label") == (
            "Target path · New / Move / Save copy"
        )
        assert save_copy.display and not save_copy.disabled
        workspace.query_one("#file-notes-save-status").scroll_visible(
            animate=False,
            top=True,
        )
        await pilot.pause()
        path.focus(scroll_visible=False)
        await pilot.pause()
        assert path.has_focus
        assert (
            workspace.query_one("#file-notes-save-status").region.bottom
            <= path_label.region.y
        )
        for control, copy in (
            (path_label, "Target path · New / Move / Save copy"),
            (path, "saved-copy.md"),
        ):
            assert workspace.region.contains_region(control.region)
            painted = _painted_text_in_region(pilot.app, control.region)
            if control is path_label and size[0] < 80:
                assert painted.startswith("Target path · New / Move /")
            else:
                assert copy in painted

        save_copy.focus()
        save_copy.scroll_visible(animate=False)
        await pilot.pause()
        assert save_copy.has_focus
        assert workspace.region.contains_region(save_copy.region)
        assert "Save copy" in _painted_text_in_region(
            pilot.app,
            save_copy.region,
        )

        save_copy.press()
        await _wait_until(
            pilot,
            lambda: (root / "saved-copy.md").exists(),
            "Save copy did not preserve the draft during reload confirmation",
        )
        await _wait_until(
            pilot,
            lambda: not workspace.reload_confirmation_active,
            "Save copy did not dismiss the reload confirmation",
        )
        assert (root / "saved-copy.md").read_text(encoding="utf-8") == (
            "draft to preserve\n"
        )
        assert source.read_text(encoding="utf-8") == "disk\n"
        assert not workspace.reload_confirmation_active

    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_file_notes_discloses_actions_by_editor_state_and_redirects_focus(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "state.md").write_text("saved\n", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        autosave_delay=10,
        poll_interval=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert _visible_editor_action_ids(workspace) == {
            "file-notes-new",
            "file-notes-maintenance-toggle",
        }

        assert await workspace.open_path("state.md")
        assert _visible_primary_action_labels(workspace) == (
            "New",
            "Move",
            "Delete",
            "More file actions",
        )

        delete = workspace.query_one("#file-notes-delete", Button)
        editor = workspace.query_one("#file-notes-editor", TextArea)
        delete.focus()
        await pilot.pause()
        assert delete.has_focus
        with workspace._hold_path_transition() as transition:
            assert transition is not None
            editor.focus()
            await pilot.pause()
            assert editor.has_focus
        await pilot.pause()
        assert editor.has_focus

        _replace_editor_text(editor, "dirty recovery")
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "dirty",
            "draft did not disclose recovery action",
        )
        assert _visible_editor_action_ids(workspace) == {
            "file-notes-new",
            "file-notes-move",
            "file-notes-delete",
            "file-notes-save-copy",
            "file-notes-maintenance-toggle",
        }

        assert await workspace.flush_pending_work()
        delete.focus()
        delete.press()
        await _wait_until(
            pilot,
            lambda: str(delete.label) == "Confirm delete",
            "delete confirmation did not arm",
        )
        assert _static_text(workspace, "#file-notes-action-status") == (
            "Activate Delete again to confirm."
        )
        assert delete.display and delete.has_focus
        delete.press()
        restore = workspace.query_one("#file-notes-restore", Button)
        await _wait_until(
            pilot,
            lambda: restore.display and workspace._selected_deleted_path == "state.md",
            "delete did not project tombstone actions",
        )
        assert _visible_editor_action_ids(workspace) == {
            "file-notes-new",
            "file-notes-restore",
            "file-notes-maintenance-toggle",
        }
        await _wait_until(
            pilot,
            lambda: workspace.app.focused is not None,
            "focus did not redirect after delete",
        )
        focused = workspace.app.focused
        assert focused is not None
        assert not delete.has_focus
        assert focused.display

        with workspace._hold_path_transition() as transition:
            assert transition is not None
            visible_buttons = tuple(
                button
                for toolbar in workspace.query(".file-notes-toolbar")
                for button in toolbar.query(Button)
                if button.display
            )
            assert visible_buttons
            assert all(button.disabled for button in visible_buttons)
            assert (
                "temporarily unavailable"
                in _static_text(
                    workspace,
                    "#file-notes-action-status",
                ).lower()
            )

    replica.close()


@pytest.mark.parametrize("size", ((120, 40), (40, 20)))
@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_file_notes_delete_is_spatially_separated_from_new(
    tmp_path: Path,
    size: tuple[int, int],
) -> None:
    """Verify Delete stays away from the routine New action.

    Args:
        tmp_path: Temporary directory used as the File Notes root.
        size: Terminal width and height used to exercise the responsive layout.
    """
    root = tmp_path / "notes"
    root.mkdir()
    (root / "safety.md").write_text("keep me\n", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        autosave_delay=10,
        poll_interval=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=size) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("safety.md")
        await pilot.pause()
        await pilot.pause()

        toolbar = workspace.query_one("#file-notes-file-actions")
        new = workspace.query_one("#file-notes-new", Button)
        delete = workspace.query_one("#file-notes-delete", Button)
        spacer = workspace.query_one("#file-notes-delete-spacer", Static)
        button_ids = [button.id for button in toolbar.query(Button) if button.display]

        assert button_ids == [
            "file-notes-new",
            "file-notes-move",
            "file-notes-delete",
            "file-notes-maintenance-toggle",
        ]
        assert new.display and delete.display
        assert new.render_line(0).text.strip() == "New"
        assert delete.render_line(0).text.strip() == "Delete"
        if size[0] == 120:
            assert not workspace.has_class("-stack-editor-actions")
            assert spacer.display
            assert delete.region.x > new.region.right
            assert delete.region.right <= toolbar.content_region.right
        else:
            assert workspace.has_class("-stack-editor-actions")
            assert not spacer.display
            assert delete.region.y > new.region.y
            assert delete.region.width == toolbar.content_region.width

    replica.close()


@pytest.mark.asyncio
async def test_disabled_file_notes_actions_carry_marker_and_visible_recovery(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "state.md").write_text("saved\n", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        autosave_delay=10,
        poll_interval=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("state.md")

        with workspace._hold_path_transition() as transition:
            assert transition is not None
            await pilot.pause()
            new_button = workspace.query_one("#file-notes-new", Button)
            assert new_button.disabled and new_button.display
            assert str(new_button.label).startswith(
                f"{LIBRARY_DISABLED_ACTION_MARKER} "
            )
            busy_reason = _static_text(workspace, "#file-notes-action-status")
            assert "temporarily unavailable" in busy_reason.lower()
            assert "wait" in busy_reason.lower()

        panel, check_remote = _show_disabled_git_result(workspace)
        await pilot.pause()
        assert panel.display
        assert check_remote.display and check_remote.disabled
        assert str(check_remote.label).startswith(f"{LIBRARY_DISABLED_ACTION_MARKER} ")
        reason = panel.query_one("#file-notes-git-push-result-reason")
        assert reason.display
        assert "Restore network access" in str(reason.renderable)
        assert "Check remote again" in str(reason.renderable)

    replica.close()


@pytest.mark.asyncio
async def test_disabled_file_notes_actions_meet_contrast_in_every_shipped_theme(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "contrast.md").write_text("saved\n", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        autosave_delay=10,
        poll_interval=10,
    )
    theme_names = tuple(
        dict.fromkeys(
            ("textual-dark", "textual-light", *(theme.name for theme in ALL_THEMES))
        )
    )

    async with _production_workspace_context(workspace, size=(120, 40)) as pilot:
        assert await workspace.open_path("contrast.md")
        with workspace._hold_path_transition() as transition:
            assert transition is not None
            panel, git_button = _show_disabled_git_result(workspace)
            await pilot.pause()
            workspace_button = workspace.query_one("#file-notes-new", Button)
            workspace_reason = workspace.query_one("#file-notes-action-status")
            git_reason = panel.query_one("#file-notes-git-push-result-reason")

            for theme_name in theme_names:
                pilot.app.theme = theme_name
                workspace._navigator_mode = "files"
                workspace._narrow_view = "editor"
                workspace._apply_responsive_layout(workspace.size.width)
                await pilot.pause()
                await pilot.pause()
                assert workspace_button.disabled and workspace_button.display
                assert workspace_button.styles.opacity == 1.0
                workspace_label = str(workspace_button.label)
                assert workspace_label.startswith(f"{LIBRARY_DISABLED_ACTION_MARKER} ")
                _assert_legible_painted_text(
                    pilot.app,
                    workspace_button,
                    workspace_label,
                    theme_name=theme_name,
                )
                _assert_legible_painted_text(
                    pilot.app,
                    workspace_reason,
                    "File operation in progress",
                    theme_name=theme_name,
                )

                workspace._navigator_mode = "git"
                workspace._narrow_view = "navigator"
                workspace._apply_responsive_layout(workspace.size.width)
                await pilot.pause()
                assert git_button.disabled and git_button.display
                assert git_button.styles.opacity == 1.0
                git_label = str(git_button.label)
                assert git_label.startswith(f"{LIBRARY_DISABLED_ACTION_MARKER} ")
                _assert_legible_painted_text(
                    pilot.app,
                    git_button,
                    f"{LIBRARY_DISABLED_ACTION_MARKER} Check",
                    theme_name=theme_name,
                )
                _assert_legible_painted_text(
                    pilot.app,
                    git_reason,
                    "Restore network access",
                    theme_name=theme_name,
                )

    replica.close()


@pytest.mark.asyncio
async def test_disabled_file_notes_actions_keep_legibility_at_40_columns(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "compact.md").write_text("saved\n", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        autosave_delay=10,
        poll_interval=10,
    )

    async with _CssTrueWorkspaceHarness(workspace).run_test(size=(40, 20)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("compact.md")
        with workspace._hold_path_transition() as transition:
            assert transition is not None
            workspace._narrow_view = "editor"
            workspace._apply_responsive_layout(workspace.size.width)
            await pilot.pause()
            workspace_button = workspace.query_one("#file-notes-new", Button)
            workspace_reason = workspace.query_one("#file-notes-action-status")

            for theme_name in (
                "textual-dark",
                "textual-light",
                "high_contrast_yellow_black",
            ):
                pilot.app.theme = theme_name
                await pilot.pause()
                label = str(workspace_button.label)
                _assert_legible_painted_text(
                    pilot.app,
                    workspace_button,
                    label,
                    theme_name=theme_name,
                )
                _assert_legible_painted_text(
                    pilot.app,
                    workspace_reason,
                    "File operation in progress",
                    theme_name=theme_name,
                )

            panel, git_button = _show_disabled_git_result(workspace)
            git_reason = panel.query_one("#file-notes-git-push-result-reason")
            git_reason.scroll_visible(animate=False)
            await pilot.pause()
            for theme_name in (
                "textual-dark",
                "textual-light",
                "high_contrast_yellow_black",
            ):
                pilot.app.theme = theme_name
                await pilot.pause()
                label = str(git_button.label)
                _assert_legible_painted_text(
                    pilot.app,
                    git_button,
                    f"{LIBRARY_DISABLED_ACTION_MARKER} Check",
                    theme_name=theme_name,
                )
                _assert_legible_painted_text(
                    pilot.app,
                    git_reason,
                    "Restore network access",
                    theme_name=theme_name,
                )

    replica.close()


@pytest.mark.parametrize("size", ((160, 45), (120, 40), (64, 28), (40, 20)))
@pytest.mark.asyncio
async def test_file_notes_disclosed_actions_fit_wide_and_compact_layouts(
    tmp_path: Path,
    size: tuple[int, int],
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "layout.md").write_text("layout\n", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        autosave_delay=10,
        poll_interval=10,
    )

    async with _WorkspaceHarness(workspace).run_test(size=size) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("layout.md")
        _replace_editor_text(
            workspace.query_one("#file-notes-editor", TextArea),
            "recovery layout",
        )
        await _wait_until(
            pilot,
            lambda: workspace.save_state == "dirty",
            "recovery action did not appear",
        )
        await pilot.pause()
        assert workspace.editor_visible
        _assert_visible_editor_actions_fit(workspace)

        workspace._set_delete_confirmation("layout.md")
        await pilot.pause()
        _assert_visible_editor_actions_fit(workspace)

    replica.close()


@pytest.mark.parametrize("size", ((120, 40), (40, 20)))
@pytest.mark.asyncio
async def test_production_folder_files_path_and_recovery_actions_are_painted(
    tmp_path: Path,
    size: tuple[int, int],
) -> None:
    """The shipped Library hierarchy keeps compact recovery controls reachable."""
    root = tmp_path / "notes"
    root.mkdir()
    (root / "layout.md").write_text("layout\n", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        autosave_delay=10,
        poll_interval=10,
    )

    async with _production_workspace_context(workspace, size=size) as pilot:
        assert await workspace.open_path("layout.md")
        workspace._set_save_state("conflict", "file changed on disk")
        await pilot.pause()

        path_label = workspace.query_one("#file-notes-path-label", Static)
        path = workspace.query_one("#file-notes-path", Input)
        path.focus()
        path.scroll_visible(animate=False)
        await pilot.pause()
        assert path.has_focus
        assert path_label.region.bottom <= path.region.y
        assert workspace.region.contains_region(path_label.region)
        assert workspace.region.contains_region(path.region)
        painted_label = _painted_text_in_region(pilot.app, path_label.region)
        if size[0] >= 80:
            assert "Target path · New / Move / Save copy" in painted_label
        else:
            assert painted_label.startswith("Target path · New / Move /")

        for selector, expected_label in (
            ("#file-notes-move", "Move"),
            ("#file-notes-reload", "Reload from disk"),
            ("#file-notes-save-copy", "Save copy"),
            ("#file-notes-delete", "Delete"),
            ("#file-notes-maintenance-toggle", "More file actions"),
        ):
            button = workspace.query_one(selector, Button)
            assert button.display and not button.disabled
            button.scroll_visible(animate=False)
            await pilot.pause()
            button.focus()
            await pilot.pause()
            assert button.has_focus
            assert workspace.region.contains_region(button.region), (
                selector,
                button.region,
                workspace.region,
                workspace.classes,
                button.parent.region if button.parent is not None else None,
            )
            assert expected_label in _painted_text_in_region(
                pilot.app,
                button.region,
            )

    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_production_compact_folder_files_disclosure_and_states_are_painted(
    tmp_path: Path,
) -> None:
    """The 40-column product path paints disclosed and contextual actions."""
    root = tmp_path / "notes"
    root.mkdir()
    (root / "states.md").write_text("states\n", encoding="utf-8")
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        autosave_delay=10,
        poll_interval=10,
    )

    async with _production_workspace_context(workspace, size=(40, 20)) as pilot:
        assert await workspace.open_path("states.md")
        opened = workspace.current_document
        assert opened is not None

        toggle = workspace.query_one("#file-notes-maintenance-toggle", Button)
        toggle.focus()
        await pilot.press("enter")
        await _wait_until(
            pilot,
            lambda: workspace.query_one("#file-notes-maintenance-actions").display,
            "More file actions did not disclose compact maintenance actions",
        )
        for selector, label in (
            ("#file-notes-reload", "Reload"),
            ("#file-notes-protect", "Protect"),
            ("#file-notes-refresh", "Refresh"),
        ):
            action = workspace.query_one(selector, Button)
            assert action.display and not action.disabled
            await pilot.press("tab")
            await pilot.pause()
            assert action.has_focus
            assert workspace.region.contains_region(action.region)
            assert label in _painted_text_in_region(pilot.app, action.region)

        workspace._opened = replace(opened, protected=True)
        workspace._update_controls()
        unprotect = workspace.query_one("#file-notes-protect", Button)
        unprotect.scroll_visible(animate=False)
        await pilot.pause()
        unprotect.focus()
        await pilot.pause()
        assert "Unprotect" in _painted_text_in_region(
            pilot.app,
            unprotect.region,
        )

        workspace._maintenance_expanded = False
        workspace._opened = replace(
            opened,
            editable=False,
            is_excerpt=True,
            read_only_reason="large_file",
        )
        workspace._update_controls()
        export = workspace.query_one("#file-notes-save-copy", Button)
        export.scroll_visible(animate=False)
        await pilot.pause()
        export.focus()
        await pilot.pause()
        assert export.display and not export.disabled and export.has_focus
        assert "Export exact copy" in _painted_text_in_region(
            pilot.app,
            export.region,
        )

        workspace._opened = opened
        workspace._set_save_state("conflict")
        resolve = workspace.query_one("#file-notes-resolve-conflict", Button)
        resolve.press()
        await _wait_until(
            pilot,
            lambda: workspace.conflict_resolution_active,
            "compact conflict resolution did not open",
        )
        for selector, label in (
            ("#file-notes-resolution-keep", "Keep editing"),
            ("#file-notes-resolution-save-new", "Save draft as new note"),
            ("#file-notes-resolution-discard", "Discard draft and load disk"),
        ):
            action = workspace.query_one(selector, Button)
            action.scroll_visible(animate=False)
            await pilot.pause()
            action.focus()
            await pilot.pause()
            assert action.has_focus
            assert workspace.region.contains_region(action.region)
            assert str(action.label) == label
            assert action.render().plain == label
            painted_label = _painted_text_in_region(
                pilot.app,
                action.region,
            ).strip()
            assert painted_label
            assert label.startswith(painted_label)

        workspace._set_conflict_resolution(False)
        workspace._deleted_paths = ("states.md",)
        assert workspace.select_deleted("states.md")
        restore = workspace.query_one("#file-notes-restore", Button)
        restore.scroll_visible(animate=False)
        await pilot.pause()
        restore.focus()
        await pilot.pause()
        assert restore.display and not restore.disabled and restore.has_focus
        assert workspace.region.contains_region(restore.region)
        assert "Restore" in _painted_text_in_region(pilot.app, restore.region)

    await workspace.shutdown()
    replica.close()


@pytest.mark.asyncio
async def test_file_notes_large_navigators_publish_bounded_keyboard_pages(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    workspace = LibraryFileNotesWorkspace(root=root, replica=None, poll_interval=10)
    sibling_entries = tuple(
        FileNoteEntry(
            relative_path=f"note-{index:04d}.md",
            size=1,
            mtime_ns=index,
            content_hash=str(index),
            editable=True,
        )
        for index in range(5_000)
    )

    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        workspace._adopt_scan_state(ScanResult("ok", sibling_entries), ())
        started = perf_counter()
        workspace._rebuild_tree()
        publication_seconds = perf_counter() - started
        tree = workspace.query_one("#file-notes-tree", Tree)

        assert publication_seconds < 0.1
        assert _tree_node_count(tree) <= workspace_module.FILE_TREE_BATCH_SIZE + 2
        await pilot.pause()
        load_more = next(
            node
            for node in tree.root.children
            if getattr(node.label, "plain", str(node.label)).startswith("Load more")
        )
        first_count = len(tree.root.children)
        tree.focus()
        tree.move_cursor(load_more)
        await pilot.pause()
        assert tree.cursor_node is load_more
        await pilot.press("enter")
        await _wait_until(
            pilot,
            lambda: len(tree.root.children) > first_count,
            "keyboard Load more did not append the next bounded batch",
        )
        assert (
            len(tree.root.children) <= (2 * workspace_module.FILE_TREE_BATCH_SIZE) + 1
        )
        assert tree.has_focus

        search_paths = tuple(f"result-{index:04d}.md" for index in range(5_000))
        started = perf_counter()
        workspace._rebuild_search_results(search_paths)
        search_seconds = perf_counter() - started
        search = workspace.query_one("#file-notes-search-results", Tree)
        assert search_seconds < 0.1
        assert _tree_node_count(search) <= workspace_module.FILE_TREE_BATCH_SIZE + 2


@pytest.mark.asyncio
async def test_file_notes_deep_tree_materializes_only_expanded_levels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    workspace = LibraryFileNotesWorkspace(root=root, replica=None, poll_interval=10)
    deep_path = "/".join(f"folder-{index:04d}" for index in range(500))
    entry = FileNoteEntry(
        relative_path=f"{deep_path}/note.md",
        size=1,
        mtime_ns=1,
        content_hash="hash",
        editable=True,
    )

    async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        started = perf_counter()
        assert workspace._apply_scan(ScanResult("ok", (entry,)), ())
        publication_seconds = perf_counter() - started
        tree = workspace.query_one("#file-notes-tree", Tree)
        assert publication_seconds < 0.1
        assert _tree_node_count(tree) == 2

        first_folder = tree.root.children[0]
        first_folder.expand()
        await _wait_until(
            pilot,
            lambda: bool(first_folder.children),
            "expanding a folder did not materialize its next level",
        )
        assert _tree_node_count(tree) == 3


@pytest.mark.parametrize("size", ((120, 40), (40, 20)))
@pytest.mark.asyncio
async def test_large_file_uses_labeled_excerpt_and_exports_exact_disk_bytes(
    tmp_path: Path,
    size: tuple[int, int],
) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    content = "x" * (INTERACTIVE_FILE_CHARS + 1)
    source = root / "large.md"
    source.write_text(content, encoding="utf-8")
    workspace = LibraryFileNotesWorkspace(root=root, replica=None, poll_interval=10)

    async with _WorkspaceHarness(workspace).run_test(size=size) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path(source.name)
        await pilot.pause()

        opened = workspace.current_document
        assert opened is not None and opened.is_excerpt
        editor = workspace.query_one("#file-notes-editor", TextArea)
        preview = workspace.query_one("#file-notes-preview-status", Static)
        export = workspace.query_one("#file-notes-save-copy", Button)
        assert editor.read_only
        assert editor.text == content[:LARGE_FILE_EXCERPT_CHARS]
        assert len(editor.text) == LARGE_FILE_EXCERPT_CHARS
        assert preview.display
        preview_copy = str(preview.renderable)
        assert "Read-only excerpt: first 100,000" in preview_copy
        assert f"{len(content):,} characters" in preview_copy
        assert f"{len(content.encode()):,} bytes" in preview_copy
        assert preview.region.right <= workspace.region.right
        assert str(export.label) == "Export exact copy"
        assert export.display and not export.disabled

        workspace.query_one("#file-notes-path", Input).value = "exact-copy.md"
        export.press()
        await _wait_until(
            pilot,
            lambda: (root / "exact-copy.md").exists(),
            "exact export did not create the destination",
        )
        assert (root / "exact-copy.md").read_bytes() == source.read_bytes()


class _PollCoverModal(ModalScreen[None]):
    """Modal cover for the TASK-22219 visibility-gate probes."""


class _PollCoverScreen(Screen[None]):
    """Plain pushed-screen cover for the TASK-22219 visibility-gate probes."""


@pytest.mark.asyncio
@pytest.mark.parametrize("cover_factory", (_PollCoverModal, _PollCoverScreen))
async def test_poll_reconcile_pauses_while_covered_and_resumes_on_return(
    tmp_path: Path,
    cover_factory: type,
) -> None:
    """TASK-22219: the poll must not walk the notes root while covered.

    Covers both cover shapes the live app stacks over Library: a pushed
    ``ModalScreen`` (dialogs, command palette) and a pushed plain ``Screen``
    (help panels). Textual 8.2.8's ``Screen.is_active`` is
    ``app.screen is self`` -- true only for the top of the stack -- so both
    covers must gate the timer fire, and popping the cover must let the very
    next tick reconcile again (the resume path IS the still-ticking timer;
    nothing needs re-arming).
    """
    root = tmp_path / "notes"
    root.mkdir()
    (root / "open.md").write_text("body", encoding="utf-8")
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=None,
        poll_interval=0.05,
    )

    reconcile_times: list[float] = []
    real_reconcile = FileNotesService.reconcile

    def counting_reconcile(service: FileNotesService, *args, **kwargs):
        reconcile_times.append(perf_counter())
        return real_reconcile(service, *args, **kwargs)

    def poll_settled() -> bool:
        worker = workspace._poll_worker
        return worker is None or worker.is_finished

    with patch.object(FileNotesService, "reconcile", counting_reconcile):
        async with _WorkspaceHarness(workspace).run_test(size=(120, 40)) as pilot:
            await _wait_until(
                pilot,
                lambda: workspace.initialized,
                "scan did not finish",
            )
            visible_baseline = len(reconcile_times)
            await _wait_until(
                pilot,
                lambda: len(reconcile_times) > visible_baseline,
                "poll did not reconcile while the screen was active",
            )

            pilot.app.push_screen(cover_factory())
            await pilot.pause()
            assert not workspace.screen.is_active
            # Let a fire admitted before the push settle, so the covered
            # window below counts only timer fires made while covered.
            await _wait_until(
                pilot,
                poll_settled,
                "in-flight poll never settled after the cover was pushed",
            )
            await pilot.pause(0.1)

            covered_baseline = len(reconcile_times)
            await pilot.pause(0.6)  # 12 poll intervals under the cover
            covered_fires = len(reconcile_times) - covered_baseline
            assert covered_fires == 0, (
                f"reconcile fired {covered_fires}x while covered by "
                f"{cover_factory.__name__}"
            )

            resume_baseline = len(reconcile_times)
            pilot.app.pop_screen()
            resumed_at = perf_counter()
            await pilot.pause()
            assert workspace.screen.is_active
            await _wait_until(
                pilot,
                lambda: len(reconcile_times) > resume_baseline,
                "poll did not resume after the cover was popped",
            )
            resume_delay = reconcile_times[resume_baseline] - resumed_at
            # The contract is "the next tick reconciles" (one 0.05s interval);
            # the bound is generous only for CI scheduling slack.
            assert resume_delay < 2.0, f"resume took {resume_delay:.3f}s"
    await workspace.shutdown()
