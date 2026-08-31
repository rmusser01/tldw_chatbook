"""Behavior contracts for the read-only Console Workspace Files modal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.Console.console_workspace_files_modal import (
    ConsoleWorkspaceFilesModal,
    WorkspaceFilesBinding,
)
from tldw_chatbook.Workspaces.file_inspector import (
    BindingScope,
    DirectoryEntry,
    DirectoryPage,
    DirectoryStatus,
    FileReadKind,
    FileReadResult,
    FilterResult,
    FilterStatus,
)


@dataclass
class _Inspector:
    """A real modal boundary fake; it records only public service operations."""

    calls: list[tuple[str, object]]

    def list_directory(self, scope: BindingScope, directory_parts=(), *, continuation=None):
        self.calls.append(("list", (scope, directory_parts, continuation)))
        return DirectoryPage(
            DirectoryStatus.COMPLETE,
            (DirectoryEntry(("unsafe[bold]\\n",), "unsafe[bold]\\n", False),),
        )

    def filter_paths(self, scope: BindingScope, query: str, *, is_cancelled=None):
        self.calls.append(("filter", (scope, query)))
        return FilterResult(FilterStatus.EMPTY, status_copy="No matching paths.")

    def read_file(self, scope: BindingScope, raw_parts, *, page_offset=None, expected_revision=None):
        self.calls.append(("read", (scope, raw_parts, page_offset, expected_revision)))
        return FileReadResult(FileReadKind.TEXT, text="safe\\npreview")


class _Host(App[None]):
    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Button("Open files", id="files-opener")


def _scope() -> BindingScope:
    return BindingScope("ws-a", "binding-a", "fingerprint", "/not-read", 1, 1)


@pytest.mark.asyncio
async def test_modal_shows_pinned_read_only_identity_and_loads_its_only_binding() -> None:
    """Catch a modal that hides the inspected scope or does not begin a safe root listing."""
    inspector = _Inspector([])
    modal = ConsoleWorkspaceFilesModal(
        inspector=inspector,
        inspected_workspace_id="ws-a",
        inspected_workspace_name="Workspace A",
        active_workspace_id="ws-b",
        active_workspace_name="Workspace B",
        bindings=(WorkspaceFilesBinding("binding-a", "Project", _scope()),),
    )
    app = _Host()

    async with app.run_test(size=(120, 40)) as pilot:
        opener = app.query_one("#files-opener", Button)
        opener.focus()
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()

        assert str(modal.query(".console-modal-header").first(Static).renderable) == "Workspace Files"
        assert str(modal.query_one("#console-workspace-files-pinned", Static).renderable) == "Inspector only · Console remains Workspace B"
        assert str(modal.query_one("#console-workspace-files-contract", Static).renderable) == "Viewing Workspace A · Read-only access"
        assert inspector.calls and inspector.calls[0][0] == "list"


@pytest.mark.asyncio
async def test_unavailable_binding_is_selected_without_falling_back_to_another_scope() -> None:
    """Catch a stale binding click that silently keeps a previously valid folder."""
    inspector = _Inspector([])
    modal = ConsoleWorkspaceFilesModal(
        inspector=inspector,
        inspected_workspace_id="ws-a",
        inspected_workspace_name="Workspace A",
        active_workspace_id="ws-a",
        active_workspace_name="Workspace A",
        bindings=(
            WorkspaceFilesBinding("binding-a", "Available", _scope()),
            WorkspaceFilesBinding("binding-b", "Changed", None, available=False, availability_copy="Unavailable"),
        ),
    )
    app = _Host()

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.click("#console-workspace-files-binding-1")
        await pilot.pause()

        assert modal.state.selected_binding_id == "binding-b"
        assert modal.state.status_copy == "Selected binding is unavailable."
        assert [name for name, _call in inspector.calls] == ["list"]


def test_modal_declares_the_shared_safe_dismissal_boundary() -> None:
    """Catch a future modal change that bypasses the Console safe-modal contract."""
    assert ConsoleWorkspaceFilesModal.SAFE_MODAL_CONTENT == "#console-workspace-files-modal"
    assert ConsoleWorkspaceFilesModal.BINDINGS[0].action == "request_safe_cancel"
