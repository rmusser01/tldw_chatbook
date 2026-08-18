"""Shared workspace creation modal (spec 2026-08-17 §4).

Used by the Console rail, Settings ▸ Workspaces, and Library. The modal
owns the create/bind service calls so failures render inline; surfaces
own post-create UI sync via their dismissal callbacks (§4.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Static

from tldw_chatbook.Third_Party.textual_fspicker import SelectDirectory
from tldw_chatbook.Workspaces.registry_service import (
    LocalWorkspaceRegistryService,
    WorkspaceRegistryServiceError,
    next_local_workspace_identity,
    validate_folder_binding_path,
)

_WORKSPACE_EXPLAINER = (
    "A workspace scopes the Console to one project. Conversations started "
    "in it are grouped together, agents' project file access comes only "
    "from the folders you bind here (read-only unless you grant write in "
    "Settings), and retrieval can be narrowed to the workspace's items via "
    "its RAG Scope. Binding your project's folder is what makes a "
    "workspace more than a label — without one, agents have no file "
    "access. You can add or change folders later in Settings ▸ Workspaces."
)


@dataclass(frozen=True)
class WorkspaceCreateResult:
    """Outcome of a completed create dialog (spec §4.3)."""

    workspace_id: str
    name: str
    bound_folders: tuple[str, ...] = ()
    failed_folders: tuple[tuple[str, str], ...] = ()  # (path, error message)
    make_active: bool = True
    #: ProjectSkillsDiscovery entries for bound folders containing .SKILLS/.
    #: Stays empty until project-skills discovery ships (spec §5.5 / PR B).
    project_skills: tuple = ()


class WorkspaceCreateModal(ModalScreen["WorkspaceCreateResult | None"]):
    """Collect a workspace name + optional folder bindings."""

    DEFAULT_CSS = """
    WorkspaceCreateModal {
        align: center middle;
    }

    #workspace-create-modal {
        width: 72;
        height: auto;
        max-height: 32;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #workspace-create-explainer {
        color: $text-muted;
        margin: 0 0 1 0;
    }

    #workspace-create-error {
        color: $error;
        height: auto;
    }

    #workspace-create-folder-row {
        height: 3;
        min-height: 3;
    }

    #workspace-create-folder-path {
        width: 1fr;
    }

    #workspace-create-actions {
        height: 3;
        min-height: 3;
        margin: 1 0 0 0;
        align-horizontal: right;
    }
    """

    BINDINGS = [("escape", "dismiss", "Cancel")]
    AUTO_FOCUS = "#workspace-create-name"

    def __init__(self, *, registry_service: LocalWorkspaceRegistryService) -> None:
        super().__init__()
        self._registry = registry_service
        self._folders: list[str] = []
        self._error = ""

    def compose(self) -> ComposeResult:
        _, suggested_name = next_local_workspace_identity(self._registry)
        with Vertical(id="workspace-create-modal"):
            yield Static("New Workspace", classes="console-modal-header")
            yield Static(
                _WORKSPACE_EXPLAINER, id="workspace-create-explainer", markup=False
            )
            yield Input(
                value=suggested_name,
                id="workspace-create-name",
                placeholder="Workspace name",
            )
            with Horizontal(id="workspace-create-folder-row"):
                yield Input(
                    id="workspace-create-folder-path",
                    placeholder="~/path/to/project (optional)",
                )
                yield Button("Browse…", id="workspace-create-browse", compact=True)
            with Horizontal(id="workspace-create-actions"):
                yield Button("Cancel", id="workspace-create-cancel", compact=True)

    def action_dismiss(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#workspace-create-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#workspace-create-browse")
    def _browse(self, event: Button.Pressed) -> None:
        event.stop()

        def _picked(selected: Path | None) -> None:
            if selected is not None:
                self.query_one(
                    "#workspace-create-folder-path", Input
                ).value = str(selected)

        self.app.push_screen(
            SelectDirectory(title="Bind Project Folder"), _picked
        )
