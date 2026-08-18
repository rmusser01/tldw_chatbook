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
        height: auto;
    }

    #workspace-create-folder-path {
        width: 1fr;
    }

    #workspace-create-actions {
        height: auto;
        margin: 1 0 0 0;
        align-horizontal: right;
    }

    #workspace-create-folder-list {
        height: auto;
    }

    .workspace-create-folder-item {
        height: auto;
    }

    .workspace-create-folder-locator {
        width: 1fr;
        content-align: left middle;
    }
    """

    BINDINGS = [("escape", "dismiss", "Cancel")]
    AUTO_FOCUS = "#workspace-create-name"

    def __init__(self, *, registry_service: LocalWorkspaceRegistryService) -> None:
        super().__init__()
        self._registry = registry_service
        self._folders: list[str] = []
        self._error = ""
        # Captured once so recompose()s triggered by add/remove-folder don't
        # clobber a user-edited name back to the original suggestion.
        _, self._suggested_name = next_local_workspace_identity(self._registry)
        self._name_value = self._suggested_name
        self._folder_path_value = ""
        self._make_active_value = True

    def compose(self) -> ComposeResult:
        with Vertical(id="workspace-create-modal"):
            yield Static("New Workspace", classes="console-modal-header")
            yield Static(
                _WORKSPACE_EXPLAINER, id="workspace-create-explainer", markup=False
            )
            yield Input(
                value=self._name_value,
                id="workspace-create-name",
                placeholder="Workspace name",
                compact=True,
            )
            with Horizontal(id="workspace-create-folder-row"):
                yield Input(
                    value=self._folder_path_value,
                    id="workspace-create-folder-path",
                    placeholder="~/path/to/project (optional)",
                    compact=True,
                )
                yield Button("Browse…", id="workspace-create-browse", compact=True)
                yield Button(
                    "Add folder", id="workspace-create-folder-add", compact=True
                )
            with Vertical(id="workspace-create-folder-list"):
                for index, folder in enumerate(self._folders):
                    with Horizontal(classes="workspace-create-folder-item"):
                        yield Static(
                            folder,
                            classes="workspace-create-folder-locator",
                            markup=False,
                        )
                        yield Button(
                            "Remove",
                            id=f"workspace-create-folder-remove-{index}",
                            compact=True,
                        )
            yield Static(self._error, id="workspace-create-error", markup=False)
            yield Checkbox(
                "Switch to this workspace",
                self._make_active_value,
                id="workspace-create-make-active",
                compact=True,
            )
            with Horizontal(id="workspace-create-actions"):
                yield Button("Cancel", id="workspace-create-cancel", compact=True)
                yield Button("Create", id="workspace-create-confirm", compact=True)

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

    def _stash_form_state(self) -> None:
        """Capture live Input/Checkbox values before a recompose discards them.

        `refresh(recompose=True)` tears down and rebuilds every child widget
        from `compose()`, which re-reads `self._name_value`,
        `self._folder_path_value`, and `self._make_active_value`. Without
        capturing the live values here first, a folder add/remove would
        silently reset a user-edited name back to the original suggestion
        (and an unchecked "make active" box back to checked).
        """
        self._name_value = self.query_one("#workspace-create-name", Input).value
        self._folder_path_value = self.query_one(
            "#workspace-create-folder-path", Input
        ).value
        self._make_active_value = self.query_one(
            "#workspace-create-make-active", Checkbox
        ).value

    def _set_error(self, message: str) -> None:
        self._error = message
        self.query_one("#workspace-create-error", Static).update(message)

    @on(Button.Pressed, "#workspace-create-folder-add")
    def _add_folder(self, event: Button.Pressed) -> None:
        event.stop()
        raw = self.query_one("#workspace-create-folder-path", Input).value.strip()
        if not raw:
            return
        try:
            resolved = validate_folder_binding_path(raw, self._folders)
        except WorkspaceRegistryServiceError as exc:
            self._set_error(str(exc))
            return
        self._folders.append(str(resolved))
        self._error = ""
        self._stash_form_state()
        self._folder_path_value = ""  # the just-consumed path is cleared
        self.refresh(recompose=True)

    @on(Button.Pressed, "#workspace-create-folder-list Button")
    def _remove_folder(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = event.button.id or ""
        try:
            index = int(button_id.rsplit("-", 1)[-1])
        except ValueError:
            return
        if 0 <= index < len(self._folders):
            del self._folders[index]
            self._stash_form_state()
            self.refresh(recompose=True)

    @on(Input.Submitted, "#workspace-create-name")
    def _name_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._create()

    @on(Button.Pressed, "#workspace-create-confirm")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self._create()

    def _create(self) -> None:
        name = self.query_one("#workspace-create-name", Input).value.strip()
        workspace_id, generated_name = next_local_workspace_identity(self._registry)
        try:
            self._registry.create_workspace(
                workspace_id=workspace_id,
                name=name or generated_name,
                description="Created from the workspace setup dialog.",
            )
        except WorkspaceRegistryServiceError as exc:
            self._set_error(str(exc))
            return
        bound: list[str] = []
        failed: list[tuple[str, str]] = []
        for folder in self._folders:
            try:
                self._registry.add_folder_binding(workspace_id, folder)
                bound.append(folder)
            except WorkspaceRegistryServiceError as exc:
                failed.append((folder, str(exc)))
        self.dismiss(
            WorkspaceCreateResult(
                workspace_id=workspace_id,
                name=name or generated_name,
                bound_folders=tuple(bound),
                failed_folders=tuple(failed),
                make_active=self.query_one(
                    "#workspace-create-make-active", Checkbox
                ).value,
            )
        )
