"""Folder binding and project-index composition for the Settings workspace card."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Input, Static

from ...Workspaces.models import RuntimeBindingStatus
from ...Workspaces.registry_service import (
    LocalWorkspaceRegistryService,
    binding_exclusion_entries,
)

if TYPE_CHECKING:
    from ..Screens.settings_screen import SettingsScreen


def render_workspace_folder_bindings(
    self: SettingsScreen,
    registry: LocalWorkspaceRegistryService,
    workspace_id: str,
) -> ComposeResult:
    """Render the folder-bindings editor for the selected workspace (task 10).

    One row per bound folder with its access level and freshness
    (recomputed from disk by `list_folder_bindings`), a per-row
    ro/rw toggle and remove button, then an add row. Below each
    binding sit its exclusion rows (paths the agent cannot see) with
    an Unexclude button each, plus a per-binding "Exclude" input row.
    Toggle/remove buttons stash `binding_id` as a plain attribute at
    compose time (mirrors the conversation browser's `conversation_id`
    stash) so the handler never has to parse a uuid out of the button
    id; exclusion buttons stash `exclusion_*` attributes instead (the
    Unexclude button shares the folder-remove CSS class, and the
    class-selected folder handler must stay a no-op on it).
    """
    from tldw_chatbook.Widgets.Settings_Widgets.project_index_binding_card import (
        ProjectIndexBindingCard,
    )

    yield Static("Folders (agent file-tool access)", classes="destination-section")
    for binding in registry.list_folder_bindings(workspace_id):
        access = binding.metadata.get("access", "ro")
        freshness = (
            "ready" if binding.status == RuntimeBindingStatus.READY else "missing"
        )
        yield Static(
            f"{binding.locator} [{access}] {freshness}",
            id=f"settings-workspace-folder-{binding.binding_id}",
            classes="settings-detail-row",
            markup=False,
        )
        with Horizontal(classes="settings-input-row"):
            toggle_button = Button(
                "Allow write" if access != "rw" else "Read-only",
                id=f"settings-workspace-folder-toggle-{binding.binding_id}",
                classes="settings-workspace-folder-toggle",
                compact=True,
            )
            toggle_button.binding_id = binding.binding_id
            yield toggle_button
            remove_button = Button(
                "Remove",
                id=f"settings-workspace-folder-remove-{binding.binding_id}",
                classes="settings-workspace-folder-remove",
                compact=True,
            )
            remove_button.binding_id = binding.binding_id
            yield remove_button
        yield self._workspace_folder_result_widget(workspace_id, binding.binding_id)
        yield ProjectIndexBindingCard(
            binding.binding_id,
            workspace_id=workspace_id,
            root=binding.locator,
            available=binding.status == RuntimeBindingStatus.READY,
        )
        # Defensive reader (never the raw metadata list): malformed
        # persisted exclusions must render as "Excluded (0)" rather than
        # crash pane composition.
        exclusions = binding_exclusion_entries(binding)
        yield Static(
            f"  Excluded ({len(exclusions)}): agent cannot see these",
            classes="settings-detail-row",
        )
        for index, entry in enumerate(exclusions):
            with Horizontal(classes="settings-input-row"):
                yield Static(
                    f"  {entry.path} ({entry.kind})",
                    id=f"settings-workspace-excl-label-{binding.binding_id}-{index}",
                    classes="settings-detail-row",
                )
                unexclude_button = Button(
                    "Unexclude",
                    id=f"settings-workspace-excl-remove-{binding.binding_id}-{index}",
                    classes="settings-workspace-folder-remove",
                    compact=True,
                )
                # ``exclusion_*`` rather than ``binding_id``: the legacy
                # folder-remove class handler also matches this class and
                # reads ``binding_id`` -- these attrs keep it a no-op here.
                unexclude_button.exclusion_binding_id = binding.binding_id
                unexclude_button.exclusion_path = str(entry.path)
                yield unexclude_button
        with Horizontal(classes="settings-input-row"):
            yield Input(
                placeholder="path/to/exclude (relative to folder)",
                id=f"settings-workspace-excl-path-{binding.binding_id}",
                classes="settings-compact-input",
            )
            add_exclusion_button = Button(
                "Exclude",
                id=f"settings-workspace-excl-add-{binding.binding_id}",
                compact=True,
            )
            add_exclusion_button.exclusion_binding_id = binding.binding_id
            yield add_exclusion_button
    with Horizontal(classes="settings-input-row"):
        yield Input(
            placeholder="~/path/to/folder",
            id="settings-workspace-folder-path",
            classes="settings-compact-input",
        )
        yield Button("Add folder", id="settings-workspace-folder-add", compact=True)
    yield self._workspace_folder_result_widget(workspace_id)
