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

from tldw_chatbook.Skills_Interop.project_skills_discovery import (
    ProjectSkillsDiscovery,
    discover_project_skills,
)
from tldw_chatbook.Third_Party.textual_fspicker import SelectDirectory
from tldw_chatbook.Utils.input_validation import sanitize_string
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin
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
    #: One ProjectSkillsDiscovery per bound folder whose root contains a
    #: non-empty .SKILLS/ (spec §5.5 create-modal chaining). Empty when no
    #: bound folder had project skills, or none were bound at all.
    project_skills: tuple[ProjectSkillsDiscovery, ...] = ()


class WorkspaceCreateModal(
    SafeModalDismissMixin, ModalScreen["WorkspaceCreateResult | None"]
):
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

    SAFE_MODAL_CONTENT = "#workspace-create-modal"
    BINDINGS = [("escape", "request_safe_cancel", "Cancel")]
    AUTO_FOCUS = "#workspace-create-name"

    def __init__(self, *, registry_service: LocalWorkspaceRegistryService) -> None:
        super().__init__()
        self._registry = registry_service
        self._folders: list[str] = []
        #: Folder locator -> its project-skills discovery, populated only
        #: for folders whose root contains a non-empty .SKILLS/ (spec §5.5
        #: create-modal chaining). Keyed by the same resolved-locator string
        #: stored in ``self._folders`` so `_create()` can filter by
        #: ``bound_folders`` without a second discovery pass.
        self._folder_discoveries: dict[str, ProjectSkillsDiscovery] = {}
        self._error = ""
        # Finding 1: one-shot guard against a double-submit (rapid
        # Enter-Enter, or a double-click race on Create) running
        # create_workspace()/add_folder_binding() twice before the modal
        # actually pops off the screen stack.
        self._committed = False
        # Finding 7: once a workspace has actually been created, a Create
        # press retries only the remaining folder bindings -- the
        # name/sanitize/create_workspace logic below is skipped entirely.
        self._created_workspace_id: str | None = None
        self._created_workspace_name: str = ""
        self._make_active_result: bool = True
        self._bound_folders: tuple[str, ...] = ()
        # (path -> message) for whatever is still failing after the most
        # recent bind attempt; used to build the partial result if the user
        # cancels instead of retrying.
        self._failed_folder_messages: dict[str, str] = {}
        # Captured once so recompose()s triggered by add/remove-folder don't
        # clobber a user-edited name back to the original suggestion.
        try:
            _, self._suggested_name = next_local_workspace_identity(self._registry)
        except WorkspaceRegistryServiceError:
            # Finding 6a: a raise here (list_workspaces() failing) must not
            # crash the modal before it can even mount -- fall back to an
            # empty suggestion and surface the failure inline so the user
            # can still type a name and retry.
            self._suggested_name = ""
            self._error = (
                "Workspace registry could not be read — you can still type "
                "a name and retry."
            )
        self._name_value = self._suggested_name
        self._folder_path_value = ""
        self._make_active_value = True

    def compose(self) -> ComposeResult:
        """Build the name/folder-binding form and its action buttons.

        Returns:
            The modal's widget tree: header, explainer, name input, folder
            add/list controls, inline error area, make-active checkbox, and
            the Cancel/Create action row.
        """
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
                    discovery = self._folder_discoveries.get(folder)
                    if discovery is not None and discovery.entries:
                        label = (
                            f"{folder} — contains "
                            f"{len(discovery.entries)} project skill(s)"
                        )
                    else:
                        label = folder
                    with Horizontal(classes="workspace-create-folder-item"):
                        yield Static(
                            label,
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

    @on(Button.Pressed, "#workspace-create-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    async def _perform_safe_cancel(self, *, source: str) -> None:
        """Route Cancel/Escape/backdrop to a partial result once created.

        Finding 7: a workspace that already exists (folder-binding retry in
        progress) is a fact, not something Cancel can undo -- deliver the
        current state as a result instead of ``None`` so callers still sync
        their workspace list/active-workspace UI.
        """
        if self._created_workspace_id is None:
            await super()._perform_safe_cancel(source=source)
            return
        self.dismiss_safe_once(
            WorkspaceCreateResult(
                workspace_id=self._created_workspace_id,
                name=self._created_workspace_name,
                bound_folders=self._bound_folders,
                failed_folders=tuple(
                    (folder, self._failed_folder_messages.get(folder, ""))
                    for folder in self._folders
                ),
                make_active=self._make_active_result,
                project_skills=self._project_skills_for(self._bound_folders),
            )
        )

    def _project_skills_for(self, folders: tuple[str, ...]) -> tuple:
        """Discoveries for bound folders whose root carries a .SKILLS/ dir.

        Args:
            folders: Locator strings of successfully bound folders.

        Returns:
            The ``ProjectSkillsDiscovery`` entries recorded at Add time for
            those folders, in binding order (spec 2026-08-17 §5.5).
        """
        return tuple(
            self._folder_discoveries[folder]
            for folder in folders
            if folder in self._folder_discoveries
        )

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
            self._set_error("")
            return
        try:
            resolved = validate_folder_binding_path(raw, self._folders)
        except WorkspaceRegistryServiceError as exc:
            self._set_error(str(exc))
            return
        resolved_locator = str(resolved)
        self._folders.append(resolved_locator)
        # Pure filesystem scan (no side effects, no trust decisions) so the
        # row can flag "contains N project skill(s)" up front; only stored
        # when there's something to offer later (spec §5.5).
        discovery = discover_project_skills(resolved)
        if discovery is not None and discovery.entries:
            self._folder_discoveries[resolved_locator] = discovery
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
            self._error = ""
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

    def _bind_folders(
        self, workspace_id: str, folders: list[str]
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """Attempt to bind each folder, collecting per-folder failures.

        Args:
            workspace_id: The workspace to bind folders onto.
            folders: Folder locators to attempt to bind, in order.

        Returns:
            A ``(bound, failed)`` pair: successfully bound folder paths, and
            ``(path, error message)`` pairs for the ones that failed.
        """
        bound: list[str] = []
        failed: list[tuple[str, str]] = []
        for folder in folders:
            try:
                self._registry.add_folder_binding(workspace_id, folder)
                bound.append(folder)
            except WorkspaceRegistryServiceError as exc:
                failed.append((folder, str(exc)))
        return bound, failed

    def _finish_or_retry_bindings(
        self, newly_bound: list[str], failed: list[tuple[str, str]]
    ) -> None:
        """Dismiss on full success, otherwise leave the modal open to retry.

        Finding 7: a per-folder binding failure must not discard the
        already-created workspace or the successful bindings -- it renders
        the failures inline, narrows ``self._folders`` to just the
        remainder, and resets ``_committed`` so a subsequent Create press
        retries only what is still outstanding.

        Args:
            newly_bound: Folders that just succeeded in this attempt.
            failed: ``(path, error message)`` pairs that just failed.
        """
        all_bound = tuple(self._bound_folders) + tuple(newly_bound)
        if failed:
            self._bound_folders = all_bound
            self._folders = [folder for folder, _message in failed]
            self._failed_folder_messages = dict(failed)
            self._error = "\n".join(
                f"{folder}: {message}" for folder, message in failed
            )
            self._committed = False
            self._stash_form_state()
            self.refresh(recompose=True)
            return
        self._bound_folders = all_bound
        self._failed_folder_messages = {}
        self.dismiss_safe_once(
            WorkspaceCreateResult(
                workspace_id=self._created_workspace_id,
                name=self._created_workspace_name,
                bound_folders=all_bound,
                failed_folders=(),
                make_active=self._make_active_result,
                project_skills=self._project_skills_for(all_bound),
            )
        )

    def _create(self) -> None:
        # Finding 1: guard against a double-submit (rapid Enter-Enter on the
        # name Input, or a double-click on Create) re-running the side
        # effects below before the modal has actually popped off the screen
        # stack -- without this, a second queued call would create another
        # workspace (auto-generated names don't collide across calls) and
        # then crash with ScreenStackError trying to dismiss twice.
        if self._committed:
            return
        self._committed = True

        if self._created_workspace_id is not None:
            # Finding 7 retry path: the workspace already exists -- only
            # re-attempt the folders that are still outstanding. Name/
            # sanitize/create_workspace logic below is intentionally skipped.
            bound, failed = self._bind_folders(
                self._created_workspace_id, list(self._folders)
            )
            self._finish_or_retry_bindings(bound, failed)
            return

        # Finding 2: validate the name at the boundary before it reaches the
        # registry. Control characters or an overlong name are rejected
        # inline rather than silently truncated/stripped into the DB; a
        # blank name still falls back to the generated suggestion below.
        raw_name = self.query_one("#workspace-create-name", Input).value.strip()
        sanitized_name = sanitize_string(raw_name, 100).strip()
        if raw_name and (sanitized_name != raw_name or not sanitized_name):
            self._committed = False
            self._set_error(
                "Workspace name is too long or contains unsupported characters."
            )
            return
        name = sanitized_name

        # Finding 6b: identity generation (next_local_workspace_identity)
        # can also raise WorkspaceRegistryServiceError -- it must be inside
        # this try so a raise resets _committed instead of permanently
        # locking the Create button for the rest of the session.
        try:
            workspace_id, generated_name = next_local_workspace_identity(
                self._registry
            )
            self._registry.create_workspace(
                workspace_id=workspace_id,
                name=name or generated_name,
                description="Created from the workspace setup dialog.",
            )
        except WorkspaceRegistryServiceError as exc:
            # Nothing was committed -- let the user fix the name/folder and
            # retry rather than permanently locking the dialog.
            self._committed = False
            self._set_error(str(exc))
            return

        self._created_workspace_id = workspace_id
        self._created_workspace_name = name or generated_name
        self._make_active_result = self.query_one(
            "#workspace-create-make-active", Checkbox
        ).value

        bound, failed = self._bind_folders(workspace_id, list(self._folders))
        self._finish_or_retry_bindings(bound, failed)
