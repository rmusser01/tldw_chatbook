"""Submit workspace lifecycle actions through the selected canonical owner."""

from collections.abc import Callable, Sequence

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static

from ...Research_Workspace.contracts import (
    QualifiedWorkspaceRef,
    ResearchWorkspaceSummary,
    WorkspaceDataSource,
    require_capability,
)
from ...Research_Workspace.quick_notes import ResearchNoteSaveRequest
from ...Research_Workspace.workspace_templates import WORKSPACE_TEMPLATES
from ...Widgets.confirmation_dialog import ConfirmationDialog
from ...Widgets.modal_dismissal import SafeModalDismissMixin
from ..Navigation.main_navigation import NavigateToScreen
from .quick_notes_section import ResearchQuickNotesSection
from .studio_region import ResearchStudioRegion
from .workspace_menu import ResearchPaneModeStrip


class ResearchWorkspaceName(SafeModalDismissMixin, ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    SAFE_MODAL_CONTENT = "#research-workspace-name-dialog"
    AUTO_FOCUS = "#research-workspace-name"

    def __init__(
        self,
        title: str,
        initial: str = "",
        *,
        description: str = "",
        label: str = "Workspace name",
        max_length: int = 256,
        allow_blank: bool = False,
    ) -> None:
        super().__init__()
        self.title = title
        self.initial = initial
        self.description = description
        self.label = label
        self.max_length = max_length
        self.allow_blank = allow_blank

    def compose(self) -> ComposeResult:
        with Vertical(
            id="research-workspace-name-dialog", classes="research-transfer-dialog"
        ):
            yield Static(self.title, classes="research-pane-title", markup=False)
            if self.description:
                yield Static(self.description, markup=False)
            yield Input(
                self.initial,
                placeholder=self.label,
                max_length=self.max_length,
                id="research-workspace-name",
            )
            yield Static("", id="research-workspace-name-status", markup=False)
            yield Button(
                "Save",
                id="research-workspace-name-confirm",
                compact=True,
                variant="primary",
            )
            yield Button("Cancel", id="research-workspace-name-cancel", compact=True)

    @on(Button.Pressed, "#research-workspace-name-confirm")
    @on(Input.Submitted, "#research-workspace-name")
    def submit(self) -> None:
        value = self.query_one(Input).value.strip()
        if (not value and not self.allow_blank) or any(
            ord(char) < 32 for char in value
        ):
            self.query_one("#research-workspace-name-status", Static).update(
                f"Enter {self.label.lower()} on one line."
            )
            return
        self.dismiss(value)

    @on(Button.Pressed, "#research-workspace-name-cancel")
    def cancel(self) -> None:
        self.dismiss(None)


class ResearchWorkspacePicker(
    SafeModalDismissMixin, ModalScreen[QualifiedWorkspaceRef | None]
):
    BINDINGS = [Binding("escape", "request_safe_cancel", "Close", show=False)]
    SAFE_MODAL_CONTENT = "#research-workspace-list-dialog"

    def __init__(
        self,
        authority: WorkspaceDataSource,
        rows: Sequence[ResearchWorkspaceSummary],
        *,
        mode: str,
    ) -> None:
        super().__init__()
        self.authority = authority
        self.rows = rows
        self.mode = mode

    def compose(self) -> ComposeResult:
        with Vertical(
            id="research-workspace-list-dialog", classes="research-transfer-dialog"
        ):
            yield Static(
                f"{self.mode.title()} {self.authority.value.title()} workspaces",
                classes="research-pane-title",
            )
            if self.rows:
                yield Select(
                    ((Text(row.name), row.ref) for row in self.rows),
                    prompt="Choose a workspace",
                    id=f"research-{self.mode}-picker",
                )
                yield Button(
                    "Restore" if self.mode == "archived" else "Open",
                    id="research-workspace-restore"
                    if self.mode == "archived"
                    else f"research-workspace-open-{self.mode}",
                    classes="research-workspace-picker-confirm",
                    compact=True,
                    disabled=True,
                )
            else:
                yield Static(f"No {self.mode} workspaces.")
            yield Button("Close", id="research-workspace-picker-close", compact=True)

    @on(Select.Changed)
    def selected(self, event: Select.Changed) -> None:
        self.query_one(
            ".research-workspace-picker-confirm", Button
        ).disabled = not isinstance(event.value, QualifiedWorkspaceRef)

    @on(Button.Pressed, ".research-workspace-picker-confirm")
    def choose(self) -> None:
        target = self.query_one(Select).value
        if isinstance(target, QualifiedWorkspaceRef) and any(
            row.ref == target for row in self.rows
        ):
            self.dismiss(target)

    @on(Button.Pressed, "#research-workspace-picker-close")
    def close(self) -> None:
        self.dismiss(None)


class ResearchWorkspaceTemplates(SafeModalDismissMixin, ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "request_safe_cancel", "Close", show=False)]
    SAFE_MODAL_CONTENT = "#research-templates-dialog"

    def __init__(self, authority: WorkspaceDataSource) -> None:
        super().__init__()
        self.authority = authority

    def compose(self) -> ComposeResult:
        with Vertical(
            id="research-templates-dialog", classes="research-transfer-dialog"
        ):
            yield Static(
                f"{self.authority.value.title()} workspace templates",
                classes="research-pane-title",
            )
            for key, template in WORKSPACE_TEMPLATES.items():
                yield Button(
                    template.name,
                    id=f"research-template-{key}",
                    name=key,
                    classes="research-workspace-template",
                    compact=True,
                )
                yield Static(template.description, markup=False)
            yield Button("Close", id="research-templates-close", compact=True)

    @on(Button.Pressed, ".research-workspace-template")
    def choose(self, event: Button.Pressed) -> None:
        if event.button.name in WORKSPACE_TEMPLATES:
            self.dismiss(event.button.name)

    @on(Button.Pressed, "#research-templates-close")
    def close(self) -> None:
        self.dismiss(None)


LIFECYCLE_ACTIONS = frozenset(
    {"new", "rename", "archive", "archived", "duplicate", "delete-local", "templates"}
)


async def run_workspace_lifecycle(
    screen, action: str, authority: WorkspaceDataSource, current: Callable[[], bool]
) -> None:
    if action == "delete-local":
        screen.post_message(NavigateToScreen("settings", {"category": "workspaces"}))
        return
    port = screen.controller.port_for_data_source(authority)
    if port is None:
        raise ValueError(
            f"The {authority.value.title()} workspace catalog is unavailable."
        )
    workspace = screen.controller.visible_workspace
    notes = screen.query_one(ResearchQuickNotesSection) if action == "templates" else None
    studio = screen.query_one(ResearchStudioRegion) if action == "templates" else None
    if action == "archived":
        rows = tuple(
            row
            for row in await port.list_workspaces(include_archived=True)
            if row.archived
        )
        if any(row.ref.data_source is not authority for row in rows):
            raise ValueError("Archive results belong to a different authority.")
        if not current():
            return
        selected = await screen.app.push_screen_wait(
            ResearchWorkspacePicker(authority, rows, mode="archived")
        )
        if (
            selected is None
            or not current()
            or not await screen.flush_pending_work()
            or not current()
        ):
            return
        archived = next(row for row in rows if row.ref == selected)
        require_capability(await port.capabilities(selected), "restore")
        if current():
            restored = await port.restore_workspace(
                selected, expected_version=archived.version
            )
            if current():
                await screen._refresh_workspace_catalog(restored.ref)
                await screen.remember_workspace_visit(restored.ref)
        return

    if workspace is not None:
        capability = {"rename": "update", "new": "create", "templates": "create"}.get(
            action, action
        )
        require_capability(await port.capabilities(workspace.ref), capability)
        if not current():
            return
    if action == "archive" and workspace is not None:
        confirmed = await screen.app.push_screen_wait(
            ConfirmationDialog(
                f"Archive {workspace.name}?",
                "Open Archived workspaces to restore it later.",
                confirm_label="Archive",
            )
        )
        if confirmed and current() and await screen.flush_pending_work() and current():
            await port.archive_workspace(
                workspace.ref, expected_version=workspace.version
            )
            if current():
                await screen._refresh_workspace_catalog()
        return

    template = None
    if action == "templates":
        key = await screen.app.push_screen_wait(ResearchWorkspaceTemplates(authority))
        if key is None or not current():
            return
        template = WORKSPACE_TEMPLATES[key]
        dialog = ResearchWorkspaceName(
            f"New {authority.value.title()} workspace · {template.name}",
            template.name,
            description=f"{template.description} Studio will open {template.note_title} as a Quick Note draft.",
        )
    elif action == "new":
        dialog = ResearchWorkspaceName(f"New {authority.value.title()} workspace")
    elif workspace is not None and action == "rename":
        dialog = ResearchWorkspaceName("Rename workspace", workspace.name)
    elif (
        workspace is not None
        and action == "duplicate"
        and authority is WorkspaceDataSource.LOCAL
    ):
        dialog = ResearchWorkspaceName(
            "Duplicate workspace details",
            f"{workspace.name} copy"[:256],
            description="Start a new workspace with the same description.",
        )
    else:
        return
    name = await screen.app.push_screen_wait(dialog)
    if (
        name is None
        or not current()
        or not await screen.flush_pending_work()
        or not current()
    ):
        return
    if action in {"new", "templates"}:
        description = (
            template.description
            if template and authority is WorkspaceDataSource.LOCAL
            else ""
        )
        saved = await port.create_workspace(name=name, description=description)
    elif action == "rename":
        saved = await port.update_workspace(
            workspace.ref, name=name, expected_version=workspace.version
        )
    else:
        saved = await port.duplicate_workspace(workspace.ref, name=name)
    if current():
        await screen._refresh_workspace_catalog(saved.ref)
        if action != "rename":
            await screen.remember_workspace_visit(saved.ref)
        if (
            template
            and notes is not None and notes.is_attached
            and studio is not None and studio.is_attached
            and screen.controller.selected_ref == saved.ref
        ):
            screen.select_pane_mode(ResearchPaneModeStrip.Selected("studio"))
            notes.new_draft(
                ResearchNoteSaveRequest(
                    template.note_title, template.note, tags=("template",)
                )
            )
            studio.jump_to_quick_notes()
