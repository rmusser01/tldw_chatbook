"""Console workspace switcher modal."""

from __future__ import annotations

from typing import ClassVar

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Static

from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin
from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID, WorkspaceRecord

#: TASK-714: the switcher dismisses with an (action, workspace_id) tuple -
#: "switch", "rename", or "archive" - or None on cancel.
WorkspaceSwitcherResult = tuple[str, str]


def workspace_persona_label_suffix(
    app_instance: object, workspace: WorkspaceRecord
) -> str:
    """Return ``" · {persona_label}"`` for a workspace with an available
    default assistant persona, or ``""``.

    Task 11 (workspace-assistant-defaults): the switcher annotates each
    workspace whose effective assistant default resolves ``available``.
    Guarded end to end (missing registry/persona services, lookup errors,
    deleted personas) — any failure is a silent omit, never a broken modal.
    """
    try:
        from tldw_chatbook.Workspaces.assistant_defaults import (
            resolve_effective_assistant_default,
        )

        registry = getattr(app_instance, "workspace_registry_service", None)
        record = None
        if registry is not None and hasattr(registry, "get_workspace"):
            try:
                record = registry.get_workspace(workspace.workspace_id)
            except Exception:  # noqa: BLE001 - display-only, degrade silently
                record = None
        if record is None:
            record = workspace
        defaults = getattr(record, "assistant_defaults", None)
        personas = getattr(app_instance, "local_character_persona_service", None)

        def lookup(persona_id: str):
            if personas is None or not hasattr(
                personas, "get_persona_profile"
            ):
                return None
            try:
                return personas.get_persona_profile(persona_id)
            except Exception:  # noqa: BLE001 - display-only
                return None

        effective = resolve_effective_assistant_default(defaults, lookup)
        if effective.status == "available" and effective.label:
            return f" · {effective.label}"
    except Exception:  # noqa: BLE001 - display-only, degrade silently
        pass
    return ""


class ConsoleWorkspaceSwitcherModal(
    SafeModalDismissMixin, ModalScreen[WorkspaceSwitcherResult | None]
):
    """Choose the active workspace for Console context.

    Args:
        workspaces: Workspace records available for selection in the modal.
        active_workspace_id: Workspace id that should render as the current
            non-actionable row, or ``None`` when no workspace is active.
    """

    DEFAULT_CSS = """
    ConsoleWorkspaceSwitcherModal {
        align: center middle;
    }

    #console-workspace-switcher-modal {
        width: 64;
        height: auto;
        max-height: 28;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #console-workspace-switcher-list {
        height: auto;
        max-height: 18;
        margin: 1 0;
    }

    .console-workspace-switcher-row {
        width: 100%;
        height: 3;
        min-height: 3;
    }

    /* TASK-714: the option shares its row with compact Rename/Archive
       buttons - 1fr (not 100%) so the lifecycle controls keep real width
       instead of being pushed past the modal clip (the TASK-712 failure
       class). */
    .console-workspace-switcher-option {
        width: 1fr;
        height: 3;
        min-height: 3;
        margin: 0;
    }

    .console-workspace-switcher-lifecycle {
        width: auto;
        min-width: 9;
        height: 3;
        min-height: 3;
        margin: 0 0 0 1;
    }

    .console-workspace-switcher-current {
        content-align: center middle;
        background: $surface;
        color: $text;
        text-style: bold;
    }

    #console-workspace-switcher-actions {
        height: 3;
        min-height: 3;
        align-horizontal: right;
    }

    #console-workspace-switcher-cancel {
        width: 10;
        min-width: 10;
        height: 3;
        min-height: 3;
    }
    """

    SAFE_MODAL_CONTENT = "#console-workspace-switcher-modal"
    BINDINGS = [
        ("escape", "request_safe_cancel", "Cancel"),
        # TASK-722: arrow ergonomics on top of Tab/Shift+Tab focus cycling.
        ("down", "focus_next", "Next"),
        ("up", "focus_previous", "Previous"),
    ]

    # TASK-722: land focus on the first actionable workspace option so the
    # modal is operable start-to-finish without a pointer (Enter selects).
    AUTO_FOCUS = "Button.console-workspace-switcher-option"

    def __init__(
        self,
        *,
        workspaces: tuple[WorkspaceRecord, ...],
        active_workspace_id: str | None,
        show_archived: bool = False,
    ) -> None:
        super().__init__()
        self._workspaces = workspaces
        self._active_workspace_id = active_workspace_id
        self._show_archived = show_archived

    def compose(self) -> ComposeResult:
        with Vertical(id="console-workspace-switcher-modal"):
            yield Static("Change Workspace", classes="console-modal-header")
            yield Static(
                "Switching changes Console context only; Library and Notes stay globally visible.",
                id="console-workspace-switcher-copy",
                markup=False,
            )
            yield Checkbox(
                "Show archived",
                self._show_archived,
                id="console-workspace-show-archived",
            )
            with VerticalScroll(id="console-workspace-switcher-list"):
                for index, workspace in enumerate(self._workspaces):
                    row = Horizontal(classes="console-workspace-switcher-row")
                    row.display = not workspace.archived or self._show_archived
                    with row:
                        # ADR-027 (TASK-723): the switcher and the browser
                        # must tell one story - Default holds everyday chats
                        # (the browser's Chats section), named workspaces get
                        # grouped contexts.
                        display_name = (
                            f"{workspace.name} (everyday chats)"
                            if workspace.workspace_id == DEFAULT_WORKSPACE_ID
                            else workspace.name
                        )
                        # Task 11: annotate workspaces whose effective
                        # default assistant resolves available; silent omit
                        # otherwise (helper degrades on any lookup failure).
                        display_name += workspace_persona_label_suffix(
                            self.app, workspace
                        )
                        if workspace.archived:
                            yield Static(
                                f"(archived) {display_name}",
                                classes="console-workspace-switcher-option",
                                markup=False,
                            )
                            yield Button(
                                "Restore",
                                id=f"console-workspace-restore-{index}",
                                classes="console-workspace-switcher-lifecycle",
                                compact=True,
                            )
                            yield Button(
                                "Restore as…",
                                id=f"console-workspace-restore_as-{index}",
                                classes="console-workspace-switcher-lifecycle",
                                compact=True,
                            )
                            continue
                        if workspace.workspace_id == self._active_workspace_id:
                            yield Static(
                                f"{display_name} (current)",
                                id=f"console-workspace-switch-current-{index}",
                                classes=(
                                    "console-workspace-switcher-option "
                                    "console-workspace-switcher-current"
                                ),
                                markup=False,
                            )
                        else:
                            button = Button(
                                Text(display_name),
                                id=f"console-workspace-switch-{index}",
                                classes="console-workspace-switcher-option",
                                compact=True,
                            )
                            button.tooltip = (
                                f"Use {workspace.name} as the active Console workspace"
                            )
                            yield button
                        # TASK-714: lifecycle controls. The built-in Default
                        # workspace keeps its identity (rail copy and runtime
                        # rules reference it by name), so it gets neither.
                        if workspace.workspace_id != DEFAULT_WORKSPACE_ID:
                            rename = Button(
                                "Rename",
                                id=f"console-workspace-rename-{index}",
                                classes="console-workspace-switcher-lifecycle",
                                compact=True,
                            )
                            rename.tooltip = f"Rename {workspace.name}"
                            yield rename
                            archive = Button(
                                "Archive",
                                id=f"console-workspace-archive-{index}",
                                classes="console-workspace-switcher-lifecycle",
                                compact=True,
                            )
                            archive.tooltip = (
                                f"Archive {workspace.name}. Its conversations "
                                "stay saved and remain visible in Library."
                            )
                            yield archive
            with Horizontal(id="console-workspace-switcher-actions"):
                yield Button(
                    "Cancel", id="console-workspace-switcher-cancel", compact=True
                )

    @on(Checkbox.Changed, "#console-workspace-show-archived")
    def _toggle_archived(self, event: Checkbox.Changed) -> None:
        event.stop()
        self._show_archived = event.value
        for row, workspace in zip(
            self.query(".console-workspace-switcher-row"), self._workspaces
        ):
            row.display = not workspace.archived or event.value

    @on(Button.Pressed, "#console-workspace-switcher-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    def _workspace_at(self, button_id: str) -> WorkspaceRecord | None:
        try:
            index = int(button_id.rsplit("-", 1)[-1])
        except ValueError:
            return None
        if 0 <= index < len(self._workspaces):
            return self._workspaces[index]
        return None

    @on(Button.Pressed, ".console-workspace-switcher-option")
    def _select_workspace(self, event: Button.Pressed) -> None:
        event.stop()
        workspace = self._workspace_at(event.button.id or "")
        if workspace is not None:
            self.dismiss(("switch", workspace.workspace_id))

    @on(Button.Pressed, ".console-workspace-switcher-lifecycle")
    def _lifecycle_action(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = event.button.id or ""
        workspace = self._workspace_at(button_id)
        if workspace is None:
            return
        action = button_id.removeprefix("console-workspace-").rsplit("-", 1)[0]
        self.dismiss((action, workspace.workspace_id))


class ConsoleWorkspaceRenameModal(SafeModalDismissMixin, ModalScreen[str | None]):
    """Prompt for a new workspace name (TASK-714)."""

    DEFAULT_CSS = """
    ConsoleWorkspaceRenameModal {
        align: center middle;
    }

    #console-workspace-rename-modal {
        width: 56;
        height: auto;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #console-workspace-rename-input {
        width: 100%;
        margin: 1 0 0 0;
    }

    #console-workspace-rename-actions {
        height: 3;
        min-height: 3;
        margin: 1 0 0 0;
        align-horizontal: right;
    }
    """

    SAFE_MODAL_CONTENT = "#console-workspace-rename-modal"
    BINDINGS = [("escape", "request_safe_cancel", "Cancel")]

    AUTO_FOCUS = "#console-workspace-rename-input"

    def __init__(self, *, current_name: str, restoring: bool = False) -> None:
        super().__init__()
        self._current_name = current_name
        self._restoring = restoring

    def compose(self) -> ComposeResult:
        with Vertical(id="console-workspace-rename-modal"):
            yield Static(
                "Restore workspace as" if self._restoring else "Rename Workspace",
                classes="console-modal-header",
            )
            yield Input(
                value=self._current_name,
                id="console-workspace-rename-input",
                placeholder="Workspace name",
            )
            with Horizontal(id="console-workspace-rename-actions"):
                yield Button(
                    "Cancel", id="console-workspace-rename-cancel", compact=True
                )
                yield Button(
                    "Restore" if self._restoring else "Save",
                    id="console-workspace-rename-save",
                    compact=True,
                )

    def _submit(self) -> None:
        value = self.query_one("#console-workspace-rename-input", Input).value.strip()
        if value:
            self.dismiss(value)

    @on(Button.Pressed, "#console-workspace-rename-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    @on(Button.Pressed, "#console-workspace-rename-save")
    def _save(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit()

    @on(Input.Submitted, "#console-workspace-rename-input")
    def _submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._submit()


class WorkspaceArchiveReceiptModal(SafeModalDismissMixin, ModalScreen[str | None]):
    """Persistent recovery choices, dismissed only by deliberate interaction."""

    DEFAULT_CSS = """
    WorkspaceArchiveReceiptModal { align: center middle; }
    #workspace-archive-receipt { width: 60; height: auto; border: tall $primary;
        background: $surface; padding: 1 2; }
    #workspace-archive-receipt-actions { height: 3; }
    #workspace-archive-receipt-actions Button { width: auto; min-width: 8; margin-right: 1; }
    """
    SAFE_MODAL_CONTENT = "#workspace-archive-receipt"
    BINDINGS: ClassVar = [("escape", "request_safe_cancel", "Done")]
    AUTO_FOCUS = "#workspace-archive-undo"

    def __init__(
        self, *, name: str, kind: str = "Workspace", description: str | None = None
    ) -> None:
        super().__init__()
        self._name = name
        self._kind = kind
        self._description = description

    def compose(self) -> ComposeResult:
        with Vertical(id="workspace-archive-receipt"):
            yield Static(
                f"{self._kind} archived", classes="console-modal-header", markup=False
            )
            yield Static(
                self._description
                if self._description is not None
                else f"{self._name} is archived. Saved conversations stay in Library. Restore it here or with Show archived in the workspace switcher.",
                markup=False,
            )
            with Horizontal(id="workspace-archive-receipt-actions"):
                yield Button("Undo", id="workspace-archive-undo", compact=True)
                yield Button("View archived", id="workspace-archive-view", compact=True)
                yield Button("Done", id="workspace-archive-done", compact=True)

    @on(Button.Pressed)
    async def _choose(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "workspace-archive-done":
            await self.request_safe_cancel(source="button")
        else:
            self.dismiss(
                "undo" if event.button.id == "workspace-archive-undo" else "view"
            )
