"""Console workspace switcher modal."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID, WorkspaceRecord

#: TASK-714: the switcher dismisses with an (action, workspace_id) tuple -
#: "switch", "rename", or "archive" - or None on cancel.
WorkspaceSwitcherResult = tuple[str, str]


class ConsoleWorkspaceSwitcherModal(ModalScreen[WorkspaceSwitcherResult | None]):
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

    BINDINGS = [
        ("escape", "dismiss", "Cancel"),
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
    ) -> None:
        super().__init__()
        self._workspaces = workspaces
        self._active_workspace_id = active_workspace_id

    def compose(self) -> ComposeResult:
        with Vertical(id="console-workspace-switcher-modal"):
            yield Static("Change Workspace", classes="console-modal-header")
            yield Static(
                "Switching changes Console context only; Library and Notes stay globally visible.",
                id="console-workspace-switcher-copy",
                markup=False,
            )
            with Vertical(id="console-workspace-switcher-list"):
                for index, workspace in enumerate(self._workspaces):
                    label = workspace.name
                    with Horizontal(
                        classes="console-workspace-switcher-row",
                    ):
                        if workspace.workspace_id == self._active_workspace_id:
                            yield Static(
                                f"{workspace.name} (current)",
                                id=f"console-workspace-switch-current-{index}",
                                classes=(
                                    "console-workspace-switcher-option "
                                    "console-workspace-switcher-current"
                                ),
                                markup=False,
                            )
                        else:
                            button = Button(
                                label,
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

    def action_dismiss(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#console-workspace-switcher-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

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
        action = "rename" if button_id.startswith(
            "console-workspace-rename-"
        ) else "archive"
        self.dismiss((action, workspace.workspace_id))


class ConsoleWorkspaceRenameModal(ModalScreen[str | None]):
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

    BINDINGS = [("escape", "dismiss", "Cancel")]

    AUTO_FOCUS = "#console-workspace-rename-input"

    def __init__(self, *, current_name: str) -> None:
        super().__init__()
        self._current_name = current_name

    def compose(self) -> ComposeResult:
        with Vertical(id="console-workspace-rename-modal"):
            yield Static("Rename Workspace", classes="console-modal-header")
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
                    "Save", id="console-workspace-rename-save", compact=True
                )

    def action_dismiss(self) -> None:
        self.dismiss(None)

    def _submit(self) -> None:
        value = self.query_one(
            "#console-workspace-rename-input", Input
        ).value.strip()
        if value:
            self.dismiss(value)

    @on(Button.Pressed, "#console-workspace-rename-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#console-workspace-rename-save")
    def _save(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit()

    @on(Input.Submitted, "#console-workspace-rename-input")
    def _submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._submit()
