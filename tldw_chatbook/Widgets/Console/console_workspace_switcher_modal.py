"""Console workspace switcher modal."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from tldw_chatbook.Workspaces.models import WorkspaceRecord


class ConsoleWorkspaceSwitcherModal(ModalScreen[str | None]):
    """Choose the active workspace for Console context.

    Args:
        workspaces: Workspace records available for selection in the modal.
        active_workspace_id: Workspace id that should render as the current
            disabled choice, or ``None`` when no workspace is active.
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

    .console-workspace-switcher-option {
        width: 100%;
        height: 3;
        min-height: 3;
        margin: 0;
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

    BINDINGS = [("escape", "dismiss", "Cancel")]

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
            yield Static("Change Workspace", classes="console-transcript-action-row")
            yield Static(
                "Switching changes Console context only; Library and Notes stay globally visible.",
                id="console-workspace-switcher-copy",
                markup=False,
            )
            with Vertical(id="console-workspace-switcher-list"):
                for index, workspace in enumerate(self._workspaces):
                    label = workspace.name
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
                        button.tooltip = f"Use {workspace.name} as the active Console workspace"
                        yield button
            with Horizontal(id="console-workspace-switcher-actions"):
                yield Button("Cancel", id="console-workspace-switcher-cancel", compact=True)

    def action_dismiss(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#console-workspace-switcher-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, ".console-workspace-switcher-option")
    def _select_workspace(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = event.button.id or ""
        try:
            index = int(button_id.rsplit("-", 1)[-1])
        except ValueError:
            return
        if 0 <= index < len(self._workspaces):
            self.dismiss(self._workspaces[index].workspace_id)
