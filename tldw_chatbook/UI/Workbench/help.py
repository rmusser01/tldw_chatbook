"""Contextual help for visible Workbench actions."""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from tldw_chatbook.UI.Workbench.workbench_state import WorkbenchAction


@dataclass(frozen=True)
class WorkbenchHelpState:
    """Plain-text help content for the current Workbench route."""

    route_id: str
    title: str
    actions: tuple[WorkbenchAction, ...] = ()
    shortcuts: tuple[tuple[str, str], ...] = ()

    def render_text(self) -> str:
        """Render visible actions and explicit shortcuts as plain text."""
        lines = [self.title]
        visible_actions = tuple(action for action in self.actions if not action.disabled)
        if visible_actions:
            lines.append("Actions:")
            lines.extend(f"- {action.label}" for action in visible_actions)
        if self.shortcuts:
            lines.append("Shortcuts:")
            lines.extend(f"- {key}: {label}" for key, label in self.shortcuts)
        return "\n".join(lines)


class WorkbenchHelpPanel(ModalScreen[None]):
    """Modal panel showing contextual Workbench help."""

    def __init__(self, state: WorkbenchHelpState) -> None:
        super().__init__()
        self.state = state

    def compose(self) -> ComposeResult:
        with Vertical(id="workbench-help-panel", classes="workbench-help-panel"):
            yield Static(self.state.render_text(), id="workbench-help-body")
            yield Button("Close", id="workbench-help-close", compact=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dismiss the panel from its close button."""
        if event.button.id == "workbench-help-close":
            event.stop()
            self.dismiss(None)
