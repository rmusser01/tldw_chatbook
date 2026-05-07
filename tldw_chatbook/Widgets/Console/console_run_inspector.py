"""Console-native run inspector."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_display_state import (
    ConsoleDisplayRow,
    ConsoleInspectorAction,
    ConsoleInspectorState,
)


_ROW_IDS = {
    "Live work": "console-inspector-live-work",
    "Provider": "console-inspector-provider",
    "Tools": "console-inspector-tools",
    "RAG/source": "console-inspector-rag-source",
    "Artifacts": "console-inspector-artifacts",
    "Approvals": "console-inspector-approvals",
}


class ConsoleRunInspector(Vertical):
    """Render Console run readiness, recovery, and action affordances."""

    def __init__(self, state: ConsoleInspectorState, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state = state

    def sync_state(self, state: ConsoleInspectorState) -> None:
        """Refresh the mounted inspector from a new display-state snapshot."""
        self.state = state
        self.refresh(recompose=True)

    @staticmethod
    def _row_id(row: ConsoleDisplayRow, index: int) -> str:
        return _ROW_IDS.get(row.label, f"console-inspector-row-{index}")

    @staticmethod
    def _button_for_action(action: ConsoleInspectorAction) -> Button:
        button = Button(
            action.label,
            id=action.widget_id,
            classes=action.classes,
            variant="primary" if action.enabled else "default",
            tooltip=action.tooltip,
        )
        button.disabled = not action.enabled
        return button

    def compose(self) -> ComposeResult:
        yield Static(
            "Run Inspector",
            id="console-run-inspector-title",
            classes="destination-section",
        )
        for index, row in enumerate(self.state.rows):
            yield Static(
                row.text,
                id=self._row_id(row, index),
                classes=f"console-inspector-row console-inspector-row-{row.status}",
            )
        for action in self.state.actions:
            yield self._button_for_action(action)
            if not action.enabled and action.disabled_reason:
                yield Static(
                    action.disabled_reason,
                    id=f"{action.widget_id}-reason",
                    classes="console-inspector-disabled-reason",
                )
