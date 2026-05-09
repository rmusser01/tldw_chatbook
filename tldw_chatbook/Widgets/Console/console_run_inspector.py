"""Console-native run inspector."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_display_state import (
    CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID,
    CONSOLE_INSPECTOR_REVIEW_TOOL_CALL_ID,
    CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID,
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

_ROW_GROUPS = (
    (
        "Run State",
        "console-inspector-run-state-heading",
        ("Live work", "Provider", "Tools"),
    ),
    (
        "Approvals",
        "console-inspector-approvals-heading",
        ("Approvals",),
    ),
    (
        "Source Readiness",
        "console-inspector-source-readiness-heading",
        ("RAG/source", "Artifacts"),
    ),
)

_ACTION_GROUPS = {
    "Run State": (CONSOLE_INSPECTOR_REVIEW_TOOL_CALL_ID,),
    "Approvals": (CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID,),
    "Source Readiness": (CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID,),
}


class ConsoleRunInspector(Vertical):
    """Render Console run readiness, recovery, and action affordances."""

    def __init__(self, state: ConsoleInspectorState, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.styles.height = "auto"
        self.styles.min_height = 0

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
            tooltip=action.tooltip if action.enabled else "",
        )
        button.disabled = not action.enabled
        if action.enabled:
            button.styles.height = 1
            button.styles.min_height = 1
        else:
            button.styles.display = "none"
            button.styles.width = 0
            button.styles.min_width = 0
            button.styles.height = 0
            button.styles.min_height = 0
        return button

    def _compose_action(self, action: ConsoleInspectorAction) -> ComposeResult:
        yield self._button_for_action(action)
        if not action.enabled and action.disabled_reason:
            reason = Static(
                action.disabled_reason,
                id=f"{action.widget_id}-reason",
                classes="console-inspector-disabled-reason console-hidden-control",
            )
            reason.styles.display = "none"
            reason.styles.height = 0
            reason.styles.min_height = 0
            yield reason

    def compose(self) -> ComposeResult:
        yield Static(
            "Run Inspector",
            id="console-run-inspector-title",
            classes="destination-section",
        )
        rows_by_label = {row.label: (index, row) for index, row in enumerate(self.state.rows)}
        rendered_labels: set[str] = set()
        rendered_action_ids: set[str] = set()

        for heading, heading_id, labels in _ROW_GROUPS:
            yield Static(
                heading,
                id=heading_id,
                classes="console-inspector-group-heading destination-section",
            )
            for label in labels:
                row_entry = rows_by_label.get(label)
                if row_entry is None:
                    continue
                index, row = row_entry
                rendered_labels.add(label)
                yield Static(
                    row.text,
                    id=self._row_id(row, index),
                    classes=f"console-inspector-row console-inspector-row-{row.status}",
                )

            action_ids = _ACTION_GROUPS.get(heading, ())
            if action_ids:
                for action in self.state.actions:
                    if action.widget_id not in action_ids:
                        continue
                    rendered_action_ids.add(action.widget_id)
                    yield from self._compose_action(action)

        for index, row in enumerate(self.state.rows):
            if row.label in rendered_labels:
                continue
            yield Static(
                row.text,
                id=self._row_id(row, index),
                classes=f"console-inspector-row console-inspector-row-{row.status}",
            )
        for action in self.state.actions:
            if action.widget_id in rendered_action_ids:
                continue
            yield from self._compose_action(action)
