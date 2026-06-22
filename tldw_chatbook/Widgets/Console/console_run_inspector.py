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
    "Setup": "console-inspector-setup",
    "Send blocked": "console-inspector-send-blocked",
    "Recovery action": "console-inspector-recovery-action",
    "Provider": "console-inspector-provider",
    "Tools": "console-inspector-tools",
    "RAG/source": "console-inspector-rag-source",
    "Evidence": "console-inspector-evidence",
    "Authority": "console-inspector-authority",
    "Artifacts": "console-inspector-artifacts",
    "Approvals": "console-inspector-approvals",
    "Selected message": "console-inspector-selected-message",
    "Selected conversation": "console-inspector-selected-conversation",
    "Conversation source": "console-inspector-conversation-source",
    "Workspace": "console-inspector-workspace",
    "Resume state": "console-inspector-resume-state",
    "Session provider": "console-inspector-session-provider",
    "Session model": "console-inspector-session-model",
    "Session endpoint": "console-inspector-session-endpoint",
    "Session sampling": "console-inspector-session-sampling",
    "Session persona": "console-inspector-session-persona",
    "Message actions": "console-inspector-message-actions",
    "Keyboard": "console-inspector-message-keyboard",
    "Variants": "console-inspector-message-variants",
    "Excerpt": "console-inspector-message-excerpt",
    "Delete confirmation": "console-inspector-delete-confirmation",
}

_ROW_GROUPS = (
    (
        "Selected Conversation",
        "console-inspector-selected-conversation-heading",
        ("Selected conversation", "Conversation source", "Workspace", "Resume state"),
    ),
    (
        "Session Defaults",
        "console-inspector-session-defaults-heading",
        (
            "Session provider",
            "Session model",
            "Session endpoint",
            "Session sampling",
            "Session persona",
        ),
    ),
    (
        "Run State",
        "console-inspector-run-state-heading",
        ("Live work", "Setup", "Send blocked", "Recovery action", "Provider"),
    ),
    (
        "Source Readiness",
        "console-inspector-source-readiness-heading",
        ("RAG/source", "Evidence", "Authority", "Artifacts"),
    ),
    (
        "Tools",
        "console-inspector-tools-heading",
        ("Tools",),
    ),
    (
        "Approvals",
        "console-inspector-approvals-heading",
        ("Approvals",),
    ),
    (
        "Selected Message",
        "console-inspector-selected-message-heading",
        ("Selected message", "Message actions", "Keyboard", "Variants", "Excerpt", "Delete confirmation"),
    ),
)

_ACTION_GROUPS = {
    "Source Readiness": (CONSOLE_INSPECTOR_SAVE_CHATBOOK_ID,),
    "Tools": (CONSOLE_INSPECTOR_REVIEW_TOOL_CALL_ID,),
    "Approvals": (CONSOLE_INSPECTOR_REVIEW_APPROVAL_ID,),
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

    def _status_summary(self) -> str:
        """Return the primary run-inspector state in one scannable row."""
        rows = {row.label: row for row in self.state.rows}
        provider = rows.get("Provider")
        approvals = rows.get("Approvals")
        rag_source = rows.get("RAG/source")
        if provider is not None and provider.status == "blocked":
            return "Status: Blocked"
        if approvals is not None and approvals.status == "blocked":
            return "Status: Needs approval"
        if rag_source is not None and rag_source.status == "blocked":
            return "Status: Source blocked"
        return "Status: Ready"

    def compose(self) -> ComposeResult:
        yield Static(
            self._status_summary(),
            id="console-inspector-run-status-summary",
            classes="console-inspector-status-summary",
        )
        rows_by_label = {row.label: (index, row) for index, row in enumerate(self.state.rows)}
        rendered_labels: set[str] = set()
        rendered_action_ids: set[str] = set()

        for heading, heading_id, labels in _ROW_GROUPS:
            group_labels = [label for label in labels if label in rows_by_label]
            action_ids = _ACTION_GROUPS.get(heading, ())
            group_actions = [
                action
                for action in self.state.actions
                if action.widget_id in action_ids
            ]
            if not group_labels and not group_actions:
                continue

            yield Static(
                heading,
                id=heading_id,
                classes="console-inspector-group-heading destination-section",
            )
            for label in group_labels:
                row_entry = rows_by_label[label]
                index, row = row_entry
                rendered_labels.add(label)
                yield Static(
                    row.text,
                    id=self._row_id(row, index),
                    classes=f"console-inspector-row console-inspector-row-{row.status}",
                    markup=False,
                )

            for action in group_actions:
                rendered_action_ids.add(action.widget_id)
                yield from self._compose_action(action)

        for index, row in enumerate(self.state.rows):
            if row.label in rendered_labels:
                continue
            yield Static(
                row.text,
                id=self._row_id(row, index),
                classes=f"console-inspector-row console-inspector-row-{row.status}",
                markup=False,
            )
        for action in self.state.actions:
            if action.widget_id in rendered_action_ids:
                continue
            yield from self._compose_action(action)
