"""Console-native workspace context tray."""

from __future__ import annotations

import re
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Workspaces.display_state import ConsoleWorkspaceContextState


_TRAILING_SHORT_ID_RE = re.compile(r"\s+\[[0-9a-fA-F]{8}\]$")
_STATUS_LABELS = {
    "workspace-thread": "workspace",
    "workspace": "workspace",
    "active": "active",
    "open": "open",
}
_STATUS_DETAIL_LABELS = {
    "workspace-thread": "saved workspace",
    "workspace": "saved workspace",
    "active": "active session",
    "open": "open session",
}
_MAX_CONVERSATION_ROW_TITLE = 20


class ConsoleWorkspaceStatusPair(Horizontal):
    """A structured label/value row for workspace authority metadata."""

    def __init__(
        self,
        label: str,
        value: str,
        *,
        label_id: str,
        value_id: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(classes="console-workspace-status-pair", **kwargs)
        self.label = label
        self.value = value
        self.label_id = label_id
        self.value_id = value_id
        self.styles.height = "auto"
        self.styles.min_height = 1

    def compose(self) -> ComposeResult:
        """Render the pair as queryable Textual widgets."""
        label_widget = Static(
            self.label,
            id=self.label_id,
            classes="console-workspace-status-label",
            markup=False,
        )
        label_widget.styles.width = 10
        label_widget.styles.min_width = 10
        yield label_widget

        value_widget = Static(
            self.value,
            id=self.value_id,
            classes="console-workspace-status-value",
            markup=False,
        )
        value_widget.styles.width = "1fr"
        value_widget.styles.min_width = 0
        yield value_widget


class ConsoleWorkspaceContextTray(Vertical):
    """Render workspace selection, conversation scope, and recovery copy."""

    def __init__(self, state: ConsoleWorkspaceContextState, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.state = state

    def sync_state(self, state: ConsoleWorkspaceContextState) -> None:
        """Refresh the mounted workspace context tray from new display state."""
        self.state = state
        self.refresh(recompose=True)

    @staticmethod
    def _static(text: str, *, id: str, classes: str = "") -> Static:
        return Static(str(text), id=id, classes=classes, markup=False)

    @staticmethod
    def _split_status_row(text: str, fallback_label: str) -> tuple[str, str]:
        """Return a scannable label/value pair from legacy status copy."""
        raw = str(text or "").strip()
        label, separator, value = raw.partition(":")
        if separator:
            clean_label = label.strip()
            clean_value = value.strip()
            if clean_label and clean_value:
                return clean_label, clean_value
            if clean_label:
                return clean_label, "unavailable"
        return fallback_label, raw or "unavailable"

    def _status_pair(
        self,
        text: str,
        *,
        label_id: str,
        value_id: str,
        fallback_label: str,
    ) -> ComposeResult:
        """Build a two-column status row for scan-heavy workspace metadata."""
        label, value = self._split_status_row(text, fallback_label)
        if fallback_label == "Handoff" and label != fallback_label:
            value = f"{label}: {value}"
            label = fallback_label
        yield ConsoleWorkspaceStatusPair(
            label,
            value,
            label_id=label_id,
            value_id=value_id,
        )

    @staticmethod
    def _conversation_button(
        text: str,
        *,
        id: str,
        conversation_id: str,
        tooltip_label: str | None = None,
        selected: bool = False,
    ) -> Button:
        button = Button(
            Text(str(text)),
            id=id,
            classes="console-workspace-conversation-row",
            compact=True,
        )
        button.conversation_id = conversation_id
        button.tooltip = f"Switch to {tooltip_label or text.lstrip('> ').strip()}"
        button.set_class(selected, "console-workspace-conversation-row-selected")
        button.styles.height = 2
        button.styles.min_height = 2
        return button

    def compose(self) -> ComposeResult:
        yield self._static(
            self.state.heading,
            id="console-workspace-context-title",
            classes="destination-section",
        )
        yield self._static(
            self._workspace_selector_label(),
            id="console-active-workspace",
            classes="console-workspace-status-row console-workspace-selector-row",
        )
        if self.state.change_workspace_enabled:
            yield Button(
                "Change workspace",
                id="console-change-workspace",
                classes="console-workspace-action",
                compact=True,
            )
            if self.state.change_workspace_recovery:
                yield self._static(
                    self.state.change_workspace_recovery,
                    id="console-change-workspace-recovery",
                    classes="console-workspace-recovery",
                )
        if self.state.recovery_copy:
            yield self._static(
                self.state.recovery_copy,
                id="console-workspace-recovery",
                classes="console-workspace-recovery",
            )
        yield self._static(
            "Conversations",
            id="console-workspace-conversations-title",
            classes="destination-section",
        )
        conversation_count = max(1, len(self.state.conversation_rows) * 3)
        conversation_list = Vertical(id="console-workspace-conversations")
        conversation_list.styles.height = conversation_count
        conversation_list.styles.min_height = conversation_count
        with conversation_list:
            if self.state.conversation_rows:
                for index, row in enumerate(self.state.conversation_rows):
                    marker = "> " if row.selected else "  "
                    title = self._conversation_title(row.title)
                    visible_title = self._conversation_visible_title(title)
                    status = self._conversation_status(row.status)
                    detail = self._conversation_detail_status(row.status)
                    status_suffix = f" [{status}]" if status else ""
                    secondary = detail or "conversation"
                    yield self._conversation_button(
                        f"{marker}{visible_title}\n  {secondary}",
                        id=f"console-workspace-conversation-{index}",
                        conversation_id=row.conversation_id,
                        tooltip_label=f"{title}{status_suffix}",
                        selected=row.selected,
                    )
            else:
                yield self._static(
                    self.state.conversation_empty_copy,
                    id="console-workspace-empty-conversations",
                    classes="console-workspace-empty-copy",
                )

        if self.state.new_conversation_enabled:
            yield Button(
                "New conversation",
                id="console-new-workspace-conversation",
                classes="console-workspace-action",
                compact=True,
            )
            if self.state.new_conversation_recovery:
                yield self._static(
                    self.state.new_conversation_recovery,
                    id="console-new-workspace-conversation-recovery",
                    classes="console-workspace-recovery",
                )
        yield from self._status_pair(
            self.state.authority_label,
            label_id="console-workspace-authority-label",
            value_id="console-workspace-authority-value",
            fallback_label="Authority",
        )
        yield from self._status_pair(
            self.state.sync_label,
            label_id="console-workspace-sync-label",
            value_id="console-workspace-sync-value",
            fallback_label="Sync",
        )
        yield from self._status_pair(
            self.state.runtime_label,
            label_id="console-workspace-runtime-label",
            value_id="console-workspace-runtime-value",
            fallback_label="Runtime",
        )
        yield from self._status_pair(
            self.state.server_readiness_label,
            label_id="console-workspace-server-readiness-label",
            value_id="console-workspace-server-readiness-value",
            fallback_label="Server",
        )
        yield self._static(
            self.state.server_readiness_detail,
            id="console-workspace-server-readiness-detail",
            classes="console-workspace-recovery",
        )
        yield self._static(
            "Handoff readiness",
            id="console-workspace-handoff-title",
            classes="destination-section",
        )
        with Vertical(id="console-workspace-handoff-rows"):
            if self.state.handoff_rows:
                for index, row in enumerate(self.state.handoff_rows):
                    portability = "" if row.portable else " (not portable)"
                    yield self._static(
                        f"{row.title} - {row.transfer_policy.value}{portability}",
                        id=f"console-workspace-handoff-{index}",
                        classes="console-workspace-status-row",
                    )
            else:
                yield self._static(
                    "No workspace items ready for handoff preflight.",
                    id="console-workspace-handoff-empty",
                    classes="console-workspace-empty-copy",
                )
        yield from self._status_pair(
            self.state.acp_handoff_label,
            label_id="console-workspace-handoff-label",
            value_id="console-workspace-handoff-value",
            fallback_label="Handoff",
        )
        yield self._static(
            self.state.acp_handoff_detail,
            id="console-workspace-acp-handoff-detail",
            classes="console-workspace-recovery",
        )
        yield self._static(
            self.state.acp_handoff_audit,
            id="console-workspace-acp-handoff-audit",
            classes="console-workspace-recovery",
        )

    def _workspace_selector_label(self) -> str:
        """Return the visible active-workspace selector affordance."""
        workspace_label = self.state.workspace_label
        if workspace_label.startswith("Workspace: "):
            workspace_label = workspace_label.removeprefix("Workspace: ").strip()
        return workspace_label

    @staticmethod
    def _conversation_title(title: str) -> str:
        """Return a readable conversation label without raw disambiguation IDs."""
        return _TRAILING_SHORT_ID_RE.sub("", str(title).strip()) or "Untitled conversation"

    @staticmethod
    def _conversation_visible_title(title: str) -> str:
        """Return a rail-safe visible title that does not clip in narrow panes."""
        readable = ConsoleWorkspaceContextTray._conversation_title(title)
        if len(readable) <= _MAX_CONVERSATION_ROW_TITLE:
            return readable
        return f"{readable[: _MAX_CONVERSATION_ROW_TITLE - 3].rstrip()}..."

    @staticmethod
    def _conversation_status(status: str) -> str:
        """Return a short user-facing conversation status badge."""
        normalized = str(status or "").strip().lower()
        if not normalized:
            return ""
        return _STATUS_LABELS.get(normalized, normalized.replace("-", " "))

    @staticmethod
    def _conversation_detail_status(status: str) -> str:
        """Return second-line row metadata for row disambiguation."""
        normalized = str(status or "").strip().lower()
        if not normalized:
            return ""
        return _STATUS_DETAIL_LABELS.get(normalized, normalized.replace("-", " "))
