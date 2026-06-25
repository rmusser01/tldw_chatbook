"""Console-native workspace context tray."""

from __future__ import annotations

import re
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Static

from tldw_chatbook.Workspaces.display_state import (
    CONSOLE_WORKSPACE_CONVERSATION_ROW_HEIGHT,
    ConsoleWorkspaceContextState,
    ConsoleWorkspaceConversationSectionState,
    console_workspace_conversation_visible_rows,
)


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
_AUTHORITY_LABELS = {
    "local registry ready": "local",
    "local-only": "local",
    "server-backed": "server backed",
    "syncing-to-server": "syncing to server",
    "syncing-from-server": "syncing from server",
    "conflict": "conflict",
    "detached": "detached",
    "remote-only": "remote only",
    "runtime-missing": "runtime missing",
}
_MAX_CONVERSATION_ROW_TITLE = 20
_CONVERSATION_SECTION_FIXED_ROWS = 7
_CONVERSATION_SECTION_STATUS_RESERVE_ROWS = 3
_CONVERSATION_SECTION_CHROME_HEIGHT = (
    _CONVERSATION_SECTION_FIXED_ROWS + _CONVERSATION_SECTION_STATUS_RESERVE_ROWS
)


class ConsoleWorkspaceStatusPair(Horizontal):
    """Render workspace authority metadata as a structured status row.

    Attributes:
        label: User-facing row label.
        value: User-facing row value.
        label_id: Textual widget id for the label cell.
        value_id: Textual widget id for the value cell.
    """

    def __init__(
        self,
        label: str,
        value: str,
        *,
        label_id: str,
        value_id: str,
        **kwargs: Any,
    ) -> None:
        """Initialize the label/value status row.

        Args:
            label: User-facing row label.
            value: User-facing row value.
            label_id: Textual widget id for the label cell.
            value_id: Textual widget id for the value cell.
            **kwargs: Additional keyword arguments passed to ``Horizontal``.
        """
        super().__init__(classes="console-workspace-status-pair", **kwargs)
        self.label = label
        self.value = value
        self.label_id = label_id
        self.value_id = value_id
        self.styles.height = "auto"
        self.styles.min_height = 1

    def compose(self) -> ComposeResult:
        """Render the row as queryable Textual widgets.

        Returns:
            ComposeResult containing the label and value widgets.
        """
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


class ConsoleWorkspaceContextTray(VerticalScroll):
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

    def _conversation_section(self) -> ConsoleWorkspaceConversationSectionState:
        """Return section state, adapting legacy row-only snapshots."""
        section = self.state.conversation_section
        if section is not None:
            return section
        selected = next(
            (row for row in self.state.conversation_rows if row.selected),
            None,
        )
        selected_summary = (
            (
                f"{self._conversation_title(selected.title)} - "
                f"{self._conversation_detail_status(selected.status) or 'conversation'}"
            )
            if selected is not None
            else "No active conversation."
        )
        return ConsoleWorkspaceConversationSectionState(
            workspace_id="",
            collapsed=False,
            query="",
            selected_summary=selected_summary,
            rows=self.state.conversation_rows,
            workspace_total_count=len(self.state.conversation_rows),
            result_total_count=None,
            status_copy="",
            empty_copy=self.state.conversation_empty_copy,
            search_enabled=False,
            new_conversation_enabled=self.state.new_conversation_enabled,
        )

    @staticmethod
    def _conversation_count_title(
        section: ConsoleWorkspaceConversationSectionState,
    ) -> str:
        """Return the Conversations heading with a stable workspace count."""
        count = section.workspace_total_count
        if count is None:
            count = len(section.rows)
        return f"Conversations ({count})"

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

    def _visible_conversation_rows(self) -> int:
        """Return the conversation rows that fit inside the mounted tray."""

        parent_region = getattr(getattr(self, "parent", None), "region", None)
        base_rows = console_workspace_conversation_visible_rows(
            parent_region.height if parent_region is not None else None
        )
        tray_height = int(getattr(getattr(self, "region", None), "height", 0) or 0)
        if tray_height <= 0:
            return base_rows

        available_list_height = max(
            1,
            tray_height - _CONVERSATION_SECTION_CHROME_HEIGHT,
        )
        fitted_rows = max(
            1,
            available_list_height // CONSOLE_WORKSPACE_CONVERSATION_ROW_HEIGHT,
        )
        return min(base_rows, fitted_rows)

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

        section = self._conversation_section()
        section_controls_enabled = self.state.conversation_section is not None
        with Horizontal(
            id="console-workspace-conversations-header",
            classes="console-workspace-conversations-header",
        ):
            title = self._static(
                self._conversation_count_title(section),
                id="console-workspace-conversations-title",
                classes="destination-section",
            )
            title.styles.width = "1fr"
            yield title
            toggle_label = "+" if section.collapsed else "-"
            toggle = Button(
                toggle_label,
                id="console-workspace-conversations-toggle",
                classes=(
                    "console-workspace-action "
                    "console-workspace-conversations-toggle"
                ),
                compact=True,
                disabled=not section_controls_enabled,
            )
            toggle.tooltip = (
                "Expand Conversations"
                if section.collapsed
                else "Collapse Conversations"
            )
            toggle.styles.width = 3
            toggle.styles.min_width = 3
            yield toggle
        yield self._static(
            section.selected_summary or "No active conversation.",
            id="console-workspace-selected-conversation",
            classes="console-workspace-selected-conversation",
        )
        if not section.collapsed:
            with Horizontal(
                id="console-workspace-conversation-search-row",
                classes="console-workspace-conversation-search-row",
            ):
                search_input = Input(
                    value=section.query,
                    placeholder="Search workspace conversations",
                    id="console-workspace-conversation-search",
                    classes="console-workspace-conversation-search",
                    disabled=not section.search_enabled,
                )
                search_input.styles.width = "1fr"
                yield search_input
                clear_button = Button(
                    "Clear",
                    id="console-workspace-conversation-search-clear",
                    classes=(
                        "console-workspace-action "
                        "console-workspace-conversation-search-clear"
                    ),
                    compact=True,
                    disabled=(
                        not section.search_enabled
                        or not bool(str(section.query or "").strip())
                    ),
                )
                clear_button.tooltip = "Clear conversation search"
                yield clear_button
            if section.status_copy:
                yield self._static(
                    section.status_copy,
                    id="console-workspace-conversation-search-status",
                    classes="console-workspace-empty-copy",
                )
            if section.error_copy:
                yield self._static(
                    section.error_copy,
                    id="console-workspace-conversation-search-error",
                    classes="console-workspace-recovery",
                )
            visible_rows = self._visible_conversation_rows()
            conversation_list = VerticalScroll(id="console-workspace-conversations")
            conversation_list.styles.height = max(1, visible_rows * 3)
            conversation_list.styles.min_height = max(1, visible_rows * 3)
            with conversation_list:
                if section.rows:
                    for index, row in enumerate(section.rows):
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
                        section.empty_copy or self.state.conversation_empty_copy,
                        id="console-workspace-empty-conversations",
                        classes="console-workspace-empty-copy",
                    )
            if section.new_conversation_enabled:
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
            self._friendly_status_label(self.state.authority_label),
            label_id="console-workspace-authority-label",
            value_id="console-workspace-authority-value",
            fallback_label="Storage",
        )
        yield from self._status_pair(
            self._friendly_status_label(self.state.sync_label),
            label_id="console-workspace-sync-label",
            value_id="console-workspace-sync-value",
            fallback_label="Sync",
        )
        yield from self._status_pair(
            self._friendly_status_label(self.state.runtime_label),
            label_id="console-workspace-runtime-label",
            value_id="console-workspace-runtime-value",
            fallback_label="File tools",
        )
        yield from self._status_pair(
            self._friendly_status_label(self.state.server_readiness_label),
            label_id="console-workspace-server-readiness-label",
            value_id="console-workspace-server-readiness-value",
            fallback_label="Server handoff",
        )
        yield self._static(
            self._friendly_detail_copy(self.state.server_readiness_detail),
            id="console-workspace-server-readiness-detail",
            classes="console-workspace-recovery",
        )
        yield self._static(
            "Handoff",
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
                    "No handoff package is ready.",
                    id="console-workspace-handoff-empty",
                    classes="console-workspace-empty-copy",
                )
        yield from self._status_pair(
            self._friendly_status_label(self.state.acp_handoff_label),
            label_id="console-workspace-handoff-label",
            value_id="console-workspace-handoff-value",
            fallback_label="Handoff",
        )
        yield self._static(
            self._friendly_detail_copy(self.state.acp_handoff_detail),
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

    @staticmethod
    def _friendly_status_label(label: str) -> str:
        """Return user-facing workspace status copy for the Console rail."""
        raw = str(label or "").strip()
        normalized = raw.lower()
        if normalized.startswith("authority: unavailable"):
            return "Storage: Unavailable"
        if normalized.startswith("authority:"):
            authority = normalized.partition(":")[2].strip()
            readable = _AUTHORITY_LABELS.get(authority, authority.replace("-", " "))
            return f"Storage: {readable or 'unavailable'}"
        if normalized == "sync: not configured":
            return "Sync: Off"
        if normalized.startswith("runtime: none, file tools disabled"):
            return "File tools: Off in Default workspace"
        if normalized.startswith("runtime: none"):
            return "File tools: Off"
        if normalized.startswith("runtime:"):
            readiness = re.search(r"(\d+) ready(?:,\s+(\d+) missing)?", normalized)
            if readiness:
                label = f"File tools: {readiness.group(1)} ready"
                if readiness.group(2):
                    label = f"{label}, {readiness.group(2)} missing"
                return label
            return raw.replace("Runtime:", "File tools:", 1)
        if normalized == "server: local fallback":
            return "Server handoff: Not configured"
        if normalized.startswith("server: unavailable"):
            return "Server handoff: Unavailable"
        if normalized.startswith("server:"):
            return raw.replace("Server:", "Server handoff:", 1)
        if normalized.startswith("acp task/run: unavailable"):
            return "ACP handoff: Not configured"
        if normalized.startswith("acp task/run:"):
            return raw.replace("ACP task/run:", "ACP handoff:", 1)
        return raw

    @staticmethod
    def _friendly_detail_copy(copy: str) -> str:
        """Return first-run readable detail while preserving diagnostic intent."""
        raw = str(copy or "").strip()
        normalized = raw.lower()
        if (
            "local registry fallback is active" in normalized
            or "local registry is authoritative" in normalized
        ):
            return "Chats stay local. Connect a server later for explicit handoff."
        if "acp task/run package handoff is not wired" in normalized:
            return "ACP task/run package handoff is not configured yet."
        return raw
