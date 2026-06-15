"""Console-native workspace context tray."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Workspaces.display_state import ConsoleWorkspaceContextState


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
        conversation_count = max(1, len(self.state.conversation_rows))
        conversation_list = Vertical(id="console-workspace-conversations")
        conversation_list.styles.height = conversation_count
        conversation_list.styles.min_height = conversation_count
        with conversation_list:
            if self.state.conversation_rows:
                for index, row in enumerate(self.state.conversation_rows):
                    marker = "> " if row.selected else "  "
                    status = f" [{row.status}]" if row.status else ""
                    yield self._static(
                        f"{marker}{row.title}{status}",
                        id=f"console-workspace-conversation-{index}",
                        classes="console-workspace-conversation-row",
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
        yield self._static(
            self.state.authority_label,
            id="console-workspace-authority",
            classes="console-workspace-status-row",
        )
        yield self._static(
            self.state.sync_label,
            id="console-workspace-sync",
            classes="console-workspace-status-row",
        )
        yield self._static(
            self.state.runtime_label,
            id="console-workspace-runtime",
            classes="console-workspace-status-row",
        )
        yield self._static(
            self.state.server_readiness_label,
            id="console-workspace-server-readiness",
            classes="console-workspace-status-row",
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
        yield self._static(
            self.state.acp_handoff_label,
            id="console-workspace-acp-handoff",
            classes="console-workspace-status-row",
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
