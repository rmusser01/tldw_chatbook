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
            self.state.workspace_label,
            id="console-active-workspace",
            classes="console-workspace-status-row",
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
        with Vertical(id="console-workspace-conversations"):
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

        if self.state.change_workspace_enabled:
            yield Button(
                "Change workspace",
                id="console-change-workspace",
                classes="console-workspace-action",
            )
            if self.state.change_workspace_recovery:
                yield self._static(
                    self.state.change_workspace_recovery,
                    id="console-change-workspace-recovery",
                    classes="console-workspace-recovery",
                )

        if self.state.new_conversation_enabled:
            yield Button(
                "New conversation",
                id="console-new-workspace-conversation",
                classes="console-workspace-action",
            )
            if self.state.new_conversation_recovery:
                yield self._static(
                    self.state.new_conversation_recovery,
                    id="console-new-workspace-conversation-recovery",
                    classes="console-workspace-recovery",
                )
