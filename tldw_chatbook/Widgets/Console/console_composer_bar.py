"""Console-native composer action row."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.widgets import Button, Static


class ConsoleComposerBar(Horizontal):
    """Expose Console-owned composer actions while reusing active chat sessions."""

    DEFAULT_STATUS = "Use the active session input below. Actions route through the active chat session."

    def sync_session_data(self, session_data: Any | None) -> None:
        """Refresh composer status copy from the active chat session contract."""
        if session_data is None:
            status = self.DEFAULT_STATUS
        else:
            title = getattr(session_data, "title", None) or "Untitled session"
            backend = getattr(session_data, "runtime_backend", None) or "local"
            assistant = getattr(session_data, "assistant_id", None) or getattr(
                session_data,
                "character_name",
                None,
            ) or "General"
            workspace = getattr(session_data, "workspace_id", None) or "global"
            status = (
                f"Active session: {title} | Backend: {backend} | "
                f"Assistant: {assistant} | Scope: {workspace}"
            )

        try:
            self.query_one("#console-composer-status", Static).update(status)
        except NoMatches:
            return

    def compose(self) -> ComposeResult:
        yield Static("Composer", id="console-composer-title", classes="destination-section")
        yield Static(
            self.DEFAULT_STATUS,
            id="console-composer-status",
            classes="console-composer-status",
            markup=False,
        )
        yield Button(
            "Send",
            id="console-send-message",
            classes="destination-action-button console-send-button",
            variant="primary",
            tooltip="Send the active Console session draft.",
        )
        yield Button(
            "Stop",
            id="console-stop-generation",
            classes="destination-action-button console-stop-button",
            tooltip="Stop generation in the active Console session.",
        )
        yield Button(
            "Attach",
            id="console-attach-context",
            classes="destination-action-button console-attach-button",
            tooltip="Attach files or context through the active Console session.",
        )
        yield Button(
            "Save Chatbook",
            id="console-save-chatbook",
            classes="destination-action-button console-save-chatbook-button",
            tooltip="Compatibility adapter: save Chatbook export is still owned by Artifacts/Chatbooks.",
        )
