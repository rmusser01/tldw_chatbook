"""Console-native composer action row."""

from __future__ import annotations

from typing import Any

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.widgets import Button, Static


class ConsoleComposerBar(Horizontal):
    """Expose Console-owned composer actions while reusing active chat sessions."""

    DEFAULT_STATUS = "Use the active session input below. Actions route through the active chat session."

    @staticmethod
    def _bounded_button(label: str, *, width: int, **kwargs: Any) -> Button:
        button = Button(label, **kwargs)
        button.styles.width = width
        button.styles.min_width = width
        return button

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
            self.query_one("#console-composer-status", Static).update(escape(status))
        except NoMatches:
            return

    def compose(self) -> ComposeResult:
        title = Static("Composer", id="console-composer-title", classes="destination-section")
        title.styles.width = 8
        title.styles.min_width = 8
        yield title
        status = Static(
            self.DEFAULT_STATUS,
            id="console-composer-status",
            classes="console-composer-status",
        )
        status.styles.width = 1
        status.styles.min_width = 0
        yield status
        yield self._bounded_button(
            "Send",
            width=8,
            id="console-send-message",
            classes="destination-action-button console-send-button",
            variant="primary",
            tooltip="Send the active Console session draft.",
        )
        yield self._bounded_button(
            "Stop",
            width=8,
            id="console-stop-generation",
            classes="destination-action-button console-stop-button",
            tooltip="Stop generation in the active Console session.",
        )
        yield self._bounded_button(
            "Attach",
            width=10,
            id="console-attach-context",
            classes="destination-action-button console-attach-button",
            tooltip="Attach files or context through the active Console session.",
        )
        yield self._bounded_button(
            "Save Chatbook",
            width=12,
            id="console-save-chatbook",
            classes="destination-action-button console-save-chatbook-button",
            tooltip="Compatibility adapter: save Chatbook export is still owned by Artifacts/Chatbooks.",
        )
