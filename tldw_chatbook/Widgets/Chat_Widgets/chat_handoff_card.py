"""Visible staged-context card for Chat handoffs."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container
from textual.message import Message
from textual.widgets import Button, Static

from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload


SOURCE_LABELS = {
    "notes": "Notes",
    "workspace": "Workspace",
    "media": "Media",
    "search-rag": "RAG Search",
    "search-web": "Web Search",
}


class ChatHandoffCard(Container):
    DEFAULT_CSS = """
    ChatHandoffCard {
        width: 100%;
        padding: 1;
        margin-bottom: 1;
        border: round $primary;
        background: $boost;
    }
    """

    class ClearRequested(Message):
        """Request that the owning chat session clear staged context."""

        def __init__(self, payload: ChatHandoffPayload) -> None:
            super().__init__()
            self.payload = payload

    def __init__(self, payload: ChatHandoffPayload, clear_action_id: str | None = None, **kwargs):
        super().__init__(**kwargs)
        normalized_payload = ChatHandoffPayload.from_dict(payload)
        if normalized_payload is None:
            raise ValueError("ChatHandoffCard requires a handoff payload")
        self.payload = normalized_payload
        self.clear_action_id = clear_action_id

    def render_text(self) -> str:
        status = "Context sent" if self.payload.status == "sent" else "Context staged"
        source_label = SOURCE_LABELS.get(
            self.payload.source,
            self.payload.source.replace("-", " ").title(),
        )
        summary = self.payload.display_summary or self.payload.body[:240]
        metadata = " | ".join(
            f"{key}: {value}"
            for key, value in sorted((self.payload.metadata or {}).items())
            if value not in (None, "")
        )
        parts = [
            f"{status} from {source_label}",
            f"Title: {self.payload.title}",
            f"Type: {self.payload.item_type}",
            f"Summary: {summary or 'No preview available.'}",
        ]
        if self.payload.runtime_backend:
            parts.append(f"Backend: {self.payload.runtime_backend}")
        if self.payload.source_owner or self.payload.source_selector_state:
            parts.append(f"Source: {self._source_chip_label()}")
        if self.payload.active_server_profile_id:
            parts.append(f"Server: {self.payload.active_server_profile_id}")
        if self.payload.workspace_id:
            parts.append(f"Workspace: {self.payload.workspace_id}")
        if self.payload.sync_dry_run_report:
            parts.append("Sync: dry-run only")
            parts.extend(self._sync_dry_run_labels())
        if self.payload.body_truncated:
            parts.append("Content: preview truncated")
        if self.payload.unsupported_reports:
            parts.append(f"Unsupported actions: {len(self.payload.unsupported_reports)}")
            parts.extend(self._unsupported_report_labels())
        if metadata:
            parts.append(metadata)
        parts.append("Review the draft below and send when ready.")
        return "\n".join(parts)

    def compose(self) -> ComposeResult:
        yield Static(self.render_text(), classes="chat-handoff-card-body")
        if self.payload.status != "sent" and self.clear_action_id:
            yield Button(
                "Clear staged context",
                id=self.clear_action_id,
                classes="chat-handoff-clear-button",
                tooltip="Remove this context from the next message",
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self.clear_action_id and event.button.id == self.clear_action_id:
            event.stop()
            self.post_message(self.ClearRequested(self.payload))

    def _source_chip_label(self) -> str:
        state = self.payload.source_selector_state or self.payload.source_owner
        labels = {
            "local": "Local source",
            "server": "Server source",
            "workspace": "Workspace source",
            "shared": "Shared source",
        }
        return labels.get(str(state), str(state).replace("_", " ").title())

    def _sync_dry_run_labels(self) -> list[str]:
        report = self.payload.sync_dry_run_report or {}
        labels: list[str] = []
        if report.get("write_enabled") is False:
            labels.append("Write sync is not enabled.")
        user_message = report.get("user_message")
        if user_message:
            labels.append(str(user_message))
        return labels

    def _unsupported_report_labels(self) -> list[str]:
        labels: list[str] = []
        for report in self.payload.unsupported_reports[:3]:
            user_message = None
            if isinstance(report, dict):
                user_message = report.get("user_message") or report.get("unsupported_user_message")
            if user_message:
                labels.append(str(user_message))
        return labels
