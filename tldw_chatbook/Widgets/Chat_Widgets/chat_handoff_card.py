"""Visible staged-context card for Chat handoffs."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

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

    def __init__(self, payload: ChatHandoffPayload, **kwargs):
        super().__init__(**kwargs)
        normalized_payload = ChatHandoffPayload.from_dict(payload)
        if normalized_payload is None:
            raise ValueError("ChatHandoffCard requires a handoff payload")
        self.payload = normalized_payload

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
        if self.payload.body_truncated:
            parts.append("Content: preview truncated")
        if self.payload.unsupported_reports:
            parts.append(f"Unsupported actions: {len(self.payload.unsupported_reports)}")
        if metadata:
            parts.append(metadata)
        parts.append("Review the draft below and send when ready.")
        return "\n".join(parts)

    def compose(self) -> ComposeResult:
        yield Static(self.render_text(), classes="chat-handoff-card-body")

    def _source_chip_label(self) -> str:
        state = self.payload.source_selector_state or self.payload.source_owner
        labels = {
            "local": "Local source",
            "server": "Server source",
            "workspace": "Workspace source",
            "shared": "Shared source",
        }
        return labels.get(str(state), str(state).replace("_", " ").title())
