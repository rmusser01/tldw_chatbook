"""Detail panel for selected Writing Suite outline nodes."""

from __future__ import annotations

from typing import Any, Mapping

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Label, Static, TextArea


class WritingDetailPanel(Vertical):
    """Read-only detail shell for the currently selected writing node."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.selected_node: dict[str, Any] | None = None
        self.title = "No selection"
        self.detail_text = "Select a project, manuscript, chapter, or scene."

    def compose(self) -> ComposeResult:
        yield Label("Writing Detail")
        yield Static(self.title, id="writing-detail-title")
        yield TextArea(
            self.detail_text,
            id="writing-detail-editor",
            read_only=True,
        )

    def clear(self) -> None:
        self.selected_node = None
        self.title = "No selection"
        self.detail_text = "Select a project, manuscript, chapter, or scene."
        self._refresh_mounted()

    def load_node(self, node_data: Mapping[str, Any]) -> None:
        self.selected_node = dict(node_data)
        self.title = str(node_data.get("title") or "Untitled")
        kind = str(node_data.get("kind") or "item")
        source = str(node_data.get("source") or "local")
        version = node_data.get("version")
        version_text = f"v{version}" if version is not None else "unversioned"
        self.detail_text = f"{kind} from {source} ({version_text})"
        self._refresh_mounted()

    def _refresh_mounted(self) -> None:
        if not self.is_mounted:
            return
        try:
            self.query_one("#writing-detail-title", Static).update(self.title)
        except Exception:
            pass
        try:
            self.query_one("#writing-detail-editor", TextArea).text = self.detail_text
        except Exception:
            pass
