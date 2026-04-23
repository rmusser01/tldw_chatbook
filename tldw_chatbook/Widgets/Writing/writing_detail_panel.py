"""Detail panel for selected Writing Suite outline nodes."""

from __future__ import annotations

from typing import Any, Mapping

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Label, Static, TextArea


class WritingDetailPanel(Vertical):
    """Read-only detail shell for the currently selected writing node."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.selected_node: dict[str, Any] | None = None
        self.entity: Any | None = None
        self.title = "No selection"
        self.detail_text = "Select a project, manuscript, chapter, or scene."
        self.body_editor_enabled = False
        self.create_version_enabled = False
        self.version_list_read_only = True
        self.versions: list[Any] = []
        self.version_labels: list[str] = []
        self.selected_version_id: str | None = None
        self.version_preview_text = ""
        self.trash_entries: list[Any] = []
        self.trash_labels: list[str] = []
        self.unsupported_reasons: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Label("Writing Detail")
        yield Static(self.title, id="writing-detail-title")
        yield TextArea(
            self.detail_text,
            id="writing-detail-editor",
            read_only=True,
        )
        with Horizontal(classes="writing-detail-actions"):
            yield Button("Save", id="writing-save-current", disabled=True)
            yield Button("Delete", id="writing-delete-current", disabled=True)
            yield Button("Create New Version", id="writing-create-version", disabled=True)
            yield Button("Restore To Working Draft", id="writing-restore-version", disabled=True)
        yield Static("", id="writing-version-list")
        yield Static("", id="writing-version-preview")
        yield Static("", id="writing-trash-list")

    def clear(self) -> None:
        self.selected_node = None
        self.title = "No selection"
        self.detail_text = "Select a project, manuscript, chapter, or scene."
        self._clear_editable_state()
        self._refresh_mounted()

    def load_node(self, node_data: Mapping[str, Any]) -> None:
        self.selected_node = dict(node_data)
        self._clear_editable_state()
        self.title = str(node_data.get("title") or "Untitled")
        kind = str(node_data.get("kind") or "item")
        source = str(node_data.get("source") or "local")
        version = node_data.get("version")
        version_text = f"v{version}" if version is not None else "unversioned"
        self.detail_text = f"{kind} from {source} ({version_text})"
        self._refresh_mounted()

    def load_entity(self, node_data: Mapping[str, Any], entity: Any) -> None:
        self.selected_node = dict(node_data)
        self.entity = entity
        self.unsupported_reasons = {}
        kind = str(node_data.get("kind") or "")
        source = str(node_data.get("source") or "local")
        self.title = self._record_title(entity, fallback=str(node_data.get("title") or "Untitled"))
        self.body_editor_enabled = kind == "scene"
        self.create_version_enabled = source == "local" and kind in {"manuscript", "chapter", "scene"}
        self.detail_text = (
            str(self._record_get(entity, "body_markdown", "") or "")
            if self.body_editor_enabled
            else self._metadata_preview(kind, entity)
        )
        self._refresh_mounted()

    def set_versions(self, versions: list[Any]) -> None:
        self.versions = list(versions or [])
        self.version_labels = [
            f"v{self._record_get(version, 'version_number', index + 1)}"
            for index, version in enumerate(self.versions)
        ]
        self.version_list_read_only = True
        self.selected_version_id = (
            str(self._record_get(self.versions[0], "id"))
            if self.versions
            else None
        )
        self.version_preview_text = (
            self._version_preview(self.versions[0])
            if self.versions
            else ""
        )
        self._refresh_mounted()

    def set_trash(self, entries: list[Any]) -> None:
        self.trash_entries = list(entries or [])
        self.trash_labels = [
            f"{self._record_get(entry, 'entity_kind', 'item')}: "
            f"{self._record_title(entry, fallback='Untitled')}"
            for entry in self.trash_entries
        ]
        self._refresh_mounted()

    def set_unsupported_reason(self, action: str, reason: str | None) -> None:
        if reason:
            self.unsupported_reasons[action] = reason
        else:
            self.unsupported_reasons.pop(action, None)
        self._refresh_mounted()

    def _refresh_mounted(self) -> None:
        if not self.is_mounted:
            return
        try:
            self.query_one("#writing-detail-title", Static).update(self.title)
        except Exception:
            pass
        try:
            editor = self.query_one("#writing-detail-editor", TextArea)
            editor.read_only = not self.body_editor_enabled
            editor.text = self.detail_text
        except Exception:
            pass
        try:
            self.query_one("#writing-save-current", Button).disabled = self.entity is None
            self.query_one("#writing-delete-current", Button).disabled = self.entity is None
            self.query_one("#writing-create-version", Button).disabled = not self.create_version_enabled
            self.query_one("#writing-restore-version", Button).disabled = not bool(self.selected_version_id)
        except Exception:
            pass
        try:
            self.query_one("#writing-version-list", Static).update("\n".join(self.version_labels))
            self.query_one("#writing-version-preview", Static).update(self.version_preview_text)
            self.query_one("#writing-trash-list", Static).update("\n".join(self.trash_labels))
        except Exception:
            pass

    def _clear_editable_state(self) -> None:
        self.entity = None
        self.body_editor_enabled = False
        self.create_version_enabled = False
        self.versions = []
        self.version_labels = []
        self.selected_version_id = None
        self.version_preview_text = ""
        self.trash_entries = []
        self.trash_labels = []
        self.unsupported_reasons = {}

    def current_payload(self) -> dict[str, Any]:
        if self.selected_node is None:
            return {}
        kind = str(self.selected_node.get("kind") or "")
        if kind == "scene":
            return {"body_markdown": self.detail_text}
        return self._metadata_payload(self.entity)

    @classmethod
    def _metadata_preview(cls, kind: str, entity: Any) -> str:
        fields = ["title", "status", "synopsis", "word_count"]
        if kind == "project":
            fields.extend(["subtitle", "author", "genre", "target_word_count"])
        elif kind in {"manuscript", "chapter", "scene"}:
            fields.append("sort_order")
        lines = []
        for field in fields:
            value = cls._record_get(entity, field)
            if value not in {None, ""}:
                lines.append(f"{field}: {value}")
        return "\n".join(lines)

    @classmethod
    def _metadata_payload(cls, entity: Any) -> dict[str, Any]:
        if entity is None:
            return {}
        payload = {}
        for field in (
            "title",
            "subtitle",
            "author",
            "genre",
            "status",
            "synopsis",
            "target_word_count",
            "word_count",
            "sort_order",
        ):
            value = cls._record_get(entity, field)
            if value is not None:
                payload[field] = value
        return payload

    @classmethod
    def _version_preview(cls, version: Any) -> str:
        body = cls._record_get(version, "body_markdown")
        if body:
            return str(body)
        metadata = cls._record_get(version, "metadata", {}) or {}
        if isinstance(metadata, Mapping):
            if metadata.get("assembled_markdown"):
                return str(metadata["assembled_markdown"])
            return "\n".join(f"{key}: {value}" for key, value in metadata.items())
        return str(metadata)

    @staticmethod
    def _record_get(record: Any, key: str, default: Any = None) -> Any:
        if isinstance(record, Mapping):
            return record.get(key, default)
        return getattr(record, key, default)

    @classmethod
    def _record_title(cls, record: Any, *, fallback: str) -> str:
        return str(cls._record_get(record, "title", None) or fallback)
