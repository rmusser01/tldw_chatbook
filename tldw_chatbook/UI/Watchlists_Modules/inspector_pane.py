"""Inspector pane for the watchlists screen."""

from __future__ import annotations

from typing import Any

from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Static


class PreviewRequested(Message):
    """Posted when the user requests a preview of the selected entity."""

    def __init__(self, entity: dict[str, Any] | None) -> None:
        self.entity = entity
        super().__init__()


class CheckNowRequested(Message):
    """Posted when the user requests an immediate check of the selected source."""

    def __init__(self, entity: dict[str, Any] | None) -> None:
        self.entity = entity
        super().__init__()


class StageInConsoleRequested(Message):
    """Posted when the user requests staging the selected entity in Console."""

    def __init__(self, entity: dict[str, Any] | None) -> None:
        self.entity = entity
        super().__init__()


class DeleteRequested(Message):
    """Posted when the user requests deletion of the selected entity."""

    def __init__(self, entity: dict[str, Any] | None) -> None:
        self.entity = entity
        super().__init__()


class InspectorPane(Vertical):
    """Context-aware inspector showing actions for the selected entity."""

    selected_entity = reactive[dict[str, Any] | None](None, recompose=True)

    def compose(self):
        yield Static("Inspector", classes="pane-title")
        entity = self.selected_entity
        if entity is None:
            yield Static(
                "Select a source or run to see actions.",
                id="inspector-empty-state",
            )
            return

        entity_type = self._entity_type(entity)
        title = entity.get("name") or entity.get("source_title") or entity.get("title") or "Untitled"
        yield Static(f"Selected: {title}", id="inspector-entity-title")
        yield Static(f"Type: {entity_type}", id="inspector-entity-type")

        with Vertical(id="inspector-actions"):
            if entity_type == "source":
                yield Button("Preview", id="inspector-preview-button", variant="primary")
                yield Button("Check now", id="inspector-check-now-button", variant="primary")
            if entity_type in ("source", "run"):
                yield Button("Stage in Console", id="inspector-stage-console-button")
            if entity_type in ("source", "run", "rule"):
                yield Button("Delete", id="inspector-delete-button", variant="error")

    @staticmethod
    def _entity_type(entity: dict[str, Any]) -> str:
        if "source_type" in entity or "url" in entity:
            return "source"
        if "status" in entity and ("found_count" in entity or "processed_count" in entity):
            return "run"
        if "item_id" in entity:
            return "item"
        return "unknown"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = str(event.button.id)
        entity = self.selected_entity
        if button_id == "inspector-preview-button":
            self.post_message(PreviewRequested(entity))
        elif button_id == "inspector-check-now-button":
            self.post_message(CheckNowRequested(entity))
        elif button_id == "inspector-stage-console-button":
            self.post_message(StageInConsoleRequested(entity))
        elif button_id == "inspector-delete-button":
            self.post_message(DeleteRequested(entity))
        event.stop()
