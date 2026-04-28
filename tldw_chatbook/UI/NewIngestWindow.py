"""Compatibility NewIngest window used by legacy tests.

The production ingest UI now lives in MediaIngestWindowRebuilt. This module
keeps the older public import path operational while UX work is in flight.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static

from ..Widgets.NewIngest import FilesSelected, ProcessingDashboard, SmartFileDropZone, UnifiedProcessor


class MediaTypeSelected(Message):
    def __init__(self, media_type: str) -> None:
        super().__init__()
        self.media_type = media_type


class MediaTypeCard(Widget):
    def __init__(self, media_type: str, title: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.media_type = media_type
        self.title = title

    def compose(self) -> ComposeResult:
        with Container(classes="media-card"):
            yield Static(self.title)
            yield Button(f"Select {self.title}", id=f"select-{self.media_type}")

    def on_click(self) -> None:
        self.post_message(MediaTypeSelected(self.media_type))


class NewIngestWindow(Widget):
    def __init__(self, app_instance: Any | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.app_instance = app_instance

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("New Ingest", classes="new-ingest-title")
            with Vertical(classes="main-content"):
                with Horizontal(classes="media-type-grid"):
                    yield MediaTypeCard("video", "Video")
                    yield MediaTypeCard("audio", "Audio")
                    yield MediaTypeCard("document", "Document")
                yield SmartFileDropZone()
                yield UnifiedProcessor(self.app_instance)
                yield ProcessingDashboard()

    def on_files_selected(self, message: FilesSelected) -> None:
        try:
            processor = self.query_one(UnifiedProcessor)
        except Exception:
            return
        processor.selected_files = list(message.files)
