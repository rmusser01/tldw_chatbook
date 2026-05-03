"""Library destination shell for source material and Search/RAG."""

from textual.app import ComposeResult
from textual import on
from textual.containers import Vertical
from textual.widgets import Button, Static

from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen


class LibraryScreen(BaseAppScreen):
    """Source material, imports/exports, conversations, and Search/RAG entry."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "library", **kwargs)

    def compose_content(self) -> ComposeResult:
        with Vertical(id="library-shell"):
            yield Static("Library", id="library-title", classes="ds-destination-header")
            yield Static(
                "Source material, notes, media, conversations, imports/exports, and Search/RAG.",
                id="library-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="library-sections", classes="ds-panel"):
                yield Button("Open Notes", id="library-open-notes")
                yield Button("Open Media", id="library-open-media")
                yield Button("Open Conversations", id="library-open-conversations")
                yield Button("Import/Export Sources", id="library-open-import-export")
                yield Button("Search/RAG", id="library-open-search")

    @on(Button.Pressed, "#library-open-notes")
    def open_notes(self) -> None:
        self.post_message(NavigateToScreen("notes"))

    @on(Button.Pressed, "#library-open-media")
    def open_media(self) -> None:
        self.post_message(NavigateToScreen("media"))

    @on(Button.Pressed, "#library-open-conversations")
    def open_conversations(self) -> None:
        self.post_message(NavigateToScreen("conversation"))

    @on(Button.Pressed, "#library-open-import-export")
    def open_import_export(self) -> None:
        self.post_message(NavigateToScreen("ingest"))

    @on(Button.Pressed, "#library-open-search")
    def open_search(self) -> None:
        self.post_message(NavigateToScreen("search"))
