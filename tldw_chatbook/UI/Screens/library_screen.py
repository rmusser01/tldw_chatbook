"""Library destination shell for source material and Search/RAG."""

from textual.app import ComposeResult
from textual import on
from textual.containers import Vertical
from textual.widgets import Button, Static

from ...Chat.chat_handoff_models import ChatHandoffPayload
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
                yield Button("Open Notes", id="library-open-notes", tooltip="Open saved notes and workspaces.")
                yield Button("Open Media", id="library-open-media", tooltip="Open ingested media and transcripts.")
                yield Button(
                    "Open Conversations",
                    id="library-open-conversations",
                    tooltip="Open saved conversation browsing inside Library.",
                )
                yield Button(
                    "Import/Export Sources",
                    id="library-open-import-export",
                    tooltip="Open source import and export tools.",
                )
                yield Button("Search/RAG", id="library-open-search", tooltip="Search or ask over indexed sources.")
                yield Button(
                    "Use in Console",
                    id="library-use-in-console",
                    tooltip="Stage Library context in Console.",
                )

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

    @on(Button.Pressed, "#library-use-in-console")
    def use_in_console(self) -> None:
        self.app_instance.open_chat_with_handoff(
            ChatHandoffPayload(
                source="library",
                item_type="library-context",
                title="Library context",
                body="Stage Library source material, notes, media, conversations, imports/exports, or Search/RAG results.",
            )
        )
