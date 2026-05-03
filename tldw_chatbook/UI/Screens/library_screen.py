"""Library destination shell for source material and Search/RAG."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..Navigation.base_app_screen import BaseAppScreen


class LibraryScreen(BaseAppScreen):
    """Source material, imports/exports, conversations, and Search/RAG entry."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "library", **kwargs)

    def compose_content(self) -> ComposeResult:
        with Vertical(id="library-shell"):
            yield Static("Library", id="library-title", classes="ds-destination-header")
            yield Static(
                "Source material, conversations, imports/exports, and Search/RAG.",
                id="library-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="library-sections", classes="ds-panel"):
                yield Static("Notes | Media | Conversations | Import/Export | Search/RAG")
