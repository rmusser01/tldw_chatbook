"""Library-owned conversation browsing route shell."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..Navigation.base_app_screen import BaseAppScreen


class LibraryConversationsScreen(BaseAppScreen):
    """Saved conversation access inside the Library destination."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, "library", **kwargs)

    def compose_content(self) -> ComposeResult:
        with Vertical(id="library-conversations-shell"):
            yield Static("Library: Conversations", id="library-conversations-title", classes="ds-destination-header")
            yield Static(
                "Saved conversations as source material for browsing, reuse, and Search/RAG.",
                id="library-conversations-purpose",
                classes="destination-purpose",
            )
            with Vertical(id="library-conversations-sections", classes="ds-panel"):
                yield Static("Conversation library | Search | Open in Console")
