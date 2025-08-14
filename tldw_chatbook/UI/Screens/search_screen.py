"""Search/RAG screen implementation."""

from typing import TYPE_CHECKING

from textual.app import ComposeResult

from ..Navigation.base_app_screen import BaseAppScreen
from ..SearchRAGWindow import SearchRAGWindow

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class SearchScreen(BaseAppScreen):
    """
    Search/RAG screen wrapper.
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(app_instance, "search", **kwargs)
        self.search_window = None
    
    def compose_content(self) -> ComposeResult:
        """Compose the search window content."""
        self.search_window = SearchRAGWindow(self.app_instance)
        # Yield the window widget directly
        yield self.search_window
    
    def save_state(self):
        """Save search window state."""
        state = super().save_state()
        # Add any search-specific state here
        return state
    
    def restore_state(self, state):
        """Restore search window state."""
        super().restore_state(state)
        # Restore any search-specific state here