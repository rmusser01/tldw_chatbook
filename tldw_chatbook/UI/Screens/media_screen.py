"""Media screen implementation."""

from typing import TYPE_CHECKING

from textual.app import ComposeResult

from ..Navigation.base_app_screen import BaseAppScreen
from ..MediaWindow_v2 import MediaWindow

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class MediaScreen(BaseAppScreen):
    """
    Media management screen wrapper.
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(app_instance, "media", **kwargs)
        self.media_window = None
    
    def compose_content(self) -> ComposeResult:
        """Compose the media window content."""
        self.media_window = MediaWindow(self.app_instance)
        # Yield the window widget directly
        yield self.media_window
    
    def save_state(self):
        """Save media window state."""
        state = super().save_state()
        # Add any media-specific state here
        return state
    
    def restore_state(self, state):
        """Restore media window state."""
        super().restore_state(state)
        # Restore any media-specific state here