"""Stats screen implementation."""

from typing import TYPE_CHECKING

from textual.app import ComposeResult

from ..Navigation.base_app_screen import BaseAppScreen
from ..Stats_Window import StatsWindow

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class StatsScreen(BaseAppScreen):
    """
    Stats screen wrapper.
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(app_instance, "stats", **kwargs)
        self.stats_window = None
    
    def compose_content(self) -> ComposeResult:
        """Compose the stats window content."""
        self.stats_window = StatsWindow(self.app_instance)
        # Yield the window widget directly
        yield self.stats_window
    
    def save_state(self):
        """Save stats window state."""
        state = super().save_state()
        # Add any stats-specific state here
        return state
    
    def restore_state(self, state):
        """Restore stats window state."""
        super().restore_state(state)
        # Restore any stats-specific state here