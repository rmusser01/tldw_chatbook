"""Evaluations screen implementation."""

from typing import TYPE_CHECKING

from textual.app import ComposeResult

from ..Navigation.base_app_screen import BaseAppScreen
from ..evals_window_v2 import EvalsWindow

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class EvalsScreen(BaseAppScreen):
    """
    Evaluations screen wrapper.
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(app_instance, "evals", **kwargs)
        self.evals_window = None
    
    def compose_content(self) -> ComposeResult:
        """Compose the evals window content."""
        self.evals_window = EvalsWindow(self.app_instance, classes="window")
        # Yield the window widget directly
        yield self.evals_window
    
    def save_state(self):
        """Save evals window state."""
        state = super().save_state()
        # Add any evals-specific state here
        return state
    
    def restore_state(self, state):
        """Restore evals window state."""
        super().restore_state(state)
        # Restore any evals-specific state here