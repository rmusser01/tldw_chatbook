"""Evaluations screen implementation."""

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

from ..Navigation.base_app_screen import BaseAppScreen
from ..Evals.navigation import EvalNavigationScreen

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class EvalsScreen(BaseAppScreen):
    """
    Evaluations screen wrapper that directly pushes the navigation screen.
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(app_instance, "evals", **kwargs)
    
    def compose_content(self) -> ComposeResult:
        """Compose a placeholder that will be replaced by the navigation screen."""
        # Create a placeholder container
        yield Container(
            Static("Loading Evaluation Lab..."),
            id="evals-placeholder"
        )
    
    def on_mount(self) -> None:
        """When mounted, push the actual evaluation navigation screen."""
        super().on_mount()
        
        # Push the evaluation navigation screen
        eval_nav_screen = EvalNavigationScreen(self.app_instance)
        self.app.push_screen(eval_nav_screen)
    
    def save_state(self):
        """Save evals screen state."""
        state = super().save_state()
        # Add any evals-specific state here
        return state
    
    def restore_state(self, state):
        """Restore evals screen state."""
        super().restore_state(state)
        # Restore any evals-specific state here