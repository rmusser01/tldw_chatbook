"""LLM Management screen implementation."""

from typing import TYPE_CHECKING

from textual.app import ComposeResult

from ..Navigation.base_app_screen import BaseAppScreen
from ..LLM_Management_Window import LLMManagementWindow

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class LLMScreen(BaseAppScreen):
    """
    LLM Management screen wrapper.
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(app_instance, "llm", **kwargs)
        self.llm_window = None
    
    def compose_content(self) -> ComposeResult:
        """Compose the LLM management window content."""
        self.llm_window = LLMManagementWindow(self.app_instance, classes="window")
        # Yield the window widget directly
        yield self.llm_window
    
    def save_state(self):
        """Save LLM window state."""
        state = super().save_state()
        # Add any LLM-specific state here
        return state
    
    def restore_state(self, state):
        """Restore LLM window state."""
        super().restore_state(state)
        # Restore any LLM-specific state here