"""LLM Management screen implementation."""

from typing import TYPE_CHECKING

from textual.app import ComposeResult

from ..Navigation.base_app_screen import BaseAppScreen
from ..LLM_Management_Window import LLMManagementWindow
from ..Workbench.workbench_state import WorkbenchHeaderState
from ..Workbench.workbench_widgets import DestinationHeader
from .lab_mode_strip import LabModeStrip

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class LLMScreen(BaseAppScreen):
    """
    LLM Management screen wrapper.
    """

    def __init__(self, app_instance: "TldwCli", **kwargs):
        super().__init__(app_instance, "llm", **kwargs)
        self.llm_window = None

    def compose_content(self) -> ComposeResult:
        """Compose the LLM management window content with its destination header."""
        yield DestinationHeader(
            WorkbenchHeaderState(
                title="Models",
                subtitle="Manage providers, models, and endpoints.",
                status="ready",
            ),
            id="llm-destination-header",
        )
        yield LabModeStrip(active_route="llm", id="lab-mode-strip")
        self.llm_window = LLMManagementWindow(self.app_instance, classes="window")
        # Leave room for the destination header above the window.
        self.llm_window.styles.height = "1fr"
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
