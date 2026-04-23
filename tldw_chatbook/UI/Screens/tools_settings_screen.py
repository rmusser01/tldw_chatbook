"""Tools & Settings screen implementation."""

from typing import TYPE_CHECKING
from loguru import logger

from textual.app import ComposeResult
from textual.widgets import Button, Markdown

from ..Navigation.base_app_screen import BaseAppScreen
from ..Tools_Settings_Window import ToolsSettingsWindow

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class ToolsSettingsScreen(BaseAppScreen):
    """
    Tools & Settings screen wrapper.
    """
    
    def __init__(self, app_instance: 'TldwCli', **kwargs):
        super().__init__(app_instance, "tools_settings", **kwargs)
        self.tools_window = None
    
    def compose_content(self) -> ComposeResult:
        """Compose the tools settings window content."""
        self.tools_window = ToolsSettingsWindow(self.app_instance, classes="window")
        # Yield the window widget directly
        yield self.tools_window
    
    def save_state(self):
        """Save tools window state."""
        state = super().save_state()
        if self.tools_window:
            state["unified_mcp_view_state"] = self.tools_window.get_unified_mcp_view_state()
        return state
    
    def restore_state(self, state):
        """Restore tools window state."""
        super().restore_state(state)
        if self.tools_window and isinstance(state, dict):
            self.tools_window.set_unified_mcp_view_state(state.get("unified_mcp_view_state"))

    async def handle_runtime_backend_changed(self, runtime_backend: str) -> None:
        """Refresh runtime-sensitive child content when the active source changes."""
        if self.tools_window:
            await self.tools_window.handle_runtime_backend_changed(runtime_backend)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Forward button events to the ToolsSettingsWindow handler."""
        if self.tools_window:
            await self.tools_window.on_button_pressed(event)
    
    async def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Forward markdown link clicks to the ToolsSettingsWindow handler."""
        if self.tools_window:
            await self.tools_window.on_markdown_link_clicked(event)
