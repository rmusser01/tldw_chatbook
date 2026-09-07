"""Tools & Settings screen implementation.

DEPRECATED (TASK-1346): This wrapper around the legacy ToolsSettingsWindow is not
routed — the "tools_settings" route resolves to MCPScreen (see
UI/Navigation/screen_registry.py). The canonical settings surface is
UI/Screens/settings_screen.py (the F9 Settings destination).
"""

from typing import TYPE_CHECKING

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
