"""
Chatbooks Screen
Screen wrapper for Chatbooks functionality in screen-based navigation.
"""

from textual.app import ComposeResult
from loguru import logger

from tldw_chatbook.Constants import TAB_CHATBOOKS

from ..Chatbooks_Window_Improved import ChatbooksWindowImproved
from ..Navigation.base_app_screen import BaseAppScreen


class ChatbooksScreen(BaseAppScreen):
    """Screen wrapper for Chatbooks functionality."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, TAB_CHATBOOKS, **kwargs)

    def compose_content(self) -> ComposeResult:
        """Compose the Chatbooks screen with the Chatbooks window."""
        logger.info("Composing Chatbooks screen")
        yield ChatbooksWindowImproved(self.app_instance)

    async def on_mount(self) -> None:
        """Initialize Chatbooks when screen is mounted."""
        # No super().on_mount(): the dispatcher already invokes
        # BaseAppScreen.on_mount separately for this Mount event.
        logger.info("Chatbooks screen mounted")

    async def on_screen_resume(self) -> None:
        """Restore state when screen is resumed."""
        logger.debug("Chatbooks screen resumed")
        chatbooks_window = self.query_one(ChatbooksWindowImproved)
        if hasattr(chatbooks_window, "_refresh_chatbooks"):
            await chatbooks_window._refresh_chatbooks()

