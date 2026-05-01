"""
Chatbooks Screen
Screen wrapper for Chatbooks functionality in screen-based navigation.
"""

from textual.app import ComposeResult
from textual.reactive import reactive
from typing import Optional, List, Dict, Any
from loguru import logger

from tldw_chatbook.Constants import TAB_CHATBOOKS

from ..Chatbooks_Window_Improved import ChatbooksWindowImproved
from ..Navigation.base_app_screen import BaseAppScreen


class ChatbooksScreen(BaseAppScreen):
    """Screen wrapper for Chatbooks functionality."""

    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, TAB_CHATBOOKS, **kwargs)
    
    # Screen-specific state
    current_chatbook: reactive[Optional[Dict[str, Any]]] = reactive(None)
    chatbook_list: reactive[List[Dict[str, Any]]] = reactive([])
    is_editing: reactive[bool] = reactive(False)
    selected_chatbook_id: reactive[Optional[int]] = reactive(None)
    
    def compose_content(self) -> ComposeResult:
        """Compose the Chatbooks screen with the Chatbooks window."""
        logger.info("Composing Chatbooks screen")
        yield ChatbooksWindowImproved(self.app_instance)
    
    async def on_mount(self) -> None:
        """Initialize Chatbooks when screen is mounted."""
        super().on_mount()
        logger.info("Chatbooks screen mounted")
        chatbooks_window = self.query_one(ChatbooksWindowImproved)
        self.chatbook_list = list(chatbooks_window.chatbooks)
    
    async def on_screen_suspend(self) -> None:
        """Save state when screen is suspended (navigated away)."""
        logger.debug("Chatbooks screen suspended")
        self.is_editing = False
    
    async def on_screen_resume(self) -> None:
        """Restore state when screen is resumed."""
        logger.debug("Chatbooks screen resumed")
        chatbooks_window = self.query_one(ChatbooksWindowImproved)
        if hasattr(chatbooks_window, '_refresh_chatbooks'):
            await chatbooks_window._refresh_chatbooks()
        self.chatbook_list = list(chatbooks_window.chatbooks)
    
    def create_new_chatbook(self, title: str, description: str = "") -> None:
        """Create a new chatbook."""
        new_chatbook = {
            "title": title,
            "description": description,
            "created_at": None,  # Will be set by ChatbooksWindow
            "chapters": []
        }
        self.current_chatbook = new_chatbook
        self.is_editing = True
        logger.info(f"Creating new chatbook: {title}")
    
    def open_chatbook(self, chatbook_id: int) -> None:
        """Open an existing chatbook for viewing/editing."""
        self.selected_chatbook_id = chatbook_id
        
        # Find the chatbook in the list
        for chatbook in self.chatbook_list:
            if chatbook.get("id") == chatbook_id:
                self.current_chatbook = chatbook
                break
        
        logger.info(f"Opened chatbook ID: {chatbook_id}")
    
    def delete_chatbook(self, chatbook_id: int) -> None:
        """Mark a chatbook for deletion."""
        # Remove from local list
        self.chatbook_list = [
            cb for cb in self.chatbook_list 
            if cb.get("id") != chatbook_id
        ]
        
        # Clear current if it was the deleted one
        if self.selected_chatbook_id == chatbook_id:
            self.selected_chatbook_id = None
            self.current_chatbook = None
        
        logger.info(f"Deleted chatbook ID: {chatbook_id}")
