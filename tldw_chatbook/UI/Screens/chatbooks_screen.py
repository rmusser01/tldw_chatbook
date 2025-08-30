"""
Chatbooks Screen
Screen wrapper for Chatbooks functionality in screen-based navigation.
"""

from textual.screen import Screen
from textual.app import ComposeResult
from textual.reactive import reactive
from typing import Optional, List, Dict, Any
from loguru import logger

from ..Chatbooks_Window import ChatbooksWindow


class ChatbooksScreen(Screen):
    """Screen wrapper for Chatbooks functionality."""
    
    # Screen-specific state
    current_chatbook: reactive[Optional[Dict[str, Any]]] = reactive(None)
    chatbook_list: reactive[List[Dict[str, Any]]] = reactive([])
    is_editing: reactive[bool] = reactive(False)
    selected_chatbook_id: reactive[Optional[int]] = reactive(None)
    
    def compose(self) -> ComposeResult:
        """Compose the Chatbooks screen with the Chatbooks window."""
        logger.info("Composing Chatbooks screen")
        yield ChatbooksWindow()
    
    async def on_mount(self) -> None:
        """Initialize Chatbooks when screen is mounted."""
        logger.info("Chatbooks screen mounted")
        
        # Get the Chatbooks window
        chatbooks_window = self.query_one(ChatbooksWindow)
        
        # Load chatbooks list
        if hasattr(chatbooks_window, 'load_chatbooks'):
            chatbooks = await chatbooks_window.load_chatbooks()
            self.chatbook_list = chatbooks
        
        # Initialize chatbooks features
        if hasattr(chatbooks_window, 'initialize'):
            await chatbooks_window.initialize()
    
    async def on_screen_suspend(self) -> None:
        """Save state when screen is suspended (navigated away)."""
        logger.debug("Chatbooks screen suspended")
        
        # Save current chatbook if editing
        if self.is_editing and self.current_chatbook:
            chatbooks_window = self.query_one(ChatbooksWindow)
            if hasattr(chatbooks_window, 'save_chatbook'):
                await chatbooks_window.save_chatbook(self.current_chatbook)
        
        self.is_editing = False
    
    async def on_screen_resume(self) -> None:
        """Restore state when screen is resumed."""
        logger.debug("Chatbooks screen resumed")
        
        # Refresh chatbooks list
        chatbooks_window = self.query_one(ChatbooksWindow)
        if hasattr(chatbooks_window, 'refresh_chatbooks'):
            chatbooks = await chatbooks_window.refresh_chatbooks()
            self.chatbook_list = chatbooks
        
        # Restore selected chatbook if any
        if self.selected_chatbook_id:
            if hasattr(chatbooks_window, 'select_chatbook'):
                await chatbooks_window.select_chatbook(self.selected_chatbook_id)
    
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