"""
Chat Tab Initializer - Handles initialization for the chat tab.

This module manages the initialization logic when the chat tab is shown,
including populating prompts, characters, and focusing the input field.
"""

from typing import TYPE_CHECKING
from textual.widgets import TextArea
from textual.css.query import QueryError

from .base_initializer import BaseTabInitializer

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class ChatTabInitializer(BaseTabInitializer):
    """Handles chat tab initialization."""
    
    def get_tab_id(self) -> str:
        """Return the chat tab ID."""
        from tldw_chatbook.Constants import TAB_CHAT
        return TAB_CHAT
    
    async def on_tab_shown(self) -> None:
        """Initialize the chat tab when shown."""
        self.log_initialization("Chat tab shown, initializing...")
        
        # Focus the chat input field
        try:
            chat_input = self.app.query_one("#chat-input", TextArea)
            chat_input.focus()
            self.log_initialization("Focused chat input field")
        except QueryError:
            self.logger.warning("Could not find chat input field to focus")
        
        # Import here to avoid circular imports
        from tldw_chatbook.Event_Handlers.Chat_Events import chat_events
        
        # Populate prompts with empty search term
        self.call_async_handler(
            chat_events.handle_chat_sidebar_prompt_search_changed,
            self.app,
            ""  # Empty search term to show all prompts
        )
        self.log_initialization("Initiated prompt population")
        
        # Populate character list
        self.call_async_handler(
            chat_events._populate_chat_character_search_list,
            self.app
        )
        self.log_initialization("Initiated character list population")
        
        self.log_initialization("Chat tab initialization complete")