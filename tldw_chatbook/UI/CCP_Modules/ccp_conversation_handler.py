"""Handler for conversation-related operations in the CCP window."""

from typing import TYPE_CHECKING, Optional, List, Dict, Any
from loguru import logger
from textual import work
from textual.widgets import ListView, ListItem, Input, TextArea, Button

from .ccp_messages import ConversationMessage, ViewChangeMessage

if TYPE_CHECKING:
    from ..Conv_Char_Window import CCPWindow

logger = logger.bind(module="CCPConversationHandler")


class CCPConversationHandler:
    """Handles all conversation-related operations for the CCP window."""
    
    def __init__(self, window: 'CCPWindow'):
        """Initialize the conversation handler.
        
        Args:
            window: Reference to the parent CCP window
        """
        self.window = window
        self.app_instance = window.app_instance
        self.current_conversation_id: Optional[int] = None
        self.search_results: List[Dict[str, Any]] = []
        
        logger.debug("CCPConversationHandler initialized")
    
    async def handle_search(self, search_term: str, search_type: str = "title") -> None:
        """Handle conversation search (async wrapper).
        
        Args:
            search_term: The term to search for
            search_type: Type of search ("title", "content", "tags")
        """
        logger.debug(f"Starting conversation search: term='{search_term}', type={search_type}")
        
        # Run the sync search in a worker thread
        self.window.run_worker(
            self._search_conversations_sync,
            search_term,
            search_type,
            thread=True,
            exclusive=True,
            name="conversation_search"
        )
    
    @work(thread=True)
    def _search_conversations_sync(self, search_term: str, search_type: str = "title") -> None:
        """Sync method to perform conversation search in a worker thread.
        
        Args:
            search_term: The term to search for
            search_type: Type of search ("title", "content", "tags")
        """
        logger.debug(f"Searching conversations: term='{search_term}', type={search_type}")
        
        # Import here to avoid circular imports
        from ...Chat.Chat_Functions import search_conversations_by_keywords, fetch_all_conversations
        
        try:
            if search_type == "title":
                # Search by title
                if search_term:
                    self.search_results = self._search_by_title_sync(search_term)
                else:
                    # Fetch all conversations if no search term
                    self.search_results = fetch_all_conversations()
            elif search_type == "content":
                # Search by content keywords
                self.search_results = search_conversations_by_keywords(search_term) if search_term else []
            elif search_type == "tags":
                # Search by tags
                self.search_results = self._search_by_tags_sync(search_term)
            
            # Update the search results list on main thread
            self.window.call_from_thread(self._update_search_results_ui)
            
            logger.info(f"Found {len(self.search_results)} conversations matching '{search_term}'")
            
        except Exception as e:
            logger.error(f"Error searching conversations: {e}", exc_info=True)
    
    async def _search_by_title(self, search_term: str) -> List[Dict[str, Any]]:
        """Search conversations by title.
        
        Args:
            search_term: The title to search for
            
        Returns:
            List of matching conversations
        """
        from ...Chat.Chat_Functions import fetch_all_conversations
        
        all_conversations = fetch_all_conversations()
        search_lower = search_term.lower()
        
        return [
            conv for conv in all_conversations
            if search_lower in conv.get('name', '').lower()
        ]
    
    async def _search_by_tags(self, tags_str: str) -> List[Dict[str, Any]]:
        """Search conversations by tags.
        
        Args:
            tags_str: Comma-separated tags to search for
            
        Returns:
            List of matching conversations
        """
        # Parse tags
        tags = [tag.strip().lower() for tag in tags_str.split(',') if tag.strip()]
        if not tags:
            return []
        
        from ...Chat.Chat_Functions import fetch_all_conversations
        
        all_conversations = fetch_all_conversations()
        results = []
        
        for conv in all_conversations:
            # Check if conversation has matching tags in keywords
            conv_keywords = conv.get('keywords', '').lower()
            if any(tag in conv_keywords for tag in tags):
                results.append(conv)
        
        return results
    
    async def _update_search_results_ui(self) -> None:
        """Update the search results ListView in the UI."""
        try:
            results_list = self.window.query_one("#conv-char-search-results-list", ListView)
            results_list.clear()
            
            for conv in self.search_results:
                title = conv.get('name', 'Untitled')
                conv_id = conv.get('conversation_id', conv.get('id'))
                list_item = ListItem(Static(title), id=f"conv-result-{conv_id}")
                results_list.append(list_item)
                
        except Exception as e:
            logger.error(f"Error updating search results UI: {e}")
    
    async def handle_load_selected(self) -> None:
        """Handle loading the selected conversation."""
        try:
            results_list = self.window.query_one("#conv-char-search-results-list", ListView)
            
            if results_list.highlighted_child:
                # Extract conversation ID from the list item ID
                item_id = results_list.highlighted_child.id
                if item_id and item_id.startswith("conv-result-"):
                    conv_id = int(item_id.replace("conv-result-", ""))
                    await self.load_conversation(conv_id)
            else:
                logger.warning("No conversation selected to load")
                
        except Exception as e:
            logger.error(f"Error loading selected conversation: {e}", exc_info=True)
    
    async def load_conversation(self, conversation_id: int) -> None:
        """Load a conversation and display it.
        
        Args:
            conversation_id: The ID of the conversation to load
        """
        logger.info(f"Loading conversation {conversation_id}")
        
        # Run the sync database operation in a worker thread
        self.window.run_worker(
            self._load_conversation_sync,
            conversation_id,
            thread=True,
            exclusive=True,
            name=f"load_conversation_{conversation_id}"
        )
    
    @work(thread=True)
    def _load_conversation_sync(self, conversation_id: int) -> None:
        """Sync method to load conversation data in a worker thread.
        
        Args:
            conversation_id: The ID of the conversation to load
        """
        try:
            from ...Chat.Chat_Functions import load_conversation
            
            # Load the conversation (sync database operation)
            success = load_conversation(conversation_id)
            
            if success:
                self.current_conversation_id = conversation_id
                
                # Post messages from worker thread using call_from_thread
                self.window.call_from_thread(
                    self.window.post_message,
                    ConversationMessage.Loaded(conversation_id, [])
                )
                
                # Switch view to show conversation
                self.window.call_from_thread(
                    self.window.post_message,
                    ViewChangeMessage.Requested("conversation_messages")
                )
                
                # Update UI on main thread
                self.window.call_from_thread(self._display_conversation_messages)
                
                logger.info(f"Conversation {conversation_id} loaded successfully")
            else:
                logger.error(f"Failed to load conversation {conversation_id}")
                
        except Exception as e:
            logger.error(f"Error loading conversation {conversation_id}: {e}", exc_info=True)
    
    def _display_conversation_messages(self) -> None:
        """Display conversation messages in the UI."""
        try:
            from ...DB.ChaChaNotes_DB import get_messages_from_conversation
            
            if not self.current_conversation_id:
                return
            
            # Get messages for the conversation
            messages = get_messages_from_conversation(self.current_conversation_id)
            
            # Get the messages container
            messages_view = self.window.query_one("#ccp-conversation-messages-view")
            
            # Clear existing messages (keep the title)
            for widget in list(messages_view.children):
                if widget.id != "ccp-center-pane-title-conv":
                    widget.remove()
            
            # Display messages
            from ...Widgets.chat_message_enhanced import ChatMessageEnhanced
            
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                # Create message widget
                message_widget = ChatMessageEnhanced(
                    content=content,
                    role=role,
                    message_id=msg.get('id'),
                    timestamp=msg.get('timestamp'),
                    is_streamed=False
                )
                
                messages_view.mount(message_widget)
            
            logger.debug(f"Displayed {len(messages)} messages")
            
        except Exception as e:
            logger.error(f"Error displaying conversation messages: {e}", exc_info=True)
    
    async def handle_save_details(self, title: str, keywords: str) -> None:
        """Save conversation details.
        
        Args:
            title: The conversation title
            keywords: The conversation keywords
        """
        if not self.current_conversation_id:
            logger.warning("No conversation loaded to save details for")
            return
        
        try:
            from ...Chat.Chat_Functions import update_conversation_metadata
            
            success = update_conversation_metadata(
                self.current_conversation_id,
                title=title,
                keywords=keywords
            )
            
            if success:
                logger.info(f"Saved details for conversation {self.current_conversation_id}")
                
                # Post update message
                self.window.post_message(
                    ConversationMessage.Updated(
                        self.current_conversation_id,
                        title,
                        keywords
                    )
                )
            else:
                logger.error(f"Failed to save details for conversation {self.current_conversation_id}")
                
        except Exception as e:
            logger.error(f"Error saving conversation details: {e}", exc_info=True)
    
    async def handle_export(self, format: str = "json") -> None:
        """Export the current conversation.
        
        Args:
            format: Export format ("json" or "text")
        """
        if not self.current_conversation_id:
            logger.warning("No conversation loaded to export")
            return
        
        try:
            from ...Chat.document_generator import DocumentGenerator
            
            generator = DocumentGenerator()
            
            if format == "json":
                file_path = await generator.export_conversation_json(self.current_conversation_id)
            else:
                file_path = await generator.export_conversation_text(self.current_conversation_id)
            
            if file_path:
                logger.info(f"Exported conversation to {file_path}")
                # Could show a notification here
            else:
                logger.error(f"Failed to export conversation")
                
        except Exception as e:
            logger.error(f"Error exporting conversation: {e}", exc_info=True)
    
    async def handle_import(self, file_path: str) -> None:
        """Import a conversation from file.
        
        Args:
            file_path: Path to the conversation file
        """
        try:
            # Implementation would depend on the import format
            # This is a placeholder for the import logic
            logger.info(f"Importing conversation from {file_path}")
            
            # Post message when import is complete
            # self.window.post_message(ConversationMessage.Created(...))
            
        except Exception as e:
            logger.error(f"Error importing conversation: {e}", exc_info=True)
    
    def refresh_conversation_list(self) -> None:
        """Refresh the conversation search results."""
        # Re-run the last search to refresh results
        try:
            search_input = self.window.query_one("#conv-char-search-input", Input)
            if search_input.value:
                self.window.run_worker(
                    self.handle_search,
                    search_input.value,
                    "title",
                    thread=True,
                    exclusive=True,
                    name="refresh_search"
                )
        except Exception as e:
            logger.error(f"Error refreshing conversation list: {e}")