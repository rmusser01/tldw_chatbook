"""Handler for conversation-related operations in the Personas screen."""

from typing import TYPE_CHECKING, Optional, List, Dict, Any, Tuple, Union
from loguru import logger
from textual import work
from textual.widgets import ListView, ListItem, Input, TextArea, Button, Static

from ...config import get_chachanotes_db_lazy
from .ccp_messages import ConversationMessage, ViewChangeMessage

if TYPE_CHECKING:
    from ..Screens.personas_screen import PersonasScreen

logger = logger.bind(module="CCPConversationHandler")
ConversationId = Union[int, str]


def normalize_conversation_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize DB rows into a stable CCP conversation contract."""
    normalized = dict(row or {})
    conversation_id = normalized.get("id", normalized.get("conversation_id"))
    if conversation_id is not None:
        normalized["id"] = str(conversation_id)
        normalized["conversation_id"] = str(conversation_id)
    if normalized.get("character_id") is not None:
        normalized["character_id"] = str(normalized["character_id"])
    if normalized.get("discovery_entity_id") is not None:
        normalized["discovery_entity_id"] = str(normalized["discovery_entity_id"])
    normalized["title"] = normalized.get("title") or normalized.get("name") or "Untitled"
    normalized["runtime_backend"] = normalized.get("runtime_backend") or "local"
    normalized["discovery_owner"] = normalized.get("discovery_owner") or "general_chat"
    normalized["scope_type"] = normalized.get("scope_type") or "global"
    normalized["workspace_id"] = normalized.get("workspace_id")
    return normalized


class CCPConversationHandler:
    """Handles all conversation-related operations for the Personas screen."""
    
    def __init__(self, window: "PersonasScreen"):
        """Initialize the conversation handler.
        
        Args:
            window: Reference to the parent Personas screen
        """
        self.window = window
        self.app_instance = window.app_instance
        self.current_conversation_id: Optional[ConversationId] = None
        self.current_conversation_data: Dict[str, Any] = {}
        self.search_results: List[Dict[str, Any]] = []
        
        logger.debug("CCPConversationHandler initialized")

    def _conversation_db(self):
        """Return the DB used for CCP conversation discovery."""
        app_attrs = vars(self.app_instance) if hasattr(self.app_instance, "__dict__") else {}
        return app_attrs.get("chachanotes_db") or get_chachanotes_db_lazy()

    def _current_scope(self) -> Tuple[str, Optional[str]]:
        """Resolve the currently selected discovery scope for CCP browsing."""
        state = getattr(self.window, "state", None)
        selected_persona_id = getattr(state, "selected_persona_id", None)
        if selected_persona_id:
            return ("ccp_persona", str(selected_persona_id))
        selected_character_id = getattr(state, "selected_character_id", None)
        if selected_character_id:
            return ("ccp_character", str(selected_character_id))
        return ("general_chat", None)

    def _matches_active_scope(self, conversation: Dict[str, Any]) -> bool:
        """Return True when a conversation belongs in the current CCP scope."""
        owner, entity_id = self._current_scope()
        normalized = normalize_conversation_row(conversation)
        if normalized.get("scope_type") == "workspace":
            return False
        conversation_owner = normalized.get("discovery_owner", "general_chat")
        if owner == "general_chat":
            return conversation_owner == "general_chat"
        if conversation_owner != owner:
            return False
        conversation_entity_id = normalized.get("discovery_entity_id")
        if conversation_entity_id is not None:
            return conversation_entity_id == entity_id
        if owner == "ccp_character":
            return normalized.get("character_id") == entity_id
        return False

    def _filter_conversations_for_scope(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep only conversations that belong to the currently selected scope."""
        normalized = [normalize_conversation_row(row) for row in conversations]
        return [row for row in normalized if self._matches_active_scope(row)]
    
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
        
        try:
            db = self._conversation_db()
            if db is None:
                self.search_results = []
                self.window.call_from_thread(self._update_search_results_ui)
                logger.warning("Conversation DB unavailable; returning empty CCP results")
                return

            if search_type == "title":
                # Search by title
                if search_term:
                    results = self._search_by_title_sync(search_term)
                else:
                    # Fetch all conversations if no search term
                    results = db.list_all_active_conversations(limit=100)
            elif search_type == "content":
                # Search by content keywords
                results = db.search_conversations_by_content(search_term, limit=100) if search_term else []
            elif search_type == "tags":
                # Search by tags
                results = self._search_by_tags_sync(search_term)
            else:
                results = []

            self.search_results = self._filter_conversations_for_scope(list(results or []))
            
            # Update the search results list on main thread
            self.window.call_from_thread(self._update_search_results_ui)
            
            logger.info(f"Found {len(self.search_results)} conversations matching '{search_term}'")
            
        except Exception as e:
            logger.error(f"Error searching conversations: {e}", exc_info=True)
    
    def _search_by_title_sync(self, search_term: str) -> List[Dict[str, Any]]:
        """Search conversations by title (sync version for worker).
        
        Args:
            search_term: The title to search for
            
        Returns:
            List of matching conversations
        """
        db = self._conversation_db()
        if db is None:
            return []
        return list(db.search_conversations_by_title(search_term, limit=100) or [])
    
    def _search_by_tags_sync(self, tags_str: str) -> List[Dict[str, Any]]:
        """Search conversations by tags (sync version for worker).
        
        Args:
            tags_str: Comma-separated tags to search for
            
        Returns:
            List of matching conversations
        """
        # Parse tags
        tags = [tag.strip().lower() for tag in tags_str.split(',') if tag.strip()]
        if not tags:
            return []
        
        db = self._conversation_db()
        if db is None:
            return []

        all_conversations = list(db.list_all_active_conversations(limit=200) or [])
        results = []
        
        for conv in all_conversations:
            # Check if conversation has matching tags in keywords
            conv_keywords = str(conv.get('keywords', '') or '').lower()
            if any(tag in conv_keywords for tag in tags):
                results.append(conv)
        
        return results
    
    async def _update_search_results_ui(self) -> None:
        """Update the search results ListView in the UI."""
        try:
            results_list = self.window.query_one("#conv-char-search-results-list", ListView)
            results_list.clear()
            
            for conv in self.search_results:
                title = conv.get('title') or conv.get('name') or 'Untitled'
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
                    conv_id = item_id.replace("conv-result-", "")
                    await self.load_conversation(conv_id)
            else:
                logger.warning("No conversation selected to load")
                
        except Exception as e:
            logger.error(f"Error loading selected conversation: {e}", exc_info=True)
    
    async def load_conversation(self, conversation_id: ConversationId) -> None:
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
    def _load_conversation_sync(self, conversation_id: ConversationId) -> None:
        """Sync method to load conversation data in a worker thread.
        
        Args:
            conversation_id: The ID of the conversation to load
        """
        try:
            db = self._conversation_db()
            conversation_row = db.get_conversation_by_id(str(conversation_id)) if db is not None else None
            conversation_data = normalize_conversation_row(conversation_row) if conversation_row else {}
            
            if conversation_data and self._matches_active_scope(conversation_data):
                self.current_conversation_id = conversation_data["id"]
                self.current_conversation_data = conversation_data
                
                # Post messages from worker thread using call_from_thread
                self.window.call_from_thread(
                    self.window.post_message,
                    ConversationMessage.Loaded(
                        self.current_conversation_id,
                        [],
                        conversation_data=self.current_conversation_data,
                    )
                )
                
                # Switch view to show conversation
                self.window.call_from_thread(
                    self.window.post_message,
                    ViewChangeMessage.Requested("conversation_messages")
                )

                logger.info(f"Conversation {conversation_id} loaded successfully")
            else:
                logger.error(f"Failed to load conversation {conversation_id}")
                
        except Exception as e:
            logger.error(f"Error loading conversation {conversation_id}: {e}", exc_info=True)

    def get_conversation_contract(self, conversation_id: Optional[ConversationId] = None) -> Dict[str, Any]:
        """Return the normalized conversation metadata used for chat launching."""
        target_id = str(conversation_id or self.current_conversation_id or "")
        if not target_id:
            return {}
        if self.current_conversation_data and self.current_conversation_data.get("id") == target_id:
            return dict(self.current_conversation_data)
        db = self._conversation_db()
        if db is None:
            return {}
        row = db.get_conversation_by_id(target_id)
        if not row:
            return {}
        return normalize_conversation_row(row)

    def _display_conversation_messages(self) -> None:
        """Display conversation messages in the UI."""
        try:
            db = self._conversation_db()
            if db is None or not self.current_conversation_id:
                return
            
            # Get messages for the conversation
            messages = db.get_messages_for_conversation(str(self.current_conversation_id), limit=200)
            
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
