"""Manager for displaying conversation messages in the CCP window."""

from typing import TYPE_CHECKING, List, Dict, Any, Optional
from loguru import logger
from textual.widgets import Static
from textual import work

if TYPE_CHECKING:
    from ..Conv_Char_Window import CCPWindow

logger = logger.bind(module="CCPMessageManager")


class CCPMessageManager:
    """Manages the display of conversation messages in the CCP window."""
    
    def __init__(self, window: 'CCPWindow'):
        """Initialize the message manager.
        
        Args:
            window: Reference to the parent CCP window
        """
        self.window = window
        self.app_instance = window.app_instance
        self.current_messages: List[Dict[str, Any]] = []
        self.message_widgets: List[Any] = []
        
        logger.debug("CCPMessageManager initialized")
    
    def clear_messages(self) -> None:
        """Clear all displayed messages."""
        try:
            messages_view = self.window.query_one("#ccp-conversation-messages-view")
            
            # Remove all message widgets but keep the title
            for widget in list(messages_view.children):
                if widget.id != "ccp-center-pane-title-conv":
                    widget.remove()
            
            self.message_widgets.clear()
            self.current_messages.clear()
            
            logger.debug("Cleared all conversation messages")
            
        except Exception as e:
            logger.error(f"Error clearing messages: {e}", exc_info=True)
    
    @work(thread=True)
    async def load_conversation_messages(self, conversation_id: int) -> None:
        """Load and display messages for a conversation.
        
        Args:
            conversation_id: The ID of the conversation to load messages for
        """
        logger.info(f"Loading messages for conversation {conversation_id}")
        
        try:
            from ...DB.ChaChaNotes_DB import get_messages_from_conversation
            
            # Get messages from database
            messages = get_messages_from_conversation(conversation_id)
            
            if messages:
                self.current_messages = messages
                
                # Display on main thread
                self.window.call_from_thread(self._display_messages)
                
                logger.info(f"Loaded {len(messages)} messages for conversation {conversation_id}")
            else:
                logger.warning(f"No messages found for conversation {conversation_id}")
                self.window.call_from_thread(self.clear_messages)
                
        except Exception as e:
            logger.error(f"Error loading conversation messages: {e}", exc_info=True)
    
    def _display_messages(self) -> None:
        """Display the loaded messages in the UI."""
        try:
            # Clear existing messages first
            self.clear_messages()
            
            messages_view = self.window.query_one("#ccp-conversation-messages-view")
            
            # Import message widget here to avoid circular imports
            from ...Widgets.chat_message_enhanced import ChatMessageEnhanced
            
            for msg in self.current_messages:
                message_widget = self._create_message_widget(msg)
                if message_widget:
                    messages_view.mount(message_widget)
                    self.message_widgets.append(message_widget)
            
            logger.debug(f"Displayed {len(self.message_widgets)} message widgets")
            
        except Exception as e:
            logger.error(f"Error displaying messages: {e}", exc_info=True)
    
    def _create_message_widget(self, message_data: Dict[str, Any]) -> Optional[Any]:
        """Create a message widget from message data.
        
        Args:
            message_data: The message data dictionary
            
        Returns:
            A message widget or None if creation fails
        """
        try:
            from ...Widgets.chat_message_enhanced import ChatMessageEnhanced
            
            # Extract message fields
            content = message_data.get('content', '')
            role = message_data.get('role', 'user')
            message_id = message_data.get('id')
            timestamp = message_data.get('timestamp')
            
            # Handle tool messages if present
            tool_calls = message_data.get('tool_calls')
            tool_call_id = message_data.get('tool_call_id')
            
            # Create appropriate widget based on message type
            if tool_calls:
                # Tool call message
                from ...Widgets.tool_message_widgets import ToolCallMessage
                return ToolCallMessage(
                    tool_calls=tool_calls,
                    message_id=message_id,
                    timestamp=timestamp
                )
            elif tool_call_id:
                # Tool result message
                from ...Widgets.tool_message_widgets import ToolResultMessage
                return ToolResultMessage(
                    tool_call_id=tool_call_id,
                    content=content,
                    message_id=message_id,
                    timestamp=timestamp
                )
            else:
                # Regular chat message
                return ChatMessageEnhanced(
                    content=content,
                    role=role,
                    message_id=message_id,
                    timestamp=timestamp,
                    is_streamed=False
                )
                
        except Exception as e:
            logger.error(f"Error creating message widget: {e}", exc_info=True)
            return None
    
    def add_message(self, message_data: Dict[str, Any]) -> None:
        """Add a single message to the display.
        
        Args:
            message_data: The message data to add
        """
        try:
            messages_view = self.window.query_one("#ccp-conversation-messages-view")
            
            message_widget = self._create_message_widget(message_data)
            if message_widget:
                messages_view.mount(message_widget)
                self.message_widgets.append(message_widget)
                self.current_messages.append(message_data)
                
                # Scroll to the new message
                message_widget.scroll_visible()
                
                logger.debug(f"Added message from {message_data.get('role', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error adding message: {e}", exc_info=True)
    
    def update_message(self, message_id: int, new_content: str) -> None:
        """Update an existing message's content.
        
        Args:
            message_id: The ID of the message to update
            new_content: The new content for the message
        """
        try:
            # Find the message widget
            for widget in self.message_widgets:
                if hasattr(widget, 'message_id') and widget.message_id == message_id:
                    if hasattr(widget, 'update_content'):
                        widget.update_content(new_content)
                        logger.debug(f"Updated message {message_id}")
                        break
            
            # Update in our cached messages
            for msg in self.current_messages:
                if msg.get('id') == message_id:
                    msg['content'] = new_content
                    break
                    
        except Exception as e:
            logger.error(f"Error updating message {message_id}: {e}", exc_info=True)
    
    def remove_message(self, message_id: int) -> None:
        """Remove a message from the display.
        
        Args:
            message_id: The ID of the message to remove
        """
        try:
            # Find and remove the message widget
            for i, widget in enumerate(self.message_widgets):
                if hasattr(widget, 'message_id') and widget.message_id == message_id:
                    widget.remove()
                    self.message_widgets.pop(i)
                    logger.debug(f"Removed message widget {message_id}")
                    break
            
            # Remove from cached messages
            self.current_messages = [
                msg for msg in self.current_messages 
                if msg.get('id') != message_id
            ]
            
        except Exception as e:
            logger.error(f"Error removing message {message_id}: {e}", exc_info=True)
    
    def highlight_message(self, message_id: int) -> None:
        """Highlight a specific message.
        
        Args:
            message_id: The ID of the message to highlight
        """
        try:
            for widget in self.message_widgets:
                if hasattr(widget, 'message_id'):
                    if widget.message_id == message_id:
                        # Add highlight class
                        widget.add_class("highlighted")
                        widget.scroll_visible()
                    else:
                        # Remove highlight from others
                        widget.remove_class("highlighted")
                        
        except Exception as e:
            logger.error(f"Error highlighting message {message_id}: {e}", exc_info=True)
    
    def get_message_count(self) -> int:
        """Get the current number of messages displayed.
        
        Returns:
            The number of messages currently displayed
        """
        return len(self.current_messages)
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all currently displayed messages.
        
        Returns:
            List of message data dictionaries
        """
        return self.current_messages.copy()
    
    def scroll_to_bottom(self) -> None:
        """Scroll to the bottom of the messages view."""
        try:
            if self.message_widgets:
                last_widget = self.message_widgets[-1]
                last_widget.scroll_visible()
                logger.debug("Scrolled to bottom of messages")
        except Exception as e:
            logger.error(f"Error scrolling to bottom: {e}", exc_info=True)
    
    def scroll_to_top(self) -> None:
        """Scroll to the top of the messages view."""
        try:
            messages_view = self.window.query_one("#ccp-conversation-messages-view")
            messages_view.scroll_home()
            logger.debug("Scrolled to top of messages")
        except Exception as e:
            logger.error(f"Error scrolling to top: {e}", exc_info=True)