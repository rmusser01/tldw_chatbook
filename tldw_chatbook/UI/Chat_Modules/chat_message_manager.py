"""
Chat Message Manager Module

Handles all message-related functionality including:
- Message display and formatting
- Message editing and actions
- Message focus and navigation
- Message history management
- Enhanced message features
"""

import asyncio
from typing import TYPE_CHECKING, Optional, List, Union
from loguru import logger
from textual.widgets import Button
from textual.css.query import NoMatches

if TYPE_CHECKING:
    from ..Chat_Window_Enhanced import ChatWindowEnhanced

logger = logger.bind(module="ChatMessageManager")


class ChatMessageManager:
    """Handles message display, editing, and management."""
    
    def __init__(self, chat_window: 'ChatWindowEnhanced'):
        """Initialize the message manager.
        
        Args:
            chat_window: Parent ChatWindowEnhanced instance
        """
        self.chat_window = chat_window
        self.app_instance = chat_window.app_instance
    
    async def edit_focused_message(self):
        """Edit the currently focused message."""
        from ...Event_Handlers.Chat_Events import chat_events
        
        try:
            # Get the chat log container
            chat_log = self.chat_window._chat_log
            if not chat_log:
                logger.debug("Chat log not cached")
                return
            
            # Find the focused widget
            focused_widget = self.app_instance.focused
            
            # Check if the focused widget is a ChatMessage or if we need to find one
            from ...Widgets.Chat_Widgets.chat_message import ChatMessage
            from ...Widgets.Chat_Widgets.chat_message_enhanced import ChatMessageEnhanced
            
            if isinstance(focused_widget, (ChatMessage, ChatMessageEnhanced)):
                message_widget = focused_widget
            else:
                # Try to find the last message in the chat log as a fallback
                message_widget = self._find_last_message(chat_log)
                if not message_widget:
                    logger.debug("No messages found to edit")
                    return
                message_widget.focus()
            
            # Find the edit button in the message widget
            try:
                edit_button = message_widget.query_one(".edit-button", Button)
                # Trigger the edit action by simulating button press
                await chat_events.handle_chat_action_button_pressed(
                    self.app_instance, 
                    edit_button, 
                    message_widget
                )
            except (AttributeError, NoMatches) as e:
                logger.debug(f"Could not find or click edit button: {e}")
                
        except NoMatches as e:
            logger.debug(f"No message widget found to edit: {e}")
        except AttributeError as e:
            logger.error(f"Error in edit_focused_message: {e}")
            self.app_instance.notify("Could not enter edit mode", severity="warning")
    
    def _find_last_message(self, chat_log) -> Optional[Union['ChatMessage', 'ChatMessageEnhanced']]:
        """Find the last message in the chat log.
        
        Args:
            chat_log: The chat log container
            
        Returns:
            The last message widget, or None if no messages found
        """
        from ...Widgets.Chat_Widgets.chat_message import ChatMessage
        from ...Widgets.Chat_Widgets.chat_message_enhanced import ChatMessageEnhanced
        
        messages = chat_log.query(ChatMessage)
        enhanced_messages = chat_log.query(ChatMessageEnhanced)
        all_messages = list(messages) + list(enhanced_messages)
        
        if all_messages:
            return all_messages[-1]
        return None
    
    def get_all_messages(self) -> List[Union['ChatMessage', 'ChatMessageEnhanced']]:
        """Get all messages in the chat log.
        
        Returns:
            List of all message widgets
        """
        from ...Widgets.Chat_Widgets.chat_message import ChatMessage
        from ...Widgets.Chat_Widgets.chat_message_enhanced import ChatMessageEnhanced
        
        chat_log = self.chat_window._chat_log
        if not chat_log:
            return []
        
        messages = chat_log.query(ChatMessage)
        enhanced_messages = chat_log.query(ChatMessageEnhanced)
        return list(messages) + list(enhanced_messages)
    
    def get_message_by_id(self, message_id: str) -> Optional[Union['ChatMessage', 'ChatMessageEnhanced']]:
        """Get a specific message by its ID.
        
        Args:
            message_id: The ID of the message to find
            
        Returns:
            The message widget, or None if not found
        """
        all_messages = self.get_all_messages()
        for message in all_messages:
            if hasattr(message, 'message_id') and message.message_id == message_id:
                return message
        return None
    
    async def add_message(self, content: str, role: str = "user", **kwargs):
        """Add a new message to the chat log.
        
        Args:
            content: Message content
            role: Message role (user/assistant/system)
            **kwargs: Additional message parameters
        """
        from ...Widgets.Chat_Widgets.chat_message_enhanced import ChatMessageEnhanced
        
        chat_log = self.chat_window._chat_log
        if not chat_log:
            logger.error("Chat log not available")
            return
        
        # Create new message widget
        message = ChatMessageEnhanced(
            content=content,
            role=role,
            **kwargs
        )
        
        # Add to chat log
        await chat_log.mount(message)
        
        # Scroll to show new message
        message.scroll_visible()
        
        logger.debug(f"Added {role} message to chat")
    
    async def update_message(self, message_id: str, new_content: str):
        """Update the content of an existing message.
        
        Args:
            message_id: ID of the message to update
            new_content: New content for the message
        """
        message = self.get_message_by_id(message_id)
        if message:
            if hasattr(message, 'update_content'):
                message.update_content(new_content)
            elif hasattr(message, 'content'):
                message.content = new_content
                message.refresh()
            logger.debug(f"Updated message {message_id}")
        else:
            logger.warning(f"Message {message_id} not found for update")
    
    async def remove_message(self, message_id: str):
        """Remove a message from the chat log.
        
        Args:
            message_id: ID of the message to remove
        """
        message = self.get_message_by_id(message_id)
        if message:
            await message.remove()
            logger.debug(f"Removed message {message_id}")
        else:
            logger.warning(f"Message {message_id} not found for removal")
    
    def focus_message(self, message_id: str):
        """Set focus to a specific message.
        
        Args:
            message_id: ID of the message to focus
        """
        message = self.get_message_by_id(message_id)
        if message:
            message.focus()
            message.scroll_visible()
            logger.debug(f"Focused message {message_id}")
        else:
            logger.warning(f"Message {message_id} not found for focus")
    
    def navigate_messages(self, direction: str = "next"):
        """Navigate between messages.
        
        Args:
            direction: 'next' or 'previous'
        """
        all_messages = self.get_all_messages()
        if not all_messages:
            return
        
        focused = self.app_instance.focused
        
        # Find current message index
        current_index = -1
        for i, message in enumerate(all_messages):
            if message == focused:
                current_index = i
                break
        
        # Navigate to next/previous message
        if direction == "next":
            new_index = min(current_index + 1, len(all_messages) - 1)
        else:  # previous
            new_index = max(current_index - 1, 0)
        
        if 0 <= new_index < len(all_messages):
            all_messages[new_index].focus()
            all_messages[new_index].scroll_visible()
    
    def clear_all_messages(self):
        """Clear all messages from the chat log."""
        chat_log = self.chat_window._chat_log
        if chat_log:
            # Remove all child widgets that are messages
            all_messages = self.get_all_messages()
            for message in all_messages:
                message.remove()
            logger.info("Cleared all messages from chat")
    
    def get_message_count(self) -> int:
        """Get the total number of messages.
        
        Returns:
            Number of messages in the chat log
        """
        return len(self.get_all_messages())
    
    def get_messages_by_role(self, role: str) -> List[Union['ChatMessage', 'ChatMessageEnhanced']]:
        """Get all messages with a specific role.
        
        Args:
            role: The role to filter by (user/assistant/system)
            
        Returns:
            List of messages with the specified role
        """
        all_messages = self.get_all_messages()
        return [msg for msg in all_messages 
                if hasattr(msg, 'role') and msg.role == role]
    
    async def handle_message_action(self, action: str, message_widget, **kwargs):
        """Handle actions on message widgets.
        
        Args:
            action: Action to perform (edit, copy, delete, etc.)
            message_widget: The message widget to act on
            **kwargs: Additional action parameters
        """
        from ...Event_Handlers.Chat_Events import chat_events
        
        actions = {
            "edit": lambda: chat_events.handle_chat_action_button_pressed(
                self.app_instance, None, message_widget
            ),
            "copy": lambda: self._copy_message_content(message_widget),
            "delete": lambda: self.remove_message(
                message_widget.message_id if hasattr(message_widget, 'message_id') else None
            ),
            "regenerate": lambda: self._regenerate_message(message_widget)
        }
        
        if action in actions:
            await actions[action]()
        else:
            logger.warning(f"Unknown message action: {action}")
    
    def _copy_message_content(self, message_widget):
        """Copy message content to clipboard.
        
        Args:
            message_widget: The message widget to copy from
        """
        if hasattr(message_widget, 'content'):
            # Would need clipboard integration here
            content = message_widget.content
            logger.info(f"Copied message content: {len(content)} characters")
            self.app_instance.notify("Message copied to clipboard")
    
    async def _regenerate_message(self, message_widget):
        """Regenerate an assistant message.
        
        Args:
            message_widget: The message widget to regenerate
        """
        if hasattr(message_widget, 'role') and message_widget.role == 'assistant':
            # Would trigger regeneration logic here
            logger.info("Regenerating assistant message")
            self.app_instance.notify("Regenerating response...")
    
    def highlight_message(self, message_id: str, highlight_class: str = "highlighted"):
        """Highlight a specific message.
        
        Args:
            message_id: ID of the message to highlight
            highlight_class: CSS class to apply for highlighting
        """
        message = self.get_message_by_id(message_id)
        if message:
            message.add_class(highlight_class)
            # Auto-remove highlight after 2 seconds
            asyncio.create_task(self._remove_highlight(message, highlight_class))
    
    async def _remove_highlight(self, message_widget, highlight_class: str):
        """Remove highlight from a message after delay.
        
        Args:
            message_widget: The message widget
            highlight_class: CSS class to remove
        """
        await asyncio.sleep(2)
        message_widget.remove_class(highlight_class)