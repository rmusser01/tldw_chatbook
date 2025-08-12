"""
Event handler bridge to convert existing event handlers to Textual message patterns.

This bridge allows Chat v99 to use existing event handlers while maintaining
the reactive message-based architecture of Textual.
"""

from typing import Any, Dict, Optional, Callable, Awaitable, TYPE_CHECKING
from dataclasses import dataclass
import asyncio

from textual.message import Message
from loguru import logger

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli
    from ..models import ChatSession, ChatMessage


# ==================== Custom Textual Messages ====================

class ChatEventMessage(Message):
    """Base class for all chat-related messages."""
    
    def __init__(self, data: Any = None):
        super().__init__()
        self.data = data


class SendMessageEvent(ChatEventMessage):
    """Message sent when user sends a chat message."""
    
    def __init__(self, content: str, attachments: Optional[list] = None):
        super().__init__()
        self.content = content
        self.attachments = attachments or []


class StopGenerationEvent(ChatEventMessage):
    """Message sent to stop LLM generation."""
    pass


class LoadSessionEvent(ChatEventMessage):
    """Message sent to load a chat session."""
    
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.session_id = session_id


class SaveSessionEvent(ChatEventMessage):
    """Message sent to save current session."""
    
    def __init__(self, session: 'ChatSession'):
        super().__init__(session)
        self.session = session


class NewSessionEvent(ChatEventMessage):
    """Message sent to create a new session."""
    
    def __init__(self, ephemeral: bool = False):
        super().__init__()
        self.ephemeral = ephemeral


class CharacterLoadedEvent(ChatEventMessage):
    """Message sent when a character is loaded."""
    
    def __init__(self, character_id: int, character_data: Dict[str, Any]):
        super().__init__()
        self.character_id = character_id
        self.character_data = character_data


class TemplateAppliedEvent(ChatEventMessage):
    """Message sent when a prompt template is applied."""
    
    def __init__(self, template_name: str, template_content: str):
        super().__init__()
        self.template_name = template_name
        self.template_content = template_content


class MessageEditedEvent(ChatEventMessage):
    """Message sent when a message is edited."""
    
    def __init__(self, message_index: int, new_content: str):
        super().__init__()
        self.message_index = message_index
        self.new_content = new_content


class MessageDeletedEvent(ChatEventMessage):
    """Message sent when a message is deleted."""
    
    def __init__(self, message_index: int):
        super().__init__()
        self.message_index = message_index


class ExportRequestedEvent(ChatEventMessage):
    """Message sent to export conversation."""
    
    def __init__(self, format: str = "markdown"):
        super().__init__()
        self.format = format


# ==================== Event Handler Bridge ====================

class EventHandlerBridge:
    """
    Bridge between Chat v99's message-based events and existing event handlers.
    
    This class:
    - Converts Textual messages to handler calls
    - Manages async execution of handlers
    - Provides backward compatibility
    - Handles errors gracefully
    """
    
    def __init__(self, app_instance: Optional['TldwCli'] = None):
        """
        Initialize the event handler bridge.
        
        Args:
            app_instance: Reference to the main TldwCli app
        """
        self.app_instance = app_instance
        self._handlers: Dict[type, Callable] = {}
        self._legacy_handlers: Dict[str, Any] = {}
        
        # Import legacy handlers lazily
        self._import_legacy_handlers()
        
        logger.info("EventHandlerBridge initialized")
    
    def _import_legacy_handlers(self):
        """Import existing event handlers for compatibility."""
        try:
            # Import chat event handlers
            from tldw_chatbook.Event_Handlers.Chat_Events import chat_events
            self._legacy_handlers['chat'] = chat_events
            
            # Import character handlers
            from tldw_chatbook.Event_Handlers import conv_char_events
            self._legacy_handlers['character'] = conv_char_events
            
            logger.debug("Legacy handlers imported successfully")
            
        except ImportError as e:
            logger.warning(f"Some legacy handlers not available: {e}")
    
    def register_handler(self, message_type: type, handler: Callable):
        """
        Register a handler for a specific message type.
        
        Args:
            message_type: The message class to handle
            handler: The handler function
        """
        self._handlers[message_type] = handler
        logger.debug(f"Registered handler for {message_type.__name__}")
    
    async def handle_message(self, message: ChatEventMessage) -> Any:
        """
        Handle a chat event message.
        
        Args:
            message: The message to handle
            
        Returns:
            Result from the handler, if any
        """
        message_type = type(message)
        
        # Check for registered handler
        if message_type in self._handlers:
            handler = self._handlers[message_type]
            try:
                if asyncio.iscoroutinefunction(handler):
                    return await handler(message)
                else:
                    return handler(message)
            except Exception as e:
                logger.error(f"Error in handler for {message_type.__name__}: {e}")
                raise
        
        # Fall back to built-in handlers
        return await self._handle_builtin(message)
    
    async def _handle_builtin(self, message: ChatEventMessage) -> Any:
        """
        Handle messages with built-in handlers.
        
        Args:
            message: The message to handle
            
        Returns:
            Result from the handler, if any
        """
        if isinstance(message, SendMessageEvent):
            return await self._handle_send_message(message)
        
        elif isinstance(message, StopGenerationEvent):
            return await self._handle_stop_generation(message)
        
        elif isinstance(message, LoadSessionEvent):
            return await self._handle_load_session(message)
        
        elif isinstance(message, SaveSessionEvent):
            return await self._handle_save_session(message)
        
        elif isinstance(message, NewSessionEvent):
            return await self._handle_new_session(message)
        
        elif isinstance(message, CharacterLoadedEvent):
            return await self._handle_character_loaded(message)
        
        elif isinstance(message, TemplateAppliedEvent):
            return await self._handle_template_applied(message)
        
        elif isinstance(message, MessageEditedEvent):
            return await self._handle_message_edited(message)
        
        elif isinstance(message, MessageDeletedEvent):
            return await self._handle_message_deleted(message)
        
        elif isinstance(message, ExportRequestedEvent):
            return await self._handle_export_requested(message)
        
        else:
            logger.warning(f"No handler for message type: {type(message).__name__}")
            return None
    
    # ==================== Built-in Handlers ====================
    
    async def _handle_send_message(self, message: SendMessageEvent) -> Dict[str, Any]:
        """Handle sending a chat message."""
        logger.debug(f"Handling send message: {message.content[:50]}...")
        
        # Use legacy handler if available
        if self.app_instance and 'chat' in self._legacy_handlers:
            try:
                # Create a mock event for the legacy handler
                class MockEvent:
                    def __init__(self):
                        self.button = type('obj', (object,), {'id': 'send-stop-chat'})()
                
                # Call legacy handler
                await self._legacy_handlers['chat'].handle_chat_send_button_pressed(
                    self.app_instance,
                    MockEvent()
                )
                
                return {"status": "sent", "content": message.content}
                
            except Exception as e:
                logger.error(f"Legacy handler failed: {e}")
                raise
        
        # Fallback response
        return {"status": "sent", "content": message.content}
    
    async def _handle_stop_generation(self, message: StopGenerationEvent) -> bool:
        """Handle stopping LLM generation."""
        logger.debug("Handling stop generation")
        
        if self.app_instance and 'chat' in self._legacy_handlers:
            try:
                await self._legacy_handlers['chat'].handle_stop_chat_generation_pressed(
                    self.app_instance,
                    None
                )
                return True
            except Exception as e:
                logger.error(f"Failed to stop generation: {e}")
                return False
        
        return True
    
    async def _handle_load_session(self, message: LoadSessionEvent) -> Optional['ChatSession']:
        """Handle loading a chat session."""
        logger.debug(f"Handling load session: {message.session_id}")
        
        # Use database adapter to load session
        from ..db.adapter import ChatV99DatabaseAdapter
        
        adapter = ChatV99DatabaseAdapter()
        try:
            session = await adapter.load_session(message.session_id)
            return session
        finally:
            adapter.close()
    
    async def _handle_save_session(self, message: SaveSessionEvent) -> bool:
        """Handle saving a chat session."""
        logger.debug(f"Handling save session: {message.session.id}")
        
        # Use database adapter to save session
        from ..db.adapter import ChatV99DatabaseAdapter
        
        adapter = ChatV99DatabaseAdapter()
        try:
            success = await adapter.save_session(message.session)
            return success
        finally:
            adapter.close()
    
    async def _handle_new_session(self, message: NewSessionEvent) -> 'ChatSession':
        """Handle creating a new session."""
        logger.debug(f"Handling new session (ephemeral: {message.ephemeral})")
        
        from ..models import ChatSession
        
        # Create new session
        session = ChatSession()
        
        if not message.ephemeral:
            # Save to database
            from ..db.adapter import ChatV99DatabaseAdapter
            
            adapter = ChatV99DatabaseAdapter()
            try:
                session = await adapter.create_session(session)
            finally:
                adapter.close()
        
        return session
    
    async def _handle_character_loaded(self, message: CharacterLoadedEvent) -> None:
        """Handle character being loaded."""
        logger.debug(f"Handling character loaded: {message.character_id}")
        
        # Update app state if available
        if self.app_instance:
            self.app_instance.current_character_id = message.character_id
            self.app_instance.current_character_data = message.character_data
    
    async def _handle_template_applied(self, message: TemplateAppliedEvent) -> None:
        """Handle template being applied."""
        logger.debug(f"Handling template applied: {message.template_name}")
        
        # This would update the system prompt in the UI
        pass
    
    async def _handle_message_edited(self, message: MessageEditedEvent) -> bool:
        """Handle message being edited."""
        logger.debug(f"Handling message edited at index {message.message_index}")
        
        # This would update the message in the current session
        return True
    
    async def _handle_message_deleted(self, message: MessageDeletedEvent) -> bool:
        """Handle message being deleted."""
        logger.debug(f"Handling message deleted at index {message.message_index}")
        
        # This would remove the message from the current session
        return True
    
    async def _handle_export_requested(self, message: ExportRequestedEvent) -> str:
        """Handle export request."""
        logger.debug(f"Handling export request: {message.format}")
        
        # Use legacy export functionality if available
        if self.app_instance and 'chat' in self._legacy_handlers:
            try:
                from tldw_chatbook.Chat.document_generator import generate_chat_document
                
                # Get current session data
                # This would be implemented based on current session
                content = "# Exported Chat\n\nExport functionality to be implemented."
                
                return content
                
            except Exception as e:
                logger.error(f"Export failed: {e}")
                raise
        
        return "Export not available"
    
    # ==================== Utility Methods ====================
    
    def create_mock_button_event(self, button_id: str):
        """
        Create a mock button event for legacy handlers.
        
        Args:
            button_id: The button ID to simulate
            
        Returns:
            Mock event object
        """
        class MockButton:
            def __init__(self, id):
                self.id = id
        
        class MockEvent:
            def __init__(self, button_id):
                self.button = MockButton(button_id)
            
            def stop(self):
                pass
        
        return MockEvent(button_id)
    
    def wrap_legacy_handler(self, handler: Callable) -> Callable:
        """
        Wrap a legacy handler to work with the new message system.
        
        Args:
            handler: The legacy handler function
            
        Returns:
            Wrapped handler function
        """
        async def wrapped(message: ChatEventMessage):
            # Convert message to legacy format
            mock_event = self.create_mock_button_event("wrapped-handler")
            
            # Call legacy handler
            if asyncio.iscoroutinefunction(handler):
                return await handler(self.app_instance, mock_event)
            else:
                return handler(self.app_instance, mock_event)
        
        return wrapped