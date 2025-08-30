"""
Chat Messages Module

Implements Textual's Message system for loose coupling between chat components.
Following official Textual best practices for event-driven architecture.

This module defines all custom messages used by the chat system to communicate
between components without direct dependencies.
"""

from typing import Optional, Any, Dict
from textual.message import Message
from textual.widgets import Button
from pathlib import Path


class ChatMessage(Message):
    """Base class for all chat-related messages."""
    
    def __init__(self, session_id: str = "default") -> None:
        """Initialize with session ID for multi-tab support.
        
        Args:
            session_id: The chat session ID
        """
        self.session_id = session_id
        super().__init__()


class ChatInputMessage(ChatMessage):
    """Messages related to chat input operations."""
    
    class SendRequested(ChatMessage):
        """User requested to send a message."""
        
        def __init__(self, text: str, attachments: Optional[list] = None, session_id: str = "default") -> None:
            """Initialize send request.
            
            Args:
                text: The message text to send
                attachments: Optional list of attachments
                session_id: The chat session ID
            """
            self.text = text
            self.attachments = attachments or []
            super().__init__(session_id)
    
    class StopRequested(ChatMessage):
        """User requested to stop generation."""
        pass
    
    class InputCleared(ChatMessage):
        """Chat input was cleared."""
        pass
    
    class TextInserted(ChatMessage):
        """Text was inserted into chat input."""
        
        def __init__(self, text: str, position: Optional[tuple] = None, session_id: str = "default") -> None:
            """Initialize text insertion message.
            
            Args:
                text: Text that was inserted
                position: Optional cursor position (row, col)
                session_id: The chat session ID
            """
            self.text = text
            self.position = position
            super().__init__(session_id)


class ChatAttachmentMessage(ChatMessage):
    """Messages related to file attachments."""
    
    class FileSelected(ChatMessage):
        """File was selected for attachment."""
        
        def __init__(self, file_path: Path, session_id: str = "default") -> None:
            """Initialize file selection message.
            
            Args:
                file_path: Path to the selected file
                session_id: The chat session ID
            """
            self.file_path = file_path
            super().__init__(session_id)
    
    class FileProcessed(ChatMessage):
        """File processing completed."""
        
        def __init__(self, file_path: Path, result: Dict[str, Any], session_id: str = "default") -> None:
            """Initialize file processed message.
            
            Args:
                file_path: Path to the processed file
                result: Processing result data
                session_id: The chat session ID
            """
            self.file_path = file_path
            self.result = result
            super().__init__(session_id)
    
    class FileError(ChatMessage):
        """File processing error occurred."""
        
        def __init__(self, file_path: Path, error: str, session_id: str = "default") -> None:
            """Initialize file error message.
            
            Args:
                file_path: Path to the file that failed
                error: Error message
                session_id: The chat session ID
            """
            self.file_path = file_path
            self.error = error
            super().__init__(session_id)
    
    class AttachmentCleared(ChatMessage):
        """All attachments were cleared."""
        pass


class ChatVoiceMessage(ChatMessage):
    """Messages related to voice input."""
    
    class RecordingStarted(ChatMessage):
        """Voice recording started."""
        pass
    
    class RecordingStopped(ChatMessage):
        """Voice recording stopped."""
        pass
    
    class TranscriptReceived(ChatMessage):
        """Voice transcript received."""
        
        def __init__(self, text: str, is_final: bool = False, session_id: str = "default") -> None:
            """Initialize transcript message.
            
            Args:
                text: Transcribed text
                is_final: Whether this is the final transcript
                session_id: The chat session ID
            """
            self.text = text
            self.is_final = is_final
            super().__init__(session_id)
    
    class VoiceError(ChatMessage):
        """Voice recording/processing error."""
        
        def __init__(self, error: str, session_id: str = "default") -> None:
            """Initialize voice error message.
            
            Args:
                error: Error message
                session_id: The chat session ID
            """
            self.error = error
            super().__init__(session_id)


class ChatSidebarMessage(ChatMessage):
    """Messages related to sidebar operations."""
    
    class SidebarToggled(ChatMessage):
        """Sidebar visibility toggled."""
        
        def __init__(self, sidebar_id: str, visible: bool, session_id: str = "default") -> None:
            """Initialize sidebar toggle message.
            
            Args:
                sidebar_id: ID of the sidebar
                visible: New visibility state
                session_id: The chat session ID
            """
            self.sidebar_id = sidebar_id
            self.visible = visible
            super().__init__(session_id)
    
    class CharacterLoaded(ChatMessage):
        """Character was loaded."""
        
        def __init__(self, character_id: str, character_data: Dict, session_id: str = "default") -> None:
            """Initialize character loaded message.
            
            Args:
                character_id: ID of the loaded character
                character_data: Character data dictionary
                session_id: The chat session ID
            """
            self.character_id = character_id
            self.character_data = character_data
            super().__init__(session_id)
    
    class PromptSelected(ChatMessage):
        """Prompt was selected."""
        
        def __init__(self, prompt_id: str, prompt_text: str, session_id: str = "default") -> None:
            """Initialize prompt selected message.
            
            Args:
                prompt_id: ID of the selected prompt
                prompt_text: The prompt text
                session_id: The chat session ID
            """
            self.prompt_id = prompt_id
            self.prompt_text = prompt_text
            super().__init__(session_id)
    
    class NotesToggled(ChatMessage):
        """Notes area was expanded/collapsed."""
        
        def __init__(self, expanded: bool, session_id: str = "default") -> None:
            """Initialize notes toggle message.
            
            Args:
                expanded: Whether notes are expanded
                session_id: The chat session ID
            """
            self.expanded = expanded
            super().__init__(session_id)


class ChatMessageDisplayMessage(ChatMessage):
    """Messages related to message display/management."""
    
    class MessageAdded(ChatMessage):
        """New message added to chat."""
        
        def __init__(self, message_id: str, content: str, role: str, session_id: str = "default") -> None:
            """Initialize message added event.
            
            Args:
                message_id: Unique message ID
                content: Message content
                role: Message role (user/assistant/system)
                session_id: The chat session ID
            """
            self.message_id = message_id
            self.content = content
            self.role = role
            super().__init__(session_id)
    
    class MessageUpdated(ChatMessage):
        """Message content updated."""
        
        def __init__(self, message_id: str, new_content: str, session_id: str = "default") -> None:
            """Initialize message updated event.
            
            Args:
                message_id: ID of the updated message
                new_content: New message content
                session_id: The chat session ID
            """
            self.message_id = message_id
            self.new_content = new_content
            super().__init__(session_id)
    
    class MessageDeleted(ChatMessage):
        """Message was deleted."""
        
        def __init__(self, message_id: str, session_id: str = "default") -> None:
            """Initialize message deleted event.
            
            Args:
                message_id: ID of the deleted message
                session_id: The chat session ID
            """
            self.message_id = message_id
            super().__init__(session_id)
    
    class MessageFocused(ChatMessage):
        """Message received focus."""
        
        def __init__(self, message_id: str, session_id: str = "default") -> None:
            """Initialize message focused event.
            
            Args:
                message_id: ID of the focused message
                session_id: The chat session ID
            """
            self.message_id = message_id
            super().__init__(session_id)
    
    class EditRequested(ChatMessage):
        """User requested to edit a message."""
        
        def __init__(self, message_id: str, session_id: str = "default") -> None:
            """Initialize edit request.
            
            Args:
                message_id: ID of the message to edit
                session_id: The chat session ID
            """
            self.message_id = message_id
            super().__init__(session_id)


class ChatStreamingMessage(ChatMessage):
    """Messages related to streaming responses."""
    
    class StreamStarted(ChatMessage):
        """Streaming response started."""
        
        def __init__(self, message_id: str, session_id: str = "default") -> None:
            """Initialize stream started event.
            
            Args:
                message_id: ID of the streaming message
                session_id: The chat session ID
            """
            self.message_id = message_id
            super().__init__(session_id)
    
    class StreamChunk(ChatMessage):
        """Streaming chunk received."""
        
        def __init__(self, message_id: str, chunk: str, session_id: str = "default") -> None:
            """Initialize stream chunk event.
            
            Args:
                message_id: ID of the streaming message
                chunk: The text chunk
                session_id: The chat session ID
            """
            self.message_id = message_id
            self.chunk = chunk
            super().__init__(session_id)
    
    class StreamCompleted(ChatMessage):
        """Streaming response completed."""
        
        def __init__(self, message_id: str, final_content: str, session_id: str = "default") -> None:
            """Initialize stream completed event.
            
            Args:
                message_id: ID of the streaming message
                final_content: Final complete content
                session_id: The chat session ID
            """
            self.message_id = message_id
            self.final_content = final_content
            super().__init__(session_id)
    
    class StreamError(ChatMessage):
        """Streaming error occurred."""
        
        def __init__(self, message_id: str, error: str, session_id: str = "default") -> None:
            """Initialize stream error event.
            
            Args:
                message_id: ID of the streaming message
                error: Error message
                session_id: The chat session ID
            """
            self.message_id = message_id
            self.error = error
            super().__init__(session_id)