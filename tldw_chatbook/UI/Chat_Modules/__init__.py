"""
Chat Window Enhanced Modules

This package contains the modularized components of the ChatWindowEnhanced class,
following Textual best practices for separation of concerns.

Modules:
- chat_input_handler: Handles chat input and send/stop functionality
- chat_attachment_handler: Manages file attachments and image handling
- chat_voice_handler: Voice input and recording functionality
- chat_sidebar_handler: Sidebar interactions and toggling
- chat_message_manager: Message display, editing, and management
- chat_messages: Textual Message system for loose coupling
"""

from .chat_input_handler import ChatInputHandler
from .chat_attachment_handler import ChatAttachmentHandler
from .chat_voice_handler import ChatVoiceHandler
from .chat_sidebar_handler import ChatSidebarHandler
from .chat_message_manager import ChatMessageManager
from .chat_messages import (
    ChatInputMessage,
    ChatAttachmentMessage,
    ChatVoiceMessage,
    ChatSidebarMessage,
    ChatMessageDisplayMessage,
    ChatStreamingMessage
)

__all__ = [
    'ChatInputHandler',
    'ChatAttachmentHandler', 
    'ChatVoiceHandler',
    'ChatSidebarHandler',
    'ChatMessageManager',
    'ChatInputMessage',
    'ChatAttachmentMessage',
    'ChatVoiceMessage',
    'ChatSidebarMessage',
    'ChatMessageDisplayMessage',
    'ChatStreamingMessage'
]