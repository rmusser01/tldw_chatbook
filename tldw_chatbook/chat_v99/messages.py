"""Custom Textual messages for event-driven communication."""

from textual.message import Message
from typing import Optional, List
from .models import ChatMessage, ChatSession


class SessionChanged(Message):
    """Posted when active session changes."""
    def __init__(self, session: Optional[ChatSession]):
        super().__init__()
        self.session = session


class MessageSent(Message):
    """Posted when user sends a message."""
    def __init__(self, content: str, attachments: List[str] = None):
        super().__init__()
        self.content = content
        self.attachments = attachments or []


class MessageReceived(Message):
    """Posted when LLM responds."""
    def __init__(self, message: ChatMessage):
        super().__init__()
        self.message = message


class StreamingChunk(Message):
    """Posted during streaming responses."""
    def __init__(self, content: str, message_id: Optional[int] = None, done: bool = False):
        super().__init__()
        self.content = content
        self.message_id = message_id
        self.done = done


class ErrorOccurred(Message):
    """Posted when an error occurs."""
    def __init__(self, error: str, severity: str = "error"):
        super().__init__()
        self.error = error
        self.severity = severity


class SidebarToggled(Message):
    """Posted when sidebar visibility changes."""
    def __init__(self, visible: bool):
        super().__init__()
        self.visible = visible


class SessionSaved(Message):
    """Posted when a session is saved."""
    def __init__(self, session: ChatSession, path: Optional[str] = None):
        super().__init__()
        self.session = session
        self.path = path


class SessionLoaded(Message):
    """Posted when a session is loaded."""
    def __init__(self, session: ChatSession):
        super().__init__()
        self.session = session