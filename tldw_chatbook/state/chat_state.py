"""
Chat state management.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class ChatSession:
    """Represents a single chat session."""
    
    id: str
    conversation_id: Optional[int] = None
    is_ephemeral: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    
    # Chat content
    messages: List[Dict[str, Any]] = field(default_factory=list)
    character_data: Optional[Dict[str, Any]] = None
    
    # Session metadata
    title: str = ""
    keywords: List[str] = field(default_factory=list)
    
    def add_message(self, role: str, content: str, **kwargs) -> None:
        """Add a message to the session."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.messages.append(message)
    
    def clear_messages(self) -> None:
        """Clear all messages from the session."""
        self.messages.clear()
    
    def to_persistent(self, conversation_id: int) -> None:
        """Convert ephemeral session to persistent."""
        self.is_ephemeral = False
        self.conversation_id = conversation_id


@dataclass
class ChatState:
    """Manages all chat-related state."""
    
    # Provider settings
    provider: str = "openai"
    model: str = "gpt-4"
    
    # Active session
    active_session_id: Optional[str] = None
    sessions: Dict[str, ChatSession] = field(default_factory=dict)
    
    # UI state
    sidebar_collapsed: bool = False
    right_sidebar_collapsed: bool = False
    sidebar_width: int = 30
    
    # Prompt management
    selected_prompt_id: Optional[int] = None
    loaded_prompt: Optional[Dict[str, Any]] = None
    
    # Streaming state
    is_streaming: bool = False
    current_stream_id: Optional[str] = None
    
    def create_session(self, session_id: str) -> ChatSession:
        """Create a new chat session."""
        session = ChatSession(id=session_id)
        self.sessions[session_id] = session
        self.active_session_id = session_id
        return session
    
    def get_active_session(self) -> Optional[ChatSession]:
        """Get the currently active session."""
        if self.active_session_id:
            return self.sessions.get(self.active_session_id)
        return None
    
    def delete_session(self, session_id: str) -> None:
        """Delete a chat session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            if self.active_session_id == session_id:
                self.active_session_id = None
    
    def switch_session(self, session_id: str) -> Optional[ChatSession]:
        """Switch to a different session."""
        if session_id in self.sessions:
            self.active_session_id = session_id
            return self.sessions[session_id]
        return None
    
    def set_provider(self, provider: str, model: str) -> None:
        """Update provider and model."""
        self.provider = provider
        self.model = model
    
    def toggle_sidebar(self, which: str = "left") -> bool:
        """Toggle sidebar visibility."""
        if which == "left":
            self.sidebar_collapsed = not self.sidebar_collapsed
            return self.sidebar_collapsed
        else:
            self.right_sidebar_collapsed = not self.right_sidebar_collapsed
            return self.right_sidebar_collapsed