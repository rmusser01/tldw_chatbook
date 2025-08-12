"""Data models for the chat application using Pydantic."""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any
from datetime import datetime


class ChatMessage(BaseModel):
    """Individual chat message model."""
    id: Optional[int] = None
    role: Literal["user", "assistant", "system", "tool", "tool_result"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    attachments: List[str] = Field(default_factory=list)
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


class ChatSession(BaseModel):
    """Chat session model."""
    id: Optional[int] = None
    title: str = "New Chat"
    messages: List[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Settings(BaseModel):
    """Application settings model."""
    provider: str = "openai"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    streaming: bool = True
    api_key: Optional[str] = None
    system_prompt: Optional[str] = None
    theme: str = "dark"