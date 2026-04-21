"""
Textual Message classes for chat events.

This module defines all the message types used in the chat system,
following Textual's message-based architecture for proper reactive updates.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from textual.message import Message


# ==================== Base Chat Messages ====================

class ChatMessage(Message):
    """Base class for all chat-related messages."""
    bubble = True  # Allow messages to bubble up the DOM


# ==================== User Action Messages ====================

class UserMessageSent(ChatMessage):
    """Posted when user sends a message."""
    
    def __init__(self, content: str, attachments: Optional[List[str]] = None):
        super().__init__()
        self.content = content
        self.attachments = attachments or []
        self.timestamp = datetime.now()


class StopGenerationRequested(ChatMessage):
    """Posted when user requests to stop generation."""
    pass


class ClearChatRequested(ChatMessage):
    """Posted when user requests to clear chat."""
    pass


class RegenerateRequested(ChatMessage):
    """Posted when user requests to regenerate last response."""
    
    def __init__(self, message_index: int):
        super().__init__()
        self.message_index = message_index


class ContinueResponseRequested(ChatMessage):
    """Posted when user wants to continue a response."""
    
    def __init__(self, message_index: int):
        super().__init__()
        self.message_index = message_index


class EditMessageRequested(ChatMessage):
    """Posted when user wants to edit a message."""
    
    def __init__(self, message_index: int, new_content: str):
        super().__init__()
        self.message_index = message_index
        self.new_content = new_content


class DeleteMessageRequested(ChatMessage):
    """Posted when user wants to delete a message."""
    
    def __init__(self, message_index: int):
        super().__init__()
        self.message_index = message_index


class CopyMessageRequested(ChatMessage):
    """Posted when user wants to copy a message."""
    
    def __init__(self, message_index: int):
        super().__init__()
        self.message_index = message_index


# ==================== LLM Response Messages ====================

class LLMResponseStarted(ChatMessage):
    """Posted when LLM starts generating a response."""
    
    def __init__(self, session_id: Optional[str] = None):
        super().__init__()
        self.session_id = session_id


class LLMResponseChunk(ChatMessage):
    """Posted when a chunk of LLM response is received."""
    
    def __init__(self, chunk: str, session_id: Optional[str] = None):
        super().__init__()
        self.chunk = chunk
        self.session_id = session_id


class LLMResponseCompleted(ChatMessage):
    """Posted when LLM finishes generating."""
    
    def __init__(self, full_response: str, session_id: Optional[str] = None):
        super().__init__()
        self.full_response = full_response
        self.session_id = session_id
        self.timestamp = datetime.now()


class LLMResponseError(ChatMessage):
    """Posted when LLM generation fails."""
    
    def __init__(self, error: str, session_id: Optional[str] = None):
        super().__init__()
        self.error = error
        self.session_id = session_id


# ==================== Session Management Messages ====================

class NewSessionRequested(ChatMessage):
    """Posted when user wants a new chat session."""
    
    def __init__(self, ephemeral: bool = False):
        super().__init__()
        self.ephemeral = ephemeral


class SaveSessionRequested(ChatMessage):
    """Posted when user wants to save current session."""
    
    def __init__(self, title: Optional[str] = None, keywords: Optional[List[str]] = None):
        super().__init__()
        self.title = title
        self.keywords = keywords


class LoadSessionRequested(ChatMessage):
    """Posted when user wants to load a session."""
    
    def __init__(self, session_id: str):
        super().__init__()
        self.session_id = session_id


class SessionLoaded(ChatMessage):
    """Posted when a session has been loaded."""
    
    def __init__(self, session_id: str, messages: List[Dict[str, Any]]):
        super().__init__()
        self.session_id = session_id
        self.messages = messages


class DeleteSessionRequested(ChatMessage):
    """Posted when user wants to delete a session."""
    
    def __init__(self, session_id: str):
        super().__init__()
        self.session_id = session_id


class CloneSessionRequested(ChatMessage):
    """Posted when user wants to clone a session."""
    
    def __init__(self, session_id: str):
        super().__init__()
        self.session_id = session_id


class ExportSessionRequested(ChatMessage):
    """Posted when user wants to export a session."""
    
    def __init__(self, session_id: str, format: str = "markdown"):
        super().__init__()
        self.session_id = session_id
        self.format = format


# ==================== Character System Messages ====================

class CharacterLoadRequested(ChatMessage):
    """Posted when user wants to load a character."""
    
    def __init__(self, character_id: int):
        super().__init__()
        self.character_id = character_id


class CharacterLoaded(ChatMessage):
    """Posted when a character has been loaded."""
    
    def __init__(self, character_id: int, character_data: Dict[str, Any]):
        super().__init__()
        self.character_id = character_id
        self.character_data = character_data


class CharacterCleared(ChatMessage):
    """Posted when character is cleared."""
    pass


# ==================== Template System Messages ====================

class TemplateApplyRequested(ChatMessage):
    """Posted when user wants to apply a template."""
    
    def __init__(self, template_name: str, template_content: str):
        super().__init__()
        self.template_name = template_name
        self.template_content = template_content


class TemplateApplied(ChatMessage):
    """Posted when a template has been applied."""
    
    def __init__(self, template_name: str):
        super().__init__()
        self.template_name = template_name


# ==================== RAG System Messages ====================

class RAGSearchRequested(ChatMessage):
    """Posted when RAG search is needed."""
    
    def __init__(self, query: str):
        super().__init__()
        self.query = query


class RAGResultsReceived(ChatMessage):
    """Posted when RAG search completes."""
    
    def __init__(self, results: List[Dict[str, Any]], context: str):
        super().__init__()
        self.results = results
        self.context = context


# ==================== File Attachment Messages ====================

class FileAttached(ChatMessage):
    """Posted when a file is attached."""
    
    def __init__(self, file_path: str, file_type: str, content: Optional[str] = None):
        super().__init__()
        self.file_path = file_path
        self.file_type = file_type
        self.content = content


class FileProcessed(ChatMessage):
    """Posted when file processing completes."""
    
    def __init__(self, file_path: str, processed_content: Any):
        super().__init__()
        self.file_path = file_path
        self.processed_content = processed_content


class FileCleared(ChatMessage):
    """Posted when attached file is cleared."""
    pass


# ==================== UI State Messages ====================

class SidebarToggled(ChatMessage):
    """Posted when sidebar visibility changes."""
    
    def __init__(self, visible: bool):
        super().__init__()
        self.visible = visible


class TabSwitched(ChatMessage):
    """Posted when chat tab is switched."""
    
    def __init__(self, tab_id: str):
        super().__init__()
        self.tab_id = tab_id


class TokenCountUpdated(ChatMessage):
    """Posted when token count changes."""
    
    def __init__(self, count: int, max_tokens: int):
        super().__init__()
        self.count = count
        self.max_tokens = max_tokens


# ==================== Error Messages ====================

class ChatError(ChatMessage):
    """Posted when an error occurs in chat."""
    
    def __init__(self, error: str, severity: str = "error"):
        super().__init__()
        self.error = error
        self.severity = severity  # "info", "warning", "error"
        self.timestamp = datetime.now()


# ==================== Tool Calling Messages ====================

class ToolCallRequested(ChatMessage):
    """Posted when a tool call is requested."""
    
    def __init__(self, tool_name: str, tool_args: Dict[str, Any]):
        super().__init__()
        self.tool_name = tool_name
        self.tool_args = tool_args


class ToolCallCompleted(ChatMessage):
    """Posted when a tool call completes."""
    
    def __init__(self, tool_name: str, result: Any):
        super().__init__()
        self.tool_name = tool_name
        self.result = result


class ToolCallFailed(ChatMessage):
    """Posted when a tool call fails."""
    
    def __init__(self, tool_name: str, error: str):
        super().__init__()
        self.tool_name = tool_name
        self.error = error


class ToolResultReceived(Message):
    """Tool execution result received."""
    def __init__(self, tool_name: str, result: Any, error: Optional[str] = None):
        self.tool_name = tool_name
        self.result = result
        self.error = error
        super().__init__()


# ==================== Conversation Management ====================

class ConversationSearchChanged(Message):
    """Conversation search query changed."""
    def __init__(self, query: str):
        self.query = query
        super().__init__()


class ConversationSearchResults(Message):
    """Conversation search results."""
    def __init__(self, results: List[Dict[str, Any]]):
        self.results = results
        super().__init__()


class ClearConversationRequested(Message):
    """Request to clear current conversation."""
    pass


class ExportConversationRequested(Message):
    """Request to export conversation."""
    def __init__(self, conversation_id: Optional[str], format: str = "markdown"):
        self.conversation_id = conversation_id
        self.format = format
        super().__init__()


class ExportConversationCompleted(Message):
    """Conversation export completed."""
    def __init__(self, filepath: str, format: str):
        self.filepath = filepath
        self.format = format
        super().__init__()


# ==================== Response Control ====================

class GenerationStopped(Message):
    """LLM generation was stopped."""
    pass


class ContinueResponseRequestedNew(Message):
    """Request to continue a partial response."""
    def __init__(self, message_id: Optional[str], partial_content: str):
        self.message_id = message_id
        self.partial_content = partial_content
        super().__init__()


# ==================== Templates & Prompts ====================

class ViewPromptRequested(Message):
    """Request to view a prompt."""
    def __init__(self, prompt_id: str):
        self.prompt_id = prompt_id
        super().__init__()


class CopyToClipboard(Message):
    """Copy content to clipboard."""
    def __init__(self, content: str, description: str = ""):
        self.content = content
        self.description = description
        super().__init__()