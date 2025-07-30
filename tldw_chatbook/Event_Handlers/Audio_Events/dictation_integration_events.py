# dictation_integration_events.py
"""
Events for integrating dictation output with other app features.
Enables voice input across the application.
"""

from typing import Optional, Dict, Any, Literal
from textual.message import Message
from dataclasses import dataclass


@dataclass
class DictationOutputEvent(Message):
    """
    Event fired when dictation produces output that should be sent elsewhere.
    
    This enables dictation results to be used in:
    - Chat conversations
    - Notes
    - Search queries
    - Any text input field
    """
    
    text: str
    target: Literal["chat", "notes", "search", "clipboard", "active_input"]
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class VoiceInputRequestEvent(Message):
    """
    Event to request voice input from any part of the app.
    The dictation service will handle this and send results back.
    """
    
    source_widget_id: str
    prompt: Optional[str] = None
    target_type: Literal["chat", "notes", "search", "general"] = "general"
    max_duration: Optional[int] = None  # Maximum recording duration in seconds
    auto_punctuation: bool = True
    language: Optional[str] = None
    

@dataclass
class VoiceInputResponseEvent(Message):
    """
    Response event containing transcribed text from voice input request.
    """
    
    source_widget_id: str
    text: str
    success: bool
    error_message: Optional[str] = None
    duration: Optional[float] = None
    word_count: Optional[int] = None


@dataclass
class DictationServiceStatusEvent(Message):
    """
    Event to broadcast dictation service status changes.
    Useful for updating UI elements that depend on dictation availability.
    """
    
    is_available: bool
    is_active: bool
    current_state: str
    error_message: Optional[str] = None
    

@dataclass
class InsertDictationTextEvent(Message):
    """
    Event to insert dictated text at the current cursor position
    in the active text input widget.
    """
    
    text: str
    append_space: bool = True
    preserve_selection: bool = False


# Voice command events for app-wide control
@dataclass
class AppVoiceCommandEvent(Message):
    """
    Voice command that controls app-level features.
    """
    
    command: str
    parameters: Optional[Dict[str, Any]] = None
    
    # Common commands
    SWITCH_TO_CHAT = "switch_to_chat"
    SWITCH_TO_NOTES = "switch_to_notes" 
    SWITCH_TO_SEARCH = "switch_to_search"
    CREATE_NEW_NOTE = "create_new_note"
    START_NEW_CONVERSATION = "start_new_conversation"
    SAVE_CURRENT = "save_current"
    CLOSE_CURRENT = "close_current"
    SHOW_HELP = "show_help"