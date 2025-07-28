# dictation_events.py
"""
Event handlers for live dictation functionality.
"""

from typing import Optional, List, Dict, Any
from textual.message import Message
from loguru import logger


class DictationEvent(Message):
    """Base class for dictation events."""
    pass


class DictationStartedEvent(DictationEvent):
    """Event fired when dictation starts."""
    
    def __init__(self, provider: str, model: Optional[str] = None):
        super().__init__()
        self.provider = provider
        self.model = model


class DictationStoppedEvent(DictationEvent):
    """Event fired when dictation stops."""
    
    def __init__(self, transcript: str, duration: float, word_count: int):
        super().__init__()
        self.transcript = transcript
        self.duration = duration
        self.word_count = word_count


class DictationPausedEvent(DictationEvent):
    """Event fired when dictation is paused."""
    pass


class DictationResumedEvent(DictationEvent):
    """Event fired when dictation is resumed."""
    pass


class PartialTranscriptEvent(DictationEvent):
    """Event fired with partial transcript updates."""
    
    def __init__(self, text: str):
        super().__init__()
        self.text = text


class FinalTranscriptEvent(DictationEvent):
    """Event fired with final transcript segments."""
    
    def __init__(self, text: str, timestamp: float):
        super().__init__()
        self.text = text
        self.timestamp = timestamp


class DictationStateChangeEvent(DictationEvent):
    """Event fired when dictation state changes."""
    
    def __init__(self, old_state: str, new_state: str):
        super().__init__()
        self.old_state = old_state
        self.new_state = new_state


class VoiceCommandEvent(DictationEvent):
    """Event fired when a voice command is detected."""
    
    def __init__(self, command: str, raw_text: str):
        super().__init__()
        self.command = command
        self.raw_text = raw_text


class DictationErrorEvent(DictationEvent):
    """Event fired when a dictation error occurs."""
    
    def __init__(self, error: Exception):
        super().__init__()
        self.error = error