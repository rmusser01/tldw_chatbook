# dictation_events.py
"""
Event handlers for live dictation functionality.
"""

from typing import Optional
from textual.message import Message


class DictationEvent(Message):
    """Base class for dictation events."""

    pass


class VoiceCommandEvent(DictationEvent):
    """Event fired when a voice command is detected."""

    def __init__(self, command: str, raw_text: str):
        super().__init__()
        self.command = command
        self.raw_text = raw_text
