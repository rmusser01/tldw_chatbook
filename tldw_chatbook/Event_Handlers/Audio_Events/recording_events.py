# recording_events.py
"""
Event handlers for audio recording functionality.
"""

from typing import Optional
from textual.message import Message
from loguru import logger


class AudioRecordingEvent(Message):
    """Base class for audio recording events."""
    pass


class RecordingStartedEvent(AudioRecordingEvent):
    """Event fired when audio recording starts."""
    
    def __init__(self, device_name: Optional[str] = None):
        super().__init__()
        self.device_name = device_name


class RecordingStoppedEvent(AudioRecordingEvent):
    """Event fired when audio recording stops."""
    
    def __init__(self, duration: float, audio_data: Optional[bytes] = None):
        super().__init__()
        self.duration = duration
        self.audio_data = audio_data


class RecordingErrorEvent(AudioRecordingEvent):
    """Event fired when a recording error occurs."""
    
    def __init__(self, error: Exception):
        super().__init__()
        self.error = error


class AudioLevelUpdateEvent(AudioRecordingEvent):
    """Event fired with audio level updates."""
    
    def __init__(self, level: float):
        super().__init__()
        self.level = level  # 0.0 to 1.0


class AudioDeviceChangedEvent(AudioRecordingEvent):
    """Event fired when audio device changes."""
    
    def __init__(self, device_id: Optional[int], device_name: str):
        super().__init__()
        self.device_id = device_id
        self.device_name = device_name