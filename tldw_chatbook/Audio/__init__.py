# Audio module for speech recording and live dictation
"""
Audio recording and dictation functionality for tldw_chatbook.
Provides cross-platform audio capture and real-time transcription.
"""

from .recording_service import AudioRecordingService, AudioRecordingError
from .dictation_service import LiveDictationService, DictationResult, DictationState

__all__ = [
    'AudioRecordingService',
    'AudioRecordingError', 
    'LiveDictationService',
    'DictationResult',
    'DictationState'
]