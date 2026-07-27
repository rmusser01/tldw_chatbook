# Audio module for speech recording and live dictation
"""
Audio recording and dictation functionality for tldw_chatbook.
Provides cross-platform audio capture and real-time transcription.
"""

from importlib import import_module
from typing import Any

from .recording_service import AudioRecordingService, AudioRecordingError

__all__ = [
    "AudioRecordingService",
    "AudioRecordingError",
    "LiveDictationService",
    "DictationResult",
    "DictationState",
]

_LAZY_DICTATION_EXPORTS = {
    "LiveDictationService",
    "DictationResult",
    "DictationState",
}


def __getattr__(name: str) -> Any:
    """Load the legacy live-dictation stack only when explicitly requested."""
    if name not in _LAZY_DICTATION_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".dictation_service", __name__), name)
    globals()[name] = value
    return value
