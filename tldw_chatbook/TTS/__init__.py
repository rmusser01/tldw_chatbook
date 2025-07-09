# TTS Module
# This module provides text-to-speech functionality with support for multiple providers

from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest, NormalizationOptions
from tldw_chatbook.TTS.TTS_Backends import TTSBackendBase, TTSBackendManager
from tldw_chatbook.TTS.TTS_Generation import TTSService, get_tts_service, close_tts_resources

__all__ = [
    # Schemas
    "OpenAISpeechRequest",
    "NormalizationOptions",
    # Backend classes
    "TTSBackendBase",
    "TTSBackendManager",
    # Service
    "TTSService",
    "get_tts_service",
    "close_tts_resources",
]