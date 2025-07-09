# tldw_chatbook/Local_Ingestion/__init__.py
"""
Local file ingestion module for programmatic access.

This module provides functions to ingest various file types (PDFs, documents, 
e-books, audio, video, etc.) into the Media database without using the UI.
"""

from .local_file_ingestion import (
    ingest_local_file,
    batch_ingest_files,
    ingest_directory,
    quick_ingest,
    detect_file_type,
    get_supported_extensions,
    FileIngestionError
)

# Audio/Video processing classes for direct access
from .audio_processing import LocalAudioProcessor, AudioProcessingError
from .video_processing import LocalVideoProcessor, VideoProcessingError
from .transcription_service import TranscriptionService, TranscriptionError

__all__ = [
    'ingest_local_file',
    'batch_ingest_files', 
    'ingest_directory',
    'quick_ingest',
    'detect_file_type',
    'get_supported_extensions',
    'FileIngestionError',
    # Audio/Video processing
    'LocalAudioProcessor',
    'LocalVideoProcessor',
    'TranscriptionService',
    'AudioProcessingError',
    'VideoProcessingError',
    'TranscriptionError'
]