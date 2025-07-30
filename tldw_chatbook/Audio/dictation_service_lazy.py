# dictation_service_lazy.py
"""
Improved live dictation service with lazy initialization and better resource management.
This implementation addresses the issues identified in the architectural review.
"""

import os
import sys
import threading
import queue
import time
import weakref
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from loguru import logger
from contextlib import contextmanager

# Local imports
from ..config import get_cli_setting, save_setting_to_cli_config


@dataclass
class DictationResult:
    """Result from a dictation session."""
    transcript: str
    segments: List[Dict[str, Any]]
    duration: float
    audio_data: Optional[bytes] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class DictationState:
    """Enumeration of dictation states."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    STARTING = "starting"
    LISTENING = "listening"
    PROCESSING = "processing"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class LazyLiveDictationService:
    """
    Improved dictation service with lazy initialization and better resource management.
    
    Key improvements:
    - Lazy initialization of audio backends
    - Graceful degradation when hardware unavailable
    - Simplified threading for single-user app
    - Privacy-first approach to history
    - Better error messages
    """
    
    # Audio buffer settings
    BUFFER_DURATION_MS = 500  # Default, now configurable
    MIN_SPEECH_DURATION_MS = 300
    
    # Privacy settings keys
    PRIVACY_KEY_PREFIX = "dictation.privacy"
    
    def __init__(
        self,
        transcription_provider: str = 'auto',
        transcription_model: Optional[str] = None,
        language: str = 'en',
        enable_punctuation: bool = True,
        enable_commands: bool = True,
        audio_backend: Optional[str] = None
    ):
        """
        Initialize dictation service with lazy loading.
        
        Audio and transcription services are not initialized until first use.
        """
        self.transcription_provider = transcription_provider
        self.transcription_model = transcription_model
        self.language = language
        self.enable_punctuation = enable_punctuation
        self.enable_commands = enable_commands
        self.audio_backend_preference = audio_backend
        
        # Lazy-loaded services
        self._audio_service = None
        self._transcription_service = None
        self._audio_init_error = None
        self._transcription_init_error = None
        
        # State management
        self.state = DictationState.IDLE
        self.state_lock = threading.Lock()
        
        # Audio buffering
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.last_speech_time = 0
        
        # Transcription management
        self.transcript_segments = []
        self.current_transcript = ""
        self.transcript_lock = threading.Lock()
        
        # Streaming transcriber
        self.streaming_transcriber = None
        
        # Callbacks
        self.on_partial_transcript = None
        self.on_final_transcript = None
        self.on_state_change = None
        self.on_error = None
        self.on_command = None
        
        # Processing thread
        self.processing_thread = None
        self.processing_queue = queue.Queue()
        self.stop_processing = threading.Event()
        
        # Statistics
        self.start_time = None
        self.total_duration = 0
        
        # Privacy settings
        self._load_privacy_settings()
        
        # Buffer configuration
        self.buffer_duration_ms = get_cli_setting(
            'dictation.buffer_duration_ms', 
            self.BUFFER_DURATION_MS
        )
        
        logger.info(
            f"LazyLiveDictationService initialized (services will load on demand) "
            f"provider: {transcription_provider}, privacy: {self.privacy_settings}"
        )
    
    def _load_privacy_settings(self):
        """Load privacy settings from configuration."""
        self.privacy_settings = {
            'save_history': get_cli_setting(f'{self.PRIVACY_KEY_PREFIX}.save_history', False),
            'encrypt_history': get_cli_setting(f'{self.PRIVACY_KEY_PREFIX}.encrypt_history', True),
            'local_only': get_cli_setting(f'{self.PRIVACY_KEY_PREFIX}.local_only', True),
            'auto_clear_buffer': get_cli_setting(f'{self.PRIVACY_KEY_PREFIX}.auto_clear_buffer', True),
        }
    
    def update_privacy_settings(self, settings: Dict[str, Any]):
        """Update privacy settings."""
        for key, value in settings.items():
            if key in self.privacy_settings:
                self.privacy_settings[key] = value
                save_setting_to_cli_config(
                    'dictation.privacy', 
                    key.replace('_', '.'), 
                    value
                )
        logger.info(f"Privacy settings updated: {self.privacy_settings}")
    
    @property
    def audio_service(self):
        """Lazy-load audio recording service."""
        if self._audio_service is None and self._audio_init_error is None:
            try:
                from .recording_service import AudioRecordingService
                
                # Try to initialize with preferences
                self._audio_service = AudioRecordingService(
                    backend=self.audio_backend_preference,
                    use_vad=True,
                    vad_aggressiveness=2,
                    chunk_size=int(self.buffer_duration_ms * 16)  # 16 samples/ms at 16kHz
                )
                logger.info("Audio recording service initialized successfully")
            except Exception as e:
                self._audio_init_error = str(e)
                logger.error(f"Failed to initialize audio service: {e}")
                raise AudioInitializationError(
                    "Unable to access microphone. Please check:\n"
                    "• Microphone is connected\n"
                    "• App has microphone permissions\n"
                    "• No other app is using the microphone\n"
                    f"\nTechnical details: {e}"
                )
        elif self._audio_init_error:
            raise AudioInitializationError(self._audio_init_error)
        
        return self._audio_service
    
    @property
    def transcription_service(self):
        """Lazy-load transcription service."""
        if self._transcription_service is None and self._transcription_init_error is None:
            try:
                from ..Local_Ingestion.transcription_service import TranscriptionService
                
                self._transcription_service = TranscriptionService()
                logger.info("Transcription service initialized successfully")
            except Exception as e:
                self._transcription_init_error = str(e)
                logger.error(f"Failed to initialize transcription service: {e}")
                raise TranscriptionInitializationError(
                    "Unable to initialize transcription. Please check:\n"
                    "• Required models are installed\n"
                    "• Sufficient disk space available\n"
                    f"\nTechnical details: {e}"
                )
        elif self._transcription_init_error:
            raise TranscriptionInitializationError(self._transcription_init_error)
        
        return self._transcription_service
    
    def start_dictation(
        self,
        on_partial_transcript: Optional[Callable[[str], None]] = None,
        on_final_transcript: Optional[Callable[[str], None]] = None,
        on_state_change: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_command: Optional[Callable[[str], None]] = None,
        save_audio: bool = False
    ) -> bool:
        """
        Start live dictation with improved initialization.
        """
        with self.state_lock:
            if self.state != DictationState.IDLE:
                logger.warning(f"Cannot start dictation in state: {self.state}")
                return False
            
            self.state = DictationState.INITIALIZING
        
        try:
            # Set callbacks first
            self.on_partial_transcript = on_partial_transcript
            self.on_final_transcript = on_final_transcript
            self.on_state_change = on_state_change
            self.on_error = on_error
            self.on_command = on_command
            
            self._notify_state_change()
            
            # Initialize services (lazy loading happens here)
            try:
                # This will trigger lazy initialization
                audio_svc = self.audio_service
                trans_svc = self.transcription_service
            except (AudioInitializationError, TranscriptionInitializationError) as e:
                self._notify_error(e)
                with self.state_lock:
                    self.state = DictationState.ERROR
                self._notify_state_change()
                return False
            
            with self.state_lock:
                self.state = DictationState.STARTING
            self._notify_state_change()
            
            # Reset state
            self.transcript_segments = []
            self.current_transcript = ""
            self.audio_buffer = []
            self.start_time = time.time()
            self.save_audio = save_audio and not self.privacy_settings['local_only']
            
            # Initialize streaming transcriber
            self._initialize_streaming_transcriber()
            
            # Start processing thread (simplified)
            self.stop_processing.clear()
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True,
                name="DictationProcessor"
            )
            self.processing_thread.start()
            
            # Start audio recording
            success = audio_svc.start_recording(
                callback=self._audio_callback
            )
            
            if success:
                with self.state_lock:
                    self.state = DictationState.LISTENING
                self._notify_state_change()
                logger.info("Started live dictation")
                return True
            else:
                self._cleanup()
                self._notify_error(
                    Exception("Failed to start audio recording. Please check your microphone.")
                )
                return False
        
        except Exception as e:
            logger.error(f"Failed to start dictation: {e}")
            self._cleanup()
            self._notify_error(e)
            return False
    
    def _initialize_streaming_transcriber(self):
        """Initialize streaming transcriber if available."""
        if self.privacy_settings['local_only']:
            # Only use local providers when in privacy mode
            allowed_providers = ['parakeet-mlx', 'faster-whisper', 'lightning-whisper']
            if self.transcription_provider not in allowed_providers:
                logger.info(
                    f"Provider '{self.transcription_provider}' not allowed in privacy mode. "
                    f"Using local provider instead."
                )
                self.transcription_provider = 'parakeet-mlx'  # Default local provider
        
        try:
            self.streaming_transcriber = self.transcription_service.create_streaming_transcriber(
                provider=self.transcription_provider,
                model=self.transcription_model,
                language=self.language
            )
            
            if self.streaming_transcriber:
                logger.info("Streaming transcriber initialized")
            else:
                logger.info("Streaming not available, will use chunked transcription")
        
        except Exception as e:
            logger.warning(f"Failed to initialize streaming transcriber: {e}")
            self.streaming_transcriber = None
    
    def _audio_callback(self, audio_chunk: bytes):
        """Callback for audio chunks with auto-clear if privacy enabled."""
        try:
            # Add to buffer
            with self.buffer_lock:
                self.audio_buffer.append(audio_chunk)
            
            # Queue for processing
            self.processing_queue.put(('audio', audio_chunk))
            
            # Update last speech time
            self.last_speech_time = time.time()
        
        except Exception as e:
            logger.error(f"Audio callback error: {e}")
    
    def _processing_loop(self):
        """Simplified processing loop for single-user app."""
        accumulated_audio = []
        last_process_time = time.time()
        
        while not self.stop_processing.is_set():
            try:
                # Get items from queue with timeout
                try:
                    item_type, data = self.processing_queue.get(timeout=0.1)
                    
                    if item_type == 'audio':
                        accumulated_audio.append(data)
                
                except queue.Empty:
                    pass
                
                # Process accumulated audio periodically
                current_time = time.time()
                buffer_duration_sec = self.buffer_duration_ms / 1000
                
                if accumulated_audio and (current_time - last_process_time) >= buffer_duration_sec:
                    audio_data = b''.join(accumulated_audio)
                    self._process_audio_buffer(audio_data)
                    
                    # Clear accumulated audio
                    accumulated_audio = []
                    last_process_time = current_time
                    
                    # Auto-clear buffer if privacy enabled
                    if self.privacy_settings['auto_clear_buffer']:
                        with self.buffer_lock:
                            # Keep only last few chunks for context
                            if len(self.audio_buffer) > 10:
                                self.audio_buffer = self.audio_buffer[-5:]
                
                # Check for silence timeout
                if self.last_speech_time and (current_time - self.last_speech_time) > 2.0:
                    # Finalize current segment after 2 seconds of silence
                    self._finalize_current_segment()
                    self.last_speech_time = 0
            
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                self._notify_error(e)
    
    def _cleanup(self):
        """Clean up resources with privacy considerations."""
        with self.state_lock:
            self.state = DictationState.IDLE
        
        # Clear sensitive data immediately if privacy mode
        if self.privacy_settings['auto_clear_buffer']:
            self.audio_buffer = []
            if not self.privacy_settings['save_history']:
                self.transcript_segments = []
                self.current_transcript = ""
        
        self.streaming_transcriber = None
        
        # Note: We don't clear the lazy-loaded services themselves
        # They can be reused for the next session
        
        self._notify_state_change()
    
    def get_audio_devices(self) -> List[Dict[str, Any]]:
        """Get available audio input devices with error handling."""
        try:
            return self.audio_service.get_audio_devices()
        except AudioInitializationError as e:
            logger.error(f"Cannot get audio devices: {e}")
            return []
    
    def set_buffer_duration(self, duration_ms: int):
        """Set audio buffer duration dynamically."""
        self.buffer_duration_ms = max(100, min(2000, duration_ms))  # Clamp between 100-2000ms
        save_setting_to_cli_config('dictation', 'buffer_duration_ms', self.buffer_duration_ms)
        logger.info(f"Buffer duration set to {self.buffer_duration_ms}ms")
    
    # Include other methods from original implementation with appropriate modifications...
    # (I'm including the key ones here, others remain similar)
    
    def _notify_error(self, error: Exception):
        """Notify error callback with sanitized error messages."""
        with self.state_lock:
            self.state = DictationState.ERROR
        
        # Sanitize error message to remove sensitive paths
        safe_error = type(error)(str(error).replace(os.path.expanduser('~'), '~'))
        
        if self.on_error:
            try:
                self.on_error(safe_error)
            except Exception as e:
                logger.error(f"Error callback error: {e}")


class AudioInitializationError(Exception):
    """Raised when audio initialization fails with user-friendly message."""
    pass


class TranscriptionInitializationError(Exception):
    """Raised when transcription initialization fails with user-friendly message."""
    pass