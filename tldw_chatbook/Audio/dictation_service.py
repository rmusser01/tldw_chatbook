# dictation_service.py
"""
Live dictation service combining audio recording with real-time transcription.
Provides a high-level interface for speech-to-text functionality.
"""

import os
import sys
import threading
import queue
import time
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

# Import numpy as required dependency
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    raise ImportError(
        "NumPy is required for audio dictation functionality.\n"
        "Please install it with: pip install numpy"
    )

# Local imports
from .recording_service import AudioRecordingService, AudioRecordingError
from ..Local_Ingestion.transcription_service import TranscriptionService


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
    STARTING = "starting"
    LISTENING = "listening"
    PROCESSING = "processing"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class LiveDictationService:
    """
    Combines audio recording with real-time transcription for live dictation.
    
    Features:
    - Real-time speech-to-text
    - Multiple transcription provider support
    - Streaming transcription updates
    - Pause/resume functionality
    - Command detection
    - Automatic punctuation
    """
    
    # Audio buffer settings
    BUFFER_DURATION_MS = 500  # Milliseconds of audio to buffer
    MIN_SPEECH_DURATION_MS = 300  # Minimum speech duration to process
    
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
        Initialize live dictation service.
        
        Args:
            transcription_provider: Provider for transcription ('auto', 'parakeet-mlx', etc.)
            transcription_model: Specific model to use
            language: Language code for transcription
            enable_punctuation: Whether to add automatic punctuation
            enable_commands: Whether to detect voice commands
            audio_backend: Audio backend to use (None for auto)
        """
        self.transcription_provider = transcription_provider
        self.transcription_model = transcription_model
        self.language = language
        self.enable_punctuation = enable_punctuation
        self.enable_commands = enable_commands
        
        # Initialize services
        try:
            self.audio_service = AudioRecordingService(
                backend=audio_backend,
                use_vad=True,
                vad_aggressiveness=2
            )
        except AudioRecordingError as e:
            logger.error(f"Failed to initialize audio service: {e}")
            raise
        
        self.transcription_service = TranscriptionService()
        
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
        
        logger.info(f"LiveDictationService initialized with provider: {transcription_provider}")
    
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
        Start live dictation with transcription callbacks.
        
        Args:
            on_partial_transcript: Callback for partial transcription updates
            on_final_transcript: Callback for final transcription segments
            on_state_change: Callback for state changes
            on_error: Callback for errors
            on_command: Callback for detected commands
            save_audio: Whether to save audio data
            
        Returns:
            True if dictation started successfully
        """
        with self.state_lock:
            if self.state != DictationState.IDLE:
                logger.warning(f"Cannot start dictation in state: {self.state}")
                return False
            
            self.state = DictationState.STARTING
        
        try:
            # Set callbacks
            self.on_partial_transcript = on_partial_transcript
            self.on_final_transcript = on_final_transcript
            self.on_state_change = on_state_change
            self.on_error = on_error
            self.on_command = on_command
            
            # Reset state
            self.transcript_segments = []
            self.current_transcript = ""
            self.audio_buffer = []
            self.start_time = time.time()
            self.save_audio = save_audio
            
            # Initialize streaming transcriber
            self._initialize_streaming_transcriber()
            
            # Start processing thread
            self.stop_processing.clear()
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True
            )
            self.processing_thread.start()
            
            # Start audio recording
            success = self.audio_service.start_recording(
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
                return False
        
        except Exception as e:
            logger.error(f"Failed to start dictation: {e}")
            self._cleanup()
            self._notify_error(e)
            return False
    
    def _initialize_streaming_transcriber(self):
        """Initialize streaming transcriber if available."""
        try:
            # Create streaming transcriber
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
        """Callback for audio chunks from recording service."""
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
        """Main processing loop for transcription."""
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
                if accumulated_audio and (current_time - last_process_time) >= (self.BUFFER_DURATION_MS / 1000):
                    audio_data = b''.join(accumulated_audio)
                    self._process_audio_buffer(audio_data)
                    accumulated_audio = []
                    last_process_time = current_time
                
                # Check for silence timeout
                if self.last_speech_time and (current_time - self.last_speech_time) > 2.0:
                    # Finalize current segment after 2 seconds of silence
                    self._finalize_current_segment()
                    self.last_speech_time = 0
            
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                self._notify_error(e)
    
    def _process_audio_buffer(self, audio_data: bytes):
        """Process audio buffer for transcription."""
        if not audio_data or self.state != DictationState.LISTENING:
            return
        
        try:
            # Use streaming transcriber if available
            if self.streaming_transcriber:
                result = self.streaming_transcriber.process_audio(audio_data)
                if result and result.get('partial'):
                    self._handle_partial_transcript(result['partial'])
                if result and result.get('final'):
                    self._handle_final_transcript(result['final'])
            else:
                # Fallback to chunked transcription
                # Save audio data to temporary file since TranscriptionService only accepts file paths
                import tempfile
                import wave
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    try:
                        # Write audio data to WAV file
                        with wave.open(tmp_file.name, 'wb') as wf:
                            wf.setnchannels(self.audio_service.channels)
                            wf.setsampwidth(2)  # 16-bit
                            wf.setframerate(self.audio_service.sample_rate)
                            wf.writeframes(audio_data)
                        
                        # Transcribe the temporary file
                        result = self.transcription_service.transcribe(
                            tmp_file.name,
                            provider=self.transcription_provider,
                            model=self.transcription_model,
                            language=self.language
                        )
                        
                        if result and result.get('text'):
                            self._handle_partial_transcript(result['text'])
                    
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(tmp_file.name)
                        except:
                            pass
        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
    
    def _handle_partial_transcript(self, text: str):
        """Handle partial transcription update."""
        if not text.strip():
            return
        
        # Process text
        if self.enable_punctuation:
            text = self._add_punctuation(text)
        
        # Check for commands
        if self.enable_commands:
            command = self._detect_command(text)
            if command:
                self._notify_command(command)
                return
        
        # Update current transcript
        with self.transcript_lock:
            self.current_transcript = text
        
        # Notify callback
        if self.on_partial_transcript:
            try:
                self.on_partial_transcript(text)
            except Exception as e:
                logger.error(f"Partial transcript callback error: {e}")
    
    def _handle_final_transcript(self, text: str):
        """Handle final transcription segment."""
        if not text.strip():
            return
        
        # Add to segments
        segment = {
            'text': text,
            'timestamp': time.time() - self.start_time,
            'duration': 0  # Will be calculated later
        }
        
        with self.transcript_lock:
            self.transcript_segments.append(segment)
            self.current_transcript = ""
        
        # Notify callback
        if self.on_final_transcript:
            try:
                self.on_final_transcript(text)
            except Exception as e:
                logger.error(f"Final transcript callback error: {e}")
    
    def _finalize_current_segment(self):
        """Finalize current transcript as a segment."""
        with self.transcript_lock:
            if self.current_transcript:
                self._handle_final_transcript(self.current_transcript)
    
    def _add_punctuation(self, text: str) -> str:
        """Add automatic punctuation to text."""
        # Simple rule-based punctuation
        # TODO: Implement more sophisticated punctuation model
        
        # Capitalize first letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Add period at end if missing
        if text and not text[-1] in '.!?':
            text += '.'
        
        return text
    
    def _detect_command(self, text: str) -> Optional[str]:
        """Detect voice commands in text."""
        text_lower = text.lower().strip()
        
        # Command patterns
        commands = {
            'new paragraph': 'new_paragraph',
            'new line': 'new_line',
            'comma': 'insert_comma',
            'period': 'insert_period',
            'question mark': 'insert_question',
            'exclamation mark': 'insert_exclamation',
            'delete last': 'delete_last',
            'clear all': 'clear_all',
            'stop dictation': 'stop_dictation'
        }
        
        for phrase, command in commands.items():
            if phrase in text_lower:
                return command
        
        return None
    
    def pause_dictation(self) -> bool:
        """
        Pause dictation temporarily.
        
        Returns:
            True if paused successfully
        """
        with self.state_lock:
            if self.state != DictationState.LISTENING:
                logger.warning(f"Cannot pause in state: {self.state}")
                return False
            
            self.state = DictationState.PAUSED
        
        # Finalize current segment
        self._finalize_current_segment()
        
        self._notify_state_change()
        logger.info("Dictation paused")
        return True
    
    def resume_dictation(self) -> bool:
        """
        Resume paused dictation.
        
        Returns:
            True if resumed successfully
        """
        with self.state_lock:
            if self.state != DictationState.PAUSED:
                logger.warning(f"Cannot resume in state: {self.state}")
                return False
            
            self.state = DictationState.LISTENING
        
        self._notify_state_change()
        logger.info("Dictation resumed")
        return True
    
    def stop_dictation(self) -> DictationResult:
        """
        Stop dictation and return final result.
        
        Returns:
            DictationResult with final transcript and metadata
        """
        with self.state_lock:
            if self.state == DictationState.IDLE:
                logger.warning("Not currently dictating")
                return DictationResult("", [], 0)
            
            self.state = DictationState.STOPPING
        
        self._notify_state_change()
        
        try:
            # Finalize current segment
            self._finalize_current_segment()
            
            # Stop processing
            self.stop_processing.set()
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)
            
            # Stop audio recording
            audio_data = self.audio_service.stop_recording()
            
            # Calculate duration
            duration = time.time() - self.start_time if self.start_time else 0
            
            # Combine all segments
            full_transcript = ' '.join(seg['text'] for seg in self.transcript_segments)
            
            # Create result
            result = DictationResult(
                transcript=full_transcript,
                segments=self.transcript_segments.copy(),
                duration=duration,
                audio_data=audio_data if self.save_audio else None
            )
            
            logger.info(f"Dictation stopped. Duration: {duration:.1f}s, Words: {len(full_transcript.split())}")
            
            return result
        
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources."""
        with self.state_lock:
            self.state = DictationState.IDLE
        
        self.streaming_transcriber = None
        self.transcript_segments = []
        self.current_transcript = ""
        self.audio_buffer = []
        
        self._notify_state_change()
    
    def _notify_state_change(self):
        """Notify state change callback."""
        if self.on_state_change:
            try:
                self.on_state_change(self.state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")
    
    def _notify_error(self, error: Exception):
        """Notify error callback."""
        with self.state_lock:
            self.state = DictationState.ERROR
        
        if self.on_error:
            try:
                self.on_error(error)
            except Exception as e:
                logger.error(f"Error callback error: {e}")
    
    def _notify_command(self, command: str):
        """Notify command detection callback."""
        if self.on_command:
            try:
                self.on_command(command)
            except Exception as e:
                logger.error(f"Command callback error: {e}")
    
    def get_state(self) -> str:
        """Get current dictation state."""
        with self.state_lock:
            return self.state
    
    def get_current_transcript(self) -> str:
        """Get current partial transcript."""
        with self.transcript_lock:
            return self.current_transcript
    
    def get_full_transcript(self) -> str:
        """Get full transcript including current partial."""
        with self.transcript_lock:
            segments_text = ' '.join(seg['text'] for seg in self.transcript_segments)
            if self.current_transcript:
                return f"{segments_text} {self.current_transcript}".strip()
            return segments_text
    
    def get_audio_devices(self) -> List[Dict[str, Any]]:
        """Get available audio input devices."""
        return self.audio_service.get_audio_devices()
    
    def set_audio_device(self, device_id: Optional[int]) -> bool:
        """Set audio input device."""
        return self.audio_service.set_device(device_id)
    
    def get_audio_level(self) -> float:
        """Get current audio input level (0.0 to 1.0)."""
        return self.audio_service.get_audio_level()
    
    def is_available(self) -> bool:
        """Check if dictation service is available."""
        return self.audio_service.is_available()