# recording_service.py
"""
Cross-platform audio recording service for live dictation and speech capture.
Supports PyAudio as primary backend with sounddevice as fallback.
"""

import os
import sys
import threading
import queue
import time
import wave
import tempfile
from typing import Optional, Callable, List, Dict, Any, Tuple
from pathlib import Path
from contextlib import contextmanager
from loguru import logger

# Try to import numpy as optional dependency
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available. Some audio processing features will be limited.")

# Try to import audio backends
PYAUDIO_AVAILABLE = False
SOUNDDEVICE_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
    logger.info("PyAudio backend available")
except ImportError:
    logger.warning("PyAudio not available. Install with: pip install pyaudio")

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
    logger.info("Sounddevice backend available")
except ImportError:
    logger.warning("Sounddevice not available. Install with: pip install sounddevice")

# Import VAD if available
VAD_AVAILABLE = False
try:
    import webrtcvad
    VAD_AVAILABLE = True
    logger.info("WebRTC VAD available for voice activity detection")
except ImportError:
    logger.warning("WebRTC VAD not available. Install with: pip install webrtcvad")


class AudioRecordingError(Exception):
    """Base exception for audio recording errors"""
    pass


class NoAudioBackendError(AudioRecordingError):
    """Raised when no audio backend is available"""
    pass


class AudioDeviceError(AudioRecordingError):
    """Raised when there's an issue with audio device"""
    pass


class AudioRecordingService:
    """
    Cross-platform audio recording service with streaming support.
    
    Features:
    - Multiple backend support (PyAudio, sounddevice)
    - Voice Activity Detection (VAD)
    - Real-time streaming callbacks
    - Device enumeration and selection
    - Automatic gain control
    """
    
    # Audio configuration defaults
    DEFAULT_SAMPLE_RATE = 16000  # 16kHz is standard for speech recognition
    DEFAULT_CHANNELS = 1  # Mono
    DEFAULT_CHUNK_SIZE = 1024  # Samples per chunk
    DEFAULT_AUDIO_FORMAT = 'int16'  # 16-bit PCM
    
    def __init__(
        self,
        backend: Optional[str] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        use_vad: bool = True,
        vad_aggressiveness: int = 2
    ):
        """
        Initialize audio recording service.
        
        Args:
            backend: Audio backend to use ('pyaudio', 'sounddevice', or None for auto)
            sample_rate: Sample rate in Hz
            channels: Number of audio channels (1 for mono, 2 for stereo)
            chunk_size: Number of samples per chunk
            use_vad: Whether to use Voice Activity Detection
            vad_aggressiveness: VAD aggressiveness (0-3, higher is more aggressive)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.use_vad = use_vad and VAD_AVAILABLE
        self.vad_aggressiveness = max(0, min(3, vad_aggressiveness))
        
        # Initialize backend
        self.backend = self._initialize_backend(backend)
        if not self.backend:
            raise NoAudioBackendError(
                "No audio backend available. Install pyaudio or sounddevice."
            )
        
        # Recording state
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        self.callback = None
        
        # Device info
        self.current_device_id = None
        self.device_info = None
        
        # VAD setup
        self.vad = None
        if self.use_vad:
            try:
                self.vad = webrtcvad.Vad()
                self.vad.set_mode(self.vad_aggressiveness)
                logger.info(f"VAD initialized with aggressiveness {self.vad_aggressiveness}")
            except Exception as e:
                logger.warning(f"Failed to initialize VAD: {e}")
                self.use_vad = False
        
        # Audio stream
        self.stream = None
        self.pyaudio_instance = None
        
        logger.info(f"AudioRecordingService initialized with backend: {self.backend}")
    
    def _initialize_backend(self, backend: Optional[str]) -> Optional[str]:
        """Initialize and select audio backend."""
        if backend:
            backend = backend.lower()
            if backend == 'pyaudio' and PYAUDIO_AVAILABLE:
                return 'pyaudio'
            elif backend == 'sounddevice' and SOUNDDEVICE_AVAILABLE:
                return 'sounddevice'
            else:
                logger.warning(f"Requested backend '{backend}' not available")
        
        # Auto-select backend
        if PYAUDIO_AVAILABLE:
            return 'pyaudio'
        elif SOUNDDEVICE_AVAILABLE:
            return 'sounddevice'
        
        return None
    
    def get_audio_devices(self) -> List[Dict[str, Any]]:
        """
        Get list of available audio input devices.
        
        Returns:
            List of device info dictionaries
        """
        devices = []
        
        try:
            if self.backend == 'pyaudio':
                if not self.pyaudio_instance:
                    self.pyaudio_instance = pyaudio.PyAudio()
                
                for i in range(self.pyaudio_instance.get_device_count()):
                    info = self.pyaudio_instance.get_device_info_by_index(i)
                    if info['maxInputChannels'] > 0:
                        devices.append({
                            'id': i,
                            'name': info['name'],
                            'channels': info['maxInputChannels'],
                            'sample_rate': int(info['defaultSampleRate']),
                            'is_default': i == self.pyaudio_instance.get_default_input_device_info()['index']
                        })
            
            elif self.backend == 'sounddevice':
                for i, device in enumerate(sd.query_devices()):
                    if device['max_input_channels'] > 0:
                        devices.append({
                            'id': i,
                            'name': device['name'],
                            'channels': device['max_input_channels'],
                            'sample_rate': int(device['default_samplerate']),
                            'is_default': i == sd.default.device[0]
                        })
        
        except Exception as e:
            logger.error(f"Error getting audio devices: {e}")
        
        return devices
    
    def set_device(self, device_id: Optional[int] = None) -> bool:
        """
        Set the active recording device.
        
        Args:
            device_id: Device ID or None for default
            
        Returns:
            True if successful
        """
        try:
            if self.is_recording:
                logger.warning("Cannot change device while recording")
                return False
            
            self.current_device_id = device_id
            
            # Validate device
            if device_id is not None:
                devices = self.get_audio_devices()
                if not any(d['id'] == device_id for d in devices):
                    logger.error(f"Invalid device ID: {device_id}")
                    return False
            
            logger.info(f"Set recording device to: {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting device: {e}")
            return False
    
    def start_recording(
        self,
        callback: Optional[Callable[[bytes], None]] = None,
        save_to_file: Optional[str] = None
    ) -> bool:
        """
        Start recording audio from microphone.
        
        Args:
            callback: Optional callback function for audio chunks
            save_to_file: Optional file path to save recording
            
        Returns:
            True if recording started successfully
        """
        if self.is_recording:
            logger.warning("Already recording")
            return False
        
        try:
            self.callback = callback
            self.save_file = save_to_file
            self.audio_buffer = []
            self.is_recording = True
            
            # Start recording thread
            self.recording_thread = threading.Thread(
                target=self._recording_loop,
                daemon=True
            )
            self.recording_thread.start()
            
            logger.info("Started audio recording")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.is_recording = False
            return False
    
    def _recording_loop(self):
        """Main recording loop running in separate thread."""
        try:
            if self.backend == 'pyaudio':
                self._pyaudio_recording_loop()
            elif self.backend == 'sounddevice':
                self._sounddevice_recording_loop()
        except Exception as e:
            logger.error(f"Recording error: {e}")
            self.is_recording = False
    
    def _pyaudio_recording_loop(self):
        """PyAudio-specific recording loop."""
        if not self.pyaudio_instance:
            self.pyaudio_instance = pyaudio.PyAudio()
        
        try:
            # Open stream
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.current_device_id,
                frames_per_buffer=self.chunk_size,
                stream_callback=None
            )
            
            logger.info("PyAudio stream opened")
            
            while self.is_recording:
                try:
                    # Read audio chunk
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Process chunk
                    self._process_audio_chunk(data)
                    
                except Exception as e:
                    logger.error(f"Error reading audio: {e}")
                    time.sleep(0.01)
        
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            logger.info("PyAudio stream closed")
    
    def _sounddevice_recording_loop(self):
        """Sounddevice-specific recording loop."""
        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Sounddevice status: {status}")
            
            if self.is_recording:
                # Convert float32 to int16
                if NUMPY_AVAILABLE:
                    audio_data = (indata * 32767).astype(np.int16).tobytes()
                else:
                    # Fallback without numpy - convert manually
                    import struct
                    audio_data = b''
                    for sample in indata.flatten():
                        # Clamp to int16 range and pack
                        int_sample = int(max(-32768, min(32767, sample * 32767)))
                        audio_data += struct.pack('<h', int_sample)
                self._process_audio_chunk(audio_data)
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                device=self.current_device_id,
                callback=audio_callback,
                blocksize=self.chunk_size,
                dtype='float32'
            ):
                logger.info("Sounddevice stream opened")
                
                while self.is_recording:
                    time.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Sounddevice error: {e}")
            self.is_recording = False
    
    def _process_audio_chunk(self, chunk: bytes):
        """Process audio chunk with optional VAD filtering."""
        # Apply VAD if enabled
        if self.use_vad and self.vad:
            # VAD requires 16-bit PCM at specific frame sizes
            # For 16kHz: 10, 20, or 30 ms frames
            frame_duration_ms = 20
            frame_size = int(self.sample_rate * frame_duration_ms / 1000) * 2  # 2 bytes per sample
            
            # Process in VAD-compatible frames
            for i in range(0, len(chunk) - frame_size, frame_size):
                frame = chunk[i:i + frame_size]
                if self.vad.is_speech(frame, self.sample_rate):
                    self._handle_audio_chunk(frame)
        else:
            # No VAD, process entire chunk
            self._handle_audio_chunk(chunk)
    
    def _handle_audio_chunk(self, chunk: bytes):
        """Handle processed audio chunk."""
        # Add to buffer
        self.audio_buffer.append(chunk)
        
        # Add to queue
        self.audio_queue.put(chunk)
        
        # Call callback if provided
        if self.callback:
            try:
                self.callback(chunk)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def stop_recording(self) -> Optional[bytes]:
        """
        Stop recording and return audio data.
        
        Returns:
            Recorded audio data as bytes, or None if not recording
        """
        if not self.is_recording:
            logger.warning("Not currently recording")
            return None
        
        logger.info("Stopping audio recording")
        self.is_recording = False
        
        # Wait for recording thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        # Get all audio data
        audio_data = b''.join(self.audio_buffer)
        
        # Save to file if requested
        if self.save_file and audio_data:
            self._save_audio_file(audio_data, self.save_file)
        
        # Cleanup
        self.audio_buffer = []
        self.callback = None
        
        # Close PyAudio if needed
        if self.backend == 'pyaudio' and self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            self.pyaudio_instance = None
        
        return audio_data
    
    def _save_audio_file(self, audio_data: bytes, filename: str):
        """Save audio data to WAV file."""
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)
            
            logger.info(f"Saved audio to: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save audio file: {e}")
    
    def get_audio_level(self) -> float:
        """
        Get current audio input level (0.0 to 1.0).
        
        Returns:
            Normalized audio level
        """
        try:
            # Get recent audio data
            recent_chunks = []
            while not self.audio_queue.empty() and len(recent_chunks) < 5:
                recent_chunks.append(self.audio_queue.get_nowait())
            
            if not recent_chunks:
                return 0.0
            
            # Re-queue chunks
            for chunk in recent_chunks:
                self.audio_queue.put(chunk)
            
            # Calculate RMS
            if NUMPY_AVAILABLE:
                audio_data = np.frombuffer(b''.join(recent_chunks), dtype=np.int16)
                rms = np.sqrt(np.mean(audio_data**2))
            else:
                # Fallback without numpy - calculate RMS manually
                import struct
                audio_bytes = b''.join(recent_chunks)
                num_samples = len(audio_bytes) // 2
                if num_samples == 0:
                    return 0.0
                
                sum_squares = 0
                for i in range(0, len(audio_bytes), 2):
                    sample = struct.unpack('<h', audio_bytes[i:i+2])[0]
                    sum_squares += sample * sample
                
                rms = (sum_squares / num_samples) ** 0.5
            
            # Normalize (16-bit max is 32767)
            level = min(1.0, rms / 32767.0)
            
            return level
            
        except Exception:
            return 0.0
    
    def is_available(self) -> bool:
        """Check if audio recording is available."""
        return self.backend is not None
    
    @contextmanager
    def recording_session(self, callback: Optional[Callable] = None):
        """
        Context manager for recording sessions.
        
        Example:
            with recorder.recording_session() as session:
                # Recording is active here
                time.sleep(5)
            # Recording stops automatically
        """
        try:
            self.start_recording(callback=callback)
            yield self
        finally:
            self.stop_recording()
    
    def __del__(self):
        """Cleanup on deletion."""
        if self.is_recording:
            self.stop_recording()
        
        if self.backend == 'pyaudio' and self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except:
                pass