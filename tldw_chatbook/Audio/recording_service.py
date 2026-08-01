# recording_service.py
"""
Cross-platform audio recording service for live dictation and speech capture.
Supports PyAudio as primary backend with sounddevice as fallback.
"""

import threading
import queue
import time
import wave
from collections import deque
from types import SimpleNamespace
from typing import Optional, Callable, Deque, List, Dict, Any
from contextlib import contextmanager

from loguru import logger

# Try to import numpy as optional dependency for audio functionality
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available. Audio recording will use fallback methods.")
    logger.info("For better performance, install numpy: pip install numpy")

# Try to import audio backends
PYAUDIO_AVAILABLE = False
SOUNDDEVICE_AVAILABLE = False

try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
    logger.info("PyAudio backend available")
except ImportError:
    pyaudio = SimpleNamespace(PyAudio=None, paInt16=8)
    logger.warning("PyAudio not available. Install with: pip install pyaudio")

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
    logger.info("Sounddevice backend available")
except ImportError:
    sd = SimpleNamespace(
        InputStream=None,
        query_devices=lambda: [],
        default=SimpleNamespace(device=(None, None)),
    )
    logger.warning("Sounddevice not available. Install with: pip install sounddevice")

# Import VAD if available
VAD_AVAILABLE = False
try:
    import webrtcvad

    VAD_AVAILABLE = True
    logger.info("WebRTC VAD available for voice activity detection")
except ImportError:
    webrtcvad = SimpleNamespace(Vad=None)
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

    #: Class-level fallback so an instance built via `__new__` (as some
    #: tests do, to skip the constructor's device-opening backend probe)
    #: still has somewhere to put rejected frames instead of raising
    #: AttributeError from `_process_audio_chunk`. `maxlen=0` makes
    #: `.append()` a safe no-op -- nothing is ever actually retained by this
    #: shared class-level object, so there is no cross-instance state
    #: leakage despite the attribute being mutable. `__init__` always
    #: replaces this with a real per-instance deque sized from
    #: `vad_preroll_ms`.
    _preroll_frames: Deque[bytes] = deque(maxlen=0)

    # Audio configuration defaults
    DEFAULT_SAMPLE_RATE = 16000  # 16kHz is standard for speech recognition
    DEFAULT_CHANNELS = 1  # Mono
    DEFAULT_CHUNK_SIZE = 1024  # Samples per chunk
    DEFAULT_AUDIO_FORMAT = "int16"  # 16-bit PCM

    #: Frame duration `_process_audio_chunk` slices VAD input into. WebRTC
    #: VAD only accepts 10/20/30 ms frames; this service has always used 20.
    VAD_FRAME_DURATION_MS = 20

    #: How much recently-*rejected* audio `_process_audio_chunk` replays the
    #: instant VAD accepts a frame after a silence run (incident: live
    #: dictation on real hardware with parakeet-mlx transcribed "stop" as
    #: "dot"/"top"-like forms and "send" as "and" -- the leading consonant
    #: gone). At `vad_aggressiveness=3`, low-energy speech onsets --
    #: word-initial fricatives especially -- are classified as non-speech,
    #: so the first frame(s) of an utterance were dropped before
    #: transcription ever saw them. 240 ms (12 x 20 ms frames) is chosen to
    #: comfortably cover a fricative onset while staying short enough that
    #: replaying it does not meaningfully dilute VAD gating (i.e. does not
    #: risk dragging a run of ambient noise into "speech"). Configurable as
    #: `dictation.vad_preroll_ms`.
    VAD_PREROLL_MS = 240

    def __init__(
        self,
        backend: Optional[str] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        use_vad: bool = True,
        vad_aggressiveness: int = 2,
        vad_preroll_ms: int = VAD_PREROLL_MS,
        max_buffer_bytes: Optional[int] = None,
        on_buffer_limit: Optional[Callable[[], None]] = None,
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
            vad_preroll_ms: How many milliseconds of rejected audio to keep
                on hand and replay when VAD accepts a frame after a silence
                run, to recover a clipped speech onset. Negative values are
                clamped to 0 (pre-roll disabled).
            max_buffer_bytes: Optional hard limit for retained PCM bytes
            on_buffer_limit: Optional callback invoked once on a daemon
                notification thread when the limit is reached
        """
        # Check for numpy requirement first
        if not NUMPY_AVAILABLE:
            raise AudioRecordingError(
                "Audio recording functionality requires NumPy for efficient real-time processing.\n"
                "NumPy is essential for:\n"
                "  - Real-time audio level calculations\n"
                "  - Efficient audio format conversions\n"
                "  - Processing audio streams without performance issues\n\n"
                "To enable audio features, install NumPy:\n"
                "  pip install numpy\n\n"
                "Without NumPy, audio processing would cause high CPU usage and UI stuttering."
            )

        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.use_vad = use_vad and VAD_AVAILABLE
        self.vad_aggressiveness = max(0, min(3, vad_aggressiveness))
        self.max_buffer_bytes = (
            max(0, int(max_buffer_bytes)) if max_buffer_bytes is not None else None
        )
        self.on_buffer_limit = on_buffer_limit
        self.vad_preroll_ms = max(0, int(vad_preroll_ms))
        preroll_frame_count = max(
            0, round(self.vad_preroll_ms / self.VAD_FRAME_DURATION_MS)
        )
        # Ring buffer of the most recently *rejected* VAD frames, flushed
        # ahead of the next accepted frame after a silence run. See
        # `VAD_PREROLL_MS` above for the incident this exists to fix.
        self._preroll_frames: Deque[bytes] = deque(maxlen=preroll_frame_count)

        # Initialize backend
        self.backend = self._initialize_backend(backend)
        if not self.backend:
            raise NoAudioBackendError(
                "No audio backend available. Install pyaudio or sounddevice."
            )

        # Recording state
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.audio_buffer: List[bytes] = []
        self._audio_buffer_bytes = 0
        self._buffer_limit_reached = False
        self.recording_thread = None
        self.callback = None
        self.save_file = None

        # Device info
        self.current_device_id = None
        self.device_info = None
        self._last_devices: List[Dict[str, Any]] = []

        # VAD setup
        self.vad = None
        if self.use_vad:
            try:
                self.vad = webrtcvad.Vad()
                self.vad.set_mode(self.vad_aggressiveness)
                logger.info(
                    f"VAD initialized with aggressiveness {self.vad_aggressiveness}"
                )
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
            if backend == "pyaudio" and PYAUDIO_AVAILABLE:
                return "pyaudio"
            elif backend == "sounddevice" and SOUNDDEVICE_AVAILABLE:
                return "sounddevice"
            else:
                logger.warning(f"Requested backend '{backend}' not available")

        # Auto-select backend
        if PYAUDIO_AVAILABLE:
            return "pyaudio"
        elif SOUNDDEVICE_AVAILABLE:
            return "sounddevice"

        return None

    def get_audio_devices(self) -> List[Dict[str, Any]]:
        """
        Get list of available audio input devices.

        Returns:
            List of device info dictionaries
        """
        devices = []

        try:
            if self.backend == "pyaudio":
                if not self.pyaudio_instance:
                    self.pyaudio_instance = pyaudio.PyAudio()

                default_input_index = None
                try:
                    default_info = self.pyaudio_instance.get_default_input_device_info()
                    if isinstance(default_info, dict):
                        default_input_index = default_info.get("index")
                except Exception as exc:
                    logger.debug(
                        f"Could not determine default PyAudio input device: {exc}"
                    )

                for i in range(self.pyaudio_instance.get_device_count()):
                    info = self.pyaudio_instance.get_device_info_by_index(i)
                    if info["maxInputChannels"] > 0:
                        devices.append(
                            {
                                "id": i,
                                "name": info["name"],
                                "channels": info["maxInputChannels"],
                                "sample_rate": int(info["defaultSampleRate"]),
                                "is_default": i == default_input_index,
                            }
                        )

            elif self.backend == "sounddevice":
                for i, device in enumerate(sd.query_devices()):
                    if device["max_input_channels"] > 0:
                        devices.append(
                            {
                                "id": i,
                                "name": device["name"],
                                "channels": device["max_input_channels"],
                                "sample_rate": int(device["default_samplerate"]),
                                "is_default": i == sd.default.device[0],
                            }
                        )

        except Exception as e:
            logger.error(f"Error getting audio devices: {e}")
            if self._last_devices:
                return list(self._last_devices)

        if devices:
            self._last_devices = list(devices)

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
                if not any(d["id"] == device_id for d in devices):
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
        save_to_file: Optional[str] = None,
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
            self.callback = callback if callback is not None else self.callback
            self.save_file = save_to_file
            self.audio_buffer = []
            self._audio_buffer_bytes = 0
            self._buffer_limit_reached = False
            self.is_recording = True

            # Start recording thread
            self.recording_thread = threading.Thread(
                target=self._recording_loop, daemon=True
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
            if self.backend == "pyaudio":
                self._pyaudio_recording_loop()
            elif self.backend == "sounddevice":
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
                stream_callback=None,
            )

            logger.info("PyAudio stream opened")

            while self.is_recording:
                try:
                    # Read audio chunk
                    data = self.stream.read(
                        self.chunk_size, exception_on_overflow=False
                    )
                    if not isinstance(data, (bytes, bytearray)):
                        logger.error(
                            f"Audio backend returned invalid chunk type: {type(data).__name__}"
                        )
                        self.is_recording = False
                        break

                    # Process chunk
                    self._process_audio_chunk(bytes(data))

                except Exception as e:
                    logger.error(f"Error reading audio: {e}")
                    self.is_recording = False
                    break

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
                if NUMPY_AVAILABLE and np is not None:
                    audio_data = (indata * 32767).astype(np.int16).tobytes()
                else:
                    # Fallback: manual conversion
                    import struct

                    samples = struct.unpack(
                        f"{frames * self.channels}f", indata.tobytes()
                    )
                    audio_data = b"".join(
                        struct.pack("h", int(min(32767, max(-32768, sample * 32767))))
                        for sample in samples
                    )
                self._process_audio_chunk(audio_data)

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                device=self.current_device_id,
                callback=audio_callback,
                blocksize=self.chunk_size,
                dtype="float32",
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
            frame_duration_ms = self.VAD_FRAME_DURATION_MS
            frame_size = (
                int(self.sample_rate * frame_duration_ms / 1000) * 2
            )  # 2 bytes per sample

            # Process in VAD-compatible frames
            for i in range(0, len(chunk) - frame_size + 1, frame_size):
                frame = chunk[i : i + frame_size]
                if self.vad.is_speech(frame, self.sample_rate):
                    # Speech onset after a silence run: replay the buffered
                    # pre-roll frames first so the onset they contain (a
                    # fricative/plosive VAD just rejected) reaches the
                    # transcriber ahead of this frame. `_preroll_frames` is
                    # only ever non-empty here on a silence -> speech
                    # transition -- it is cleared immediately after this
                    # flush, so two consecutive accepted frames never
                    # re-flush anything.
                    if self._preroll_frames:
                        for buffered_frame in self._preroll_frames:
                            self._handle_audio_chunk(buffered_frame)
                        self._preroll_frames.clear()
                    self._handle_audio_chunk(frame)
                else:
                    # Not (yet) speech: hold onto it in case it turns out to
                    # be the clipped onset of an utterance that starts on
                    # the very next frame. `maxlen` keeps only the most
                    # recent `vad_preroll_ms` worth.
                    self._preroll_frames.append(frame)
        else:
            # No VAD, process entire chunk
            self._handle_audio_chunk(chunk)

    def _handle_audio_chunk(self, chunk: bytes):
        """Handle processed audio chunk."""
        retained = chunk
        hit_limit = False
        if self.max_buffer_bytes is not None:
            frame_bytes = 2 * self.channels
            remaining = max(0, self.max_buffer_bytes - self._audio_buffer_bytes)
            retained_bytes = min(len(chunk), remaining)
            retained_bytes -= retained_bytes % frame_bytes
            retained = chunk[:retained_bytes]
            hit_limit = retained_bytes < len(chunk) or (
                self._audio_buffer_bytes + retained_bytes + frame_bytes
                > self.max_buffer_bytes
            )

        # Add to buffer
        if retained:
            self.audio_buffer.append(retained)
            self._audio_buffer_bytes += len(retained)

        # Add to queue
        if retained:
            self.audio_queue.put(retained)

        # Call callback if provided
        if retained and self.callback:
            try:
                self.callback(retained)
            except Exception as e:
                logger.error(f"Callback error: {e}")

        if hit_limit:
            self.is_recording = False
            if not self._buffer_limit_reached:
                self._buffer_limit_reached = True
                self._notify_buffer_limit()

    def _notify_buffer_limit(self) -> None:
        """Invoke the buffer-limit callback away from the recording thread."""
        callback = self.on_buffer_limit
        if callback is None:
            return

        def invoke() -> None:
            try:
                callback()
            except Exception as e:
                logger.error(f"Buffer limit callback error: {e}")

        threading.Thread(
            target=invoke,
            name="AudioBufferLimitCallback",
            daemon=True,
        ).start()

    def stop_recording(self) -> Optional[bytes]:
        """
        Stop recording and return audio data.

        Returns:
            Recorded audio data as bytes, or None if not recording
        """
        if not self.is_recording and not getattr(self, "audio_buffer", None):
            logger.warning("Not currently recording")
            return None

        logger.info("Stopping audio recording")
        self.is_recording = False

        # Wait for recording thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)

        # Get all audio data
        audio_data = b"".join(self.audio_buffer)

        # Save to file if requested
        if self.save_file and audio_data:
            self._save_audio_file(audio_data, self.save_file)

        # Cleanup
        self.audio_buffer = []
        self._audio_buffer_bytes = 0
        self.callback = None

        # Close PyAudio if needed
        if self.backend == "pyaudio" and self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            self.pyaudio_instance = None

        return audio_data

    def _save_audio_file(self, audio_data: bytes, filename: str):
        """Save audio data to WAV file."""
        try:
            with wave.open(filename, "wb") as wf:
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

            # Calculate peak level for responsive UI metering.
            combined_audio = b"".join(recent_chunks)
            if NUMPY_AVAILABLE and np is not None:
                audio_data = np.frombuffer(combined_audio, dtype=np.int16).astype(
                    np.float64
                )
                peak = np.max(np.abs(audio_data)) if audio_data.size else 0
            else:
                # Fallback: manual peak calculation
                import struct

                samples = struct.unpack(f"{len(combined_audio) // 2}h", combined_audio)
                peak = max((abs(sample) for sample in samples), default=0)

            # Normalize (16-bit max is 32767)
            level = min(1.0, peak / 32767.0)

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
        if getattr(self, "is_recording", False):
            self.stop_recording()

        if getattr(self, "backend", None) == "pyaudio" and getattr(
            self, "pyaudio_instance", None
        ):
            try:
                self.pyaudio_instance.terminate()
            except Exception:
                pass
