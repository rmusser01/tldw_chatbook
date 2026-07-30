# dictation_service_lazy.py
"""
Improved live dictation service with lazy initialization and better resource management.
This implementation addresses the issues identified in the architectural review.
"""

import math
import os
import sys
import threading
import queue
import time
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

# Optional numpy import for audio level calculation
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False
    logger.warning(
        "NumPy not available. Audio level monitoring will use fallback method. Install with: pip install numpy"
    )

# Local imports
from ..config import get_cli_setting, save_setting_to_cli_config

# One catalogue of local providers, shared with the Console's resolver. Import
# free of heavy dependencies by design (`find_spec` only), so this costs
# nothing here and cannot drift from what the resolver picks.
from ..Utils.local_stt_providers import (
    LOCAL_STT_PROVIDERS,
    installed_local_providers,
    provider_is_local,
)


@dataclass
class DictationResult:
    """Result from a dictation session.

    Attributes:
        transcript: Everything the recognizer finalized, space-joined.
        segments: The individual finalized segments, with timestamps.
        duration: Wall-clock seconds the capture ran for.
        audio_data: Raw PCM, only when the caller asked for it.
        timestamp: When the result was produced.
        captured_bytes: PCM bytes the recorder actually delivered. Zero means
            the microphone produced nothing -- the only case in which an empty
            transcript is a capture problem.
        transcription_complete: False when the processing thread was still
            working when its join expired, so audio was dropped unread. An
            empty transcript then says nothing about the microphone.
    """

    transcript: str
    segments: List[Dict[str, Any]]
    duration: float
    audio_data: Optional[bytes] = None
    timestamp: datetime = None
    captured_bytes: int = 0
    transcription_complete: bool = True

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

    #: How long `stop_dictation()` waits for the processing thread to drain
    #: and transcribe what is left. This used to be a hard-coded 2.0s, which
    #: is shorter than a single warm transcription (~1s) and orders of
    #: magnitude shorter than a cold one, so a first capture's audio was
    #: routinely thrown away with no signal at all. Configurable as
    #: `dictation.stop_join_timeout_seconds`.
    STOP_JOIN_TIMEOUT_SECONDS = 30.0

    #: How long the recorder must go without delivering a speech frame before
    #: `_processing_loop` finalizes the segment in progress. The gate is the
    #: recorder's own VAD: `AudioRecordingService._process_audio_chunk` splits
    #: capture into 20 ms frames and only hands VAD-positive ones to our
    #: callback, so "no delivery" already means "no speech". Without
    #: `webrtcvad` the recorder delivers everything and this check never
    #: fires mid-capture (finals at stop only, as before). Configurable as
    #: `dictation.silence_threshold_seconds`.
    SILENCE_THRESHOLD_SECONDS = 2.0

    #: Instance defaults, declared on the class so a service built with
    #: `__new__` (teardown-path tests do exactly that) still stops cleanly
    #: instead of raising AttributeError while releasing the microphone.
    stop_join_timeout_seconds = STOP_JOIN_TIMEOUT_SECONDS
    silence_threshold_seconds = SILENCE_THRESHOLD_SECONDS
    captured_bytes = 0
    max_buffer_bytes: Optional[int] = None
    on_buffer_limit: Optional[Callable[[], None]] = None

    # Privacy settings keys
    PRIVACY_KEY_PREFIX = "dictation.privacy"

    def __init__(
        self,
        transcription_provider: str = "auto",
        transcription_model: Optional[str] = None,
        language: str = "en",
        enable_punctuation: bool = True,
        enable_commands: bool = True,
        audio_backend: Optional[str] = None,
        max_buffer_bytes: Optional[int] = None,
        on_buffer_limit: Optional[Callable[[], None]] = None,
    ):
        """Initialize dictation service with lazy loading.

        Audio and transcription services are not initialized until first use.

        Args:
            transcription_provider: Provider name, or "auto" to let the
                transcription service choose.
            transcription_model: Model identifier for that provider.
            language: Language code passed to the recognizer.
            enable_punctuation: Whether to ask the provider to punctuate.
            enable_commands: Whether spoken commands are interpreted.
            audio_backend: Preferred capture backend, or None to auto-detect.
            max_buffer_bytes: Hard cap on the PCM the recorder retains for one
                capture. `None` (the default, and what every non-Console
                caller uses) leaves the recorder unbounded, which is only safe
                for captures bounded some other way -- a continuous capture
                accumulates ~32 KB/s in the recorder's buffer, again in its
                undrained queue, and again in `self.audio_buffer`.
            on_buffer_limit: Invoked once when `max_buffer_bytes` is reached,
                from a daemon notification thread the recorder spawns. Ignored
                unless `max_buffer_bytes` is set.
        """
        self.transcription_provider = transcription_provider
        self.transcription_model = transcription_model
        self.language = language
        self.enable_punctuation = enable_punctuation
        self.enable_commands = enable_commands
        self.audio_backend_preference = audio_backend
        self.max_buffer_bytes = max_buffer_bytes
        self.on_buffer_limit = on_buffer_limit

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
        self._current_audio_level = 0.0

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
            "dictation.buffer_duration_ms", self.BUFFER_DURATION_MS
        )
        self.stop_join_timeout_seconds = self._resolve_stop_join_timeout()
        self.silence_threshold_seconds = self._resolve_silence_threshold()

        # Bytes the recorder has handed us this session. The one fact that
        # separates "the microphone produced nothing" from "the transcriber
        # produced nothing", which callers previously had to guess at.
        self.captured_bytes = 0

        logger.info(
            f"LazyLiveDictationService initialized (services will load on demand) "
            f"provider: {transcription_provider}, privacy: {self.privacy_settings}"
        )

    @classmethod
    def _resolve_stop_join_timeout(cls) -> float:
        """Read `dictation.stop_join_timeout_seconds`, falling back to the default.

        Returns:
            A positive number of seconds to wait for the processing thread.
        """
        raw = get_cli_setting(
            "dictation.stop_join_timeout_seconds", cls.STOP_JOIN_TIMEOUT_SECONDS
        )
        try:
            timeout = float(raw)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid dictation.stop_join_timeout_seconds {!r}; using {}s",
                raw,
                cls.STOP_JOIN_TIMEOUT_SECONDS,
            )
            return cls.STOP_JOIN_TIMEOUT_SECONDS
        # `nan` and `inf` are valid TOML floats and both survive `float()`.
        # `nan <= 0` is False, so a bare positivity check waves them through --
        # and `Thread.join(timeout=nan)` raises ValueError from inside the stop
        # worker, which used to abandon a live microphone behind an idle state
        # machine. `inf` would simply hang the stop forever.
        if not math.isfinite(timeout) or timeout <= 0:
            logger.warning(
                "dictation.stop_join_timeout_seconds must be a positive, finite "
                "number (got {!r}); using {}s",
                raw,
                cls.STOP_JOIN_TIMEOUT_SECONDS,
            )
            return cls.STOP_JOIN_TIMEOUT_SECONDS
        return timeout

    @classmethod
    def _resolve_silence_threshold(cls) -> float:
        """Read `dictation.silence_threshold_seconds`, falling back to the default.

        Returns:
            A positive number of seconds of silence `_processing_loop` waits
            for before finalizing the segment in progress.
        """
        raw = get_cli_setting(
            "dictation.silence_threshold_seconds", cls.SILENCE_THRESHOLD_SECONDS
        )
        try:
            threshold = float(raw)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid dictation.silence_threshold_seconds {!r}; using {}s",
                raw,
                cls.SILENCE_THRESHOLD_SECONDS,
            )
            return cls.SILENCE_THRESHOLD_SECONDS
        # Same trap as the stop-join timeout: `nan`/`inf` are valid TOML
        # floats that survive `float()`. `nan <= 0` is False, so a bare
        # positivity check waves `nan` through to a silence check that never
        # fires; `inf` would have the same effect.
        if not math.isfinite(threshold) or threshold <= 0:
            logger.warning(
                "dictation.silence_threshold_seconds must be a positive, finite "
                "number (got {!r}); using {}s",
                raw,
                cls.SILENCE_THRESHOLD_SECONDS,
            )
            return cls.SILENCE_THRESHOLD_SECONDS
        return threshold

    def _load_privacy_settings(self):
        """Load privacy settings from configuration."""
        self.privacy_settings = {
            "save_history": get_cli_setting(
                f"{self.PRIVACY_KEY_PREFIX}.save_history", False
            ),
            "encrypt_history": get_cli_setting(
                f"{self.PRIVACY_KEY_PREFIX}.encrypt_history", True
            ),
            "local_only": get_cli_setting(
                f"{self.PRIVACY_KEY_PREFIX}.local_only", True
            ),
            "auto_clear_buffer": get_cli_setting(
                f"{self.PRIVACY_KEY_PREFIX}.auto_clear_buffer", True
            ),
        }

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
                    chunk_size=int(
                        self.buffer_duration_ms * 16
                    ),  # 16 samples/ms at 16kHz
                    # Both default to None, so every caller that does not ask
                    # for a bound gets exactly the behaviour it had before.
                    max_buffer_bytes=self.max_buffer_bytes,
                    on_buffer_limit=self.on_buffer_limit,
                )
                logger.info("Audio recording service initialized successfully")
            except Exception as e:
                self._audio_init_error = str(e)
                logger.error(f"Failed to initialize audio service: {e}")
                # Special handling for macOS permissions
                if sys.platform == "darwin" and (
                    "Invalid input device" in str(e) or "no default" in str(e).lower()
                ):
                    raise AudioInitializationError(
                        "No microphone access on macOS. Please:\n"
                        "1. Open System Settings > Privacy & Security > Microphone\n"
                        "2. Find and enable Terminal (or your IDE/Python app)\n"
                        "3. Restart this application\n"
                        "\nNote: You must restart after granting permissions."
                    )
                elif "numpy" in str(e).lower():
                    raise AudioInitializationError(
                        "Audio recording requires NumPy for real-time processing.\n\n"
                        "To enable voice input features, install NumPy:\n"
                        "  pip install numpy\n\n"
                        "NumPy is required for:\n"
                        "• Real-time audio level monitoring\n"
                        "• Efficient audio format conversions\n"
                        "• Voice activity detection\n\n"
                        "Without NumPy, audio processing would cause high CPU usage and UI freezing."
                    )
                else:
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
        if (
            self._transcription_service is None
            and self._transcription_init_error is None
        ):
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
        save_audio: bool = False,
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
            self.captured_bytes = 0
            self.start_time = time.time()
            self.save_audio = save_audio and not self.privacy_settings["local_only"]

            # Initialize streaming transcriber
            self._initialize_streaming_transcriber()

            # Start processing thread (simplified)
            self.stop_processing.clear()
            self.processing_thread = threading.Thread(
                target=self._processing_loop, daemon=True, name="DictationProcessor"
            )
            self.processing_thread.start()

            # Start audio recording
            success = audio_svc.start_recording(callback=self._audio_callback)

            if success:
                with self.state_lock:
                    self.state = DictationState.LISTENING
                self._notify_state_change()
                logger.info("Started live dictation")
                return True
            else:
                self._cleanup()
                self._notify_error(
                    Exception(
                        "Failed to start audio recording. Please check your microphone."
                    )
                )
                return False

        except Exception as e:
            logger.error(f"Failed to start dictation: {e}")
            self._cleanup()
            self._notify_error(e)
            return False

    @staticmethod
    def _preferred_local_provider() -> str:
        """Pick a local provider to fall back to under privacy mode.

        Prefers one that is actually installed on this machine. The old
        hard-coded `parakeet-mlx` is Apple-Silicon-only, so on Linux it swapped
        a working provider for one that fails on every chunk.

        Returns:
            An installed local provider id, or the first local provider when
            none is installed (nothing will work either way; this at least
            keeps the reported provider inside the local catalogue).
        """
        installed = installed_local_providers()
        return installed[0] if installed else LOCAL_STT_PROVIDERS[0]

    def _initialize_streaming_transcriber(self):
        """Initialize streaming transcriber if available."""
        if self.privacy_settings["local_only"]:
            # Privacy mode means "audio never leaves this machine", so the test
            # is exactly `provider_is_local()` -- the same catalogue the Console
            # resolver picks from (`Utils/local_stt_providers`). A second,
            # hand-maintained list lived here and drifted twice: once on a
            # misspelled id ("lightning-whisper"), and once by staying at three
            # providers while the resolver grew to seven, which made the Console
            # download and announce one model and then transcribe with another.
            # Do not reintroduce a literal list here.
            if not provider_is_local(self.transcription_provider):
                fallback = self._preferred_local_provider()
                logger.info(
                    f"Provider '{self.transcription_provider}' is not local; "
                    f"privacy mode requires one. Using '{fallback}' instead."
                )
                self.transcription_provider = fallback

        try:
            self.streaming_transcriber = (
                self.transcription_service.create_streaming_transcriber(
                    provider=self.transcription_provider,
                    model=self.transcription_model,
                    language=self.language,
                )
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
            # Calculate audio level (RMS)
            try:
                if NUMPY_AVAILABLE and np is not None:
                    # Use numpy for efficient calculation if available
                    audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
                    if len(audio_array) > 0:
                        rms = np.sqrt(np.mean(audio_array.astype(float) ** 2))
                        # Normalize to 0.0-1.0 range (assuming 16-bit audio)
                        self._current_audio_level = min(1.0, rms / 32768.0)
                else:
                    # Fallback: simple RMS calculation without numpy
                    import struct

                    # Unpack 16-bit samples
                    samples = struct.unpack(f"{len(audio_chunk) // 2}h", audio_chunk)
                    if samples:
                        # Calculate RMS manually
                        sum_squares = sum(s * s for s in samples)
                        rms = (sum_squares / len(samples)) ** 0.5
                        # Normalize to 0.0-1.0 range (assuming 16-bit audio)
                        self._current_audio_level = min(1.0, rms / 32768.0)
                    else:
                        self._current_audio_level = 0.0
            except Exception as e:
                logger.debug(f"Could not calculate audio level: {e}")
                self._current_audio_level = 0.0

            # Add to buffer. `captured_bytes` is a running total and must not
            # be derived from `audio_buffer`, which privacy mode trims to the
            # last few chunks while the capture is still running.
            with self.buffer_lock:
                self.audio_buffer.append(audio_chunk)
                self.captured_bytes += len(audio_chunk)

            # Every delivered chunk is queued and refreshes the finalize
            # deadline. Silence is filtered *upstream*, not here:
            # `AudioRecordingService._process_audio_chunk` splits capture into
            # 20 ms frames and only calls this callback for VAD-positive ones,
            # so "a chunk arrived" already means "speech arrived". Re-gating
            # here on a locally-built VAD is worse than redundant -- the
            # recorder's frames are 640 bytes at 16 kHz, so any window larger
            # than one frame matches nothing and silently drops the entire
            # capture. Without `webrtcvad` the recorder delivers everything
            # unconditionally and finals fire only at stop, as before.
            # Queue for processing
            self.processing_queue.put(("audio", audio_chunk))

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

                    if item_type == "audio":
                        accumulated_audio.append(data)

                except queue.Empty:
                    pass

                # Process accumulated audio periodically
                current_time = time.time()
                buffer_duration_sec = self.buffer_duration_ms / 1000

                if (
                    accumulated_audio
                    and (current_time - last_process_time) >= buffer_duration_sec
                ):
                    audio_data = b"".join(accumulated_audio)
                    self._process_audio_buffer(audio_data)

                    # Clear accumulated audio
                    accumulated_audio = []
                    last_process_time = current_time

                    # Auto-clear buffer if privacy enabled
                    if self.privacy_settings["auto_clear_buffer"]:
                        with self.buffer_lock:
                            # Keep only last few chunks for context
                            if len(self.audio_buffer) > 10:
                                self.audio_buffer = self.audio_buffer[-5:]

                # Check for silence timeout. Runs every ~0.1s iteration
                # independent of chunk arrival -- deliberately not derived
                # from queue activity, so it still fires while the recorder's
                # VAD is withholding frames during a pause.
                if (
                    self.last_speech_time
                    and (current_time - self.last_speech_time)
                    > self.silence_threshold_seconds
                ):
                    # Finalize current segment after a threshold pause.
                    self._finalize_current_segment()
                    self.last_speech_time = 0

            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                self._notify_error(e)

        # `stop_dictation()` sets `stop_processing` and the `while` above
        # exits on its very next iteration, abandoning whatever is still in
        # `accumulated_audio` plus anything left unread in
        # `processing_queue`. Without this, a capture shorter than one
        # `buffer_duration_ms` window is transcribed as nothing at all, and
        # the tail of every longer capture (audio queued since the last
        # periodic flush) is silently dropped. Drain the queue and flush
        # whatever remains before the thread returns.
        try:
            while True:
                try:
                    item_type, data = self.processing_queue.get_nowait()
                except queue.Empty:
                    break

                if item_type == "audio":
                    accumulated_audio.append(data)

            if accumulated_audio:
                audio_data = b"".join(accumulated_audio)
                self._process_audio_buffer(audio_data)
                accumulated_audio = []
        except Exception as e:
            logger.error(f"Processing loop final flush error: {e}")
            self._notify_error(e)

    def _cleanup(self):
        """Clean up resources with privacy considerations."""
        with self.state_lock:
            self.state = DictationState.IDLE

        # Clear sensitive data immediately if privacy mode
        if self.privacy_settings["auto_clear_buffer"]:
            self.audio_buffer = []
            if not self.privacy_settings["save_history"]:
                self.transcript_segments = []
                self.current_transcript = ""

        self.streaming_transcriber = None

        # Note: We don't clear the lazy-loaded services themselves
        # They can be reused for the next session

        self._notify_state_change()

    def get_audio_devices(self) -> List[Dict[str, Any]]:
        """Get available audio input devices with error handling."""
        try:
            devices = self.audio_service.get_audio_devices()
            if not devices:
                logger.warning(
                    "No audio input devices found. Check microphone permissions."
                )
            return devices
        except AudioInitializationError as e:
            logger.error(f"Cannot get audio devices: {e}")
            return []
        except Exception as e:
            if "Invalid input device" in str(e) or "no default" in str(e).lower():
                logger.error(
                    "No microphone access. Please:\n"
                    "1. Open System Settings > Privacy & Security > Microphone\n"
                    "2. Grant access to Terminal or your Python app\n"
                    "3. Restart the application"
                )
            else:
                logger.error(f"Error getting audio devices: {e}")
            return []

    def set_buffer_duration(self, duration_ms: int):
        """Set audio buffer duration dynamically."""
        self.buffer_duration_ms = max(
            100, min(2000, duration_ms)
        )  # Clamp between 100-2000ms
        save_setting_to_cli_config(
            "dictation", "buffer_duration_ms", self.buffer_duration_ms
        )
        logger.info(f"Buffer duration set to {self.buffer_duration_ms}ms")

    def _process_audio_buffer(self, audio_data: bytes):
        """Transcribe one buffered chunk of PCM and publish what it says.

        Two paths, in order of preference:

        1. The session's streaming transcriber, when one was built.
        2. ``TranscriptionService.transcribe_buffer()``, which takes raw PCM.

        Never ``TranscriptionService.transcribe()``: that one takes an *audio
        file path* and reaches ``Path(audio_path)`` before any provider is
        dispatched, so handing it PCM raised ``TypeError`` on every single
        chunk -- a capture produced no partials, no finals and no transcript,
        for every provider. See ``Tests/Audio/test_dictation_lazy_transcription.py``.
        """
        if not audio_data:
            return

        try:
            if self.streaming_transcriber is not None:
                result = None
                try:
                    result = self.streaming_transcriber.process_audio(audio_data)
                except Exception as e:
                    # Includes the transcriber simply not speaking this
                    # protocol; the buffer path below is a complete fallback.
                    logger.warning(
                        f"Streaming transcription failed, "
                        f"falling back to buffer transcription: {e}"
                    )

                if isinstance(result, dict):
                    partial = self._streamed_partial_text(result)
                    if partial:
                        self._handle_partial_text(partial)

                    final = result.get("final")
                    if isinstance(final, str) and final.strip():
                        self._handle_streamed_final(final)
                    return

            service = self.transcription_service
            if service is None:
                return

            # The recorder built for *this* capture, never the `audio_service`
            # property: reading that property lazily CONSTRUCTS a recorder and
            # opens an audio device.
            recorder = self._audio_service
            sample_rate = getattr(recorder, "sample_rate", None) or 16000
            channels = getattr(recorder, "channels", None) or 1

            # Pass the provider the caller resolved. Omitting it silently falls
            # back to the transcription service's own default provider, which
            # would make the "using X instead of your configured Y" notice a lie.
            result = service.transcribe_buffer(
                audio_data=audio_data,
                sample_rate=sample_rate,
                channels=channels,
                sample_width=2,  # 16-bit PCM
                provider=self.transcription_provider,
                model=self.transcription_model,
                language=self.language,
            )

            if result and result.get("text"):
                self._handle_partial_text(result["text"])

        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            self._notify_error(e)

    @staticmethod
    def _streamed_partial_text(result: Dict[str, Any]) -> Optional[str]:
        """Pull the in-progress text out of a streaming transcriber's result.

        Two shapes exist in the wild: ``{"partial": "<text>"}`` and
        ``{"partial": True, "text": "<text>"}`` (what
        ``ParakeetMLXStreamingTranscriber`` returns). Treat both as partials
        rather than handing a bool to the transcript accumulator.
        """
        partial = result.get("partial")
        if isinstance(partial, str):
            return partial
        text = result.get("text")
        if isinstance(text, str):
            return text
        return None

    def _handle_partial_text(self, text: str):
        """Accumulate one chunk's text and publish the segment so far.

        Chunks arrive roughly every ``buffer_duration_ms``; replacing the
        transcript with each one (as this used to) would leave a segment
        holding only its last half-second of speech.
        """
        chunk = (text or "").strip()
        if not chunk:
            return

        with self.transcript_lock:
            self.current_transcript = (
                f"{self.current_transcript} {chunk}"
                if self.current_transcript
                else chunk
            )
            accumulated = self.current_transcript

        # Consumers redraw the preview from each partial, so send the whole
        # segment, not just this chunk.
        if self.on_partial_transcript:
            try:
                self.on_partial_transcript(accumulated)
            except Exception as e:
                logger.error(f"Partial transcript callback error: {e}")

        # Commands are looked for in the new speech only; the accumulated text
        # would re-trigger a command already handled on an earlier chunk.
        if self.enable_commands and self.on_command:
            command = self._detect_command(chunk)
            if command:
                try:
                    self.on_command(command)
                except Exception as e:
                    logger.error(f"Command callback error: {e}")

    def _handle_streamed_final(self, text: str):
        """Commit a segment the streaming transcriber has already finalised."""
        final = text.strip()
        if not final:
            return

        # A streaming final supersedes the partial hypotheses that previewed it.
        with self.transcript_lock:
            self.current_transcript = final

        self._finalize_current_segment()

    def _finalize_current_segment(self):
        """Finalize the current transcript segment."""
        with self.transcript_lock:
            text = self.current_transcript
            self.current_transcript = ""

        if not text:
            return

        # Add to segments
        self.transcript_segments.append({"text": text, "timestamp": time.time()})

        # Notify final transcript
        if self.on_final_transcript:
            try:
                self.on_final_transcript(text)
            except Exception as e:
                logger.error(f"Final transcript callback error: {e}")

    def _detect_command(self, text: str) -> Optional[str]:
        """Detect voice commands in transcript."""
        text_lower = text.lower()

        # Common voice commands
        commands = {
            "stop dictation": "stop",
            "new paragraph": "new_paragraph",
            "new line": "new_line",
            "clear all": "clear",
            "undo": "undo",
        }

        for phrase, command in commands.items():
            if phrase in text_lower:
                return command

        return None

    def stop_dictation(self) -> DictationResult:
        """Stop dictation and return results.

        Returns:
            The transcript, plus the two facts a caller needs to explain an
            empty one honestly: how many bytes the recorder delivered, and
            whether the processing thread finished before the join expired.
        """
        logger.info("Stopping dictation...")

        # Change state
        with self.state_lock:
            if self.state != DictationState.LISTENING:
                logger.warning("Dictation not active")
                return DictationResult(transcript="", segments=[], duration=0.0)
            self.state = DictationState.IDLE

        # Stop processing
        if self.stop_processing:
            self.stop_processing.set()

        # Wait for the processing thread to drain and transcribe what is left.
        # The old hard-coded 2.0s was shorter than a single warm transcription,
        # so on any real capture this expired and the work in flight was thrown
        # away silently -- indistinguishable, downstream, from a dead
        # microphone. Report the expiry instead of hiding it.
        transcription_complete = True
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=self.stop_join_timeout_seconds)
            transcription_complete = not self.processing_thread.is_alive()
            if not transcription_complete:
                logger.warning(
                    "Dictation transcription did not finish within {}s; "
                    "audio still in flight was dropped",
                    self.stop_join_timeout_seconds,
                )

        # Finalize any remaining transcript
        self._finalize_current_segment()

        # Calculate duration
        duration = time.time() - self.start_time if self.start_time else 0.0

        # Build final transcript
        final_transcript = " ".join(seg["text"] for seg in self.transcript_segments)

        # Create result
        result = DictationResult(
            transcript=final_transcript,
            segments=self.transcript_segments.copy(),
            duration=duration,
            captured_bytes=self.captured_bytes,
            transcription_complete=transcription_complete,
        )

        # Release capture explicitly. The non-lazy service does this in its own
        # stop_dictation; this one never did, so every successful stop left the
        # microphone live. Use the private attribute, not the `audio_service`
        # property -- reading the property lazily CONSTRUCTS a recorder, which
        # would open an audio device during teardown.
        recorder = self._audio_service
        if recorder is not None:
            try:
                recorder.stop_recording()
            except Exception:  # noqa: BLE001 - teardown must never raise
                logger.opt(exception=True).warning("Failed to release audio capture")

        # Cleanup
        self._cleanup()

        word_count = len(result.transcript.split()) if result.transcript else 0
        logger.info(
            f"Dictation stopped. Words: {word_count}, Duration: {result.duration:.1f}s"
        )
        return result

    def pause_dictation(self):
        """Pause dictation (temporarily stop processing)."""
        with self.state_lock:
            if self.state == DictationState.LISTENING:
                self.state = DictationState.PAUSED
                self._notify_state_change()
                logger.info("Dictation paused")

    def resume_dictation(self):
        """Resume paused dictation."""
        with self.state_lock:
            if self.state == DictationState.PAUSED:
                self.state = DictationState.LISTENING
                self._notify_state_change()
                logger.info("Dictation resumed")

    def update_privacy_settings(self, settings: dict):
        """Update privacy settings dynamically."""
        self.privacy_settings.update(settings)
        save_setting_to_cli_config(
            "dictation.privacy", "local_only", settings.get("local_only", True)
        )
        save_setting_to_cli_config(
            "dictation.privacy", "save_history", settings.get("save_history", False)
        )
        save_setting_to_cli_config(
            "dictation.privacy",
            "auto_clear_buffer",
            settings.get("auto_clear_buffer", True),
        )
        logger.info(f"Privacy settings updated: {settings}")

    def get_full_transcript(self) -> str:
        """Get the full transcript as a single string."""
        if self.transcript_segments:
            return " ".join(seg["text"] for seg in self.transcript_segments)
        return self.current_transcript or ""

    def set_audio_device(self, device_id: Optional[str]):
        """Set the audio input device."""
        self.audio_device_id = device_id
        if self.audio_service:
            try:
                self.audio_service.set_device(device_id)
            except Exception as e:
                logger.warning(f"Could not set audio device: {e}")

    def get_audio_level(self) -> float:
        """Get current audio input level (0.0 to 1.0)."""
        if hasattr(self, "_current_audio_level"):
            return self._current_audio_level
        return 0.0

    # Include other methods from original implementation with appropriate modifications...
    # (I'm including the key ones here, others remain similar)

    def _notify_state_change(self):
        """Notify state change callback."""
        if self.on_state_change:
            try:
                self.on_state_change(self.state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")

    def _notify_error(self, error: Exception):
        """Notify error callback with sanitized error messages."""
        with self.state_lock:
            self.state = DictationState.ERROR

        # Sanitize error message to remove sensitive paths
        safe_error = type(error)(str(error).replace(os.path.expanduser("~"), "~"))

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
