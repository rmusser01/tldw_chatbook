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

    #: How aggressively the recorder's VAD classifies a frame as speech
    #: (0 = least aggressive / most permissive, 3 = most aggressive). This
    #: used to be hard-coded to 2, and measurement on real hardware showed
    #: ambient silence still registering ~30 speech-positive frames per 4s at
    #: that level -- enough that `last_speech_time` never goes stale and a
    #: pause never finalizes a segment. Aggressiveness 3 measured 0 false
    #: positives on the same room. Configurable as
    #: `dictation.vad_aggressiveness`.
    VAD_AGGRESSIVENESS = 3

    #: How many milliseconds of recently-*rejected* audio the recorder
    #: replays the instant its VAD accepts a frame after a silence run, to
    #: recover a clipped speech onset. Incident: live dictation on real
    #: hardware with parakeet-mlx transcribed "stop" as "dot"/"top"-like
    #: forms and "send" as "and" -- at `vad_aggressiveness=3`, low-energy
    #: onsets (word-initial fricatives especially) were classified as
    #: non-speech and dropped before transcription ever saw them. See
    #: `AudioRecordingService.VAD_PREROLL_MS` for the 240ms/12-frame
    #: rationale this mirrors. Configurable as `dictation.vad_preroll_ms`.
    VAD_PREROLL_MS = 240

    #: Hard safety net on the non-streaming (buffer-API) regime's in-progress
    #: segment, expressed as a duration rather than a bare byte count so it
    #: tracks whatever sample rate/width the capture is actually using (see
    #: `_max_non_streaming_segment_bytes`). Review finding (PR #1171,
    #: Finding 2): `_processing_loop`'s non-streaming path only transcribes at
    #: the silence gate or at stop. `AudioRecordingService` sets
    #: `use_vad=False` whenever `webrtcvad` is missing or `Vad()` init fails,
    #: and in that state the recorder forwards EVERY chunk unconditionally --
    #: so `last_speech_time` never goes stale, the silence gate never fires,
    #: and `segment_audio` grows for the whole capture: unbounded memory, and
    #: nothing transcribed until `stop_dictation()`'s tail-drain (itself
    #: behind `stop_join_timeout_seconds`, 30s default). This bound is a
    #: safety net, not a VAD workaround -- it fires regardless of VAD state,
    #: since a user who simply never pauses hits the identical wall with VAD
    #: fully working. 30s mirrors `STOP_JOIN_TIMEOUT_SECONDS`: the worst-case
    #: forced-transcription latency this introduces is one the user already
    #: tolerates at stop, and it is long enough that no ordinary spoken
    #: sentence (silence-gated well before this) is ever chopped by it.
    MAX_NON_STREAMING_SEGMENT_SECONDS = 30.0

    #: Instance defaults, declared on the class so a service built with
    #: `__new__` (teardown-path tests do exactly that) still stops cleanly
    #: instead of raising AttributeError while releasing the microphone.
    stop_join_timeout_seconds = STOP_JOIN_TIMEOUT_SECONDS
    silence_threshold_seconds = SILENCE_THRESHOLD_SECONDS
    vad_aggressiveness = VAD_AGGRESSIVENESS
    vad_preroll_ms = VAD_PREROLL_MS
    captured_bytes = 0
    max_buffer_bytes: Optional[int] = None
    on_buffer_limit: Optional[Callable[[], None]] = None
    #: `_processing_loop` reads this unconditionally on its very first line
    #: (to pick its streaming vs. non-streaming regime), so a service built
    #: via `__new__` without an explicit `_initialize_streaming_transcriber()`
    #: call needs this default too, or the processing thread dies with an
    #: `AttributeError` the instant it starts.
    streaming_transcriber: Optional[Any] = None
    #: Same `__new__`-safety reasoning as `streaming_transcriber` just above:
    #: `_transcribe_segment_audio` reads this unconditionally, so a service
    #: built via `__new__` without going through `start_dictation()` needs
    #: this class-level default too.
    on_segment_transcribing: Optional[Callable[[bool], None]] = None

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
        self.on_segment_transcribing = None

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
        self.vad_aggressiveness = self._resolve_vad_aggressiveness()
        self.vad_preroll_ms = self._resolve_vad_preroll_ms()

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

    @classmethod
    def _resolve_vad_aggressiveness(cls) -> int:
        """Read `dictation.vad_aggressiveness`, falling back to the default.

        Returns:
            An integer from 0 to 3 (inclusive) controlling how aggressively
            the recorder's VAD filters ambient noise out of "speech". Higher
            values classify more borderline audio as non-speech, which is
            what lets a pause finalize a segment instead of ambient noise
            holding `last_speech_time` fresh forever.
        """
        raw = get_cli_setting(
            "dictation.vad_aggressiveness", cls.VAD_AGGRESSIVENESS
        )
        try:
            aggressiveness = int(raw)
        except (TypeError, ValueError, OverflowError):
            # Same trap as the two resolvers above: `nan`/`inf` are valid
            # TOML floats. `int(float("nan"))` raises `ValueError` and
            # `int(float("inf"))` raises `OverflowError` (not `ValueError`),
            # so both must be caught here or a typo'd config value crashes
            # dictation start instead of falling back.
            logger.warning(
                "Invalid dictation.vad_aggressiveness {!r}; using {}",
                raw,
                cls.VAD_AGGRESSIVENESS,
            )
            return cls.VAD_AGGRESSIVENESS
        if not 0 <= aggressiveness <= 3:
            logger.warning(
                "dictation.vad_aggressiveness must be an integer between 0 "
                "and 3 (got {!r}); using {}",
                raw,
                cls.VAD_AGGRESSIVENESS,
            )
            return cls.VAD_AGGRESSIVENESS
        return aggressiveness

    @classmethod
    def _resolve_vad_preroll_ms(cls) -> int:
        """Read `dictation.vad_preroll_ms`, falling back to the default.

        Returns:
            A non-negative number of milliseconds of recently-rejected audio
            the recorder replays the instant its VAD accepts a frame after a
            silence run. See `VAD_PREROLL_MS` above for why this exists.
        """
        raw = get_cli_setting("dictation.vad_preroll_ms", cls.VAD_PREROLL_MS)
        try:
            preroll_ms = int(raw)
        except (TypeError, ValueError, OverflowError):
            # Same trap as the resolvers above: `nan`/`inf` are valid TOML
            # floats. `int(float("nan"))` raises `ValueError` and
            # `int(float("inf"))` raises `OverflowError` (not `ValueError`),
            # so both must be caught here or a typo'd config value crashes
            # dictation start instead of falling back.
            logger.warning(
                "Invalid dictation.vad_preroll_ms {!r}; using {}",
                raw,
                cls.VAD_PREROLL_MS,
            )
            return cls.VAD_PREROLL_MS
        if preroll_ms < 0:
            logger.warning(
                "dictation.vad_preroll_ms must be a non-negative integer "
                "(got {!r}); using {}",
                raw,
                cls.VAD_PREROLL_MS,
            )
            return cls.VAD_PREROLL_MS
        return preroll_ms

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
                    vad_aggressiveness=self.vad_aggressiveness,
                    vad_preroll_ms=self.vad_preroll_ms,
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
        on_segment_transcribing: Optional[Callable[[bool], None]] = None,
        save_audio: bool = False,
    ) -> bool:
        """
        Start live dictation with improved initialization.

        Args:
            on_segment_transcribing: Fired from `_transcribe_segment_audio`,
                on the processing thread, TWICE per segment, symmetrically --
                both at the mid-capture silence gate and at the stop-path
                tail-fold -- with a single `bool` argument:

                * `False` right when a (potentially seconds-long)
                  whole-segment transcription starts.
                * `True` right after that transcription call returns, on
                  EVERY segment completion, including one that transcribes to
                  blank/whitespace (routine for room noise or a too-short VAD
                  sliver -- see `_transcribe_segment_audio`'s `if not
                  produced_text:` branch). A blank result fires neither
                  `on_partial_transcript` nor `on_final_transcript`, so
                  without this second, unconditional signal a consumer that
                  shows a "transcribing" indicator on `False` and hides it on
                  the next partial/final would have nothing to hide it on --
                  it would stay shown for the rest of the capture.

                Never invoked for the streaming-transcriber regime, whose
                `process_audio()` calls are cheap incremental pushes, not a
                from-scratch transcription.
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
            self.on_segment_transcribing = on_segment_transcribing

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
            candidate = self.transcription_service.create_streaming_transcriber(
                provider=self.transcription_provider,
                model=self.transcription_model,
                language=self.language,
            )

            # The processing loop's streaming regime calls `process_audio()`
            # per chunk. `ParakeetMLXStreamingTranscriber` exposes
            # `add_audio`/`finalize` instead, so engaging it meant every
            # cadence tick raised AttributeError and fell back to per-window
            # buffer transcription -- the chopped-segment architecture the
            # segment-at-silence rework removed ("console stop" transcribed
            # as 'Consoles.' + 'Stop.', which no command can match, plus
            # per-window noise hallucinations; observed live 2026-07-31).
            # A transcriber that does not speak the loop's protocol is worse
            # than none: refuse it and take the working segment path.
            if candidate is not None and not hasattr(candidate, "process_audio"):
                logger.info(
                    "Streaming transcriber for '{}' lacks process_audio; "
                    "using segment transcription instead",
                    self.transcription_provider,
                )
                candidate = None
            self.streaming_transcriber = candidate

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
        """Drain queued audio into the segment it belongs to.

        Two entirely different regimes, chosen once at the top -- checked
        against `self.streaming_transcriber` at the moment this thread
        starts, since `start_dictation()` builds it (if at all) before this
        thread is spawned and nothing reassigns it mid-session:

        * A streaming transcriber -- in this codebase, only ever built for
          `parakeet-mlx` on Apple Silicon (see
          `Local_Ingestion/transcription_service.py`
          `create_streaming_transcriber`; every other provider this app ships,
          including the default `faster-whisper`, returns `None`) -- already
          pushes its own finals through `_handle_streamed_final`, and its
          `process_audio()` calls are cheap incremental pushes, not a
          from-scratch transcription. This regime is UNCHANGED: transcribe
          whatever accumulated on the old `buffer_duration_ms` cadence, and
          let the silence check finalize whatever partial is outstanding as a
          fallback when the backend never sends an explicit final.

        * Every other provider has no streaming transcriber, so
          `_process_audio_buffer()` always took the buffer API, which is a
          full, synchronous, from-scratch transcription of whatever bytes it
          is handed (~4-5s for a 0.5s window measured live, on
          distil-large-v3, on a loaded machine). Calling that on a cadence
          starves the silence check that shares this same thread: while the
          loop is inside `_transcribe_buffer_with_faster_whisper`, it is not
          polling `last_speech_time`, so a threshold-length pause can elapse
          invisibly -- live captures showed `last_speech_time` age reach 8.6s
          against a 2.0s threshold, with a non-empty `current_transcript` and
          no final ever firing while the capture ran. Worse, on the rare
          iteration the loop *did* get a turn, it would finalize whatever
          fraction of one utterance it happened to be holding, chopping one
          utterance into unrelated finals ("console stop" -> "consoles." /
          "stop." on two different windows).

          So for this regime the loop does no transcription of its own: it
          only accumulates VAD-gated chunks into `segment_audio`. The only
          two places that ever call `_process_audio_buffer()` for this
          regime are the silence check below and the tail-drain after the
          loop exits -- each transcribing the whole segment exactly once. A
          segment transcription can take seconds; chunks that arrive while
          one is in flight simply queue in `processing_queue` (the audio
          callback never blocks on this thread) and open the *next*
          segment once this call returns and the loop resumes draining --
          nothing is lost or mixed into the segment already sent for
          transcription, at the cost of delaying that next segment's own
          silence-finalize until this call returns. Accepted: the
          alternative (transcribing on the shared loop thread) is the
          defect this method exists to fix.

          `segment_audio`'s size would otherwise be bounded only by however
          long the in-progress segment runs before something finalizes it --
          and without `webrtcvad` (`recording_service.py` forwards every
          chunk unconditionally in that state, a documented degrade path),
          or for an utterance that simply never pauses, NOTHING would
          finalize it until `stop_dictation()`: `last_speech_time` never
          goes stale, so the silence check above never fires, and the entire
          capture -- however long -- would sit in `segment_audio` and be
          transcribed as a single call inside the tail-drain, behind
          `stop_join_timeout_seconds` (30s default). Two different callers
          already had two different outcomes here: the Console
          (`UI/Screens/chat_screen.py`) passes
          `max_buffer_bytes=CONSOLE_DICTATION_MAX_BYTES` to the recorder,
          whose `on_buffer_limit` callback stops the capture once that many
          bytes have been delivered (`_handle_console_dictation_limit`), so
          `segment_audio` was bounded there too, indirectly, at the same
          ~60s/~1.9MB ceiling. `UI/Dictation_Window_Improved.py` builds this
          service with no `max_buffer_bytes` at all, so for that caller
          `segment_audio` was genuinely unbounded by anything but the user
          choosing to stop. Fixed by `MAX_NON_STREAMING_SEGMENT_SECONDS`
          below: the loop force-finalizes the in-progress segment once it
          crosses that bound, through the same transcribe+finalize helper
          the silence gate uses (`_finalize_non_streaming_segment`), so
          `segment_audio` cannot grow past it regardless of VAD state or
          `max_buffer_bytes`. See
          `.superpowers/sdd/2026-07-29-console-voice-control-v2/dictation-loop-fix-report.md`
          for the original review that first flagged this as future work,
          and `.superpowers/sdd/2026-07-29-console-voice-control-v2/qodo-findings-report.md`
          for the review that turned it into this fix.
        """
        streaming = self.streaming_transcriber is not None
        segment_audio = []
        last_process_time = time.time()

        while not self.stop_processing.is_set():
            try:
                # Get items from queue with timeout, THEN drain whatever else
                # is already waiting (non-blocking) before doing anything
                # else this iteration. A single `get(timeout=0.1)` per
                # iteration was previously enough -- until a whole
                # multi-second `_transcribe_segment_audio()` call (below) can
                # block this loop long enough for an entire second utterance
                # to arrive, finish, and go silent while the loop cannot look
                # at it. Resuming with only ONE of those queued frames
                # drained, `last_speech_time` (refreshed by the audio
                # callback on ITS OWN thread throughout, unaffected by this
                # loop being blocked) is already stale relative to the
                # silence check just below -- so it fired on that single
                # frame alone, stranding the rest of the utterance behind a
                # just-zeroed `last_speech_time` until the next pause or
                # stop. Draining to empty first means the silence check
                # always sees the FULL backlog that arrived while this loop
                # was away, not an arbitrary one-frame slice of it.
                try:
                    item_type, data = self.processing_queue.get(timeout=0.1)

                    if item_type == "audio":
                        segment_audio.append(data)

                except queue.Empty:
                    pass
                else:
                    while True:
                        try:
                            item_type, data = self.processing_queue.get_nowait()
                        except queue.Empty:
                            break
                        if item_type == "audio":
                            segment_audio.append(data)

                current_time = time.time()
                buffer_duration_sec = self.buffer_duration_ms / 1000
                cadence_elapsed = (
                    segment_audio
                    and (current_time - last_process_time) >= buffer_duration_sec
                )

                if streaming and cadence_elapsed:
                    audio_data = b"".join(segment_audio)
                    self._process_audio_buffer(audio_data)
                    segment_audio = []
                    last_process_time = current_time
                    self._trim_privacy_audio_buffer()
                elif cadence_elapsed:
                    # Non-streaming: accumulate only -- see method docstring
                    # for why transcribing here is exactly the defect this
                    # rework fixes. Still tick the cadence clock and trim the
                    # privacy buffer on the same schedule as before; only the
                    # transcribe call moved.
                    last_process_time = current_time
                    self._trim_privacy_audio_buffer()

                # Hard safety net on the non-streaming segment (review
                # finding, PR #1171, "Finding 2"; see
                # `MAX_NON_STREAMING_SEGMENT_SECONDS` for the full rationale):
                # runs every ~0.1s iteration regardless of VAD state, so it
                # fires exactly when the silence gate below cannot -- no
                # `webrtcvad`, or an utterance that simply never pauses. Not
                # a duplicate of the silence gate's transcribe+finalize call:
                # both go through `_finalize_non_streaming_segment` so a
                # forced finalize behaves identically to a silence-triggered
                # one (same start/done signals, same empty-result drop).
                if not streaming and segment_audio:
                    segment_bytes = sum(len(chunk) for chunk in segment_audio)
                    if segment_bytes >= self._max_non_streaming_segment_bytes():
                        segment_audio = self._finalize_non_streaming_segment(
                            segment_audio
                        )

                # Check for silence timeout. Runs every ~0.1s iteration
                # independent of chunk arrival -- deliberately not derived
                # from queue activity, so it still fires while the recorder's
                # VAD is withholding frames during a pause. Cheap for BOTH
                # regimes now: the non-streaming loop never blocks on a
                # transcription except right here, once per segment.
                if (
                    self.last_speech_time
                    and (current_time - self.last_speech_time)
                    > self.silence_threshold_seconds
                ):
                    self.last_speech_time = 0
                    if streaming:
                        # Finalizes whatever partial text streaming pushed in
                        # via `_handle_partial_text`, as a fallback for a
                        # backend that never sent an explicit final.
                        self._finalize_current_segment()
                    else:
                        segment_audio = self._finalize_non_streaming_segment(
                            segment_audio
                        )

            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                self._notify_error(e)

        # `stop_dictation()` sets `stop_processing` and the `while` above
        # exits on its very next iteration, abandoning whatever is still in
        # `segment_audio` plus anything left unread in `processing_queue`.
        # Without this, a capture shorter than one `buffer_duration_ms`
        # window (non-streaming: shorter than one silence pause) is
        # transcribed as nothing at all, and the tail of every longer
        # capture is silently dropped. Drain the queue and transcribe
        # whatever remains before the thread returns; `stop_dictation()`
        # calls `_finalize_current_segment()` itself right after this thread
        # is joined, exactly as it always has -- this only ever sets
        # `current_transcript`, never finalizes it.
        try:
            while True:
                try:
                    item_type, data = self.processing_queue.get_nowait()
                except queue.Empty:
                    break

                if item_type == "audio":
                    segment_audio.append(data)

            if segment_audio:
                if streaming:
                    audio_data = b"".join(segment_audio)
                    self._process_audio_buffer(audio_data)
                    segment_audio = []
                else:
                    pending, segment_audio = segment_audio, []
                    self._transcribe_segment_audio(pending)
        except Exception as e:
            logger.error(f"Processing loop final flush error: {e}")
            self._notify_error(e)

    def _trim_privacy_audio_buffer(self) -> None:
        """Keep only the last few raw chunks in `self.audio_buffer` for context.

        Unrelated to segment/transcript tracking -- `self.audio_buffer` is the
        raw-PCM history `_audio_callback()` appends to, trimmed periodically
        (on the `buffer_duration_ms` cadence, same as before this rework) so a
        long capture does not hold its entire raw audio in memory when
        `dictation.privacy.auto_clear_buffer` is on (the default).
        """
        if self.privacy_settings["auto_clear_buffer"]:
            with self.buffer_lock:
                if len(self.audio_buffer) > 10:
                    self.audio_buffer = self.audio_buffer[-5:]

    def _max_non_streaming_segment_bytes(self) -> int:
        """Convert `MAX_NON_STREAMING_SEGMENT_SECONDS` to bytes for this capture.

        Reads the real recorder's sample rate/channels when one has already
        been constructed (`self._audio_service`, the same fields
        `_process_audio_buffer` reads for the actual `transcribe_buffer()`
        call), falling back to the recorder's own defaults otherwise --
        `_processing_loop` must not read the `audio_service` property here,
        since that lazily CONSTRUCTS a recorder and opens an audio device.
        Sample width is hard-coded at 2 bytes (16-bit PCM), the same literal
        `_process_audio_buffer` passes to `transcribe_buffer()`; nothing in
        this codebase ever records at a different width.
        """
        recorder = self._audio_service
        sample_rate = getattr(recorder, "sample_rate", None) or 16000
        channels = getattr(recorder, "channels", None) or 1
        sample_width = 2  # 16-bit PCM
        return int(
            self.MAX_NON_STREAMING_SEGMENT_SECONDS
            * sample_rate
            * channels
            * sample_width
        )

    def _finalize_non_streaming_segment(
        self, segment_audio: List[bytes]
    ) -> List[bytes]:
        """Transcribe and finalize one non-streaming segment, exactly once.

        Shared by every place `_processing_loop`'s non-streaming regime
        decides a segment is complete mid-capture -- the silence gate and
        the hard segment-size bound (`MAX_NON_STREAMING_SEGMENT_SECONDS`) --
        so a forced finalize behaves identically to a silence-triggered one:
        `_transcribe_segment_audio` fires the transcribing start/done
        signals and drops an empty result silently, and
        `_finalize_current_segment` publishes whatever text it produced (a
        no-op when that was blank). Not used by the tail-drain at stop,
        which deliberately transcribes without finalizing -- `stop_dictation()`
        finalizes once itself, right after the processing thread is joined.

        Args:
            segment_audio: The chunks accumulated since the last finalize.

        Returns:
            A fresh empty list, ready for the caller to resume accumulating
            into.
        """
        if segment_audio:
            self._transcribe_segment_audio(segment_audio)
        self._finalize_current_segment()
        return []

    def _transcribe_segment_audio(self, segment_audio: List[bytes]) -> None:
        """Transcribe one whole non-streaming segment's audio, exactly once.

        Called only from `_processing_loop`'s non-streaming regime -- via
        `_finalize_non_streaming_segment` (the silence gate and the hard
        segment-size bound) and directly from the tail-drain (stop) -- never
        on a cadence, and never for the streaming regime, whose own finals
        arrive push-style through `_handle_streamed_final`. Reuses
        `_process_audio_buffer()` unchanged (same `transcribe_buffer()` call
        shape: provider/model/language passthrough, sample params), so it
        sets `self.current_transcript` exactly as every other caller of that
        method always has; the caller decides whether/when to finalize it.

        Args:
            segment_audio: The chunks accumulated since the last finalize.
                A no-op when empty (nothing to transcribe).
        """
        if not segment_audio:
            return
        # Fired right here, before the call that can take seconds: this is
        # the ONLY place `_processing_loop` ever calls this method (the
        # mid-capture silence gate and the stop-path tail-fold both funnel
        # through it), so one call site covers both without duplicating the
        # notification at each caller. There is otherwise zero signal in this
        # gap under the segment-at-silence architecture -- no live partial
        # text, nothing -- so a multi-second transcription looks identical to
        # a dead capture without it.
        self._notify_segment_transcribing(done=False)
        audio_data = b"".join(segment_audio)
        # Snapshot before the call and compare after, rather than reading
        # `current_transcript` alone post-call: this call is the only writer
        # while it runs (this thread, sequential with `_finalize_current_
        # segment()`), so any change is exactly what THIS call produced.
        # `current_transcript` should always be "" going in -- the prior
        # segment's own call is always immediately followed by a finalize
        # that clears it -- but reading only the post-call value would
        # misreport a genuinely blank result as "produced text" if residue
        # ever did carry over (e.g. a future change to the streaming ->
        # buffer-API fallback inside `_process_audio_buffer`), since
        # `_handle_partial_text` leaves `current_transcript` untouched, not
        # cleared, when its input is blank.
        with self.transcript_lock:
            before = self.current_transcript
        self._process_audio_buffer(audio_data)
        with self.transcript_lock:
            produced_text = self.current_transcript != before
        # Unconditional, and always the LAST thing this method does: a
        # blank/whitespace-only result (routine -- see the log line below)
        # fires neither `on_partial_transcript` nor `on_final_transcript`,
        # since `_handle_partial_text` no-ops on blank input and
        # `_finalize_current_segment()` no-ops on empty text. Without an
        # unconditional completion signal here, a consumer that shows a
        # "transcribing" indicator on the `done=False` call above and hides
        # it on the next partial/final has nothing to hide it on for a blank
        # segment -- it would stay shown for the rest of the capture (review
        # finding M1). `_process_audio_buffer` never raises (it catches and
        # reports through `on_error` internally), so this always runs
        # immediately after it, symmetric with the `done=False` call above:
        # exactly one of each per segment.
        self._notify_segment_transcribing(done=True)
        if not produced_text:
            # Whisper-family models routinely return empty or whitespace-only
            # text for room noise or a too-short VAD sliver -- routine, not a
            # failure, so this stays a debug log rather than `on_error`/a
            # user-facing notice. `_finalize_current_segment()` already no-ops
            # on empty text, so nothing further is needed to drop the segment.
            logger.debug(
                "Dictation segment ({} bytes) produced no usable transcript; "
                "dropping silently",
                len(audio_data),
            )

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

        For the streaming regime, chunks arrive roughly every
        ``buffer_duration_ms``, and this accumulation is what keeps a
        segment from holding only its most recent half-second of speech
        instead of everything said so far. For the non-streaming regime this
        is called exactly once per segment (see ``_transcribe_segment_audio``),
        so there is nothing to accumulate onto -- but it is still the same
        method, since ``current_transcript`` is always empty going in either
        way (the previous segment's own call is immediately followed by a
        finalize that clears it).
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

    def _notify_segment_transcribing(self, done: bool):
        """Tell the caller a whole-segment transcription is starting or has ended.

        Called only from `_transcribe_segment_audio`, on this thread: once
        with `done=False` right before the call that can take seconds, once
        with `done=True` right after it returns -- unconditionally, including
        a blank/whitespace-only result (see that method for why the second
        call must be unconditional). Advisory, like every other
        `_notify_*`/callback invocation in this class: never lets a raising
        callback escape into the processing loop.

        Args:
            done: False for the "started" signal, True for the "completed"
                signal.
        """
        if self.on_segment_transcribing:
            try:
                self.on_segment_transcribing(done)
            except Exception as e:
                logger.error(f"Segment-transcribing callback error: {e}")


class AudioInitializationError(Exception):
    """Raised when audio initialization fails with user-friendly message."""

    pass


class TranscriptionInitializationError(Exception):
    """Raised when transcription initialization fails with user-friendly message."""

    pass
