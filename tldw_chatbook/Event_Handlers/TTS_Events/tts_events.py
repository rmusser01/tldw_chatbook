# tts_events.py
# Description: Event handlers for TTS functionality
#
# Imports
import asyncio
import inspect
import re
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from functools import partial
from typing import Dict, Literal, Optional, TypeVar
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from uuid import uuid4
from loguru import logger

# Third-party imports
from textual.message import Message

# Local imports
from tldw_chatbook.Audio.streaming_sink import (
    SinkStarted,
    SinkUnderrun,
    StreamingPcmSink,
    pump,
    sink_available,
    stop_live_sink,
)
from tldw_chatbook.Chat.console_speech import (
    ConsoleSpeechSnapshotRejected,
    TTSMessageSpeechSnapshot,
)
from tldw_chatbook.Chat.console_speech_preferences import (
    is_console_speech_destination,
)
from tldw_chatbook.TTS import (
    CharacterTTSRequestResolution,
    CharacterTTSRequestResolver,
    CharacterTTSResolutionError,
    TTSRequestedSelectionSnapshot,
    get_tts_service,
)
from tldw_chatbook.TTS.base_backends import TTSBackendConnectionError
from tldw_chatbook.TTS.character_request_resolver import TTSVoiceRefusalDomain
from tldw_chatbook.TTS.legacy_bridge import UnknownLegacyModelError
from tldw_chatbook.TTS.adapter_types import (
    TTSConfigurationRevisionError,
    TTSOperationError,
    TTSProgress,
    TTSProviderReconfiguringError,
    TTSProviderUnavailableError,
    TTSRequest,
    TTSRegistryClosedError,
)
from tldw_chatbook.TTS.default_profile_request_resolver import (
    resolve_default_profile,
)
from tldw_chatbook.TTS.pcm_stream import SinkPlan, sink_plan
from tldw_chatbook.TTS.effective_settings import (
    TTSCharacterProfileSelection,
    TTSDefaultProfileSelection,
    TTSEffectiveSettingsResolver,
    TTSSelectionOverrides,
)
from tldw_chatbook.TTS.openai_compatible_config import (
    is_loopback_openai_compatible_endpoint,
    normalize_openai_compatible_endpoint,
    openai_destination_fingerprint,
)
from tldw_chatbook.TTS.profile_types import CharacterRef
from tldw_chatbook.Utils.secure_temp_files import get_temp_manager, secure_delete_file

_T = TypeVar("_T")
_TTS_ARTIFACT_WRITE_BATCH_BYTES = 64 * 1024
_TTS_IO_CANCELLATION_JOIN_TIMEOUT_SECONDS = 1.0
_TTS_SECURE_DELETE_TIMEOUT_SECONDS = 1.0
_TTS_RETAINED_WORK_DRAIN_TIMEOUT_SECONDS = 1.0
_GLOBAL_OVERRIDE_TOKEN_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
# F4 fix-round: hard cap, in bytes, on the in-memory WAV body the
# opportunistic sink-upgrade attempt (see `wav_collect` in `_generate_tts`)
# will accumulate before giving up and falling through to the legacy,
# already-fully-written disk artifact. Derived from the streaming sink's
# own BUFFER_CAP_SECONDS (`Audio/streaming_sink.py`, 60s -- the most audio
# it could ever actually buffer and play) at a deliberately generous
# worst-case PCM16 rate -- 48kHz stereo, 2 bytes/sample:
#   60s * 48_000Hz * 2ch * 2bytes = 11_520_000 bytes (~11.5MB)
# rounded up to a clean 16MiB. A WAV response bigger than this could never
# be fully played through the sink's own buffer anyway (`feed()` rejects
# the tail past BUFFER_CAP_SECONDS of buffered audio at the response's
# REAL sample rate), so abandoning the attempt here costs nothing in
# correctness -- only saves the memory/CPU of accumulating (and then
# copying) a body this large just to discover that.
_MAX_WAV_SINK_UPGRADE_BYTES = 16 * 1024 * 1024
# Task-4 review F2: `_play_legacy_clip_and_await_completion` polls
# `SimpleAudioPlayer.get_state()`/`get_current_file()` (off the event loop)
# until a just-started legacy clip stops being the player's current one,
# rather than reporting completion the instant `play()` hands the clip to a
# background process (see that function's own docstring for why the old
# behavior amounted to truncating every sentence but the last). There is no
# real audio duration available to bound the poll against --
# `AudioPlayerInfo.duration` (`TTS/audio_player.py`) is declared but never
# populated by `play()` -- so the bound is estimated from the SYNTHESIZED
# TEXT's own length, at a deliberately slow (over-estimating) rate, divided
# by the resolved provider speed (task-4 review N3 -- `default_speed` is
# user-configurable and only validated as "finite positive", so a bound
# that ignores it can under-estimate at a slower-than-1x setting), plus a
# fixed startup/latency margin, capped at an absolute ceiling so a
# pathological input can never poll indefinitely.
_LEGACY_PLAYBACK_MIN_CHARS_PER_SECOND = 8.0
_LEGACY_PLAYBACK_POLL_MARGIN_SECONDS = 3.0
# Task-4 review N3: raised from 120s -- a 5000-char utterance (the max
# `_prepare_tts_text` allows) at the conservative assumed rate needs ~625s;
# no fixed ceiling can cover an arbitrarily slow real reading of arbitrarily
# long text, but this is far more generous while staying bounded (and an
# operator/system can still cancel a stuck utterance -- task-4 review F4's
# `_active_tasks` registration, N4's prompt cancel-stop).
_LEGACY_PLAYBACK_POLL_MAX_SECONDS = 300.0
_LEGACY_PLAYBACK_POLL_INTERVAL_SECONDS = 0.05
_ANY_ARTIFACT_OWNER = object()


class _TTSResponseContractError(RuntimeError):
    """Raised when a synthesized response violates the Console audio contract."""


class _TTSAutomaticDestinationChangedError(RuntimeError):
    """Automatic speech no longer targets its consented authority."""


class _TTSArtifactIOTimeout(RuntimeError):
    """Raised when bounded artifact I/O continues in a retained worker."""


@dataclass(frozen=True, slots=True)
class ConsoleTTSDestination:
    """Text-free effective destination used for per-conversation consent."""

    fingerprint: str
    provider_label: str
    sanitized_destination: str
    charges_may_apply: bool

    def __post_init__(self) -> None:
        if type(self.fingerprint) is not str:
            raise ValueError("fingerprint must be a string")
        if type(self.provider_label) is not str or not self.provider_label:
            raise ValueError("provider_label must be a non-empty string")
        if type(self.sanitized_destination) is not str:
            raise ValueError("sanitized_destination must be a string")
        if type(self.charges_may_apply) is not bool:
            raise ValueError("charges_may_apply must be a boolean")


#######################################################################################################################
#
# TTS Event Messages


class TTSRequestEvent(Message):
    """Explicit trusted global-speech request without a Console message."""

    def __init__(
        self, text: str, message_id: Optional[str] = None, voice: Optional[str] = None
    ):
        super().__init__()
        self.text = text
        self.message_id = message_id  # ID of the chat message
        self.voice = voice  # Optional voice override


class TTSMessageSpeechRequestEvent(Message):
    """Request speech for one store-issued immutable Console snapshot."""

    def __init__(
        self,
        snapshot: TTSMessageSpeechSnapshot,
        validator: Callable[[TTSMessageSpeechSnapshot], str],
        outcome_callback: Callable[[bool], None] | None = None,
        expected_destination_fingerprint: str | None = None,
        retry_failed_auto: bool = False,
        playback_lifecycle: "TTSPlaybackLifecycle | None" = None,
    ) -> None:
        super().__init__()
        if type(snapshot) is not TTSMessageSpeechSnapshot:
            raise ValueError("snapshot must be TTSMessageSpeechSnapshot")
        if not callable(validator):
            raise ValueError("validator must be callable")
        if outcome_callback is not None and not callable(outcome_callback):
            raise ValueError("outcome_callback must be callable or None")
        if (
            expected_destination_fingerprint is not None
            and not is_console_speech_destination(
                expected_destination_fingerprint
            )
        ):
            raise ValueError(
                "expected_destination_fingerprint must be canonical or None"
            )
        if type(retry_failed_auto) is not bool:
            raise ValueError("retry_failed_auto must be an exact boolean")
        if retry_failed_auto and expected_destination_fingerprint is None:
            raise ValueError("automatic retry requires a destination fingerprint")
        if playback_lifecycle is not None and (
            type(playback_lifecycle) is not TTSPlaybackLifecycle
            or playback_lifecycle.message_id != snapshot.message_id
        ):
            raise ValueError("playback_lifecycle must match the snapshot message")
        self.snapshot = snapshot
        self.validator = validator
        self._outcome_callback = outcome_callback
        self.expected_destination_fingerprint = expected_destination_fingerprint
        self.retry_failed_auto = retry_failed_auto
        self.playback_lifecycle = playback_lifecycle
        self._outcome_reported = False

    @property
    def message_id(self) -> str:
        """Expose the native message id without duplicating caller text."""
        return self.snapshot.message_id

    def report_outcome(self, succeeded: bool) -> None:
        """Report one bounded terminal result without exposing request data."""
        if self._outcome_reported or self._outcome_callback is None:
            return
        self._outcome_reported = True
        try:
            self._outcome_callback(succeeded is True)
        except Exception:
            logger.warning("Console speech outcome callback failed")

    @property
    def has_outcome_callback(self) -> bool:
        """Return whether this request needs an automatic-speech result."""
        return self._outcome_callback is not None


class TTSStreamingEvent(Message):
    """Event for streaming TTS audio chunks"""

    def __init__(self, chunk: bytes, message_id: str, is_final: bool = False):
        super().__init__()
        self.chunk = chunk
        self.message_id = message_id
        self.is_final = is_final


class TTSCompleteEvent(Message):
    """Event when TTS generation is complete"""

    def __init__(
        self,
        message_id: str,
        audio_file: Optional[Path] = None,
        error: Optional[str] = None,
        global_override_token: str | None = None,
        playback_lifecycle: "TTSPlaybackLifecycle | None" = None,
    ):
        super().__init__()
        if global_override_token is not None and not (
            type(global_override_token) is str
            and _GLOBAL_OVERRIDE_TOKEN_PATTERN.fullmatch(global_override_token)
        ):
            raise ValueError("global override token must be lowercase hexadecimal")
        self.message_id = message_id
        self.audio_file = audio_file
        self.error = error
        self.global_override_token = global_override_token
        if playback_lifecycle is not None and (
            type(playback_lifecycle) is not TTSPlaybackLifecycle
            or playback_lifecycle.message_id != message_id
        ):
            raise ValueError("playback_lifecycle must match message_id")
        self.playback_lifecycle = playback_lifecycle


PlaybackLifecycleState = Literal["playing", "stopped", "failed"]


class TTSPlaybackLifecycle:
    """Bounded request-owned playback signal without message content."""

    __slots__ = (
        "message_id",
        "request_id",
        "_validator",
        "_callback",
        "_state",
        "_lock",
    )

    def __init__(
        self,
        *,
        message_id: str,
        request_id: int,
        validator: Callable[[], bool],
        callback: Callable[[PlaybackLifecycleState], None],
    ) -> None:
        if type(message_id) is not str or not message_id:
            raise ValueError("message_id must be a non-empty string")
        if type(request_id) is not int or request_id < 1:
            raise ValueError("request_id must be a positive integer")
        if not callable(validator) or not callable(callback):
            raise ValueError("validator and callback must be callable")
        self.message_id = message_id
        self.request_id = request_id
        self._validator = validator
        self._callback = callback
        self._state: Literal["generating", "playing", "stopped", "failed"] = (
            "generating"
        )
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def is_current(self) -> bool:
        with self._lock:
            if self._state in {"stopped", "failed"}:
                return False
        try:
            return self._validator() is True
        except Exception:
            return False

    def report(self, state: PlaybackLifecycleState) -> bool:
        if state not in {"playing", "stopped", "failed"}:
            raise ValueError("unsupported playback lifecycle state")
        if not self.is_current():
            return False
        return self._transition(state)

    def report_terminal(self, state: Literal["stopped", "failed"]) -> bool:
        """Settle an exact handler-owned lifecycle after ownership matched.

        Terminal acknowledgements must survive a screen/session validator
        becoming stale while the handler is physically stopping its owner.
        Admission and playback-start signals still use :meth:`report`.
        """
        if state not in {"stopped", "failed"}:
            raise ValueError("terminal playback state must be stopped or failed")
        return self._transition(state)

    def _transition(self, state: PlaybackLifecycleState) -> bool:
        with self._lock:
            current = self._state
            if current in {"stopped", "failed"}:
                return False
            if state == "playing" and current != "generating":
                return False
            if state in {"stopped", "failed"} and current not in {
                "generating",
                "playing",
            }:
                return False
            self._state = state
        try:
            self._callback(state)
        except Exception:
            logger.warning("Console playback lifecycle callback failed")
        return True


@dataclass(slots=True)
class _ConsoleGenerationOwner:
    """One exact handler-owned Console generation task."""

    lifecycle: TTSPlaybackLifecycle
    task: asyncio.Task[None] | None = None
    cancel_as_success: bool = False


class TTSGlobalOverrideDecisionEvent(Message):
    """Accept or decline one handler-issued global-voice fallback capability."""

    def __init__(self, token: str, accepted: bool) -> None:
        super().__init__()
        if type(token) is not str or not _GLOBAL_OVERRIDE_TOKEN_PATTERN.fullmatch(
            token
        ):
            raise ValueError("token must be 32 lowercase hexadecimal characters")
        if type(accepted) is not bool:
            raise ValueError("accepted must be a boolean")
        self.token = token
        self.accepted = accepted


class TTSPlaybackEvent(Message):
    """Event to control TTS playback"""

    def __init__(
        self,
        action: str,
        message_id: Optional[str] = None,
        *,
        playback_lifecycle: TTSPlaybackLifecycle | None = None,
        outcome_callback: Callable[[bool], None] | None = None,
    ):
        super().__init__()
        if playback_lifecycle is not None and (
            type(playback_lifecycle) is not TTSPlaybackLifecycle
            or playback_lifecycle.message_id != message_id
        ):
            raise ValueError("playback_lifecycle must match message_id")
        if outcome_callback is not None and not callable(outcome_callback):
            raise ValueError("outcome_callback must be callable or None")
        self.action = action  # "play", "pause", "stop"
        self.message_id = message_id
        self.playback_lifecycle = playback_lifecycle
        self._outcome_callback = outcome_callback
        self._outcome_reported = False

    def report_outcome(self, accepted: bool) -> None:
        if self._outcome_reported or self._outcome_callback is None:
            return
        self._outcome_reported = True
        try:
            self._outcome_callback(accepted is True)
        except Exception:
            logger.warning("TTS playback control callback failed")


class TTSProgressEvent(Message):
    """Event for TTS generation progress updates"""

    def __init__(
        self,
        message_id: str,
        progress: float,
        status: str,
        estimated_time_remaining: Optional[float] = None,
    ):
        super().__init__()
        self.message_id = message_id
        self.progress = progress  # 0.0 to 1.0
        self.status = status  # e.g., "Processing", "Generating", "Finalizing"
        self.estimated_time_remaining = estimated_time_remaining  # seconds


@dataclass
class TTSUsageRecord:
    """Single TTS usage record for local cost tracking."""

    provider: str
    model: str
    characters: int
    voice: str
    format: str
    estimated_cost: float
    created_at: datetime


@dataclass(frozen=True, slots=True)
class _PendingGlobalOverride:
    """Private admission state retained behind an opaque one-use token.

    ``voice_domain`` (review round 2) records which configured-voice
    domain actually failed and issued this token -- read back, without
    consuming the token, by `TTSEventHandler.peek_global_override_voice_
    domain` so `app.py::_offer_tts_global_override` can render a
    confirmation dialog that names the domain the user is really
    consenting to bypass, instead of a hardcoded assumption.
    """

    snapshot: TTSMessageSpeechSnapshot
    validator: Callable[[TTSMessageSpeechSnapshot], str]
    created_at: float
    voice_domain: TTSVoiceRefusalDomain = "character"

    def __post_init__(self) -> None:
        if self.voice_domain not in ("character", "default_profile"):
            raise ValueError("voice_domain must be bounded")


class CostTracker:
    """Lightweight local TTS usage and cost tracker."""

    DEFAULT_COSTS = {
        ("openai", "tts-1"): {"cost_per_1k_chars": 0.015, "free_tier_chars": 0},
        ("openai", "tts-1-hd"): {"cost_per_1k_chars": 0.030, "free_tier_chars": 0},
        ("local", "*"): {"cost_per_1k_chars": 0.0, "free_tier_chars": 0},
    }

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else None
        self._costs: Dict[tuple[str, str], Dict[str, float]] = {
            key: value.copy() for key, value in self.DEFAULT_COSTS.items()
        }
        self._usage: list[TTSUsageRecord] = []

    def update_cost_info(
        self,
        provider: str,
        cost_per_1k_chars: float,
        free_tier_chars: int = 0,
        model: str = "*",
    ) -> None:
        self._costs[(provider.lower(), model.lower())] = {
            "cost_per_1k_chars": float(cost_per_1k_chars),
            "free_tier_chars": int(free_tier_chars),
        }

    def estimate_cost(self, provider: str, model: str, characters: int) -> float:
        provider_key = provider.lower()
        model_key = model.lower()
        cost_info = (
            self._costs.get((provider_key, model_key))
            or self._costs.get((provider_key, "*"))
            or {"cost_per_1k_chars": 0.0, "free_tier_chars": 0}
        )
        free_tier_chars = int(cost_info.get("free_tier_chars", 0))
        used_chars = self._monthly_usage_for_provider(provider_key)
        billable_chars = max(0, int(characters) - max(0, free_tier_chars - used_chars))
        return (billable_chars / 1000.0) * float(
            cost_info.get("cost_per_1k_chars", 0.0)
        )

    def track_usage(
        self,
        provider: str,
        model: str,
        text: str,
        voice: str,
        format: str,
    ) -> TTSUsageRecord:
        characters = len(text)
        record = TTSUsageRecord(
            provider=provider,
            model=model,
            characters=characters,
            voice=voice,
            format=format,
            estimated_cost=self.estimate_cost(provider, model, characters),
            created_at=datetime.now(),
        )
        self._usage.append(record)
        return record

    def get_monthly_usage(self) -> int:
        return sum(record.characters for record in self._usage)

    def get_monthly_cost(self) -> float:
        return sum(record.estimated_cost for record in self._usage)

    def _monthly_usage_for_provider(self, provider: str) -> int:
        return sum(
            record.characters
            for record in self._usage
            if record.provider.lower() == provider
        )


#######################################################################################################################
#
# TTS Event Handler Mixin


class TTSEventHandler:
    """
    Mixin class for handling TTS events.

    Note: Rate limiting is handled by the TTSService itself with a global
    semaphore of 4 concurrent requests. This handler only implements
    cooldown periods to prevent rapid repeated requests for the same message.
    """

    # Cooldown tracking to prevent rapid repeated requests
    _request_cooldown: Dict[str, float] = {}  # Track last request time per message
    COOLDOWN_SECONDS = 2.0  # Minimum time between requests for same message
    COOLDOWN_CLEANUP_INTERVAL = 300.0  # Clean up old entries every 5 minutes
    MAX_COOLDOWN_ENTRIES = 1000  # Maximum entries to keep in memory
    GLOBAL_OVERRIDE_TTL_SECONDS = 300.0
    MAX_PENDING_GLOBAL_OVERRIDES = 32

    def __init__(
        self,
        profile_service_loader: Callable[[], Awaitable[object | None]] | None = None,
        default_profile_id_reader: Callable[[], object | None] | None = None,
    ):
        self._tts_service = None
        self._profile_service_loader = profile_service_loader
        # Task-4 (slice 3): reads the persisted `[app_tts] default_profile_id`
        # setting -- injected, like `profile_service_loader` above, so tests
        # never touch real config, and `None` means "no app-default voice is
        # wired up here", identical in effect to it being unconfigured.
        self._default_profile_id_reader = default_profile_id_reader
        self._pending_global_overrides: dict[str, _PendingGlobalOverride] = {}
        self._temp_manager = get_temp_manager()
        self._audio_files: Dict[str, Path] = {}  # Track audio files by message_id
        self._audio_file_owners: Dict[
            str, TTSPlaybackLifecycle | None
        ] = {}  # Same bounded key set as _audio_files; None is legacy ownership.
        self._artifact_cleanup_retry: set[Path] = set()
        self._retained_tts_io_tasks: set[asyncio.Task] = set()
        self._retained_tts_cleanup_tasks: set[asyncio.Task] = set()
        # Task-4 review N5: kept SEPARATE from `_retained_tts_cleanup_
        # tasks` above -- that set's tasks are genuinely in-flight I/O
        # `cleanup_tts_resources()` gives a bounded chance to finish
        # naturally. `_schedule_legacy_playback_cleanup`'s tasks are pure
        # `asyncio.sleep(5)`-then-delete timers; `cleanup_tts_resources()`
        # deletes the same artifact directly moments later regardless, so
        # AWAITING one (even boundedly) only wastes shutdown time for zero
        # benefit -- it is CANCELLED there instead, never drained.
        self._pending_legacy_cleanup_timers: set[asyncio.Task] = set()
        self._retained_tts_cleanup_paths: set[Path] = set()
        self._retained_tts_cleanup_requeue: dict[Path, str] = {}
        # task-559 fix round 1: which file the player last loaded, tracked
        # independently of `_audio_files` -- that cache is deleted 5s after
        # playback STARTS (see handle_tts_playback's "play" branch), well
        # before longer clips finish, so a stop-guard reading `_audio_files`
        # alone silently stops working past that window.
        #
        # fix round 2 (Qodo PR #867): a single `(message_id, path)` slot,
        # not a per-message dict -- `SimpleAudioPlayer` (`TTS/audio_player.
        # py`) is itself a single-slot global singleton (only one clip can
        # be "current" system-wide; every `play()` stops whatever was
        # previously loaded first), so a dict entry per message could only
        # ever grow (every auto-played message adds one; only an explicit
        # stop or shutdown removes one) without ever reflecting more than
        # one real, simultaneously-relevant entry. Overwritten on every
        # play; cleared on a matching stop or handler shutdown. Protected
        # by the same lock as `_audio_files` (related bookkeeping, always
        # touched together).
        self._last_played: Optional[tuple[str, Path]] = None
        # Console playback has one process-global audio owner. These slots
        # mirror that bound and let an explicit Stop reach the pre-player
        # handoff window without retaining per-message history.
        self._active_file_playback_task: asyncio.Task[None] | None = None
        self._active_file_playback_owner: TTSPlaybackLifecycle | None = None
        self._active_file_playback_stop: tuple[str, threading.Event] | None = None
        self._active_file_playback_started: threading.Event | None = None
        self._active_stream_playback_owner: TTSPlaybackLifecycle | None = None
        self._file_play_admission_lock = asyncio.Lock()
        self._playback_handoff_lock = asyncio.Lock()
        self._console_generation_owner: _ConsoleGenerationOwner | None = None
        # Task-4 review round 2 (F3+N2), round 3 (D1): one `threading.Event`
        # per IN-FLIGHT `_play_utterance_legacy_artifact` play-and-poll call
        # (added just before `player.play()` runs, discarded just after the
        # poll returns) -- a SET, not a single bool, because a bool cleared
        # unconditionally in a `finally` left the FIRST of two overlapping
        # handoffs to exit clearing protection for the other (round-2's
        # `_legacy_handoff_in_flight` bug, confirmed by a forced-overlap
        # probe; `speak_utterance` is a public entry and two concurrent
        # calls are reachable even though the sequencer serializes them in
        # practice). A bare/global stop (`bare_stop=True`) sets EVERY
        # in-flight event -- "stop everything" is a bare stop's whole
        # semantics -- so `_stop_prior_legacy_clip` gates its unconditional
        # branch on this set being non-empty, and each handoff's own worker
        # thread (round 3's F3 fix, `_play_legacy_clip_and_await_
        # completion`) checks ITS OWN event to close the window a
        # same-tick `player.stop()` call cannot: a stop issued BEFORE
        # `Popen` has no process to kill, and `play()` proceeds past its
        # own internal `self.stop()` and starts the clip regardless --
        # only the worker, which alone knows when `Popen` actually
        # happened, can catch that. Read/written only from the event loop
        # (the SET itself, that is -- the `Event` objects it holds are
        # deliberately cross-thread: `threading.Event.set()`/`.is_set()`
        # are the whole point).
        self._legacy_handoff_stop_events: set[threading.Event] = set()
        self._audio_files_lock = asyncio.Lock()  # Lock for audio files dictionary
        self._active_tasks: set[asyncio.Task] = set()  # Track active async tasks
        self._active_tasks_lock = asyncio.Lock()  # Lock for active tasks set
        self._last_cooldown_cleanup = 0.0  # Track last cleanup time

    async def initialize_tts(self) -> None:
        """Initialize TTS service"""
        try:
            self._tts_service = await get_tts_service()
            logger.info("TTS service initialized successfully")
        except Exception as error:
            logger.error(
                "Failed to initialize TTS service ({})",
                type(error).__name__,
            )
            self._tts_service = None

    def _cleanup_cooldown_dict(self, current_time: float) -> None:
        """Clean up old entries from cooldown dictionary"""
        # Remove entries older than 5 minutes
        cutoff_time = current_time - 300.0
        keys_to_remove = [
            key
            for key, timestamp in self._request_cooldown.items()
            if timestamp < cutoff_time
        ]
        for key in keys_to_remove:
            del self._request_cooldown[key]

        if keys_to_remove:
            logger.debug(f"Cleaned up {len(keys_to_remove)} old cooldown entries")

    def _enforce_cooldown_limit(self) -> None:
        """Trim cooldown tracking to the maximum configured entries."""
        if len(self._request_cooldown) <= self.MAX_COOLDOWN_ENTRIES:
            return

        sorted_entries = sorted(self._request_cooldown.items(), key=lambda x: x[1])
        overflow = len(sorted_entries) - self.MAX_COOLDOWN_ENTRIES
        for key, _ in sorted_entries[:overflow]:
            del self._request_cooldown[key]

    async def _post_tts_message(self, message: Message) -> bool:
        """Post a message through Textual app wiring or a direct test handler."""
        try:
            app = getattr(self, "app", None)
            if app is not None and hasattr(app, "post_message"):
                result = app.post_message(message)
            else:
                post_message = getattr(self, "post_message", None)
                if not callable(post_message):
                    return False
                result = post_message(message)
            if asyncio.iscoroutine(result):
                result = await result
            return result is not False
        except Exception:
            return False

    async def handle_tts_request(
        self,
        event: TTSRequestEvent | TTSMessageSpeechRequestEvent,
    ) -> None:
        """Admit a trusted request, then run the shared TTS generation path."""
        if isinstance(event, TTSMessageSpeechRequestEvent):
            try:
                await self._handle_trusted_message_speech_request(event)
            except asyncio.CancelledError:
                event.report_outcome(False)
                raise
            except Exception as error:  # noqa: BLE001 - terminal trust boundary
                logger.warning(
                    "Trusted Console speech request failed "
                    "(exception_category={})",
                    type(error).__name__,
                )
                try:
                    await self._post_tts_message(
                        TTSCompleteEvent(
                            message_id=event.message_id,
                            error="Speech could not be generated.",
                        )
                    )
                except Exception as post_error:  # noqa: BLE001
                    logger.warning(
                        "Trusted Console speech failure notice was rejected "
                        "(exception_category={})",
                        type(post_error).__name__,
                    )
                finally:
                    event.report_outcome(False)
            return

        request_text = event.text
        request_message_id = event.message_id
        request_voice = event.voice
        # Preserve the legacy explicit-request maintenance behavior even
        # when no service is available.
        self._enforce_cooldown_limit()

        effective_message_id = (
            request_message_id
            if isinstance(request_message_id, str) and request_message_id
            else "adhoc"
        )
        text = await self._prepare_tts_text(
            request_text,
            effective_message_id,
        )
        if text is None:
            if isinstance(event, TTSMessageSpeechRequestEvent):
                event.report_outcome(False)
            return

        await self._admit_tts_generation(
            text=text,
            message_id=effective_message_id,
            voice=request_voice,
            resolution=None,
        )

    async def _handle_trusted_message_speech_request(
        self,
        event: TTSMessageSpeechRequestEvent,
    ) -> None:
        request_text = await self._validate_message_speech_snapshot(
            event.snapshot,
            event.validator,
            playback_lifecycle=event.playback_lifecycle,
        )
        if request_text is None:
            event.report_outcome(False)
            return
        text = await self._prepare_tts_text(
            request_text,
            event.message_id,
            playback_lifecycle=event.playback_lifecycle,
        )
        if text is None:
            event.report_outcome(False)
            return
        try:
            resolution = await self._resolve_message_speech_request(
                text,
                event.snapshot,
            )
        except CharacterTTSResolutionError as error:
            token = None
            if error.allow_global_override:
                token = self._issue_global_override(
                    event.snapshot,
                    event.validator,
                    error.domain,
                )
            logger.warning(
                "Console speech voice resolution failed (outcome_code={})",
                error.code,
            )
            await self._post_tts_message(
                TTSCompleteEvent(
                    message_id=event.message_id,
                    error=str(error),
                    global_override_token=token,
                    playback_lifecycle=event.playback_lifecycle,
                )
            )
            event.report_outcome(False)
            return

        expected = event.expected_destination_fingerprint
        if expected is not None:
            destination = await self._destination_for_resolution(resolution)
            if destination is None or destination.fingerprint != expected:
                logger.warning(
                    "Automatic Console speech destination changed "
                    "(outcome_code=destination_changed)"
                )
                await self._post_tts_message(
                    TTSCompleteEvent(
                        message_id=event.message_id,
                        error=(
                            "The speech destination changed. Confirm Speak replies "
                            "again."
                        ),
                        playback_lifecycle=event.playback_lifecycle,
                    )
                )
                event.report_outcome(False)
                return

        await self._admit_tts_generation(
            text=text,
            message_id=event.message_id,
            voice=None,
            resolution=resolution,
            outcome_callback=(
                event.report_outcome if event.has_outcome_callback else None
            ),
            expected_destination_fingerprint=expected,
            retry_failed_auto=event.retry_failed_auto,
            playback_lifecycle=event.playback_lifecycle,
        )

    async def speak_utterance(
        self,
        text: str,
        *,
        on_finished: Callable[[bool], None],
        quiet: bool = False,
    ) -> None:
        """Speak one cooldown-free hands-free-loop utterance.

        The hands-free reply-speech path (`Chat/reply_sentence_sequencer.py`
        's `SentenceSequencer`) dispatches one sentence-sized utterance at a
        time, often within milliseconds of the previous one finishing --
        the OPPOSITE of "rapid repeated requests for the same message" the
        message-cooldown gate (`_admit_tts_generation`, keyed on
        `COOLDOWN_SECONDS`) exists to throttle. This entry therefore skips
        that gate -- and the `_enforce_cooldown_limit()` maintenance call
        the ad-hoc `TTSRequestEvent` branch of `handle_tts_request` runs
        early (:420-436) -- entirely, following the shape of the
        `TTSMessageSpeechRequestEvent` branch instead, which never touches
        it either (task-4 review F9: that branch DOES still reach the real
        per-message-id cooldown throttle through `_admit_tts_generation` --
        only the early maintenance call at :420-436 is ad-hoc-branch-
        specific -- this entry skips BOTH by never calling
        `_admit_tts_generation` at all). Each call is assigned its own
        fresh message id, so even the bookkeeping this bypasses
        (`_request_cooldown`) could never collide across utterances even if
        something else touched it later.

        Reuses the exact same generation + playback machinery every other
        TTS caller shares (`_generate_tts`, streaming-sink branch
        included), so one-voice displacement (a new utterance silences
        whatever the sink or legacy player was doing) keeps working
        identically -- see `_generate_tts`'s own `on_finished` parameter,
        which this threads straight through as the completion signal.

        `on_finished` fires EXACTLY ONCE per call, on every path: a
        completed drain, an interrupted/stopped sink (barge-in), a
        synthesis failure, or a legacy-player failure. The `finally` below
        is a last-resort single-fire guard, not the primary signal source
        -- real terminal points fire `fire` directly (see
        `_stream_response_via_sink` and `_play_utterance_legacy_artifact`);
        this only catches paths that reach here without ever calling it
        (e.g. text validation rejecting the utterance before generation
        ever starts). `fire` also isolates a RAISING `on_finished` (task-4
        review F6) -- that is a caller bug (in production, `utterance_
        finished -> speak -> ...`, real caller code), logged and swallowed
        here rather than let it unwind into `_generate_tts`'s own `try`,
        where it would be misreported as a TTS generation failure.

        The generation is registered in `_active_tasks` (task-4 review F4)
        so `cleanup_tts_resources()` can find and cancel an in-flight
        hands-free utterance at shutdown, exactly like every other TTS
        caller's task -- `speak_utterance` still awaits it directly (this
        entry's own external completion timing is unchanged), only the
        bookkeeping is new.

        Args:
            text: The utterance text to speak (one sentence-sized chunk, in
                the hands-free caller's usage).
            on_finished: Completion callback. Fires EXACTLY ONCE per call,
                on every path (see above) -- a completed drain, an
                interrupted/stopped sink (barge-in), a synthesis failure, or
                a legacy-player failure -- with `True` for a successful
                drain and `False` for anything else.
            quiet: When True, suppresses the user-facing `TTSCompleteEvent
                (error=...)` toast for a text-validation rejection or a
                synthesis failure (task-4 review F5) -- the underlying
                condition is still logged and `on_finished(False)` still
                fires normally; only the toast is skipped. `speak_utterance`
                is deliberately stateless across replies (a hard brief
                constraint), so it has no per-reply toast-aggregation
                policy of its own -- this is the mechanism a stateful
                caller (task 5) can use to show at most one toast per
                reply and quietly skip the rest.
        """
        fired = False

        def fire(ok: bool) -> None:
            nonlocal fired
            if fired:
                return
            fired = True
            try:
                on_finished(ok)
            except Exception:
                logger.warning(
                    "speak_utterance's on_finished callback raised; "
                    "treating as a caller bug, not a TTS generation failure"
                )

        message_id = f"handsfree-{uuid4().hex}"
        try:
            prepared_text = await self._prepare_tts_text(text, message_id, quiet=quiet)
            if prepared_text is None:
                return
            generation_task = asyncio.create_task(
                self._generate_tts(
                    prepared_text,
                    message_id,
                    voice=None,
                    resolution=None,
                    on_finished=fire,
                    quiet=quiet,
                )
            )
            await self._add_active_task(generation_task)
            await generation_task
        finally:
            fire(False)

    async def handle_tts_global_override_decision(
        self,
        event: TTSGlobalOverrideDecisionEvent,
    ) -> None:
        """Consume one fallback decision and re-admit its original snapshot.

        Args:
            event: One opaque, message-scoped global-voice decision.
        """
        pending = self._consume_global_override(event.token)
        if pending is None or not event.accepted:
            return

        request_text = await self._validate_message_speech_snapshot(
            pending.snapshot,
            pending.validator,
        )
        if request_text is None:
            return
        text = await self._prepare_tts_text(
            request_text,
            pending.snapshot.message_id,
        )
        if text is None:
            return

        resolver = CharacterTTSRequestResolver(None)
        resolution = resolver.resolve_explicit_global_override(text=text)

        await self._admit_tts_generation(
            text=text,
            message_id=pending.snapshot.message_id,
            voice=None,
            resolution=resolution,
        )

    async def _validate_message_speech_snapshot(
        self,
        snapshot: TTSMessageSpeechSnapshot,
        validator: Callable[[TTSMessageSpeechSnapshot], str],
        *,
        playback_lifecycle: TTSPlaybackLifecycle | None = None,
    ) -> str | None:
        """Validate one handler-retained snapshot without exposing its content."""
        try:
            request_text = validator(snapshot)
        except ConsoleSpeechSnapshotRejected as error:
            logger.warning(
                "Console speech snapshot rejected (outcome_code={})",
                error.code.value,
            )
            await self._post_tts_message(
                TTSCompleteEvent(
                    message_id=snapshot.message_id,
                    error=str(error),
                    playback_lifecycle=playback_lifecycle,
                )
            )
            return None
        except Exception:
            logger.warning(
                "Console speech snapshot rejected (outcome_code=validator_failure)"
            )
            await self._post_tts_message(
                TTSCompleteEvent(
                    message_id=snapshot.message_id,
                    error=ConsoleSpeechSnapshotRejected.USER_COPY,
                    playback_lifecycle=playback_lifecycle,
                )
            )
            return None
        if type(request_text) is not str:
            logger.warning(
                "Console speech snapshot rejected "
                "(outcome_code=invalid_validator_result)"
            )
            await self._post_tts_message(
                TTSCompleteEvent(
                    message_id=snapshot.message_id,
                    error=ConsoleSpeechSnapshotRejected.USER_COPY,
                    playback_lifecycle=playback_lifecycle,
                )
            )
            return None
        return request_text

    async def _prepare_tts_text(
        self,
        request_text: object,
        message_id: str,
        *,
        quiet: bool = False,
        playback_lifecycle: TTSPlaybackLifecycle | None = None,
    ) -> str | None:
        """Validate and normalize text before assignment or cooldown admission.

        Args:
            quiet: Task-4 review F5 -- suppress the user-facing
                `TTSCompleteEvent(error=...)` toast for a rejection.
                Validation itself, its return value, and the (pre-existing)
                logging are all unaffected; only the toast post is skipped.
                `False` for every existing caller.
        """
        if not self._tts_service:
            logger.error("TTS service not initialized")
            if not quiet:
                await self._post_tts_message(
                    TTSCompleteEvent(
                        message_id=message_id,
                        error="TTS service not available",
                        playback_lifecycle=playback_lifecycle,
                    )
                )
            return None
        if type(request_text) is not str or not request_text:
            if not quiet:
                await self._post_tts_message(
                    TTSCompleteEvent(
                        message_id=message_id,
                        error="No text provided for TTS generation",
                        playback_lifecycle=playback_lifecycle,
                    )
                )
            return None

        max_tts_length = 5000
        if len(request_text) > max_tts_length:
            logger.warning("TTS text exceeds the configured length limit")
            if not quiet:
                await self._post_tts_message(
                    TTSCompleteEvent(
                        message_id=message_id,
                        error=(
                            "Text is too long for TTS. Maximum "
                            f"{max_tts_length} characters allowed."
                        ),
                        playback_lifecycle=playback_lifecycle,
                    )
                )
            return None

        text = " ".join(request_text.split())
        if not text:
            if not quiet:
                await self._post_tts_message(
                    TTSCompleteEvent(
                        message_id=message_id,
                        error="Text contains only whitespace",
                        playback_lifecycle=playback_lifecycle,
                    )
                )
            return None
        return text

    async def _resolve_message_speech_request(
        self,
        text: str,
        snapshot: TTSMessageSpeechSnapshot,
    ) -> CharacterTTSRequestResolution:
        """Resolve an exact character assignment, else the app default voice.

        A per-character voice always wins when one is assigned -- the
        default profile is never even loaded in that case (Task 4's
        honesty requirement runs the other way too: a MORE specific voice
        must not be silently overridden by a less specific one). Only once
        `CharacterTTSRequestResolver` itself lands on `"global"` (no
        character voice applies, whether because this message has no
        character context at all or because its character has no
        assignment) is the app-wide default profile even consulted, and
        only when one is actually configured.
        """
        return await self._resolve_speech_request_identity(
            text=text,
            assistant_kind=snapshot.assistant_kind,
            character_ref=snapshot.character_ref,
        )

    async def _resolve_speech_request_identity(
        self,
        *,
        text: str,
        assistant_kind: str | None,
        character_ref: CharacterRef | None,
    ) -> CharacterTTSRequestResolution:
        """Resolve character/default/global authority without retaining text."""
        profile_service: object | None = None
        profile_service_loaded = False

        async def ensure_profile_service() -> object | None:
            nonlocal profile_service, profile_service_loaded
            if not profile_service_loaded:
                loader = self._profile_service_loader
                if loader is not None:
                    try:
                        profile_service = await loader()
                    except asyncio.CancelledError:
                        raise
                    except Exception as error:
                        logger.warning(
                            "TTS profile service load failed (exception_category={})",
                            type(error).__name__,
                        )
                profile_service_loaded = True
            return profile_service

        if assistant_kind == "character":
            await ensure_profile_service()

        resolver = CharacterTTSRequestResolver(profile_service)
        resolution = await resolver.resolve(
            text=text,
            assistant_kind=assistant_kind,
            character_ref=character_ref,
        )
        if resolution.source != "global":
            return resolution

        default_profile_id = self._read_default_profile_id()
        if default_profile_id is None:
            return resolution

        default_profile_service = await ensure_profile_service()
        return await resolve_default_profile(
            text=text,
            default_profile_id=default_profile_id,
            profile_service=default_profile_service,
        )

    async def resolve_console_speech_destination(
        self,
        assistant_kind: str | None,
        character_ref: CharacterRef | None,
    ) -> ConsoleTTSDestination | None:
        """Resolve the effective TTS authority without using message text."""
        service = self._tts_service
        if service is None:
            return None
        resolution = await self._resolve_speech_request_identity(
            text="destination-resolution",
            assistant_kind=assistant_kind,
            character_ref=character_ref,
        )
        return await self._destination_for_resolution(resolution)

    async def _destination_for_resolution(
        self,
        resolution: CharacterTTSRequestResolution,
    ) -> ConsoleTTSDestination | None:
        """Resolve the network authority for the exact effective selection."""
        service = self._tts_service
        if service is None:
            return None
        character_profile = None
        default_profile = None
        if resolution.source in {"assigned", "default_profile"}:
            request = resolution.request
            if (
                request is None
                or resolution.repository_generation is None
                or resolution.profile_id is None
                or resolution.profile_revision is None
            ):
                return None
            selection = TTSSelectionOverrides(
                provider_id=request.provider_id,
                model_mode="exact",
                model_id=request.model_id,
                voice_mode="server_default" if request.voice is None else "exact",
                voice_id=request.voice,
                response_format=request.response_format,
                speed=request.speed,
                provider_options=request.options,
            )
            if resolution.source == "assigned":
                character_profile = TTSCharacterProfileSelection(
                    selection=selection,
                    repository_generation=resolution.repository_generation,
                    profile_revision=resolution.profile_revision,
                    profile_id=resolution.profile_id,
                    reference=resolution.reference,
                )
            else:
                default_profile = TTSDefaultProfileSelection(
                    selection=selection,
                    repository_generation=resolution.repository_generation,
                    profile_revision=resolution.profile_revision,
                    profile_id=resolution.profile_id,
                    reference=resolution.reference,
                )

        effective = await TTSEffectiveSettingsResolver().resolve_non_studio(
            global_preferences=service.preferences_snapshot(),
            global_preferences_revision=service.preferences_generation(),
            provider_revision_reader=service.configuration_revision,
            catalog_reader=service.get_catalog,
            character_profile=character_profile,
            default_profile=default_profile,
        )
        provider_id = effective.provider_id
        provider_label = next(
            (
                descriptor.display_name
                for descriptor in service.provider_descriptors()
                if descriptor.provider_id == provider_id
            ),
            provider_id,
        )
        raw_endpoint = await self._effective_provider_endpoint(service, provider_id)
        endpoint = normalize_openai_compatible_endpoint(raw_endpoint)
        return ConsoleTTSDestination(
            fingerprint=(
                "sha256:"
                f"{openai_destination_fingerprint(provider_id, endpoint)}"
            ),
            provider_label=provider_label,
            sanitized_destination=endpoint.origin,
            charges_may_apply=(
                provider_id in {"openai", "elevenlabs", "alltalk"}
                and not is_loopback_openai_compatible_endpoint(endpoint)
            ),
        )

    @staticmethod
    async def _effective_provider_endpoint(service, provider_id: str) -> str:
        if provider_id == "elevenlabs":
            return "https://api.elevenlabs.io"
        if provider_id not in {"audio_cpp", "openai", "alltalk"}:
            return "http://localhost"

        resolver = getattr(service, "resolve_provider_outbound_endpoint", None)
        if callable(resolver):
            resolved = resolver(provider_id)
            if inspect.isawaitable(resolved):
                return await resolved

        configuration = await service.registry.provider_configuration_snapshot(
            provider_id
        )
        return TTSEventHandler._provider_endpoint_from_applied_config(
            provider_id,
            configuration.applied_config,
        )

    @staticmethod
    def _provider_endpoint_from_applied_config(
        provider_id: str,
        applied: Mapping[str, object],
    ) -> str:
        if provider_id == "audio_cpp":
            if applied.get("mode") == "managed":
                return "http://localhost"
            raw_endpoint = applied.get("base_url")
            if isinstance(raw_endpoint, str) and raw_endpoint:
                return raw_endpoint
            return "http://localhost"
        if provider_id == "elevenlabs":
            return "https://api.elevenlabs.io"
        if provider_id not in {"openai", "alltalk"}:
            return "http://localhost"
        app_config = applied.get("app_config") if isinstance(applied, Mapping) else None
        app_tts = (
            app_config.get("app_tts")
            if isinstance(app_config, Mapping)
            else None
        )
        if isinstance(app_tts, Mapping):
            settings = (
                ("OPENAI_BASE_URL",)
                if provider_id == "openai"
                else ("ALLTALK_TTS_URL", "ALLTALK_TTS_URL_DEFAULT")
            )
            for setting in settings:
                raw_endpoint = app_tts.get(setting)
                if isinstance(raw_endpoint, str) and raw_endpoint:
                    return raw_endpoint
        if provider_id == "alltalk":
            return "http://127.0.0.1:7851"
        return "https://api.openai.com/v1/audio/speech"

    def _read_default_profile_id(self) -> str | None:
        """Return the non-blank configured default profile id, or None.

        Normalizes exactly like Task 3's own loader
        (`settings_speech_tts.py::_normalize_default_profile_id`): absent,
        non-string, empty, and whitespace-only values all mean "not
        configured" and are indistinguishable from the default profile
        never having been set. A non-blank string is passed through
        as-is, whether or not it is a well-formed UUID -- Task 2's loader
        deliberately keeps a malformed value as a defined dangling state
        rather than discarding it, and `resolve_default_profile` is where
        that state is finally interpreted (as unusable, refusing honestly,
        never silently dropped).
        """
        reader = self._default_profile_id_reader
        if reader is None:
            return None
        raw_value = reader()
        if not isinstance(raw_value, str):
            return None
        stripped = raw_value.strip()
        return stripped or None

    def _issue_global_override(
        self,
        snapshot: TTSMessageSpeechSnapshot,
        validator: Callable[[TTSMessageSpeechSnapshot], str],
        voice_domain: TTSVoiceRefusalDomain = "character",
    ) -> str:
        """Retain bounded private fallback state behind a random capability."""
        current_time = asyncio.get_event_loop().time()
        self._prune_global_overrides(current_time)
        if len(self._pending_global_overrides) >= self.MAX_PENDING_GLOBAL_OVERRIDES:
            oldest_token = min(
                self._pending_global_overrides,
                key=lambda token: self._pending_global_overrides[token].created_at,
            )
            del self._pending_global_overrides[oldest_token]

        token = uuid4().hex
        while token in self._pending_global_overrides:
            token = uuid4().hex
        self._pending_global_overrides[token] = _PendingGlobalOverride(
            snapshot=snapshot,
            validator=validator,
            created_at=current_time,
            voice_domain=voice_domain,
        )
        return token

    def peek_global_override_voice_domain(
        self,
        token: str,
    ) -> TTSVoiceRefusalDomain | None:
        """Return a still-pending token's voice domain without consuming it.

        Read-only and advisory (review round 2): the sole caller is
        `app.py::_offer_tts_global_override`, which needs to know whether
        the refusal that issued this token was about a per-character
        assignment or the app-wide default voice profile so its
        confirmation dialog names the right one. Deliberately does not
        enforce the TTL or remove the entry -- the real admission decision
        still goes through `_consume_global_override`, the only method
        that actually spends the capability; a token this call cannot find
        (unknown, already consumed, or -- in practice never, since this is
        always called moments after issuance -- expired) simply returns
        `None`, and the caller falls back to domain-neutral copy.
        """
        pending = self._pending_global_overrides.get(token)
        return None if pending is None else pending.voice_domain

    def _consume_global_override(
        self,
        token: str,
    ) -> _PendingGlobalOverride | None:
        """Atomically consume a non-expired capability."""
        pending = self._pending_global_overrides.pop(token, None)
        if pending is None:
            return None
        current_time = asyncio.get_event_loop().time()
        if current_time - pending.created_at > self.GLOBAL_OVERRIDE_TTL_SECONDS:
            return None
        self._prune_global_overrides(current_time)
        return pending

    def _prune_global_overrides(self, current_time: float) -> None:
        cutoff = current_time - self.GLOBAL_OVERRIDE_TTL_SECONDS
        expired = [
            token
            for token, pending in self._pending_global_overrides.items()
            if pending.created_at < cutoff
        ]
        for token in expired:
            del self._pending_global_overrides[token]

    async def _admit_tts_generation(
        self,
        *,
        text: str,
        message_id: str,
        voice: str | None,
        resolution: CharacterTTSRequestResolution | None,
        outcome_callback: Callable[[bool], None] | None = None,
        expected_destination_fingerprint: str | None = None,
        retry_failed_auto: bool = False,
        playback_lifecycle: TTSPlaybackLifecycle | None = None,
    ) -> None:
        """Apply cooldown only after validation and character resolution."""
        current_time = asyncio.get_event_loop().time()
        if current_time - self._last_cooldown_cleanup > self.COOLDOWN_CLEANUP_INTERVAL:
            self._cleanup_cooldown_dict(current_time)
            self._last_cooldown_cleanup = current_time
        self._enforce_cooldown_limit()

        if retry_failed_auto:
            self._request_cooldown.pop(message_id, None)
        if message_id in self._request_cooldown:
            time_since_last = current_time - self._request_cooldown[message_id]
            if time_since_last < self.COOLDOWN_SECONDS:
                wait_seconds = self.COOLDOWN_SECONDS - time_since_last
                logger.warning(
                    "TTS request rejected by message cooldown (wait_seconds={:.1f})",
                    wait_seconds,
                )
                await self._post_tts_message(
                    TTSCompleteEvent(
                        message_id=message_id,
                        error=(
                            f"Please wait {wait_seconds:.1f} seconds before "
                            "requesting TTS again"
                        ),
                        playback_lifecycle=playback_lifecycle,
                    )
                )
                if outcome_callback is not None:
                    outcome_callback(False)
                return

        self._request_cooldown[message_id] = current_time
        self._enforce_cooldown_limit()

        owner: _ConsoleGenerationOwner | None = None
        if playback_lifecycle is not None:
            await self._cancel_console_generation(superseded=True)
            owner = _ConsoleGenerationOwner(playback_lifecycle)

        if owner is not None:
            generation = self._generate_tts_with_rate_limit(
                text,
                message_id,
                voice,
                resolution,
                outcome_callback=outcome_callback,
                expected_destination_fingerprint=expected_destination_fingerprint,
                playback_lifecycle=playback_lifecycle,
                cancellation_is_success=lambda: owner.cancel_as_success,
            )
        elif outcome_callback is None and expected_destination_fingerprint is None:
            generation = self._generate_tts_with_rate_limit(
                text,
                message_id,
                voice,
                resolution,
            )
        elif outcome_callback is None:
            generation = self._generate_tts_with_rate_limit(
                text,
                message_id,
                voice,
                resolution,
                expected_destination_fingerprint=expected_destination_fingerprint,
            )
        else:
            generation = self._generate_tts_with_rate_limit(
                text,
                message_id,
                voice,
                resolution,
                outcome_callback=outcome_callback,
                expected_destination_fingerprint=(
                    expected_destination_fingerprint
                ),
            )
        task = asyncio.create_task(generation)
        if owner is not None:
            owner.task = task
            self._console_generation_owner = owner

            def clear_owner(done: asyncio.Task) -> None:
                if (
                    self._console_generation_owner is owner
                    and owner.task is done
                ):
                    self._console_generation_owner = None

            task.add_done_callback(clear_owner)
            await self._add_active_task(task)
        else:
            asyncio.create_task(self._add_active_task(task))

    async def _cancel_console_generation(
        self,
        *,
        message_id: str | None = None,
        lifecycle: TTSPlaybackLifecycle | None = None,
        superseded: bool,
    ) -> bool:
        """Cancel and join the exact current Console generation, if matched."""
        owner = self._console_generation_owner
        if owner is None or owner.task is None:
            return False
        if message_id and lifecycle is None:
            return False
        if message_id and owner.lifecycle.message_id != message_id:
            return False
        if lifecycle is not None and owner.lifecycle is not lifecycle:
            return False
        task = owner.task
        if task.done():
            if self._console_generation_owner is owner:
                self._console_generation_owner = None
            return False
        owner.cancel_as_success = superseded
        if not task.cancel():
            return False
        await asyncio.gather(task, return_exceptions=True)
        if self._console_generation_owner is owner:
            self._console_generation_owner = None
        return True

    async def _generate_tts_with_rate_limit(
        self,
        text: str,
        message_id: Optional[str],
        voice: Optional[str],
        resolution: CharacterTTSRequestResolution | None = None,
        *,
        outcome_callback: Callable[[bool], None] | None = None,
        expected_destination_fingerprint: str | None = None,
        playback_lifecycle: TTSPlaybackLifecycle | None = None,
        cancellation_is_success: Callable[[], bool] | None = None,
    ) -> None:
        """Generate TTS audio (rate limiting handled by TTSService)"""
        try:
            await self._generate_tts(
                text,
                message_id,
                voice,
                resolution,
                outcome_callback=outcome_callback,
                expected_destination_fingerprint=(
                    expected_destination_fingerprint
                ),
                playback_lifecycle=playback_lifecycle,
                cancellation_is_success=cancellation_is_success,
            )
        except asyncio.CancelledError:
            logger.info("TTS generation cancelled")
            raise

    async def _generate_tts(
        self,
        text: str,
        message_id: Optional[str],
        voice: Optional[str],
        resolution: CharacterTTSRequestResolution | None = None,
        *,
        on_finished: Callable[[bool], None] | None = None,
        outcome_callback: Callable[[bool], None] | None = None,
        expected_destination_fingerprint: str | None = None,
        quiet: bool = False,
        playback_lifecycle: TTSPlaybackLifecycle | None = None,
        cancellation_is_success: Callable[[], bool] | None = None,
    ) -> None:
        """Generate one complete resolved TTS response and publish its artifact.

        Args:
            on_finished: Task-4 cooldown-free-utterance completion signal.
                ``None`` for every existing caller (spoken feedback,
                character speech, ad-hoc requests) -- behavior for them is
                completely unchanged. When supplied (only `speak_utterance`
                does this), fires exactly once, from whichever terminal
                branch this generation actually takes: the streaming sink's
                own drained/stopped/failed outcome (reported from inside
                `_stream_response_via_sink`), or -- for a response that
                falls through to the legacy write path -- the outcome of
                ALSO playing the just-written artifact directly here (see
                `_play_utterance_legacy_artifact`), since a hands-free
                utterance has no per-message widget for the app's own
                `TTSCompleteEvent` handler to auto-play through.
            quiet: Task-4 review F5 -- suppress the user-facing
                `TTSCompleteEvent(error=...)` toast this method's own
                `except Exception` branch posts. The failure is still
                logged and `on_finished`/`outcome_code` are unaffected.
                `False` for every existing caller.
        """
        from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram

        normalized_message_id = message_id or "adhoc"
        resolution_source = (
            resolution.source
            if resolution is not None
            else ("explicit_override" if voice is not None else "global")
        )
        start_time = asyncio.get_event_loop().time()
        outcome_code = "generation_failed"
        outcome_reported = False

        def report_outcome(ok: bool) -> None:
            nonlocal outcome_reported
            if outcome_reported or outcome_callback is None:
                return
            outcome_reported = True
            try:
                outcome_callback(ok is True)
            except Exception:
                logger.warning("Console speech outcome callback failed")
        provider_id: str | None = None
        # Task-4 review N3: the resolved provider speed, when available --
        # folded into the legacy completion poll's timeout estimate
        # (`_legacy_playback_timeout_seconds`) so a slower-than-1x
        # configuration does not silently under-estimate playback duration.
        effective_speed: float = 1.0
        response = None
        artifact_path: Path | None = None

        def authorize_destination(
            admitted_provider_id: str,
            admitted_endpoint: str,
        ) -> bool:
            if expected_destination_fingerprint is None:
                return True
            try:
                endpoint = normalize_openai_compatible_endpoint(
                    admitted_endpoint
                )
                admitted_fingerprint = (
                    "sha256:"
                    f"{openai_destination_fingerprint(admitted_provider_id, endpoint)}"
                )
            except Exception:
                return False
            return admitted_fingerprint == expected_destination_fingerprint

        admission_authorizer = (
            authorize_destination
            if expected_destination_fingerprint is not None
            else None
        )

        try:
            service = self._tts_service
            if service is None:
                raise TTSProviderUnavailableError("TTS service is unavailable")
            if expected_destination_fingerprint is not None:
                if resolution is None:
                    raise _TTSAutomaticDestinationChangedError
                destination = await self._destination_for_resolution(resolution)
                if (
                    destination is None
                    or destination.fingerprint
                    != expected_destination_fingerprint
                ):
                    raise _TTSAutomaticDestinationChangedError

            exact_request = (
                resolution.request
                if resolution is not None
                and resolution.source in ("assigned", "default_profile")
                else None
            )
            if exact_request is not None:
                provider_id = exact_request.provider_id
            else:
                try:
                    preferences = service.preferences_snapshot()
                    candidate_provider_id = getattr(
                        preferences,
                        "provider_id",
                        None,
                    )
                    if isinstance(candidate_provider_id, str) and candidate_provider_id:
                        provider_id = candidate_provider_id
                    candidate_speed = getattr(preferences, "speed", None)
                    if (
                        isinstance(candidate_speed, (int, float))
                        and not isinstance(candidate_speed, bool)
                        and candidate_speed > 0
                    ):
                        effective_speed = float(candidate_speed)
                except Exception:
                    logger.debug("TTS metric provider snapshot is unavailable")

            await self._post_tts_message(
                TTSProgressEvent(
                    message_id=normalized_message_id,
                    progress=0.0,
                    status="Initializing TTS generation",
                )
            )

            current_progress = 0.0

            async def progress_sink(progress: TTSProgress) -> None:
                nonlocal current_progress
                fraction = progress.fraction
                if fraction is None and progress.processed is not None:
                    if progress.total is not None and progress.total > 0:
                        fraction = progress.processed / progress.total
                if isinstance(fraction, (int, float)):
                    current_progress = max(
                        current_progress,
                        min(0.9, max(0.0, float(fraction))),
                    )
                await self._post_tts_message(
                    TTSProgressEvent(
                        message_id=normalized_message_id,
                        progress=current_progress,
                        status="Generating audio",
                    )
                )

            primary_error: BaseException | None = None
            try:
                if exact_request is not None:
                    assert resolution is not None
                    assert resolution.repository_generation is not None
                    assert resolution.profile_revision is not None
                    exact_selection = TTSSelectionOverrides(
                        provider_id=exact_request.provider_id,
                        model_mode="exact",
                        model_id=exact_request.model_id,
                        voice_mode=(
                            "server_default" if exact_request.voice is None else "exact"
                        ),
                        voice_id=exact_request.voice,
                        response_format=exact_request.response_format,
                        speed=exact_request.speed,
                        provider_options=exact_request.options,
                    )
                    authorization_kwargs = (
                        {"admission_authorizer": admission_authorizer}
                        if admission_authorizer is not None
                        else {}
                    )
                    if resolution.source == "assigned":
                        (
                            response,
                            effective_selection,
                        ) = await service.synthesize_effective(
                            text=text,
                            character_profile=TTSCharacterProfileSelection(
                                selection=exact_selection,
                                repository_generation=(
                                    resolution.repository_generation
                                ),
                                profile_revision=resolution.profile_revision,
                                profile_id=resolution.profile_id,
                                reference=resolution.reference,
                            ),
                            progress_sink=progress_sink,
                            **authorization_kwargs,
                        )
                    else:
                        assert resolution.source == "default_profile"
                        (
                            response,
                            effective_selection,
                        ) = await service.synthesize_effective(
                            text=text,
                            default_profile=TTSDefaultProfileSelection(
                                selection=exact_selection,
                                repository_generation=(
                                    resolution.repository_generation
                                ),
                                profile_revision=resolution.profile_revision,
                                profile_id=resolution.profile_id,
                                reference=resolution.reference,
                            ),
                            progress_sink=progress_sink,
                            **authorization_kwargs,
                        )
                    requested_selection = TTSRequestedSelectionSnapshot(
                        provider_id=effective_selection.provider_id,
                        model_id=effective_selection.model_id,
                        voice_id=effective_selection.voice_id,
                        response_format=effective_selection.response_format,
                        speed=effective_selection.speed,
                        options=effective_selection.provider_options,
                        configuration_revision=(
                            effective_selection.revisions.provider_configuration
                        ),
                    )
                    self._validate_exact_selection(
                        exact_request,
                        requested_selection,
                    )
                else:
                    authorization_kwargs = (
                        {"admission_authorizer": admission_authorizer}
                        if admission_authorizer is not None
                        else {}
                    )
                    response = await service.synthesize_default(
                        text=text,
                        voice_override=voice,
                        progress_sink=progress_sink,
                        **authorization_kwargs,
                    )
                if (
                    not isinstance(response.provider_id, str)
                    or not response.provider_id
                ):
                    raise _TTSResponseContractError
                if (
                    exact_request is not None
                    and response.provider_id != exact_request.provider_id
                ):
                    raise _TTSResponseContractError
                provider_id = response.provider_id
                if not isinstance(response.model_id, str) or not response.model_id:
                    raise _TTSResponseContractError
                if (
                    exact_request is not None
                    and response.model_id != exact_request.model_id
                ):
                    raise _TTSResponseContractError
                audio_format = self._response_audio_format(response.audio_format)
                if (
                    exact_request is not None
                    and audio_format != exact_request.response_format
                ):
                    raise _TTSResponseContractError

                # --- streaming PCM sink seam (task-4) ------------------------
                # Raw PCM has no container-declared length, so `sink_plan`
                # can decide eligibility from the response's own metadata
                # alone (`response.sample_rate`) -- zero bytes read, zero
                # timing impact either way. This is the first point that
                # metadata AND the validated format are both in hand, and it
                # sits strictly BEFORE any file-writing begins below
                # (`_create_tts_artifact`), so an eligible pcm response never
                # touches disk at all. Decides purely from the RESPONSE's own
                # shape (format/rate), never from provider identity (Global
                # Constraints) -- applies uniformly to every `_generate_tts`
                # caller, not just Console spoken feedback specifically.
                # Falls through unchanged whenever `sink_available()` is
                # False or the format isn't "pcm".
                #
                # WAV is handled differently (see below, AFTER the write
                # loop): `sink_plan` can only validate a WAV body's
                # structure -- and thus decide eligibility -- against the
                # COMPLETE body (see its own docstring: "a streaming WAV
                # source would need to buffer the whole body before calling
                # this"). Deciding here, before writing, would mean fully
                # draining `response.byte_stream` up front for EVERY wav
                # response whenever a sink is merely available -- changing
                # cancellation/mid-stream-failure/write-batching timing for
                # every wav caller regardless of the eventual eligibility
                # verdict, which is exactly what broke
                # `test_console_audio_cpp_native.py`'s cancellation and
                # partial-artifact pins in an earlier version of this seam
                # (verified: reverting to the pre-file-write WAV probe
                # reproduces all four failures). The legacy write loop below
                # is therefore left completely untouched for wav; eligibility
                # is decided from the bytes it already wrote, afterward.
                eligible_pcm_plan: SinkPlan | None = None
                if sink_available() and audio_format == "pcm":
                    eligible_pcm_plan = sink_plan("pcm", response.sample_rate, None)

                if eligible_pcm_plan is not None:
                    if playback_lifecycle is not None and not playback_lifecycle.is_current():
                        outcome_code = "superseded"
                        return
                    streamed_outcome_code = await self._stream_response_via_sink(
                        eligible_pcm_plan,
                        response.byte_stream,
                        message_id=normalized_message_id,
                        on_finished=on_finished,
                        playback_lifecycle=playback_lifecycle,
                    )
                    if streamed_outcome_code is not None:
                        outcome_code = streamed_outcome_code
                        return
                    # `None`: the sink failed to open (a device-level
                    # failure, not a response-shape one) -- fall through to
                    # the unmodified legacy write/play path below for THIS
                    # utterance. `outcome_code` is untouched here -- the
                    # legacy path's own outcome (below) still decides it.

                def remember_cancelled_creation(path: Path) -> None:
                    nonlocal artifact_path
                    artifact_path = path

                def cleanup_late_creation(path: Path) -> None:
                    self._schedule_cancelled_artifact_cleanup(
                        normalized_message_id,
                        path,
                    )

                created_artifact_path = await self._run_blocking_tts_io(
                    lambda: self._create_tts_artifact(audio_format),
                    on_cancelled_result=remember_cancelled_creation,
                    on_late_cancelled_result=cleanup_late_creation,
                )
                artifact_path = created_artifact_path
                buffered_chunks: list[bytes] = []
                buffered_bytes = 0
                # Collected alongside `buffered_chunks` (which gets cleared
                # on every flush) ONLY for a wav response a sink could
                # actually consume -- see the streaming-seam comment above
                # `_create_tts_artifact` for why WAV eligibility is decided
                # from these, AFTER the unmodified write loop below, rather
                # than before it. Fix-round F3: gated on `sink_available()`
                # too, not format alone -- `buffered_chunks` is already
                # flushed and cleared in `_TTS_ARTIFACT_WRITE_BATCH_BYTES`
                # batches, keeping peak memory bounded; `wav_collect`
                # retains the WHOLE body for the lifetime of the call, so
                # holding one when no sink could ever use it (the exact
                # case the Global Constraint says must be left alone) would
                # regress that bound for every wav response on a machine
                # with no `sounddevice` at all -- e.g. a headless/CI box.
                #
                # Fix-round F4: a `bytearray` accumulator, not a
                # `list[bytes]` later joined with `b"".join(...)` -- the
                # list-then-join shape briefly (and, since `wav_collect`
                # itself was never dereferenced afterward, not so briefly)
                # held BOTH the complete list of every streamed chunk AND
                # the freshly `b"".join`-ed copy of the same bytes at once.
                # `bytearray.extend()` grows one buffer in place instead;
                # each `chunk` is copied in and then immediately eligible
                # for GC (nothing else retains it beyond `buffered_chunks`,
                # which is cleared every `_TTS_ARTIFACT_WRITE_BATCH_BYTES`).
                wav_collect: bytearray | None = (
                    bytearray() if _wants_wav_collection(audio_format) else None
                )

                def cleanup_late_write() -> None:
                    self._schedule_cancelled_artifact_cleanup(
                        normalized_message_id,
                        created_artifact_path,
                    )

                async def flush_artifact_batch() -> None:
                    nonlocal buffered_bytes
                    if not buffered_chunks:
                        return
                    batch = b"".join(buffered_chunks)
                    buffered_chunks.clear()
                    buffered_bytes = 0
                    await self._run_blocking_tts_io(
                        partial(
                            self._append_tts_artifact_chunk,
                            created_artifact_path,
                            batch,
                        ),
                        on_late_completion=cleanup_late_write,
                    )

                async for chunk in response.byte_stream:
                    if not chunk:
                        continue
                    if wav_collect is not None:
                        wav_collect.extend(chunk)
                        if len(wav_collect) > _MAX_WAV_SINK_UPGRADE_BYTES:
                            # F4: oversized body -- abandon the in-memory
                            # upgrade attempt (and free what was collected
                            # so far) rather than keep accumulating a body
                            # too big to ever fully fit the sink's own
                            # buffer anyway. The legacy write loop below is
                            # completely unaffected: it already has (or
                            # will have) the complete file on disk.
                            wav_collect = None
                    buffered_chunks.append(chunk)
                    buffered_bytes += len(chunk)
                    if buffered_bytes >= _TTS_ARTIFACT_WRITE_BATCH_BYTES:
                        await flush_artifact_batch()
                await flush_artifact_batch()

                # --- streaming PCM sink seam (task-4), WAV half ------------
                # The response has now been written to `created_artifact_path`
                # in full, byte-identically to how it always was (see the
                # comment above `_create_tts_artifact`). ONLY now -- from the
                # bytes already collected, no extra reads -- decide whether
                # it was ALSO sink-eligible; if so, play it live through the
                # sink and delete the now-redundant artifact rather than
                # exposing it for the legacy file-based play path (avoiding
                # the double-audio failure mode this seam exists to avoid).
                # A sink failure of ANY kind here (open OR mid-stream) simply
                # abandons the sink attempt and falls through to the normal
                # completion below -- unlike the pcm branch above, there is a
                # complete, valid, already-written file to fall back to, so
                # there is no reason to ever surface a user-facing failure
                # for this opportunistic upgrade not panning out.
                if wav_collect and sink_available():
                    # The one copy `sink_plan`/`validate_pcm16_wav` actually
                    # demands: both are typed (and, `validate_pcm16_wav`'s
                    # `struct.unpack_from`/slicing aside, documented) around
                    # an immutable `bytes` body, not a still-growable
                    # `bytearray` -- converting here, once, at that boundary
                    # (fix-round F4) is the single unavoidable copy; dropping
                    # `wav_collect` immediately after frees the accumulator
                    # instead of holding both for the rest of this call.
                    wav_body = bytes(wav_collect)
                    wav_collect = None
                    wav_plan = sink_plan("wav", None, wav_body)
                    if wav_plan is not None:
                        if playback_lifecycle is not None and not playback_lifecycle.is_current():
                            outcome_code = "superseded"
                            await self._discard_tts_artifact(
                                normalized_message_id,
                                created_artifact_path,
                            )
                            artifact_path = None
                            return
                        streamed_outcome_code = await self._stream_response_via_sink(
                            wav_plan,
                            _replay_drained_bytes(wav_body),
                            message_id=normalized_message_id,
                            fallback_on_failure=True,
                            on_finished=on_finished,
                            playback_lifecycle=playback_lifecycle,
                        )
                        if streamed_outcome_code is not None:
                            outcome_code = streamed_outcome_code
                            # Same retry-tracked cleanup a failed/cancelled
                            # generation already uses below (`_artifact_
                            # cleanup_retry`) -- this artifact was never
                            # registered into `_audio_files` (that only
                            # happens further down, which this `return`
                            # skips), so the cache-eviction half of this
                            # call is a harmless no-op; the retry bookkeeping
                            # half is what matters for a stalled delete.
                            await self._discard_tts_artifact(
                                normalized_message_id,
                                created_artifact_path,
                            )
                            artifact_path = None
                            return
                        # `None`: abandon the sink attempt (open OR
                        # mid-stream failure) -- fall through to the normal
                        # completion below, reporting the file already
                        # written above exactly as if this branch had never
                        # run.
            except BaseException as error:
                primary_error = error
                raise
            finally:
                if response is not None:
                    try:
                        await response.aclose()
                    except BaseException:
                        if primary_error is None:
                            raise
                        logger.warning(
                            "TTS response close failed after {}",
                            type(primary_error).__name__,
                        )

            assert artifact_path is not None
            if playback_lifecycle is not None and not playback_lifecycle.is_current():
                await self._discard_tts_artifact(
                    normalized_message_id,
                    artifact_path,
                )
                artifact_path = None
                outcome_code = "superseded"
                return
            await self._cache_audio_file(
                normalized_message_id,
                artifact_path,
                playback_lifecycle,
            )

            await self._post_tts_message(
                TTSProgressEvent(
                    message_id=normalized_message_id,
                    progress=1.0,
                    status="Audio generation complete",
                )
            )
            completion_accepted = await self._post_tts_message(
                TTSCompleteEvent(
                    message_id=normalized_message_id,
                    # Task-4 review F1: `None`, not `artifact_path`, when a
                    # hands-free caller is about to play this artifact
                    # directly below -- exactly the rule the streaming-sink
                    # branch already follows ("so nothing downstream
                    # auto-plays a file that was already played live", see
                    # `_stream_response_via_sink`'s own docstring).
                    # Advertising the real path here let the app's
                    # `TTSCompleteEvent` handler (`app.py`'s
                    # `handle_tts_complete_event`) auto-play it a SECOND
                    # time: no `ChatMessage`/`ChatMessageEnhanced` widget
                    # ever claims a `handsfree-<uuid4>` id, so that handler
                    # always took its own auto-play branch on top of the
                    # direct play below -- a real double voice on every
                    # legacy-path utterance, confirmed by the reviewer's
                    # probe.
                    audio_file=artifact_path if on_finished is None else None,
                    playback_lifecycle=playback_lifecycle,
                )
            )
            if not completion_accepted and on_finished is None:
                if playback_lifecycle is not None:
                    playback_lifecycle.report("failed")
                await self._discard_tts_artifact(
                    normalized_message_id,
                    artifact_path,
                )
                artifact_path = None
            if on_finished is not None:
                # Task-4: a hands-free utterance has no per-message widget
                # for the app's own `TTSCompleteEvent` handler to route an
                # auto-play through (see `_play_utterance_legacy_artifact`'s
                # own docstring) -- play it directly here instead, gated so
                # every OTHER caller (on_finished is always None for them)
                # is completely unaffected.
                await self._play_utterance_legacy_artifact(
                    normalized_message_id,
                    artifact_path,
                    text,
                    on_finished,
                    speed=effective_speed,
                )
            outcome_code = "success" if completion_accepted else "delivery_rejected"
        except asyncio.CancelledError as cancellation:
            outcome_code = (
                "superseded"
                if cancellation_is_success is not None
                and cancellation_is_success()
                else "cancelled"
            )
            if artifact_path is not None:
                cleanup = self._schedule_cancelled_artifact_cleanup(
                    normalized_message_id,
                    artifact_path,
                )
                if cleanup is not None:
                    while not cleanup.done():
                        try:
                            await asyncio.wait({cleanup})
                        except asyncio.CancelledError:
                            continue
            raise cancellation
        except Exception as error:
            destination_changed = isinstance(
                error,
                (
                    _TTSAutomaticDestinationChangedError,
                    TTSConfigurationRevisionError,
                ),
            ) and expected_destination_fingerprint is not None
            outcome_code = (
                "destination_changed"
                if destination_changed
                else self._tts_outcome_code(error)
            )
            await self._discard_tts_artifact(normalized_message_id, artifact_path)
            logger.error(
                "TTS generation failed (outcome_code={})",
                outcome_code,
            )
            if not quiet:
                await self._post_tts_message(
                    TTSCompleteEvent(
                        message_id=normalized_message_id,
                        error=(
                            "The speech destination changed. Confirm Speak replies "
                            "again."
                            if destination_changed
                            else self._tts_error_copy(error)
                        ),
                        playback_lifecycle=playback_lifecycle,
                    )
                )
        finally:
            if playback_lifecycle is not None:
                if outcome_code not in {"success", "interrupted", "superseded"}:
                    playback_lifecycle.report("failed")
                    report_outcome(False)
                elif outcome_code == "superseded":
                    report_outcome(True)
            else:
                report_outcome(outcome_code == "success")
            if provider_id is not None:
                labels = {
                    "provider_id": provider_id,
                    "resolution_source": resolution_source,
                    "outcome_code": outcome_code,
                }
                latency = asyncio.get_event_loop().time() - start_time
                try:
                    log_counter(
                        "tts_generation_total",
                        labels=labels,
                    )
                    log_histogram(
                        "tts_generation_latency_seconds",
                        latency,
                        labels=labels,
                    )
                except Exception:
                    logger.debug("TTS metric publication failed")

    async def _stream_response_via_sink(
        self,
        plan: SinkPlan,
        byte_source: AsyncIterator[bytes],
        *,
        message_id: str,
        fallback_on_failure: bool = False,
        on_finished: Callable[[bool], None] | None = None,
        playback_lifecycle: TTSPlaybackLifecycle | None = None,
    ) -> str | None:
        """Play one response live through the streaming PCM sink.

        Owns the WHOLE streaming branch: silencing any still-playing legacy
        clip, opening the sink, pumping `byte_source` into it, and posting
        the same `TTSProgressEvent`/`TTSCompleteEvent` pair the legacy path
        would have -- just with `audio_file=None` (so nothing downstream
        auto-plays a file that was already played live; see
        `TldwCli`'s `TTSCompleteEvent` handler, which only acts when
        `audio_file` is truthy). Callers must not ALSO run the legacy
        write/play path when this returns non-`None` -- that would be the
        double-audio failure mode this seam exists to avoid.

        Args:
            plan: The eligible `SinkPlan` for this response.
            byte_source: The (unconsumed) audio bytes to feed the sink.
            message_id: The generation's message id, for the events posted.
            fallback_on_failure: `False` (the pcm caller, which reaches this
                BEFORE writing anything -- there is nothing else to fall
                back to) surfaces a mid-stream pump failure as one error
                `TTSCompleteEvent`, same as `open()` failing. `True` (the
                WAV caller, which reaches this AFTER already writing a
                complete, valid artifact) additionally treats a mid-stream
                failure the same as an `open()` failure -- silently
                returning `None` so the caller falls back to the file it
                already has, rather than surfacing a user-facing error for
                an opportunistic upgrade that simply didn't pan out.
            on_finished: Task-4 completion signal, threaded straight through
                from `_generate_tts`'s own `on_finished` parameter (see its
                docstring). Fired at most once, and ONLY for a definite
                terminal outcome (a non-`None` return): `True` for a drain,
                `False` for a stopped/interrupted sink or a streaming
                failure. Never fired for a `None` return (sink open failure,
                or -- only when `fallback_on_failure` is `True` -- a
                mid-stream failure) -- those fall through to the caller's
                own alternate path, which is what ultimately decides (and
                signals) the real terminal outcome.

        Returns:
            The metrics `outcome_code`: `"success"` for a natural drain,
            `"interrupted"` for a deliberate stop (e.g. one-voice
            displacement or a later barge-in -- not an error, but distinct
            from a completed drain), or `"streaming_failed"` (only when
            `fallback_on_failure` is `False`). `None` when the sink failed
            to OPEN, or -- only when `fallback_on_failure` is `True` -- also
            when it failed mid-stream: either way, the caller must fall
            through to its own alternate path, so this method posts nothing
            and touches no bookkeeping in that case.

        Note:
            The response/provider lease this generation is holding (see
            `_generate_tts`'s `response.aclose()`, in its own `finally`)
            stays held for the FULL DURATION of `pump()` below -- i.e.
            through actual playback, not just until the bytes were
            received -- since `_generate_tts` cannot release it until this
            method returns. Fix-round F8: worth noting explicitly as a
            trade-off, not a bug -- pre-Task-4 the lease was released right
            after the legacy write loop finished receiving bytes, well
            before the file was ever played. With
            `TTSService`'s 4-concurrent-operation cap this is not a
            practical blocker in Console's single-utterance-at-a-time
            usage, but it does mean a provider reconfiguration (or another
            concurrent request) can now be delayed by up to one utterance's
            playback length.
        """
        if playback_lifecycle is not None and not playback_lifecycle.is_current():
            return "superseded"
        await self._stop_prior_legacy_clip()
        last_underrun_frames = 0
        event_loop = asyncio.get_running_loop()
        playback_started = threading.Event()

        def _on_event(event: object) -> None:
            nonlocal last_underrun_frames
            if isinstance(event, SinkUnderrun):
                # Fix-round M2: `SinkUnderrun` is the only signal that live
                # playback is stuttering -- `_post_sink_event` alone drops
                # it into a debug-only log line, invisible to both the
                # user and metrics on the one feature whose entire premise
                # is live playback quality. Recorded here so it can be
                # reported once, at utterance end (below), rather than per
                # throttled event. `on_event` may be invoked concurrently
                # from multiple threads (Audio/streaming_sink.py's thread
                # contract), but `SinkUnderrun` specifically always
                # originates from the sink's own notify thread, never a
                # caller thread -- so this plain assignment has exactly
                # one writer; the read below happens once, after `pump()`
                # has already returned, from this coroutine's own thread.
                last_underrun_frames = event.frames
            if playback_lifecycle is not None and isinstance(event, SinkStarted):
                playback_started.set()
                event_loop.call_soon_threadsafe(
                    playback_lifecycle.report,
                    "playing",
                )
            self._post_sink_event(event)

        sink = StreamingPcmSink(on_event=_on_event)
        # Fix-round F2: `open()` is documented thread-safe and never
        # raises, but it is NOT free -- lazily importing `sounddevice` on
        # first use plus building/starting the real `OutputStream` measured
        # ~65-110ms on a quiet machine, and CoreAudio device-open latency
        # can be much worse when another process holds the device. Every
        # OTHER blocking call in this class already goes through this same
        # `_run_blocking_tts_io` offload seam (`>100ms -> worker`); `open()`
        # is no exception, and reusing the seam (rather than a bare
        # `asyncio.to_thread`) gets its cancellation handling for free: if
        # this coroutine is cancelled while `open()` is still running on
        # its own thread (which cannot itself be interrupted), the
        # `on_cancelled_result`/`on_late_cancelled_result` hooks below
        # still guarantee `sink.stop()` runs once `open()` actually
        # finishes -- otherwise a sink that reached "open" only AFTER the
        # cancellation had already unwound this call stack would never
        # reach a terminal state at all (the carried terminal-call
        # guarantee). `sink.stop()` is a safe, idempotent no-op if `open()`
        # instead reached "failed" on its own.
        await self._run_blocking_tts_io(
            lambda: sink.open(plan.sample_rate, plan.channels),
            on_cancelled_result=lambda _: sink.stop(),
            on_late_cancelled_result=lambda _: sink.stop(),
        )
        if sink.state == "failed":
            return None
        if playback_lifecycle is not None:
            self._active_stream_playback_owner = playback_lifecycle

        try:
            result = await pump(
                sink,
                byte_source,
                skip_bytes=plan.skip_bytes,
                max_bytes=plan.data_bytes,
            )
        finally:
            if self._active_stream_playback_owner is playback_lifecycle:
                self._active_stream_playback_owner = None

        if (
            playback_lifecycle is not None
            and playback_started.is_set()
            and playback_lifecycle.state == "generating"
        ):
            playback_lifecycle.report("playing")

        # Fix-round M2: report on utterance end, regardless of terminal
        # outcome -- an underrun can happen on the way to ANY of them, not
        # just a successful drain. Minimal and honest: one INFO log line
        # with the cumulative frame count, plus one bump of the existing
        # `tts_generation_total` counter with a dedicated `"underrun"`
        # `outcome_code` value (same pattern F8 already established for
        # `"interrupted"`) -- no UI.
        if last_underrun_frames > 0:
            logger.info(
                "Streaming TTS playback underrun ({} frames) for message {}",
                last_underrun_frames,
                message_id,
            )
            try:
                from tldw_chatbook.Metrics.metrics_logger import log_counter

                log_counter(
                    "tts_generation_total",
                    labels={"outcome_code": "underrun"},
                )
            except Exception:
                logger.debug("TTS underrun metric publication failed")

        # Fix-round F8: the `outcome_code` values returned below
        # ("success"/"interrupted"/"streaming_failed") are a DELIBERATE
        # expansion of `_generate_tts`'s metric label set, not covered by
        # `_tts_outcome_code`'s bounded exception-type mapping (that
        # function only ever runs for the legacy path's `except Exception`
        # branch) -- listed together here as the closed set this method
        # can produce, for anyone auditing metric label cardinality.
        if result.outcome in ("drained", "stopped"):
            await self._post_tts_message(
                TTSProgressEvent(
                    message_id=message_id,
                    progress=1.0,
                    status="Audio generation complete",
                )
            )
            completion_accepted = await self._post_tts_message(
                TTSCompleteEvent(
                    message_id=message_id,
                    audio_file=None,
                    playback_lifecycle=playback_lifecycle,
                )
            )
            if not completion_accepted:
                if on_finished is not None:
                    on_finished(False)
                return "delivery_rejected"
            if result.outcome == "drained":
                if playback_lifecycle is not None:
                    if playback_lifecycle.state == "playing":
                        playback_lifecycle.report_terminal("stopped")
                    elif playback_lifecycle.state == "generating":
                        playback_lifecycle.report_terminal("failed")
                if on_finished is not None:
                    on_finished(True)
                return "success"
            # "stopped": the sink was interrupted -- one-voice displacement
            # by a later utterance, an explicit stop, or a barge-in -- not
            # a natural drain. Still posted as a normal completion (no
            # error: an interruption is not a failure, and there is
            # nothing more useful to tell the user), but fix-round F8:
            # labeled distinctly in the metric rather than folded into
            # "success", which would misrepresent playback that was cut
            # short as if it had played out in full.
            if playback_lifecycle is not None:
                playback_lifecycle.report_terminal("stopped")
            if on_finished is not None:
                on_finished(False)
            return "interrupted"

        logger.warning(
            "Streaming TTS playback failed (outcome={}, reason={}, "
            "fallback_on_failure={})",
            result.outcome,
            result.reason,
            fallback_on_failure,
        )
        if fallback_on_failure:
            return None
        if playback_lifecycle is not None:
            playback_lifecycle.report_terminal("failed")
        await self._post_tts_message(
            TTSCompleteEvent(
                message_id=message_id,
                error="TTS playback failed; retry",
                playback_lifecycle=playback_lifecycle,
            )
        )
        if on_finished is not None:
            on_finished(False)
        return "streaming_failed"

    async def _stop_prior_legacy_clip(self, *, bare_stop: bool = False) -> bool:
        """Silence any currently-playing legacy file clip before streaming.

        The streaming sink plays through its own `sounddevice.OutputStream`,
        entirely independent of the legacy `SimpleAudioPlayer` singleton --
        opening a new sink does nothing, on its own, to stop a still-playing
        legacy clip the way the sink registry's one-voice displacement
        already stops a PRIOR sink (see `streaming_sink._register_live_sink`).
        Mirrors `handle_tts_playback`'s message-scoped stop branch (the same
        `_last_played`/`stop_audio_playback_if_current` pair), called
        directly here rather than round-tripping through a posted
        `TTSPlaybackEvent` since generation already runs on its own worker.

        Task-4 review round 2 (F3+N2)/round 3 (F3, D1), replacing earlier
        fixes that did not close the window they targeted -- see git
        history for the analysis of each. `bare_stop` is `True` for
        exactly one caller, `handle_tts_playback`'s bare/global-stop
        branch. When it is true AND `_legacy_handoff_stop_events` is
        non-empty (one or more `_play_utterance_legacy_artifact` calls
        have a play-and-poll in progress -- see `__init__`'s docstring for
        why this is a SET, not a single flag), this:

        1. Sets EVERY in-flight handoff's own `threading.Event` -- each
           worker thread (`_play_legacy_clip_and_await_completion`) checks
           its own event immediately after `player.play()` returns (round
           3's F3 fix: a same-tick `player.stop()` call issued BEFORE
           `Popen` has no process to kill, and `play()` proceeds past its
           own internal `self.stop()` regardless -- only the worker, which
           alone knows when `Popen` actually happened, can catch that) and
           on each subsequent poll iteration.
        2. ALSO stops the shared player directly, unconditionally, right
           here -- the identity check `stop_audio_playback_if_current`
           normally uses cannot succeed yet in the pre-`Popen` window (the
           player does not own the file until `play()` returns), so the
           tracked branch below would silently no-op for the whole handoff
           window otherwise; this direct call is what makes a POST-`Popen`
           barge-in prompt (~0ms) rather than waiting for the worker's own
           next poll tick. `SimpleAudioPlayer.stop()` is a cheap, safe
           no-op when nothing is actually loaded/playing.

        Deliberately NOT applied when `bare_stop` is false (the OTHER
        caller, `_stream_response_via_sink`, on the shared path EVERY TTS
        caller uses to silence a legacy clip before opening a new sink) --
        an ordinary, non-hands-free utterance silencing unrelated audio
        elsewhere in the app (watchlists, the STTS playground, ...) via the
        SAME process-global player singleton is a real scope regression
        (task-4 review N2), not a fix. That call site keeps the original,
        purely-tracked behavior: a deliberate no-op when nothing is
        tracked, exactly like `stop_audio_playback_if_current`'s own
        message-scoped-stop sibling documents ("stopping message A must
        never silence a different, still-playing message B").
        """
        if bare_stop and self._legacy_handoff_stop_events:
            for stop_requested in list(self._legacy_handoff_stop_events):
                stop_requested.set()

            from tldw_chatbook.TTS.audio_player import get_audio_player

            # Task 5 (D7, carried from task-4-review.md): `SimpleAudioPlayer.
            # stop()` takes `_lock` and can hold it for up to ~2.5s waiting
            # out an unresponsive player process (see `stop()`'s own
            # terminate/kill/wait sequence) -- calling it synchronously here
            # would block the WHOLE event loop for that long, matching the
            # `await asyncio.to_thread(player.stop)` idiom this same file
            # already uses 3 lines away in `_play_utterance_legacy_artifact`
            # for the identical call.
            await asyncio.to_thread(get_audio_player().stop)
            async with self._audio_files_lock:
                self._last_played = None
            return True

        async with self._audio_files_lock:
            last_played = self._last_played
            self._last_played = None
        if last_played is not None:
            return stop_audio_playback_if_current(last_played[1])
        return False

    async def _play_utterance_legacy_artifact(
        self,
        message_id: str,
        audio_file: Path,
        text: str,
        on_finished: Callable[[bool], None],
        *,
        speed: float = 1.0,
    ) -> None:
        """Play one just-written legacy artifact for a cooldown-free utterance.

        `speak_utterance` (task-4) has no per-message widget for the app's
        `TTSCompleteEvent` handler to route an auto-play through -- that
        routing (`app.py`'s `handle_tts_complete_event`) only posts a
        `TTSPlaybackEvent(action="play")` when NO `ChatMessage`/
        `ChatMessageEnhanced` widget claims the message id, and even then
        only reaches `handle_tts_playback` after a full Textual message
        round-trip. This plays the artifact directly instead, mirroring
        `handle_tts_playback`'s own "play" action (silence any live sink
        first, so the two independent audio-output paths never overlap
        into a double voice; track `_last_played` so a later stop can
        interrupt it, and schedule the same delayed cleanup) -- but
        additionally waits (bounded, off the event loop) for the clip to
        actually stop being current before reporting completion through
        `on_finished` -- see `_play_legacy_clip_and_await_completion`'s own
        docstring for why (task-4 review F2, the headline finding: firing
        at handoff instead of completion truncated every sentence but the
        last).

        Task-4 review round 2 (F3+N2)/round 3 (F3, D1): a fresh
        `threading.Event` (`stop_requested`) is created for THIS handoff
        and registered in `_legacy_handoff_stop_events` (a set, per-handoff
        -- see `__init__`'s docstring for why a single bool was wrong) for
        the full duration of the play-and-poll call below, so a concurrent
        bare/global stop can reach this specific handoff even before the
        player actually owns the file. Two earlier fixes did not close
        this window (see git history for each): round 1's early `_last_
        played` registration just moved the no-op from one guard to
        another; round 2's handler-side-only `player.stop()` fires but has
        no process to kill in the pre-`Popen` sub-window, so `play()`
        proceeds past its own internal `self.stop()` and starts the clip
        regardless. Round 3 closes it from the WORKER side instead --
        `_play_legacy_clip_and_await_completion` checks `stop_requested`
        immediately after `player.play()` returns (Popen has then
        definitely either happened or failed) and on every poll iteration,
        which is the only vantage point that actually knows when `Popen`
        ran.

        Task-4 review N4 (round 2), D2 guard (round 3): the play-and-poll
        call is offloaded via a bare `asyncio.create_task(asyncio.to_
        thread(...))` + `asyncio.shield` here, NOT the shared `_run_
        blocking_tts_io` seam every other blocking call in this class
        uses. That seam's cancellation handling does a BOUNDED JOIN
        (`_TTS_IO_CANCELLATION_JOIN_TIMEOUT_SECONDS`, up to 1s) before its
        own `on_cancelled_result` hook ever fires -- fine for a quick
        artifact write, but it means cancelling this coroutine would NOT
        promptly silence a clip that could still be playing for many more
        seconds. Reimplemented narrowly here so the `except asyncio.
        CancelledError` below runs IMMEDIATELY when this coroutine is
        cancelled (e.g. via `cleanup_tts_resources()` cancelling the
        `_active_tasks`-registered generation task, task-4 review F4),
        setting `stop_requested` (closing the SAME pre-`Popen` race F3
        closes, regardless of whether cancellation lands before, during,
        or after the worker's own `play()` call) and stopping the player
        directly -- but ONLY when it still owns THIS clip, or owns nothing
        yet (D2: a DIFFERENT clip could have displaced ours in the same
        tick, before the poll's own next iteration would have noticed;
        stopping unconditionally would kill that unrelated clip). The
        abandoned worker is retained in this file's existing `_retained_
        tts_io_tasks` set so it is never garbage-collected mid-flight even
        though nothing awaits it once this coroutine unwinds -- it settles
        on its own once the prompt stop (or `stop_requested`) makes the
        poll observe the state change.

        Args:
            speed: Task-4 review N3 -- the resolved provider speed, folded
                into the completion poll's timeout estimate (see
                `_legacy_playback_timeout_seconds`) so a slower-than-1x
                configuration does not silently under-estimate how long
                the clip could still be playing. `1.0` (unchanged bound)
                when the caller could not determine it.
        """
        from tldw_chatbook.TTS.audio_player import get_audio_player

        stop_live_sink()
        stop_requested = threading.Event()
        self._legacy_handoff_stop_events.add(stop_requested)
        try:
            async with self._audio_files_lock:
                self._last_played = (message_id, audio_file)

            player = get_audio_player()
            timeout_seconds = _legacy_playback_timeout_seconds(len(text), speed)
            worker: asyncio.Task[bool] = asyncio.create_task(
                asyncio.to_thread(
                    _play_legacy_clip_and_await_completion,
                    player,
                    audio_file,
                    timeout_seconds=timeout_seconds,
                    stop_requested=stop_requested,
                )
            )
            ok = False
            try:
                ok = await asyncio.shield(worker)
            except asyncio.CancelledError:
                # Task-4 review N4 (round 2) + D2 (round 3): stop the
                # player IMMEDIATELY, off the event loop -- no bounded
                # join first -- but only when it still owns OUR clip (or
                # owns nothing yet, the pre-`Popen` case) -- D2: cancelling
                # while a DIFFERENT clip has already displaced ours in the
                # same tick must not kill that unrelated clip. Also sets
                # `stop_requested` so the worker's own F3 check catches
                # this regardless of exactly when cancellation landed
                # relative to the worker's own `play()` call.
                stop_requested.set()
                self._retained_tts_io_tasks.add(worker)
                worker.add_done_callback(self._retained_tts_io_tasks.discard)
                # Task 5 (D7): `get_current_file()` takes the SAME `_lock`
                # `stop()` can hold for up to ~2.5s -- calling it
                # synchronously here would block the event loop for that
                # long too, same as the bare-stop branch above.
                current_file = await asyncio.to_thread(player.get_current_file)
                if current_file in (None, audio_file):
                    await asyncio.to_thread(player.stop)
                raise
            except Exception:
                logger.warning("Legacy TTS playback failed for one utterance")
                ok = False
        finally:
            self._legacy_handoff_stop_events.discard(stop_requested)

        self._schedule_legacy_playback_cleanup(message_id)
        on_finished(bool(ok))

    def _schedule_legacy_playback_cleanup(self, message_id: str) -> None:
        """Schedule the delayed artifact cleanup `_play_utterance_legacy_
        artifact` needs, holding a strong reference (task-4 review F7 --
        the event loop only weak-refs tasks with no other strong
        reference) in `_pending_legacy_cleanup_timers`, NOT the file's
        older `_retained_tts_cleanup_tasks` retention idiom (task-4 review
        N5: that set's tasks are AWAITED, boundedly, at shutdown -- correct
        for genuinely in-flight I/O, but this is a pure `asyncio.sleep(5)`
        timer that `cleanup_tts_resources()` has no reason to wait out,
        since it deletes the same artifact directly moments later
        regardless; see `cleanup_tts_resources`'s own cancel-not-await
        handling of this set). Both legacy cleanup call sites bind the
        explicit `None` cache owner so neither can delete a newer
        lifecycle-owned replacement for the same message id.
        """
        cleanup = asyncio.create_task(
            self._cleanup_audio_file(
                message_id,
                delay=5.0,
                artifact_owner=None,
            )
        )
        self._pending_legacy_cleanup_timers.add(cleanup)

        def observe(completed: asyncio.Task) -> None:
            self._pending_legacy_cleanup_timers.discard(completed)
            try:
                completed.result()
            except asyncio.CancelledError:
                # Expected at shutdown: cleanup_tts_resources() cancels
                # this timer outright rather than waiting it out, then
                # deletes the same artifact itself moments later.
                pass
            except BaseException:
                logger.warning("Retained TTS audio cleanup did not complete")

        cleanup.add_done_callback(observe)

    def _post_sink_event(self, event: object) -> None:
        """Record one streaming-sink lifecycle event.

        `StreamingPcmSink.on_event` may be invoked CONCURRENTLY, from
        multiple threads at once, and must never do sink work itself --
        only marshal (see `Audio/streaming_sink.py`'s module docstring,
        "Thread contract"). Nothing in the UI consumes these yet: spoken
        feedback only needs `pump()`'s own synchronously-awaited
        `PumpResult` to know how generation ended, and
        `_stream_response_via_sink` already reports that through the normal
        `TTSProgressEvent`/`TTSCompleteEvent` path. Wiring these granular
        events into the UI (a live underrun indicator, a "speaking" state,
        etc.) is deferred to a later phase that has an actual listener for
        them -- this is intentionally a debug-log recorder, not a
        `post_message`-marshaled Textual event, for now. `logger.debug` is
        safe to call concurrently from multiple threads, satisfying the
        thread-safety requirement without introducing any shared mutable
        state here.
        """
        logger.debug("streaming TTS sink event: {}", type(event).__name__)

    @staticmethod
    def _validate_exact_selection(
        request: TTSRequest,
        selection: object,
    ) -> None:
        """Reject exact-service provenance that differs from the admitted request."""
        if type(selection) is not TTSRequestedSelectionSnapshot:
            raise _TTSResponseContractError
        expected = (
            request.provider_id,
            request.model_id,
            request.voice,
            request.response_format,
            request.speed,
            request.options,
        )
        actual = (
            selection.provider_id,
            selection.model_id,
            selection.voice_id,
            selection.response_format,
            selection.speed,
            selection.options,
        )
        if actual != expected:
            raise _TTSResponseContractError

    async def _run_blocking_tts_io(
        self,
        operation: Callable[[], _T],
        *,
        on_cancelled_result: Callable[[_T], None] | None = None,
        on_late_cancelled_result: Callable[[_T], None] | None = None,
        on_late_completion: Callable[[], None] | None = None,
        operation_timeout_seconds: float | None = None,
    ) -> _T:
        """Run artifact I/O off-loop with a bounded cancellation join."""
        worker = asyncio.create_task(asyncio.to_thread(operation))
        try:
            if operation_timeout_seconds is None:
                return await asyncio.shield(worker)
            await asyncio.wait({worker}, timeout=operation_timeout_seconds)
            if not worker.done():
                self._retain_tts_io_after_cancellation(
                    worker,
                    on_late_cancelled_result=on_late_cancelled_result,
                    on_late_completion=on_late_completion,
                )
                raise _TTSArtifactIOTimeout
            return worker.result()
        except asyncio.CancelledError as cancellation:
            deadline = (
                asyncio.get_running_loop().time()
                + _TTS_IO_CANCELLATION_JOIN_TIMEOUT_SECONDS
            )
            worker_error: BaseException | None = None
            while not worker.done():
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    self._retain_tts_io_after_cancellation(
                        worker,
                        on_late_cancelled_result=on_late_cancelled_result,
                        on_late_completion=on_late_completion,
                    )
                    logger.warning(
                        "TTS artifact I/O exceeded the cancellation join timeout; "
                        "late cleanup was retained"
                    )
                    raise cancellation
                try:
                    await asyncio.wait({worker}, timeout=remaining)
                except asyncio.CancelledError:
                    continue
                except BaseException as error:
                    worker_error = error
                    break
            if worker_error is None:
                try:
                    result = worker.result()
                except BaseException as error:
                    worker_error = error
                else:
                    if on_cancelled_result is not None:
                        on_cancelled_result(result)
            if worker_error is not None:
                logger.warning(
                    "TTS artifact I/O did not complete while cancellation was pending"
                )
            # Outside this already-cancelled branch, process-control
            # BaseExceptions still propagate from the initial await above.
            raise cancellation

    def _retain_tts_io_after_cancellation(
        self,
        worker: asyncio.Task[_T],
        *,
        on_late_cancelled_result: Callable[[_T], None] | None,
        on_late_completion: Callable[[], None] | None,
    ) -> None:
        """Observe one timed-out worker and dispatch its eventual result."""
        self._retained_tts_io_tasks.add(worker)

        def observe(completed: asyncio.Task[_T]) -> None:
            self._retained_tts_io_tasks.discard(completed)
            try:
                try:
                    result = completed.result()
                except BaseException:
                    logger.warning(
                        "Retained TTS artifact I/O did not complete successfully"
                    )
                else:
                    if on_late_cancelled_result is not None:
                        try:
                            on_late_cancelled_result(result)
                        except BaseException:
                            logger.warning(
                                "Late TTS artifact cleanup could not be scheduled"
                            )
            finally:
                if on_late_completion is not None:
                    try:
                        on_late_completion()
                    except BaseException:
                        logger.warning(
                            "Late TTS artifact completion could not be processed"
                        )

        worker.add_done_callback(observe)

    def _schedule_cancelled_artifact_cleanup(
        self,
        message_id: str,
        artifact_path: Path,
    ) -> asyncio.Task[None] | None:
        """Retain cleanup for an artifact exposed after cancellation returned."""
        self._artifact_cleanup_retry.add(artifact_path)
        if artifact_path in self._retained_tts_cleanup_paths:
            self._retained_tts_cleanup_requeue[artifact_path] = message_id
            return None
        self._retained_tts_cleanup_paths.add(artifact_path)
        cleanup = asyncio.create_task(
            self._discard_tts_artifact(message_id, artifact_path)
        )
        self._retained_tts_cleanup_tasks.add(cleanup)

        def observe(completed: asyncio.Task[None]) -> None:
            self._retained_tts_cleanup_tasks.discard(completed)
            self._retained_tts_cleanup_paths.discard(artifact_path)
            requeued_message_id = self._retained_tts_cleanup_requeue.pop(
                artifact_path,
                None,
            )
            try:
                completed.result()
            except BaseException:
                logger.warning("Retained TTS artifact cleanup did not complete")
            if (
                requeued_message_id is not None
                and artifact_path in self._artifact_cleanup_retry
            ):
                self._schedule_cancelled_artifact_cleanup(
                    requeued_message_id,
                    artifact_path,
                )

        cleanup.add_done_callback(observe)
        return cleanup

    def _create_tts_artifact(self, audio_format: str) -> Path:
        """Create one owner-only Console audio artifact."""
        return Path(
            self._temp_manager.create_temp_file(
                content=b"",
                suffix=f".{audio_format}",
                prefix="tts_audio_",
            )
        )

    @staticmethod
    def _append_tts_artifact_chunk(artifact_path: Path, chunk: bytes) -> None:
        """Append and flush one ordered response batch."""
        with artifact_path.open("ab") as audio_file:
            audio_file.write(chunk)
            audio_file.flush()

    @staticmethod
    def _secure_delete_tts_artifact(artifact_path: Path) -> bool:
        """Delete an artifact and treat an already-absent path as complete."""
        deleted = secure_delete_file(artifact_path)
        return deleted is True or not artifact_path.exists()

    async def _try_secure_delete_tts_artifact(
        self,
        artifact_path: Path,
        *,
        on_late_success: Callable[[], None] | None = None,
    ) -> bool:
        """Attempt secure deletion without exposing the artifact path in logs.

        Args:
            artifact_path: Owned artifact to delete.
            on_late_success: Event-loop callback used when a timed-out delete
                eventually succeeds.

        Returns:
            ``True`` when deletion completed within the bounded attempt.
        """

        def observe_late_delete(deleted: bool) -> None:
            if deleted:
                self._artifact_cleanup_retry.discard(artifact_path)
                if on_late_success is not None:
                    on_late_success()

        try:
            deleted = await self._run_blocking_tts_io(
                lambda: self._secure_delete_tts_artifact(artifact_path),
                on_late_cancelled_result=observe_late_delete,
                operation_timeout_seconds=_TTS_SECURE_DELETE_TIMEOUT_SECONDS,
            )
        except Exception:
            deleted = False
        if not deleted:
            logger.warning("Incomplete TTS artifact cleanup will be retried")
        return deleted

    async def _cache_audio_file(
        self,
        message_id: str,
        artifact_path: Path,
        artifact_owner: TTSPlaybackLifecycle | None,
    ) -> None:
        """Publish one path and its exact lifecycle owner as one cache record."""
        async with self._audio_files_lock:
            replaced_path = self._audio_files.get(message_id)
            replaced_owner = self._audio_file_owners.get(message_id)
            self._audio_files[message_id] = artifact_path
            self._audio_file_owners[message_id] = artifact_owner

        if replaced_path is not None and replaced_path != artifact_path:
            await self._discard_tts_artifact(
                message_id,
                replaced_path,
                artifact_owner=replaced_owner,
            )

    async def _release_audio_file_if_current(
        self,
        message_id: str,
        artifact_path: Path,
        artifact_owner: TTSPlaybackLifecycle | None | object,
    ) -> None:
        """Release a cache entry only when it still owns the deleted artifact."""
        async with self._audio_files_lock:
            cached_owner = self._audio_file_owners.get(message_id)
            if (
                self._audio_files.get(message_id) == artifact_path
                and (
                    artifact_owner is _ANY_ARTIFACT_OWNER
                    or cached_owner is artifact_owner
                )
            ):
                del self._audio_files[message_id]
                self._audio_file_owners.pop(message_id, None)

    def _schedule_audio_file_release_if_current(
        self,
        message_id: str,
        artifact_path: Path,
        artifact_owner: TTSPlaybackLifecycle | None | object,
    ) -> None:
        """Track cache bookkeeping triggered by a retained delete worker."""
        release = asyncio.create_task(
            self._release_audio_file_if_current(
                message_id,
                artifact_path,
                artifact_owner,
            )
        )
        self._retained_tts_cleanup_tasks.add(release)

        def observe(completed: asyncio.Task[None]) -> None:
            self._retained_tts_cleanup_tasks.discard(completed)
            try:
                completed.result()
            except BaseException:
                logger.warning("Retained TTS audio cache cleanup did not complete")

        release.add_done_callback(observe)

    async def _drain_retained_tts_artifact_work(self) -> None:
        """Bound shutdown waiting for retained artifact I/O and cleanup."""
        deadline = (
            asyncio.get_running_loop().time() + _TTS_RETAINED_WORK_DRAIN_TIMEOUT_SECONDS
        )

        async def drain(tasks: set[asyncio.Task]) -> None:
            while tasks:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    return
                await asyncio.wait(set(tasks), timeout=remaining)
                await asyncio.sleep(0)

        await drain(self._retained_tts_io_tasks)
        await drain(self._retained_tts_cleanup_tasks)

    async def _discard_tts_artifact(
        self,
        message_id: str,
        artifact_path: Path | None,
        *,
        artifact_owner: TTSPlaybackLifecycle | None | object = _ANY_ARTIFACT_OWNER,
    ) -> None:
        """Remove one failed or cancelled artifact from cache and disk."""
        if artifact_path is None:
            return
        async with self._audio_files_lock:
            cached_path = self._audio_files.get(message_id)
            cached_owner = self._audio_file_owners.get(message_id)
            if (
                cached_path == artifact_path
                and artifact_owner is not _ANY_ARTIFACT_OWNER
                and cached_owner is not artifact_owner
            ):
                return
            if cached_path == artifact_path:
                del self._audio_files[message_id]
                self._audio_file_owners.pop(message_id, None)
            self._artifact_cleanup_retry.add(artifact_path)

        if await self._try_secure_delete_tts_artifact(artifact_path):
            async with self._audio_files_lock:
                self._artifact_cleanup_retry.discard(artifact_path)

    async def discard_stale_console_completion(
        self,
        message_id: str,
        artifact_path: Path | None,
        lifecycle: TTSPlaybackLifecycle,
    ) -> None:
        """Discard a completion that lost Console playback ownership."""
        await self._discard_tts_artifact(
            message_id,
            artifact_path,
            artifact_owner=lifecycle,
        )
        lifecycle.report_terminal("stopped")

    @staticmethod
    def _response_audio_format(audio_format: object) -> str:
        """Return one safe canonical extension from response-owned metadata."""
        if not isinstance(audio_format, str):
            raise _TTSResponseContractError
        normalized = audio_format.lower().strip().removeprefix(".")
        if normalized not in {"mp3", "opus", "aac", "flac", "wav", "pcm"}:
            raise _TTSResponseContractError
        return normalized

    @staticmethod
    def _tts_outcome_code(error: Exception) -> str:
        """Map failures to bounded metric outcome codes."""
        if isinstance(error, TTSProviderReconfiguringError):
            return "reconfiguring"
        if isinstance(error, TTSProviderUnavailableError):
            return "unavailable"
        if isinstance(error, TTSConfigurationRevisionError):
            return "revision_mismatch"
        if isinstance(error, TTSRegistryClosedError):
            return "unavailable"
        if isinstance(error, TTSOperationError):
            return error.code
        if isinstance(error, _TTSResponseContractError):
            return "audio_response_invalid"
        if isinstance(error, UnknownLegacyModelError):
            # An id the compatibility bridge cannot route is a model
            # configuration problem; the generic bucket's "retry" framing
            # hid that for weeks of TASK-15420's window (TASK-15422).
            return "model_invalid"
        if isinstance(error, TTSBackendConnectionError):
            # Reachability, not configuration — this must precede the
            # generic ValueError bucket it subclasses (TASK-15530).
            return "connection_unavailable"
        if isinstance(error, ValueError):
            return "configuration_invalid"
        return "generation_failed"

    @staticmethod
    def _tts_error_copy(error: Exception) -> str:
        """Map failures to fixed actionable UI copy without upstream details."""
        if isinstance(error, TTSProviderReconfiguringError):
            return "TTS settings are being applied; retry shortly"
        if isinstance(error, TTSProviderUnavailableError):
            return "TTS is unavailable; check STTS Settings and Retry/Reconnect"
        if isinstance(error, TTSConfigurationRevisionError):
            return "TTS settings changed before speech started; retry"
        if isinstance(error, TTSRegistryClosedError):
            return "The TTS service is unavailable"
        if isinstance(error, TTSOperationError):
            if error.code in {"configuration_invalid", "not_configured"}:
                return "TTS is not configured; open STTS Settings"
            if error.code == "contract_incompatible":
                return "The configured TTS service is incompatible"
            if error.code == "server_busy":
                return "The TTS service is busy; retry shortly"
            if error.code == "generation_timeout":
                return "TTS generation timed out; retry"
            if error.code == "audio_response_invalid":
                return (
                    "The TTS service returned invalid audio; "
                    "check provider compatibility"
                )
            if error.code == "cleanup_failed":
                return "TTS cleanup did not complete; restart Chatbook before retrying"
            return "TTS generation failed; retry"
        if isinstance(error, _TTSResponseContractError):
            return (
                "The TTS service returned invalid audio; check provider compatibility"
            )
        if isinstance(error, UnknownLegacyModelError):
            return (
                "The selected TTS model is not available for this provider; "
                "check the model in STTS Settings"
            )
        if isinstance(error, TTSBackendConnectionError):
            return (
                "Unable to reach the TTS server; check that it is running "
                "and the Base URL in STTS Settings"
            )
        if isinstance(error, ValueError):
            return "TTS is not configured; open STTS Settings"
        return "Unexpected TTS generation failure; retry"

    async def _run_owned_file_playback(
        self,
        message_id: str,
        audio_file: Path,
        lifecycle: TTSPlaybackLifecycle,
        stop_requested: threading.Event,
        playback_started: threading.Event,
    ) -> None:
        """Start and monitor one Console clip from the real player state."""
        from tldw_chatbook.TTS.audio_player import get_audio_player

        loop = asyncio.get_running_loop()

        def report_started() -> None:
            playback_started.set()
            loop.call_soon_threadsafe(lifecycle.report, "playing")

        try:
            finished = await asyncio.to_thread(
                _play_legacy_clip_and_await_completion,
                get_audio_player(),
                audio_file,
                timeout_seconds=_LEGACY_PLAYBACK_POLL_MAX_SECONDS,
                stop_requested=stop_requested,
                on_started=report_started,
            )
            await asyncio.sleep(0)
            if stop_requested.is_set():
                lifecycle.report_terminal("stopped")
            elif finished:
                lifecycle.report_terminal("stopped")
            else:
                lifecycle.report_terminal("failed")
        except asyncio.CancelledError:
            stop_requested.set()
            lifecycle.report_terminal("stopped")
            raise
        except Exception:
            lifecycle.report_terminal("failed")
        finally:
            current_task = asyncio.current_task()
            if self._active_file_playback_task is current_task:
                self._active_file_playback_task = None
            if self._active_file_playback_owner is lifecycle:
                self._active_file_playback_owner = None
            if self._active_file_playback_stop == (message_id, stop_requested):
                self._active_file_playback_stop = None
            if self._active_file_playback_started is playback_started:
                self._active_file_playback_started = None
            async with self._audio_files_lock:
                if self._last_played == (message_id, audio_file):
                    self._last_played = None
            await self._cleanup_audio_file(
                message_id,
                artifact_owner=lifecycle,
            )

    async def handle_tts_playback(self, event: TTSPlaybackEvent) -> None:
        """Handle TTS playback control"""
        logger.info(
            f"TTS playback action: {event.action} for message {event.message_id}"
        )

        stream_stop_accepted = False
        generation_stop_accepted = False
        file_stop_accepted = False
        file_stop_retryable_failure = False
        if event.action == "stop":
            generation_owner = self._console_generation_owner
            generation_stop_accepted = await self._cancel_console_generation(
                message_id=event.message_id,
                lifecycle=event.playback_lifecycle,
                superseded=True,
            )
            if generation_stop_accepted and generation_owner is not None:
                generation_owner.lifecycle.report_terminal("stopped")

            stream_owner = self._active_stream_playback_owner
            bare_stop = not event.message_id
            stream_owner_matches = bool(
                stream_owner is not None
                and (
                    bare_stop
                    or (
                        stream_owner.message_id == event.message_id
                        and stream_owner is event.playback_lifecycle
                    )
                )
            )
            if bare_stop or stream_owner_matches:
                stop_live_sink()
            if stream_owner_matches and stream_owner is not None:
                stream_stop_accepted = True
                if self._active_stream_playback_owner is stream_owner:
                    self._active_stream_playback_owner = None
                stream_owner.report_terminal("stopped")

            async with self._playback_handoff_lock:
                file_owner = self._active_file_playback_owner
                file_owner_matches = bool(
                    file_owner is not None
                    and (
                        bare_stop
                        or (
                            file_owner.message_id == event.message_id
                            and file_owner is event.playback_lifecycle
                        )
                    )
                )
                handoff = self._active_file_playback_stop
                playback_started = self._active_file_playback_started
                if file_owner_matches and file_owner is not None:
                    matching_handoff = bool(
                        handoff is not None and handoff[0] == file_owner.message_id
                    )
                    pending_start = bool(
                        matching_handoff
                        and playback_started is not None
                        and not playback_started.is_set()
                    )
                    if pending_start:
                        file_stop_accepted = True
                        handoff[1].set()
                        if self._active_file_playback_owner is file_owner:
                            self._active_file_playback_owner = None
                        if self._active_file_playback_stop == handoff:
                            self._active_file_playback_stop = None
                        if self._active_file_playback_started is playback_started:
                            self._active_file_playback_started = None
                    else:
                        async with self._audio_files_lock:
                            last_played = self._last_played
                            if (
                                last_played is None
                                or last_played[0] != file_owner.message_id
                            ):
                                last_played = None

                            if last_played is not None:
                                try:
                                    file_stop_accepted = (
                                        stop_audio_playback_if_current(last_played[1])
                                    )
                                except Exception as exc:
                                    file_stop_retryable_failure = True
                                    logger.warning(
                                        "Owned TTS file stop failed; ownership retained "
                                        "for retry ({})",
                                        type(exc).__name__,
                                    )
                                else:
                                    file_stop_retryable_failure = not file_stop_accepted
                            elif matching_handoff:
                                # Older internal state may have a handoff
                                # without the newer started marker.
                                file_stop_accepted = True
                            else:
                                file_stop_retryable_failure = True

                            if file_stop_accepted:
                                if matching_handoff and handoff is not None:
                                    handoff[1].set()
                                if self._last_played == last_played:
                                    self._last_played = None
                                if self._active_file_playback_owner is file_owner:
                                    self._active_file_playback_owner = None
                                if self._active_file_playback_stop == handoff:
                                    self._active_file_playback_stop = None
                                if self._active_file_playback_started is playback_started:
                                    self._active_file_playback_started = None

            if file_stop_accepted and file_owner is not None:
                file_owner.report_terminal("stopped")

            if bare_stop and file_owner is None:
                file_stop_accepted = (
                    await self._stop_prior_legacy_clip(bare_stop=True)
                    or file_stop_accepted
                )

        if (
            event.action == "play"
            and event.message_id
            and event.playback_lifecycle is not None
        ):
            lifecycle = event.playback_lifecycle
            if not lifecycle.is_current():
                event.report_outcome(False)
                return
            stop_requested = threading.Event()
            playback_started = threading.Event()
            audio_file = None
            owner_mismatch = False
            playback_cancelled = False
            async with self._file_play_admission_lock:
                async with self._playback_handoff_lock:
                    if not lifecycle.is_current():
                        event.report_outcome(False)
                        return
                    prior_owner = self._active_file_playback_owner
                    prior_handoff = self._active_file_playback_stop
                    prior_started = self._active_file_playback_started
                    self._active_file_playback_owner = lifecycle
                    self._active_file_playback_stop = (
                        event.message_id,
                        stop_requested,
                    )
                    self._active_file_playback_started = playback_started

                # Keep the pending reservation visible while artifact lookup
                # waits, so an exact Stop can cancel this Play before start.
                async with self._audio_files_lock:
                    pass

                async with self._playback_handoff_lock:
                    reservation_active = bool(
                        self._active_file_playback_owner is lifecycle
                        and self._active_file_playback_stop
                        == (event.message_id, stop_requested)
                    )
                    playback_cancelled = bool(
                        stop_requested.is_set()
                        or not reservation_active
                        or not lifecycle.is_current()
                    )
                    async with self._audio_files_lock:
                        audio_file = self._audio_files.get(event.message_id)
                        cached_owner = self._audio_file_owners.get(event.message_id)
                        owner_mismatch = bool(
                            audio_file is not None
                            and cached_owner is not None
                            and cached_owner is not lifecycle
                        )
                        if owner_mismatch or playback_cancelled:
                            if self._active_file_playback_owner in (None, lifecycle):
                                self._active_file_playback_owner = prior_owner
                                self._active_file_playback_stop = prior_handoff
                                self._active_file_playback_started = prior_started
                        else:
                            if (
                                audio_file is not None
                                and cached_owner is None
                                and event.message_id not in self._audio_file_owners
                            ):
                                self._audio_file_owners[event.message_id] = lifecycle
                            if (
                                prior_owner is not None
                                and prior_owner is not lifecycle
                            ):
                                if (
                                    prior_handoff is not None
                                    and prior_handoff[0] == prior_owner.message_id
                                ):
                                    prior_handoff[1].set()
                                prior_owner.report_terminal("stopped")
                            if audio_file is not None and audio_file.exists():
                                self._last_played = (event.message_id, audio_file)
            if owner_mismatch:
                lifecycle.report_terminal("failed")
                event.report_outcome(False)
                return
            if playback_cancelled:
                async with self._playback_handoff_lock:
                    if self._active_file_playback_owner is lifecycle:
                        self._active_file_playback_owner = None
                    if self._active_file_playback_stop == (
                        event.message_id,
                        stop_requested,
                    ):
                        self._active_file_playback_stop = None
                    if self._active_file_playback_started is playback_started:
                        self._active_file_playback_started = None
                event.report_outcome(False)
                return
            if audio_file is None or not audio_file.exists():
                async with self._playback_handoff_lock:
                    if self._active_file_playback_owner is lifecycle:
                        self._active_file_playback_owner = None
                    if self._active_file_playback_stop == (
                        event.message_id,
                        stop_requested,
                    ):
                        self._active_file_playback_stop = None
                    if self._active_file_playback_started is playback_started:
                        self._active_file_playback_started = None
                await self._cleanup_audio_file(
                    event.message_id,
                    artifact_owner=lifecycle,
                )
                lifecycle.report_terminal("failed")
                event.report_outcome(False)
                return

            stop_live_sink()
            task = asyncio.create_task(
                self._run_owned_file_playback(
                    event.message_id,
                    audio_file,
                    lifecycle,
                    stop_requested,
                    playback_started,
                )
            )
            self._active_file_playback_task = task
            await self._add_active_task(task)
            event.report_outcome(True)
            return

        if event.action == "play" and event.message_id:
            # Get audio file with lock
            async with self._audio_files_lock:
                audio_file = self._audio_files.get(event.message_id)
                if self._audio_file_owners.get(event.message_id) is not None:
                    audio_file = None

            if audio_file and audio_file.exists():
                # Fix-round F1/N1: symmetric with `_stop_prior_legacy_clip`
                # (which silences a legacy clip before a NEW streaming
                # utterance starts) -- a legacy play request must silence a
                # currently LIVE sink first, or the two independent
                # audio-output paths (the sink's own `sounddevice.
                # OutputStream` vs. the legacy `SimpleAudioPlayer`) would
                # overlap into a double voice. Deliberately placed AFTER
                # the file-exists check (N1 fix-round): this branch is the
                # only one that will actually replace whatever is
                # currently playing -- stopping a live sink for a `play`
                # whose cached artifact has already been cleaned up (the
                # 5s cache cleanup below) would silence real audio and
                # play nothing back, a strictly worse outcome than leaving
                # it alone. A no-op when nothing is currently live.
                stop_live_sink()
                # Play the audio file
                play_audio_file(audio_file)
                # Record what the player now has loaded, independent of
                # `_audio_files` (task-559 fix round 1: that cache is
                # deleted by the cleanup scheduled right below, 5s after
                # playback STARTS, not after it finishes, so a stop-guard
                # keyed off `_audio_files` alone goes blind for any clip
                # over 5s -- the common case, since Console auto-plays
                # every spoken message). A single slot, overwritten here on
                # every play (fix round 2) -- never cleared by the timed
                # cache cleanup below, only by a matching stop or shutdown.
                async with self._audio_files_lock:
                    self._last_played = (event.message_id or "adhoc", audio_file)
                # Schedule cleanup after playback
                asyncio.create_task(
                    self._cleanup_audio_file(
                        event.message_id,
                        delay=5.0,
                        artifact_owner=None,
                    )
                )
            else:
                logger.warning(f"Audio file not found for message {event.message_id}")

        elif event.action == "pause" and event.message_id:
            # The audio player doesn't support pause, so we stop playback
            # but keep the audio file for resuming
            logger.info(f"Pausing audio for message {event.message_id}")
            # This will stop any playing audio but won't delete the file

        elif event.action == "stop" and event.message_id:
            # Interrupt in-flight playback (task-559 unit 2) -- only when
            # this message's audio is the one currently loaded in the
            # shared single-slot player, so stopping message A can never
            # silence a different, actively-playing message B. Two checks,
            # both required: (1) here, the requested message id must match
            # the single tracked `_last_played` slot -- NOT `_audio_files`,
            # which is routinely gone-by-now (see the "play" branch above);
            # (2) inside `stop_audio_playback_if_current`, the player's own
            # `get_current_file()` must still match the tracked path (kept
            # correct even after the file itself was deleted, since
            # deletion doesn't rewrite the player's recorded Path).
            normalized_id = event.message_id or "adhoc"
            last_played = None
            async with self._audio_files_lock:
                cached_artifact = self._audio_files.get(normalized_id)
                cached_artifact_owner = self._audio_file_owners.get(normalized_id)
            exact_cached_artifact = bool(
                cached_artifact is not None
                and event.playback_lifecycle is not None
                and cached_artifact_owner is event.playback_lifecycle
            )
            legacy_file_stop_branch = bool(
                event.playback_lifecycle is None
                and self._active_file_playback_owner is None
                and cached_artifact_owner is None
            )
            if legacy_file_stop_branch:
                async with self._audio_files_lock:
                    last_played = self._last_played
                    if last_played is not None and last_played[0] == normalized_id:
                        self._last_played = None
                    else:
                        last_played = None
            stopped = False
            handoff = self._active_file_playback_stop
            handoff_accepted = bool(
                self._active_file_playback_owner is None
                and handoff is not None
                and handoff[0] == event.message_id
            )
            if handoff_accepted and handoff is not None:
                handoff[1].set()
            if last_played is not None:
                stopped = stop_audio_playback_if_current(last_played[1])
            accepted = (
                stopped
                or handoff_accepted
                or file_stop_accepted
                or stream_stop_accepted
                or generation_stop_accepted
                or (
                    exact_cached_artifact
                    and self._active_file_playback_owner is None
                )
            )
            if accepted:
                logger.info(f"Stopped playback for message {event.message_id}")
            else:
                logger.debug(
                    f"Stop requested for message {event.message_id}; "
                    "nothing was playing"
                )
            # Clean up the (likely already-gone) cached file entry too.
            if file_stop_accepted and file_owner is not None:
                await self._cleanup_audio_file(
                    event.message_id,
                    artifact_owner=file_owner,
                )
            elif (
                exact_cached_artifact
                and self._active_file_playback_owner is None
            ):
                await self._cleanup_audio_file(
                    event.message_id,
                    artifact_owner=event.playback_lifecycle,
                )
            elif legacy_file_stop_branch:
                await self._cleanup_audio_file(
                    event.message_id,
                    artifact_owner=None,
                )
            if event.playback_lifecycle is not None:
                if accepted:
                    event.playback_lifecycle.report_terminal("stopped")
                elif not file_stop_retryable_failure:
                    event.playback_lifecycle.report_terminal("failed")
            event.report_outcome(accepted)
        elif event.action == "stop":
            event.report_outcome(
                stream_stop_accepted
                or file_stop_accepted
                or generation_stop_accepted
            )

    async def _cleanup_audio_file(
        self,
        message_id: str,
        delay: float = 0,
        *,
        artifact_owner: TTSPlaybackLifecycle | None | object = _ANY_ARTIFACT_OWNER,
    ) -> None:
        """Clean up audio file after playback"""
        if delay > 0:
            await asyncio.sleep(delay)

        async with self._audio_files_lock:
            audio_file = self._audio_files.get(message_id)
            cached_owner = self._audio_file_owners.get(message_id)
            if (
                audio_file is not None
                and artifact_owner is not _ANY_ARTIFACT_OWNER
                and cached_owner is not artifact_owner
            ):
                audio_file = None
        if audio_file is None:
            return

        if await self._try_secure_delete_tts_artifact(
            audio_file,
            on_late_success=partial(
                self._schedule_audio_file_release_if_current,
                message_id,
                audio_file,
                artifact_owner,
            ),
        ):
            await self._release_audio_file_if_current(
                message_id,
                audio_file,
                artifact_owner,
            )
            logger.debug(f"Cleaned up audio file for message {message_id}")

    def on_tts_request_event(self, event: TTSRequestEvent) -> None:
        """Handle TTS request event"""
        task = asyncio.create_task(self.handle_tts_request(event))
        # Use create_task to add task safely
        asyncio.create_task(self._add_active_task(task))

    def on_tts_message_speech_request_event(
        self,
        event: TTSMessageSpeechRequestEvent,
    ) -> None:
        """Handle one trusted Console message speech request event."""
        task = asyncio.create_task(self.handle_tts_request(event))
        asyncio.create_task(self._add_active_task(task))

    def on_tts_playback_event(self, event: TTSPlaybackEvent) -> None:
        """Handle TTS playback event"""
        task = asyncio.create_task(self.handle_tts_playback(event))
        # Use create_task to add task safely
        asyncio.create_task(self._add_active_task(task))

    async def _add_active_task(self, task: asyncio.Task) -> None:
        """Add task to active tasks set with lock"""
        async with self._active_tasks_lock:
            self._active_tasks.add(task)
            task.add_done_callback(
                lambda t: asyncio.create_task(self._remove_active_task(t))
            )

    async def _remove_active_task(self, task: asyncio.Task) -> None:
        """Remove task from active tasks set with lock"""
        async with self._active_tasks_lock:
            self._active_tasks.discard(task)

    async def cleanup_tts_resources(self) -> None:
        """Clean up all TTS resources"""
        self._pending_global_overrides.clear()

        # Task-4 review N5: cancel outright, never drain/await -- these are
        # pure `asyncio.sleep(5)`-then-delete timers (`_schedule_legacy_
        # playback_cleanup`), and this method's own artifact-deletion pass
        # below deletes the same files directly moments later regardless.
        # Measured cost of the old behavior (implicitly draining them via
        # `_retained_tts_cleanup_tasks`): ~2s added to shutdown for zero
        # benefit, since a 5s timer can never finish within `_drain_
        # retained_tts_artifact_work`'s two 1s bounds anyway. `asyncio.
        # sleep` responds to cancellation immediately, so this is fast.
        pending_legacy_timers = list(self._pending_legacy_cleanup_timers)
        for timer in pending_legacy_timers:
            if not timer.done():
                timer.cancel()
        if pending_legacy_timers:
            await asyncio.gather(*pending_legacy_timers, return_exceptions=True)

        # Cancel all active tasks with lock
        async with self._active_tasks_lock:
            tasks_to_cancel = list(self._active_tasks)

        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete cancellation
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        await self._drain_retained_tts_artifact_work()

        # Snapshot owned files without dropping failed-deletion bookkeeping.
        async with self._audio_files_lock:
            files_to_clean = [
                (
                    message_id,
                    audio_file,
                    self._audio_file_owners.get(
                        message_id,
                        _ANY_ARTIFACT_OWNER,
                    ),
                )
                for message_id, audio_file in self._audio_files.items()
            ]
            retries_to_clean = list(self._artifact_cleanup_retry)
            self._last_played = None

        for message_id, audio_file, artifact_owner in files_to_clean:
            if await self._try_secure_delete_tts_artifact(
                audio_file,
                on_late_success=partial(
                    self._schedule_audio_file_release_if_current,
                    message_id,
                    audio_file,
                    artifact_owner,
                ),
            ):
                await self._release_audio_file_if_current(
                    message_id,
                    audio_file,
                    artifact_owner,
                )
                logger.debug(f"Cleaned up audio file for message {message_id}")

        for audio_file in retries_to_clean:
            if await self._try_secure_delete_tts_artifact(audio_file):
                async with self._audio_files_lock:
                    self._artifact_cleanup_retry.discard(audio_file)

        await self._drain_retained_tts_artifact_work()

        # Clear active tasks with lock
        async with self._active_tasks_lock:
            self._active_tasks.clear()


#######################################################################################################################
#
# Helper Functions


def _wants_wav_collection(audio_format: str) -> bool:
    """Whether wav bytes should be retained for the post-write sink-eligibility check.

    Fix-round F3 (task-4 review): gated on BOTH the format AND
    `sink_available()` -- not format alone -- so a machine with no
    `sounddevice` at all (the exact case the Global Constraint says the
    legacy path must be left byte- and memory-profile-identical for) never
    pays the whole-body memory cost the legacy write loop's own
    `_TTS_ARTIFACT_WRITE_BATCH_BYTES` batching was specifically designed to
    avoid. `sink_available()` is a cheap `importlib.util.find_spec` probe
    (measured at ~0.008ms), so hoisting it into this gate costs nothing.
    A plain function (not inlined) so the gate itself is directly
    unit-testable without needing to drive a full `_generate_tts` call.
    """
    return audio_format == "wav" and sink_available()


async def _replay_drained_bytes(data: bytes) -> AsyncIterator[bytes]:
    """Replay one already-collected buffer as a single-chunk async source.

    Used only for the WAV half of the streaming seam in `_generate_tts`:
    WAV eligibility is decided AFTER the legacy write loop has already
    collected the response's bytes (see the comment above
    `_create_tts_artifact`), so playing an eligible one through the sink
    needs to replay those already-in-memory bytes as an `AsyncIterator`,
    the same shape `pump()` expects a live `response.byte_stream` to be.
    """
    yield data


def _legacy_playback_timeout_seconds(text_length: int, speed: float = 1.0) -> float:
    """A generous, text-length-and-speed-derived bound on how long one
    legacy-path utterance could plausibly still be playing (task-4 review
    F2, refined by N3).

    There is no real audio duration available to bound
    `_play_legacy_clip_and_await_completion`'s poll loop against --
    `AudioPlayerInfo.duration` (`TTS/audio_player.py`) is declared but never
    populated by `SimpleAudioPlayer.play()`. This estimates instead from the
    SYNTHESIZED TEXT's own character count, at a deliberately slow
    (over-estimating) assumed speech rate, divided by `speed` (task-4
    review N3: `default_speed` is user-configurable and `TTS/preferences.
    py`'s `_require_speed` only enforces "finite positive" -- a bound that
    ignores it can under-estimate, and silently, since a timeout is
    indistinguishable from a natural finish to the caller: see
    `_play_legacy_clip_and_await_completion`'s own `True`-on-timeout
    return), plus a fixed margin for playback startup/executor-hop
    latency, capped at an absolute ceiling so a pathological input can
    never poll indefinitely (`speak_utterance`'s generation task is
    cancellable via `_active_tasks` regardless -- see task-4 review F4,
    and N4's prompt cancel-stop -- but a sane ceiling is still worth
    keeping on its own merits).

    Args:
        text_length: Length of the synthesized text.
        speed: The resolved provider speed (1.0 = normal pace). Defensively
            floored to a small positive value -- `_require_speed` already
            guarantees "finite positive" upstream, but this function must
            never divide by zero or a negative number regardless of what a
            caller passes.
    """
    effective_speed = speed if speed > 0 else 1.0
    estimated = text_length / (_LEGACY_PLAYBACK_MIN_CHARS_PER_SECOND * effective_speed)
    return min(
        _LEGACY_PLAYBACK_POLL_MAX_SECONDS,
        estimated + _LEGACY_PLAYBACK_POLL_MARGIN_SECONDS,
    )


def _play_legacy_clip_and_await_completion(
    player,
    audio_file: Path,
    *,
    timeout_seconds: float,
    stop_requested: threading.Event | None = None,
    poll_interval_seconds: float = _LEGACY_PLAYBACK_POLL_INTERVAL_SECONDS,
    on_started: Callable[[], None] | None = None,
) -> bool:
    """Blocking: start playback on `player` and wait (bounded) for this clip
    to stop being the player's current one (task-4 review F2, the
    headline finding).

    `SimpleAudioPlayer.play()` (`TTS/audio_player.py`) launches a
    background player process and returns as soon as it has started --
    NOT when playback finishes -- with a daemon `_monitor_playback` thread
    doing the actual `process.wait()` and flipping `get_state()` to
    `FINISHED` on a natural end. Reporting completion the instant `play()`
    returns (the pre-fix behavior) meant `speak_utterance`'s `on_finished`
    fired while audio was still playing; since `play()` itself calls
    `self.stop()` first (a single-slot global singleton), the VERY NEXT
    utterance's handoff would kill the current one mid-word the instant it
    landed -- confirmed by the reviewer's timing probe (a completion ->
    playback-end gap of ~0ms for a 0.35s clip). This must run entirely off
    the event loop -- callers MUST invoke it through the existing
    `_run_blocking_tts_io` offload seam, the same one `sink.open()` already
    uses for exactly this "blocking work belongs on a worker thread" reason.

    Task-4 review round 3 (F3, the third pass on this finding): `player.
    play()` (the REAL `SimpleAudioPlayer`) itself calls `self.stop()`
    FIRST, then does file/format checks and (on macOS with `afplay`) a
    `time.sleep(0.1)`, and only THEN calls `Popen`. A stop request that
    lands on the event-loop side, calling `player.stop()` directly, DURING
    that pre-`Popen` window has no process to kill -- a no-op -- and
    `play()` proceeds past its own internal `self.stop()` and starts the
    clip regardless, playing through to this poll's own timeout. Only this
    WORKER (which alone knows when `Popen` actually ran, since it is the
    one that called `play()`) can close that window: `stop_requested` is
    checked immediately after `player.play()` returns (at that point,
    `Popen` has definitively either happened or `play()` itself failed --
    either way this second `player.stop()` call is not a no-op if a stop
    truly was requested mid-handoff) and again on every poll iteration
    thereafter, for a barge-in landing later.

    Args:
        player: A `SimpleAudioPlayer`-shaped object (`play`, `get_state`,
            `get_current_file`, `stop`) -- injected rather than looked up
            via `get_audio_player()` internally so tests can pass a fake.
        audio_file: The clip this call is responsible for.
        timeout_seconds: Poll bound -- see `_legacy_playback_timeout_
            seconds`. Because an explicit `stop()` resets `get_state()` to
            `IDLE` and clears `get_current_file()` immediately
            (`audio_player.py`'s own `stop()`), a barge-in exits this poll
            promptly rather than pinning the worker for the full bound.
        stop_requested: Set by `_stop_prior_legacy_clip`'s bare-stop branch
            (task-4 review round 3) for THIS specific handoff. `None` is
            treated the same as an event that is never set.
        poll_interval_seconds: Sleep between checks.

    Returns:
        `False` if `play()` itself never started the clip, if a stop was
        requested (checked immediately after `play()` returns, and on
        every subsequent poll iteration), or if the clip stopped being
        current for any OTHER reason before reaching `FINISHED` --
        displacement by a different clip (a race with another `play()`
        call, or the single-slot player's own next caller). `True` for a
        natural finish, or -- best-effort -- if the poll bound was reached
        while the clip was STILL current and still `PLAYING` (assume it
        played through rather than penalize a long clip for an
        under-estimated bound).
    """
    from tldw_chatbook.TTS.audio_player import PlaybackState

    started = player.play(audio_file)
    if not started:
        return False
    if stop_requested is not None and stop_requested.is_set():
        # A stop landed in the pre-`Popen` window while `play()` was still
        # running -- `Popen` has now definitively either happened or
        # failed, so THIS `stop()` call has something real to act on (or
        # is a safe no-op if `play()` itself failed). Task-4 review D2:
        # identity-guarded, same as the async-side cancel handler -- by
        # the time this worker notices, a DIFFERENT clip could already
        # have displaced ours (another handoff's own `play()`, or an
        # unrelated caller of the same process-global singleton); this
        # utterance still completes as stopped/interrupted (`return
        # False`) either way, but must not reach out and kill audio that
        # is no longer ours to stop.
        if player.get_current_file() in (None, audio_file):
            player.stop()
        return False
    if player.get_current_file() != audio_file:
        return False
    if on_started is not None:
        on_started()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if stop_requested is not None and stop_requested.is_set():
            if player.get_current_file() in (None, audio_file):
                player.stop()
            return False
        if player.get_current_file() != audio_file:
            return False
        if player.get_state() == PlaybackState.FINISHED:
            return True
        time.sleep(poll_interval_seconds)
    # Task-4 review N3: the estimate under-ran -- make that DISTINGUISHABLE
    # in the logs rather than silently indistinguishable from a natural
    # finish (both return `True`). `logger` (loguru) is documented
    # thread-safe; this runs off the event loop, on the same worker thread
    # as the poll above.
    logger.warning(
        "Legacy TTS playback poll timed out before observing FINISHED "
        "(bound={:.1f}s); assuming it played through",
        timeout_seconds,
    )
    return True


def play_audio_file(file_path: Path) -> None:
    """Play an audio file using system default player"""
    # Use the centralized audio player from audio_player module
    from tldw_chatbook.TTS.audio_player import play_audio_file as play_audio

    # Delegate to the secure implementation
    success = play_audio(file_path)
    if not success:
        logger.error(f"Failed to play audio file: {file_path}")


def stop_audio_playback_if_current(file_path: Path) -> bool:
    """Stop the shared system audio player, but only if it currently owns ``file_path``.

    `SimpleAudioPlayer` (`TTS/audio_player.py`) is a single-slot global
    singleton -- only one clip can be "current" system-wide at any time,
    since every `play()` call stops whatever was previously loaded first.
    Comparing before stopping keeps a stop request scoped to one message
    from silencing a different, unrelated message's still-playing audio
    (task-559 unit 2; a real scenario for legacy chat, where audio is not
    auto-played and several messages can sit cached-but-never-played
    simultaneously while a different one is actively playing). The
    comparison stays correct even after the underlying file was deleted by
    the 5s cache cleanup (fix round 1) -- `get_current_file()` reports the
    player's own recorded path, not disk state.

    Args:
        file_path: The audio file to check against the player's currently
            loaded clip (`SimpleAudioPlayer.get_current_file()`) before
            deciding whether to stop it.

    Returns:
        ``True`` if the player was actually told to stop, ``False`` if
        ``file_path`` wasn't the one currently loaded (a no-op).
    """
    from tldw_chatbook.TTS.audio_player import get_audio_player

    player = get_audio_player()
    if player.get_current_file() == file_path:
        player.stop()
        return True
    return False


#
# End of tts_events.py
#######################################################################################################################
