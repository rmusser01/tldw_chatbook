# tts_events.py
# Description: Event handlers for TTS functionality
#
# Imports
import asyncio
import re
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Dict, Optional, TypeVar
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from uuid import uuid4
from loguru import logger

# Third-party imports
from textual.message import Message

# Local imports
from tldw_chatbook.Chat.console_speech import (
    ConsoleSpeechSnapshotRejected,
    TTSMessageSpeechSnapshot,
)
from tldw_chatbook.TTS import (
    CharacterTTSRequestResolution,
    CharacterTTSRequestResolver,
    CharacterTTSResolutionError,
    TTSRequestedSelectionSnapshot,
    get_tts_service,
)
from tldw_chatbook.TTS.adapter_types import (
    TTSConfigurationRevisionError,
    TTSOperationError,
    TTSProgress,
    TTSProviderReconfiguringError,
    TTSProviderUnavailableError,
    TTSRequest,
    TTSRegistryClosedError,
)
from tldw_chatbook.Utils.secure_temp_files import get_temp_manager, secure_delete_file

_T = TypeVar("_T")
_TTS_ARTIFACT_WRITE_BATCH_BYTES = 64 * 1024
_TTS_IO_CANCELLATION_JOIN_TIMEOUT_SECONDS = 1.0
_TTS_SECURE_DELETE_TIMEOUT_SECONDS = 1.0
_TTS_RETAINED_WORK_DRAIN_TIMEOUT_SECONDS = 1.0
_GLOBAL_OVERRIDE_TOKEN_PATTERN = re.compile(r"[0-9a-f]{32}\Z")


class _TTSResponseContractError(RuntimeError):
    """Raised when a synthesized response violates the Console audio contract."""


class _TTSArtifactIOTimeout(RuntimeError):
    """Raised when bounded artifact I/O continues in a retained worker."""


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
    ) -> None:
        super().__init__()
        if type(snapshot) is not TTSMessageSpeechSnapshot:
            raise ValueError("snapshot must be TTSMessageSpeechSnapshot")
        if not callable(validator):
            raise ValueError("validator must be callable")
        self.snapshot = snapshot
        self.validator = validator

    @property
    def message_id(self) -> str:
        """Expose the native message id without duplicating caller text."""
        return self.snapshot.message_id


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

    def __init__(self, action: str, message_id: Optional[str] = None):
        super().__init__()
        self.action = action  # "play", "pause", "stop"
        self.message_id = message_id


class TTSExportEvent(Message):
    """Event to export TTS audio with custom naming"""

    def __init__(
        self, message_id: str, output_path: Path, include_metadata: bool = True
    ):
        super().__init__()
        self.message_id = message_id
        self.output_path = output_path
        self.include_metadata = include_metadata


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
    """Private admission state retained behind an opaque one-use token."""

    snapshot: TTSMessageSpeechSnapshot
    validator: Callable[[TTSMessageSpeechSnapshot], str]
    created_at: float


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
        profile_service_loader: Callable[[], Awaitable[object | None]]
        | None = None,
    ):
        self._tts_service = None
        self._profile_service_loader = profile_service_loader
        self._pending_global_overrides: dict[str, _PendingGlobalOverride] = {}
        self._temp_manager = get_temp_manager()
        self._audio_files: Dict[str, Path] = {}  # Track audio files by message_id
        self._artifact_cleanup_retry: set[Path] = set()
        self._retained_tts_io_tasks: set[asyncio.Task] = set()
        self._retained_tts_cleanup_tasks: set[asyncio.Task] = set()
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

    async def _post_tts_message(self, message: Message) -> None:
        """Post a message through Textual app wiring or a direct test handler."""
        app = getattr(self, "app", None)
        if app is not None and hasattr(app, "post_message"):
            app.post_message(message)
            return

        post_message = getattr(self, "post_message", None)
        if callable(post_message):
            result = post_message(message)
            if asyncio.iscoroutine(result):
                await result

    async def handle_tts_request(
        self,
        event: TTSRequestEvent | TTSMessageSpeechRequestEvent,
    ) -> None:
        """Admit a trusted request, then run the shared TTS generation path."""
        if isinstance(event, TTSMessageSpeechRequestEvent):
            request_text = await self._validate_message_speech_snapshot(
                event.snapshot,
                event.validator,
            )
            if request_text is None:
                return
            request_message_id: str | None = event.message_id
            request_voice: str | None = None
        else:
            request_text = event.text
            request_message_id = event.message_id
            request_voice = event.voice
            # Preserve the legacy explicit-request maintenance behavior even
            # when no service is available. Trusted snapshots keep their
            # stricter validate/resolve-before-cooldown ordering below.
            self._enforce_cooldown_limit()

        text = await self._prepare_tts_text(
            request_text,
            request_message_id or "unknown",
        )
        if text is None:
            return

        resolution: CharacterTTSRequestResolution | None = None
        if isinstance(event, TTSMessageSpeechRequestEvent):
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
                    )
                logger.warning(
                    "Console character speech resolution failed "
                    "(outcome_code={})",
                    error.code,
                )
                await self._post_tts_message(
                    TTSCompleteEvent(
                        message_id=event.message_id,
                        error=str(error),
                        global_override_token=token,
                    )
                )
                return

        await self._admit_tts_generation(
            text=text,
            message_id=request_message_id or "adhoc",
            voice=request_voice,
            resolution=resolution,
        )

    async def handle_tts_global_override_decision(
        self,
        event: TTSGlobalOverrideDecisionEvent,
    ) -> None:
        """Consume one fallback decision and re-admit its original snapshot."""
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
                )
            )
            return None
        except Exception:
            logger.warning(
                "Console speech snapshot rejected "
                "(outcome_code=validator_failure)"
            )
            await self._post_tts_message(
                TTSCompleteEvent(
                    message_id=snapshot.message_id,
                    error=ConsoleSpeechSnapshotRejected.USER_COPY,
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
                )
            )
            return None
        return request_text

    async def _prepare_tts_text(
        self,
        request_text: object,
        message_id: str,
    ) -> str | None:
        """Validate and normalize text before assignment or cooldown admission."""
        if not self._tts_service:
            logger.error("TTS service not initialized")
            await self._post_tts_message(
                TTSCompleteEvent(
                    message_id=message_id,
                    error="TTS service not available",
                )
            )
            return None
        if type(request_text) is not str or not request_text:
            await self._post_tts_message(
                TTSCompleteEvent(
                    message_id=message_id,
                    error="No text provided for TTS generation",
                )
            )
            return None

        max_tts_length = 5000
        if len(request_text) > max_tts_length:
            logger.warning("TTS text exceeds the configured length limit")
            await self._post_tts_message(
                TTSCompleteEvent(
                    message_id=message_id,
                    error=(
                        "Text is too long for TTS. Maximum "
                        f"{max_tts_length} characters allowed."
                    ),
                )
            )
            return None

        text = " ".join(request_text.split())
        if not text:
            await self._post_tts_message(
                TTSCompleteEvent(
                    message_id=message_id,
                    error="Text contains only whitespace",
                )
            )
            return None
        return text

    async def _resolve_message_speech_request(
        self,
        text: str,
        snapshot: TTSMessageSpeechSnapshot,
    ) -> CharacterTTSRequestResolution:
        """Resolve an exact assignment only for verified character authorship."""
        profile_service = None
        if snapshot.assistant_kind == "character":
            loader = self._profile_service_loader
            if loader is not None:
                try:
                    profile_service = await loader()
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    logger.warning(
                        "TTS profile service load failed "
                        "(exception_category={})",
                        type(error).__name__,
                    )
        resolver = CharacterTTSRequestResolver(profile_service)
        return await resolver.resolve(
            text=text,
            assistant_kind=snapshot.assistant_kind,
            character_ref=snapshot.character_ref,
        )

    def _issue_global_override(
        self,
        snapshot: TTSMessageSpeechSnapshot,
        validator: Callable[[TTSMessageSpeechSnapshot], str],
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
        )
        return token

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
    ) -> None:
        """Apply cooldown only after validation and character resolution."""
        current_time = asyncio.get_event_loop().time()
        if current_time - self._last_cooldown_cleanup > self.COOLDOWN_CLEANUP_INTERVAL:
            self._cleanup_cooldown_dict(current_time)
            self._last_cooldown_cleanup = current_time
        self._enforce_cooldown_limit()

        if message_id in self._request_cooldown:
            time_since_last = current_time - self._request_cooldown[message_id]
            if time_since_last < self.COOLDOWN_SECONDS:
                wait_seconds = self.COOLDOWN_SECONDS - time_since_last
                logger.warning(
                    "TTS request rejected by message cooldown "
                    "(wait_seconds={:.1f})",
                    wait_seconds,
                )
                await self._post_tts_message(
                    TTSCompleteEvent(
                        message_id=message_id,
                        error=(
                            f"Please wait {wait_seconds:.1f} seconds before "
                            "requesting TTS again"
                        ),
                    )
                )
                return

        self._request_cooldown[message_id] = current_time
        self._enforce_cooldown_limit()

        task = asyncio.create_task(
            self._generate_tts_with_rate_limit(
                text,
                message_id,
                voice,
                resolution,
            )
        )
        asyncio.create_task(self._add_active_task(task))

    async def _generate_tts_with_rate_limit(
        self,
        text: str,
        message_id: Optional[str],
        voice: Optional[str],
        resolution: CharacterTTSRequestResolution | None = None,
    ) -> None:
        """Generate TTS audio (rate limiting handled by TTSService)"""
        try:
            await self._generate_tts(text, message_id, voice, resolution)
        except asyncio.CancelledError:
            logger.info(f"TTS generation cancelled for message {message_id}")
            raise

    async def _generate_tts(
        self,
        text: str,
        message_id: Optional[str],
        voice: Optional[str],
        resolution: CharacterTTSRequestResolution | None = None,
    ) -> None:
        """Generate one complete resolved TTS response and publish its artifact."""
        from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram

        normalized_message_id = message_id or "adhoc"
        resolution_source = (
            resolution.source
            if resolution is not None
            else ("explicit_override" if voice is not None else "global")
        )
        start_time = asyncio.get_event_loop().time()
        outcome_code = "generation_failed"
        provider_id: str | None = None
        response = None
        artifact_path: Path | None = None

        try:
            service = self._tts_service
            if service is None:
                raise TTSProviderUnavailableError("TTS service is unavailable")

            exact_request = (
                resolution.request
                if resolution is not None and resolution.source == "assigned"
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
                    if (
                        isinstance(candidate_provider_id, str)
                        and candidate_provider_id
                    ):
                        provider_id = candidate_provider_id
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
                    response, requested_selection = await service.synthesize_exact(
                        exact_request,
                        progress_sink=progress_sink,
                    )
                    self._validate_exact_selection(
                        exact_request,
                        requested_selection,
                    )
                else:
                    response = await service.synthesize_default(
                        text=text,
                        voice_override=voice,
                        progress_sink=progress_sink,
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
                    buffered_chunks.append(chunk)
                    buffered_bytes += len(chunk)
                    if buffered_bytes >= _TTS_ARTIFACT_WRITE_BATCH_BYTES:
                        await flush_artifact_batch()
                await flush_artifact_batch()
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
            async with self._audio_files_lock:
                self._audio_files[normalized_message_id] = artifact_path

            await self._post_tts_message(
                TTSProgressEvent(
                    message_id=normalized_message_id,
                    progress=1.0,
                    status="Audio generation complete",
                )
            )
            await self._post_tts_message(
                TTSCompleteEvent(
                    message_id=normalized_message_id,
                    audio_file=artifact_path,
                )
            )
            outcome_code = "success"
        except asyncio.CancelledError as cancellation:
            outcome_code = "cancelled"
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
            outcome_code = self._tts_outcome_code(error)
            await self._discard_tts_artifact(normalized_message_id, artifact_path)
            logger.error(
                "TTS generation failed (outcome_code={})",
                outcome_code,
            )
            await self._post_tts_message(
                TTSCompleteEvent(
                    message_id=normalized_message_id,
                    error=self._tts_error_copy(error),
                )
            )
        finally:
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

    async def _release_audio_file_if_current(
        self,
        message_id: str,
        artifact_path: Path,
    ) -> None:
        """Release a cache entry only when it still owns the deleted artifact."""
        async with self._audio_files_lock:
            if self._audio_files.get(message_id) == artifact_path:
                del self._audio_files[message_id]

    def _schedule_audio_file_release_if_current(
        self,
        message_id: str,
        artifact_path: Path,
    ) -> None:
        """Track cache bookkeeping triggered by a retained delete worker."""
        release = asyncio.create_task(
            self._release_audio_file_if_current(message_id, artifact_path)
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
    ) -> None:
        """Remove one failed or cancelled artifact from cache and disk."""
        if artifact_path is None:
            return
        async with self._audio_files_lock:
            if self._audio_files.get(message_id) == artifact_path:
                del self._audio_files[message_id]
            self._artifact_cleanup_retry.add(artifact_path)

        if await self._try_secure_delete_tts_artifact(artifact_path):
            async with self._audio_files_lock:
                self._artifact_cleanup_retry.discard(artifact_path)

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
            return "TTS generation failed; retry"
        if isinstance(error, _TTSResponseContractError):
            return (
                "The TTS service returned invalid audio; check provider compatibility"
            )
        if isinstance(error, ValueError):
            return "TTS is not configured; open STTS Settings"
        return "Unexpected TTS generation failure; retry"

    async def handle_tts_playback(self, event: TTSPlaybackEvent) -> None:
        """Handle TTS playback control"""
        logger.info(
            f"TTS playback action: {event.action} for message {event.message_id}"
        )

        if event.action == "play" and event.message_id:
            # Get audio file with lock
            async with self._audio_files_lock:
                audio_file = self._audio_files.get(event.message_id)

            if audio_file and audio_file.exists():
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
                    self._cleanup_audio_file(event.message_id, delay=5.0)
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
            async with self._audio_files_lock:
                last_played = self._last_played
                if last_played is not None and last_played[0] == normalized_id:
                    self._last_played = None
                else:
                    last_played = None
            stopped = False
            if last_played is not None:
                stopped = stop_audio_playback_if_current(last_played[1])
            if stopped:
                logger.info(f"Stopped playback for message {event.message_id}")
            else:
                logger.debug(
                    f"Stop requested for message {event.message_id}; "
                    "nothing was playing"
                )
            # Clean up the (likely already-gone) cached file entry too.
            await self._cleanup_audio_file(event.message_id)

    async def handle_tts_export(self, event: TTSExportEvent) -> None:
        """Handle TTS audio export"""
        import shutil
        import json

        # Get audio file
        async with self._audio_files_lock:
            source_file = self._audio_files.get(event.message_id)

        if not source_file or not source_file.exists():
            logger.error(f"No audio file found for message {event.message_id}")
            self.notify("No audio file found to export", severity="error")
            return

        try:
            # Validate output path
            output_path = event.output_path
            if not output_path.suffix:
                # Add extension from source file
                output_path = output_path.with_suffix(source_file.suffix)

            # Create parent directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy audio file
            shutil.copy2(source_file, output_path)
            logger.info(f"Exported audio to {output_path}")

            # Add metadata if requested
            if event.include_metadata:
                metadata = {
                    "message_id": event.message_id,
                    "export_time": datetime.now().isoformat(),
                    "format": source_file.suffix[1:],  # Remove dot
                    "source": "tldw_chatbook_tts",
                }

                # Save metadata as JSON sidecar file
                metadata_path = output_path.with_suffix(output_path.suffix + ".json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                logger.info(f"Saved metadata to {metadata_path}")

            self.notify(f"Audio exported to {output_path.name}", severity="success")

        except Exception as e:
            logger.error(f"Failed to export audio: {e}")
            self.notify(f"Failed to export audio: {str(e)}", severity="error")

    async def _cleanup_audio_file(self, message_id: str, delay: float = 0) -> None:
        """Clean up audio file after playback"""
        if delay > 0:
            await asyncio.sleep(delay)

        async with self._audio_files_lock:
            audio_file = self._audio_files.get(message_id)
        if audio_file is None:
            return

        if await self._try_secure_delete_tts_artifact(
            audio_file,
            on_late_success=partial(
                self._schedule_audio_file_release_if_current,
                message_id,
                audio_file,
            ),
        ):
            await self._release_audio_file_if_current(message_id, audio_file)
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

    def on_tts_export_event(self, event: TTSExportEvent) -> None:
        """Handle TTS export event"""
        task = asyncio.create_task(self.handle_tts_export(event))
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
            files_to_clean = list(self._audio_files.items())
            retries_to_clean = list(self._artifact_cleanup_retry)
            self._last_played = None

        for message_id, audio_file in files_to_clean:
            if await self._try_secure_delete_tts_artifact(
                audio_file,
                on_late_success=partial(
                    self._schedule_audio_file_release_if_current,
                    message_id,
                    audio_file,
                ),
            ):
                await self._release_audio_file_if_current(message_id, audio_file)
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
