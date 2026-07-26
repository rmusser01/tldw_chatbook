# tts_events.py
# Description: Event handlers for TTS functionality
#
# Imports
import asyncio
from typing import Optional, Dict
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from loguru import logger

# Third-party imports
from textual.message import Message

# Local imports
from tldw_chatbook.TTS import get_tts_service
from tldw_chatbook.TTS.adapter_types import (
    TTSConfigurationRevisionError,
    TTSOperationError,
    TTSProgress,
    TTSProviderReconfiguringError,
    TTSProviderUnavailableError,
    TTSRegistryClosedError,
)
from tldw_chatbook.Utils.secure_temp_files import get_temp_manager, secure_delete_file

#######################################################################################################################
#
# TTS Event Messages


class TTSRequestEvent(Message):
    """Event to request TTS generation"""

    def __init__(
        self, text: str, message_id: Optional[str] = None, voice: Optional[str] = None
    ):
        super().__init__()
        self.text = text
        self.message_id = message_id  # ID of the chat message
        self.voice = voice  # Optional voice override


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
    ):
        super().__init__()
        self.message_id = message_id
        self.audio_file = audio_file
        self.error = error


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

    def __init__(self):
        self._tts_service = None
        self._temp_manager = get_temp_manager()
        self._audio_files: Dict[str, Path] = {}  # Track audio files by message_id
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

    async def handle_tts_request(self, event: TTSRequestEvent) -> None:
        """Handle TTS generation request"""
        current_time = asyncio.get_event_loop().time()
        if current_time - self._last_cooldown_cleanup > self.COOLDOWN_CLEANUP_INTERVAL:
            self._cleanup_cooldown_dict(current_time)
            self._last_cooldown_cleanup = current_time
        self._enforce_cooldown_limit()

        if not self._tts_service:
            logger.error("TTS service not initialized")
            await self._post_tts_message(
                TTSCompleteEvent(
                    message_id=event.message_id or "unknown",
                    error="TTS service not available",
                )
            )
            return

        # Validate input text
        if not event.text:
            await self._post_tts_message(
                TTSCompleteEvent(
                    message_id=event.message_id or "unknown",
                    error="No text provided for TTS generation",
                )
            )
            return

        # Check text length limits
        MAX_TTS_LENGTH = 5000  # Maximum characters for TTS
        if len(event.text) > MAX_TTS_LENGTH:
            logger.warning(f"TTS text too long: {len(event.text)} characters")
            await self._post_tts_message(
                TTSCompleteEvent(
                    message_id=event.message_id or "unknown",
                    error=f"Text is too long for TTS. Maximum {MAX_TTS_LENGTH} characters allowed.",
                )
            )
            return

        # Basic sanitization - remove excessive whitespace
        text = " ".join(event.text.split())
        if len(text) < 1:
            await self._post_tts_message(
                TTSCompleteEvent(
                    message_id=event.message_id or "unknown",
                    error="Text contains only whitespace",
                )
            )
            return

        # Check rate limiting for this message
        message_id = event.message_id or "adhoc"

        if message_id in self._request_cooldown:
            time_since_last = current_time - self._request_cooldown[message_id]
            if time_since_last < self.COOLDOWN_SECONDS:
                logger.warning(
                    f"TTS request too soon for message {message_id}. Please wait {self.COOLDOWN_SECONDS - time_since_last:.1f}s"
                )
                await self._post_tts_message(
                    TTSCompleteEvent(
                        message_id=message_id,
                        error=f"Please wait {self.COOLDOWN_SECONDS - time_since_last:.1f} seconds before requesting TTS again",
                    )
                )
                return

        # Update cooldown tracker
        self._request_cooldown[message_id] = current_time

        # Check if we need to evict old entries (LRU style)
        self._enforce_cooldown_limit()

        # Start TTS generation task
        task = asyncio.create_task(
            self._generate_tts_with_rate_limit(
                text,  # Use sanitized text
                message_id,
                event.voice,
            )
        )
        # Track the task
        asyncio.create_task(self._add_active_task(task))

    async def _generate_tts_with_rate_limit(
        self, text: str, message_id: Optional[str], voice: Optional[str]
    ) -> None:
        """Generate TTS audio (rate limiting handled by TTSService)"""
        try:
            await self._generate_tts(text, message_id, voice)
        except asyncio.CancelledError:
            logger.info(f"TTS generation cancelled for message {message_id}")
            raise

    async def _generate_tts(
        self, text: str, message_id: Optional[str], voice: Optional[str]
    ) -> None:
        """Generate one complete default TTS response and publish its artifact."""
        from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram

        normalized_message_id = message_id or "adhoc"
        resolution_source = "explicit_override" if voice is not None else "global"
        start_time = asyncio.get_event_loop().time()
        outcome_code = "generation_failed"
        provider_id: str | None = None
        response = None
        artifact_path: Path | None = None

        try:
            service = self._tts_service
            if service is None:
                raise TTSProviderUnavailableError("TTS service is unavailable")

            try:
                preferences = service.preferences_snapshot()
                candidate_provider_id = getattr(preferences, "provider_id", None)
                if isinstance(candidate_provider_id, str) and candidate_provider_id:
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
                response = await service.synthesize_default(
                    text=text,
                    voice_override=voice,
                    progress_sink=progress_sink,
                )
                if (
                    not isinstance(response.provider_id, str)
                    or not response.provider_id
                ):
                    raise ValueError("TTS response provider metadata is invalid")
                if not isinstance(response.model_id, str) or not response.model_id:
                    raise ValueError("TTS response model metadata is invalid")
                provider_id = response.provider_id
                audio_format = self._response_audio_format(response.audio_format)
                artifact_path = Path(
                    self._temp_manager.create_temp_file(
                        content=b"",
                        suffix=f".{audio_format}",
                        prefix="tts_audio_",
                    )
                )
                with artifact_path.open("ab") as audio_file:
                    async for chunk in response.byte_stream:
                        audio_file.write(chunk)
                    audio_file.flush()
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
        except asyncio.CancelledError:
            outcome_code = "cancelled"
            await self._discard_tts_artifact(normalized_message_id, artifact_path)
            raise
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
        try:
            secure_delete_file(artifact_path)
        except Exception:
            logger.warning("Failed to remove incomplete TTS artifact")

    @staticmethod
    def _response_audio_format(audio_format: object) -> str:
        """Return one safe canonical extension from response-owned metadata."""
        if not isinstance(audio_format, str):
            raise ValueError("TTS response format metadata is invalid")
        normalized = audio_format.lower().strip().removeprefix(".")
        if normalized not in {"mp3", "opus", "aac", "flac", "wav", "pcm"}:
            raise ValueError("TTS response format metadata is invalid")
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
            return "TTS generation failed; retry"
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
            if message_id in self._audio_files:
                audio_file = self._audio_files[message_id]
                if secure_delete_file(audio_file):
                    logger.debug(f"Cleaned up audio file for message {message_id}")
                del self._audio_files[message_id]

    def on_tts_request_event(self, event: TTSRequestEvent) -> None:
        """Handle TTS request event"""
        task = asyncio.create_task(self.handle_tts_request(event))
        # Use create_task to add task safely
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
        # Cancel all active tasks with lock
        async with self._active_tasks_lock:
            tasks_to_cancel = list(self._active_tasks)

        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete cancellation
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        # Clean up audio files with lock
        async with self._audio_files_lock:
            files_to_clean = list(self._audio_files.items())
            self._audio_files.clear()
            self._last_played = None

        for message_id, audio_file in files_to_clean:
            secure_delete_file(audio_file)
            logger.debug(f"Cleaned up audio file for message {message_id}")

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
