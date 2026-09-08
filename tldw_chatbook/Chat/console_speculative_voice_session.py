"""Production composition root for one view-scoped speculative voice session."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy
from functools import partial
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_voice_process import ConsoleVoiceProcess

from tldw_chatbook.Audio.duplex_contracts import (
    AcousticSafetyPath,
    RenderBoundary,
    RenderSubmission,
)
from tldw_chatbook.Chat.console_speculative_voice import (
    AttemptDispatchPrepared,
    AttemptGenerationCompleted,
    AttemptOutputDelta,
    AttemptPlaybackFailed,
    AttemptPlaybackStarted,
    AttemptTtsFailed,
    SpeculativeTurnCoordinator,
)
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnExecutionContext
from tldw_chatbook.Chat.console_voice_attempts import (
    AttemptCleanupManager,
    AttemptCleanupOutcome,
    ProviderAttemptFailed,
    VoiceAttempt,
    VoiceAttemptDelta,
    VoiceAttemptRequest,
    VoiceAttemptToolRequest,
)
from tldw_chatbook.Chat.console_voice_preflight import (
    voice_failure_category,
)
from tldw_chatbook.Chat.console_voice_promotion import ConsoleSessionBindingOrigin
from tldw_chatbook.Chat.console_voice_supervisor import VoiceDispatchSupervisor
from tldw_chatbook.Chat.voice_phrase_sequencer import PhraseSpeechSequencer
from tldw_chatbook.Utils.persistent_diagnostics import persist_event


_VOICE_FIRST_STAGES = frozenset(
    {"provider_delta", "eligible_phrase", "synthesis_complete", "render_receipt"}
)
_RENDER_FRAME_DURATION_NS = 10_000_000
_voice_diagnostic_sink: ContextVar[Any] = ContextVar(
    "voice_diagnostic_sink", default=None
)


def _persist_voice_event(event: str, **fields: Any) -> None:
    """Best-effort content-free stages for diagnosing the one live smoke test."""

    with contextlib.suppress(Exception):
        sink = _voice_diagnostic_sink.get()
        if sink is None:
            persist_event("speculative_voice", event, **fields)
        else:
            sink(event, fields)


from tldw_chatbook.Audio.voice_process_core import (  # noqa: E402
    ConsoleSpeculativeVoiceSession as _AudioVoiceSession,
    SpeculativeVoiceCompositionError as SpeculativeVoiceCompositionError,
    _Transport as _Transport,
    _SessionEffects as _SessionEffects,
    _native_fault_snapshot as _native_fault_snapshot,
    _native_snapshot_has_fault as _native_snapshot_has_fault,
    _transport_fault_type as _transport_fault_type,
)
from tldw_chatbook.Audio.voice_transcription import (  # noqa: E402
    _SerialSttWorker as _SerialSttWorker,
    _NativeStreamingStt as _NativeStreamingStt,
    _RollingWindowStt as _RollingWindowStt,
    _close_streaming_candidate as _close_streaming_candidate,
    _prepare_streaming_candidate as _prepare_streaming_candidate,
    _stt_failure_metadata as _stt_failure_metadata,
    _native_stream_realtime_capable as _native_stream_realtime_capable,
    _merge_streaming_text as _merge_streaming_text,
    _prewarm_parakeet_stream as _prewarm_parakeet_stream,
    _parakeet_mlx_audio as _parakeet_mlx_audio,
    _UNPREPARED_STREAMING_CANDIDATE as _UNPREPARED_STREAMING_CANDIDATE,
    _ROLLING_DEBOUNCE_SECONDS as _ROLLING_DEBOUNCE_SECONDS,
    _ROLLING_MIN_WINDOW_NS as _ROLLING_MIN_WINDOW_NS,
)


class ConsoleSpeculativeVoiceSession(_AudioVoiceSession):
    """In-process compatibility owner retaining original parent preparation."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("coordinator_factory", SpeculativeTurnCoordinator)
        kwargs.setdefault(
            "diagnostic_sink",
            lambda event, fields: _persist_voice_event(event, **fields),
        )
        super().__init__(**kwargs)


class _LazyHandsFreeTts:
    async def synthesize_hands_free(self, *, text: str) -> Any:
        from tldw_chatbook.TTS import get_tts_service

        service = await get_tts_service()
        return await service.synthesize_hands_free(text=text)


@dataclass(frozen=True, slots=True)
class VoicePromotionSeed:
    """Content-free session/leaf identity frozen with provider dispatch."""

    promotion_id: str
    attempt_id: str
    terminal_boundary_id: str
    origin: ConsoleSessionBindingOrigin
    expected_native_leaf_id: str | None
    expected_persisted_leaf_id: str | None
    capture_eligible_at_dispatch: bool
    capture_policy: FrozenTracePolicy | None = field(default=None, repr=False)
    next_trace_privacy_revision: int | None = None

    def __post_init__(self) -> None:
        for name in ("promotion_id", "attempt_id", "terminal_boundary_id"):
            value = getattr(self, name)
            if type(value) is not str or not value or len(value) > 512:
                raise ValueError(f"{name} must be a bounded non-empty string")
        if not isinstance(self.origin, ConsoleSessionBindingOrigin):
            raise TypeError("origin must be a ConsoleSessionBindingOrigin")
        for name in ("expected_native_leaf_id", "expected_persisted_leaf_id"):
            value = getattr(self, name)
            if value is not None and (
                type(value) is not str or not value or len(value) > 512
            ):
                raise ValueError(f"{name} must be a bounded string or None")
        if type(self.capture_eligible_at_dispatch) is not bool:
            raise TypeError("capture_eligible_at_dispatch must be a bool")
        if (
            self.capture_policy is not None
            and type(self.capture_policy) is not FrozenTracePolicy
        ):
            raise TypeError("capture_policy must be frozen")
        if self.next_trace_privacy_revision is not None and (
            type(self.next_trace_privacy_revision) is not int
            or self.next_trace_privacy_revision < 0
        ):
            raise ValueError("next_trace_privacy_revision")


@dataclass(frozen=True, slots=True)
class PreparedSpeculativeVoiceAttempt:
    """Immutable controller output for one effect-free provisional dispatch."""

    request: VoiceAttemptRequest = field(repr=False)
    frozen_session_context: ConsoleTurnExecutionContext = field(repr=False)
    promotion_seed: VoicePromotionSeed | None = field(default=None, repr=False)
    requires_citation_creation: bool = False
    requires_pre_dispatch_authority: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.request, VoiceAttemptRequest):
            raise TypeError("request must be a VoiceAttemptRequest")
        if not isinstance(self.frozen_session_context, ConsoleTurnExecutionContext):
            raise TypeError(
                "frozen_session_context must be a ConsoleTurnExecutionContext"
            )
        if self.promotion_seed is not None and not isinstance(
            self.promotion_seed,
            VoicePromotionSeed,
        ):
            raise TypeError("promotion_seed must be a VoicePromotionSeed or None")
        if type(self.requires_citation_creation) is not bool:
            raise TypeError("requires_citation_creation must be a bool")
        if type(self.requires_pre_dispatch_authority) is not bool:
            raise TypeError("requires_pre_dispatch_authority must be a bool")


@dataclass(slots=True)
class _AttemptLifecycle:
    turn_id: str
    attempt_epoch: int
    transcript: str = field(repr=False)
    started_ns: int = field(default_factory=time.monotonic_ns, repr=False)
    first_stages_seen: set[str] = field(default_factory=set, repr=False)
    first_render_submission: RenderSubmission | None = field(default=None, repr=False)
    preparation_task: asyncio.Task[None] | None = field(default=None, repr=False)
    prepared: PreparedSpeculativeVoiceAttempt | None = field(default=None, repr=False)
    attempt: VoiceAttempt | None = field(default=None, repr=False)
    speech: PhraseSpeechSequencer | None = field(default=None, repr=False)
    events: asyncio.Queue[object | None] | None = field(default=None, repr=False)
    event_worker: asyncio.Task[None] | None = field(default=None, repr=False)
    assistant_text: str = field(default="", repr=False)
    current: bool = True
    rendering: bool = False
    tool_requested: bool = False
    failed: bool = False
    preparation_failure_category: str | None = None
    draft_preservation_queued: bool = False
    preparation_cancelled: bool = False


_PrepareAttempt = Callable[..., Awaitable[PreparedSpeculativeVoiceAttempt]]
_SubmitEvent = Callable[[object], Awaitable[object]]
_PromoteWinner = Callable[..., Any]


class SpeculativeVoiceAttemptEffects:
    """Compose preparation, provider generation, phrase speech, and cleanup."""

    def __init__(
        self,
        *,
        submit_event: _SubmitEvent,
        prepare_attempt: _PrepareAttempt,
        gateway: Any,
        synthesizer: Any,
        transport: Any,
        promotion: _PromoteWinner,
        promotion_owner: Any,
        dispatch_supervisor: VoiceDispatchSupervisor,
        project_preview: Callable[[Any], None],
        clear_preview: Callable[[], None],
        submit_accepted_voice_turn: Callable[[str, ConsoleTurnExecutionContext], Any],
        classify_spoken_command: Callable[[str], bool] | None = None,
        handle_spoken_command: Callable[..., Any] | None = None,
        preserve_draft: Callable[..., Any] | None = None,
        rebuild_audio: Callable[[int, int], Any] | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(dispatch_supervisor, VoiceDispatchSupervisor):
            raise TypeError("dispatch_supervisor must be a VoiceDispatchSupervisor")
        self._submit_event = submit_event
        self._prepare_attempt = prepare_attempt
        self._gateway = gateway
        self._synthesizer = synthesizer
        self._transport = transport
        self._promotion = promotion
        self._promotion_owner = promotion_owner
        self._dispatch_supervisor = dispatch_supervisor
        self._project_preview = project_preview
        self._clear_preview = clear_preview
        self._submit_accepted_voice_turn = submit_accepted_voice_turn
        self._classify_spoken_command = classify_spoken_command or (lambda _text: False)
        self._handle_spoken_command = handle_spoken_command or (lambda **_kwargs: None)
        self._preserve_draft = preserve_draft or (lambda **_kwargs: None)
        self._rebuild_audio = rebuild_audio or (lambda _old, _epoch: None)
        self._clock = clock
        self._cleanup = AttemptCleanupManager(dispatch_supervisor)
        self._attempts: dict[int, _AttemptLifecycle] = {}
        self._background: set[asyncio.Task[Any]] = set()
        self._audio_safety_path = AcousticSafetyPath.WARMING
        self._fenced = False

    @property
    def assistant_rendering(self) -> bool:
        """Return whether current assistant PCM may still reach the device."""

        return any(item.current and item.rendering for item in self._attempts.values())

    def final_render_submission(self, attempt_epoch: int) -> RenderSubmission | None:
        """Read the final speech target synchronously on the voice owner."""
        lifecycle = self._attempts.get(attempt_epoch)
        if not self._is_epoch_current(attempt_epoch) or lifecycle.speech is None:
            return None
        return lifecycle.speech.final_submission

    def first_render_submission(self, attempt_epoch: int) -> RenderSubmission | None:
        """Read the first accepted speech target synchronously on the voice owner."""

        lifecycle = self._attempts.get(attempt_epoch)
        if not self._is_epoch_current(attempt_epoch) or lifecycle is None:
            return None
        return lifecycle.first_render_submission

    def voice_dispatch_quarantined(self) -> bool:
        return self._dispatch_supervisor.is_quarantined

    def publish_audio_capability(self, path: AcousticSafetyPath) -> None:
        """Project whether playback-period speech admission is available."""

        if not isinstance(path, AcousticSafetyPath):
            raise TypeError("audio safety path must be an AcousticSafetyPath")
        self._audio_safety_path = path
        for lifecycle in self._attempts.values():
            if lifecycle.current and lifecycle.rendering:
                self._project_lifecycle(
                    lifecycle,
                    status=self._playback_status(),
                )

    def dispatch_attempt(
        self,
        *,
        turn_id: str,
        attempt_epoch: int,
        transcript: str,
    ) -> None:
        if self._fenced:
            raise RuntimeError("speculative voice effects are fenced")
        _persist_voice_event("attempt_dispatched", status="started")
        lifecycle = _AttemptLifecycle(
            turn_id,
            attempt_epoch,
            transcript,
            started_ns=self._clock(),
        )
        self._attempts[attempt_epoch] = lifecycle
        task = asyncio.create_task(
            self._prepare(lifecycle),
            name=f"console-voice-prepare-{attempt_epoch}",
        )
        lifecycle.preparation_task = task
        self._own(task)

    async def _prepare(self, lifecycle: _AttemptLifecycle) -> None:
        try:
            prepared = await self._prepare_attempt(
                turn_id=lifecycle.turn_id,
                attempt_epoch=lifecycle.attempt_epoch,
                transcript=lifecycle.transcript,
            )
            if not isinstance(prepared, PreparedSpeculativeVoiceAttempt):
                raise TypeError("voice preparation returned an invalid result")
            if prepared.request.attempt_epoch != lifecycle.attempt_epoch:
                raise ValueError("voice preparation changed the attempt epoch")
            if not lifecycle.current or self._fenced:
                trace = prepared.request.provisional_trace_attempt
                if trace is not None:
                    self._gateway.abandon_provisional_voice_trace(trace)
                return
            lifecycle.prepared = prepared
            _persist_voice_event(
                "attempt_prepared",
                status=(
                    "ordinary_handoff"
                    if prepared.requires_pre_dispatch_authority
                    else "provisional"
                ),
            )
            await self._submit_event(
                AttemptDispatchPrepared(
                    lifecycle.attempt_epoch,
                    prepared.frozen_session_context,
                    prepared.request.prepared,
                    prepared.requires_citation_creation,
                    prepared.requires_pre_dispatch_authority,
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if lifecycle.current and not self._fenced and not lifecycle.failed:
                category = voice_failure_category(exc)
                if category == "stale":
                    return
                lifecycle.failed = True
                lifecycle.preparation_failure_category = category
                _persist_voice_event(
                    "attempt_prepare_failed",
                    status="failed",
                    error_category=lifecycle.preparation_failure_category,
                    exception_type=type(exc).__name__,
                )
                await self._submit_event(
                    ProviderAttemptFailed(
                        lifecycle.attempt_epoch,
                        lifecycle.preparation_failure_category,
                    )
                )

    def start_prepared_attempt(self, attempt_epoch: int) -> None:
        lifecycle = self._attempts.get(attempt_epoch)
        if (
            lifecycle is None
            or not lifecycle.current
            or lifecycle.prepared is None
            or lifecycle.attempt is not None
        ):
            raise RuntimeError("prepared voice attempt is unavailable")
        lifecycle.events = asyncio.Queue()
        speech = PhraseSpeechSequencer(
            epoch=attempt_epoch,
            synthesizer=self._synthesizer,
            sink=self._transport,
            on_playback_started=self._on_playback_started,
            on_failed=self._on_tts_failed,
            on_first_eligible_phrase=self._on_first_eligible_phrase,
            on_first_synthesis_complete=self._on_first_synthesis_complete,
        )
        lifecycle.speech = speech
        _persist_voice_event("provider_generation_started", status="started")
        attempt = VoiceAttempt(
            request=lifecycle.prepared.request,
            gateway=self._gateway,
            is_epoch_current=self._is_epoch_current,
            on_delta=self._on_delta,
            on_tool_request=self._on_tool_request,
            on_failed=self._on_provider_failed,
        )
        lifecycle.attempt = attempt

        async def cancel_speech() -> None:
            await speech.cancel(attempt_epoch)
            await speech.wait_for_cleanup()

        attempt.register_tts_canceller(cancel_speech)
        worker = asyncio.create_task(
            self._run_attempt_events(lifecycle),
            name=f"console-voice-events-{attempt_epoch}",
        )
        lifecycle.event_worker = worker
        self._own(worker)
        runner = attempt.start()
        runner.add_done_callback(
            lambda _task, owned=lifecycle: self._on_generation_exit(owned)
        )

    def _is_epoch_current(self, attempt_epoch: int) -> bool:
        lifecycle = self._attempts.get(attempt_epoch)
        return bool(not self._fenced and lifecycle is not None and lifecycle.current)

    def _queue_attempt_event(self, attempt_epoch: int, event: object) -> None:
        lifecycle = self._attempts.get(attempt_epoch)
        if not self._is_epoch_current(attempt_epoch) or lifecycle is None:
            return
        queue = lifecycle.events
        if queue is not None:
            queue.put_nowait(event)

    def _on_delta(self, event: VoiceAttemptDelta) -> None:
        if event.text:
            self._record_first_stage(event.attempt_epoch, "provider_delta")
        self._queue_attempt_event(event.attempt_epoch, event)

    def _on_first_eligible_phrase(self, attempt_epoch: int) -> None:
        self._record_first_stage(attempt_epoch, "eligible_phrase")

    def _on_first_synthesis_complete(self, attempt_epoch: int) -> None:
        self._record_first_stage(attempt_epoch, "synthesis_complete")

    def observe_render_receipt(
        self,
        attempt_epoch: int,
        boundary: RenderBoundary,
    ) -> None:
        """Record the first exact paired callback receipt for a live attempt."""

        lifecycle = self._attempts.get(attempt_epoch)
        if (
            lifecycle is None
            or lifecycle.first_render_submission is None
            or not isinstance(boundary, RenderBoundary)
        ):
            return
        submission = lifecycle.first_render_submission
        if (
            boundary.generation,
            boundary.output_epoch,
            boundary.submission_id,
        ) != (
            submission.generation,
            submission.output_epoch,
            submission.submission_id,
        ):
            return
        dac_started_ns = boundary.ended_ns - _RENDER_FRAME_DURATION_NS
        latency_ns = dac_started_ns - lifecycle.started_ns
        self._record_first_stage(
            attempt_epoch,
            "render_receipt",
            latency_ms=latency_ns // 1_000_000 if latency_ns >= 0 else None,
        )

    def _record_first_stage(
        self,
        attempt_epoch: int,
        phase: str,
        *,
        latency_ms: int | None = None,
    ) -> None:
        lifecycle = self._attempts.get(attempt_epoch)
        if (
            phase not in _VOICE_FIRST_STAGES
            or lifecycle is None
            or not self._is_epoch_current(attempt_epoch)
            or lifecycle.failed
            or phase in lifecycle.first_stages_seen
        ):
            return
        lifecycle.first_stages_seen.add(phase)
        duration_ms = max(0, self._clock() - lifecycle.started_ns) // 1_000_000
        _persist_voice_event(
            "voice_first_stage",
            phase=phase,
            status="ok",
            duration_ms=duration_ms,
            **({"latency_ms": latency_ms} if latency_ms is not None else {}),
        )

    def _on_tool_request(self, event: VoiceAttemptToolRequest) -> None:
        lifecycle = self._attempts.get(event.attempt_epoch)
        if lifecycle is not None:
            lifecycle.tool_requested = True
        self._queue_attempt_event(event.attempt_epoch, event)

    def _on_provider_failed(self, event: ProviderAttemptFailed) -> None:
        _persist_voice_event(
            "provider_generation_failed",
            status="failed",
            exception_type=event.error_class,
        )
        lifecycle = self._attempts.get(event.attempt_epoch)
        if lifecycle is not None:
            lifecycle.failed = True
        self._queue_attempt_event(event.attempt_epoch, event)

    def _on_generation_exit(self, lifecycle: _AttemptLifecycle) -> None:
        queue = lifecycle.events
        if queue is None:
            return
        if lifecycle.current and not lifecycle.tool_requested and not lifecycle.failed:
            _persist_voice_event("provider_generation_completed", status="ok")
            queue.put_nowait(AttemptGenerationCompleted(lifecycle.attempt_epoch, ""))
        else:
            queue.put_nowait(None)

    async def _run_attempt_events(self, lifecycle: _AttemptLifecycle) -> None:
        queue = lifecycle.events
        speech = lifecycle.speech
        if queue is None or speech is None:
            return
        while True:
            event = await queue.get()
            try:
                if event is None:
                    return
                if not lifecycle.current or self._fenced:
                    continue
                if isinstance(event, VoiceAttemptDelta):
                    lifecycle.assistant_text += event.text
                    await self._submit_event(
                        AttemptOutputDelta(event.attempt_epoch, event.text)
                    )
                    await speech.feed(event.attempt_epoch, event.text)
                    continue
                if isinstance(event, VoiceAttemptToolRequest):
                    await self._submit_event(event)
                    continue
                if isinstance(event, ProviderAttemptFailed):
                    await self._submit_event(event)
                    return
                if isinstance(event, AttemptGenerationCompleted):
                    attempt = lifecycle.attempt
                    if attempt is None:
                        return
                    snapshot = attempt.snapshot
                    await self._submit_event(
                        AttemptGenerationCompleted(
                            lifecycle.attempt_epoch,
                            snapshot.response_text,
                        )
                    )
                    await speech.finish(lifecycle.attempt_epoch)
                    return
            finally:
                queue.task_done()

    def _on_playback_started(self, attempt_epoch: int) -> None:
        lifecycle = self._attempts.get(attempt_epoch)
        if lifecycle is not None and lifecycle.current:
            if (
                lifecycle.first_render_submission is None
                and lifecycle.speech is not None
            ):
                lifecycle.first_render_submission = lifecycle.speech.first_submission
            _persist_voice_event("playback_started", status="started")
            lifecycle.rendering = True
            self._project_lifecycle(lifecycle, status=self._playback_status())
            self._post(AttemptPlaybackStarted(attempt_epoch))

    def _on_tts_failed(self, attempt_epoch: int, error_class: str) -> None:
        _persist_voice_event(
            "tts_failed",
            status="failed",
            exception_type=error_class,
        )
        if error_class == "output_rejected":
            self._post(AttemptPlaybackFailed(attempt_epoch))
        else:
            self._post(AttemptTtsFailed(attempt_epoch, error_class))

    def fence_attempt(self, attempt_epoch: int) -> None:
        lifecycle = self._attempts.get(attempt_epoch)
        if lifecycle is None or not lifecycle.current:
            return
        lifecycle.first_render_submission = None
        if lifecycle.speech is not None:
            lifecycle.speech.fence(attempt_epoch)
        else:
            self._transport.fence_output()
        lifecycle.current = False
        lifecycle.rendering = False
        if lifecycle.attempt is not None:
            lifecycle.attempt.invalidate()

    def cancel_attempt(self, attempt_epoch: int) -> Awaitable[AttemptCleanupOutcome]:
        lifecycle = self._attempts.get(attempt_epoch)
        if lifecycle is None:
            return self._completed_cleanup()
        lifecycle.preparation_cancelled = True
        queue = lifecycle.events
        if queue is not None:
            queue.put_nowait(None)
        if lifecycle.attempt is not None:
            result = self._cleanup.cancel(lifecycle.attempt)
        else:
            result = asyncio.create_task(self._cancel_preparation(lifecycle))
        result.add_done_callback(
            lambda _done, epoch=attempt_epoch: self._attempts.pop(epoch, None)
        )
        return result

    async def _cancel_preparation(
        self,
        lifecycle: _AttemptLifecycle,
    ) -> AttemptCleanupOutcome:
        task = lifecycle.preparation_task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return AttemptCleanupOutcome.CLEAN

    @staticmethod
    def _completed_cleanup() -> asyncio.Future[AttemptCleanupOutcome]:
        result = asyncio.get_running_loop().create_future()
        result.set_result(AttemptCleanupOutcome.CLEAN)
        return result

    def abort_output(self, attempt_epoch: int) -> Awaitable[Any]:
        lifecycle = self._attempts.get(attempt_epoch)
        if lifecycle is not None:
            lifecycle.first_render_submission = None
        if lifecycle is not None and lifecycle.speech is not None:
            lifecycle.rendering = False
            lifecycle.speech.fence(attempt_epoch)
            return lifecycle.speech.cancel(attempt_epoch)
        if lifecycle is not None and lifecycle.current:
            self._transport.fence_output()
        return self._completed_cleanup()

    def publish_preview(self, attempt_epoch: int, delta: str) -> None:
        del delta
        lifecycle = self._attempts.get(attempt_epoch)
        if lifecycle is None or not lifecycle.current:
            return
        self._project_lifecycle(
            lifecycle,
            status=self._playback_status() if lifecycle.rendering else "responding",
        )

    def _playback_status(self) -> str:
        if self._audio_safety_path is AcousticSafetyPath.WARMING:
            return "aec warming"
        if self._audio_safety_path is AcousticSafetyPath.HALF_DUPLEX:
            return "half duplex"
        return "speaking"

    def _project_lifecycle(
        self,
        lifecycle: _AttemptLifecycle,
        *,
        status: str,
    ) -> None:
        from tldw_chatbook.Widgets.Console import VoicePreviewProjection

        self._project_preview(
            VoicePreviewProjection(
                turn_id=lifecycle.turn_id,
                attempt_epoch=lifecycle.attempt_epoch,
                user_text=lifecycle.transcript,
                assistant_text=lifecycle.assistant_text,
                status=status,
            )
        )

    def clear_preview(self, _attempt_epoch: int) -> None:
        self._clear_preview()

    def preserve_draft(self, **kwargs: Any) -> Any:
        if kwargs.get("reason") == "provider_failed":
            lifecycle = next(
                (
                    item
                    for item in reversed(self._attempts.values())
                    if item.turn_id == kwargs.get("turn_id")
                ),
                None,
            )
            if (
                lifecycle is not None
                and lifecycle.preparation_failure_category is not None
            ):
                if lifecycle.draft_preservation_queued:
                    return None
                lifecycle.draft_preservation_queued = True

                # The reducer normally fences a failed attempt before preserving
                # it. Guard against cancellation/replacement, not that fence.
                def is_current() -> bool:
                    return (
                        not self._fenced
                        and not lifecycle.preparation_cancelled
                        and self._attempts.get(lifecycle.attempt_epoch) is lifecycle
                        and max(self._attempts) == lifecycle.attempt_epoch
                    )

                kwargs.update(
                    preparation_failure_category=lifecycle.preparation_failure_category,
                    is_current=is_current,
                )
        return self._preserve_draft(**kwargs)

    def promote(
        self,
        *,
        turn_id: str,
        attempt_epoch: int,
        transcript: str,
        assistant_text: str,
        terminal_boundary_ns: int | None,
    ) -> Any:
        lifecycle = self._attempts.get(attempt_epoch)
        if lifecycle is None or lifecycle.attempt is None or lifecycle.prepared is None:
            raise RuntimeError("winning voice attempt is unavailable")
        snapshot = lifecycle.attempt.snapshot
        lifecycle.current = False
        lifecycle.rendering = False
        self._clear_preview()
        try:
            return self._promotion(
                owner=self._promotion_owner,
                prepared=lifecycle.prepared,
                snapshot=snapshot,
                turn_id=turn_id,
                attempt_epoch=attempt_epoch,
                transcript=transcript,
                assistant_text=assistant_text,
                terminal_boundary_ns=terminal_boundary_ns,
            )
        finally:
            self._attempts.pop(attempt_epoch, None)

    def classify_spoken_command(self, transcript: str) -> bool:
        return self._classify_spoken_command(transcript)

    def handle_spoken_command(self, **kwargs: Any) -> Any:
        return self._handle_spoken_command(**kwargs)

    def rebuild_audio(self, old_clock_generation: int, rebuild_epoch: int) -> Any:
        return self._rebuild_audio(old_clock_generation, rebuild_epoch)

    def submit_accepted_voice_turn(
        self,
        exact_user_text: str,
        frozen_session_context: ConsoleTurnExecutionContext,
    ) -> Any:
        return self._submit_accepted_voice_turn(
            exact_user_text,
            frozen_session_context,
        )

    def fence_all(self) -> None:
        if self._fenced:
            return
        self._fenced = True
        for epoch in tuple(self._attempts):
            self.fence_attempt(epoch)

    async def close(self) -> None:
        self.fence_all()
        cleanups = [self.cancel_attempt(epoch) for epoch in tuple(self._attempts)]
        if cleanups:
            await asyncio.gather(*cleanups, return_exceptions=True)
        background = tuple(self._background)
        if background:
            await asyncio.gather(*background, return_exceptions=True)

    def _post(self, event: object) -> None:
        if self._fenced:
            return
        self._own(asyncio.create_task(self._submit_event(event)))

    def _own(self, task: asyncio.Task[Any]) -> None:
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        task.add_done_callback(_consume_task_result)


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    with contextlib.suppress(BaseException):
        task.result()


def create_console_speculative_voice_session(
    *,
    app_instance: Any,
    view: Any,
    promotion_owner: Any,
    dispatch_supervisor: VoiceDispatchSupervisor,
    project_preview: Callable[[Any], None],
    clear_preview: Callable[[], None],
    on_runtime_failure: Callable[[BaseException], None] | None = None,
    worker: Any = None,
    process_supervisor: Any = None,
    entry_current: Callable[[], bool] | None = None,
) -> ConsoleVoiceProcess:
    """Build one qualified view-local pipeline from existing app services."""

    from tldw_chatbook.Chat.console_voice_input import resolve as resolve_stt
    from tldw_chatbook.Chat.console_voice_promotion import (
        VoicePromotionContext,
        VoicePromotionOwner,
        VoiceWinningPromotion,
    )
    from tldw_chatbook.Chat.console_voice_settings import (
        pipeline_aec_enabled,
        response_eagerness_ms,
    )
    from tldw_chatbook.Chat.provider_usage import ProviderUsage
    from tldw_chatbook.config import get_cli_setting

    if type(promotion_owner) is not VoicePromotionOwner:
        raise SpeculativeVoiceCompositionError("voice promotion owner is unavailable")
    if not isinstance(dispatch_supervisor, VoiceDispatchSupervisor):
        raise SpeculativeVoiceCompositionError(
            "voice dispatch supervisor is unavailable"
        )
    if not callable(project_preview) or not callable(clear_preview):
        raise SpeculativeVoiceCompositionError(
            "voice preview projection is unavailable"
        )

    ensure_controller = getattr(view, "_ensure_console_chat_controller", None)
    if not callable(ensure_controller):
        raise SpeculativeVoiceCompositionError("voice app services are unavailable")
    controller = ensure_controller()
    prepare_attempt = getattr(controller, "prepare_speculative_voice_attempt", None)
    accepted_handoff = getattr(controller, "submit_accepted_voice_turn", None)
    gateway = getattr(controller, "provider_gateway", None)
    if (
        not callable(prepare_attempt)
        or not callable(accepted_handoff)
        or gateway is None
    ):
        raise SpeculativeVoiceCompositionError(
            "voice controller services are unavailable"
        )

    stt_config = resolve_stt()
    if stt_config is None:
        raise SpeculativeVoiceCompositionError(
            "no qualified speech recognizer is available"
        )
    configured_eagerness = get_cli_setting(
        "dictation",
        "response_eagerness_ms",
        700,
    )
    eagerness_ms = response_eagerness_ms(
        {"dictation": {"response_eagerness_ms": configured_eagerness}}
    )
    configured_aec = get_cli_setting(
        "dictation",
        "pipeline_aec_enabled",
        True,
    )
    aec_enabled = pipeline_aec_enabled(
        {"dictation": {"pipeline_aec_enabled": configured_aec}}
    )

    def promote_winner(
        *,
        owner: Any,
        prepared: PreparedSpeculativeVoiceAttempt,
        snapshot: Any,
        transcript: str,
        assistant_text: str,
        terminal_boundary_ns: int | None,
        on_claim: Callable[[Any], None] | None = None,
        **_kwargs: Any,
    ) -> Any:
        if owner is not promotion_owner:
            raise RuntimeError("winning voice authority is unavailable")
        seed = prepared.promotion_seed
        if seed is None:
            raise RuntimeError("winning voice identity is unavailable")
        usage: ProviderUsage | None = None
        resolution = prepared.request.resolution
        for payload in snapshot.usage_payloads:
            item = ProviderUsage.from_provider_payload(
                payload,
                provider=resolution.provider,
                model=resolution.model or "",
            )
            if item is not None:
                usage = item if usage is None else usage.plus(item)
        context = VoicePromotionContext(
            promotion_id=seed.promotion_id,
            attempt_id=seed.attempt_id,
            origin=seed.origin,
            expected_native_leaf_id=seed.expected_native_leaf_id,
            expected_persisted_leaf_id=seed.expected_persisted_leaf_id,
            user_text=transcript,
            assistant_text=assistant_text,
            usage_json=usage.to_json() if usage is not None else None,
            terminal_boundary_id=seed.terminal_boundary_id,
            capture_eligible_at_dispatch=seed.capture_eligible_at_dispatch,
        )
        from tldw_chatbook.Chat.console_voice_trace_gateway import (
            VoiceTraceImportContext,
        )
        from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService

        registry = gateway.provisional_trace_registry
        persistence = store.persistence

        def trace_context(_context, commit):
            if seed.capture_policy is None:
                raise RuntimeError("winning trace policy is unavailable")
            return VoiceTraceImportContext(
                import_id=seed.promotion_id,
                conversation_id=commit.conversation_id,
                user_message_id=commit.user_message_id,
                user_revision_id=commit.user_revision_id,
                assistant_message_id=commit.assistant_message_id,
                assistant_revision_id=commit.assistant_revision_id,
                turn_id=commit.user_message_id,
                run_id=f"voice:{seed.attempt_id}",
                policy=seed.capture_policy,
            )

        def import_trace(manifest, envelopes, import_context):
            # VoiceWinningPromotion runs this after pair publication through
            # its existing off-thread runner. Trace failure cannot undo audio.
            return ConsoleTraceService(
                repository=persistence.console_trace_repository,
            ).import_provisional_voice_trace(
                persistence.db,
                registry,
                manifest,
                envelopes,
                import_context,
            )

        winner = VoiceWinningPromotion(
            promotion_owner,
            trace_registry=registry,
            trace_context_factory=trace_context,
            trace_importer=import_trace,
        )

        def claimed(claim):
            if seed.next_trace_privacy_revision is not None:
                try:
                    store.consume_session_next_trace_privacy(
                        seed.origin.session_id,
                        expected_next_revision=seed.next_trace_privacy_revision,
                    )
                except Exception:
                    _persist_voice_event("privacy_claim_consumption_failed")
            if on_claim is not None:
                on_claim(claim)

        return winner.promote(context, snapshot, on_claim=claimed)

    from uuid import uuid4
    from tldw_chatbook.Chat.console_voice_process import (
        ConsoleVoiceProcess,
        bootstrap_record,
    )
    from tldw_chatbook.Chat.console_runtime import ensure_console_runtime

    runtime = ensure_console_runtime(app_instance, view=view)
    supervisor = process_supervisor or runtime.voice_process_supervisor
    bootstrap = bootstrap_record(
        generation=view._hands_free._qualified_voice_generation,
        request_id=uuid4().hex,
        stt_provider=stt_config.provider,
        stt_model=stt_config.model,
        language=stt_config.language,
        stt_device=get_cli_setting("transcription", "device", None),
        stt_compute_type=get_cli_setting("transcription", "compute_type", None),
        stt_precision=get_cli_setting("transcription", "parakeet_precision", None),
        response_eagerness_ms=eagerness_ms,
        aec_enabled=aec_enabled,
        vad_aggressiveness=get_cli_setting("dictation", "vad_aggressiveness", 2),
        vad_preroll_ms=get_cli_setting("dictation", "vad_preroll_ms", 240),
    )

    async def preflight():
        # The mounted path has already validated the original readiness stamp.
        # A direct factory caller obtains the same typed controller readiness.
        nonlocal entry_current
        if entry_current is None:
            stamp = await controller.validate_speculative_voice_entry()

            def entry_current():
                return controller.is_speculative_voice_entry_current(stamp)

        return entry_current() is True

    store = controller.store
    session_id = store.active_session_id
    owning_session = next(
        (item for item in store.sessions() if item.id == session_id), None
    )
    owning_epoch = store.active_session_epoch()

    def preserve_original_draft(
        text, *, preparation_failure_category=None, exception_type=None, is_current=None
    ):
        if owning_session is None or not any(
            item is owning_session for item in store.sessions()
        ):
            return False
        if is_current is not None:
            if (
                not is_current()
                or not getattr(view, "is_mounted", False)
                or store.active_session_id != session_id
                or store.active_session_epoch() != owning_epoch
                or view._hands_free._qualified_voice_generation
                != bootstrap.header["generation"]
            ):
                return False
            view._append_to_console_draft(text)
            if preparation_failure_category is not None:
                from .console_voice_preflight import voice_failure_message

                _persist_voice_event(
                    "attempt_prepare_failed",
                    status="failed",
                    error_category=preparation_failure_category,
                    exception_type=exception_type,
                )
                app_instance.notify(
                    voice_failure_message(preparation_failure_category),
                    severity="error",
                )
            return True
        if (
            getattr(view, "is_mounted", False)
            and store.active_session_id == session_id
            and view._hands_free._qualified_voice_generation
            == bootstrap.header["generation"]
        ):
            view._append_to_console_draft(text)
        else:
            current = store.session_draft(session_id)
            store.set_session_draft(
                session_id, current + view._draft_addition(current, text)
            )
        return True

    engine = ConsoleVoiceProcess(
        supervisor.device_lease,
        bootstrap=bootstrap,
        preflight=preflight,
        current=lambda: (
            entry_current is not None
            and entry_current()
            and getattr(view, "is_mounted", False)
            and view._console_dictation_state == "idle"
        ),
        prepare_attempt=prepare_attempt,
        gateway=gateway,
        dispatch_supervisor=dispatch_supervisor,
        promote=partial(
            promote_winner, owner=promotion_owner, terminal_boundary_ns=None
        ),
        accepted_handoff=accepted_handoff,
        project_preview=project_preview,
        clear_preview=clear_preview,
        preserve_draft=preserve_original_draft,
        on_runtime_failure=on_runtime_failure,
    )
    supervisor.retain(engine)
    return engine


__all__ = [
    "ConsoleSpeculativeVoiceSession",
    "PreparedSpeculativeVoiceAttempt",
    "SpeculativeVoiceCompositionError",
    "SpeculativeVoiceAttemptEffects",
    "VoicePromotionSeed",
    "create_console_speculative_voice_session",
]
