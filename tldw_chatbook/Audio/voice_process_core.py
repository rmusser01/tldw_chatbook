"""Audio-critical session owner without app, provider, or UI imports."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from tldw_chatbook.Audio.duplex_contracts import (
    AcousticDemotionReason,
    AcousticSafetyPath,
    AecDelayEvidence,
    AecHealth,
    AudioFrame,
    CaptureDrainError,
    DuplexMode,
    RenderBoundary,
    RenderSubmission,
    RouteKind,
)
from tldw_chatbook.Audio.native_duplex_stream import AudioShutdownUnconfirmed
from tldw_chatbook.Audio.rolling_transcript import TranscriptEngine, TranscriptRevision
from tldw_chatbook.Audio.voice_preprocessor import (
    PendingClassificationWatermark,
    VoicePreprocessor,
)
from tldw_chatbook.Audio.voice_process_types import ControlKind, VoiceTransportFailure
from tldw_chatbook.Audio.voice_turn_coordinator import (
    AdmittedSpeechFrame,
    AttemptPlaybackBoundaryKnown,
    AttemptPlaybackFailed,
    AttemptPlaybackTerminal,
    AudioRouteReady,
    AudioRouteRebuildFailed,
    AudioCapabilityChanged,
    CaptureGated,
    ControlAction,
    SpeculativeTurnCoordinator,
    SpeculativeVoiceState,
)

_PUMP_INTERVAL_SECONDS = 0.005
_MAX_LIVE_TRANSCRIPTS = 2
_MAX_NATIVE_COUNTER = 2**64 - 1


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    with contextlib.suppress(BaseException):
        task.result()


def _transport_fault_type(evidence: AecDelayEvidence | None) -> str:
    """Classify unsafe transport timing without retaining captured content."""

    if evidence is None:
        return "missing-timing"
    if evidence.status_flags:
        return "status-" + "+".join(evidence.status_flags)
    if not evidence.occupancy_bounded:
        return "buffer-overflow"
    if evidence.clock_drift:
        return "clock-drift"
    if evidence.timing_discontinuity:
        return "timestamp-discontinuity"
    return "frame-discontinuity"


def _native_fault_snapshot(
    transport: Any,
    evidence: AecDelayEvidence | None,
    *,
    observed_ns: int | None,
) -> dict[str, int]:
    """Project only validated native counters at owner-loop drain time."""

    try:
        snapshot = transport.native_counters
    except Exception:
        return {}
    if not isinstance(snapshot, dict) or not snapshot:
        return {}
    try:
        capacities = transport.buffer_capacities
        capture_capacity = capacities.capture_frames
        render_capacity = capacities.render_frames
    except Exception:
        capture_capacity = render_capacity = -1
    projected: dict[str, int] = {}
    fields = (
        ("callback_count", "native_callback_count", None),
        ("fatal_status_bits", "native_fatal_status_bits", None),
        ("capture_overflows", "native_capture_overflows", None),
        ("invalid_frames", "native_invalid_frames", None),
        ("invalid_timing", "native_invalid_timing", None),
        (
            "capture_occupancy",
            "native_capture_occupancy",
            capture_capacity,
        ),
        (
            "render_occupancy",
            "native_render_occupancy",
            render_capacity,
        ),
    )
    for native_name, log_name, capacity in fields:
        value = snapshot.get(native_name)
        if (
            type(value) is int
            and 0 <= value <= _MAX_NATIVE_COUNTER
            and (
                capacity is None
                or (
                    type(capacity) is int
                    and 0 <= capacity <= _MAX_NATIVE_COUNTER
                    and value <= capacity
                )
            )
        ):
            projected[log_name] = value
    if (
        evidence is not None
        and type(evidence.observed_ns) is int
        and type(observed_ns) is int
        and 0 <= evidence.observed_ns <= observed_ns <= _MAX_NATIVE_COUNTER
    ):
        projected["lag_ms"] = (observed_ns - evidence.observed_ns) // 1_000_000
    return projected


def _native_snapshot_has_fault(snapshot: dict[str, int]) -> bool:
    return any(
        snapshot.get(field, 0)
        for field in (
            "native_fatal_status_bits",
            "native_capture_overflows",
            "native_invalid_frames",
            "native_invalid_timing",
        )
    )


class _Transport(Protocol):
    @property
    def clock_generation(self) -> int: ...

    async def start(self) -> None: ...

    async def close(self) -> None: ...

    def request_close(self) -> None: ...

    def fence_output(self) -> None: ...

    async def abort_output(self) -> None: ...

    async def drain_capture_through(self, render_boundary_ns: int) -> Any: ...

    def notify_route_changed(self, route_kind: RouteKind) -> Awaitable[None]: ...

    def pop_capture(self) -> AudioFrame | None: ...

    def pop_render_reference(self) -> AudioFrame | None: ...

    def pop_control_event(self) -> object | None: ...

    def queue_render(self, pcm16: bytes) -> object | None: ...

    def acknowledge_capture(
        self,
        sequence: int,
        *,
        clock_generation: int,
        dsp_ok: bool,
        vad_ok: bool,
    ) -> None: ...


class _TranscriptPort(Protocol):
    turn_id: str

    def append_admitted_frame(self, frame: AudioFrame) -> None: ...

    async def seal_through(self, admitted_sequence: int) -> TranscriptRevision: ...

    async def close(self) -> None: ...


_PreprocessorFactory = Callable[..., VoicePreprocessor]
_TranscriptFactory = Callable[
    [str, Callable[[TranscriptRevision], None]],
    _TranscriptPort,
]


class SpeculativeVoiceCompositionError(RuntimeError):
    """Raised when the safe production session cannot be composed."""


class _SessionEffects:
    """Delegate policy effects while owning causal audio/transcript barriers."""

    def __init__(self, owner: "ConsoleSpeculativeVoiceSession", delegate: Any) -> None:
        self._owner = owner
        self._delegate = delegate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def dispatch_attempt(self, **kwargs: Any) -> Any:
        if self._owner._fenced:
            return None
        return self._delegate.dispatch_attempt(**kwargs)

    def start_prepared_attempt(self, *args: Any, **kwargs: Any) -> Any:
        if self._owner._fenced:
            return None
        return self._delegate.start_prepared_attempt(*args, **kwargs)

    def submit_accepted_voice_turn(self, *args: Any, **kwargs: Any) -> Any:
        if self._owner._fenced:
            raise VoiceTransportFailure("audio_transport_failed")
        return self._delegate.submit_accepted_voice_turn(*args, **kwargs)

    def fence_attempt(self, attempt_epoch: int) -> Any:
        self._owner._playback_target = None
        return self._delegate.fence_attempt(attempt_epoch)

    def promote(self, **kwargs: Any) -> Any:
        # Causal seals succeeded; publication may outlive the audio deadline.
        if self._owner._fenced:
            raise VoiceTransportFailure("audio_transport_failed")
        self._owner._playback_target = None
        return self._delegate.promote(**kwargs)

    async def drain_capture_through(self, render_boundary_ns: int) -> Any:
        return await self._owner._transport.drain_capture_through(render_boundary_ns)

    async def drain_pending_classification_through(
        self,
        render_boundary_ns: int,
        clock_generation: int,
    ) -> None:
        # Half duplex gates playback-period speech acoustically, but idle/gap
        # speech still needs capture coverage before a terminal seal can win.
        await self.drain_capture_through(render_boundary_ns)
        await self._owner._drain_pending_classification_through(
            render_boundary_ns,
            clock_generation,
        )

    async def seal_transcript_through(
        self,
        admitted_sequence: int,
    ) -> TranscriptRevision:
        return await self._owner._seal_transcript_through(admitted_sequence)


@dataclass(slots=True)
class _PlaybackTarget:
    attempt_epoch: int
    submission: RenderSubmission
    delivery_deadline_ns: int
    boundary: RenderBoundary | None = None
    terminal_sent: bool = False


class ConsoleSpeculativeVoiceSession:
    """Own the view-local audio pump and serialized speculative coordinator."""

    def __init__(
        self,
        *,
        transport: _Transport,
        preprocessor_factory: _PreprocessorFactory,
        transcript_factory: _TranscriptFactory,
        effects: Any,
        response_eagerness_ms: object = 700,
        initial_duplex_mode: DuplexMode = DuplexMode.HALF_DUPLEX,
        deferred_attempt_preparation: bool = False,
        pump_interval_seconds: float = _PUMP_INTERVAL_SECONDS,
        on_runtime_failure: Callable[[BaseException], None] | None = None,
        prepare_transcription: Callable[[], Awaitable[None]] | None = None,
        interrupt_transcription: Callable[[], Awaitable[None]] | None = None,
        close_transcription: Callable[[], Awaitable[None]] | None = None,
        coordinator_factory: Callable[
            ..., SpeculativeTurnCoordinator
        ] = SpeculativeTurnCoordinator,
        diagnostic_sink: Callable[[str, dict[str, object]], None] | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(initial_duplex_mode, DuplexMode):
            raise TypeError("initial_duplex_mode must be a DuplexMode")
        if pump_interval_seconds <= 0:
            raise ValueError("pump interval must be positive")
        self._transport = transport
        self._clock = clock
        self._playback_target: _PlaybackTarget | None = None
        self._shutdown_error: AudioShutdownUnconfirmed | None = None
        self._transcript_factory = transcript_factory
        self._delegate_effects = effects
        self._effects = _SessionEffects(self, effects)
        self._coordinator = coordinator_factory(
            effects=self._effects,
            response_eagerness_ms=response_eagerness_ms,
            initial_duplex_mode=initial_duplex_mode,
            deferred_attempt_preparation=deferred_attempt_preparation,
        )
        self._pending_classification: PendingClassificationWatermark | None = None
        self._capture_admission_pending: AudioFrame | None = None
        self._classification_release_pending = False
        self._classification_failed_generation: int | None = None
        self._classification_waiters: set[asyncio.Future[None]] = set()
        self._preprocessor = preprocessor_factory(
            on_admitted_frame=self._on_admitted_frame,
            on_processed=self._on_processed,
            on_classification_changed=self._on_classification_changed,
        )
        self._pump_interval_seconds = pump_interval_seconds
        self._on_runtime_failure = on_runtime_failure
        self._prepare_transcription = prepare_transcription
        self._interrupt_transcription = interrupt_transcription
        self._close_transcription = close_transcription
        self._diagnostic_sink = diagnostic_sink
        self._admitted: list[AudioFrame] = []
        self._transcripts: list[_TranscriptPort] = []
        self._transcript_retirements: set[asyncio.Task[None]] = set()
        self._render_references: deque[AudioFrame] = deque()
        self._advertised_capability: (
            tuple[DuplexMode, AecHealth, AcousticSafetyPath] | None
        ) = None
        self._render_activity_active = False
        self._reported_transport_fault_generation: int | None = None
        self._initial_transport_recovery_attempted = False
        self._started = False
        self._prepared = False
        self._prepare_task: asyncio.Task[None] | None = None
        self._startup_task: asyncio.Task[None] | None = None
        self._fenced = False
        self._closed = False
        self._pump_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._resource_cleanup_task: asyncio.Task[None] | None = None
        self._transport_failure_report_task: asyncio.Task[None] | None = None
        self._close_lock = asyncio.Lock()

    @property
    def state(self) -> object:
        """Expose the coordinator state for the existing hands-free facade."""

        return self._coordinator.snapshot.state

    @property
    def coordinator(self) -> SpeculativeTurnCoordinator:
        """Return the serialized coordinator for qualification observations."""

        return self._coordinator

    def _persist_voice_event(self, event: str, **fields: object) -> None:
        if self._diagnostic_sink is not None:
            with contextlib.suppress(Exception):
                self._diagnostic_sink(event, fields)

    async def prepare(self) -> None:
        """Prepare transcription without opening audio capture."""

        if self._fenced:
            raise RuntimeError("speculative voice session is closed")
        if self._prepare_task is None:
            self._prepare_task = asyncio.create_task(
                self._prepare_once(), name="console-speculative-voice-prepare"
            )
            self._prepare_task.add_done_callback(_consume_task_result)
        await asyncio.shield(self._prepare_task)

    async def _prepare_once(self) -> None:
        if self._fenced:
            raise RuntimeError("speculative voice session is closed")
        if self._prepare_transcription is not None:
            await self._prepare_transcription()
        if self._fenced:
            raise RuntimeError("speculative voice session is closed")
        self._prepared = True

    async def start_audio(self) -> None:
        """Start capture after the parent grants its separate permission."""

        if self._fenced:
            raise RuntimeError("speculative voice session is closed")
        if self._startup_task is None:
            self._startup_task = asyncio.create_task(
                self._start_audio_once(), name="console-speculative-voice-start"
            )
            self._startup_task.add_done_callback(_consume_task_result)
        await asyncio.shield(self._startup_task)

    async def _start_audio_once(self) -> None:
        try:
            await self.prepare()
            if self._fenced:
                raise RuntimeError("speculative voice session is closed")
            self._started = True
            await self._coordinator.start()
            await self._transport.start()
            if self._fenced:
                await self._transport.close()
                raise RuntimeError("speculative voice session is closed")
        except BaseException:
            if not self._fenced:
                self._fence_callbacks()
            await self._close_once()
            raise
        self._persist_voice_event("session_ready", status="ok")
        self._pump_task = asyncio.create_task(
            self._pump_audio(),
            name="console-speculative-voice-audio",
        )

    async def enter(self, *, capture_live: bool) -> None:
        """Start one duplex session; an existing legacy capture is never adopted."""

        del capture_live
        if self._fenced:
            raise RuntimeError("speculative voice session is closed")
        await self.start_audio()

    def fence_audio_admission(self) -> None:
        """Stop native capture and start its deadline from the facade's thread."""
        self._transport.request_close()

    def fence_and_close(self, reason: ControlKind) -> asyncio.Future[None]:
        """Fence callbacks synchronously, then return asynchronous cleanup."""

        if not isinstance(reason, ControlKind):
            raise TypeError("voice close reason must be a ControlKind")
        if not self._fenced:
            self._fence_callbacks()
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_with_reason(reason),
                name="console-speculative-voice-close",
            )
            self._close_task.add_done_callback(_consume_task_result)
        return asyncio.shield(self._close_task)

    async def submit(self, event: object) -> bool:
        """Submit one component event unless the view fence already closed."""

        if self._fenced:
            return False
        await self._coordinator.submit(event)
        observe = getattr(self._delegate_effects, "observe_recovery_currentness", None)
        if callable(observe):
            try:
                observe(self._coordinator.snapshot)
            except Exception:
                self._fail_transport()
        return True

    async def process_pending_audio(self) -> bool:
        """Process one pending capture/control item for deterministic qualification."""

        if self._fenced:
            return False
        control = self._transport.pop_control_event()
        if control is not None:
            await self.submit(control)
            return True
        await self._observe_playback_target()
        capture = self._transport.pop_capture()
        if capture is None:
            native_snapshot = _native_fault_snapshot(
                self._transport,
                None,
                observed_ns=None,
            )
            try:
                generation = self._transport.clock_generation
            except Exception:
                generation = None
            if (
                type(generation) is int
                and generation >= 0
                and self._reported_transport_fault_generation != generation
                and _native_snapshot_has_fault(native_snapshot)
            ):
                self._reported_transport_fault_generation = generation
                self._persist_voice_event(
                    "audio_transport_fault",
                    status="failed",
                    phase="capture",
                    operation="drain_snapshot",
                    result_type="missing-timing",
                    **native_snapshot,
                )
                self._fail_transport()
                return True
            return False

        self._observe_first_render_receipt(capture)

        while True:
            reference = self._transport.pop_render_reference()
            if reference is None:
                break
            self._render_references.append(reference)

        render_frames: list[AudioFrame] = []
        required_reference = capture.render_reference_sequence
        if required_reference is not None:
            while self._render_references:
                reference = self._render_references[0]
                if (
                    reference.clock_generation == capture.clock_generation
                    and reference.sequence > required_reference
                ):
                    break
                render_frames.append(self._render_references.popleft())

        assistant_rendering = capture.assistant_rendering
        if assistant_rendering is None:  # legacy fake frames only
            assistant_rendering = self._assistant_rendering() and bool(render_frames)
        if assistant_rendering != self._render_activity_active:
            self._render_activity_active = assistant_rendering
            self._persist_voice_event(
                "render_activity_changed",
                status="active" if assistant_rendering else "paused",
            )
        self._admitted.clear()
        if (
            capture.discontinuity
            and self._reported_transport_fault_generation != capture.clock_generation
        ):
            self._reported_transport_fault_generation = capture.clock_generation
            self._persist_voice_event(
                "audio_transport_fault",
                status="failed",
                phase="capture",
                operation="drain_snapshot",
                result_type=_transport_fault_type(capture.delay_evidence),
                **_native_fault_snapshot(
                    self._transport,
                    capture.delay_evidence,
                    observed_ns=self._clock(),
                ),
            )
        if (
            capture.discontinuity
            and not self._initial_transport_recovery_attempted
            and not assistant_rendering
            and self._coordinator.snapshot.turn_id is None
            and _transport_fault_type(capture.delay_evidence)
            == "timestamp-discontinuity"
        ):
            self._initial_transport_recovery_attempted = True
            self._persist_voice_event(
                "audio_transport_recovery_requested",
                status="retrying",
                phase="capture",
                result_type="timestamp-discontinuity",
            )
            await self._transport.notify_route_changed(RouteKind.DUPLEX)
            return True
        if capture.discontinuity:
            self._fail_transport()
            return True
        self._capture_admission_pending = capture
        try:
            await self._preprocessor.process_capture(
                capture,
                render_frames=render_frames,
                assistant_rendering=assistant_rendering,
            )
            capability = (
                self._preprocessor.mode,
                self._preprocessor.health,
                self._preprocessor.safety.path,
            )
            if capability != self._advertised_capability:
                self._advertised_capability = capability
                demotion_reason = self._preprocessor.safety.demotion_reason
                self._persist_voice_event(
                    "audio_capability_changed",
                    status=capability[2].value,
                    phase=capability[0].value,
                    result_type=capability[1].value,
                    **(
                        {"error_category": demotion_reason.value}
                        if demotion_reason is not None
                        else {}
                    ),
                )
                publish_capability = getattr(
                    self._delegate_effects,
                    "publish_audio_capability",
                    None,
                )
                if callable(publish_capability):
                    publish_capability(capability[2])
                await self._coordinator.submit(
                    AudioCapabilityChanged(
                        capture.clock_generation,
                        capability[0],
                        capability[1],
                        capability[2],
                    )
                )
            admitted = tuple(self._admitted)
            self._admitted.clear()
            for frame in admitted:
                await self._observe_playback_target()
                await self._accept_admitted_frame(
                    frame,
                    assistant_rendering=(
                        assistant_rendering
                        if frame.assistant_rendering is None
                        else frame.assistant_rendering
                    ),
                )
            if (
                not admitted
                and assistant_rendering
                and self._preprocessor.mode is DuplexMode.HALF_DUPLEX
            ):
                await self._coordinator.submit(
                    CaptureGated(
                        capture.sequence,
                        capture.started_ns,
                        capture.ended_ns,
                        capture.clock_generation,
                    )
                )
        except BaseException:
            self._release_pending_classification(failed=True)
            raise
        else:
            if self._classification_release_pending:
                safety = self._preprocessor.safety
                classification_failed = (
                    not safety.admission_open
                    and safety.demotion_reason
                    is not AcousticDemotionReason.CORRELATED_RENDER
                )
                self._release_pending_classification(failed=classification_failed)
        finally:
            self._capture_admission_pending = None
            self._wake_classification_waiters()
        return True

    def _observe_first_render_receipt(self, capture: AudioFrame) -> None:
        """Project exact nonfaulted callback evidence from the original capture."""

        boundary = capture.committed_render
        epoch = self._coordinator.snapshot.current_attempt_epoch
        if (
            boundary is None
            or capture.discontinuity
            or epoch is None
            or capture.clock_generation != boundary.generation
        ):
            return
        first_submission = getattr(
            self._delegate_effects, "first_render_submission", None
        )
        submission = first_submission(epoch) if callable(first_submission) else None
        if not isinstance(submission, RenderSubmission) or (
            boundary.generation,
            boundary.output_epoch,
            boundary.submission_id,
        ) != (
            submission.generation,
            submission.output_epoch,
            submission.submission_id,
        ):
            return
        observe = getattr(self._delegate_effects, "observe_render_receipt", None)
        if callable(observe):
            observe(epoch, boundary)

    async def _observe_playback_target(self) -> None:
        """Install actual final output evidence before any later capture admission."""
        if self._coordinator.snapshot.state is SpeculativeVoiceState.PROMOTING:
            return
        epoch = self._coordinator.snapshot.current_attempt_epoch
        target = self._playback_target
        if target is not None and target.attempt_epoch != epoch:
            self._playback_target = target = None
        if epoch is None:
            return
        if target is None:
            final_submission = getattr(
                self._delegate_effects, "final_render_submission", None
            )
            submission = final_submission(epoch) if callable(final_submission) else None
            if submission is None:
                return
            if not isinstance(submission, RenderSubmission):
                await self._coordinator.submit(AttemptPlaybackFailed(epoch))
                return
            counters = self._transport.native_counters
            remaining_startup = max(0, 50 - counters.get("callback_count", 0))
            latency = self._transport.stream_latency_seconds
            if latency is None:
                await self._coordinator.submit(AttemptPlaybackFailed(epoch))
                return
            target = _PlaybackTarget(
                epoch,
                submission,
                self._clock()
                + (remaining_startup + self._transport.buffer_capacities.render_frames)
                * 10_000_000
                + round(latency[1] * 1_000_000_000)
                + 500_000_000,
            )
            self._playback_target = target
        try:
            # Query again even after receipt: an abort/reset invalidates its authority.
            boundary = self._transport.render_boundary(target.submission)
        except CaptureDrainError:
            await self._coordinator.submit(AttemptPlaybackFailed(epoch))
            return
        now = self._clock()
        if target.boundary is not None and boundary != target.boundary:
            await self._coordinator.submit(AttemptPlaybackFailed(epoch))
            return
        if target.boundary is None:
            if now >= target.delivery_deadline_ns:
                await self._coordinator.submit(AttemptPlaybackFailed(epoch))
                return
            if boundary is None:
                return
            if (
                boundary.generation,
                boundary.output_epoch,
                boundary.submission_id,
            ) != (
                target.submission.generation,
                target.submission.output_epoch,
                target.submission.submission_id,
            ):
                await self._coordinator.submit(AttemptPlaybackFailed(epoch))
                return
            target.boundary = boundary
            await self._coordinator.submit(
                AttemptPlaybackBoundaryKnown(epoch, boundary.ended_ns)
            )
        boundary = target.boundary
        if now >= boundary.ended_ns + 500_000_000:
            await self._coordinator.submit(AttemptPlaybackFailed(epoch))
        elif not target.terminal_sent and now >= boundary.ended_ns:
            target.terminal_sent = True
            await self._coordinator.submit(
                AttemptPlaybackTerminal(epoch, boundary.ended_ns)
            )
            self._persist_voice_event("playback_terminal", status="ok")

    async def rebuild_audio(
        self,
        old_clock_generation: int,
        rebuild_epoch: int,
    ) -> None:
        """Reopen a changed device route and publish its fail-closed capability."""

        try:
            self._render_references.clear()
            self._release_pending_classification(failed=True)
            pre_start_generation = self._transport.clock_generation
            if pre_start_generation > self._preprocessor.active_clock_generation:
                self._preprocessor.reset_for_device_route(pre_start_generation)
                self._release_pending_classification(failed=True)
            await self._transport.start()
            clock_generation = self._transport.clock_generation
            if clock_generation <= old_clock_generation:
                raise RuntimeError("audio_route_generation_did_not_advance")
            if clock_generation > self._preprocessor.active_clock_generation:
                self._preprocessor.reset_for_device_route(clock_generation)
                self._release_pending_classification(failed=True)
            self._classification_failed_generation = None
            self._advertised_capability = None
            await self.submit(
                AudioRouteReady(
                    rebuild_epoch,
                    clock_generation,
                    self._preprocessor.mode,
                    self._preprocessor.health,
                    self._preprocessor.safety.path,
                )
            )
        except Exception as exc:
            clock_generation = max(
                int(getattr(self._transport, "clock_generation", 0)),
                old_clock_generation + 1,
            )
            await self.submit(
                AudioRouteRebuildFailed(
                    rebuild_epoch,
                    clock_generation,
                    type(exc).__name__,
                )
            )

    async def _pump_audio(self) -> None:
        try:
            while not self._fenced:
                if not await self.process_pending_audio():
                    await asyncio.sleep(self._pump_interval_seconds)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            try:
                self._fence_callbacks()
            finally:
                try:
                    if self._on_runtime_failure is not None:
                        self._on_runtime_failure(exc)
                finally:
                    try:
                        await self._close_once()
                    except AudioShutdownUnconfirmed as shutdown_error:
                        # The pump is retiring itself; expose its categorical close
                        # result through the owner callback and retain it for closers.
                        if self._on_runtime_failure is not None:
                            self._on_runtime_failure(shutdown_error)

    def _fence_callbacks(self) -> None:
        fence_output = getattr(self._transport, "fence_output", None)
        if callable(fence_output):
            fence_output()
        self.fence_audio_admission()
        self._fenced = True
        self._playback_target = None
        fence_all = getattr(self._delegate_effects, "fence_all", None)
        if not callable(fence_all):
            return
        result = fence_all()
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            elif isinstance(result, asyncio.Future):
                result.cancel()
            raise TypeError("fence_all must be synchronous")

    def _fail_transport(self) -> None:
        """Fence now; independently close audio and retain the available draft."""

        if self._fenced:
            return
        snapshot = self._coordinator.snapshot
        turn_id = snapshot.pending_next_turn_id or snapshot.turn_id
        transcript = (
            snapshot.pending_next_transcript_text
            if snapshot.pending_next_turn_id is not None
            else snapshot.transcript_text
        )
        self._fence_callbacks()

        async def preserve_and_report() -> None:
            try:
                preserve = getattr(self._delegate_effects, "preserve_draft", None)
                if callable(preserve) and turn_id and transcript:
                    result = preserve(
                        turn_id=turn_id,
                        transcript=transcript,
                        reason="audio_transport_failed",
                    )
                    if inspect.isawaitable(result):
                        await result
            except Exception as error:
                self._persist_voice_event(
                    "draft_preservation_failed",
                    status="failed",
                    exception_type=type(error).__name__,
                )

        self._transport_failure_report_task = asyncio.create_task(
            preserve_and_report(), name="voice-transport-failure-report"
        )
        self._transport_failure_report_task.add_done_callback(_consume_task_result)
        self._close_task = asyncio.create_task(
            self._close_once(), name="voice-transport-failure-close"
        )
        self._close_task.add_done_callback(_consume_task_result)
        if self._on_runtime_failure is not None:
            # Authority/off notification must not await recoverable draft IPC.
            with contextlib.suppress(Exception):
                self._on_runtime_failure(
                    VoiceTransportFailure("audio_transport_failed")
                )

    def _assistant_rendering(self) -> bool:
        value = getattr(self._delegate_effects, "assistant_rendering", False)
        value = value() if callable(value) else value
        if inspect.isawaitable(value):
            if inspect.iscoroutine(value):
                value.close()
            elif isinstance(value, asyncio.Future):
                value.cancel()
            raise TypeError("assistant_rendering must be synchronous")
        return bool(value)

    def _on_admitted_frame(self, frame: AudioFrame) -> None:
        if not self._fenced:
            pending = self._capture_admission_pending
            if pending is not None and frame.started_ns < pending.started_ns:
                # VAD preroll can precede the capture that released it.
                self._capture_admission_pending = frame
            self._admitted.append(frame)

    def _on_processed(
        self,
        sequence: int,
        clock_generation: int,
        dsp_ok: bool,
        vad_ok: bool,
    ) -> None:
        self._transport.acknowledge_capture(
            sequence,
            clock_generation=clock_generation,
            dsp_ok=dsp_ok,
            vad_ok=vad_ok,
        )

    def _on_classification_changed(
        self,
        watermark: PendingClassificationWatermark | None,
    ) -> None:
        if watermark is None:
            if self._pending_classification is not None:
                self._classification_release_pending = True
            return
        if not isinstance(watermark, PendingClassificationWatermark):
            raise TypeError("classification watermark is invalid")
        self._pending_classification = watermark
        self._classification_release_pending = False
        self._wake_classification_waiters()

    def _release_pending_classification(self, *, failed: bool = False) -> None:
        watermark = self._pending_classification
        if watermark is None and not self._classification_release_pending:
            return
        if failed and watermark is not None:
            self._classification_failed_generation = watermark.clock_generation
        self._pending_classification = None
        self._classification_release_pending = False
        self._wake_classification_waiters()

    def _wake_classification_waiters(self) -> None:
        waiters = tuple(self._classification_waiters)
        self._classification_waiters.clear()
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(None)

    async def _drain_pending_classification_through(
        self,
        render_boundary_ns: int,
        clock_generation: int,
    ) -> None:
        while True:
            active_generation = self._preprocessor.active_clock_generation
            if active_generation != clock_generation:
                raise CaptureDrainError("classification route generation changed")
            if self._classification_failed_generation == clock_generation:
                raise CaptureDrainError("classification failed before terminal drain")
            watermark = self._pending_classification
            capture = self._capture_admission_pending
            admission_pending = (
                capture is not None and capture.started_ns <= render_boundary_ns
            )
            if not admission_pending and (
                watermark is None or watermark.first_started_ns > render_boundary_ns
            ):
                return
            if watermark is not None and watermark.clock_generation != clock_generation:
                raise CaptureDrainError("classification watermark generation is stale")
            waiter = asyncio.get_running_loop().create_future()
            self._classification_waiters.add(waiter)
            try:
                await waiter
            finally:
                self._classification_waiters.discard(waiter)

    async def _accept_admitted_frame(
        self,
        frame: AudioFrame,
        *,
        assistant_rendering: bool,
    ) -> None:
        await self.submit(
            AdmittedSpeechFrame(
                frame.sequence,
                frame.started_ns,
                frame.ended_ns,
                frame.clock_generation,
                assistant_rendering,
                frame.speech_started_ns,
            )
        )
        snapshot = self._coordinator.snapshot
        turn_id = snapshot.pending_next_turn_id or snapshot.turn_id
        if turn_id is None:
            return
        self._transcript_for(turn_id).append_admitted_frame(frame)

    def _transcript_for(self, turn_id: str) -> _TranscriptPort:
        for transcript in self._transcripts:
            if transcript.turn_id == turn_id:
                return transcript
        transcript = self._transcript_factory(turn_id, self._on_transcript_revision)
        if not isinstance(transcript, TranscriptEngine) and not all(
            callable(getattr(transcript, name, None))
            for name in ("append_admitted_frame", "seal_through", "close")
        ):
            raise TypeError("transcript factory returned an invalid engine")
        self._persist_voice_event("speech_admitted", status="ok")
        self._transcripts.append(transcript)
        while len(self._transcripts) > _MAX_LIVE_TRANSCRIPTS:
            retired = self._transcripts.pop(0)
            retirement = asyncio.create_task(
                retired.close(),
                name=f"console-voice-transcript-retire-{retired.turn_id}",
            )
            self._transcript_retirements.add(retirement)
            retirement.add_done_callback(self._on_transcript_retired)
        return transcript

    def _on_transcript_retired(self, task: asyncio.Task[None]) -> None:
        self._transcript_retirements.discard(task)
        with contextlib.suppress(asyncio.CancelledError, Exception):
            task.result()

    def _on_transcript_revision(self, revision: TranscriptRevision) -> None:
        if not self._fenced:
            publish = getattr(self._delegate_effects, "record_revision", None)
            if callable(publish):
                try:
                    publish(revision)
                except Exception:
                    self._fail_transport()
                    return
            if revision.failure_code is not None:
                self._persist_voice_event(
                    "transcript_failed",
                    status="failed",
                    phase=revision.mode,
                    error_category=revision.failure_code,
                )
            asyncio.create_task(self.submit(revision))

    async def _seal_transcript_through(
        self,
        admitted_sequence: int,
    ) -> TranscriptRevision:
        last_error: ValueError | None = None
        for transcript in reversed(self._transcripts):
            try:
                return await transcript.seal_through(admitted_sequence)
            except ValueError as exc:
                last_error = exc
        raise last_error or ValueError("no transcript owns the admitted sequence")

    async def _close_with_reason(self, reason: ControlKind) -> None:
        await self._close_once(reason)

    async def _close_once(self, reason: ControlKind | None = None) -> None:
        self.fence_audio_admission()
        async with self._close_lock:
            if self._closed:
                if self._shutdown_error is not None:
                    raise self._shutdown_error
                return
            self._closed = True
            pump = self._pump_task
            self._resource_cleanup_task = asyncio.create_task(
                self._close_resources(
                    reason, None if pump is asyncio.current_task() else pump
                ),
                name="console-speculative-voice-resource-cleanup",
            )
            self._resource_cleanup_task.add_done_callback(_consume_task_result)
            # Native observation runs independently of provider/transcription
            # cleanup, and the transport retains its checked operation owner.
            try:
                await self._transport.close()
            except AudioShutdownUnconfirmed:
                self._shutdown_error = AudioShutdownUnconfirmed(
                    "audio_shutdown_unconfirmed"
                )
                raise self._shutdown_error from None
            await asyncio.shield(self._resource_cleanup_task)

    async def _close_resources(
        self, reason: ControlKind | None, pump: asyncio.Task[None] | None
    ) -> None:
        if self._started and reason is not None:
            with contextlib.suppress(RuntimeError):
                await self._coordinator.submit(ControlAction(reason))
        if pump is not None:
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)
        transcripts = tuple(self._transcripts)
        self._transcripts.clear()
        retirements = tuple(self._transcript_retirements)
        self._transcript_retirements.clear()
        self._render_references.clear()
        self._release_pending_classification(failed=True)
        if self._interrupt_transcription is not None:
            await self._interrupt_transcription()
        if transcripts or retirements:
            await asyncio.gather(
                *(transcript.close() for transcript in transcripts),
                *retirements,
                return_exceptions=True,
            )
        close_effects = getattr(self._delegate_effects, "close", None)
        effect_cleanup = close_effects() if callable(close_effects) else None
        transcription_cleanup = (
            self._close_transcription()
            if self._close_transcription is not None
            else None
        )
        await self._coordinator.close()
        failure_report = self._transport_failure_report_task
        await asyncio.gather(
            *([effect_cleanup] if inspect.isawaitable(effect_cleanup) else []),
            *(
                [transcription_cleanup]
                if inspect.isawaitable(transcription_cleanup)
                else []
            ),
            self._transport.abort_output(),
            *(
                [failure_report]
                if failure_report is not None
                and failure_report is not asyncio.current_task()
                else []
            ),
            return_exceptions=True,
        )
