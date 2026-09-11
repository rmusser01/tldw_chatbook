"""Serialized policy coordinator for provisional Console voice turns.

The coordinator owns no provider, audio device, transcript backend, or durable
store.  Every external signal and every injected timer enters one mailbox; the
single reducer is the only writer of logical-turn and attempt-epoch state.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
import contextlib
from dataclasses import dataclass, field
from enum import Enum
import inspect
import time
from typing import Any, Protocol, TypeAlias

from tldw_chatbook.Audio.duplex_contracts import (
    AcousticSafetyPath,
    AecHealth,
    DeviceRouteChanged,
    DrainReceipt,
    DuplexMode,
)
from tldw_chatbook.Audio.rolling_transcript import (
    TranscriptRevision,
    is_material_transcript_change,
)
from tldw_chatbook.Audio.voice_process_types import (
    RESPONSE_EAGERNESS_DEFAULT_MS,
    RESPONSE_EAGERNESS_MAX_MS,
    RESPONSE_EAGERNESS_MIN_MS,
    AttemptCleanupOutcome,
    AttemptDispatchPrepared,
    AttemptToolPending,
    ControlKind,
    ProviderAttemptFailed,
    VoiceRequestHandle,
    VoiceSpeculationDecision,
    VoiceTerminalDisposition,
    VoiceTurnContextHandle,
)


_CORRECTION_DEBOUNCE_NS = 120_000_000
_CORRECTION_HARD_CAP_NS = 250_000_000
_RESTART_WINDOW_NS = 10_000_000_000
_GOVERNED_EAGERNESS_NS = 1_500_000_000
_CLEANUP_SLOW_NS = 2_000_000_000
_CONSERVATIVE_QUIET_NS = 2_000_000_000
_TERMINAL_SEAL_DEADLINE_NS = 500_000_000
_EFFECT_BARRIER_NS = 2_000_000_000
_MAX_OBSOLETE_CLEANUPS = 2
_FRAME_TOLERANCE_NS = 10_000_000


def _validate_audio_safety_contract(
    *,
    clock_generation: int,
    duplex_mode: DuplexMode,
    aec_health: AecHealth,
    safety_path: AcousticSafetyPath,
) -> None:
    if type(clock_generation) is not int:
        raise TypeError("clock generation must be an integer")
    if clock_generation < 0:
        raise ValueError("clock generation must be non-negative")
    if not isinstance(duplex_mode, DuplexMode):
        raise TypeError("duplex mode must be a DuplexMode")
    if not isinstance(aec_health, AecHealth):
        raise TypeError("AEC health must be an AecHealth")
    if not isinstance(safety_path, AcousticSafetyPath):
        raise TypeError("safety path must be an AcousticSafetyPath")
    open_path = safety_path in {
        AcousticSafetyPath.AEC,
        AcousticSafetyPath.ACOUSTIC_ISOLATION,
    }
    if (duplex_mode is DuplexMode.FULL_DUPLEX) is not open_path:
        raise ValueError("safety path and duplex mode are inconsistent")
    if safety_path is AcousticSafetyPath.AEC and aec_health is not AecHealth.HEALTHY:
        raise ValueError("the AEC safety path requires healthy AEC")


class SpeculativeVoiceState(str, Enum):
    """Content-free state projected to the Console voice status surface."""

    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    GENERATING = "generating"
    SPEAKING = "speaking"
    RESPONDING_TEXT_ONLY = "responding_text_only"
    UPDATING_RESPONSE = "updating_response"
    WAITING_FOR_CLEANUP = "waiting_for_cleanup"
    SERIALIZED_CONSERVATIVE = "serialized_conservative"
    REBUILDING_AUDIO = "rebuilding_audio"
    LISTENING_AFTER_PROVIDER_FAILURE = "listening_after_provider_failure"
    SEALING = "sealing"
    PROMOTING = "promoting"
    DRAFT_SUSPENDED = "draft_suspended"
    WAITING_FOR_STABLE_TURN = "waiting for stable turn"


@dataclass(frozen=True, slots=True)
class AdmittedSpeechFrame:
    """Content-free VAD admission for one post-AEC capture frame."""

    sequence: int
    started_ns: int
    ended_ns: int
    clock_generation: int
    assistant_rendering: bool = True
    # None uses this positive frame's start; preroll points to its later onset.
    speech_started_ns: int | None = None

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("admitted sequence must be non-negative")
        if self.started_ns < 0 or self.ended_ns <= self.started_ns:
            raise ValueError("admitted frame timestamps must be ordered")
        if self.clock_generation < 0:
            raise ValueError("clock generation must be non-negative")
        if type(self.assistant_rendering) is not bool:
            raise TypeError("assistant_rendering must be a bool")
        if self.speech_started_ns is not None:
            if type(self.speech_started_ns) is not int:
                raise TypeError("speech onset must be an integer or None")
            if self.speech_started_ns < self.started_ns:
                raise ValueError("speech onset cannot precede its context frame")


@dataclass(frozen=True, slots=True)
class CaptureGated:
    """Intentional half-duplex evidence that capture was not speech-eligible."""

    sequence: int
    started_ns: int
    ended_ns: int
    clock_generation: int

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("gated sequence must be non-negative")
        if self.started_ns < 0 or self.ended_ns <= self.started_ns:
            raise ValueError("gated frame timestamps must be ordered")
        if self.clock_generation < 0:
            raise ValueError("clock generation must be non-negative")


@dataclass(frozen=True, slots=True)
class AudioCapabilityChanged:
    """Fail-closed duplex capability update within the current audio route."""

    clock_generation: int
    duplex_mode: DuplexMode
    aec_health: AecHealth
    safety_path: AcousticSafetyPath

    def __post_init__(self) -> None:
        _validate_audio_safety_contract(
            clock_generation=self.clock_generation,
            duplex_mode=self.duplex_mode,
            aec_health=self.aec_health,
            safety_path=self.safety_path,
        )


@dataclass(frozen=True, slots=True)
class AttemptOutputDelta:
    """One attempt-fenced provisional assistant delta."""

    attempt_epoch: int
    text: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class AttemptPlaybackStarted:
    """Current-attempt PCM has begun entering the render transport."""

    attempt_epoch: int


@dataclass(frozen=True, slots=True)
class AttemptGenerationCompleted:
    """The provider completed the current attempt's full textual response."""

    attempt_epoch: int
    assistant_text: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class AttemptPlaybackBoundaryKnown:
    """Actual final DAC boundary is known, but playback need not have elapsed."""

    attempt_epoch: int
    render_boundary_ns: int

    def __post_init__(self) -> None:
        if self.render_boundary_ns < 0:
            raise ValueError("render boundary must be non-negative")


@dataclass(frozen=True, slots=True)
class AttemptPlaybackTerminal:
    """The actual final current-attempt DAC boundary has elapsed."""

    attempt_epoch: int
    render_boundary_ns: int

    def __post_init__(self) -> None:
        if self.render_boundary_ns < 0:
            raise ValueError("render boundary must be non-negative")


@dataclass(frozen=True, slots=True)
class AttemptPlaybackFailed:
    """Delivery or actual render completion could not be proved."""

    attempt_epoch: int


@dataclass(frozen=True, slots=True)
class AttemptTtsFailed:
    """Content-free recoverable speech-only failure for one attempt."""

    attempt_epoch: int
    error_class: str

    def __post_init__(self) -> None:
        if not self.error_class:
            raise ValueError("TTS error class must be nonempty")


@dataclass(frozen=True, slots=True)
class AudioRouteReady:
    """A replacement clock domain has a fail-closed duplex capability."""

    rebuild_epoch: int
    clock_generation: int
    duplex_mode: DuplexMode
    aec_health: AecHealth
    safety_path: AcousticSafetyPath

    def __post_init__(self) -> None:
        if type(self.rebuild_epoch) is not int:
            raise TypeError("rebuild epoch must be an integer")
        if self.rebuild_epoch < 1:
            raise ValueError("rebuild epoch must be positive")
        _validate_audio_safety_contract(
            clock_generation=self.clock_generation,
            duplex_mode=self.duplex_mode,
            aec_health=self.aec_health,
            safety_path=self.safety_path,
        )


@dataclass(frozen=True, slots=True)
class AudioRouteRebuildFailed:
    """Content-free terminal failure of the current audio-route rebuild."""

    rebuild_epoch: int
    clock_generation: int
    error_class: str

    def __post_init__(self) -> None:
        if self.rebuild_epoch < 1:
            raise ValueError("rebuild epoch must be positive")
        if self.clock_generation < 0:
            raise ValueError("clock generation must be non-negative")
        if not self.error_class:
            raise ValueError("route failure class must be nonempty")


@dataclass(frozen=True, slots=True)
class ManualInterruption:
    """Explicit same-turn barge-in, including intentional half duplex."""


@dataclass(frozen=True, slots=True)
class ManualRetry:
    """Explicitly retry the exact retained provider-failure draft."""


@dataclass(frozen=True, slots=True)
class ControlAction:
    """Explicit discard boundary for provisional voice content."""

    kind: ControlKind

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ControlKind):
            raise TypeError("control kind must be a ControlKind")


VoiceEvent: TypeAlias = (
    AdmittedSpeechFrame
    | CaptureGated
    | AudioCapabilityChanged
    | TranscriptRevision
    | AttemptDispatchPrepared
    | AttemptOutputDelta
    | AttemptPlaybackStarted
    | AttemptGenerationCompleted
    | AttemptPlaybackBoundaryKnown
    | AttemptPlaybackFailed
    | AttemptPlaybackTerminal
    | AttemptTtsFailed
    | AudioRouteReady
    | AudioRouteRebuildFailed
    | DeviceRouteChanged
    | ProviderAttemptFailed
    | AttemptToolPending
    | ManualInterruption
    | ManualRetry
    | ControlAction
)


@dataclass(frozen=True, slots=True)
class SpeculativeVoiceSnapshot:
    """Immutable coordinator projection; text is deliberately repr-hidden."""

    state: SpeculativeVoiceState
    turn_id: str | None
    attempt_epoch: int
    current_attempt_epoch: int | None
    transcript_text: str = field(repr=False)
    revision_id: int
    last_speech_end_ns: int | None
    last_admitted_sequence: int | None
    terminal_boundary_ns: int | None
    obsolete_cleanup_count: int
    serialized_conservative: bool
    speculation_suspended: bool
    failure_class: str | None
    audio_clock_generation: int
    audio_rebuild_epoch: int
    duplex_mode: DuplexMode
    admission_open: bool
    pending_next_frame_count: int
    pending_next_turn_id: str | None
    pending_next_transcript_text: str = field(repr=False)


class _ScheduledHandle(Protocol):
    def cancel(self) -> None: ...


class _Scheduler(Protocol):
    now_ns: int | Callable[[], int]

    def call_at_ns(
        self,
        deadline_ns: int,
        callback: Callable[[], None],
    ) -> _ScheduledHandle: ...


class _Effects(Protocol):
    def dispatch_attempt(
        self,
        *,
        turn_id: str,
        attempt_epoch: int,
        transcript: str,
    ) -> Any: ...

    def start_prepared_attempt(
        self, attempt_epoch: int, request_handle: VoiceRequestHandle
    ) -> None: ...

    def fence_attempt(self, attempt_epoch: int) -> Any: ...

    def cancel_attempt(
        self, attempt_epoch: int
    ) -> Awaitable[AttemptCleanupOutcome]: ...

    def abort_output(self, attempt_epoch: int) -> Any: ...

    def clear_preview(self, attempt_epoch: int) -> Any: ...

    def publish_preview(self, attempt_epoch: int, delta: str) -> Any: ...

    def preserve_draft(
        self,
        *,
        turn_id: str,
        transcript: str,
        reason: str,
    ) -> Any: ...

    def promote(
        self,
        *,
        turn_id: str,
        attempt_epoch: int,
        transcript: str,
        assistant_text: str,
        terminal_boundary_ns: int | None,
    ) -> VoiceTerminalDisposition | Awaitable[VoiceTerminalDisposition]: ...

    def classify_spoken_command(self, transcript: str) -> bool: ...

    def handle_spoken_command(self, *, turn_id: str, transcript: str) -> Any: ...

    def rebuild_audio(self, old_clock_generation: int, rebuild_epoch: int) -> Any: ...

    async def drain_capture_through(self, render_boundary_ns: int) -> DrainReceipt: ...

    async def drain_pending_classification_through(
        self,
        render_boundary_ns: int,
        clock_generation: int,
    ) -> None: ...

    async def seal_transcript_through(
        self, admitted_sequence: int
    ) -> TranscriptRevision: ...

    def voice_dispatch_quarantined(self) -> bool: ...

    def submit_accepted_voice_turn(
        self,
        exact_user_text: str,
        turn_context_handle: VoiceTurnContextHandle,
    ) -> object: ...


class _AsyncioHandle:
    def __init__(self, handle: asyncio.TimerHandle) -> None:
        self._handle = handle

    def cancel(self) -> None:
        self._handle.cancel()


class _AsyncioScheduler:
    @property
    def now_ns(self) -> int:
        return time.monotonic_ns()

    def call_at_ns(
        self,
        deadline_ns: int,
        callback: Callable[[], None],
    ) -> _ScheduledHandle:
        delay_seconds = max(0, deadline_ns - self.now_ns) / 1_000_000_000
        return _AsyncioHandle(
            asyncio.get_running_loop().call_later(delay_seconds, callback)
        )


@dataclass(slots=True)
class _Envelope:
    event: object
    acknowledgement: asyncio.Future[None] | None


@dataclass(frozen=True, slots=True)
class _Barrier:
    pass


@dataclass(frozen=True, slots=True)
class _Shutdown:
    pass


@dataclass(frozen=True, slots=True)
class _EagernessExpired:
    token: int


@dataclass(frozen=True, slots=True)
class _CorrectionExpired:
    series: int
    soft_token: int | None
    hard: bool


@dataclass(frozen=True, slots=True)
class _CleanupSlow:
    attempt_epoch: int


@dataclass(frozen=True, slots=True)
class _CleanupFinished:
    attempt_epoch: int


@dataclass(frozen=True, slots=True)
class _EffectBarrierExpired:
    token: int


@dataclass(frozen=True, slots=True)
class _TerminalSealSucceeded:
    token: int
    revision: TranscriptRevision = field(repr=False)


@dataclass(frozen=True, slots=True)
class _TerminalSealFailed:
    token: int


@dataclass(frozen=True, slots=True)
class _TerminalSealTimedOut:
    token: int


@dataclass(frozen=True, slots=True)
class _PromotionFinished:
    token: int
    succeeded: bool


@dataclass(frozen=True, slots=True)
class _TerminalCandidate:
    token: int
    turn_id: str
    attempt_epoch: int
    transcript: str = field(repr=False)
    assistant_text: str = field(repr=False)
    render_boundary_ns: int
    admitted_sequence: int
    clock_generation: int
    activity_version: int
    capture_drain_required: bool


class SpeculativeTurnCoordinator:
    """Serialize one provisional logical voice turn and its attempt epochs."""

    def __init__(
        self,
        *,
        effects: _Effects,
        scheduler: _Scheduler | None = None,
        response_eagerness_ms: object = RESPONSE_EAGERNESS_DEFAULT_MS,
        frame_tolerance_ns: int = _FRAME_TOLERANCE_NS,
        initial_clock_generation: int = 0,
        initial_duplex_mode: DuplexMode = DuplexMode.FULL_DUPLEX,
        turn_id_factory: Callable[[int], str] | None = None,
        prepared_policy: AttemptDispatchPrepared | None = None,
        deferred_attempt_preparation: bool = False,
    ) -> None:
        if frame_tolerance_ns < 0:
            raise ValueError("frame tolerance must be non-negative")
        if initial_clock_generation < 0:
            raise ValueError("clock generation must be non-negative")
        if not isinstance(initial_duplex_mode, DuplexMode):
            raise TypeError("initial duplex mode must be a DuplexMode")
        if (
            prepared_policy is not None
            and type(prepared_policy) is not AttemptDispatchPrepared
        ):
            raise TypeError("prepared_policy must be an AttemptDispatchPrepared")
        if type(deferred_attempt_preparation) is not bool:
            raise TypeError("deferred_attempt_preparation must be a bool")
        if deferred_attempt_preparation and prepared_policy is not None:
            raise ValueError(
                "deferred attempt preparation cannot use static request context"
            )
        self._effects = effects
        self._scheduler = scheduler or _AsyncioScheduler()
        self._response_eagerness_ms = _bounded_eagerness(response_eagerness_ms)
        self._frame_tolerance_ns = frame_tolerance_ns
        self._turn_id_factory = turn_id_factory or (
            lambda serial: f"voice-turn-{serial}"
        )
        self._deferred_attempt_preparation = deferred_attempt_preparation
        self._default_turn_context_handle = (
            prepared_policy.turn_context_handle if prepared_policy is not None else None
        )
        self._default_ordinary_handoff_required = (
            prepared_policy is not None
            and prepared_policy.decision
            is VoiceSpeculationDecision.WAIT_FOR_STABLE_TURN
        )
        self._turn_context_handle = self._default_turn_context_handle
        self._ordinary_handoff_required = self._default_ordinary_handoff_required

        self._mailbox: asyncio.Queue[_Envelope] = asyncio.Queue()
        self._runner: asyncio.Task[None] | None = None
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None
        self._background: set[asyncio.Task[Any]] = set()

        self._state = SpeculativeVoiceState.IDLE
        self._turn_serial = 0
        self._turn_id: str | None = None
        self._transcript_text = ""
        self._revision_id = 0
        self._covered_through_ns = 0
        self._last_speech_end_ns: int | None = None
        self._last_speech_received_ns: int | None = None
        self._last_admitted_sequence: int | None = None
        self._activity_version = 0

        self._epoch = 0
        self._epoch_reserved = False
        self._active_epoch: int | None = None
        self._attempt_snapshot = ""
        self._generation_text: str | None = None
        self._tts_failure_version: int | None = None
        self._tts_failure_revision_id: int | None = None
        self._attempt_history_ns: deque[int] = deque()

        self._eagerness_token = 0
        self._eagerness_handle: _ScheduledHandle | None = None
        self._eagerness_expired = False
        self._correction_series = 0
        self._correction_soft_token = 0
        self._correction_started_ns: int | None = None
        self._correction_soft_handle: _ScheduledHandle | None = None
        self._correction_hard_handle: _ScheduledHandle | None = None
        self._effect_barrier_token = 0
        self._effect_barrier_handle: _ScheduledHandle | None = None
        self._effect_barrier_active = False
        self._tool_barrier_epoch: int | None = None
        self._tool_handoff_required = False

        self._obsolete: dict[int, asyncio.Future[Any]] = {}
        self._cleanup_slow_handles: dict[int, _ScheduledHandle] = {}
        self._serialized_conservative = False

        self._audio_clock_generation = initial_clock_generation
        self._duplex_mode = initial_duplex_mode
        self._admission_open = True
        self._capture_drain_required = False
        self._route_rebuild_epoch = 0
        self._route_candidate_generation: int | None = None
        self._route_retained_transcript_fresh = False
        self._route_resume_state: SpeculativeVoiceState | None = None
        self._route_resume_failure_class: str | None = None
        self._speculation_suspended = False
        self._provider_retry_suspended = False
        self._failure_class: str | None = None

        self._terminal_token = 0
        self._known_playback_boundary: AttemptPlaybackBoundaryKnown | None = None
        self._playback_terminal_received = False
        self._terminal: _TerminalCandidate | None = None
        self._terminal_task: asyncio.Task[None] | None = None
        self._terminal_deadline_handle: _ScheduledHandle | None = None
        self._pending_next_frames: deque[AdmittedSpeechFrame] = deque()
        self._pending_next_turn_id: str | None = None
        self._pending_next_transcript_text = ""
        self._pending_next_revision_id = 0
        self._pending_next_covered_through_ns = 0
        self._pending_next_failure_code: str | None = None
        self._promotion_token = 0
        self._promotion_task: asyncio.Task[None] | None = None
        self._pending_route_change: DeviceRouteChanged | None = None

    @property
    def response_eagerness_ms(self) -> int:
        """Return the qualified 500..3000 ms speculative silence setting."""

        return self._response_eagerness_ms

    @property
    def snapshot(self) -> SpeculativeVoiceSnapshot:
        """Return one immutable view of reducer-owned state."""

        terminal_boundary = (
            None
            if self._known_playback_boundary is None
            else self._known_playback_boundary.render_boundary_ns
        )
        return SpeculativeVoiceSnapshot(
            state=self._state,
            turn_id=self._turn_id,
            attempt_epoch=self._epoch,
            current_attempt_epoch=self._active_epoch,
            transcript_text=self._transcript_text,
            revision_id=self._revision_id,
            last_speech_end_ns=self._last_speech_end_ns,
            last_admitted_sequence=self._last_admitted_sequence,
            terminal_boundary_ns=terminal_boundary,
            obsolete_cleanup_count=len(self._obsolete),
            serialized_conservative=self._serialized_conservative,
            speculation_suspended=self._speculation_suspended,
            failure_class=self._failure_class,
            audio_clock_generation=self._audio_clock_generation,
            audio_rebuild_epoch=self._route_rebuild_epoch,
            duplex_mode=self._duplex_mode,
            admission_open=self._admission_open,
            pending_next_frame_count=len(self._pending_next_frames),
            pending_next_turn_id=self._pending_next_turn_id,
            pending_next_transcript_text=self._pending_next_transcript_text,
        )

    async def start(self) -> None:
        """Start the sole mailbox consumer idempotently."""

        if self._closed or self._closing:
            raise RuntimeError("speculative voice coordinator is closed")
        if self._runner is not None:
            return
        self._owner_loop = asyncio.get_running_loop()
        self._runner = asyncio.create_task(self._run())

    async def submit(self, event: VoiceEvent) -> None:
        """Enqueue and reduce one event before returning to its owner."""

        if self._closed or self._closing:
            raise RuntimeError("speculative voice coordinator is closed")
        await self.start()
        acknowledgement = asyncio.get_running_loop().create_future()
        await self._mailbox.put(_Envelope(event, acknowledgement))
        await acknowledgement

    async def flush(self) -> None:
        """Process already-ready timer and operation callbacks without wall time."""

        await self.start()
        stable_rounds = 0
        for _ in range(12):
            acknowledgement = asyncio.get_running_loop().create_future()
            await self._mailbox.put(_Envelope(_Barrier(), acknowledgement))
            await acknowledgement
            checkpoint = asyncio.get_running_loop().create_future()
            asyncio.get_running_loop().call_soon(checkpoint.set_result, None)
            await checkpoint
            ready_background = any(task.done() for task in self._background)
            if self._mailbox.empty() and not ready_background:
                stable_rounds += 1
                if stable_rounds >= 2:
                    return
            else:
                stable_rounds = 0

    async def close(self) -> None:
        """Discard provisional state and deterministically stop owned tasks."""

        if self._closed:
            return
        close_task = self._close_task
        if close_task is None:
            self._closing = True
            close_task = asyncio.create_task(self._close_once())
            self._close_task = close_task
        await asyncio.shield(close_task)

    async def _close_once(self) -> None:
        """Own teardown independently of any individual close waiter."""

        runner = self._runner
        try:
            if runner is not None:
                acknowledgement = asyncio.get_running_loop().create_future()
                await self._mailbox.put(_Envelope(_Shutdown(), acknowledgement))
                try:
                    await acknowledgement
                finally:
                    await runner
        finally:
            self._runner = None
            self._closed = True
            self._cancel_all_timers()
            owned = tuple(task for task in self._background if not task.done())
            for task in owned:
                task.cancel()
            if owned:
                await asyncio.gather(*owned, return_exceptions=True)
            self._background.clear()

    async def _run(self) -> None:
        while True:
            envelope = await self._mailbox.get()
            shutdown = isinstance(envelope.event, _Shutdown)
            try:
                if shutdown:
                    await self._discard_turn()
                else:
                    await self._reduce(envelope.event)
            except BaseException as exc:
                acknowledgement = envelope.acknowledgement
                if acknowledgement is not None and not acknowledgement.done():
                    acknowledgement.set_exception(exc)
                elif isinstance(exc, asyncio.CancelledError):
                    raise
            else:
                acknowledgement = envelope.acknowledgement
                if acknowledgement is not None and not acknowledgement.done():
                    acknowledgement.set_result(None)
            finally:
                self._mailbox.task_done()
            if shutdown:
                return

    async def _reduce(self, event: object) -> None:
        if isinstance(event, _Barrier):
            return
        if isinstance(event, AdmittedSpeechFrame):
            await self._on_speech(event)
            return
        if isinstance(event, CaptureGated):
            self._on_gated_capture(event)
            return
        if isinstance(event, AudioCapabilityChanged):
            self._on_audio_capability_changed(event)
            return
        if isinstance(event, TranscriptRevision):
            await self._on_transcript(event)
            return
        if isinstance(event, AttemptDispatchPrepared):
            await self._on_attempt_dispatch_prepared(event)
            return
        if isinstance(event, AttemptOutputDelta):
            self._on_output_delta(event)
            return
        if isinstance(event, AttemptToolPending):
            self._on_tool_request(event)
            return
        if isinstance(event, AttemptPlaybackStarted):
            self._on_playback_started(event)
            return
        if isinstance(event, AttemptGenerationCompleted):
            await self._on_generation_completed(event)
            return
        if isinstance(event, AttemptPlaybackBoundaryKnown):
            self._on_playback_boundary_known(event)
            return
        if isinstance(event, AttemptPlaybackFailed):
            if event.attempt_epoch == self._active_epoch:
                await self._fail_playback_sync()
            return
        if isinstance(event, AttemptPlaybackTerminal):
            self._on_playback_terminal(event)
            return
        if isinstance(event, AttemptTtsFailed):
            await self._on_tts_failed(event)
            return
        if isinstance(event, ProviderAttemptFailed):
            await self._on_provider_failed(event)
            return
        if isinstance(event, DeviceRouteChanged):
            await self._on_route_changed(event)
            return
        if isinstance(event, AudioRouteReady):
            await self._on_route_ready(event)
            return
        if isinstance(event, AudioRouteRebuildFailed):
            await self._on_route_rebuild_failed(event)
            return
        if isinstance(event, ManualInterruption):
            await self._on_manual_interruption()
            return
        if isinstance(event, ManualRetry):
            await self._on_manual_retry()
            return
        if isinstance(event, ControlAction):
            await self._discard_turn()
            return
        if isinstance(event, _EagernessExpired):
            await self._on_eagerness_expired(event)
            return
        if isinstance(event, _CorrectionExpired):
            await self._on_correction_expired(event)
            return
        if isinstance(event, _CleanupSlow):
            self._on_cleanup_slow(event)
            return
        if isinstance(event, _CleanupFinished):
            await self._on_cleanup_finished(event)
            return
        if isinstance(event, _EffectBarrierExpired):
            await self._on_effect_barrier_expired(event)
            return
        if isinstance(event, _TerminalSealSucceeded):
            await self._on_terminal_succeeded(event)
            return
        if isinstance(event, (_TerminalSealFailed, _TerminalSealTimedOut)):
            await self._on_terminal_failed(event.token)
            return
        if isinstance(event, _PromotionFinished):
            await self._on_promotion_finished(event)
            return
        raise TypeError(f"unsupported speculative voice event: {type(event).__name__}")

    def _on_audio_capability_changed(self, event: AudioCapabilityChanged) -> None:
        if self._state is SpeculativeVoiceState.REBUILDING_AUDIO:
            return
        if event.clock_generation != self._audio_clock_generation:
            return
        self._duplex_mode = event.duplex_mode
        if self._state in {
            SpeculativeVoiceState.SPEAKING,
            SpeculativeVoiceState.SEALING,
        }:
            self._admission_open = event.duplex_mode is DuplexMode.FULL_DUPLEX
        if (
            self._state is SpeculativeVoiceState.SPEAKING
            and event.duplex_mode is DuplexMode.FULL_DUPLEX
        ):
            self._capture_drain_required = True

    async def _on_speech(self, frame: AdmittedSpeechFrame) -> None:
        if self._state is SpeculativeVoiceState.REBUILDING_AUDIO:
            return
        if frame.clock_generation != self._audio_clock_generation:
            return
        if self._state is SpeculativeVoiceState.PROMOTING:
            self._buffer_next_turn_frame(frame)
            return
        if self._speculation_suspended:
            if self._failure_class == "voice_cleanup_stuck":
                if self._voice_dispatch_quarantined():
                    return
                self._clear_turn()
                self._begin_turn(frame)
                return
            if self._failure_class not in ("backend_failed", "fallback_failed"):
                return
        if (
            self._turn_id is not None
            and self._last_admitted_sequence is not None
            and frame.sequence <= self._last_admitted_sequence
        ):
            return
        if self._speculation_suspended:
            self._speculation_suspended = False
            self._failure_class = None
            self._admission_open = True
        boundary = self._known_playback_boundary
        if boundary is not None:
            speech_started_ns = (
                frame.started_ns
                if frame.speech_started_ns is None
                else frame.speech_started_ns
            )
            if speech_started_ns > boundary.render_boundary_ns:
                self._buffer_next_turn_frame(frame)
                return
            if (
                self._duplex_mode is DuplexMode.HALF_DUPLEX
                and frame.assistant_rendering
            ):
                return
            self._activity_version += 1
            self._cancel_terminal_seal()
            await self._fence_active(cancel=True)
            self._accept_speech_into_turn(frame)
            return
        if (
            self._duplex_mode is DuplexMode.HALF_DUPLEX
            and self._state is SpeculativeVoiceState.SPEAKING
            and frame.assistant_rendering
        ):
            return
        if self._turn_id is None:
            self._begin_turn(frame)
            return

        self._activity_version += 1
        if self._effect_barrier_active:
            self._cancel_effect_barrier()
        if self._active_epoch is not None:
            await self._fence_active(cancel=True)
        self._provider_retry_suspended = False
        self._accept_speech_into_turn(frame)

    def _buffer_next_turn_frame(self, frame: AdmittedSpeechFrame) -> None:
        last_sequence = (
            self._pending_next_frames[-1].sequence
            if self._pending_next_frames
            else self._last_admitted_sequence
        )
        if last_sequence is not None and frame.sequence <= last_sequence:
            return
        if self._pending_next_turn_id is None:
            self._pending_next_turn_id = self._allocate_turn_id()
        self._pending_next_frames.append(frame)

    def _clear_pending_next_turn(self) -> None:
        self._pending_next_frames.clear()
        self._pending_next_turn_id = None
        self._pending_next_transcript_text = ""
        self._pending_next_revision_id = 0
        self._pending_next_covered_through_ns = 0
        self._pending_next_failure_code = None

    def _allocate_turn_id(self) -> str:
        self._turn_serial += 1
        turn_id = self._turn_id_factory(self._turn_serial)
        if not turn_id:
            raise ValueError("turn id factory returned an empty identifier")
        return turn_id

    def _begin_turn(
        self,
        frame: AdmittedSpeechFrame,
        *,
        turn_id: str | None = None,
    ) -> None:
        turn_id = turn_id or self._allocate_turn_id()
        self._turn_id = turn_id
        self._transcript_text = ""
        self._revision_id = 0
        self._covered_through_ns = 0
        self._attempt_snapshot = ""
        self._generation_text = None
        self._tts_failure_version = None
        self._tts_failure_revision_id = None
        self._attempt_history_ns.clear()
        self._serialized_conservative = False
        self._provider_retry_suspended = False
        self._failure_class = None
        self._speculation_suspended = False
        self._activity_version += 1
        self._accept_speech_into_turn(frame)

    def _accept_speech_into_turn(self, frame: AdmittedSpeechFrame) -> None:
        if self._last_admitted_sequence is not None and (
            frame.sequence <= self._last_admitted_sequence
        ):
            return
        self._last_admitted_sequence = frame.sequence
        self._last_speech_end_ns = frame.ended_ns
        self._last_speech_received_ns = self._now_ns()
        self._failure_class = None
        self._route_retained_transcript_fresh = False
        self._cancel_correction()
        self._eagerness_expired = False
        if self._ordinary_handoff_required:
            self._schedule_effect_barrier()
            self._state = SpeculativeVoiceState.WAITING_FOR_STABLE_TURN
            return
        delay_ns = (
            _CONSERVATIVE_QUIET_NS
            if self._serialized_conservative
            else self._effective_eagerness_ns()
        )
        self._schedule_eagerness_at(self._now_ns() + delay_ns)
        self._state = (
            SpeculativeVoiceState.SERIALIZED_CONSERVATIVE
            if self._serialized_conservative
            else SpeculativeVoiceState.LISTENING
        )

    def _on_gated_capture(self, event: CaptureGated) -> None:
        if event.clock_generation != self._audio_clock_generation:
            return
        if self._duplex_mode is not DuplexMode.HALF_DUPLEX:
            return
        # Deliberately no turn mutation: playback-period capture was not admitted.

    async def _on_transcript(self, revision: TranscriptRevision) -> None:
        if (
            self._state
            in (SpeculativeVoiceState.SEALING, SpeculativeVoiceState.PROMOTING)
            and self._pending_next_turn_id is not None
            and revision.turn_id == self._pending_next_turn_id
        ):
            if revision.revision_id <= self._pending_next_revision_id:
                return
            self._pending_next_revision_id = revision.revision_id
            self._pending_next_covered_through_ns = revision.covered_through_ns
            self._pending_next_transcript_text = (
                f"{revision.stable_text}{revision.revisable_text}"
            )
            self._pending_next_failure_code = revision.failure_code
            return
        if self._state is SpeculativeVoiceState.PROMOTING:
            return
        if self._turn_id is None or revision.turn_id != self._turn_id:
            return
        if revision.revision_id <= self._revision_id:
            return
        before = self._transcript_text
        after = f"{revision.stable_text}{revision.revisable_text}"
        self._revision_id = revision.revision_id
        self._covered_through_ns = revision.covered_through_ns
        self._transcript_text = after
        material_from_previous = is_material_transcript_change(before, after)

        if revision.failure_code is not None:
            self._activity_version += 1
            self._cancel_eagerness()
            self._cancel_correction()
            self._cancel_effect_barrier()
            self._cancel_terminal_seal()
            if self._active_epoch is not None:
                await self._fence_active(cancel=True)
            self._speculation_suspended = True
            self._provider_retry_suspended = False
            self._failure_class = revision.failure_code
            self._admission_open = True
            self._state = SpeculativeVoiceState.DRAFT_SUSPENDED
            self._invoke_effect(
                self._effects.preserve_draft,
                turn_id=self._turn_id,
                transcript=self._transcript_text,
                reason="stt_failed",
            )
            return

        if self._effect_barrier_active:
            self._state = SpeculativeVoiceState.WAITING_FOR_STABLE_TURN
            await self._handoff_effectful_turn_if_ready()
            return

        if (
            self._tts_failure_revision_id is not None
            and revision.revision_id > self._tts_failure_revision_id
        ):
            self._activity_version += 1
            await self._fence_active(cancel=True)
            self._start_or_extend_correction()
            return

        if self._active_epoch is not None and is_material_transcript_change(
            self._attempt_snapshot,
            after,
        ):
            self._activity_version += 1
            await self._fence_active(cancel=True)
            self._start_or_extend_correction()
            return
        if self._correction_started_ns is not None and material_from_previous:
            self._activity_version += 1
            self._start_or_extend_correction()
            return
        if self._provider_retry_suspended:
            return
        if self._eagerness_expired:
            await self._try_dispatch()

    def _on_output_delta(self, event: AttemptOutputDelta) -> None:
        if event.attempt_epoch != self._active_epoch or not event.text:
            return
        if self._state not in (
            SpeculativeVoiceState.GENERATING,
            SpeculativeVoiceState.SPEAKING,
            SpeculativeVoiceState.RESPONDING_TEXT_ONLY,
        ):
            return
        self._invoke_effect(
            self._effects.publish_preview, event.attempt_epoch, event.text
        )

    async def _on_attempt_dispatch_prepared(
        self,
        event: AttemptDispatchPrepared,
    ) -> None:
        if not self._deferred_attempt_preparation:
            return
        if event.attempt_epoch != self._active_epoch:
            return
        self._turn_context_handle = event.turn_context_handle
        decision = event.decision
        self._ordinary_handoff_required = (
            decision is VoiceSpeculationDecision.WAIT_FOR_STABLE_TURN
        )
        if self._ordinary_handoff_required:
            await self._fence_active(cancel=True)
            self._schedule_effect_barrier()
            self._state = SpeculativeVoiceState.WAITING_FOR_STABLE_TURN
            await self._handoff_effectful_turn_if_ready()
            return
        try:
            result = self._effects.start_prepared_attempt(
                event.attempt_epoch, event.request_handle
            )
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                elif isinstance(result, asyncio.Future):
                    result.cancel()
                raise TypeError("prepared attempt start must be synchronous")
        except Exception as exc:
            await self._on_provider_failed(
                ProviderAttemptFailed(event.attempt_epoch, type(exc).__name__)
            )

    def _on_tool_request(self, event: AttemptToolPending) -> None:
        if event.attempt_epoch != self._active_epoch:
            return
        if self._tool_barrier_epoch is not None:
            return
        self._tool_barrier_epoch = event.attempt_epoch
        self._tool_handoff_required = True
        self._cancel_eagerness()
        self._cancel_correction()
        self._schedule_effect_barrier()
        self._state = SpeculativeVoiceState.WAITING_FOR_STABLE_TURN
        try:
            self._invoke_effect(self._effects.abort_output, event.attempt_epoch)
        except BaseException:
            pass

    def _on_playback_started(self, event: AttemptPlaybackStarted) -> None:
        if event.attempt_epoch != self._active_epoch:
            return
        if event.attempt_epoch == self._tool_barrier_epoch:
            return
        self._state = SpeculativeVoiceState.SPEAKING
        if self._duplex_mode is DuplexMode.FULL_DUPLEX:
            self._capture_drain_required = True
        if self._duplex_mode is DuplexMode.HALF_DUPLEX:
            self._admission_open = False

    async def _on_generation_completed(
        self,
        event: AttemptGenerationCompleted,
    ) -> None:
        if event.attempt_epoch != self._active_epoch:
            return
        if event.attempt_epoch == self._tool_barrier_epoch:
            return
        boundary = self._known_playback_boundary
        if (
            boundary is not None
            and self._now_ns()
            >= boundary.render_boundary_ns + _TERMINAL_SEAL_DEADLINE_NS
        ):
            await self._fail_playback_sync()
            return
        self._generation_text = event.assistant_text
        if self._tts_failure_version is not None:
            if (
                self._tts_failure_version == self._activity_version
                and self._tts_failure_revision_id == self._revision_id
            ):
                await self._promote_text_only(event.assistant_text)
        elif self._playback_terminal_received and self._known_playback_boundary:
            self._on_playback_terminal(
                AttemptPlaybackTerminal(
                    event.attempt_epoch,
                    self._known_playback_boundary.render_boundary_ns,
                )
            )

    def _on_playback_boundary_known(self, event: AttemptPlaybackBoundaryKnown) -> None:
        if (
            event.attempt_epoch != self._active_epoch
            or event.attempt_epoch == self._tool_barrier_epoch
            or self._known_playback_boundary is not None
        ):
            return
        self._known_playback_boundary = event

    def _on_playback_terminal(self, event: AttemptPlaybackTerminal) -> None:
        if event.attempt_epoch != self._active_epoch:
            return
        if event.attempt_epoch == self._tool_barrier_epoch:
            return
        self._on_playback_boundary_known(
            AttemptPlaybackBoundaryKnown(event.attempt_epoch, event.render_boundary_ns)
        )
        if self._known_playback_boundary.render_boundary_ns != event.render_boundary_ns:
            return
        self._playback_terminal_received = True
        if self._terminal is not None:
            return
        if self._generation_text is None:
            return
        if self._turn_id is None or self._last_admitted_sequence is None:
            return
        self._terminal_token += 1
        candidate = _TerminalCandidate(
            token=self._terminal_token,
            turn_id=self._turn_id,
            attempt_epoch=event.attempt_epoch,
            transcript=self._attempt_snapshot,
            assistant_text=self._generation_text,
            render_boundary_ns=event.render_boundary_ns,
            admitted_sequence=self._last_admitted_sequence,
            clock_generation=self._audio_clock_generation,
            activity_version=self._activity_version,
            capture_drain_required=(
                self._capture_drain_required
                or self._duplex_mode is DuplexMode.FULL_DUPLEX
            ),
        )
        self._terminal = candidate
        self._state = SpeculativeVoiceState.SEALING
        self._admission_open = self._duplex_mode is DuplexMode.FULL_DUPLEX
        self._terminal_deadline_handle = self._scheduler.call_at_ns(
            event.render_boundary_ns + _TERMINAL_SEAL_DEADLINE_NS,
            lambda token=candidate.token: self._post(_TerminalSealTimedOut(token)),
        )
        self._terminal_task = asyncio.create_task(self._run_terminal_seal(candidate))
        self._own_background(self._terminal_task)

    async def _on_tts_failed(self, event: AttemptTtsFailed) -> None:
        if event.attempt_epoch != self._active_epoch:
            return
        if event.attempt_epoch == self._tool_barrier_epoch:
            return
        self._tts_failure_version = self._activity_version
        self._tts_failure_revision_id = self._revision_id
        self._failure_class = event.error_class
        self._state = SpeculativeVoiceState.RESPONDING_TEXT_ONLY
        self._admission_open = True
        if self._generation_text is not None:
            await self._promote_text_only(self._generation_text)

    async def _on_provider_failed(self, event: ProviderAttemptFailed) -> None:
        if event.attempt_epoch != self._active_epoch or self._turn_id is None:
            return
        if event.attempt_epoch == self._tool_barrier_epoch:
            return
        self._activity_version += 1
        await self._fence_active(cancel=False)
        self._provider_retry_suspended = True
        self._failure_class = event.error_class
        self._cancel_eagerness()
        self._cancel_correction()
        self._state = SpeculativeVoiceState.LISTENING_AFTER_PROVIDER_FAILURE
        self._admission_open = True
        self._invoke_effect(
            self._effects.preserve_draft,
            turn_id=self._turn_id,
            transcript=self._transcript_text,
            reason="provider_failed",
        )

    async def _on_route_changed(self, event: DeviceRouteChanged) -> None:
        if event.old_clock_generation != self._audio_clock_generation:
            return
        if self._state is SpeculativeVoiceState.PROMOTING:
            self._pending_route_change = event
            return
        if self._route_resume_state is None:
            if self._state is SpeculativeVoiceState.LISTENING_AFTER_PROVIDER_FAILURE:
                self._route_resume_state = self._state
                self._route_resume_failure_class = self._failure_class
            elif (
                self._state is SpeculativeVoiceState.DRAFT_SUSPENDED
                and self._failure_class
                in (
                    "backend_failed",
                    "fallback_failed",
                    "accepted_handoff_failed",
                    "promotion_failed",
                )
            ):
                self._route_resume_state = self._state
                self._route_resume_failure_class = self._failure_class
        self._route_retained_transcript_fresh = (
            bool(self._transcript_text.strip())
            and self._last_speech_end_ns is not None
            and self._covered_through_ns + self._frame_tolerance_ns
            >= self._last_speech_end_ns
        )
        self._activity_version += 1
        self._cancel_eagerness()
        self._cancel_correction()
        if self._tool_handoff_required:
            self._pause_effect_barrier()
        else:
            self._cancel_effect_barrier()
        self._cancel_terminal_seal()
        self._clear_pending_next_turn()
        if self._active_epoch is not None:
            await self._fence_active(cancel=True)
        else:
            self._advance_idle_fence()
        self._speculation_suspended = True
        self._admission_open = False
        self._route_rebuild_epoch += 1
        self._route_candidate_generation = None
        self._state = SpeculativeVoiceState.REBUILDING_AUDIO
        self._invoke_effect(
            self._effects.rebuild_audio,
            event.old_clock_generation,
            self._route_rebuild_epoch,
        )

    async def _on_route_ready(self, event: AudioRouteReady) -> None:
        if self._state is not SpeculativeVoiceState.REBUILDING_AUDIO:
            return
        if event.rebuild_epoch != self._route_rebuild_epoch:
            return
        if event.clock_generation <= self._audio_clock_generation:
            return
        if (
            self._route_candidate_generation is not None
            and event.clock_generation < self._route_candidate_generation
        ):
            return
        self._route_candidate_generation = event.clock_generation
        self._audio_clock_generation = event.clock_generation
        self._duplex_mode = event.duplex_mode
        resume_state = self._route_resume_state
        resume_failure_class = self._route_resume_failure_class
        self._route_resume_state = None
        self._route_resume_failure_class = None
        if resume_state is SpeculativeVoiceState.LISTENING_AFTER_PROVIDER_FAILURE:
            self._speculation_suspended = False
            self._provider_retry_suspended = True
            self._failure_class = resume_failure_class
            self._admission_open = True
            self._state = resume_state
            return
        if resume_state is SpeculativeVoiceState.DRAFT_SUSPENDED:
            self._speculation_suspended = True
            self._provider_retry_suspended = False
            self._failure_class = resume_failure_class
            self._admission_open = True
            self._state = resume_state
            return
        self._speculation_suspended = False
        self._provider_retry_suspended = False
        self._failure_class = None
        self._admission_open = True
        self._eagerness_expired = False
        if self._ordinary_handoff_required or self._tool_handoff_required:
            self._schedule_effect_barrier()
            self._state = SpeculativeVoiceState.WAITING_FOR_STABLE_TURN
            return
        self._last_speech_received_ns = self._now_ns()
        delay_ns = (
            _CONSERVATIVE_QUIET_NS
            if self._serialized_conservative
            else self._effective_eagerness_ns()
        )
        self._schedule_eagerness_at(self._now_ns() + delay_ns)
        self._state = (
            SpeculativeVoiceState.SERIALIZED_CONSERVATIVE
            if self._serialized_conservative
            else SpeculativeVoiceState.LISTENING
        )

    async def _on_route_rebuild_failed(
        self,
        event: AudioRouteRebuildFailed,
    ) -> None:
        if self._state is not SpeculativeVoiceState.REBUILDING_AUDIO:
            return
        if event.rebuild_epoch != self._route_rebuild_epoch:
            return
        if event.clock_generation <= self._audio_clock_generation:
            return
        if (
            self._route_candidate_generation is not None
            and event.clock_generation < self._route_candidate_generation
        ):
            return
        self._route_candidate_generation = event.clock_generation
        self._speculation_suspended = True
        self._failure_class = event.error_class
        self._admission_open = False
        self._state = SpeculativeVoiceState.DRAFT_SUSPENDED
        if self._turn_id is not None:
            self._invoke_effect(
                self._effects.preserve_draft,
                turn_id=self._turn_id,
                transcript=self._transcript_text,
                reason="audio_rebuild_failed",
            )

    async def _on_manual_interruption(self) -> None:
        if self._state is SpeculativeVoiceState.PROMOTING:
            return
        if self._turn_id is None:
            return
        self._activity_version += 1
        self._cancel_terminal_seal()
        if self._active_epoch is not None:
            await self._fence_active(cancel=True)
        self._provider_retry_suspended = False
        self._failure_class = None
        self._admission_open = True
        self._last_speech_received_ns = self._now_ns()
        self._eagerness_expired = False
        if self._ordinary_handoff_required:
            self._schedule_effect_barrier()
            self._state = SpeculativeVoiceState.WAITING_FOR_STABLE_TURN
            return
        delay = (
            _CONSERVATIVE_QUIET_NS
            if self._serialized_conservative
            else self._effective_eagerness_ns()
        )
        self._schedule_eagerness_at(self._now_ns() + delay)
        self._state = (
            SpeculativeVoiceState.SERIALIZED_CONSERVATIVE
            if self._serialized_conservative
            else SpeculativeVoiceState.LISTENING
        )

    async def _on_manual_retry(self) -> None:
        if (
            self._state is SpeculativeVoiceState.DRAFT_SUSPENDED
            and self._failure_class in ("backend_failed", "fallback_failed")
            and self._turn_id is not None
        ):
            self._speculation_suspended = False
            self._failure_class = None
            self._eagerness_expired = True
            await self._try_dispatch(force=True)
            return
        if (
            self._state is not SpeculativeVoiceState.LISTENING_AFTER_PROVIDER_FAILURE
            or self._turn_id is None
        ):
            return
        self._provider_retry_suspended = False
        self._failure_class = None
        self._eagerness_expired = True
        await self._try_dispatch(force=True)

    async def _on_eagerness_expired(self, event: _EagernessExpired) -> None:
        if event.token != self._eagerness_token:
            return
        self._eagerness_handle = None
        self._eagerness_expired = True
        await self._try_dispatch()

    async def _on_effect_barrier_expired(self, event: _EffectBarrierExpired) -> None:
        if event.token != self._effect_barrier_token:
            return
        self._effect_barrier_handle = None
        await self._handoff_effectful_turn_if_ready()

    async def _on_correction_expired(self, event: _CorrectionExpired) -> None:
        if event.series != self._correction_series:
            return
        if self._correction_started_ns is None:
            return
        if not event.hard and event.soft_token != self._correction_soft_token:
            return
        correction_started_ns = self._correction_started_ns
        self._prune_attempt_history(self._now_ns())
        governed = len(self._attempt_history_ns) >= 3
        self._cancel_correction()
        if governed and correction_started_ns is not None:
            self._eagerness_expired = False
            self._schedule_eagerness_at(correction_started_ns + _GOVERNED_EAGERNESS_NS)
            return
        self._eagerness_expired = True
        await self._try_dispatch()

    def _on_cleanup_slow(self, event: _CleanupSlow) -> None:
        if event.attempt_epoch not in self._obsolete:
            return
        self._serialized_conservative = True
        self._cancel_eagerness()
        if self._state in (
            SpeculativeVoiceState.REBUILDING_AUDIO,
            SpeculativeVoiceState.DRAFT_SUSPENDED,
            SpeculativeVoiceState.LISTENING_AFTER_PROVIDER_FAILURE,
        ):
            return
        self._state = SpeculativeVoiceState.SERIALIZED_CONSERVATIVE
        self._schedule_conservative_deadline()

    async def _on_cleanup_finished(self, event: _CleanupFinished) -> None:
        completion = self._obsolete.pop(event.attempt_epoch, None)
        if completion is None:
            return
        try:
            outcome = completion.result()
        except BaseException:
            outcome = None
        handle = self._cleanup_slow_handles.pop(event.attempt_epoch, None)
        if handle is not None:
            handle.cancel()
        if (
            outcome is AttemptCleanupOutcome.DETACHED
            or type(outcome) is not AttemptCleanupOutcome
            or self._voice_dispatch_quarantined()
        ):
            await self._suspend_for_stuck_cleanup()
            return
        if self._turn_id is None or self._active_epoch is not None:
            return
        if self._serialized_conservative:
            self._schedule_conservative_deadline()
            if self._quiet_for_ns() >= _CONSERVATIVE_QUIET_NS:
                self._eagerness_expired = True
                await self._try_dispatch()
        elif self._eagerness_expired:
            await self._try_dispatch()

    async def _on_terminal_succeeded(self, event: _TerminalSealSucceeded) -> None:
        candidate = self._terminal
        if candidate is None or candidate.token != event.token:
            return
        if self._now_ns() >= candidate.render_boundary_ns + _TERMINAL_SEAL_DEADLINE_NS:
            await self._on_terminal_failed(event.token)
            return
        if event.revision.turn_id != candidate.turn_id:
            await self._on_terminal_failed(event.token)
            return
        if (
            candidate.attempt_epoch != self._active_epoch
            or candidate.activity_version != self._activity_version
        ):
            self._cancel_terminal_seal()
            return
        if event.revision.revision_id > self._revision_id:
            sealed_text = f"{event.revision.stable_text}{event.revision.revisable_text}"
            if is_material_transcript_change(candidate.transcript, sealed_text):
                self._cancel_terminal_seal()
                await self._on_transcript(event.revision)
                return
            await self._on_transcript(event.revision)
            current = self._terminal
            if (
                current is None
                or current.token != event.token
                or candidate.attempt_epoch != self._active_epoch
                or candidate.activity_version != self._activity_version
            ):
                return
        self._cancel_terminal_timer()
        self._terminal = None
        self._terminal_task = None
        await self._start_promotion(
            turn_id=candidate.turn_id,
            attempt_epoch=candidate.attempt_epoch,
            transcript=candidate.transcript,
            assistant_text=candidate.assistant_text,
            terminal_boundary_ns=candidate.render_boundary_ns,
        )

    async def _on_terminal_failed(self, token: int) -> None:
        candidate = self._terminal
        if candidate is None or candidate.token != token:
            return
        await self._fail_playback_sync()

    async def _fail_playback_sync(self) -> None:
        self._cancel_terminal_seal()
        if self._active_epoch is not None:
            await self._fence_active(cancel=False)
        self._speculation_suspended = True
        self._failure_class = "capture_sync_failed"
        self._admission_open = False
        self._route_rebuild_epoch += 1
        self._route_candidate_generation = None
        self._clear_pending_next_turn()
        self._state = SpeculativeVoiceState.REBUILDING_AUDIO
        if self._turn_id is not None:
            self._invoke_effect(
                self._effects.preserve_draft,
                turn_id=self._turn_id,
                transcript=self._transcript_text,
                reason="capture_sync_failed",
            )
        self._invoke_effect(
            self._effects.rebuild_audio,
            self._audio_clock_generation,
            self._route_rebuild_epoch,
        )

    async def _try_dispatch(self, *, force: bool = False) -> None:
        if self._turn_id is None or self._active_epoch is not None:
            return
        if self._ordinary_handoff_required or self._tool_handoff_required:
            if not self._effect_barrier_active:
                self._schedule_effect_barrier()
            self._state = SpeculativeVoiceState.WAITING_FOR_STABLE_TURN
            await self._handoff_effectful_turn_if_ready()
            return
        if self._voice_dispatch_quarantined():
            await self._suspend_for_stuck_cleanup()
            return
        if self._speculation_suspended or self._provider_retry_suspended:
            return
        if self._terminal is not None or self._correction_started_ns is not None:
            return
        if len(self._obsolete) >= _MAX_OBSOLETE_CLEANUPS:
            self._state = (
                SpeculativeVoiceState.SERIALIZED_CONSERVATIVE
                if self._serialized_conservative
                else SpeculativeVoiceState.WAITING_FOR_CLEANUP
            )
            return
        if self._serialized_conservative and (
            self._obsolete or self._quiet_for_ns() < _CONSERVATIVE_QUIET_NS
        ):
            self._state = SpeculativeVoiceState.SERIALIZED_CONSERVATIVE
            return
        if not force and not self._eagerness_expired:
            return
        if not self._transcript_text.strip() or not self._transcript_is_fresh():
            self._state = SpeculativeVoiceState.TRANSCRIBING
            return
        command = self._effects.classify_spoken_command(self._transcript_text)
        if inspect.isawaitable(command):
            raise TypeError("spoken-command classification must be synchronous")
        if command:
            self._invoke_effect(
                self._effects.handle_spoken_command,
                turn_id=self._turn_id,
                transcript=self._transcript_text,
            )
            self._clear_turn()
            return

        if self._epoch_reserved:
            epoch = self._epoch
            self._epoch_reserved = False
        else:
            self._epoch += 1
            epoch = self._epoch
        self._active_epoch = epoch
        self._capture_drain_required = False
        self._attempt_snapshot = self._transcript_text
        self._generation_text = None
        self._tts_failure_version = None
        self._tts_failure_revision_id = None
        self._state = SpeculativeVoiceState.GENERATING
        self._failure_class = None
        self._admission_open = True
        self._cancel_eagerness()
        now = self._now_ns()
        self._prune_attempt_history(now)
        self._attempt_history_ns.append(now)
        self._invoke_effect(
            self._effects.dispatch_attempt,
            turn_id=self._turn_id,
            attempt_epoch=epoch,
            transcript=self._attempt_snapshot,
        )

    async def _fence_active(self, *, cancel: bool) -> None:
        epoch = self._active_epoch
        if epoch is None:
            return
        self._cancel_terminal_seal()
        self._epoch = max(self._epoch, epoch) + 1
        self._epoch_reserved = True
        self._active_epoch = None
        self._generation_text = None
        self._tts_failure_version = None
        self._tts_failure_revision_id = None
        self._state = SpeculativeVoiceState.UPDATING_RESPONSE
        try:
            self._effects.fence_attempt(epoch)
        except BaseException:
            pass
        try:
            self._invoke_effect(self._effects.clear_preview, epoch)
        except BaseException:
            pass
        try:
            self._invoke_effect(self._effects.abort_output, epoch)
        except BaseException:
            pass
        if cancel:
            try:
                cleanup = self._effects.cancel_attempt(epoch)
                self._track_cleanup(epoch, cleanup)
            except BaseException:
                pass

    def _advance_idle_fence(self) -> None:
        self._epoch += 1
        self._epoch_reserved = False

    def _track_cleanup(self, epoch: int, cleanup: Any) -> None:
        if cleanup is None or not inspect.isawaitable(cleanup):
            return
        completion = asyncio.ensure_future(cleanup)
        if not completion.done() and len(self._obsolete) >= _MAX_OBSOLETE_CLEANUPS:
            _consume_future_when_done(completion)
            self._serialized_conservative = True
            self._state = SpeculativeVoiceState.SERIALIZED_CONSERVATIVE
            return
        self._obsolete[epoch] = completion
        completion.add_done_callback(_consume_future)
        completion.add_done_callback(
            lambda _done, attempt_epoch=epoch: self._post(
                _CleanupFinished(attempt_epoch)
            )
        )
        if not completion.done():
            self._cleanup_slow_handles[epoch] = self._scheduler.call_at_ns(
                self._now_ns() + _CLEANUP_SLOW_NS,
                lambda attempt_epoch=epoch: self._post(_CleanupSlow(attempt_epoch)),
            )

    def _start_or_extend_correction(self) -> None:
        now = self._now_ns()
        if self._correction_started_ns is None:
            self._correction_series += 1
            self._correction_started_ns = now
            series = self._correction_series
            self._correction_hard_handle = self._scheduler.call_at_ns(
                now + _CORRECTION_HARD_CAP_NS,
                lambda series=series: self._post(
                    _CorrectionExpired(series, None, True)
                ),
            )
        if self._correction_soft_handle is not None:
            self._correction_soft_handle.cancel()
        self._correction_soft_token += 1
        series = self._correction_series
        soft_token = self._correction_soft_token
        hard_deadline = self._correction_started_ns + _CORRECTION_HARD_CAP_NS
        soft_deadline = min(now + _CORRECTION_DEBOUNCE_NS, hard_deadline)
        self._correction_soft_handle = self._scheduler.call_at_ns(
            soft_deadline,
            lambda series=series, token=soft_token: self._post(
                _CorrectionExpired(series, token, False)
            ),
        )
        self._state = SpeculativeVoiceState.UPDATING_RESPONSE

    def _schedule_eagerness_at(self, deadline_ns: int) -> None:
        self._cancel_eagerness()
        self._eagerness_token += 1
        token = self._eagerness_token
        self._eagerness_handle = self._scheduler.call_at_ns(
            max(self._now_ns(), deadline_ns),
            lambda token=token: self._post(_EagernessExpired(token)),
        )

    def _schedule_effect_barrier(self) -> None:
        if self._turn_id is None:
            return
        if self._effect_barrier_handle is not None:
            self._effect_barrier_handle.cancel()
        self._effect_barrier_token += 1
        self._effect_barrier_active = True
        token = self._effect_barrier_token
        last_speech = self._last_speech_received_ns
        deadline = (
            self._now_ns() if last_speech is None else last_speech + _EFFECT_BARRIER_NS
        )
        self._effect_barrier_handle = self._scheduler.call_at_ns(
            max(self._now_ns(), deadline),
            lambda token=token: self._post(_EffectBarrierExpired(token)),
        )

    def _cancel_effect_barrier(self) -> None:
        self._pause_effect_barrier()
        self._tool_handoff_required = False

    def _pause_effect_barrier(self) -> None:
        if self._effect_barrier_handle is not None:
            self._effect_barrier_handle.cancel()
            self._effect_barrier_handle = None
        self._effect_barrier_token += 1
        self._effect_barrier_active = False
        self._tool_barrier_epoch = None

    async def _handoff_effectful_turn_if_ready(self) -> None:
        if not self._effect_barrier_active or self._turn_id is None:
            return
        if self._quiet_for_ns() < _EFFECT_BARRIER_NS:
            if self._effect_barrier_handle is None:
                self._schedule_effect_barrier()
            return
        if not self._transcript_text.strip() or not self._transcript_is_fresh():
            self._state = SpeculativeVoiceState.WAITING_FOR_STABLE_TURN
            return
        context = self._turn_context_handle
        if context is None:
            raise RuntimeError("accepted voice handoff has no frozen context")
        transcript = self._transcript_text
        if self._active_epoch is not None:
            await self._fence_active(cancel=True)
        self._cancel_effect_barrier()
        try:
            result = self._effects.submit_accepted_voice_turn(transcript, context)
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                elif isinstance(result, asyncio.Future):
                    result.cancel()
                raise TypeError(
                    "accepted voice handoff must transfer custody synchronously"
                )
        except Exception:
            self._speculation_suspended = True
            self._provider_retry_suspended = False
            self._failure_class = "accepted_handoff_failed"
            self._admission_open = True
            self._state = SpeculativeVoiceState.DRAFT_SUSPENDED
            try:
                self._invoke_effect(
                    self._effects.preserve_draft,
                    turn_id=self._turn_id,
                    transcript=transcript,
                    reason="accepted_handoff_failed",
                )
            except BaseException:
                pass
            return
        self._clear_turn()

    def _schedule_conservative_deadline(self) -> None:
        if self._turn_id is None:
            return
        last = self._last_speech_received_ns
        deadline = self._now_ns() if last is None else last + _CONSERVATIVE_QUIET_NS
        self._schedule_eagerness_at(deadline)

    def _effective_eagerness_ns(self) -> int:
        now = self._now_ns()
        self._prune_attempt_history(now)
        if len(self._attempt_history_ns) >= 3:
            return _GOVERNED_EAGERNESS_NS
        return self._response_eagerness_ms * 1_000_000

    def _prune_attempt_history(self, now_ns: int) -> None:
        threshold = now_ns - _RESTART_WINDOW_NS
        while self._attempt_history_ns and self._attempt_history_ns[0] < threshold:
            self._attempt_history_ns.popleft()

    def _transcript_is_fresh(self) -> bool:
        if self._route_retained_transcript_fresh:
            return True
        if self._last_speech_end_ns is None:
            return False
        return (
            self._covered_through_ns + self._frame_tolerance_ns
            >= self._last_speech_end_ns
        )

    def _quiet_for_ns(self) -> int:
        if self._last_speech_received_ns is None:
            return 0
        return max(0, self._now_ns() - self._last_speech_received_ns)

    async def _run_terminal_seal(self, candidate: _TerminalCandidate) -> None:
        try:
            if candidate.capture_drain_required:
                receipt = await self._effects.drain_capture_through(
                    candidate.render_boundary_ns
                )
                _validate_drain_receipt(receipt, candidate)
            await self._effects.drain_pending_classification_through(
                candidate.render_boundary_ns,
                candidate.clock_generation,
            )
            revision = await self._effects.seal_transcript_through(
                candidate.admitted_sequence
            )
        except asyncio.CancelledError:
            return
        except BaseException:
            self._post(_TerminalSealFailed(candidate.token))
        else:
            self._post(_TerminalSealSucceeded(candidate.token, revision))

    async def _promote_text_only(self, assistant_text: str) -> None:
        if self._turn_id is None or self._active_epoch is None:
            return
        epoch = self._active_epoch
        await self._start_promotion(
            turn_id=self._turn_id,
            attempt_epoch=epoch,
            transcript=self._attempt_snapshot,
            assistant_text=assistant_text,
            terminal_boundary_ns=None,
        )

    async def _start_promotion(
        self,
        *,
        turn_id: str,
        attempt_epoch: int,
        transcript: str,
        assistant_text: str,
        terminal_boundary_ns: int | None,
    ) -> None:
        self._promotion_token += 1
        token = self._promotion_token
        self._state = SpeculativeVoiceState.PROMOTING
        self._admission_open = True
        try:
            result = self._effects.promote(
                turn_id=turn_id,
                attempt_epoch=attempt_epoch,
                transcript=transcript,
                assistant_text=assistant_text,
                terminal_boundary_ns=terminal_boundary_ns,
            )
        except BaseException:
            await self._on_promotion_finished(_PromotionFinished(token, False))
            return
        if not inspect.isawaitable(result):
            await self._on_promotion_finished(
                _PromotionFinished(token, _promotion_succeeded(result))
            )
            return
        task = asyncio.ensure_future(self._wait_for_promotion(token, result))
        self._promotion_task = task
        self._own_background(task)

    async def _wait_for_promotion(self, token: int, result: Any) -> None:
        try:
            outcome = await result
        except asyncio.CancelledError:
            return
        except BaseException:
            succeeded = False
        else:
            succeeded = _promotion_succeeded(outcome)
        self._post(_PromotionFinished(token, succeeded))

    async def _on_promotion_finished(self, event: _PromotionFinished) -> None:
        if (
            event.token != self._promotion_token
            or self._state is not SpeculativeVoiceState.PROMOTING
        ):
            return
        self._promotion_task = None
        pending = tuple(self._pending_next_frames)
        self._pending_next_frames.clear()
        pending_turn_id = self._pending_next_turn_id
        pending_transcript = self._pending_next_transcript_text
        pending_revision_id = self._pending_next_revision_id
        pending_covered_through_ns = self._pending_next_covered_through_ns
        pending_failure_code = self._pending_next_failure_code
        pending_route = self._pending_route_change
        self._pending_route_change = None
        self._finish_successful_turn()
        self._begin_pending_turn(
            pending,
            turn_id=pending_turn_id,
            transcript=pending_transcript,
            revision_id=pending_revision_id,
            covered_through_ns=pending_covered_through_ns,
        )
        if not event.succeeded and pending:
            self._cancel_eagerness()
            self._speculation_suspended = True
            self._failure_class = "promotion_failed"
            self._state = SpeculativeVoiceState.DRAFT_SUSPENDED
            if self._turn_id is not None:
                self._invoke_effect(
                    self._effects.preserve_draft,
                    turn_id=self._turn_id,
                    transcript=self._transcript_text,
                    reason="promotion_failed_pending_next_turn",
                )
        elif event.succeeded and pending_failure_code is not None and pending:
            self._cancel_eagerness()
            self._speculation_suspended = True
            self._failure_class = pending_failure_code
            self._state = SpeculativeVoiceState.DRAFT_SUSPENDED
            if self._turn_id is not None:
                self._invoke_effect(
                    self._effects.preserve_draft,
                    turn_id=self._turn_id,
                    transcript=self._transcript_text,
                    reason="stt_failed",
                )
        if pending_route is not None:
            await self._on_route_changed(pending_route)

    def _begin_pending_turn(
        self,
        pending: tuple[AdmittedSpeechFrame, ...],
        *,
        turn_id: str | None = None,
        transcript: str = "",
        revision_id: int = 0,
        covered_through_ns: int = 0,
    ) -> None:
        for index, frame in enumerate(pending):
            if index == 0:
                self._begin_turn(frame, turn_id=turn_id)
            else:
                self._accept_speech_into_turn(frame)
        if pending:
            self._transcript_text = transcript
            self._revision_id = revision_id
            self._covered_through_ns = covered_through_ns

    def _finish_successful_turn(self) -> None:
        self._active_epoch = None
        self._epoch_reserved = False
        self._clear_turn()

    def _clear_turn(self) -> None:
        self._cancel_eagerness()
        self._cancel_correction()
        self._cancel_effect_barrier()
        self._cancel_terminal_seal()
        self._turn_id = None
        self._transcript_text = ""
        self._revision_id = 0
        self._covered_through_ns = 0
        self._last_speech_end_ns = None
        self._last_speech_received_ns = None
        self._last_admitted_sequence = None
        self._active_epoch = None
        self._capture_drain_required = False
        self._attempt_snapshot = ""
        self._generation_text = None
        self._tts_failure_version = None
        self._tts_failure_revision_id = None
        self._attempt_history_ns.clear()
        self._serialized_conservative = False
        self._provider_retry_suspended = False
        self._failure_class = None
        self._speculation_suspended = False
        self._route_retained_transcript_fresh = False
        self._route_resume_state = None
        self._route_resume_failure_class = None
        self._clear_pending_next_turn()
        self._pending_route_change = None
        self._turn_context_handle = self._default_turn_context_handle
        self._ordinary_handoff_required = self._default_ordinary_handoff_required
        self._admission_open = True
        self._state = SpeculativeVoiceState.IDLE

    async def _discard_turn(self) -> None:
        self._activity_version += 1
        self._cancel_eagerness()
        self._cancel_correction()
        self._cancel_terminal_seal()
        try:
            if self._active_epoch is not None:
                await self._fence_active(cancel=True)
            else:
                self._advance_idle_fence()
        finally:
            self._clear_turn()

    def _cancel_eagerness(self) -> None:
        if self._eagerness_handle is not None:
            self._eagerness_handle.cancel()
            self._eagerness_handle = None

    def _cancel_correction(self) -> None:
        if self._correction_soft_handle is not None:
            self._correction_soft_handle.cancel()
        if self._correction_hard_handle is not None:
            self._correction_hard_handle.cancel()
        self._correction_soft_handle = None
        self._correction_hard_handle = None
        self._correction_started_ns = None

    def _cancel_terminal_timer(self) -> None:
        if self._terminal_deadline_handle is not None:
            self._terminal_deadline_handle.cancel()
            self._terminal_deadline_handle = None

    def _cancel_terminal_seal(self) -> None:
        self._cancel_terminal_timer()
        self._known_playback_boundary = None
        self._playback_terminal_received = False
        task = self._terminal_task
        self._terminal_task = None
        self._terminal = None
        if task is not None and not task.done():
            task.cancel()

    def _cancel_all_timers(self) -> None:
        self._cancel_eagerness()
        self._cancel_correction()
        self._cancel_effect_barrier()
        self._cancel_terminal_timer()
        for handle in self._cleanup_slow_handles.values():
            handle.cancel()
        self._cleanup_slow_handles.clear()

    def _invoke_effect(
        self, callback: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        result = callback(*args, **kwargs)
        if inspect.isawaitable(result):
            task = asyncio.ensure_future(result)
            self._own_background(task)

    def _own_background(self, task: asyncio.Task[Any]) -> None:
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        task.add_done_callback(_consume_future)

    def _post(self, event: object) -> None:
        if self._closed or self._closing or self._runner is None:
            return
        self._mailbox.put_nowait(_Envelope(event, None))

    def _now_ns(self) -> int:
        value = self._scheduler.now_ns
        now = value() if callable(value) else value
        if type(now) is not int or now < 0:
            raise ValueError("scheduler now_ns must be a non-negative integer")
        return now

    def _voice_dispatch_quarantined(self) -> bool:
        quarantined = self._effects.voice_dispatch_quarantined()
        if inspect.isawaitable(quarantined):
            raise TypeError("voice dispatch quarantine check must be synchronous")
        return bool(quarantined)

    async def _suspend_for_stuck_cleanup(self) -> None:
        if self._failure_class == "voice_cleanup_stuck":
            return
        self._activity_version += 1
        self._cancel_eagerness()
        self._cancel_correction()
        self._cancel_terminal_seal()
        if self._active_epoch is not None:
            await self._fence_active(cancel=True)
        self._epoch_reserved = False
        self._speculation_suspended = True
        self._provider_retry_suspended = False
        self._failure_class = "voice_cleanup_stuck"
        self._admission_open = False
        self._clear_pending_next_turn()
        self._state = SpeculativeVoiceState.DRAFT_SUSPENDED
        if self._turn_id is not None:
            self._invoke_effect(
                self._effects.preserve_draft,
                turn_id=self._turn_id,
                transcript=self._transcript_text,
                reason="voice_cleanup_stuck",
            )


def _bounded_eagerness(value: object) -> int:
    if (
        type(value) is int
        and RESPONSE_EAGERNESS_MIN_MS <= value <= RESPONSE_EAGERNESS_MAX_MS
    ):
        return value
    return RESPONSE_EAGERNESS_DEFAULT_MS


def _promotion_succeeded(result: object) -> bool:
    """Accept only the parent's categorical successful disposition."""

    return result is VoiceTerminalDisposition.PROMOTED


def _validate_drain_receipt(
    receipt: DrainReceipt,
    candidate: _TerminalCandidate,
) -> None:
    if receipt.capture_watermark_ns <= candidate.render_boundary_ns:
        raise ValueError("capture watermark did not pass render boundary")
    if receipt.clock_generation != candidate.clock_generation:
        raise ValueError("capture receipt belongs to another clock generation")
    if (
        receipt.dsp_sequence < receipt.capture_sequence
        or receipt.vad_sequence < receipt.capture_sequence
    ):
        raise ValueError("capture receipt has an unacknowledged downstream sequence")
    if receipt.capture_sequence < candidate.admitted_sequence:
        raise ValueError("capture receipt did not acknowledge admitted audio")


def _consume_future(future: asyncio.Future[Any]) -> None:
    with contextlib.suppress(BaseException):
        future.result()


def _consume_future_when_done(future: asyncio.Future[Any]) -> None:
    if future.done():
        _consume_future(future)
    else:
        future.add_done_callback(_consume_future)


__all__ = [
    "AdmittedSpeechFrame",
    "AttemptDispatchPrepared",
    "AttemptGenerationCompleted",
    "AttemptOutputDelta",
    "AttemptPlaybackStarted",
    "AttemptPlaybackBoundaryKnown",
    "AttemptPlaybackFailed",
    "AttemptPlaybackTerminal",
    "AttemptTtsFailed",
    "AudioCapabilityChanged",
    "AudioRouteReady",
    "AudioRouteRebuildFailed",
    "CaptureGated",
    "ControlAction",
    "ControlKind",
    "ManualInterruption",
    "ManualRetry",
    "SpeculativeTurnCoordinator",
    "SpeculativeVoiceSnapshot",
    "SpeculativeVoiceState",
    "VoiceEvent",
]
