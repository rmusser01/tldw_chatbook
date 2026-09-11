"""Deterministic integration checks for the speculative voice composition root."""

from __future__ import annotations

import asyncio
from collections import deque
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from Tests.Audio.fakes.fake_duplex_backend import FakeDuplexBackend, ManualClock
from tldw_chatbook.Audio.duplex_contracts import (
    AcousticIsolationObservation,
    AcousticSafetyPath,
    AcousticSafetySnapshot,
    AecDelayEvidence,
    AecHealth,
    AudioFrame,
    CaptureDrainError,
    DuplexMode,
    NearEndDisposition,
    RenderSubmission,
    RouteKind,
)
from tldw_chatbook.Audio.duplex_transport import DuplexAudioTransport
from tldw_chatbook.Audio import voice_transcription
from tldw_chatbook.Audio.rolling_transcript import (
    TranscriptEngine,
    TranscriptHypothesis,
    TranscriptRevision,
)
from tldw_chatbook.Audio.voice_preprocessor import (
    PendingClassificationWatermark,
    VoicePreprocessor,
)
from tldw_chatbook.Chat.console_speculative_voice_session import (
    ConsoleSpeculativeVoiceSession,
    PreparedSpeculativeVoiceAttempt,
    SpeculativeVoiceAttemptEffects,
    VoicePromotionSeed,
    _LazyHandsFreeTts,
    _NativeStreamingStt,
    _RollingWindowStt,
    _merge_streaming_text,
    _native_stream_realtime_capable,
    _prepare_streaming_candidate,
    _transport_fault_type,
    create_console_speculative_voice_session,
)
from tldw_chatbook.Chat.console_speculative_voice import (
    AdmittedSpeechFrame,
    AttemptDispatchPrepared,
    AttemptGenerationCompleted,
    AttemptOutputDelta,
    AttemptPlaybackTerminal,
    AudioRouteReady,
    AudioRouteRebuildFailed,
    SpeculativeVoiceState,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleLibraryItemScopeSnapshot,
    ConsoleProviderIntent,
    ConsoleResolvedDestination,
    ConsoleTurnLibraryAuthority,
)
from tldw_chatbook.Chat.console_library_policy import (
    AUTOMATIC_LIBRARY_SOURCE_TYPES,
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.Chat.console_turn_context import (
    ConsoleTurnConfigurationSnapshot,
    ConsoleTurnExecutionContext,
)
from tldw_chatbook.Chat.console_trace_provenance import SavedRevisionTraceProvenance
from tldw_chatbook.Chat.console_voice_attempts import (
    AttemptCleanupOutcome,
    VoiceAttemptSnapshot,
    VoiceAttemptRequest,
)
from tldw_chatbook.Chat.console_voice_controls import ControlKind
from tldw_chatbook.Chat.console_voice_supervisor import VoiceDispatchSupervisor
from tldw_chatbook.Chat.console_voice_promotion import (
    ConsoleSessionBindingOrigin,
    VoicePromotionOwner,
)
from tldw_chatbook.TTS.adapter_types import TTSAudioResponse


pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        (
            AecDelayEvidence(
                observed_ns=30_000_000,
                capture_adc_ns=10_000_000,
                render_dac_ns=20_000_000,
                delay_ms=10,
                capture_occupancy_frames=0,
                render_occupancy_frames=0,
                occupancy_bounded=True,
                timing_discontinuity=True,
                clock_drift=False,
                status_flags=("input_underflow",),
            ),
            "status-input_underflow",
        ),
        (
            AecDelayEvidence(
                observed_ns=30_000_000,
                capture_adc_ns=10_000_000,
                render_dac_ns=20_000_000,
                delay_ms=10,
                capture_occupancy_frames=3,
                render_occupancy_frames=0,
                occupancy_bounded=False,
                timing_discontinuity=True,
                clock_drift=False,
            ),
            "buffer-overflow",
        ),
        (
            AecDelayEvidence(
                observed_ns=30_000_000,
                capture_adc_ns=10_000_000,
                render_dac_ns=20_000_000,
                delay_ms=10,
                capture_occupancy_frames=0,
                render_occupancy_frames=0,
                occupancy_bounded=True,
                timing_discontinuity=True,
                clock_drift=True,
            ),
            "clock-drift",
        ),
    ],
)
def test_transport_fault_diagnostic_is_content_free_and_specific(
    evidence: AecDelayEvidence,
    expected: str,
) -> None:
    assert _transport_fault_type(evidence) == expected


def test_streaming_revision_merge_never_discards_an_accepted_prefix() -> None:
    assert _merge_streaming_text("please explain", "please") == "please explain"


@pytest.mark.parametrize(
    ("prewarm_seconds", "expected"),
    [(0.25, True), (1.5, True), (1.5001, False)],
)
def test_native_stream_requires_realtime_preflight(
    prewarm_seconds: float,
    expected: bool,
) -> None:
    assert _native_stream_realtime_capable(prewarm_seconds) is expected


@pytest.mark.asyncio
async def test_slow_native_stream_is_closed_and_falls_back_before_capture(
    monkeypatch,
) -> None:
    from tldw_chatbook.Chat import console_speculative_voice_session as session_module

    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        voice_transcription,
        "_prewarm_parakeet_stream",
        lambda _candidate: 1.5001,
    )

    class Candidate:
        def __init__(self) -> None:
            self.model = SimpleNamespace(transcribe_stream=lambda **_kwargs: None)
            self.closed = False

        def close(self) -> None:
            self.closed = True

    candidate = Candidate()
    service = SimpleNamespace(
        create_streaming_transcriber=lambda **_kwargs: candidate,
    )
    worker = session_module._SerialSttWorker()
    try:
        prepared, native = await _prepare_streaming_candidate(
            service,
            provider="parakeet-mlx",
            model=None,
            language="en",
            serial_worker=worker,
            diagnostic_sink=lambda event, fields: events.append((event, fields)),
        )
    finally:
        await worker.close()

    assert prepared is None
    assert native is False
    assert candidate.closed is True
    assert events == [
        (
            "stt_native_rejected",
            {
                "provider": "parakeet-mlx",
                "status": "fallback",
                "result_type": "slower-than-realtime",
                "latency_ms": 1500,
            },
        )
    ]


@pytest.mark.asyncio
async def test_rolling_stt_failure_uses_real_injected_categorical_diagnostics():
    events: list[tuple[str, dict[str, Any]]] = []

    class Service:
        def transcribe_buffer(self, **_kwargs):
            raise RuntimeError("private transcript and provider secret")

    adapter = _RollingWindowStt(
        Service(),
        provider="fake",
        model=None,
        language="en",
        diagnostic_sink=lambda event, fields: events.append((event, fields)),
    )
    with pytest.raises(RuntimeError):
        await adapter.transcribe_window(
            pcm16=bytes(960), started_ns=0, ended_ns=10_000_000
        )
    assert events == [
        (
            "stt_window_started",
            {
                "provider": "fake",
                "status": "started",
                "duration_ms": 10,
            },
        ),
        (
            "stt_failed",
            {
                "provider": "fake",
                "status": "failed",
                "duration_ms": 10,
                "error_category": "unknown_native",
                "exception_type": "RuntimeError",
            },
        ),
    ]


def _frame(
    sequence: int,
    *,
    render_reference_sequence: int | None = None,
    clock_generation: int = 0,
) -> AudioFrame:
    started_ns = sequence * 10_000_000
    return AudioFrame(
        sequence=sequence,
        started_ns=started_ns,
        ended_ns=started_ns + 10_000_000,
        pcm16=bytes(960),
        clock_generation=clock_generation,
        render_reference_sequence=render_reference_sequence,
    )


class _Effects:
    assistant_rendering = False

    def __init__(self) -> None:
        self.fence_calls = 0

    def fence_all(self) -> None:
        self.fence_calls += 1


class _CoordinatorEffects(_Effects):
    def __init__(self) -> None:
        super().__init__()
        self.dispatches: list[tuple[str, int, str]] = []

    def dispatch_attempt(
        self,
        *,
        turn_id: str,
        attempt_epoch: int,
        transcript: str,
    ) -> None:
        self.dispatches.append((turn_id, attempt_epoch, transcript))

    def classify_spoken_command(self, _transcript: str) -> bool:
        return False

    def voice_dispatch_quarantined(self) -> bool:
        return False

    def fence_attempt(self, _attempt_epoch: int) -> None:
        return None

    def cancel_attempt(self, _attempt_epoch: int) -> None:
        return None

    def abort_output(self, _attempt_epoch: int) -> None:
        return None

    def clear_preview(self, _attempt_epoch: int) -> None:
        return None

    def publish_preview(self, _attempt_epoch: int, _delta: str) -> None:
        return None

    def preserve_draft(self, **_kwargs: Any) -> None:
        return None

    def promote(self, **_kwargs: Any) -> None:
        return None

    def handle_spoken_command(self, **_kwargs: Any) -> None:
        return None

    def rebuild_audio(self, _old_clock_generation: int, _rebuild_epoch: int) -> None:
        return None

    def submit_accepted_voice_turn(self, *_args: Any) -> None:
        return None


class _Transport:
    def __init__(self) -> None:
        self.captures: deque[AudioFrame] = deque()
        self.references: deque[AudioFrame] = deque()
        self.started = 0
        self.aborted = 0
        self.closed = 0
        self.rendered: list[AudioFrame] = []
        self.raise_on_capture = False
        self.clock_generation = 0

    async def start(self) -> None:
        self.started += 1

    async def close(self) -> None:
        self.closed += 1

    async def abort_output(self) -> None:
        self.aborted += 1

    def fence_output(self) -> None:
        pass

    def request_close(self) -> None:
        pass

    def queue_render(self, pcm16: bytes) -> AudioFrame:
        frame = _frame(len(self.rendered))
        assert len(pcm16) == len(frame.pcm16)
        self.rendered.append(frame)
        return frame

    async def drain_capture_through(self, _render_boundary_ns: int) -> object:
        return object()

    def pop_capture(self) -> AudioFrame | None:
        if self.raise_on_capture:
            self.raise_on_capture = False
            raise RuntimeError("content-free pump failure")
        return self.captures.popleft() if self.captures else None

    def pop_render_reference(self) -> AudioFrame | None:
        return self.references.popleft() if self.references else None

    def pop_control_event(self) -> None:
        return None

    def acknowledge_capture(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _Preprocessor:
    mode = DuplexMode.HALF_DUPLEX
    health = AecHealth.DEGRADED

    def __init__(self, **_kwargs: Any) -> None:
        self.render_sequences: list[tuple[int, ...]] = []
        self.safety = AcousticSafetySnapshot(
            AcousticSafetyPath.HALF_DUPLEX,
            DuplexMode.HALF_DUPLEX,
            0,
            False,
            None,
        )

    async def process_capture(
        self,
        _capture: AudioFrame,
        *,
        render_frames: list[AudioFrame],
        assistant_rendering: bool,
    ) -> None:
        assert assistant_rendering is False
        self.render_sequences.append(tuple(frame.sequence for frame in render_frames))


class _RoutePreprocessor(_Preprocessor):
    active_clock_generation = 0

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reset_generations: list[int] = []

    def reset_for_device_route(self, clock_generation: int) -> None:
        self.active_clock_generation = clock_generation
        self.reset_generations.append(clock_generation)
        self.safety = AcousticSafetySnapshot(
            AcousticSafetyPath.HALF_DUPLEX,
            DuplexMode.HALF_DUPLEX,
            clock_generation,
            False,
            None,
        )


class _RouteTransport(_Transport):
    def __init__(self, *, fail_start: bool = False) -> None:
        super().__init__()
        self.fail_start = fail_start

    async def start(self) -> None:
        if self.fail_start:
            raise RuntimeError("route unavailable")
        self.started += 1
        self.clock_generation += 1


class _RecoverableTransport(_Transport):
    def __init__(self) -> None:
        super().__init__()
        self.route_notifications: list[RouteKind] = []

    def notify_route_changed(self, route_kind: RouteKind) -> asyncio.Task[None]:
        self.route_notifications.append(route_kind)
        self.clock_generation += 1

        async def finish_teardown() -> None:
            return None

        return asyncio.create_task(finish_teardown())


class _RetirableTranscript:
    def __init__(
        self,
        turn_id: str,
        *,
        close_started: asyncio.Event,
        close_release: asyncio.Event,
    ) -> None:
        self.turn_id = turn_id
        self._close_started = close_started
        self._close_release = close_release

    def append_admitted_frame(self, _frame: AudioFrame) -> None:
        return None

    async def seal_through(self, _sequence: int) -> Any:
        raise ValueError("no admitted frames")

    async def close(self) -> None:
        self._close_started.set()
        await self._close_release.wait()


def _unused_transcript_factory(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("no speech was admitted")


class _HealthyAec:
    def analyze_render(self, _pcm16: bytes, *, delay_ms: int) -> None:
        assert delay_ms == 20

    def process_capture(self, pcm16: bytes, *, delay_ms: int) -> bytes:
        assert delay_ms == 20
        return pcm16

    def metrics(self) -> dict[str, float]:
        return {
            "erle_db": 25.0,
            "delay_ms": 20.0,
            "delay_estimate_available": 1.0,
            "delay_estimate_refined": 1.0,
            "delay_age_blocks": 1.0,
            "clock_drift": 0.0,
        }

    def reset(self) -> None:
        return None


class _TransitionIsolationMonitor:
    def observe(self, **kwargs: Any) -> AcousticIsolationObservation:
        disposition = (
            NearEndDisposition.PENDING
            if kwargs["assistant_rendering"] and kwargs["near_end_speech"]
            else None
        )
        return AcousticIsolationObservation(
            AcousticSafetySnapshot(
                AcousticSafetyPath.AEC,
                DuplexMode.FULL_DUPLEX,
                kwargs["capture"].clock_generation,
                True,
                None,
            ),
            disposition,
        )

    def reset_for_route(self, _clock_generation: int) -> None:
        return None


def _timed_aec_frame(
    sequence: int,
    *,
    base_ns: int = 0,
    marker: int = 0,
    render_reference_sequence: int | None = None,
) -> AudioFrame:
    started_ns = base_ns + sequence * 10_000_000
    return AudioFrame(
        sequence=sequence,
        started_ns=started_ns,
        ended_ns=started_ns + 10_000_000,
        pcm16=bytes([marker]) * 960,
        clock_generation=0,
        render_reference_sequence=render_reference_sequence,
        delay_evidence=AecDelayEvidence(
            observed_ns=started_ns,
            capture_adc_ns=started_ns,
            render_dac_ns=started_ns + 20_000_000,
            delay_ms=20,
            capture_occupancy_frames=0,
            render_occupancy_frames=0,
            occupancy_bounded=True,
            timing_discontinuity=False,
            clock_drift=False,
        ),
    )


class _RollingStt:
    async def transcribe_window(
        self,
        *,
        pcm16: bytes,
        started_ns: int,
        ended_ns: int,
    ) -> TranscriptHypothesis:
        assert pcm16
        assert ended_ns > started_ns
        return TranscriptHypothesis(
            stable_text="hello from rolling speech",
            covered_through_ns=ended_ns,
        )

    async def abort(self) -> None:
        return None


class _StreamingCandidate:
    def __init__(self, results: list[dict[str, str]]) -> None:
        self._results = deque(results)
        self.closed = False

    def process_audio(self, pcm16: bytes) -> dict[str, str]:
        assert pcm16
        return self._results.popleft()

    def close(self) -> None:
        self.closed = True


class _NativeStreamingService:
    def __init__(self, candidate: Any | None) -> None:
        self.candidate = candidate
        self.create_kwargs: dict[str, Any] = {}
        self.fallback_calls = 0

    def create_streaming_transcriber(self, **kwargs: Any) -> Any:
        self.create_kwargs = kwargs
        return self.candidate

    def transcribe_buffer(self, **kwargs: Any) -> dict[str, str]:
        assert kwargs["audio_data"]
        self.fallback_calls += 1
        return {"text": "fallback words"}


class _AttemptGateway:
    async def stream_chat(self, *_args: Any, **_kwargs: Any):
        yield "Hello there."


class _AudioStream:
    def __init__(self) -> None:
        self._pending = [bytes(960)]

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if not self._pending:
            raise StopAsyncIteration
        return self._pending.pop(0)

    async def aclose(self) -> None:
        self._pending.clear()


class _Synthesizer:
    async def synthesize_hands_free(self, *, text: str) -> TTSAudioResponse:
        assert text == "Hello there."
        return TTSAudioResponse(
            provider_id="test",
            model_id="test",
            audio_format="pcm",
            content_type="audio/pcm",
            byte_stream=_AudioStream(),
            sample_rate=48_000,
            metadata={"channels": 1},
        )


class _RestartGateway:
    def __init__(self) -> None:
        self.calls = 0
        self.first_started = asyncio.Event()
        self.first_cancelled = asyncio.Event()

    async def stream_chat(self, *_args: Any, **_kwargs: Any):
        self.calls += 1
        if self.calls == 1:
            try:
                yield "Old reply!"
                self.first_started.set()
                await asyncio.Event().wait()
            finally:
                self.first_cancelled.set()
            return
        yield "New reply."


class _RestartSynthesizer:
    def __init__(self) -> None:
        self.calls = 0
        self.first_started = asyncio.Event()
        self.first_cancelled = asyncio.Event()

    async def synthesize_hands_free(self, *, text: str) -> TTSAudioResponse:
        self.calls += 1
        if self.calls == 1:
            assert text == "Old reply!"
            self.first_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.first_cancelled.set()
        assert text == "New reply."
        return TTSAudioResponse(
            provider_id="test",
            model_id="test",
            audio_format="pcm",
            content_type="audio/pcm",
            byte_stream=_AudioStream(),
            sample_rate=48_000,
            metadata={"channels": 1},
        )


def _attempt_context() -> ConsoleTurnExecutionContext:
    policy = ConsoleLibraryPolicySnapshot(
        auto_retrieve=ConsoleAutoRetrieve.NEVER,
        assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
        policy_revision=1,
        source="durable",
    )
    scope = ConsoleLibraryItemScopeSnapshot((), (), True)
    configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id="session-voice",
        provider_selection=ConsoleProviderSelection(
            provider="openai",
            explicit_model="gpt-test",
        ),
        library_policy_maximum=policy,
        library_scope_maximum=scope,
    )
    return ConsoleTurnExecutionContext(
        configuration=configuration,
        library_authority=ConsoleTurnLibraryAuthority(
            policy=policy,
            direct_library_tools=True,
            source_types=AUTOMATIC_LIBRARY_SOURCE_TYPES,
            scope_snapshot=scope,
            provider_intent=ConsoleProviderIntent("openai", "gpt-test", None),
            attempt_id="authority-attempt",
        ),
        resolved_destination=ConsoleResolvedDestination(
            provider="openai",
            model="gpt-test",
            endpoint_identity="https://api.openai.com",
            egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
        ),
    )


def _prepared_attempt(epoch: int) -> PreparedSpeculativeVoiceAttempt:
    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="",
        model="gpt-test",
        ready=True,
        execution_key="openai",
    )
    prepared = ConsoleProviderGateway(http_client=object()).prepare_chat_request(  # type: ignore[arg-type]
        resolution,
        [{"role": "user", "content": "Exact rolling transcript"}],
    )
    return PreparedSpeculativeVoiceAttempt(
        VoiceAttemptRequest(epoch, resolution, prepared),
        _attempt_context(),
    )


@pytest.mark.asyncio
async def test_native_streaming_stt_publishes_partial_then_final_revision() -> None:
    candidate = _StreamingCandidate([{"partial": "hello"}, {"final": "hello world"}])
    service = _NativeStreamingService(candidate)
    transcript = TranscriptEngine(
        turn_id="turn-native",
        live_adapter=_NativeStreamingStt(
            service,
            provider="native-test",
            model=None,
            language="en",
        ),
    )

    transcript.append_admitted_frame(_frame(0))
    partial = await asyncio.wait_for(transcript.seal_through(0), timeout=1)
    assert partial.stable_text == ""
    assert partial.revisable_text == "hello"
    assert partial.mode == "live"

    transcript.append_admitted_frame(_frame(1))
    final = await asyncio.wait_for(transcript.seal_through(1), timeout=1)
    assert final.stable_text == "hello world"
    assert final.revisable_text == ""
    assert final.is_final is True

    await transcript.close()
    assert candidate.closed is True


@pytest.mark.asyncio
async def test_parakeet_package_stream_uses_genuine_incremental_api(
    monkeypatch,
) -> None:

    events: list[tuple[str, dict[str, Any]]] = []

    class PackageStream:
        def __init__(self) -> None:
            self.audio: list[bytes] = []
            self.result = SimpleNamespace(text="")

        def add_audio(self, audio: bytes) -> None:
            self.audio.append(audio)
            self.result = SimpleNamespace(text="native package words")

    class PackageContext:
        def __init__(self, stream: PackageStream) -> None:
            self.stream = stream
            self.entered = False
            self.exited = False

        def __enter__(self) -> PackageStream:
            self.entered = True
            return self.stream

        def __exit__(self, *_args: Any) -> None:
            self.exited = True

    class PackageModel:
        def __init__(self) -> None:
            self.stream = PackageStream()
            self.context = PackageContext(self.stream)
            self.stream_kwargs: dict[str, Any] = {}

        def transcribe_stream(self, **kwargs: Any) -> PackageContext:
            self.stream_kwargs = kwargs
            return self.context

    candidate = SimpleNamespace(model=PackageModel())
    service = _NativeStreamingService(candidate)
    monkeypatch.setattr(voice_transcription, "_parakeet_mlx_audio", lambda pcm16: pcm16)
    transcript = TranscriptEngine(
        turn_id="turn-native-package-stream",
        live_adapter=_NativeStreamingStt(
            service,
            provider="parakeet-mlx",
            model=None,
            language="en",
            prepared_candidate=candidate,
            diagnostic_sink=lambda event, fields: events.append((event, fields)),
        ),
    )

    for sequence in range(50):
        transcript.append_admitted_frame(_frame(sequence))
    revision = await asyncio.wait_for(transcript.seal_through(49), timeout=1)

    assert revision.mode == "live"
    assert revision.stable_text + revision.revisable_text == "native package words"
    assert candidate.model.context.entered is True
    assert candidate.model.stream_kwargs == {"context_size": (64, 64)}
    assert candidate.model.stream.audio == [
        b"".join(_frame(i).pcm16 for i in range(50))
    ]
    completed = [fields for event, fields in events if event == "stt_window_completed"]
    assert len(completed) == 1
    assert completed[0]["duration_ms"] == 500
    assert completed[0]["result_size"] == len("native package words")
    assert type(completed[0]["latency_ms"]) is int
    await transcript.close()
    assert candidate.model.context.exited is True


@pytest.mark.asyncio
async def test_parakeet_stream_coalesces_short_interword_chunks(
    monkeypatch,
) -> None:

    class PackageStream:
        def __init__(self) -> None:
            self.audio: list[bytes] = []
            self.result = SimpleNamespace(text="")

        def add_audio(self, audio: bytes) -> None:
            self.audio.append(audio)
            self.result = SimpleNamespace(text="hello how are you doing today")

    class PackageContext:
        def __init__(self, stream: PackageStream) -> None:
            self.stream = stream

        def __enter__(self) -> PackageStream:
            return self.stream

        def __exit__(self, *_args: Any) -> None:
            return None

    class PackageModel:
        def __init__(self) -> None:
            self.stream = PackageStream()

        def transcribe_stream(self, **_kwargs: Any) -> PackageContext:
            return PackageContext(self.stream)

    candidate = SimpleNamespace(model=PackageModel())
    monkeypatch.setattr(voice_transcription, "_parakeet_mlx_audio", lambda pcm16: pcm16)
    transcript = TranscriptEngine(
        turn_id="turn-native-interword-gaps",
        live_adapter=_NativeStreamingStt(
            SimpleNamespace(),
            provider="parakeet-mlx",
            model=None,
            language="en",
            prepared_candidate=candidate,
            quiet_seconds=0.05,
        ),
    )

    for first, count in ((0, 33), (33, 50), (83, 20)):
        for sequence in range(first, first + count):
            transcript.append_admitted_frame(_frame(sequence))
        await asyncio.sleep(0.01)

    revision = await asyncio.wait_for(transcript.seal_through(102), timeout=1)

    assert revision.stable_text + revision.revisable_text == (
        "hello how are you doing today"
    )
    assert candidate.model.stream.audio == [
        b"".join(_frame(i).pcm16 for i in range(103))
    ]
    await transcript.close()


@pytest.mark.asyncio
async def test_parakeet_stream_coalesces_a_real_time_inference_backlog(
    monkeypatch,
) -> None:

    class PackageStream:
        def __init__(self) -> None:
            self.audio: list[bytes] = []
            self.result = SimpleNamespace(text="")

        def add_audio(self, audio: bytes) -> None:
            self.audio.append(audio)
            self.result = SimpleNamespace(text="complete queued utterance")

    class PackageContext:
        def __init__(self, stream: PackageStream) -> None:
            self.stream = stream

        def __enter__(self) -> PackageStream:
            return self.stream

        def __exit__(self, *_args: Any) -> None:
            return None

    class PackageModel:
        def __init__(self) -> None:
            self.stream = PackageStream()

        def transcribe_stream(self, **_kwargs: Any) -> PackageContext:
            return PackageContext(self.stream)

    candidate = SimpleNamespace(model=PackageModel())
    monkeypatch.setattr(voice_transcription, "_parakeet_mlx_audio", lambda pcm16: pcm16)
    transcript = TranscriptEngine(
        turn_id="turn-native-inference-backlog",
        live_adapter=_NativeStreamingStt(
            SimpleNamespace(),
            provider="parakeet-mlx",
            model=None,
            language="en",
            prepared_candidate=candidate,
            quiet_seconds=0.01,
        ),
    )

    # Mirrors the admitted duration in the failed USB run: while one MLX
    # inference is executing, over three seconds of speech can accumulate.
    for sequence in range(336):
        transcript.append_admitted_frame(_frame(sequence))

    revision = await asyncio.wait_for(transcript.seal_through(335), timeout=1)

    assert revision.stable_text + revision.revisable_text == (
        "complete queued utterance"
    )
    assert candidate.model.stream.audio == [
        b"".join(_frame(i).pcm16 for i in range(336))
    ]
    await transcript.close()


@pytest.mark.asyncio
async def test_parakeet_stream_releases_idle_context_before_same_turn_continues(
    monkeypatch,
) -> None:

    active_contexts = 0
    maximum_active_contexts = 0

    class PackageStream:
        def __init__(self, text: str) -> None:
            self._text = text
            self.result = SimpleNamespace(text="")

        def add_audio(self, _audio: bytes) -> None:
            self.result = SimpleNamespace(text=self._text)

    class PackageContext:
        def __init__(self, text: str) -> None:
            self.stream = PackageStream(text)
            self.exited = False

        def __enter__(self) -> PackageStream:
            nonlocal active_contexts, maximum_active_contexts
            active_contexts += 1
            maximum_active_contexts = max(maximum_active_contexts, active_contexts)
            return self.stream

        def __exit__(self, *_args: Any) -> None:
            nonlocal active_contexts
            active_contexts -= 1
            self.exited = True

    class PackageModel:
        def __init__(self) -> None:
            self.contexts: list[PackageContext] = []

        def transcribe_stream(self, **_kwargs: Any) -> PackageContext:
            text = "hello" if not self.contexts else "how are you"
            context = PackageContext(text)
            self.contexts.append(context)
            return context

    candidate = SimpleNamespace(model=PackageModel())
    monkeypatch.setattr(voice_transcription, "_parakeet_mlx_audio", lambda pcm16: pcm16)
    transcript = TranscriptEngine(
        turn_id="turn-native-resumed-speech",
        live_adapter=_NativeStreamingStt(
            SimpleNamespace(),
            provider="parakeet-mlx",
            model=None,
            language="en",
            prepared_candidate=candidate,
            quiet_seconds=0.01,
        ),
    )

    transcript.append_admitted_frame(_frame(0))
    first = await asyncio.wait_for(transcript.seal_through(0), timeout=1)
    assert first.stable_text + first.revisable_text == "hello"
    assert candidate.model.contexts[0].exited is True

    transcript.append_admitted_frame(_frame(1))
    revised = await asyncio.wait_for(transcript.seal_through(1), timeout=1)
    async with asyncio.timeout(1):
        while not candidate.model.contexts[1].exited:
            await asyncio.sleep(0)

    assert revised.stable_text + revised.revisable_text == "hello how are you"
    assert len(candidate.model.contexts) == 2
    assert all(context.exited for context in candidate.model.contexts)
    assert maximum_active_contexts == 1
    await transcript.close()


@pytest.mark.asyncio
async def test_parakeet_model_lifecycle_stays_on_one_worker_thread(
    monkeypatch,
) -> None:
    from tldw_chatbook.Chat import console_speculative_voice_session as session_module

    owner_threads: list[int] = []

    class PackageStream:
        result = SimpleNamespace(text="thread-bound words")

        def add_audio(self, _audio: bytes) -> None:
            owner_threads.append(threading.get_ident())

    class PackageContext:
        def __enter__(self) -> PackageStream:
            owner_threads.append(threading.get_ident())
            return PackageStream()

        def __exit__(self, *_args: Any) -> None:
            owner_threads.append(threading.get_ident())

    class PackageModel:
        def transcribe_stream(self, **_kwargs: Any) -> PackageContext:
            owner_threads.append(threading.get_ident())
            return PackageContext()

    serial_worker = session_module._SerialSttWorker()

    def create_candidate() -> Any:
        owner_threads.append(threading.get_ident())
        return SimpleNamespace(model=PackageModel())

    candidate = await serial_worker.run(create_candidate)
    monkeypatch.setattr(voice_transcription, "_parakeet_mlx_audio", lambda pcm16: pcm16)
    await serial_worker.run(session_module._prewarm_parakeet_stream, candidate)
    transcript = TranscriptEngine(
        turn_id="turn-thread-bound-package-stream",
        live_adapter=_NativeStreamingStt(
            SimpleNamespace(),
            provider="parakeet-mlx",
            model=None,
            language="en",
            prepared_candidate=candidate,
            serial_worker=serial_worker,
        ),
    )

    for sequence in range(50):
        transcript.append_admitted_frame(_frame(sequence))
    revision = await asyncio.wait_for(transcript.seal_through(49), timeout=1)
    await transcript.close()
    await serial_worker.close()

    assert revision.stable_text + revision.revisable_text == "thread-bound words"
    assert len(set(owner_threads)) == 1


@pytest.mark.asyncio
async def test_unavailable_native_streaming_stt_replays_audio_to_rolling_fallback() -> (
    None
):
    service = _NativeStreamingService(None)
    transcript = TranscriptEngine(
        turn_id="turn-fallback",
        live_adapter=_NativeStreamingStt(
            service,
            provider="batch-only-test",
            model=None,
            language="en",
        ),
        fallback_adapter=_RollingWindowStt(
            service,
            provider="batch-only-test",
            model=None,
            language="en",
        ),
    )

    transcript.append_admitted_frame(_frame(0))
    revision = await asyncio.wait_for(transcript.seal_through(0), timeout=1)

    assert revision.mode == "rolling-window"
    assert revision.stable_text + revision.revisable_text == "fallback words"
    assert service.fallback_calls == 1
    await transcript.close()


@pytest.mark.asyncio
async def test_rolling_fallback_replaces_an_overlapping_batch_revision() -> None:
    service = _NativeStreamingService(None)
    fallback_results = deque(["first words", "revised words"])

    def transcribe_buffer(**kwargs: Any) -> dict[str, str]:
        assert kwargs["audio_data"]
        service.fallback_calls += 1
        return {"text": fallback_results.popleft()}

    service.transcribe_buffer = transcribe_buffer  # type: ignore[method-assign]
    transcript = TranscriptEngine(
        turn_id="turn-overlapping-fallback",
        live_adapter=_NativeStreamingStt(
            service,
            provider="batch-only-test",
            model=None,
            language="en",
        ),
        fallback_adapter=_RollingWindowStt(
            service,
            provider="batch-only-test",
            model=None,
            language="en",
        ),
    )

    transcript.append_admitted_frame(_frame(0))
    first = await asyncio.wait_for(transcript.seal_through(0), timeout=1)
    assert first.revisable_text == "first words"

    transcript.append_admitted_frame(_frame(1))
    revised = await asyncio.wait_for(transcript.seal_through(1), timeout=1)
    assert revised.revisable_text == "revised words"
    assert revised.failure_code is None
    assert service.fallback_calls == 2

    await transcript.close()


@pytest.mark.asyncio
async def test_session_route_rebuild_resets_dsp_before_announcing_ready() -> None:
    events: list[object] = []
    transport = _RouteTransport()
    preprocessors: list[_RoutePreprocessor] = []

    def preprocessor_factory(**kwargs: Any) -> _RoutePreprocessor:
        preprocessor = _RoutePreprocessor(**kwargs)
        preprocessors.append(preprocessor)
        return preprocessor

    session = ConsoleSpeculativeVoiceSession(
        transport=transport,
        preprocessor_factory=preprocessor_factory,
        transcript_factory=_unused_transcript_factory,
        effects=_Effects(),
    )

    async def collect(event: object) -> bool:
        events.append(event)
        return True

    session.submit = collect  # type: ignore[method-assign]
    await session.rebuild_audio(old_clock_generation=0, rebuild_epoch=3)

    assert preprocessors[0].reset_generations == [1]
    assert events == [
        AudioRouteReady(
            rebuild_epoch=3,
            clock_generation=1,
            duplex_mode=DuplexMode.HALF_DUPLEX,
            aec_health=AecHealth.DEGRADED,
            safety_path=AcousticSafetyPath.HALF_DUPLEX,
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("overflowed", [False, True])
async def test_session_recycles_only_one_idle_clock_gap_not_overflow(
    overflowed: bool,
) -> None:
    transport = _RecoverableTransport()
    preprocessors: list[_Preprocessor] = []

    def preprocessor_factory(**kwargs: Any) -> _Preprocessor:
        preprocessor = _Preprocessor(**kwargs)
        preprocessors.append(preprocessor)
        return preprocessor

    session = ConsoleSpeculativeVoiceSession(
        transport=transport,
        preprocessor_factory=preprocessor_factory,
        transcript_factory=_unused_transcript_factory,
        effects=_Effects(),
    )
    fault_evidence = AecDelayEvidence(
        observed_ns=30_000_000,
        capture_adc_ns=10_000_000,
        render_dac_ns=20_000_000,
        delay_ms=10,
        capture_occupancy_frames=0,
        render_occupancy_frames=0,
        occupancy_bounded=not overflowed,
        timing_discontinuity=True,
        clock_drift=False,
    )
    transport.captures.append(
        AudioFrame(
            sequence=0,
            started_ns=10_000_000,
            ended_ns=20_000_000,
            pcm16=bytes(960),
            clock_generation=0,
            discontinuity=True,
            delay_evidence=fault_evidence,
        )
    )

    assert await session.process_pending_audio() is True
    assert transport.route_notifications == ([] if overflowed else [RouteKind.DUPLEX])
    assert preprocessors[0].render_sequences == []
    assert session._fenced is overflowed

    transport.captures.append(
        AudioFrame(
            sequence=0,
            started_ns=40_000_000,
            ended_ns=50_000_000,
            pcm16=bytes(960),
            clock_generation=1,
            discontinuity=True,
            delay_evidence=fault_evidence,
        )
    )

    assert await session.process_pending_audio() is (not overflowed)
    assert transport.route_notifications == ([] if overflowed else [RouteKind.DUPLEX])
    assert preprocessors[0].render_sequences == []
    assert session._fenced
    await session.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
async def test_route_rebuild_prebinds_an_already_advanced_transport_generation() -> (
    None
):
    transport = _RouteTransport()
    transport.clock_generation = 1
    preprocessors: list[_RoutePreprocessor] = []

    def preprocessor_factory(**kwargs: Any) -> _RoutePreprocessor:
        preprocessor = _RoutePreprocessor(**kwargs)
        preprocessors.append(preprocessor)
        return preprocessor

    session = ConsoleSpeculativeVoiceSession(
        transport=transport,
        preprocessor_factory=preprocessor_factory,
        transcript_factory=_unused_transcript_factory,
        effects=_Effects(),
    )

    async def accept_ready(_event: object) -> bool:
        return True

    session.submit = accept_ready  # type: ignore[method-assign]
    await session.rebuild_audio(old_clock_generation=0, rebuild_epoch=1)

    assert preprocessors[0].reset_generations == [1, 2]


@pytest.mark.asyncio
async def test_route_rebuild_discards_buffered_old_generation_render_references() -> (
    None
):
    transport = _RouteTransport()
    preprocessors: list[_RoutePreprocessor] = []

    def preprocessor_factory(**kwargs: Any) -> _RoutePreprocessor:
        preprocessor = _RoutePreprocessor(**kwargs)
        preprocessors.append(preprocessor)
        return preprocessor

    session = ConsoleSpeculativeVoiceSession(
        transport=transport,
        preprocessor_factory=preprocessor_factory,
        transcript_factory=_unused_transcript_factory,
        effects=_Effects(),
    )
    session._render_references.append(_frame(9))

    async def accept_ready(_event: object) -> bool:
        return True

    session.submit = accept_ready  # type: ignore[method-assign]
    await session.rebuild_audio(old_clock_generation=0, rebuild_epoch=1)
    transport.references.append(_frame(0, clock_generation=1))
    transport.captures.append(
        _frame(0, render_reference_sequence=0, clock_generation=1)
    )

    assert await session.process_pending_audio() is True
    assert preprocessors[0].render_sequences == [(0,)]


@pytest.mark.asyncio
async def test_session_classification_barrier_defers_release_and_rejects_stale_route() -> (
    None
):
    preprocessors: list[_RoutePreprocessor] = []

    def preprocessor_factory(**kwargs: Any) -> _RoutePreprocessor:
        preprocessor = _RoutePreprocessor(**kwargs)
        preprocessors.append(preprocessor)
        return preprocessor

    session = ConsoleSpeculativeVoiceSession(
        transport=_Transport(),
        preprocessor_factory=preprocessor_factory,
        transcript_factory=_unused_transcript_factory,
        effects=_Effects(),
    )
    watermark = PendingClassificationWatermark(0, 1, 10_000_000, 4)
    session._on_classification_changed(watermark)
    waiter = asyncio.create_task(
        session._drain_pending_classification_through(50_000_000, 0)
    )
    await asyncio.sleep(0)
    assert waiter.done() is False

    session._on_classification_changed(None)
    await asyncio.sleep(0)
    assert waiter.done() is False
    session._release_pending_classification()
    await waiter

    preprocessors[0].active_clock_generation = 1
    session._on_classification_changed(watermark)
    with pytest.raises(CaptureDrainError, match="generation"):
        await session._drain_pending_classification_through(50_000_000, 1)


@pytest.mark.asyncio
async def test_session_classification_failure_dominates_a_new_pending_watermark() -> (
    None
):
    session = ConsoleSpeculativeVoiceSession(
        transport=_Transport(),
        preprocessor_factory=_RoutePreprocessor,
        transcript_factory=_unused_transcript_factory,
        effects=_Effects(),
    )
    session._on_classification_changed(
        PendingClassificationWatermark(0, 1, 10_000_000, 1)
    )
    waiter = asyncio.create_task(
        session._drain_pending_classification_through(50_000_000, 0)
    )
    await asyncio.sleep(0)

    session._release_pending_classification(failed=True)
    session._on_classification_changed(
        PendingClassificationWatermark(0, 2, 20_000_000, 2)
    )

    with pytest.raises(CaptureDrainError, match="classification failed"):
        await asyncio.wait_for(waiter, 0.1)


@pytest.mark.asyncio
async def test_session_publishes_admit_replay_before_classification_release() -> None:
    events: list[tuple[str, int | None]] = []

    class _AdmittingPreprocessor:
        mode = DuplexMode.FULL_DUPLEX
        health = AecHealth.HEALTHY
        active_clock_generation = 0
        safety = AcousticSafetySnapshot(
            AcousticSafetyPath.AEC,
            DuplexMode.FULL_DUPLEX,
            0,
            True,
            None,
        )

        def __init__(self, **kwargs: Any) -> None:
            self._admit = kwargs["on_admitted_frame"]
            self._classification = kwargs["on_classification_changed"]

        async def process_capture(self, *_args: Any, **_kwargs: Any) -> None:
            for sequence in range(5):
                self._admit(_frame(sequence))
            self._classification(None)

    transport = _Transport()
    transport.captures.append(_frame(0))
    effects = _Effects()
    effects.assistant_rendering = True
    session = ConsoleSpeculativeVoiceSession(
        transport=transport,
        preprocessor_factory=_AdmittingPreprocessor,  # type: ignore[arg-type]
        transcript_factory=_unused_transcript_factory,
        effects=effects,
    )

    async def collect(event: object) -> None:
        sequence = getattr(event, "sequence", None)
        events.append((type(event).__name__, sequence))

    session._coordinator.submit = collect  # type: ignore[method-assign]
    session._on_classification_changed(PendingClassificationWatermark(0, 0, 0, 3))

    async def observe_release() -> None:
        await session._drain_pending_classification_through(50_000_000, 0)
        events.append(("released", None))

    waiter = asyncio.create_task(observe_release())
    await asyncio.sleep(0)
    assert await session.process_pending_audio() is True
    await waiter

    admitted = [event for event in events if event[0] == "AdmittedSpeechFrame"]
    assert admitted == [("AdmittedSpeechFrame", sequence) for sequence in range(5)]
    assert events.index(("released", None)) > max(
        events.index(item) for item in admitted
    )


@pytest.mark.asyncio
async def test_pre_boundary_terminal_waits_for_post_render_contiguous_replay() -> None:
    transport = _Transport()
    effects = _Effects()
    effects.assistant_rendering = True
    session = ConsoleSpeculativeVoiceSession(
        transport=transport,
        preprocessor_factory=lambda **kwargs: VoicePreprocessor(
            aec=_HealthyAec(),
            isolation_monitor=_TransitionIsolationMonitor(),
            vad=lambda _frame: True,
            healthy_streak=1,
            **kwargs,
        ),
        transcript_factory=_unused_transcript_factory,
        effects=effects,
    )
    events: list[object] = []

    async def collect(event: object) -> None:
        events.append(event)

    session._coordinator.submit = collect  # type: ignore[method-assign]
    for sequence in range(4):
        transport.references.append(_timed_aec_frame(sequence))
        transport.captures.append(
            _timed_aec_frame(
                sequence,
                render_reference_sequence=sequence,
            )
        )
        assert await session.process_pending_audio() is True

    terminal_waiter = asyncio.create_task(
        session._drain_pending_classification_through(35_000_000, 0)
    )
    await asyncio.sleep(0)
    assert terminal_waiter.done() is False

    effects.assistant_rendering = False
    transport.captures.append(_timed_aec_frame(4, render_reference_sequence=3))
    assert await session.process_pending_audio() is True
    await terminal_waiter

    admitted = [
        event.sequence for event in events if isinstance(event, AdmittedSpeechFrame)
    ]
    assert admitted == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_post_boundary_pending_classification_does_not_delay_terminal() -> None:
    transport = _Transport()
    effects = _CoordinatorEffects()
    effects.assistant_rendering = True
    promotion_release = asyncio.get_running_loop().create_future()
    promoted: list[dict[str, Any]] = []

    def hold_promotion(**kwargs: Any) -> asyncio.Future[None]:
        promoted.append(kwargs)
        return promotion_release

    effects.promote = hold_promotion  # type: ignore[method-assign]
    transcript_frames: dict[str, list[AudioFrame]] = {}

    class _RecordingTranscript:
        def __init__(self, turn_id: str) -> None:
            self.turn_id = turn_id
            transcript_frames[turn_id] = []

        def append_admitted_frame(self, frame: AudioFrame) -> None:
            transcript_frames[self.turn_id].append(frame)

        async def seal_through(self, _sequence: int) -> Any:
            raise AssertionError("terminal sealing is not used in this test")

        async def close(self) -> None:
            return None

    session = ConsoleSpeculativeVoiceSession(
        transport=transport,
        preprocessor_factory=lambda **kwargs: VoicePreprocessor(
            aec=_HealthyAec(),
            isolation_monitor=_TransitionIsolationMonitor(),
            vad=lambda _frame: True,
            healthy_streak=1,
            **kwargs,
        ),
        transcript_factory=lambda turn_id, _callback: _RecordingTranscript(turn_id),
        effects=effects,
    )
    coordinator = session.coordinator
    try:
        await session._accept_admitted_frame(
            _timed_aec_frame(0),
            assistant_rendering=False,
        )
        prior_turn_id = coordinator.snapshot.turn_id
        assert prior_turn_id is not None
        await coordinator._start_promotion(
            turn_id=prior_turn_id,
            attempt_epoch=0,
            transcript="prior request",
            assistant_text="prior answer",
            terminal_boundary_ns=999_000_000,
        )
        assert coordinator.snapshot.state is SpeculativeVoiceState.PROMOTING

        frame_base_ns = 990_000_000
        transport.references.append(_timed_aec_frame(0, base_ns=frame_base_ns))
        transport.captures.append(
            _timed_aec_frame(
                1,
                base_ns=frame_base_ns,
                marker=0,
                render_reference_sequence=0,
            )
        )
        assert await session.process_pending_audio() is True
        await asyncio.wait_for(
            session._drain_pending_classification_through(999_000_000, 0),
            timeout=0.1,
        )
        assert session._pending_classification is not None

        effects.assistant_rendering = False
        for sequence in range(2, 6):
            transport.captures.append(
                _timed_aec_frame(
                    sequence,
                    base_ns=frame_base_ns,
                    marker=sequence - 1,
                    render_reference_sequence=0,
                )
            )
            assert await session.process_pending_audio() is True

        pending_turn_id = coordinator.snapshot.pending_next_turn_id
        assert pending_turn_id is not None
        assert pending_turn_id != prior_turn_id
        assert coordinator.snapshot.pending_next_frame_count == 5
        assert [frame.sequence for frame in transcript_frames[prior_turn_id]] == [0]
        replayed = transcript_frames[pending_turn_id]
        assert [frame.sequence for frame in replayed] == [1, 2, 3, 4, 5]
        assert [frame.pcm16[0] for frame in replayed] == [0, 1, 2, 3, 4]

        promotion_release.set_result(None)
        await coordinator.flush()

        assert len(promoted) == 1
        assert coordinator.snapshot.turn_id == pending_turn_id
        assert coordinator.snapshot.last_admitted_sequence == 5
        assert coordinator.snapshot.state is SpeculativeVoiceState.LISTENING
    finally:
        if not promotion_release.done():
            promotion_release.set_result(None)
        await session._close_once()


@pytest.mark.asyncio
async def test_session_route_rebuild_failure_reports_advanced_fenced_generation() -> (
    None
):
    events: list[object] = []
    transport = _RouteTransport(fail_start=True)
    session = ConsoleSpeculativeVoiceSession(
        transport=transport,
        preprocessor_factory=_RoutePreprocessor,
        transcript_factory=_unused_transcript_factory,
        effects=_Effects(),
    )

    async def collect(event: object) -> bool:
        events.append(event)
        return True

    session.submit = collect  # type: ignore[method-assign]
    await session.rebuild_audio(old_clock_generation=0, rebuild_epoch=4)

    assert events == [
        AudioRouteRebuildFailed(
            rebuild_epoch=4,
            clock_generation=1,
            error_class="RuntimeError",
        )
    ]


@pytest.mark.asyncio
async def test_controller_prepares_saved_capture_on_voice_attempt() -> None:
    gateway = ConsoleProviderGateway(http_client=object())  # type: ignore[arg-type]
    saved_revision_id = "d0e7f8ef-e5f8-4fc7-a7c6-95cc0d4afc37"
    origin = ConsoleSessionBindingOrigin(
        session_id="session-voice",
        session_incarnation=1,
        persisted_conversation_id="conversation-voice",
        conversation_binding_revision=0,
    )

    class _Store:
        active_session_id = "session-voice"
        persistence = SimpleNamespace(
            db=SimpleNamespace(
                get_connection=lambda: SimpleNamespace(
                    execute=lambda _query, _parameters: SimpleNamespace(
                        fetchone=lambda: (
                            saved_revision_id,
                            "conversation-voice",
                            "persisted-history",
                        )
                    )
                )
            )
        )

        def sessions(self) -> list[Any]:
            return [SimpleNamespace(id="session-voice", assistant_kind="default")]

        def pending_attachments(self, _session_id: str) -> tuple[()]:
            return ()

        def snapshot_voice_promotion_origin(self, _session_id: str) -> Any:
            return origin, None, None

        def get_message(self, message_id: str) -> Any:
            assert message_id == "native-history"
            return SimpleNamespace(persisted_message_id="persisted-history")

    controller = object.__new__(ConsoleChatController)
    controller.store = _Store()
    controller.provider_gateway = gateway
    controller._agent_runtime_enabled = False
    controller._agent_bridge = None

    async def capture_context(_session_id: str) -> Any:
        return _prepared_attempt(1).request.resolution, _attempt_context()

    async def pass_messages(messages: Any, *_args: Any) -> Any:
        return messages

    controller._capture_and_resolve_turn_execution_context = capture_context
    controller._provider_messages_for_session = lambda *_args, **_kwargs: [
        {
            "role": "user",
            "content": "saved history",
            "_native_message_id": "native-history",
        }
    ]
    controller._apply_chat_dictionaries = pass_messages
    controller._apply_world_info = pass_messages
    controller._apply_context_summary_compaction = lambda _session_id, messages: (
        messages
    )
    controller._resolve_submit_prefill = lambda _session_id: (None, False)
    controller._has_explicit_staged_evidence = lambda _session_id: False
    controller._provider_continuation_sidecar_for_session = lambda _session_id: ()
    controller._provider_thinking_sidecar_for_session = lambda _session_id: ()
    controller.context_control_inputs = lambda _session_id: (None, None, None)
    controller.capture_policy_snapshot = lambda _session_id: SimpleNamespace(
        effective=SimpleNamespace(enabled=True, detail=CaptureDetail.FULL),
        error_code=None,
        next_detail=None,
    )

    prepared = await controller.prepare_speculative_voice_attempt(
        attempt_epoch=1,
        transcript="exact rolling transcript",
        turn_id="turn-voice",
    )

    assert prepared.requires_pre_dispatch_authority is False
    assert prepared.promotion_seed.capture_eligible_at_dispatch is True
    assert prepared.request.exchange_capture_enabled is True
    assert prepared.request.capture_detail is CaptureDetail.FULL
    assert prepared.request.provisional_trace_attempt is not None
    assert prepared.request.prepared.semantic.capture_durability == "durable"
    provenance = prepared.request.prepared.semantic.provenance
    assert provenance is not None
    assert provenance.compactable[0].messages == (
        SavedRevisionTraceProvenance(saved_revision_id),
    )
    gateway.abandon_provisional_voice_trace(prepared.request.provisional_trace_attempt)


@pytest.mark.asyncio
async def test_attempt_effects_consume_lazy_tts_stream_before_final_submission(
    monkeypatch,
) -> None:
    events: list[object] = []
    previews: list[Any] = []
    promotions: list[dict[str, Any]] = []

    class SubmissionTransport(_Transport):
        def queue_render(self, pcm16):
            super().queue_render(pcm16)
            return RenderSubmission(0, 0, len(self.rendered) - 1)

    transport = SubmissionTransport()
    effects: SpeculativeVoiceAttemptEffects

    async def submit_event(event: object) -> None:
        events.append(event)
        if isinstance(event, AttemptDispatchPrepared):
            effects.start_prepared_attempt(event.attempt_epoch)
        elif isinstance(event, AttemptOutputDelta):
            effects.publish_preview(event.attempt_epoch, event.text)

    async def prepare_attempt(**kwargs: Any) -> PreparedSpeculativeVoiceAttempt:
        return _prepared_attempt(kwargs["attempt_epoch"])

    async def get_tts_service() -> _Synthesizer:
        return _Synthesizer()

    monkeypatch.setattr("tldw_chatbook.TTS.get_tts_service", get_tts_service)

    def promote(**kwargs: Any) -> None:
        promotions.append(kwargs)

    effects = SpeculativeVoiceAttemptEffects(
        submit_event=submit_event,
        prepare_attempt=prepare_attempt,
        gateway=_AttemptGateway(),
        synthesizer=_LazyHandsFreeTts(),
        transport=transport,
        promotion=promote,
        promotion_owner=object(),
        dispatch_supervisor=VoiceDispatchSupervisor(),
        project_preview=previews.append,
        clear_preview=lambda: None,
        submit_accepted_voice_turn=lambda *_args: None,
    )
    effects.publish_audio_capability(AcousticSafetyPath.WARMING)

    effects.dispatch_attempt(
        turn_id="turn-1",
        attempt_epoch=1,
        transcript="Exact rolling transcript",
    )
    async with asyncio.timeout(1):
        while effects.final_render_submission(1) is None:
            await asyncio.sleep(0)

    assert any(isinstance(event, AttemptGenerationCompleted) for event in events)
    assert effects.final_render_submission(1) == RenderSubmission(0, 0, 0)
    # Only the session owner can observe actual DAC completion and emit terminal.
    assert not any(isinstance(event, AttemptPlaybackTerminal) for event in events)
    assert previews[-1].assistant_text == "Hello there."
    assert previews[-1].status == "aec warming"
    assert len(transport.rendered) == 1

    effects.publish_audio_capability(AcousticSafetyPath.ACOUSTIC_ISOLATION)
    assert previews[-1].status == "speaking"

    effects.promote(
        turn_id="turn-1",
        attempt_epoch=1,
        transcript="Exact rolling transcript",
        assistant_text="Hello there.",
        terminal_boundary_ns=10_000_000,
    )
    assert promotions[0]["snapshot"].response_text == "Hello there."
    await effects.close()


@pytest.mark.asyncio
async def test_attempt_effects_cancel_blocked_provider_and_tts_before_restart() -> None:
    events: list[object] = []
    previews: list[Any] = []
    gateway = _RestartGateway()
    synthesizer = _RestartSynthesizer()
    effects: SpeculativeVoiceAttemptEffects

    class SubmissionTransport(_Transport):
        def queue_render(self, pcm16):
            super().queue_render(pcm16)
            return RenderSubmission(0, 0, len(self.rendered) - 1)

    async def submit_event(event: object) -> None:
        events.append(event)
        if isinstance(event, AttemptDispatchPrepared):
            effects.start_prepared_attempt(event.attempt_epoch)
        elif isinstance(event, AttemptOutputDelta):
            effects.publish_preview(event.attempt_epoch, event.text)

    async def prepare_attempt(**kwargs: Any) -> PreparedSpeculativeVoiceAttempt:
        return _prepared_attempt(kwargs["attempt_epoch"])

    effects = SpeculativeVoiceAttemptEffects(
        submit_event=submit_event,
        prepare_attempt=prepare_attempt,
        gateway=gateway,
        synthesizer=synthesizer,
        transport=SubmissionTransport(),
        promotion=lambda **_kwargs: None,
        promotion_owner=object(),
        dispatch_supervisor=VoiceDispatchSupervisor(),
        project_preview=previews.append,
        clear_preview=lambda: None,
        submit_accepted_voice_turn=lambda *_args: None,
    )
    effects.dispatch_attempt(
        turn_id="turn-1",
        attempt_epoch=1,
        transcript="Initial speech",
    )
    async with asyncio.timeout(1):
        await gateway.first_started.wait()
        await synthesizer.first_started.wait()

    effects.fence_attempt(1)
    await effects.abort_output(1)
    async with asyncio.timeout(1):
        assert await effects.cancel_attempt(1) is AttemptCleanupOutcome.CLEAN
        await gateway.first_cancelled.wait()
        await synthesizer.first_cancelled.wait()

    effects.dispatch_attempt(
        turn_id="turn-1",
        attempt_epoch=2,
        transcript="Initial speech plus correction",
    )
    async with asyncio.timeout(1):
        while effects.final_render_submission(2) is None:
            await asyncio.sleep(0)

    assert effects.final_render_submission(1) is None
    assert effects.final_render_submission(2) == RenderSubmission(0, 0, 0)
    assert not any(isinstance(event, AttemptPlaybackTerminal) for event in events)
    assert previews[-1].assistant_text == "New reply."
    assert not any(
        isinstance(item, AttemptGenerationCompleted) and item.attempt_epoch == 1
        for item in events
    )
    await effects.close()


@pytest.mark.asyncio
async def test_session_only_analyzes_render_references_visible_to_each_capture() -> (
    None
):
    transport = _Transport()
    effects = _Effects()
    preprocessors: list[_Preprocessor] = []

    def preprocessor_factory(**kwargs: Any) -> _Preprocessor:
        preprocessor = _Preprocessor(**kwargs)
        preprocessors.append(preprocessor)
        return preprocessor

    session = ConsoleSpeculativeVoiceSession(
        transport=transport,
        preprocessor_factory=preprocessor_factory,
        transcript_factory=_unused_transcript_factory,
        effects=effects,
    )
    transport.captures.extend(
        (
            _frame(0, render_reference_sequence=0),
            _frame(1, render_reference_sequence=1),
        )
    )
    transport.references.extend((_frame(0), _frame(1)))

    assert await session.process_pending_audio() is True
    assert await session.process_pending_audio() is True

    assert preprocessors[0].render_sequences == [(0,), (1,)]


@pytest.mark.asyncio
async def test_session_marks_a_sequential_tts_render_gap_as_a_pause(
    monkeypatch,
) -> None:
    from tldw_chatbook.Chat import console_speculative_voice_session as session_module

    transport = _Transport()
    effects = _Effects()
    effects.assistant_rendering = True
    observations: list[tuple[bool, tuple[int, ...]]] = []
    render_events: list[str] = []
    monkeypatch.setattr(
        session_module,
        "_persist_voice_event",
        lambda event, **fields: (
            render_events.append(fields["status"])
            if event == "render_activity_changed"
            else None
        ),
    )

    class _PauseRecordingPreprocessor(_Preprocessor):
        async def process_capture(
            self,
            _capture: AudioFrame,
            *,
            render_frames: list[AudioFrame],
            assistant_rendering: bool,
        ) -> None:
            observations.append(
                (
                    assistant_rendering,
                    tuple(frame.sequence for frame in render_frames),
                )
            )

    session = ConsoleSpeculativeVoiceSession(
        transport=transport,
        preprocessor_factory=_PauseRecordingPreprocessor,
        transcript_factory=_unused_transcript_factory,
        effects=effects,
    )
    transport.captures.extend(
        (
            _frame(0, render_reference_sequence=0),
            _frame(1, render_reference_sequence=0),
        )
    )
    transport.references.append(_frame(0))

    assert await session.process_pending_audio() is True
    assert await session.process_pending_audio() is True

    assert observations == [(True, (0,)), (False, ())]
    assert render_events == ["active", "paused"]


@pytest.mark.asyncio
async def test_session_persists_content_free_audio_capability_changes(
    monkeypatch,
) -> None:
    from tldw_chatbook.Chat import console_speculative_voice_session as session_module

    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        session_module,
        "_persist_voice_event",
        lambda event, **fields: events.append((event, fields)),
    )
    transport = _Transport()
    effects = _Effects()
    capabilities: list[AcousticSafetyPath] = []
    effects.publish_audio_capability = capabilities.append  # type: ignore[attr-defined]
    session = ConsoleSpeculativeVoiceSession(
        transport=transport,
        preprocessor_factory=_Preprocessor,
        transcript_factory=_unused_transcript_factory,
        effects=effects,
    )
    transport.captures.append(_frame(0))

    assert await session.process_pending_audio() is True

    assert events == [
        (
            "audio_capability_changed",
            {
                "status": "half-duplex",
                "phase": "half-duplex",
                "result_type": "degraded",
            },
        )
    ]
    assert capabilities == [AcousticSafetyPath.HALF_DUPLEX]


@pytest.mark.asyncio
async def test_session_reports_transcript_failure_without_text(monkeypatch) -> None:
    from tldw_chatbook.Chat import console_speculative_voice_session as session_module

    events = []
    monkeypatch.setattr(
        session_module,
        "_persist_voice_event",
        lambda event, **fields: events.append((event, fields)),
    )
    session = ConsoleSpeculativeVoiceSession(
        transport=_Transport(),
        preprocessor_factory=_Preprocessor,
        transcript_factory=_unused_transcript_factory,
        effects=_Effects(),
    )
    submitted = []

    async def submit(revision):
        submitted.append(revision)

    monkeypatch.setattr(session, "submit", submit)
    revision = TranscriptRevision(
        turn_id="private-turn",
        revision_id=1,
        stable_text="private speech",
        revisable_text="",
        covered_through_ns=0,
        mode="rolling-window",
        failure_code="backend_failed",
        is_final=True,
    )
    session._on_transcript_revision(revision)
    await asyncio.sleep(0)
    assert submitted == [revision]
    assert events == [
        (
            "transcript_failed",
            {
                "status": "failed",
                "phase": "rolling-window",
                "error_category": "backend_failed",
            },
        )
    ]


@pytest.mark.asyncio
async def test_audio_pump_failure_fences_callbacks_and_closes_resources() -> None:
    transport = _Transport()
    transport.raise_on_capture = True
    effects = _Effects()
    runtime_failures: list[BaseException] = []
    session = ConsoleSpeculativeVoiceSession(
        transport=transport,
        preprocessor_factory=_Preprocessor,
        transcript_factory=_unused_transcript_factory,
        effects=effects,
        on_runtime_failure=runtime_failures.append,
        pump_interval_seconds=0.001,
    )

    await session.enter(capture_live=False)
    async with asyncio.timeout(1):
        await session._pump_task

    assert effects.fence_calls == 1
    assert transport.aborted == 1
    assert transport.closed == 1
    assert [str(failure) for failure in runtime_failures] == [
        "content-free pump failure"
    ]
    assert await session.submit(object()) is False

    await session.fence_and_close(ControlKind.TEARDOWN)
    assert transport.closed == 1


@pytest.mark.asyncio
async def test_session_prepares_stt_before_opening_capture() -> None:
    transport = _Transport()
    prepare_started = asyncio.Event()
    release_prepare = asyncio.Event()
    events: list[str] = []

    async def prepare_transcription() -> None:
        events.append("prepare")
        prepare_started.set()
        await release_prepare.wait()

    async def close_transcription() -> None:
        events.append("close")

    original_start = transport.start

    async def start_transport() -> None:
        events.append("capture")
        await original_start()

    transport.start = start_transport  # type: ignore[method-assign]
    session = ConsoleSpeculativeVoiceSession(
        transport=transport,
        preprocessor_factory=_Preprocessor,
        transcript_factory=_unused_transcript_factory,
        effects=_Effects(),
        prepare_transcription=prepare_transcription,
        close_transcription=close_transcription,
    )

    enter_task = asyncio.create_task(session.enter(capture_live=False))
    await prepare_started.wait()
    assert transport.started == 0

    release_prepare.set()
    await enter_task
    assert events == ["prepare", "capture"]

    await session.fence_and_close(ControlKind.TEARDOWN)
    assert events == ["prepare", "capture", "close"]


@pytest.mark.asyncio
async def test_session_close_waits_for_retired_transcript_cleanup() -> None:
    close_started = [asyncio.Event() for _ in range(3)]
    close_release = [asyncio.Event() for _ in range(3)]
    transcripts: list[_RetirableTranscript] = []

    def transcript_factory(turn_id: str, _publish: Any) -> _RetirableTranscript:
        index = len(transcripts)
        transcript = _RetirableTranscript(
            turn_id,
            close_started=close_started[index],
            close_release=close_release[index],
        )
        transcripts.append(transcript)
        return transcript

    session = ConsoleSpeculativeVoiceSession(
        transport=_Transport(),
        preprocessor_factory=_Preprocessor,
        transcript_factory=transcript_factory,
        effects=_Effects(),
        pump_interval_seconds=60,
    )
    await session.enter(capture_live=False)
    session._transcript_for("turn-1")
    session._transcript_for("turn-2")
    session._transcript_for("turn-3")
    await close_started[0].wait()
    close_release[1].set()
    close_release[2].set()

    close_task = session.fence_and_close(ControlKind.TEARDOWN)
    done, _pending = await asyncio.wait({close_task}, timeout=0.05)
    assert not done

    for release in close_release:
        release.set()
    await close_task


@pytest.mark.asyncio
async def test_real_transport_aec_rolling_stt_and_coordinator_dispatch() -> None:
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    effects = _CoordinatorEffects()
    transcripts: list[TranscriptEngine] = []

    def transcript_factory(turn_id: str, publish: Any) -> TranscriptEngine:
        transcript = TranscriptEngine(
            turn_id=turn_id,
            rolling_adapter=_RollingStt(),
            on_revision=publish,
        )
        transcripts.append(transcript)
        return transcript

    session = ConsoleSpeculativeVoiceSession(
        transport=transport,
        preprocessor_factory=lambda **kwargs: VoicePreprocessor(
            aec=_HealthyAec(),
            vad=lambda _frame: True,
            healthy_streak=1,
            **kwargs,
        ),
        transcript_factory=transcript_factory,
        effects=effects,
        response_eagerness_ms=500,
        initial_duplex_mode=DuplexMode.HALF_DUPLEX,
        pump_interval_seconds=60,
    )
    await session.enter(capture_live=False)
    await asyncio.sleep(0)

    backend.stream.emit_capture(b"\x01\x00" * 480)
    assert await session.process_pending_audio() is True
    await transcripts[0].wait_idle()
    async with asyncio.timeout(1):
        while not effects.dispatches:
            await asyncio.sleep(0.01)

    assert effects.dispatches[0][2] == "hello from rolling speech"
    assert session.coordinator.snapshot.duplex_mode is DuplexMode.FULL_DUPLEX

    await session.fence_and_close(ControlKind.TEARDOWN)
    assert backend.stream.closed is True


@pytest.mark.asyncio
async def test_production_factory_rejects_batch_backed_pseudo_streaming(
    monkeypatch,
) -> None:
    from tldw_chatbook.Audio import parakeet_voice_worker as process_module
    from tldw_chatbook.Audio import voice_process_entry as entry_module
    from tldw_chatbook.Audio import duplex_transport as transport_module
    from tldw_chatbook.Audio import voice_preprocessor as preprocessor_module
    from tldw_chatbook.Audio.rolling_transcript import TranscriptBackendFailure
    from tldw_chatbook.Chat.console_voice_process import bootstrap_record

    transport = _Transport()

    class _Candidate:
        closed = False

        def add_audio(self, _samples: list[float], **_kwargs: Any) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    candidate = _Candidate()

    class _TranscriptionService:
        def __init__(self) -> None:
            self.create_calls: list[dict[str, Any]] = []
            self.fail = False
            self.cleanup_outcome = AttemptCleanupOutcome.CLEAN

        def create_streaming_transcriber(self, **kwargs: Any) -> Any:
            self.create_calls.append(kwargs)
            return candidate

        def transcribe_buffer(self, **_kwargs: Any) -> dict[str, str]:
            if self.fail:
                raise RuntimeError("private transcript and provider secret")
            return {"text": "fallback"}

        def close(self) -> None:
            return None

    transcription_service = _TranscriptionService()

    class _Pipe:
        generation = 7
        request_id = "a" * 32

        def send(self, _op: str, **_fields: Any) -> None:
            return None

    monkeypatch.setattr(
        process_module,
        "LocalVoiceSttProcess",
        lambda **_kwargs: transcription_service,
    )
    monkeypatch.setattr(transport_module, "DuplexAudioTransport", lambda: transport)
    monkeypatch.setattr(preprocessor_module, "create_webrtc_vad", lambda **_: None)
    session = entry_module._ProductionSession(
        bootstrap_record(
            generation=7,
            request_id="a" * 32,
            stt_provider="faster-whisper",
            stt_model=None,
            language="en",
            response_eagerness_ms=700,
            aec_enabled=False,
            vad_aggressiveness=2,
            vad_preroll_ms=240,
        )
    )
    session.bind_pipe(_Pipe(), lambda _code: None)

    assert transcription_service.create_calls == []
    assert await session.prepare() is False
    assert transcription_service.create_calls == [
        {
            "provider": "faster-whisper",
            "model": None,
            "source_lang": "en",
        }
    ]
    assert candidate.closed
    transcript = session.core._transcript_for("turn-factory")
    assert isinstance(transcript, TranscriptEngine)
    assert transcript._mode == "rolling-window"
    assert transcript._rolling_min_window_ns == 500_000_000
    assert transcript._rolling_debounce_seconds == 0.12
    transcript.append_admitted_frame(_frame(0))
    revision = await asyncio.wait_for(transcript.seal_through(0), timeout=1)
    assert revision.revisable_text == "fallback"
    transcription_service.fail = True
    failed_transcript = session.core._transcript_for("turn-factory-failure")
    failed_transcript.append_admitted_frame(_frame(1))
    with pytest.raises(TranscriptBackendFailure) as failure:
        await asyncio.wait_for(failed_transcript.seal_through(1), timeout=1)
    assert failure.value.code == "backend_failed"
    assert str(failure.value) == "backend_failed"
    assert "private transcript" not in repr(failure.value)
    assert "provider secret" not in repr(failure.value)
    assert transcription_service.create_calls == [
        {
            "provider": "faster-whisper",
            "model": None,
            "source_lang": "en",
        }
    ]
    native, resources = session.begin_close()
    assert await native is True
    assert await resources is AttemptCleanupOutcome.CLEAN


@pytest.mark.asyncio
async def test_production_parakeet_is_isolated_and_reaped_before_transcript_cleanup(
    monkeypatch,
) -> None:
    from Tests.Audio.test_parakeet_voice_worker import fake_runtime
    from tldw_chatbook.Audio import duplex_transport as transport_module
    from tldw_chatbook.Audio import parakeet_voice_worker as process_module
    from tldw_chatbook.Audio import voice_preprocessor as preprocessor_module
    from tldw_chatbook.Audio import voice_process_entry as entry_module
    from tldw_chatbook.Chat.console_voice_process import bootstrap_record

    services = []

    class Process(process_module.ParakeetVoiceProcess):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._runtime_factory = fake_runtime
            services.append(self)

    class _Pipe:
        generation = 7
        request_id = "a" * 32

        def send(self, _op: str, **_fields: Any) -> None:
            return None

    monkeypatch.setattr(process_module, "LocalVoiceSttProcess", Process)
    monkeypatch.setattr(transport_module, "DuplexAudioTransport", _Transport)
    monkeypatch.setattr(preprocessor_module, "create_webrtc_vad", lambda **_: None)
    session = entry_module._ProductionSession(
        bootstrap_record(
            generation=7,
            request_id="a" * 32,
            stt_provider="parakeet-mlx",
            stt_model="test-model",
            language="en",
            response_eagerness_ms=700,
            aec_enabled=False,
            vad_aggressiveness=2,
            vad_preroll_ms=240,
        )
    )
    session.bind_pipe(_Pipe(), lambda _code: None)
    assert await session.prepare() is True
    service = services[0]
    try:
        engine = session.core._transcript_for("process-turn")
        assert engine._mode == "live"
        # Model a transcript waiting for its decoder during cleanup. The process
        # must be reaped before that wait; otherwise a toggle can hang for 30 s.
        cleanup_observations = []

        async def close_transcript():
            cleanup_observations.append(service._process.is_alive())

        monkeypatch.setattr(engine, "close", close_transcript)
        native, resources = session.begin_close()
        assert await native is True
        assert await resources is AttemptCleanupOutcome.CLEAN
        assert cleanup_observations == [False]
        assert service.reaped
    finally:
        await asyncio.to_thread(service.close)


def test_production_promotion_callback_binds_no_render_boundary(
    monkeypatch,
) -> None:
    from tldw_chatbook import config as config_module
    from tldw_chatbook.Chat import console_voice_input as voice_input_module
    from tldw_chatbook.Chat import console_voice_promotion as promotion_module
    from tldw_chatbook.Chat.console_voice_process import ConsoleVoiceProcess

    promoted: list[tuple[Any, Any]] = []

    class _Winner:
        def __init__(self, _owner: Any) -> None:
            return None

        def promote(self, context: Any, snapshot: Any, *, on_claim: Any = None) -> str:
            assert on_claim is None
            promoted.append((context, snapshot))
            return "promoted"

    class _Store:
        active_session_id = "session-voice"

        def sessions(self) -> list[Any]:
            return [SimpleNamespace(id=self.active_session_id)]

        def active_session_epoch(self) -> int:
            return 1

    class _Controller:
        provider_gateway = object()
        store = _Store()

        async def prepare_speculative_voice_attempt(self, **_kwargs: Any) -> Any:
            raise AssertionError("attempt preparation must remain lazy")

        def submit_accepted_voice_turn(self, *_args: Any) -> None:
            return None

    class _View:
        _hands_free = SimpleNamespace(_qualified_voice_generation=11)
        is_mounted = True
        _console_dictation_state = "idle"

        def _ensure_console_chat_controller(self) -> _Controller:
            return _Controller()

    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda _section, _key=None, default=None: default,
    )
    monkeypatch.setattr(
        voice_input_module,
        "resolve",
        lambda: SimpleNamespace(provider="faster-whisper", model=None, language="en"),
    )
    monkeypatch.setattr(promotion_module, "VoiceWinningPromotion", _Winner)

    owner = VoicePromotionOwner(lambda: object())
    app = SimpleNamespace()
    view = _View()
    session = create_console_speculative_voice_session(
        app_instance=app,
        view=view,
        promotion_owner=owner,
        dispatch_supervisor=VoiceDispatchSupervisor(),
        project_preview=lambda _projection: None,
        clear_preview=lambda: None,
    )
    base = _prepared_attempt(1)
    prepared = PreparedSpeculativeVoiceAttempt(
        base.request,
        base.frozen_session_context,
        promotion_seed=VoicePromotionSeed(
            promotion_id="promotion-1",
            attempt_id="attempt-1",
            terminal_boundary_id="terminal-1",
            origin=ConsoleSessionBindingOrigin("session-voice", 1, None, 0),
            expected_native_leaf_id=None,
            expected_persisted_leaf_id=None,
            capture_eligible_at_dispatch=False,
        ),
    )
    snapshot = VoiceAttemptSnapshot(1, "Assistant reply.")

    assert isinstance(session, ConsoleVoiceProcess)
    assert session._bootstrap.header["generation"] == 11
    assert session in app.console_runtime.voice_process_supervisor._sessions
    _, _, _, promotion, _ = session._bindings
    assert promotion.keywords["terminal_boundary_ns"] is None
    result = promotion(
        prepared=prepared,
        snapshot=snapshot,
        transcript="Exact rolling transcript",
        assistant_text="Assistant reply.",
    )

    assert result == "promoted"
    assert promoted[0][0].assistant_text == "Assistant reply."
    assert promoted[0][1] is snapshot
