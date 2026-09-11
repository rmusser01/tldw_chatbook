from __future__ import annotations

import asyncio
from dataclasses import dataclass
import heapq
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from tldw_chatbook.Audio.duplex_contracts import (
    AcousticSafetyPath,
    AecHealth,
    DeviceRouteChanged,
    DrainReceipt,
    DuplexMode,
    RouteKind,
)
from tldw_chatbook.Audio.rolling_transcript import TranscriptRevision
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
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
from tldw_chatbook.Chat.console_prepared_request import PreparedProviderRequest
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.Chat.console_speculative_voice import (
    AdmittedSpeechFrame,
    AttemptDispatchPrepared,
    AttemptOutputDelta,
    AudioRouteReady,
    ControlAction,
    ControlKind,
    SpeculativeTurnCoordinator,
    SpeculativeVoiceState,
)
from tldw_chatbook.Chat.console_turn_context import (
    ConsoleTurnCustodyRequest,
    ConsoleTurnConfigurationSnapshot,
    ConsoleTurnExecutionContext,
)
from tldw_chatbook.Chat.console_voice_attempts import (
    AttemptCleanupOutcome,
    VoiceAttemptToolRequest,
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Read weather",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


@dataclass(slots=True)
class _Scheduled:
    deadline_ns: int
    order: int
    callback: Callable[[], None]
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


class _Scheduler:
    def __init__(self) -> None:
        self.now_ns = 0
        self._order = 0
        self._scheduled: list[tuple[int, int, _Scheduled]] = []

    def call_at_ns(
        self,
        deadline_ns: int,
        callback: Callable[[], None],
    ) -> _Scheduled:
        handle = _Scheduled(deadline_ns, self._order, callback)
        self._order += 1
        heapq.heappush(self._scheduled, (deadline_ns, handle.order, handle))
        return handle

    def advance_ms(self, milliseconds: int) -> None:
        self.now_ns += milliseconds * 1_000_000
        while self._scheduled and self._scheduled[0][0] <= self.now_ns:
            _, _, handle = heapq.heappop(self._scheduled)
            if not handle.cancelled:
                handle.callback()


class _Effects:
    def __init__(self) -> None:
        self.operations: list[tuple[Any, ...]] = []
        self.dispatches: list[tuple[str, int, str]] = []
        self.started_prepared: list[int] = []
        self.accepted_calls: list[tuple[str, ConsoleTurnExecutionContext]] = []
        self.accepted_release: asyncio.Event | None = None
        self.accepted_started = asyncio.Event()
        self.accepted_finished = asyncio.Event()
        self.accepted_cancelled = False
        self.accepted_task: asyncio.Task[None] | None = None
        self.accepted_attempts = 0
        self.accepted_error: Exception | None = None
        self.abort_error: Exception | None = None
        self.audio_rebuild_allowed = False

    def dispatch_attempt(
        self,
        *,
        turn_id: str,
        attempt_epoch: int,
        transcript: str,
    ) -> None:
        self.operations.append(("dispatch", attempt_epoch))
        self.dispatches.append((turn_id, attempt_epoch, transcript))

    def fence_attempt(self, attempt_epoch: int) -> None:
        self.operations.append(("fence", attempt_epoch))

    def start_prepared_attempt(self, attempt_epoch: int) -> None:
        self.started_prepared.append(attempt_epoch)

    def cancel_attempt(
        self, attempt_epoch: int
    ) -> asyncio.Future[AttemptCleanupOutcome]:
        self.operations.append(("cancel", attempt_epoch))
        result = asyncio.get_running_loop().create_future()
        result.set_result(AttemptCleanupOutcome.CLEAN)
        return result

    def abort_output(self, attempt_epoch: int) -> None:
        self.operations.append(("abort", attempt_epoch))
        if self.abort_error is not None:
            raise self.abort_error

    def clear_preview(self, attempt_epoch: int) -> None:
        self.operations.append(("clear", attempt_epoch))

    def publish_preview(self, attempt_epoch: int, delta: str) -> None:
        self.operations.append(("preview", attempt_epoch, delta))

    def preserve_draft(self, **kwargs: Any) -> None:
        if kwargs.get("reason") != "accepted_handoff_failed":
            raise AssertionError(
                "effect-barrier paths must not preserve another failure draft"
            )
        self.operations.append(
            (
                "preserve",
                kwargs["turn_id"],
                kwargs["transcript"],
                kwargs["reason"],
            )
        )

    def promote(self, **_kwargs: Any) -> None:
        raise AssertionError("effectful turns must not use provisional promotion")

    def classify_spoken_command(self, _transcript: str) -> bool:
        return False

    def handle_spoken_command(self, **_kwargs: Any) -> None:
        raise AssertionError("the fixture transcript is not a command")

    def rebuild_audio(self, old_clock_generation: int, rebuild_epoch: int) -> None:
        if not self.audio_rebuild_allowed:
            raise AssertionError("audio rebuild is outside this fixture")
        self.operations.append(("rebuild", old_clock_generation, rebuild_epoch))

    async def drain_capture_through(self, render_boundary_ns: int) -> DrainReceipt:
        return DrainReceipt(render_boundary_ns + 1, 0, 0, 0, 0)

    async def seal_transcript_through(
        self, _admitted_sequence: int
    ) -> TranscriptRevision:
        turn_id, _, transcript = self.dispatches[-1]
        return TranscriptRevision(
            turn_id=turn_id,
            revision_id=100,
            stable_text=transcript,
            revisable_text="",
            covered_through_ns=10**18,
            mode="live",
            is_final=True,
        )

    def voice_dispatch_quarantined(self) -> bool:
        return False

    def submit_accepted_voice_turn(
        self,
        exact_user_text: str,
        frozen_session_context: ConsoleTurnExecutionContext,
    ) -> str:
        self.accepted_attempts += 1
        self.operations.append(("accepted_attempt", exact_user_text))
        if self.accepted_error is not None:
            raise self.accepted_error
        self.operations.append(("accepted", exact_user_text))
        self.accepted_calls.append((exact_user_text, frozen_session_context))
        self.accepted_started.set()
        self.accepted_task = asyncio.create_task(self._finish_accepted())
        return "runtime-turn"

    async def _finish_accepted(self) -> None:
        try:
            if self.accepted_release is not None:
                await self.accepted_release.wait()
        except asyncio.CancelledError:
            self.accepted_cancelled = True
            raise
        self.accepted_finished.set()


def _context(
    *,
    automatic_retrieval: bool = False,
) -> ConsoleTurnExecutionContext:
    auto_retrieve = (
        ConsoleAutoRetrieve.AUTOMATIC
        if automatic_retrieval
        else ConsoleAutoRetrieve.NEVER
    )
    policy = ConsoleLibraryPolicySnapshot(
        auto_retrieve=auto_retrieve,
        assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
        policy_revision=4,
        source="durable",
    )
    scope = ConsoleLibraryItemScopeSnapshot((), (), True)
    configuration = ConsoleTurnConfigurationSnapshot.capture(
        session_id="session-voice",
        provider_selection=ConsoleProviderSelection(
            provider="openai",
            explicit_model="gpt-test",
            system_prompt="Frozen system prompt",
        ),
        tool_configuration={"agent_runtime_enabled": True},
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


def _prepared() -> PreparedProviderRequest:
    resolution = ConsoleProviderResolution(
        provider="openai",
        base_url="",
        model="gpt-test",
        ready=True,
        execution_key="openai",
    )
    return ConsoleProviderGateway(  # type: ignore[arg-type]
        http_client=object()
    ).prepare_chat_request(
        resolution,
        [
            {"role": "system", "content": "Frozen system prompt"},
            {"role": "user", "content": "Exact rolling transcript"},
        ],
        tools=TOOLS,
    )


async def _coordinator(
    *,
    automatic_retrieval: bool = False,
    requires_citation_creation: bool = False,
    requires_pre_dispatch_authority: bool = False,
) -> tuple[
    SpeculativeTurnCoordinator,
    _Scheduler,
    _Effects,
    ConsoleTurnExecutionContext,
]:
    scheduler = _Scheduler()
    effects = _Effects()
    context = _context(automatic_retrieval=automatic_retrieval)
    coordinator = SpeculativeTurnCoordinator(
        effects=effects,
        scheduler=scheduler,
        response_eagerness_ms=700,
        frozen_session_context=context,
        prepared_request=_prepared(),
        requires_citation_creation=requires_citation_creation,
        requires_pre_dispatch_authority=requires_pre_dispatch_authority,
    )
    await coordinator.start()
    return coordinator, scheduler, effects, context


def _speech(sequence: int, *, started_ns: int) -> AdmittedSpeechFrame:
    return AdmittedSpeechFrame(
        sequence=sequence,
        started_ns=started_ns,
        ended_ns=started_ns + 10_000_000,
        clock_generation=0,
    )


async def _fresh_transcript(
    coordinator: SpeculativeTurnCoordinator,
    *,
    revision_id: int,
    text: str,
) -> None:
    snapshot = coordinator.snapshot
    assert snapshot.turn_id is not None
    await coordinator.submit(
        TranscriptRevision(
            turn_id=snapshot.turn_id,
            revision_id=revision_id,
            stable_text="",
            revisable_text=text,
            covered_through_ns=snapshot.last_speech_end_ns or 0,
            mode="live",
        )
    )


async def _start_tool_attempt(
    coordinator: SpeculativeTurnCoordinator,
    scheduler: _Scheduler,
) -> int:
    await coordinator.submit(_speech(0, started_ns=scheduler.now_ns))
    await _fresh_transcript(
        coordinator,
        revision_id=1,
        text="Exact rolling transcript",
    )
    scheduler.advance_ms(700)
    await coordinator.flush()
    epoch = coordinator.snapshot.current_attempt_epoch
    assert epoch is not None
    return epoch


def test_public_accepted_voice_handoff_enters_runtime_custody() -> None:
    controller = object.__new__(ConsoleChatController)
    context = _context()
    accepted: list[ConsoleTurnCustodyRequest] = []

    class Store:
        def session_one_shot_prefill_snapshot(self, session_id: str):
            assert session_id == context.session_id
            return "captured prefill", 7

        def pending_attachments(self, session_id: str):
            assert session_id == context.session_id
            return [SimpleNamespace(attachment_id="attachment-1")]

    class Runtime:
        def snapshot_console_staged_evidence(self):
            return "captured launch", 9, 4

        def accept_turn(self, request: ConsoleTurnCustodyRequest) -> str:
            accepted.append(request)
            return request.turn_id

    controller.store = Store()  # type: ignore[assignment]
    controller.app = SimpleNamespace(console_runtime=Runtime())

    result = controller.submit_accepted_voice_turn(
        "Exact rolling transcript",
        context,
    )

    assert result == accepted[0].turn_id
    request = accepted[0]
    assert request.session_id == context.session_id
    assert request.draft == "Exact rolling transcript"
    assert request.configuration is context.configuration
    assert request.attachment_ids == ("attachment-1",)
    assert request.one_shot_prefill == "captured prefill"
    assert request.one_shot_prefill_revision == 7
    assert request.staged_evidence_launch == "captured launch"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "effect_context",
    (
        {"automatic_retrieval": True},
        {"requires_citation_creation": True},
        {"requires_pre_dispatch_authority": True},
    ),
)
async def test_effectful_context_never_dispatches_speculatively_and_hands_off_once(
    effect_context: dict[str, bool],
) -> None:
    coordinator, scheduler, effects, context = await _coordinator(**effect_context)
    try:
        await coordinator.submit(_speech(0, started_ns=0))
        await _fresh_transcript(
            coordinator,
            revision_id=1,
            text="Exact rolling transcript",
        )

        assert (
            coordinator.snapshot.state is SpeculativeVoiceState.WAITING_FOR_STABLE_TURN
        )
        scheduler.advance_ms(1_999)
        await coordinator.flush()
        assert effects.dispatches == []
        assert effects.accepted_calls == []

        scheduler.advance_ms(1)
        await coordinator.flush()
        await effects.accepted_started.wait()

        assert effects.dispatches == []
        assert effects.accepted_calls == [("Exact rolling transcript", context)]
        scheduler.advance_ms(5_000)
        await coordinator.flush()
        assert len(effects.accepted_calls) == 1
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_deferred_attempt_preparation_selects_provisional_or_stable_handoff() -> (
    None
):
    scheduler = _Scheduler()
    effects = _Effects()
    coordinator = SpeculativeTurnCoordinator(
        effects=effects,
        scheduler=scheduler,
        response_eagerness_ms=700,
        deferred_attempt_preparation=True,
    )
    await coordinator.start()
    try:
        await coordinator.submit(_speech(0, started_ns=0))
        await _fresh_transcript(
            coordinator,
            revision_id=1,
            text="Exact rolling transcript",
        )
        scheduler.advance_ms(700)
        await coordinator.flush()
        epoch = coordinator.snapshot.current_attempt_epoch
        assert epoch is not None

        provisional_context = _context()
        await coordinator.submit(
            AttemptDispatchPrepared(epoch, provisional_context, _prepared())
        )
        assert effects.started_prepared == [epoch]

        await coordinator.submit(
            VoiceAttemptToolRequest(
                epoch,
                ({"id": "call-dynamic", "function": {"name": "weather"}},),
            )
        )
        scheduler.advance_ms(1_300)
        await coordinator.flush()

        assert effects.accepted_calls == [
            ("Exact rolling transcript", provisional_context)
        ]
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_first_tool_request_freezes_output_then_discards_before_handoff() -> None:
    coordinator, scheduler, effects, context = await _coordinator()
    try:
        epoch = await _start_tool_attempt(coordinator, scheduler)
        await coordinator.submit(AttemptOutputDelta(epoch, "safe preamble"))
        tool_request = VoiceAttemptToolRequest(
            epoch,
            (
                {
                    "id": "call-speculative",
                    "type": "function",
                    "function": {"name": "weather", "arguments": "{}"},
                },
            ),
        )
        await coordinator.submit(tool_request)
        await coordinator.submit(AttemptOutputDelta(epoch, "must not speak"))

        assert (
            coordinator.snapshot.state is SpeculativeVoiceState.WAITING_FOR_STABLE_TURN
        )
        assert ("preview", epoch, "safe preamble") in effects.operations
        assert ("preview", epoch, "must not speak") not in effects.operations
        assert ("abort", epoch) in effects.operations
        assert effects.accepted_calls == []

        scheduler.advance_ms(1_299)
        await coordinator.flush()
        assert effects.accepted_calls == []
        scheduler.advance_ms(1)
        await coordinator.flush()
        await effects.accepted_started.wait()

        assert effects.accepted_calls == [("Exact rolling transcript", context)]
        assert ("fence", epoch) in effects.operations
        assert ("cancel", epoch) in effects.operations
        accepted_index = effects.operations.index(
            ("accepted", "Exact rolling transcript")
        )
        assert effects.operations.index(("fence", epoch)) < accepted_index
        assert effects.operations.index(("cancel", epoch)) < accepted_index
        assert all(
            "call-speculative" not in repr(call) for call in effects.accepted_calls
        )
        assert not hasattr(effects, "tool_executor")
        assert not hasattr(effects, "approval_hook")
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_synchronous_custody_failure_retains_one_recoverable_exact_turn() -> None:
    coordinator, scheduler, effects, _ = await _coordinator(automatic_retrieval=True)
    effects.accepted_error = RuntimeError("runtime refused custody")
    try:
        await coordinator.submit(_speech(0, started_ns=0))
        await _fresh_transcript(
            coordinator,
            revision_id=1,
            text="Exact recoverable transcript",
        )

        scheduler.advance_ms(2_000)
        await coordinator.flush()

        snapshot = coordinator.snapshot
        assert snapshot.state is SpeculativeVoiceState.DRAFT_SUSPENDED
        assert snapshot.turn_id is not None
        assert snapshot.transcript_text == "Exact recoverable transcript"
        assert snapshot.failure_class == "accepted_handoff_failed"
        assert effects.accepted_attempts == 1
        assert effects.accepted_calls == []
        assert any(
            operation[0] == "preserve"
            and operation[2] == "Exact recoverable transcript"
            for operation in effects.operations
        )

        scheduler.advance_ms(5_000)
        await coordinator.flush()
        assert effects.accepted_attempts == 1
    finally:
        await coordinator.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_triggered", (False, True))
async def test_route_rebuild_preserves_failed_accepted_handoff_without_retry(
    tool_triggered: bool,
) -> None:
    coordinator, scheduler, effects, _ = await _coordinator(
        automatic_retrieval=not tool_triggered
    )
    effects.accepted_error = RuntimeError("runtime refused custody")
    effects.audio_rebuild_allowed = True
    try:
        if tool_triggered:
            epoch = await _start_tool_attempt(coordinator, scheduler)
            await coordinator.submit(
                VoiceAttemptToolRequest(
                    epoch,
                    ({"id": "call-route-failure", "function": {"name": "weather"}},),
                )
            )
            scheduler.advance_ms(1_300)
        else:
            await coordinator.submit(_speech(0, started_ns=0))
            await _fresh_transcript(
                coordinator,
                revision_id=1,
                text="Exact rolling transcript",
            )
            scheduler.advance_ms(2_000)
        await coordinator.flush()
        assert coordinator.snapshot.failure_class == "accepted_handoff_failed"
        initial_dispatches = list(effects.dispatches)

        await coordinator.submit(DeviceRouteChanged(0, RouteKind.OUTPUT))
        await coordinator.submit(
            AudioRouteReady(
                rebuild_epoch=coordinator.snapshot.audio_rebuild_epoch,
                clock_generation=1,
                duplex_mode=DuplexMode.FULL_DUPLEX,
                aec_health=AecHealth.HEALTHY,
                safety_path=AcousticSafetyPath.AEC,
            )
        )

        snapshot = coordinator.snapshot
        assert snapshot.state is SpeculativeVoiceState.DRAFT_SUSPENDED
        assert snapshot.transcript_text == "Exact rolling transcript"
        assert snapshot.failure_class == "accepted_handoff_failed"
        scheduler.advance_ms(5_000)
        await coordinator.flush()
        assert effects.accepted_attempts == 1
        assert effects.dispatches == initial_dispatches
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_synchronous_abort_failure_cannot_strand_tool_handoff() -> None:
    coordinator, scheduler, effects, context = await _coordinator()
    effects.abort_error = RuntimeError("audio abort failed")
    try:
        epoch = await _start_tool_attempt(coordinator, scheduler)

        await coordinator.submit(
            VoiceAttemptToolRequest(
                epoch,
                ({"id": "call-abort-failed", "function": {"name": "weather"}},),
            )
        )

        assert (
            coordinator.snapshot.state is SpeculativeVoiceState.WAITING_FOR_STABLE_TURN
        )
        scheduler.advance_ms(1_300)
        await coordinator.flush()
        assert effects.accepted_calls == [("Exact rolling transcript", context)]
        assert effects.accepted_attempts == 1
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_route_rebuild_preserves_tool_handoff_obligation() -> None:
    coordinator, scheduler, effects, context = await _coordinator()
    effects.audio_rebuild_allowed = True
    try:
        epoch = await _start_tool_attempt(coordinator, scheduler)
        await coordinator.submit(
            VoiceAttemptToolRequest(
                epoch,
                ({"id": "call-route", "function": {"name": "weather"}},),
            )
        )
        scheduler.advance_ms(400)
        await coordinator.submit(DeviceRouteChanged(0, RouteKind.OUTPUT))
        rebuild_epoch = coordinator.snapshot.audio_rebuild_epoch

        await coordinator.submit(
            AudioRouteReady(
                rebuild_epoch=rebuild_epoch,
                clock_generation=1,
                duplex_mode=DuplexMode.FULL_DUPLEX,
                aec_health=AecHealth.HEALTHY,
                safety_path=AcousticSafetyPath.AEC,
            )
        )

        assert (
            coordinator.snapshot.state is SpeculativeVoiceState.WAITING_FOR_STABLE_TURN
        )
        scheduler.advance_ms(899)
        await coordinator.flush()
        assert effects.accepted_calls == []
        scheduler.advance_ms(1)
        await coordinator.flush()

        assert effects.accepted_calls == [("Exact rolling transcript", context)]
        assert effects.accepted_attempts == 1
        assert len(effects.dispatches) == 1
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_speech_before_tool_barrier_expiry_fences_and_stales_the_timer() -> None:
    coordinator, scheduler, effects, _ = await _coordinator()
    try:
        epoch = await _start_tool_attempt(coordinator, scheduler)
        await coordinator.submit(
            VoiceAttemptToolRequest(
                epoch,
                ({"id": "call-stale", "function": {"name": "weather"}},),
            )
        )
        scheduler.advance_ms(500)
        resumed = _speech(1, started_ns=scheduler.now_ns)
        await coordinator.submit(resumed)
        await _fresh_transcript(
            coordinator,
            revision_id=2,
            text="Exact rolling transcript with more speech",
        )

        assert ("fence", epoch) in effects.operations
        assert ("cancel", epoch) in effects.operations
        scheduler.advance_ms(800)
        await coordinator.flush()
        assert effects.accepted_calls == []
        assert effects.dispatches[-1][2] == "Exact rolling transcript with more speech"
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_navigation_before_handoff_discards_but_after_handoff_does_not_cancel() -> (
    None
):
    before, scheduler, before_effects, _ = await _coordinator(automatic_retrieval=True)
    try:
        await before.submit(_speech(0, started_ns=0))
        await _fresh_transcript(before, revision_id=1, text="leave before handoff")
        scheduler.advance_ms(1_000)
        await before.submit(ControlAction(ControlKind.NAVIGATION))
        scheduler.advance_ms(2_000)
        await before.flush()
        assert before_effects.accepted_calls == []
    finally:
        await before.close()

    after, scheduler, after_effects, _ = await _coordinator(automatic_retrieval=True)
    after_effects.accepted_release = asyncio.Event()
    try:
        await after.submit(_speech(0, started_ns=0))
        await _fresh_transcript(after, revision_id=1, text="survive after handoff")
        scheduler.advance_ms(2_000)
        await after.flush()
        await after_effects.accepted_started.wait()

        await after.submit(ControlAction(ControlKind.NAVIGATION))
        await asyncio.sleep(0)
        assert after_effects.accepted_cancelled is False
        assert after_effects.accepted_finished.is_set() is False

        after_effects.accepted_release.set()
        await after_effects.accepted_finished.wait()
        assert after_effects.accepted_task is not None
        await after_effects.accepted_task
    finally:
        await after.close()
