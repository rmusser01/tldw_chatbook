from __future__ import annotations

import asyncio
from dataclasses import dataclass
import heapq
from typing import Any, Callable

import pytest

from tldw_chatbook.Audio.duplex_contracts import (
    AcousticSafetyPath,
    AecHealth,
    CaptureDrainError,
    DeviceRouteChanged,
    DrainReceipt,
    DuplexMode,
    RouteKind,
)
from tldw_chatbook.Audio.rolling_transcript import TranscriptRevision
from tldw_chatbook.Chat.console_speculative_voice import (
    AdmittedSpeechFrame,
    AttemptGenerationCompleted,
    AttemptOutputDelta,
    AttemptPlaybackStarted,
    AttemptPlaybackTerminal,
    AttemptTtsFailed,
    AudioCapabilityChanged,
    AudioRouteReady,
    AudioRouteRebuildFailed,
    CaptureGated,
    ControlAction,
    ControlKind,
    ManualInterruption,
    ManualRetry,
    SpeculativeTurnCoordinator,
    SpeculativeVoiceState,
)
from tldw_chatbook.Chat.console_voice_attempts import (
    AttemptCleanupOutcome,
    ProviderAttemptFailed,
)


@dataclass(slots=True)
class _Scheduled:
    deadline_ns: int
    order: int
    callback: Callable[[], None]
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


class _FakeScheduler:
    def __init__(self) -> None:
        self.now_ns = 0
        self._next_order = 0
        self._scheduled: list[tuple[int, int, _Scheduled]] = []

    def call_at_ns(
        self,
        deadline_ns: int,
        callback: Callable[[], None],
    ) -> _Scheduled:
        handle = _Scheduled(deadline_ns, self._next_order, callback)
        self._next_order += 1
        heapq.heappush(
            self._scheduled,
            (handle.deadline_ns, handle.order, handle),
        )
        return handle

    def advance_ms(self, milliseconds: int) -> None:
        self.now_ns += milliseconds * 1_000_000
        while self._scheduled and self._scheduled[0][0] <= self.now_ns:
            _, _, handle = heapq.heappop(self._scheduled)
            if not handle.cancelled:
                handle.callback()


class _FakeEffects:
    def __init__(self, *, auto_cleanup: bool = True) -> None:
        self.auto_cleanup = auto_cleanup
        self.operations: list[tuple[Any, ...]] = []
        self.dispatches: list[tuple[str, int, str]] = []
        self.fenced_epochs: set[int] = set()
        self.abort_hook: Callable[[], None] | None = None
        self.cleanup_futures: dict[int, asyncio.Future[AttemptCleanupOutcome]] = {}
        self.orphan_quarantined = False
        self.previews: list[tuple[int, str]] = []
        self.drafts: list[tuple[str, str, str]] = []
        self.promotions: list[tuple[str, int, str, str, int | None]] = []
        self.commands: list[tuple[str, str]] = []
        self.command_texts: set[str] = set()
        self.rebuilds: list[int] = []
        self.rebuild_requests: list[tuple[int, int]] = []
        self.drain_calls: list[int] = []
        self.classification_drain_calls: list[tuple[int, int]] = []
        self.seal_calls: list[int] = []
        self.drain_receipt = DrainReceipt(1, 0, 0, 0, 0)
        self.drain_error: BaseException | None = None
        self.drain_future: asyncio.Future[DrainReceipt] | None = None
        self.classification_release: asyncio.Event | None = None
        self.classification_error: BaseException | None = None
        self.seal_error: BaseException | None = None
        self.seal_future: asyncio.Future[TranscriptRevision] | None = None
        self.seal_revision: TranscriptRevision | None = None
        self.promotion_future: asyncio.Future[object] | None = None

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
        self.fenced_epochs.add(attempt_epoch)

    def cancel_attempt(
        self, attempt_epoch: int
    ) -> asyncio.Future[AttemptCleanupOutcome]:
        self.operations.append(("cancel", attempt_epoch))
        completion = asyncio.get_running_loop().create_future()
        self.cleanup_futures[attempt_epoch] = completion
        if self.auto_cleanup:
            completion.set_result(AttemptCleanupOutcome.CLEAN)
        return completion

    def abort_output(self, attempt_epoch: int) -> None:
        self.operations.append(("abort", attempt_epoch))
        if self.abort_hook is not None:
            self.abort_hook()

    def clear_preview(self, attempt_epoch: int) -> None:
        self.operations.append(("clear", attempt_epoch))

    def publish_preview(self, attempt_epoch: int, delta: str) -> None:
        assert attempt_epoch not in self.fenced_epochs
        self.previews.append((attempt_epoch, delta))

    def preserve_draft(
        self,
        *,
        turn_id: str,
        transcript: str,
        reason: str,
    ) -> None:
        self.drafts.append((turn_id, transcript, reason))

    def promote(
        self,
        *,
        turn_id: str,
        attempt_epoch: int,
        transcript: str,
        assistant_text: str,
        terminal_boundary_ns: int | None,
    ) -> object:
        self.promotions.append(
            (
                turn_id,
                attempt_epoch,
                transcript,
                assistant_text,
                terminal_boundary_ns,
            )
        )
        assert attempt_epoch not in self.fenced_epochs
        return self.promotion_future

    def classify_spoken_command(self, transcript: str) -> bool:
        return transcript in self.command_texts

    def handle_spoken_command(self, *, turn_id: str, transcript: str) -> None:
        self.commands.append((turn_id, transcript))

    def rebuild_audio(self, old_clock_generation: int, rebuild_epoch: int) -> None:
        self.rebuilds.append(old_clock_generation)
        self.rebuild_requests.append((old_clock_generation, rebuild_epoch))

    async def drain_capture_through(self, render_boundary_ns: int) -> DrainReceipt:
        self.drain_calls.append(render_boundary_ns)
        if self.drain_error is not None:
            raise self.drain_error
        if self.drain_future is not None:
            return await self.drain_future
        return self.drain_receipt

    async def drain_pending_classification_through(
        self,
        render_boundary_ns: int,
        clock_generation: int,
    ) -> None:
        self.classification_drain_calls.append((render_boundary_ns, clock_generation))
        if self.classification_error is not None:
            raise self.classification_error
        if self.classification_release is not None:
            await self.classification_release.wait()

    async def seal_transcript_through(
        self, admitted_sequence: int
    ) -> TranscriptRevision:
        self.seal_calls.append(admitted_sequence)
        if self.seal_error is not None:
            raise self.seal_error
        if self.seal_future is not None:
            return await self.seal_future
        if self.seal_revision is not None:
            return self.seal_revision
        turn_id, _, transcript = self.dispatches[-1]
        return TranscriptRevision(
            turn_id=turn_id,
            revision_id=1_000_000,
            stable_text=transcript,
            revisable_text="",
            covered_through_ns=10**18,
            mode="live",
            is_final=True,
        )

    def complete_cleanup(
        self,
        attempt_epoch: int,
        outcome: AttemptCleanupOutcome = AttemptCleanupOutcome.CLEAN,
    ) -> None:
        completion = self.cleanup_futures[attempt_epoch]
        if not completion.done():
            completion.set_result(outcome)

    def voice_dispatch_quarantined(self) -> bool:
        return self.orphan_quarantined


async def _coordinator(
    *,
    response_eagerness_ms: object = 700,
    auto_cleanup: bool = True,
    mode: DuplexMode = DuplexMode.FULL_DUPLEX,
) -> tuple[SpeculativeTurnCoordinator, _FakeScheduler, _FakeEffects]:
    scheduler = _FakeScheduler()
    effects = _FakeEffects(auto_cleanup=auto_cleanup)
    coordinator = SpeculativeTurnCoordinator(
        effects=effects,
        scheduler=scheduler,
        response_eagerness_ms=response_eagerness_ms,
        initial_clock_generation=0,
        initial_duplex_mode=mode,
    )
    await coordinator.start()
    return coordinator, scheduler, effects


def _speech(
    sequence: int,
    *,
    started_ns: int,
    generation: int = 0,
    assistant_rendering: bool = True,
) -> AdmittedSpeechFrame:
    return AdmittedSpeechFrame(
        sequence=sequence,
        started_ns=started_ns,
        ended_ns=started_ns + 10_000_000,
        clock_generation=generation,
        assistant_rendering=assistant_rendering,
    )


def _revision(
    turn_id: str,
    revision_id: int,
    text: str,
    covered_through_ns: int,
) -> TranscriptRevision:
    return TranscriptRevision(
        turn_id=turn_id,
        revision_id=revision_id,
        stable_text="",
        revisable_text=text,
        covered_through_ns=covered_through_ns,
        mode="live",
    )


async def _start_attempt(
    coordinator: SpeculativeTurnCoordinator,
    scheduler: _FakeScheduler,
    *,
    text: str = "hello world",
    sequence: int = 0,
) -> tuple[str, int]:
    frame = _speech(sequence, started_ns=scheduler.now_ns)
    await coordinator.submit(frame)
    turn_id = coordinator.snapshot.turn_id
    assert turn_id is not None
    await coordinator.submit(
        _revision(turn_id, 1, text, frame.ended_ns),
    )
    scheduler.advance_ms(coordinator.response_eagerness_ms)
    await coordinator.flush()
    epoch = coordinator.snapshot.current_attempt_epoch
    assert epoch is not None
    return turn_id, epoch


@pytest.mark.asyncio
async def test_first_frame_starts_turn_and_each_frame_resets_fresh_dispatch() -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        first = _speech(0, started_ns=0)
        await coordinator.submit(first)
        turn_id = coordinator.snapshot.turn_id
        assert turn_id is not None
        assert coordinator.snapshot.state is SpeculativeVoiceState.LISTENING

        await coordinator.submit(_revision(turn_id, 1, "hello", first.ended_ns))
        scheduler.advance_ms(600)
        second = _speech(1, started_ns=600_000_000)
        await coordinator.submit(second)
        await coordinator.submit(_revision(turn_id, 2, "hello there", second.ended_ns))

        scheduler.advance_ms(699)
        await coordinator.flush()
        assert effects.dispatches == []
        scheduler.advance_ms(1)
        await coordinator.flush()

        assert effects.dispatches == [(turn_id, 1, "hello there")]
        assert coordinator.snapshot.state is SpeculativeVoiceState.GENERATING
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_expired_eagerness_waits_for_nonempty_fresh_tail_coverage() -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        frame = _speech(0, started_ns=20_000_000)
        await coordinator.submit(frame)
        turn_id = coordinator.snapshot.turn_id
        assert turn_id is not None

        scheduler.advance_ms(700)
        await coordinator.flush()
        assert effects.dispatches == []
        assert coordinator.snapshot.state is SpeculativeVoiceState.TRANSCRIBING

        await coordinator.submit(
            _revision(turn_id, 1, "hello", frame.ended_ns - 10_000_001)
        )
        assert effects.dispatches == []
        await coordinator.submit(_revision(turn_id, 2, "   ", frame.ended_ns))
        assert effects.dispatches == []
        await coordinator.submit(_revision(turn_id, 3, "hello", frame.ended_ns))

        assert effects.dispatches == [(turn_id, 1, "hello")]
    finally:
        await coordinator.close()


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, 700),
        (True, 700),
        (499, 700),
        (500, 500),
        (700, 700),
        (3_000, 3_000),
        (3_001, 700),
        ("700", 700),
    ],
)
@pytest.mark.asyncio
async def test_response_eagerness_is_bounded_with_700ms_default(
    configured: object,
    expected: int,
) -> None:
    coordinator, _, _ = await _coordinator(response_eagerness_ms=configured)
    try:
        assert coordinator.response_eagerness_ms == expected
    finally:
        await coordinator.close()


@pytest.mark.parametrize("playback_started", [False, True])
@pytest.mark.asyncio
async def test_speech_during_generation_or_playback_fences_and_extends_same_turn(
    playback_started: bool,
) -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        if playback_started:
            await coordinator.submit(AttemptPlaybackStarted(epoch))
        await coordinator.submit(AttemptOutputDelta(epoch, "stale answer"))

        resumed = _speech(1, started_ns=scheduler.now_ns)
        await coordinator.submit(resumed)

        assert coordinator.snapshot.turn_id == turn_id
        assert coordinator.snapshot.current_attempt_epoch is None
        assert effects.operations.index(("fence", epoch)) < effects.operations.index(
            ("abort", epoch)
        )
        assert effects.operations.index(("abort", epoch)) < effects.operations.index(
            ("cancel", epoch)
        )
        await coordinator.submit(AttemptOutputDelta(epoch, "must be ignored"))
        assert effects.previews == [(epoch, "stale answer")]

        await coordinator.submit(
            _revision(turn_id, 2, "hello world and more", resumed.ended_ns),
        )
        scheduler.advance_ms(700)
        await coordinator.flush()

        replacement = effects.dispatches[-1]
        assert replacement[0] == turn_id
        assert replacement[1] > epoch
        assert replacement[2] == "hello world and more"
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_duplicate_admitted_sequence_cannot_fence_current_attempt() -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        _, epoch = await _start_attempt(coordinator, scheduler)

        await coordinator.submit(_speech(0, started_ns=scheduler.now_ns))
        await coordinator.submit(AttemptOutputDelta(epoch, "still current"))

        assert coordinator.snapshot.current_attempt_epoch == epoch
        assert epoch not in effects.fenced_epochs
        assert effects.previews == [(epoch, "still current")]
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_terminal_stt_failure_preserves_draft_and_stops_automatic_dispatch() -> (
    None
):
    coordinator, scheduler, effects = await _coordinator()
    try:
        frame = _speech(0, started_ns=0)
        await coordinator.submit(frame)
        turn_id = coordinator.snapshot.turn_id
        assert turn_id is not None
        await coordinator.submit(
            TranscriptRevision(
                turn_id=turn_id,
                revision_id=1,
                stable_text="",
                revisable_text="editable partial request",
                covered_through_ns=frame.ended_ns,
                mode="rolling-window",
                is_final=True,
                failure_code="fallback_failed",
            )
        )
        scheduler.advance_ms(5_000)
        await coordinator.flush()

        assert effects.dispatches == []
        assert effects.drafts == [(turn_id, "editable partial request", "stt_failed")]
        assert coordinator.snapshot.state is SpeculativeVoiceState.DRAFT_SUSPENDED
        assert coordinator.snapshot.failure_class == "fallback_failed"
        assert coordinator.snapshot.admission_open is True

        await coordinator.submit(ManualRetry())
        assert effects.dispatches == [(turn_id, 1, "editable partial request")]
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_cosmetic_revision_does_not_restart_but_material_burst_coalesces() -> (
    None
):
    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, epoch = await _start_attempt(
            coordinator, scheduler, text="Hello world"
        )
        covered = 10_000_000
        await coordinator.submit(
            _revision(turn_id, 2, " hello,   WORLD! ", covered),
        )
        assert coordinator.snapshot.current_attempt_epoch == epoch
        assert effects.fenced_epochs == set()

        await coordinator.submit(_revision(turn_id, 3, "Hello brave world", covered))
        assert epoch in effects.fenced_epochs
        assert coordinator.snapshot.current_attempt_epoch is None

        scheduler.advance_ms(100)
        await coordinator.submit(
            _revision(turn_id, 4, "Hello very brave world", covered),
        )
        scheduler.advance_ms(119)
        await coordinator.flush()
        assert len(effects.dispatches) == 1
        scheduler.advance_ms(1)
        await coordinator.flush()

        assert effects.dispatches[-1][2] == "Hello very brave world"
        assert len(effects.dispatches) == 2
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_correction_debounce_extends_but_hard_caps_at_250ms() -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, _ = await _start_attempt(coordinator, scheduler, text="one")
        await coordinator.submit(_revision(turn_id, 2, "one two", 10_000_000))
        scheduler.advance_ms(100)
        await coordinator.submit(_revision(turn_id, 3, "one two three", 10_000_000))
        scheduler.advance_ms(100)
        await coordinator.submit(
            _revision(turn_id, 4, "one two three four", 10_000_000),
        )
        scheduler.advance_ms(49)
        await coordinator.flush()
        assert len(effects.dispatches) == 1
        scheduler.advance_ms(1)
        await coordinator.flush()

        assert len(effects.dispatches) == 2
        assert effects.dispatches[-1][2] == "one two three four"
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_fourth_attempt_within_ten_seconds_uses_1500ms_eagerness() -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, _ = await _start_attempt(coordinator, scheduler, text="one")
        for sequence, text in ((1, "one two"), (2, "one two three")):
            frame = _speech(sequence, started_ns=scheduler.now_ns)
            await coordinator.submit(frame)
            await coordinator.submit(
                _revision(turn_id, sequence + 1, text, frame.ended_ns),
            )
            scheduler.advance_ms(700)
            await coordinator.flush()
        assert len(effects.dispatches) == 3

        fourth_speech = _speech(3, started_ns=scheduler.now_ns)
        await coordinator.submit(fourth_speech)
        await coordinator.submit(
            _revision(turn_id, 4, "one two three four", fourth_speech.ended_ns),
        )
        scheduler.advance_ms(1_499)
        await coordinator.flush()
        assert len(effects.dispatches) == 3
        scheduler.advance_ms(1)
        await coordinator.flush()
        assert len(effects.dispatches) == 4
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_material_correction_attempt_four_obeys_1500ms_governor() -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, _ = await _start_attempt(coordinator, scheduler, text="one")
        for sequence, text in ((1, "one two"), (2, "one two three")):
            frame = _speech(sequence, started_ns=scheduler.now_ns)
            await coordinator.submit(frame)
            await coordinator.submit(
                _revision(turn_id, sequence + 1, text, frame.ended_ns)
            )
            scheduler.advance_ms(700)
            await coordinator.flush()
        assert len(effects.dispatches) == 3

        await coordinator.submit(
            _revision(
                turn_id,
                4,
                "one two three corrected",
                coordinator.snapshot.last_speech_end_ns or 0,
            )
        )
        scheduler.advance_ms(1_499)
        await coordinator.flush()
        assert len(effects.dispatches) == 3

        scheduler.advance_ms(1)
        await coordinator.flush()
        assert effects.dispatches[-1][2] == "one two three corrected"
        assert len(effects.dispatches) == 4
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_material_correction_after_restart_window_uses_correction_debounce() -> (
    None
):
    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, _ = await _start_attempt(coordinator, scheduler, text="one")
        for sequence, text in ((1, "one two"), (2, "one two three")):
            frame = _speech(sequence, started_ns=scheduler.now_ns)
            await coordinator.submit(frame)
            await coordinator.submit(
                _revision(turn_id, sequence + 1, text, frame.ended_ns)
            )
            scheduler.advance_ms(700)
            await coordinator.flush()
        assert len(effects.dispatches) == 3

        scheduler.advance_ms(10_001)
        await coordinator.submit(
            _revision(
                turn_id,
                4,
                "one two three corrected",
                coordinator.snapshot.last_speech_end_ns or 0,
            )
        )
        scheduler.advance_ms(119)
        await coordinator.flush()
        assert len(effects.dispatches) == 3
        scheduler.advance_ms(1)
        await coordinator.flush()

        assert effects.dispatches[-1][2] == "one two three corrected"
        assert len(effects.dispatches) == 4
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_cleanup_cap_pauses_and_slow_cleanup_enters_serialized_conservative() -> (
    None
):
    coordinator, scheduler, effects = await _coordinator(auto_cleanup=False)
    try:
        turn_id, first_epoch = await _start_attempt(coordinator, scheduler, text="one")
        second_speech = _speech(1, started_ns=scheduler.now_ns)
        await coordinator.submit(second_speech)
        await coordinator.submit(
            _revision(turn_id, 2, "one two", second_speech.ended_ns),
        )
        scheduler.advance_ms(700)
        await coordinator.flush()
        second_epoch = coordinator.snapshot.current_attempt_epoch
        assert second_epoch is not None and second_epoch != first_epoch

        third_speech = _speech(2, started_ns=scheduler.now_ns)
        await coordinator.submit(third_speech)
        await coordinator.submit(
            _revision(turn_id, 3, "one two three", third_speech.ended_ns),
        )
        assert coordinator.snapshot.obsolete_cleanup_count == 2

        scheduler.advance_ms(1_300)
        await coordinator.flush()
        assert (
            coordinator.snapshot.state is SpeculativeVoiceState.SERIALIZED_CONSERVATIVE
        )
        assert len(effects.dispatches) == 2

        effects.complete_cleanup(first_epoch)
        effects.complete_cleanup(second_epoch)
        await coordinator.flush()
        scheduler.advance_ms(699)
        await coordinator.flush()
        assert len(effects.dispatches) == 2
        scheduler.advance_ms(1)
        await coordinator.flush()

        assert len(effects.dispatches) == 3
        assert effects.dispatches[-1][2] == "one two three"
        assert coordinator.snapshot.serialized_conservative is True
    finally:
        for completion in effects.cleanup_futures.values():
            if not completion.done():
                completion.set_result(AttemptCleanupOutcome.CLEAN)
        await coordinator.close()


@pytest.mark.asyncio
async def test_detached_cleanup_terminates_turn_until_orphan_quarantine_clears() -> (
    None
):
    coordinator, scheduler, effects = await _coordinator(auto_cleanup=False)
    try:
        old_turn, epoch = await _start_attempt(coordinator, scheduler)
        resumed = _speech(1, started_ns=scheduler.now_ns)
        await coordinator.submit(resumed)
        await coordinator.submit(
            _revision(old_turn, 2, "hello world continued", resumed.ended_ns)
        )

        effects.orphan_quarantined = True
        effects.complete_cleanup(epoch, AttemptCleanupOutcome.DETACHED)
        await coordinator.flush()

        assert coordinator.snapshot.state is SpeculativeVoiceState.DRAFT_SUSPENDED
        assert coordinator.snapshot.failure_class == "voice_cleanup_stuck"
        assert coordinator.snapshot.current_attempt_epoch is None
        assert effects.drafts[-1] == (
            old_turn,
            "hello world continued",
            "voice_cleanup_stuck",
        )

        scheduler.advance_ms(5_000)
        await coordinator.flush()
        await coordinator.submit(_speech(2, started_ns=scheduler.now_ns))
        assert len(effects.dispatches) == 1
        assert coordinator.snapshot.turn_id == old_turn

        effects.orphan_quarantined = False
        fresh = _speech(3, started_ns=scheduler.now_ns)
        await coordinator.submit(fresh)
        new_turn = coordinator.snapshot.turn_id
        assert new_turn is not None and new_turn != old_turn
        await coordinator.submit(_revision(new_turn, 1, "new request", fresh.ended_ns))
        scheduler.advance_ms(700)
        await coordinator.flush()
        assert effects.dispatches[-1][0] == new_turn
        assert effects.dispatches[-1][2] == "new request"
    finally:
        for completion in effects.cleanup_futures.values():
            if not completion.done():
                completion.set_result(AttemptCleanupOutcome.CLEAN)
        await coordinator.close()


@pytest.mark.asyncio
async def test_slow_sibling_cleanup_cannot_overwrite_detached_draft_state() -> None:
    coordinator, scheduler, effects = await _coordinator(auto_cleanup=False)
    try:
        turn_id, first_epoch = await _start_attempt(coordinator, scheduler, text="one")
        second_speech = _speech(1, started_ns=scheduler.now_ns)
        await coordinator.submit(second_speech)
        await coordinator.submit(
            _revision(turn_id, 2, "one two", second_speech.ended_ns)
        )
        scheduler.advance_ms(700)
        await coordinator.flush()
        second_epoch = coordinator.snapshot.current_attempt_epoch
        assert second_epoch is not None

        await coordinator.submit(_speech(2, started_ns=scheduler.now_ns))
        effects.orphan_quarantined = True
        effects.complete_cleanup(first_epoch, AttemptCleanupOutcome.DETACHED)
        await coordinator.flush()
        assert coordinator.snapshot.state is SpeculativeVoiceState.DRAFT_SUSPENDED

        scheduler.advance_ms(2_000)
        await coordinator.flush()
        assert coordinator.snapshot.state is SpeculativeVoiceState.DRAFT_SUSPENDED
        assert coordinator.snapshot.failure_class == "voice_cleanup_stuck"
    finally:
        for completion in effects.cleanup_futures.values():
            if not completion.done():
                completion.set_result(AttemptCleanupOutcome.CLEAN)
        await coordinator.close()


@pytest.mark.parametrize(
    "action",
    [
        ControlKind.STOP,
        ControlKind.ESCAPE,
        ControlKind.MICROPHONE_DISABLED,
        ControlKind.HANDS_FREE_EXIT,
        ControlKind.NAVIGATION,
        ControlKind.TEARDOWN,
    ],
)
@pytest.mark.asyncio
async def test_explicit_control_discards_provisional_work_without_history(
    action: ControlKind,
) -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        _, epoch = await _start_attempt(coordinator, scheduler)
        await coordinator.submit(
            AttemptOutputDelta(epoch, "private provisional answer")
        )
        await coordinator.submit(ControlAction(action))

        assert coordinator.snapshot.state is SpeculativeVoiceState.IDLE
        assert coordinator.snapshot.turn_id is None
        assert coordinator.snapshot.transcript_text == ""
        assert effects.promotions == []
        assert effects.drafts == []
        assert epoch in effects.fenced_epochs
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_close_rejects_submission_created_during_teardown() -> None:
    coordinator, scheduler, effects = await _coordinator()
    _, _ = await _start_attempt(coordinator, scheduler)
    racing_submit: asyncio.Task[None] | None = None

    def submit_after_abort() -> None:
        nonlocal racing_submit
        if racing_submit is None:
            racing_submit = asyncio.create_task(
                coordinator.submit(_speech(1, started_ns=scheduler.now_ns))
            )

    effects.abort_hook = submit_after_abort
    await coordinator.close()

    assert racing_submit is not None
    outcome = (await asyncio.gather(racing_submit, return_exceptions=True))[0]
    assert isinstance(outcome, RuntimeError)
    assert coordinator.snapshot.state is SpeculativeVoiceState.IDLE
    assert coordinator.snapshot.turn_id is None
    assert coordinator.snapshot.transcript_text == ""


@pytest.mark.asyncio
async def test_cancelling_first_close_waiter_cannot_publish_early_completion() -> None:
    coordinator, scheduler, _ = await _coordinator()
    await _start_attempt(coordinator, scheduler)
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    discard_turn = coordinator._discard_turn

    async def blocked_discard_turn() -> None:
        entered.set()
        await release.wait()
        await discard_turn()
        finished.set()

    coordinator._discard_turn = blocked_discard_turn  # type: ignore[method-assign]
    owner = asyncio.create_task(coordinator.close())
    await entered.wait()
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner

    observer = asyncio.create_task(coordinator.close())
    await asyncio.sleep(0)
    try:
        assert not observer.done()
        assert coordinator.snapshot.state is SpeculativeVoiceState.GENERATING
    finally:
        release.set()
        await finished.wait()
        await observer

    assert coordinator.snapshot.state is SpeculativeVoiceState.IDLE
    assert coordinator.snapshot.current_attempt_epoch is None


@pytest.mark.parametrize("raising_effect", ["fence", "abort", "cancel"])
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
@pytest.mark.asyncio
async def test_close_scrubs_turn_when_cancellation_effect_raises(
    raising_effect: str,
    error_type: type[BaseException],
) -> None:
    coordinator, scheduler, effects = await _coordinator()
    await _start_attempt(coordinator, scheduler)
    calls: list[str] = []

    def fence_attempt(_epoch: int) -> None:
        calls.append("fence")
        if raising_effect == "fence":
            raise error_type("fence failed")

    def clear_preview(_epoch: int) -> None:
        calls.append("clear")

    def abort_output(_epoch: int) -> None:
        calls.append("abort")
        if raising_effect == "abort":
            raise error_type("abort failed")

    def cancel_attempt(_epoch: int) -> None:
        calls.append("cancel")
        if raising_effect == "cancel":
            raise error_type("cancel failed")

    effects.fence_attempt = fence_attempt  # type: ignore[method-assign]
    effects.clear_preview = clear_preview  # type: ignore[method-assign]
    effects.abort_output = abort_output  # type: ignore[method-assign]
    effects.cancel_attempt = cancel_attempt  # type: ignore[method-assign]

    await coordinator.close()

    assert calls == ["fence", "clear", "abort", "cancel"]
    assert coordinator.snapshot.state is SpeculativeVoiceState.IDLE
    assert coordinator.snapshot.turn_id is None
    assert coordinator.snapshot.transcript_text == ""
    assert coordinator.snapshot.current_attempt_epoch is None


@pytest.mark.parametrize("raising_effect", ["fence", "abort", "cancel"])
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
@pytest.mark.asyncio
async def test_resumed_speech_survives_synchronous_cancellation_effect_failure(
    raising_effect: str,
    error_type: type[BaseException],
) -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        calls: list[str] = []

        def fence_attempt(_epoch: int) -> None:
            calls.append("fence")
            if raising_effect == "fence":
                raise error_type("fence failed")

        def abort_output(_epoch: int) -> None:
            calls.append("abort")
            if raising_effect == "abort":
                raise error_type("abort failed")

        def cancel_attempt(_epoch: int) -> None:
            calls.append("cancel")
            if raising_effect == "cancel":
                raise error_type("cancel failed")

        effects.fence_attempt = fence_attempt  # type: ignore[method-assign]
        effects.abort_output = abort_output  # type: ignore[method-assign]
        effects.cancel_attempt = cancel_attempt  # type: ignore[method-assign]

        resumed = _speech(1, started_ns=scheduler.now_ns)
        await coordinator.submit(resumed)

        assert calls == ["fence", "abort", "cancel"]
        assert coordinator.snapshot.state is SpeculativeVoiceState.LISTENING
        assert coordinator.snapshot.turn_id == turn_id
        assert coordinator.snapshot.last_admitted_sequence == 1
        assert coordinator.snapshot.current_attempt_epoch is None

        await coordinator.submit(
            _revision(turn_id, 2, "hello world continued", resumed.ended_ns)
        )
        scheduler.advance_ms(700)
        await coordinator.flush()
        assert effects.dispatches[-1][2] == "hello world continued"
        assert effects.dispatches[-1][1] > epoch
    finally:
        await coordinator.close()


@pytest.mark.parametrize("event_path", ["route", "transcript"])
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
@pytest.mark.asyncio
async def test_route_and_stt_fences_survive_synchronous_effect_failure(
    event_path: str,
    error_type: type[BaseException],
) -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, _ = await _start_attempt(coordinator, scheduler)

        def fence_attempt(_epoch: int) -> None:
            raise error_type("fence failed")

        effects.fence_attempt = fence_attempt  # type: ignore[method-assign]
        if event_path == "route":
            await coordinator.submit(DeviceRouteChanged(0, RouteKind.DUPLEX))
            assert coordinator.snapshot.state is SpeculativeVoiceState.REBUILDING_AUDIO
            assert effects.rebuilds == [0]
        else:
            await coordinator.submit(
                _revision(turn_id, 2, "hello corrected world", 10_000_000)
            )
            scheduler.advance_ms(120)
            await coordinator.flush()
            assert effects.dispatches[-1][2] == "hello corrected world"
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_spoken_command_is_classified_before_provider_dispatch() -> None:
    coordinator, scheduler, effects = await _coordinator()
    effects.command_texts.add("stop listening")
    try:
        frame = _speech(0, started_ns=0)
        await coordinator.submit(frame)
        turn_id = coordinator.snapshot.turn_id
        assert turn_id is not None
        await coordinator.submit(
            _revision(turn_id, 1, "stop listening", frame.ended_ns),
        )
        scheduler.advance_ms(700)
        await coordinator.flush()

        assert effects.commands == [(turn_id, "stop listening")]
        assert effects.dispatches == []
        assert coordinator.snapshot.state is SpeculativeVoiceState.IDLE
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_tts_failure_commits_completed_text_only_without_newer_event() -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        await coordinator.submit(AttemptTtsFailed(epoch, "invalid_audio_stream"))
        await coordinator.submit(AttemptGenerationCompleted(epoch, "complete answer"))

        assert effects.promotions == [
            (turn_id, epoch, "hello world", "complete answer", None)
        ]
        assert coordinator.snapshot.state is SpeculativeVoiceState.IDLE
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_half_duplex_tts_failure_reopens_same_turn_admission() -> None:
    coordinator, scheduler, effects = await _coordinator(mode=DuplexMode.HALF_DUPLEX)
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        await coordinator.submit(AttemptPlaybackStarted(epoch))
        assert coordinator.snapshot.admission_open is False

        await coordinator.submit(AttemptTtsFailed(epoch, "invalid_audio_stream"))
        assert coordinator.snapshot.state is SpeculativeVoiceState.RESPONDING_TEXT_ONLY
        assert coordinator.snapshot.admission_open is True

        await coordinator.submit(_speech(1, started_ns=scheduler.now_ns))
        assert coordinator.snapshot.turn_id == turn_id
        assert coordinator.snapshot.last_admitted_sequence == 1
        assert epoch in effects.fenced_epochs
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_newer_material_revision_prevents_text_commit_after_tts_failure() -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        await coordinator.submit(AttemptTtsFailed(epoch, "invalid_audio_stream"))
        await coordinator.submit(
            _revision(turn_id, 2, "hello corrected world", 10_000_000),
        )
        await coordinator.submit(AttemptGenerationCompleted(epoch, "stale answer"))

        assert effects.promotions == []
        assert epoch in effects.fenced_epochs
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_newer_cosmetic_revision_prevents_text_commit_after_tts_failure() -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, epoch = await _start_attempt(
            coordinator, scheduler, text="Hello world"
        )
        await coordinator.submit(AttemptTtsFailed(epoch, "invalid_audio_stream"))
        await coordinator.submit(_revision(turn_id, 2, " hello, WORLD! ", 10_000_000))
        await coordinator.submit(AttemptGenerationCompleted(epoch, "stale answer"))

        assert effects.promotions == []
        assert epoch in effects.fenced_epochs
    finally:
        await coordinator.close()


@pytest.mark.parametrize(
    ("mode", "health", "path"),
    [
        (
            DuplexMode.FULL_DUPLEX,
            AecHealth.HEALTHY,
            AcousticSafetyPath.AEC,
        ),
        (
            DuplexMode.HALF_DUPLEX,
            AecHealth.DEGRADED,
            AcousticSafetyPath.HALF_DUPLEX,
        ),
    ],
)
@pytest.mark.parametrize("playback_started", [False, True])
@pytest.mark.asyncio
async def test_device_route_change_fences_and_waits_for_new_healthy_clock_and_silence(
    mode: DuplexMode,
    health: AecHealth,
    path: AcousticSafetyPath,
    playback_started: bool,
) -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        if playback_started:
            await coordinator.submit(AttemptPlaybackStarted(epoch))
        await coordinator.submit(DeviceRouteChanged(0, RouteKind.OUTPUT))

        assert coordinator.snapshot.state is SpeculativeVoiceState.REBUILDING_AUDIO
        assert coordinator.snapshot.admission_open is False
        assert coordinator.snapshot.transcript_text == "hello world"
        assert epoch in effects.fenced_epochs
        assert effects.rebuilds == [0]
        await coordinator.submit(AttemptOutputDelta(epoch, "stale route output"))
        assert effects.previews == []

        await coordinator.submit(
            AudioRouteReady(
                rebuild_epoch=1,
                clock_generation=1,
                duplex_mode=mode,
                aec_health=health,
                safety_path=path,
            )
        )
        scheduler.advance_ms(699)
        await coordinator.flush()
        assert len(effects.dispatches) == 1
        scheduler.advance_ms(1)
        await coordinator.flush()

        assert len(effects.dispatches) == 2
        assert effects.dispatches[-1][0] == turn_id
        assert effects.dispatches[-1][2] == "hello world"
        assert coordinator.snapshot.audio_clock_generation == 1
    finally:
        await coordinator.close()


@pytest.mark.parametrize(
    "health",
    [AecHealth.WARMING, AecHealth.DEGRADED, AecHealth.HEALTHY],
)
def test_isolation_capability_is_valid_without_aec_health_proof(
    health: AecHealth,
) -> None:
    capability = AudioCapabilityChanged(
        clock_generation=3,
        duplex_mode=DuplexMode.FULL_DUPLEX,
        aec_health=health,
        safety_path=AcousticSafetyPath.ACOUSTIC_ISOLATION,
    )
    ready = AudioRouteReady(
        rebuild_epoch=2,
        clock_generation=3,
        duplex_mode=DuplexMode.FULL_DUPLEX,
        aec_health=health,
        safety_path=AcousticSafetyPath.ACOUSTIC_ISOLATION,
    )

    assert capability.safety_path is AcousticSafetyPath.ACOUSTIC_ISOLATION
    assert ready.safety_path is AcousticSafetyPath.ACOUSTIC_ISOLATION


@pytest.mark.parametrize(
    ("mode", "health", "path"),
    [
        (DuplexMode.FULL_DUPLEX, AecHealth.WARMING, AcousticSafetyPath.AEC),
        (DuplexMode.FULL_DUPLEX, AecHealth.HEALTHY, AcousticSafetyPath.WARMING),
        (DuplexMode.FULL_DUPLEX, AecHealth.HEALTHY, AcousticSafetyPath.HALF_DUPLEX),
        (DuplexMode.HALF_DUPLEX, AecHealth.HEALTHY, AcousticSafetyPath.AEC),
        (
            DuplexMode.HALF_DUPLEX,
            AecHealth.WARMING,
            AcousticSafetyPath.ACOUSTIC_ISOLATION,
        ),
    ],
)
def test_audio_capability_rejects_inconsistent_closed_path_contract(
    mode: DuplexMode,
    health: AecHealth,
    path: AcousticSafetyPath,
) -> None:
    with pytest.raises(ValueError):
        AudioCapabilityChanged(0, mode, health, path)
    with pytest.raises(ValueError):
        AudioRouteReady(1, 0, mode, health, path)


@pytest.mark.parametrize("generation", [True, -1, 1.0])
def test_audio_capability_requires_exact_nonnegative_integer_generation(
    generation: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        AudioCapabilityChanged(
            generation,  # type: ignore[arg-type]
            DuplexMode.HALF_DUPLEX,
            AecHealth.DEGRADED,
            AcousticSafetyPath.HALF_DUPLEX,
        )
    with pytest.raises((TypeError, ValueError)):
        AudioRouteReady(
            1,
            generation,  # type: ignore[arg-type]
            DuplexMode.HALF_DUPLEX,
            AecHealth.DEGRADED,
            AcousticSafetyPath.HALF_DUPLEX,
        )


@pytest.mark.parametrize(
    ("mode", "health", "path"),
    [
        ("full-duplex", AecHealth.HEALTHY, AcousticSafetyPath.AEC),
        (DuplexMode.FULL_DUPLEX, "healthy", AcousticSafetyPath.AEC),
        (DuplexMode.FULL_DUPLEX, AecHealth.HEALTHY, "aec"),
    ],
)
def test_audio_capability_requires_exact_safety_enum_instances(
    mode: object,
    health: object,
    path: object,
) -> None:
    with pytest.raises(TypeError):
        AudioCapabilityChanged(0, mode, health, path)  # type: ignore[arg-type]


@pytest.mark.parametrize("rebuild_epoch", [True, 0, 1.0])
def test_audio_route_ready_requires_exact_positive_integer_epoch(
    rebuild_epoch: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        AudioRouteReady(  # type: ignore[arg-type]
            rebuild_epoch,
            0,
            DuplexMode.HALF_DUPLEX,
            AecHealth.DEGRADED,
            AcousticSafetyPath.HALF_DUPLEX,
        )


@pytest.mark.asyncio
async def test_route_recovery_preserves_provider_failure_manual_retry_gate() -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        _, epoch = await _start_attempt(coordinator, scheduler)
        await coordinator.submit(ProviderAttemptFailed(epoch, "ProviderError"))
        await coordinator.submit(DeviceRouteChanged(0, RouteKind.DUPLEX))
        await coordinator.submit(
            AudioRouteReady(
                1,
                1,
                DuplexMode.FULL_DUPLEX,
                AecHealth.HEALTHY,
                AcousticSafetyPath.AEC,
            )
        )
        scheduler.advance_ms(5_000)
        await coordinator.flush()

        assert len(effects.dispatches) == 1
        assert coordinator.snapshot.state is (
            SpeculativeVoiceState.LISTENING_AFTER_PROVIDER_FAILURE
        )
        assert coordinator.snapshot.failure_class == "ProviderError"
        assert coordinator.snapshot.admission_open is True
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_route_recovery_preserves_terminal_stt_failure_suspension() -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        frame = _speech(0, started_ns=0)
        await coordinator.submit(frame)
        turn_id = coordinator.snapshot.turn_id
        assert turn_id is not None
        await coordinator.submit(
            TranscriptRevision(
                turn_id=turn_id,
                revision_id=1,
                stable_text="",
                revisable_text="failed transcript draft",
                covered_through_ns=frame.ended_ns,
                mode="rolling-window",
                is_final=True,
                failure_code="fallback_failed",
            )
        )
        await coordinator.submit(DeviceRouteChanged(0, RouteKind.DUPLEX))
        await coordinator.submit(
            AudioRouteReady(
                1,
                1,
                DuplexMode.FULL_DUPLEX,
                AecHealth.HEALTHY,
                AcousticSafetyPath.AEC,
            )
        )
        scheduler.advance_ms(5_000)
        await coordinator.flush()

        assert effects.dispatches == []
        assert coordinator.snapshot.state is SpeculativeVoiceState.DRAFT_SUSPENDED
        assert coordinator.snapshot.failure_class == "fallback_failed"
        assert coordinator.snapshot.admission_open is True
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_device_rebuild_failure_retains_draft_and_suspends_speculation() -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, _ = await _start_attempt(coordinator, scheduler)
        await coordinator.submit(DeviceRouteChanged(0, RouteKind.DUPLEX))
        await coordinator.submit(AudioRouteRebuildFailed(1, 1, "device_unavailable"))
        scheduler.advance_ms(5_000)
        await coordinator.flush()

        assert coordinator.snapshot.state is SpeculativeVoiceState.DRAFT_SUSPENDED
        assert coordinator.snapshot.speculation_suspended is True
        assert effects.drafts == [(turn_id, "hello world", "audio_rebuild_failed")]
        assert coordinator.snapshot.failure_class == "device_unavailable"
        assert len(effects.dispatches) == 1
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_route_rebuild_rejects_stale_generation_and_epoch_callbacks() -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        await _start_attempt(coordinator, scheduler)
        await coordinator.submit(DeviceRouteChanged(0, RouteKind.DUPLEX))
        assert effects.rebuild_requests == [(0, 1)]

        await coordinator.submit(
            AudioRouteReady(
                2,
                2,
                DuplexMode.FULL_DUPLEX,
                AecHealth.WARMING,
                AcousticSafetyPath.ACOUSTIC_ISOLATION,
            )
        )
        assert coordinator.snapshot.state is SpeculativeVoiceState.REBUILDING_AUDIO
        assert coordinator.snapshot.audio_clock_generation == 0
        await coordinator.submit(
            AudioRouteReady(
                1,
                1,
                DuplexMode.FULL_DUPLEX,
                AecHealth.HEALTHY,
                AcousticSafetyPath.AEC,
            )
        )
        assert coordinator.snapshot.state is SpeculativeVoiceState.LISTENING
        assert coordinator.snapshot.audio_clock_generation == 1

        await coordinator.submit(
            AudioRouteReady(
                1,
                2,
                DuplexMode.FULL_DUPLEX,
                AecHealth.WARMING,
                AcousticSafetyPath.ACOUSTIC_ISOLATION,
            )
        )
        assert coordinator.snapshot.audio_clock_generation == 1
        await coordinator.submit(DeviceRouteChanged(1, RouteKind.DUPLEX))
        assert effects.rebuild_requests[-1] == (1, 2)

        await coordinator.submit(AudioRouteRebuildFailed(1, 3, "stale_failure"))
        assert coordinator.snapshot.state is SpeculativeVoiceState.REBUILDING_AUDIO
        await coordinator.submit(
            AudioRouteReady(
                2,
                3,
                DuplexMode.FULL_DUPLEX,
                AecHealth.WARMING,
                AcousticSafetyPath.ACOUSTIC_ISOLATION,
            )
        )
        assert coordinator.snapshot.audio_clock_generation == 3
        assert coordinator.snapshot.state is SpeculativeVoiceState.LISTENING
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_route_rebuild_cannot_make_a_stale_partial_transcript_fresh() -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        frame = _speech(0, started_ns=20_000_000)
        await coordinator.submit(frame)
        turn_id = coordinator.snapshot.turn_id
        assert turn_id is not None
        await coordinator.submit(
            _revision(turn_id, 1, "partial", frame.ended_ns - 10_000_001)
        )

        await coordinator.submit(DeviceRouteChanged(0, RouteKind.DUPLEX))
        await coordinator.submit(
            AudioRouteReady(
                1,
                1,
                DuplexMode.FULL_DUPLEX,
                AecHealth.HEALTHY,
                AcousticSafetyPath.AEC,
            )
        )
        scheduler.advance_ms(700)
        await coordinator.flush()

        assert effects.dispatches == []
        assert coordinator.snapshot.state is SpeculativeVoiceState.TRANSCRIBING

        await coordinator.submit(_revision(turn_id, 2, "complete", frame.ended_ns))
        assert effects.dispatches == [(turn_id, 2, "complete")]
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_slow_old_attempt_cleanup_cannot_clobber_audio_rebuild_state() -> None:
    coordinator, scheduler, effects = await _coordinator(auto_cleanup=False)
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        await coordinator.submit(DeviceRouteChanged(0, RouteKind.DUPLEX))

        scheduler.advance_ms(2_000)
        await coordinator.flush()
        assert coordinator.snapshot.state is SpeculativeVoiceState.REBUILDING_AUDIO
        assert coordinator.snapshot.serialized_conservative is True

        await coordinator.submit(
            AudioRouteReady(
                1,
                1,
                DuplexMode.FULL_DUPLEX,
                AecHealth.HEALTHY,
                AcousticSafetyPath.AEC,
            )
        )
        assert coordinator.snapshot.state is (
            SpeculativeVoiceState.SERIALIZED_CONSERVATIVE
        )
        scheduler.advance_ms(2_000)
        await coordinator.flush()
        assert len(effects.dispatches) == 1

        effects.complete_cleanup(epoch)
        await coordinator.flush()
        assert effects.dispatches[-1] == (turn_id, 2, "hello world")
    finally:
        for completion in effects.cleanup_futures.values():
            if not completion.done():
                completion.set_result(None)
        await coordinator.close()


@pytest.mark.asyncio
async def test_provider_failure_retains_same_turn_draft_without_automatic_retry() -> (
    None
):
    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        await coordinator.submit(ProviderAttemptFailed(epoch, "ProviderError"))
        scheduler.advance_ms(5_000)
        await coordinator.flush()

        assert coordinator.snapshot.state is (
            SpeculativeVoiceState.LISTENING_AFTER_PROVIDER_FAILURE
        )
        assert coordinator.snapshot.failure_class == "ProviderError"
        assert coordinator.snapshot.turn_id == turn_id
        assert effects.drafts == [(turn_id, "hello world", "provider_failed")]
        assert effects.promotions == []
        assert len(effects.dispatches) == 1

        resumed = _speech(1, started_ns=scheduler.now_ns)
        await coordinator.submit(resumed)
        await coordinator.submit(
            _revision(turn_id, 2, "hello world continued", resumed.ended_ns),
        )
        scheduler.advance_ms(700)
        await coordinator.flush()
        assert effects.dispatches[-1][2] == "hello world continued"
        assert len(effects.dispatches) == 2
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_slow_prior_cleanup_preserves_provider_failure_manual_retry() -> None:
    coordinator, scheduler, effects = await _coordinator(auto_cleanup=False)
    try:
        turn_id, first_epoch = await _start_attempt(coordinator, scheduler, text="one")
        speech = _speech(1, started_ns=scheduler.now_ns)
        await coordinator.submit(speech)
        await coordinator.submit(_revision(turn_id, 2, "one two", speech.ended_ns))
        scheduler.advance_ms(700)
        await coordinator.flush()
        second_epoch = coordinator.snapshot.current_attempt_epoch
        assert second_epoch is not None

        await coordinator.submit(ProviderAttemptFailed(second_epoch, "ProviderError"))
        scheduler.advance_ms(1_300)
        await coordinator.flush()
        assert coordinator.snapshot.serialized_conservative is True
        assert coordinator.snapshot.state is (
            SpeculativeVoiceState.LISTENING_AFTER_PROVIDER_FAILURE
        )

        effects.complete_cleanup(first_epoch)
        await coordinator.flush()
        await coordinator.submit(ManualRetry())

        assert len(effects.dispatches) == 3
        assert effects.dispatches[-1] == (turn_id, 3, "one two")
    finally:
        for completion in effects.cleanup_futures.values():
            if not completion.done():
                completion.set_result(AttemptCleanupOutcome.CLEAN)
        await coordinator.close()


@pytest.mark.asyncio
async def test_half_duplex_provider_failure_reopens_same_turn_admission() -> None:
    coordinator, scheduler, _ = await _coordinator(mode=DuplexMode.HALF_DUPLEX)
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        await coordinator.submit(AttemptPlaybackStarted(epoch))
        assert coordinator.snapshot.admission_open is False

        await coordinator.submit(ProviderAttemptFailed(epoch, "ProviderError"))
        assert coordinator.snapshot.state is (
            SpeculativeVoiceState.LISTENING_AFTER_PROVIDER_FAILURE
        )
        assert coordinator.snapshot.admission_open is True

        resumed = _speech(1, started_ns=scheduler.now_ns)
        await coordinator.submit(resumed)
        assert coordinator.snapshot.turn_id == turn_id
        assert coordinator.snapshot.last_admitted_sequence == 1
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_capability_demotion_recomputes_speaking_admission_projection() -> None:
    coordinator, scheduler, _ = await _coordinator()
    try:
        _, epoch = await _start_attempt(coordinator, scheduler)
        await coordinator.submit(AttemptPlaybackStarted(epoch))
        assert coordinator.snapshot.state is SpeculativeVoiceState.SPEAKING
        assert coordinator.snapshot.admission_open is True

        await coordinator.submit(
            AudioCapabilityChanged(
                0,
                DuplexMode.HALF_DUPLEX,
                AecHealth.DEGRADED,
                AcousticSafetyPath.HALF_DUPLEX,
            )
        )

        assert coordinator.snapshot.duplex_mode is DuplexMode.HALF_DUPLEX
        assert coordinator.snapshot.admission_open is False
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_manual_retry_reuses_exact_provider_failure_draft() -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        await coordinator.submit(ProviderAttemptFailed(epoch, "ProviderError"))
        await coordinator.submit(ManualRetry())

        assert effects.dispatches[-1][0] == turn_id
        assert effects.dispatches[-1][2] == "hello world"
        assert len(effects.dispatches) == 2
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_terminal_full_duplex_drains_capture_and_seals_before_promotion() -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        boundary = 1_000_000_000
        effects.drain_receipt = DrainReceipt(
            capture_watermark_ns=boundary + 1,
            capture_sequence=0,
            dsp_sequence=0,
            vad_sequence=0,
            clock_generation=0,
        )
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackStarted(epoch))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, boundary))
        await coordinator.flush()

        assert effects.drain_calls == [boundary]
        assert effects.classification_drain_calls == [(boundary, 0)]
        assert effects.seal_calls == [0]
        assert effects.promotions == [
            (turn_id, epoch, "hello world", "answer", boundary)
        ]
        assert coordinator.snapshot.state is SpeculativeVoiceState.IDLE
    finally:
        await coordinator.close()


@pytest.mark.parametrize(
    ("initial_mode", "mode_history"),
    [
        (DuplexMode.FULL_DUPLEX, (DuplexMode.HALF_DUPLEX,)),
        (
            DuplexMode.HALF_DUPLEX,
            (DuplexMode.FULL_DUPLEX, DuplexMode.HALF_DUPLEX),
        ),
    ],
    ids=["full-half", "half-full-half"],
)
@pytest.mark.asyncio
async def test_terminal_capture_drain_requirement_is_monotonic_during_playback(
    initial_mode: DuplexMode,
    mode_history: tuple[DuplexMode, ...],
) -> None:
    coordinator, scheduler, effects = await _coordinator(mode=initial_mode)
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        boundary = 1_000_000_000
        effects.drain_receipt = DrainReceipt(boundary + 1, 0, 0, 0, 0)
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackStarted(epoch))
        for mode in mode_history:
            await coordinator.submit(
                AudioCapabilityChanged(
                    0,
                    mode,
                    (
                        AecHealth.HEALTHY
                        if mode is DuplexMode.FULL_DUPLEX
                        else AecHealth.DEGRADED
                    ),
                    (
                        AcousticSafetyPath.AEC
                        if mode is DuplexMode.FULL_DUPLEX
                        else AcousticSafetyPath.HALF_DUPLEX
                    ),
                )
            )

        await coordinator.submit(AttemptPlaybackTerminal(epoch, boundary))
        await coordinator.flush()

        assert effects.drain_calls == [boundary]
        assert effects.classification_drain_calls == [(boundary, 0)]
        assert effects.promotions == [
            (turn_id, epoch, "hello world", "answer", boundary)
        ]
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_capture_drain_requirement_resets_for_new_half_duplex_attempt() -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        _, first_epoch = await _start_attempt(coordinator, scheduler)
        first_boundary = 1_000_000_000
        effects.drain_receipt = DrainReceipt(first_boundary + 1, 0, 0, 0, 0)
        await coordinator.submit(AttemptGenerationCompleted(first_epoch, "first"))
        await coordinator.submit(AttemptPlaybackStarted(first_epoch))
        await coordinator.submit(AttemptPlaybackTerminal(first_epoch, first_boundary))
        await coordinator.flush()

        await coordinator.submit(
            AudioCapabilityChanged(
                0,
                DuplexMode.HALF_DUPLEX,
                AecHealth.DEGRADED,
                AcousticSafetyPath.HALF_DUPLEX,
            )
        )
        _, second_epoch = await _start_attempt(
            coordinator,
            scheduler,
            sequence=1,
        )
        second_boundary = 2_000_000_000
        await coordinator.submit(AttemptGenerationCompleted(second_epoch, "second"))
        await coordinator.submit(AttemptPlaybackStarted(second_epoch))
        await coordinator.submit(AttemptPlaybackTerminal(second_epoch, second_boundary))
        await coordinator.flush()

        assert effects.drain_calls == [first_boundary]
        assert effects.classification_drain_calls == [
            (first_boundary, 0),
            (second_boundary, 0),
        ]
        assert len(effects.promotions) == 2
    finally:
        await coordinator.close()


@pytest.mark.parametrize(
    ("initial_mode", "mode_history"),
    [
        (DuplexMode.FULL_DUPLEX, (DuplexMode.HALF_DUPLEX,)),
        (
            DuplexMode.HALF_DUPLEX,
            (DuplexMode.FULL_DUPLEX, DuplexMode.HALF_DUPLEX),
        ),
    ],
    ids=["full-half", "half-full-half"],
)
@pytest.mark.asyncio
async def test_terminal_capture_drain_failure_after_duplex_demotion_blocks_promotion(
    initial_mode: DuplexMode,
    mode_history: tuple[DuplexMode, ...],
) -> None:
    coordinator, scheduler, effects = await _coordinator(mode=initial_mode)
    effects.drain_error = CaptureDrainError("capture drain failed")
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        boundary = 1_000_000_000
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackStarted(epoch))
        for mode in mode_history:
            await coordinator.submit(
                AudioCapabilityChanged(
                    0,
                    mode,
                    (
                        AecHealth.HEALTHY
                        if mode is DuplexMode.FULL_DUPLEX
                        else AecHealth.DEGRADED
                    ),
                    (
                        AcousticSafetyPath.AEC
                        if mode is DuplexMode.FULL_DUPLEX
                        else AcousticSafetyPath.HALF_DUPLEX
                    ),
                )
            )

        await coordinator.submit(AttemptPlaybackTerminal(epoch, boundary))
        await coordinator.flush()

        assert effects.drain_calls == [boundary]
        assert effects.promotions == []
        assert effects.drafts == [(turn_id, "hello world", "capture_sync_failed")]
        assert coordinator.snapshot.state is SpeculativeVoiceState.REBUILDING_AUDIO
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_terminal_cannot_cross_pending_classification_before_admit_replay() -> (
    None
):
    coordinator, scheduler, effects = await _coordinator()
    effects.classification_release = asyncio.Event()
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        boundary = 1_000_000_000
        effects.drain_receipt = DrainReceipt(boundary + 1, 0, 0, 0, 0)
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackStarted(epoch))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, boundary))
        await coordinator.flush()

        assert effects.classification_drain_calls == [(boundary, 0)]
        assert effects.seal_calls == []
        assert effects.promotions == []
        assert coordinator.snapshot.state is SpeculativeVoiceState.SEALING

        for sequence in range(1, 6):
            await coordinator.submit(
                _speech(sequence, started_ns=boundary - (6 - sequence) * 10_000_000)
            )

        effects.classification_release.set()
        await coordinator.flush()

        assert effects.promotions == []
        assert coordinator.snapshot.turn_id == turn_id
        assert coordinator.snapshot.last_admitted_sequence == 5
        assert epoch in effects.fenced_epochs
    finally:
        effects.classification_release.set()
        await coordinator.close()


@pytest.mark.asyncio
async def test_terminal_checks_failed_classification_after_capability_demotion() -> (
    None
):
    coordinator, scheduler, effects = await _coordinator()
    effects.classification_error = CaptureDrainError("classification failed")
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        boundary = 1_000_000_000
        effects.drain_receipt = DrainReceipt(boundary + 1, 0, 0, 0, 0)
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackStarted(epoch))
        await coordinator.submit(
            AudioCapabilityChanged(
                0,
                DuplexMode.HALF_DUPLEX,
                AecHealth.DEGRADED,
                AcousticSafetyPath.HALF_DUPLEX,
            )
        )
        await coordinator.submit(AttemptPlaybackTerminal(epoch, boundary))
        await coordinator.flush()

        assert effects.classification_drain_calls == [(boundary, 0)]
        assert effects.promotions == []
        assert effects.drafts == [(turn_id, "hello world", "capture_sync_failed")]
        assert coordinator.snapshot.state is SpeculativeVoiceState.REBUILDING_AUDIO
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_terminal_accepts_equal_capture_boundary_with_dsp_and_vad_ahead() -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        boundary = 1_000_000_000
        effects.drain_receipt = DrainReceipt(boundary + 1, 0, 2, 3, 0)
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, boundary))
        await coordinator.flush()

        assert effects.promotions == [
            (turn_id, epoch, "hello world", "answer", boundary)
        ]
    finally:
        await coordinator.close()


@pytest.mark.parametrize(
    "sealed_text",
    [
        "HELLO WORLD",
        "hello, world!",
        "  hello  world \n",
    ],
    ids=["case-only", "punctuation-only", "whitespace-only"],
)
@pytest.mark.asyncio
async def test_cosmetic_terminal_revision_promotes_exact_winning_prompt(
    sealed_text: str,
) -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        boundary = 1_000_000_000
        effects.drain_receipt = DrainReceipt(boundary + 1, 0, 0, 0, 0)
        effects.seal_revision = TranscriptRevision(
            turn_id=turn_id,
            revision_id=2,
            stable_text="",
            revisable_text=sealed_text,
            covered_through_ns=10_000_000,
            mode="live",
            is_final=True,
        )
        await coordinator.submit(AttemptGenerationCompleted(epoch, "winning answer"))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, boundary))
        await coordinator.flush()

        winning_prompt = effects.dispatches[-1][2]
        assert winning_prompt == "hello world"
        assert effects.promotions == [
            (turn_id, epoch, winning_prompt, "winning answer", boundary)
        ]
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_material_revision_returned_by_terminal_seal_fences_stale_attempt() -> (
    None
):
    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        boundary = 1_000_000_000
        effects.drain_receipt = DrainReceipt(boundary + 1, 0, 0, 0, 0)
        effects.seal_revision = TranscriptRevision(
            turn_id=turn_id,
            revision_id=2,
            stable_text="",
            revisable_text="hello corrected world",
            covered_through_ns=10_000_000,
            mode="live",
            is_final=True,
        )
        await coordinator.submit(AttemptGenerationCompleted(epoch, "stale answer"))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, boundary))
        await coordinator.flush()

        assert effects.promotions == []
        assert epoch in effects.fenced_epochs
        assert coordinator.snapshot.transcript_text == "hello corrected world"
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_failed_revision_returned_by_terminal_seal_cannot_promote_attempt() -> (
    None
):
    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        boundary = 1_000_000_000
        effects.drain_receipt = DrainReceipt(boundary + 1, 0, 0, 0, 0)
        effects.seal_revision = TranscriptRevision(
            turn_id=turn_id,
            revision_id=2,
            stable_text="",
            revisable_text="hello world",
            covered_through_ns=10_000_000,
            mode="rolling-window",
            is_final=True,
            failure_code="fallback_failed",
        )
        await coordinator.submit(AttemptGenerationCompleted(epoch, "stale answer"))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, boundary))
        await coordinator.flush()

        assert effects.promotions == []
        assert epoch in effects.fenced_epochs
        assert effects.drafts == [(turn_id, "hello world", "stt_failed")]
        assert coordinator.snapshot.state is SpeculativeVoiceState.DRAFT_SUSPENDED
        assert coordinator.snapshot.failure_class == "fallback_failed"
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_pre_boundary_speech_racing_terminal_seal_extends_same_turn() -> None:
    coordinator, scheduler, effects = await _coordinator()
    effects.drain_future = asyncio.get_running_loop().create_future()
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        boundary = 1_000_000_000
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackStarted(epoch))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, boundary))
        await coordinator.submit(_speech(1, started_ns=boundary - 1))

        assert coordinator.snapshot.turn_id == turn_id
        assert coordinator.snapshot.current_attempt_epoch is None
        assert epoch in effects.fenced_epochs
        assert effects.promotions == []
    finally:
        if effects.drain_future is not None and not effects.drain_future.done():
            effects.drain_future.cancel()
        await coordinator.close()


@pytest.mark.asyncio
async def test_post_boundary_speech_queues_new_turn_without_rewriting_winner() -> None:
    coordinator, scheduler, effects = await _coordinator()
    effects.drain_future = asyncio.get_running_loop().create_future()
    try:
        old_turn, epoch = await _start_attempt(coordinator, scheduler)
        boundary = 1_000_000_000
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackStarted(epoch))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, boundary))
        later = _speech(1, started_ns=boundary + 1)
        await coordinator.submit(later)

        assert epoch not in effects.fenced_epochs
        assert effects.promotions == []
        effects.drain_future.set_result(DrainReceipt(boundary + 10_000_000, 0, 0, 0, 0))
        await coordinator.flush()

        assert effects.promotions == [
            (old_turn, epoch, "hello world", "answer", boundary)
        ]
        assert coordinator.snapshot.turn_id is not None
        assert coordinator.snapshot.turn_id != old_turn
        assert coordinator.snapshot.last_admitted_sequence == 1
    finally:
        await coordinator.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_first", [False, True])
async def test_known_boundary_routes_speech_before_generation_or_terminal(
    terminal_first,
):
    from tldw_chatbook.Chat.console_speculative_voice import (
        AttemptPlaybackBoundaryKnown,
    )

    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        effects.drain_future = asyncio.get_running_loop().create_future()
        await coordinator.submit(AttemptPlaybackStarted(epoch))
        await coordinator.submit(AttemptPlaybackBoundaryKnown(epoch, 1_000_000_000))
        assert effects.promotions == []
        if terminal_first:
            await coordinator.submit(AttemptPlaybackTerminal(epoch, 1_000_000_000))
        await coordinator.submit(_speech(1, started_ns=1_000_000_001))
        assert coordinator.snapshot.turn_id == turn_id
        assert coordinator.snapshot.pending_next_turn_id is not None
        assert coordinator.snapshot.current_attempt_epoch == epoch
        if not terminal_first:
            await coordinator.submit(AttemptPlaybackTerminal(epoch, 1_000_000_000))
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.flush()
        assert effects.drain_calls == [1_000_000_000]
        assert effects.promotions == []
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_known_boundary_backlog_at_boundary_still_cancels_same_attempt():
    from tldw_chatbook.Chat.console_speculative_voice import (
        AttemptPlaybackBoundaryKnown,
    )

    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        await coordinator.submit(AttemptPlaybackStarted(epoch))
        await coordinator.submit(AttemptPlaybackBoundaryKnown(epoch, 1_000_000_000))
        scheduler.advance_ms(400)
        await coordinator.submit(_speech(1, started_ns=1_000_000_000))
        assert coordinator.snapshot.turn_id == turn_id
        assert coordinator.snapshot.current_attempt_epoch is None
        await coordinator.submit(AttemptPlaybackTerminal(epoch, 1_000_000_000))
        await coordinator.submit(AttemptGenerationCompleted(epoch, "obsolete"))
        assert effects.promotions == []
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_generation_arriving_after_known_boundary_deadline_cannot_win_between_pumps():
    from tldw_chatbook.Chat.console_speculative_voice import (
        AttemptPlaybackBoundaryKnown,
    )

    coordinator, scheduler, effects = await _coordinator()
    try:
        _, epoch = await _start_attempt(coordinator, scheduler)
        effects.drain_receipt = DrainReceipt(1_000_000_001, 0, 0, 0, 0)
        await coordinator.submit(AttemptPlaybackBoundaryKnown(epoch, 1_000_000_000))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, 1_000_000_000))
        scheduler.advance_ms(800)  # exactly DAC+500 ms, before another owner pump
        await coordinator.submit(AttemptGenerationCompleted(epoch, "too late"))
        await coordinator.flush()
        assert effects.promotions == []
        assert coordinator.snapshot.state is SpeculativeVoiceState.REBUILDING_AUDIO
    finally:
        await coordinator.close()


@pytest.mark.parametrize(
    "settlement",
    ("success", "failure", "deferred-route"),
)
@pytest.mark.asyncio
async def test_post_boundary_revision_survives_terminal_sealing(
    settlement: str,
) -> None:
    coordinator, scheduler, effects = await _coordinator()
    effects.drain_future = asyncio.get_running_loop().create_future()
    effects.promotion_future = asyncio.get_running_loop().create_future()
    try:
        old_turn, epoch = await _start_attempt(coordinator, scheduler)
        boundary = 1_000_000_000
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, boundary))
        assert coordinator.snapshot.state is SpeculativeVoiceState.SEALING

        later = _speech(1, started_ns=boundary + 1)
        await coordinator.submit(later)
        pending_turn = coordinator.snapshot.pending_next_turn_id
        assert pending_turn is not None
        await coordinator.submit(
            _revision(pending_turn, 1, "next request", later.ended_ns)
        )

        effects.drain_future.set_result(DrainReceipt(boundary + 1, 0, 0, 0, 0))
        await coordinator.flush()
        assert coordinator.snapshot.state is SpeculativeVoiceState.PROMOTING

        if settlement == "deferred-route":
            await coordinator.submit(DeviceRouteChanged(0, RouteKind.DUPLEX))
            effects.promotion_future.set_result(None)
        elif settlement == "failure":
            effects.promotion_future.set_exception(RuntimeError("publication failed"))
        else:
            effects.promotion_future.set_result(None)
        await coordinator.flush()

        next_turn = coordinator.snapshot.turn_id
        assert next_turn == pending_turn
        assert next_turn != old_turn
        assert coordinator.snapshot.last_admitted_sequence == later.sequence
        assert coordinator.snapshot.transcript_text == "next request"
        if settlement == "failure":
            assert coordinator.snapshot.state is SpeculativeVoiceState.DRAFT_SUSPENDED
            assert effects.drafts[-1] == (
                next_turn,
                "next request",
                "promotion_failed_pending_next_turn",
            )
        elif settlement == "deferred-route":
            assert coordinator.snapshot.state is SpeculativeVoiceState.REBUILDING_AUDIO
            await coordinator.submit(
                AudioRouteReady(
                    1,
                    1,
                    DuplexMode.FULL_DUPLEX,
                    AecHealth.HEALTHY,
                    AcousticSafetyPath.AEC,
                )
            )
            assert coordinator.snapshot.state is SpeculativeVoiceState.LISTENING
        else:
            assert coordinator.snapshot.state is SpeculativeVoiceState.LISTENING
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_speech_during_promotion_waits_for_active_leaf_publication() -> None:
    coordinator, scheduler, effects = await _coordinator()
    effects.promotion_future = asyncio.get_running_loop().create_future()
    try:
        old_turn, epoch = await _start_attempt(coordinator, scheduler)
        boundary = 1_000_000_000
        effects.drain_receipt = DrainReceipt(boundary + 1, 0, 0, 0, 0)
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, boundary))
        await coordinator.flush()

        assert coordinator.snapshot.state is SpeculativeVoiceState.PROMOTING
        later = _speech(1, started_ns=boundary + 1)
        await coordinator.submit(later)
        assert coordinator.snapshot.pending_next_frame_count == 1
        pending_turn = coordinator.snapshot.pending_next_turn_id
        assert pending_turn is not None
        await coordinator.submit(
            _revision(pending_turn, 1, "next request", later.ended_ns)
        )
        assert len(effects.dispatches) == 1

        effects.promotion_future.set_result(None)
        await coordinator.flush()

        assert coordinator.snapshot.turn_id is not None
        assert coordinator.snapshot.turn_id != old_turn
        assert coordinator.snapshot.last_admitted_sequence == later.sequence
        assert coordinator.snapshot.transcript_text == "next request"
        assert coordinator.snapshot.state is SpeculativeVoiceState.LISTENING
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_promotion_failure_preserves_buffered_speech_as_separate_draft() -> None:
    coordinator, scheduler, effects = await _coordinator()
    effects.promotion_future = asyncio.get_running_loop().create_future()
    try:
        old_turn, epoch = await _start_attempt(coordinator, scheduler)
        boundary = 1_000_000_000
        effects.drain_receipt = DrainReceipt(boundary + 1, 0, 0, 0, 0)
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, boundary))
        await coordinator.flush()
        later = _speech(1, started_ns=boundary + 1)
        await coordinator.submit(later)
        pending_turn = coordinator.snapshot.pending_next_turn_id
        assert pending_turn is not None
        await coordinator.submit(
            _revision(pending_turn, 1, "next request", later.ended_ns)
        )

        effects.promotion_future.set_exception(RuntimeError("publication failed"))
        await coordinator.flush()

        next_turn = coordinator.snapshot.turn_id
        assert next_turn is not None and next_turn != old_turn
        assert coordinator.snapshot.state is SpeculativeVoiceState.DRAFT_SUSPENDED
        assert coordinator.snapshot.failure_class == "promotion_failed"
        assert coordinator.snapshot.last_admitted_sequence == later.sequence
        assert effects.drafts[-1] == (
            next_turn,
            "next request",
            "promotion_failed_pending_next_turn",
        )
        assert len(effects.dispatches) == 1
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_manual_interruption_cannot_restart_claimed_turn_during_promotion() -> (
    None
):
    coordinator, scheduler, effects = await _coordinator()
    effects.promotion_future = asyncio.get_running_loop().create_future()
    try:
        _turn_id, epoch = await _start_attempt(coordinator, scheduler)
        boundary = 1_000_000_000
        effects.drain_receipt = DrainReceipt(boundary + 1, 0, 0, 0, 0)
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, boundary))
        await coordinator.flush()

        await coordinator.submit(ManualInterruption())

        assert coordinator.snapshot.state is SpeculativeVoiceState.PROMOTING
        assert epoch not in effects.fenced_epochs
        assert len(effects.dispatches) == 1
        effects.promotion_future.set_result(None)
        await coordinator.flush()
        assert coordinator.snapshot.state is SpeculativeVoiceState.IDLE
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_route_change_waits_for_claimed_promotion_before_rebuild() -> None:
    coordinator, scheduler, effects = await _coordinator()
    effects.promotion_future = asyncio.get_running_loop().create_future()
    try:
        _turn_id, epoch = await _start_attempt(coordinator, scheduler)
        boundary = 1_000_000_000
        effects.drain_receipt = DrainReceipt(boundary + 1, 0, 0, 0, 0)
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, boundary))
        await coordinator.flush()

        await coordinator.submit(DeviceRouteChanged(0, RouteKind.DUPLEX))

        assert coordinator.snapshot.state is SpeculativeVoiceState.PROMOTING
        assert epoch not in effects.fenced_epochs
        assert effects.rebuilds == []
        effects.promotion_future.set_result(None)
        await coordinator.flush()
        assert coordinator.snapshot.state is SpeculativeVoiceState.REBUILDING_AUDIO
        assert effects.rebuilds == [0]
        assert len(effects.dispatches) == 1
    finally:
        await coordinator.close()


@pytest.mark.parametrize("promotion_succeeds", [True, False])
@pytest.mark.asyncio
async def test_deferred_route_change_preserves_speech_buffered_during_promotion(
    promotion_succeeds: bool,
) -> None:
    coordinator, scheduler, effects = await _coordinator()
    effects.promotion_future = asyncio.get_running_loop().create_future()
    try:
        old_turn, epoch = await _start_attempt(coordinator, scheduler)
        boundary = 1_000_000_000
        effects.drain_receipt = DrainReceipt(boundary + 1, 0, 0, 0, 0)
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, boundary))
        await coordinator.flush()
        later = _speech(1, started_ns=boundary + 1)
        await coordinator.submit(later)
        pending_turn = coordinator.snapshot.pending_next_turn_id
        assert pending_turn is not None
        await coordinator.submit(
            _revision(pending_turn, 1, "next request", later.ended_ns)
        )
        await coordinator.submit(DeviceRouteChanged(0, RouteKind.DUPLEX))

        if promotion_succeeds:
            effects.promotion_future.set_result(None)
        else:
            effects.promotion_future.set_exception(RuntimeError("publication failed"))
        await coordinator.flush()

        next_turn = coordinator.snapshot.turn_id
        assert next_turn is not None and next_turn != old_turn
        assert coordinator.snapshot.state is SpeculativeVoiceState.REBUILDING_AUDIO
        assert coordinator.snapshot.last_admitted_sequence == later.sequence
        assert coordinator.snapshot.transcript_text == "next request"
        await coordinator.submit(
            AudioRouteReady(
                1,
                1,
                DuplexMode.FULL_DUPLEX,
                AecHealth.HEALTHY,
                AcousticSafetyPath.AEC,
            )
        )

        if promotion_succeeds:
            assert coordinator.snapshot.state is SpeculativeVoiceState.LISTENING
        else:
            assert coordinator.snapshot.state is SpeculativeVoiceState.DRAFT_SUSPENDED
            assert coordinator.snapshot.failure_class == "promotion_failed"
            assert effects.drafts[-1] == (
                next_turn,
                "next request",
                "promotion_failed_pending_next_turn",
            )
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_route_reset_discards_post_boundary_frames_from_old_clock() -> None:
    coordinator, scheduler, effects = await _coordinator()
    effects.drain_future = asyncio.get_running_loop().create_future()
    try:
        _, epoch = await _start_attempt(coordinator, scheduler)
        boundary = 1_000_000_000
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, boundary))
        later = _speech(1, started_ns=boundary + 1)
        await coordinator.submit(later)
        pending_turn = coordinator.snapshot.pending_next_turn_id
        assert pending_turn is not None
        await coordinator.submit(
            _revision(pending_turn, 1, "discarded next turn", later.ended_ns)
        )
        assert coordinator.snapshot.pending_next_frame_count == 1

        await coordinator.submit(DeviceRouteChanged(0, RouteKind.DUPLEX))
        assert coordinator.snapshot.state is SpeculativeVoiceState.REBUILDING_AUDIO
        assert coordinator.snapshot.pending_next_frame_count == 0
        assert coordinator.snapshot.pending_next_turn_id is None
        assert coordinator.snapshot.pending_next_transcript_text == ""
        assert coordinator._pending_next_revision_id == 0
        assert coordinator._pending_next_covered_through_ns == 0
        assert coordinator._pending_next_failure_code is None
    finally:
        if effects.drain_future is not None and not effects.drain_future.done():
            effects.drain_future.cancel()
        await coordinator.close()


@pytest.mark.parametrize("discard_path", ("seal-failure", "cleanup-failure"))
@pytest.mark.asyncio
async def test_terminal_discard_paths_clear_complete_pending_turn_state(
    discard_path: str,
) -> None:
    coordinator, scheduler, effects = await _coordinator()
    effects.drain_future = asyncio.get_running_loop().create_future()
    try:
        _, epoch = await _start_attempt(coordinator, scheduler)
        boundary = 1_000_000_000
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, boundary))
        later = _speech(1, started_ns=boundary + 1)
        await coordinator.submit(later)
        pending_turn = coordinator.snapshot.pending_next_turn_id
        assert pending_turn is not None
        await coordinator.submit(
            TranscriptRevision(
                turn_id=pending_turn,
                revision_id=1,
                stable_text="",
                revisable_text="discarded next turn",
                covered_through_ns=later.ended_ns,
                mode="rolling-window",
                is_final=True,
                failure_code="fallback_failed",
            )
        )

        if discard_path == "seal-failure":
            effects.drain_future.set_exception(CaptureDrainError("capture gap"))
            await coordinator.flush()
        else:
            await coordinator._suspend_for_stuck_cleanup()

        assert coordinator.snapshot.pending_next_frame_count == 0
        assert coordinator.snapshot.pending_next_turn_id is None
        assert coordinator.snapshot.pending_next_transcript_text == ""
        assert coordinator._pending_next_revision_id == 0
        assert coordinator._pending_next_covered_through_ns == 0
        assert coordinator._pending_next_failure_code is None
    finally:
        if effects.drain_future is not None and not effects.drain_future.done():
            effects.drain_future.cancel()
        await coordinator.close()


@pytest.mark.asyncio
async def test_multiple_post_boundary_frames_are_preserved_and_reset_new_turn_timer() -> (
    None
):
    coordinator, scheduler, effects = await _coordinator()
    effects.drain_future = asyncio.get_running_loop().create_future()
    try:
        old_turn, epoch = await _start_attempt(coordinator, scheduler)
        boundary = 1_000_000_000
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, boundary))
        await coordinator.submit(_speech(1, started_ns=boundary + 1))
        scheduler.advance_ms(100)
        latest = _speech(2, started_ns=boundary + 100_000_001)
        await coordinator.submit(latest)

        assert coordinator.snapshot.pending_next_frame_count == 2
        effects.drain_future.set_result(DrainReceipt(boundary + 1, 0, 0, 0, 0))
        await coordinator.flush()

        new_turn = coordinator.snapshot.turn_id
        assert new_turn is not None and new_turn != old_turn
        assert coordinator.snapshot.last_admitted_sequence == 2
        await coordinator.submit(_revision(new_turn, 1, "next turn", latest.ended_ns))
        scheduler.advance_ms(699)
        await coordinator.flush()
        assert len(effects.dispatches) == 1
        scheduler.advance_ms(1)
        await coordinator.flush()
        assert effects.dispatches[-1][0] == new_turn
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_half_duplex_accepts_post_boundary_speech_as_next_turn() -> None:
    coordinator, scheduler, effects = await _coordinator(mode=DuplexMode.HALF_DUPLEX)
    effects.seal_future = asyncio.get_running_loop().create_future()
    try:
        old_turn, epoch = await _start_attempt(coordinator, scheduler)
        boundary = 1_000_000_000
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackStarted(epoch))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, boundary))
        await coordinator.submit(_speech(1, started_ns=boundary + 1))

        effects.seal_future.set_result(
            TranscriptRevision(
                turn_id=old_turn,
                revision_id=2,
                stable_text="hello world",
                revisable_text="",
                covered_through_ns=10_000_000,
                mode="live",
                is_final=True,
            )
        )
        await coordinator.flush()

        assert coordinator.snapshot.turn_id is not None
        assert coordinator.snapshot.turn_id != old_turn
        assert coordinator.snapshot.last_admitted_sequence == 1
    finally:
        await coordinator.close()


@pytest.mark.parametrize(
    "receipt",
    [
        DrainReceipt(1_000_000_000, 0, 0, 0, 0),
        DrainReceipt(1_000_000_001, 1, 0, 1, 0),
        DrainReceipt(1_000_000_001, 0, 0, 0, 1),
    ],
)
@pytest.mark.asyncio
async def test_invalid_terminal_drain_receipt_preserves_draft_and_suspends(
    receipt: DrainReceipt,
) -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        effects.drain_receipt = receipt
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, 1_000_000_000))
        await coordinator.flush()

        assert effects.promotions == []
        assert effects.drafts == [(turn_id, "hello world", "capture_sync_failed")]
        assert coordinator.snapshot.state is SpeculativeVoiceState.REBUILDING_AUDIO
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_terminal_drain_timeout_at_500ms_fails_closed_without_wall_sleep() -> (
    None
):
    coordinator, scheduler, effects = await _coordinator()
    effects.drain_future = asyncio.get_running_loop().create_future()
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        boundary = 1_000_000_000
        await coordinator.submit(AttemptPlaybackTerminal(epoch, boundary))
        scheduler.advance_ms(799)
        await coordinator.flush()
        assert coordinator.snapshot.state is SpeculativeVoiceState.SEALING
        scheduler.advance_ms(1)
        await coordinator.flush()

        assert effects.promotions == []
        assert effects.drafts == [(turn_id, "hello world", "capture_sync_failed")]
        assert coordinator.snapshot.state is SpeculativeVoiceState.REBUILDING_AUDIO
    finally:
        if effects.drain_future is not None and not effects.drain_future.done():
            effects.drain_future.cancel()
        await coordinator.close()


@pytest.mark.asyncio
async def test_pending_classification_uses_terminal_deadline_as_fail_closed_bound() -> (
    None
):
    coordinator, scheduler, effects = await _coordinator()
    effects.classification_release = asyncio.Event()
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        boundary = 1_000_000_000
        effects.drain_receipt = DrainReceipt(boundary + 1, 0, 0, 0, 0)
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, boundary))
        scheduler.advance_ms(799)
        await coordinator.flush()
        assert coordinator.snapshot.state is SpeculativeVoiceState.SEALING
        assert effects.classification_drain_calls == [(boundary, 0)]

        scheduler.advance_ms(1)
        await coordinator.flush()

        assert effects.promotions == []
        assert effects.drafts == [(turn_id, "hello world", "capture_sync_failed")]
        assert coordinator.snapshot.state is SpeculativeVoiceState.REBUILDING_AUDIO
    finally:
        effects.classification_release.set()
        await coordinator.close()


@pytest.mark.asyncio
async def test_device_route_reset_cancels_terminal_seal_without_promotion() -> None:
    coordinator, scheduler, effects = await _coordinator()
    effects.drain_future = asyncio.get_running_loop().create_future()
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, 1_000_000_000))

        await coordinator.submit(DeviceRouteChanged(0, RouteKind.DUPLEX))
        await coordinator.flush()

        assert effects.promotions == []
        assert coordinator.snapshot.state is SpeculativeVoiceState.REBUILDING_AUDIO
        assert coordinator.snapshot.turn_id == turn_id
        assert epoch in effects.fenced_epochs
        assert effects.rebuilds == [0]
    finally:
        if effects.drain_future is not None and not effects.drain_future.done():
            effects.drain_future.cancel()
        await coordinator.close()


@pytest.mark.parametrize("failure", [CaptureDrainError("gap"), RuntimeError("ack")])
@pytest.mark.asyncio
async def test_terminal_drain_or_seal_failure_is_recoverable_draft(
    failure: BaseException,
) -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        if isinstance(failure, CaptureDrainError):
            effects.drain_error = failure
        else:
            effects.drain_receipt = DrainReceipt(1_000_000_001, 0, 0, 0, 0)
            effects.seal_error = failure
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, 1_000_000_000))
        await coordinator.flush()

        assert effects.promotions == []
        assert effects.drafts == [(turn_id, "hello world", "capture_sync_failed")]
        assert coordinator.snapshot.speculation_suspended is True
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_capture_sync_failure_rebuilds_before_voice_dispatch_recovers() -> None:
    coordinator, scheduler, effects = await _coordinator()
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        effects.drain_error = CaptureDrainError("gap")
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, 1_000_000_000))
        await coordinator.flush()

        assert coordinator.snapshot.state is SpeculativeVoiceState.REBUILDING_AUDIO
        assert coordinator.snapshot.admission_open is False
        assert effects.rebuilds == [0]
        assert effects.drafts == [(turn_id, "hello world", "capture_sync_failed")]

        effects.drain_error = None
        await coordinator.submit(
            AudioRouteReady(
                1,
                1,
                DuplexMode.FULL_DUPLEX,
                AecHealth.HEALTHY,
                AcousticSafetyPath.AEC,
            )
        )
        scheduler.advance_ms(699)
        await coordinator.flush()
        assert len(effects.dispatches) == 1
        scheduler.advance_ms(1)
        await coordinator.flush()
        assert effects.dispatches[-1][0] == turn_id
        assert effects.dispatches[-1][2] == "hello world"
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_half_duplex_gated_capture_cannot_extend_but_manual_interrupt_can() -> (
    None
):
    coordinator, scheduler, effects = await _coordinator(mode=DuplexMode.HALF_DUPLEX)
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        await coordinator.submit(AttemptPlaybackStarted(epoch))
        await coordinator.submit(CaptureGated(1, 20_000_000, 30_000_000, 0))
        assert coordinator.snapshot.current_attempt_epoch == epoch
        assert coordinator.snapshot.turn_id == turn_id

        await coordinator.submit(ManualInterruption())
        assert coordinator.snapshot.current_attempt_epoch is None
        assert coordinator.snapshot.turn_id == turn_id
        assert epoch in effects.fenced_epochs
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_half_duplex_render_pause_speech_interrupts_current_attempt() -> None:
    coordinator, scheduler, effects = await _coordinator(mode=DuplexMode.HALF_DUPLEX)
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        await coordinator.submit(AttemptPlaybackStarted(epoch))

        await coordinator.submit(
            _speech(
                1,
                started_ns=20_000_000,
                assistant_rendering=False,
            )
        )

        assert coordinator.snapshot.current_attempt_epoch is None
        assert coordinator.snapshot.turn_id == turn_id
        assert epoch in effects.fenced_epochs
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_half_duplex_terminal_skips_capture_drain_but_seals_transcript() -> None:
    coordinator, scheduler, effects = await _coordinator(mode=DuplexMode.HALF_DUPLEX)
    try:
        turn_id, epoch = await _start_attempt(coordinator, scheduler)
        await coordinator.submit(AttemptGenerationCompleted(epoch, "answer"))
        await coordinator.submit(AttemptPlaybackTerminal(epoch, 700_000_000))
        await coordinator.flush()

        assert effects.drain_calls == []
        assert effects.classification_drain_calls == [(700_000_000, 0)]
        assert effects.seal_calls == [0]
        assert effects.promotions == [
            (turn_id, epoch, "hello world", "answer", 700_000_000)
        ]
    finally:
        await coordinator.close()
