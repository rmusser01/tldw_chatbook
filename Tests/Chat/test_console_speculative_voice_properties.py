from __future__ import annotations

import asyncio
from dataclasses import dataclass
import heapq
from typing import Any, Callable

from hypothesis import given, settings
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
)
from hypothesis.strategies import sampled_from

from tldw_chatbook.Audio.duplex_contracts import DrainReceipt, DuplexMode
from tldw_chatbook.Audio.rolling_transcript import TranscriptRevision
from tldw_chatbook.Chat.console_speculative_voice import (
    AdmittedSpeechFrame,
    AttemptGenerationCompleted,
    AttemptOutputDelta,
    AttemptPlaybackTerminal,
    AttemptTtsFailed,
    ControlAction,
    ControlKind,
    SpeculativeTurnCoordinator,
    SpeculativeVoiceState,
)
from tldw_chatbook.Chat.console_voice_attempts import ProviderAttemptFailed
from tldw_chatbook.Chat.console_voice_attempts import AttemptCleanupOutcome


@dataclass(slots=True)
class _Handle:
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
        self._items: list[tuple[int, int, _Handle]] = []

    def call_at_ns(self, deadline_ns: int, callback: Callable[[], None]) -> _Handle:
        handle = _Handle(deadline_ns, self._order, callback)
        self._order += 1
        heapq.heappush(self._items, (deadline_ns, handle.order, handle))
        return handle

    def advance_ms(self, milliseconds: int) -> None:
        self.now_ns += milliseconds * 1_000_000
        while self._items and self._items[0][0] <= self.now_ns:
            _, _, handle = heapq.heappop(self._items)
            if not handle.cancelled:
                handle.callback()


class _Effects:
    def __init__(self) -> None:
        self.active_epochs: set[int] = set()
        self.fenced_epochs: set[int] = set()
        self.previews: list[tuple[int, str]] = []
        self.promotions: list[tuple[int, str, str]] = []
        self.drafts: list[str] = []
        self.forbidden_promotion_epochs: set[int] = set()
        self.pending_drain: asyncio.Future[DrainReceipt] | None = None
        self.classification_drains: list[tuple[int, int]] = []
        self.last_turn_id = ""
        self.last_transcript = ""

    def dispatch_attempt(
        self,
        *,
        turn_id: str,
        attempt_epoch: int,
        transcript: str,
    ) -> None:
        assert self.active_epochs == set()
        assert attempt_epoch not in self.fenced_epochs
        self.active_epochs.add(attempt_epoch)
        self.last_turn_id = turn_id
        self.last_transcript = transcript

    def fence_attempt(self, attempt_epoch: int) -> None:
        self.active_epochs.discard(attempt_epoch)
        self.fenced_epochs.add(attempt_epoch)

    def cancel_attempt(
        self, attempt_epoch: int
    ) -> asyncio.Future[AttemptCleanupOutcome]:
        del attempt_epoch
        completion = asyncio.get_running_loop().create_future()
        completion.set_result(AttemptCleanupOutcome.CLEAN)
        return completion

    def abort_output(self, attempt_epoch: int) -> None:
        del attempt_epoch

    def clear_preview(self, attempt_epoch: int) -> None:
        del attempt_epoch

    def publish_preview(self, attempt_epoch: int, delta: str) -> None:
        assert attempt_epoch in self.active_epochs
        assert attempt_epoch not in self.fenced_epochs
        self.previews.append((attempt_epoch, delta))

    def preserve_draft(
        self,
        *,
        turn_id: str,
        transcript: str,
        reason: str,
    ) -> None:
        del turn_id, transcript
        self.drafts.append(reason)

    def promote(
        self,
        *,
        turn_id: str,
        attempt_epoch: int,
        transcript: str,
        assistant_text: str,
        terminal_boundary_ns: int | None,
    ) -> None:
        del turn_id, terminal_boundary_ns
        assert attempt_epoch in self.active_epochs
        assert attempt_epoch not in self.fenced_epochs
        assert attempt_epoch not in self.forbidden_promotion_epochs
        self.active_epochs.discard(attempt_epoch)
        self.promotions.append((attempt_epoch, transcript, assistant_text))

    def classify_spoken_command(self, transcript: str) -> bool:
        del transcript
        return False

    def handle_spoken_command(self, *, turn_id: str, transcript: str) -> None:
        raise AssertionError((turn_id, transcript))

    def rebuild_audio(self, old_clock_generation: int, rebuild_epoch: int) -> None:
        del old_clock_generation, rebuild_epoch

    async def drain_capture_through(self, render_boundary_ns: int) -> DrainReceipt:
        if self.pending_drain is None:
            return DrainReceipt(render_boundary_ns + 1, 0, 0, 0, 0)
        return await self.pending_drain

    async def drain_pending_classification_through(
        self, render_boundary_ns: int, clock_generation: int
    ) -> None:
        self.classification_drains.append((render_boundary_ns, clock_generation))

    async def seal_transcript_through(
        self, admitted_sequence: int
    ) -> TranscriptRevision:
        del admitted_sequence
        assert self.classification_drains
        return TranscriptRevision(
            turn_id=self.last_turn_id,
            revision_id=1_000_000,
            stable_text=self.last_transcript,
            revisable_text="",
            covered_through_ns=10**18,
            mode="live",
            is_final=True,
        )

    def voice_dispatch_quarantined(self) -> bool:
        return False


class SpeculativeCoordinatorStateMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.scheduler = _Scheduler()
        self.effects = _Effects()
        self.coordinator = SpeculativeTurnCoordinator(
            effects=self.effects,
            scheduler=self.scheduler,
            response_eagerness_ms=500,
            initial_clock_generation=0,
            initial_duplex_mode=DuplexMode.FULL_DUPLEX,
        )
        self._run(self.coordinator.start())
        self.sequence = -1
        self.revision_by_turn: dict[str, int] = {}
        self.word = 0

    def _run(self, awaitable: Any) -> Any:
        return self.loop.run_until_complete(awaitable)

    @rule()
    def admit_speech(self) -> None:
        snapshot = self.coordinator.snapshot
        started_ns = self.scheduler.now_ns
        if snapshot.terminal_boundary_ns is not None:
            started_ns = snapshot.terminal_boundary_ns - 1
            if snapshot.current_attempt_epoch is not None:
                self.effects.forbidden_promotion_epochs.add(
                    snapshot.current_attempt_epoch
                )
        self.sequence += 1
        self._run(
            self.coordinator.submit(
                AdmittedSpeechFrame(
                    sequence=self.sequence,
                    started_ns=max(0, started_ns),
                    ended_ns=max(0, started_ns) + 10_000_000,
                    clock_generation=0,
                )
            )
        )

    @precondition(lambda self: self.coordinator.snapshot.turn_id is not None)
    @rule(presentation_only=sampled_from([False, True]))
    def revise(self, presentation_only: bool) -> None:
        snapshot = self.coordinator.snapshot
        turn_id = snapshot.turn_id
        assert turn_id is not None
        revision_id = self.revision_by_turn.get(turn_id, 0) + 1
        self.revision_by_turn[turn_id] = revision_id
        if presentation_only and snapshot.transcript_text:
            text = f"  {snapshot.transcript_text.upper()}! "
        else:
            self.word += 1
            text = f"voice word {self.word}"
        coverage = snapshot.last_speech_end_ns or 0
        self._run(
            self.coordinator.submit(
                TranscriptRevision(
                    turn_id=turn_id,
                    revision_id=revision_id,
                    stable_text="",
                    revisable_text=text,
                    covered_through_ns=coverage,
                    mode="live",
                )
            )
        )

    @rule(milliseconds=sampled_from([1, 119, 120, 250, 499, 500, 700, 1_500, 2_000]))
    def advance(self, milliseconds: int) -> None:
        self.scheduler.advance_ms(milliseconds)
        self._run(self.coordinator.flush())

    @precondition(
        lambda self: self.coordinator.snapshot.current_attempt_epoch is not None
    )
    @rule()
    def current_delta(self) -> None:
        epoch = self.coordinator.snapshot.current_attempt_epoch
        assert epoch is not None
        self._run(self.coordinator.submit(AttemptOutputDelta(epoch, "current")))

    @rule()
    def stale_delta(self) -> None:
        current = self.coordinator.snapshot.current_attempt_epoch
        stale = 0 if current is None else max(0, current - 1)
        self._run(self.coordinator.submit(AttemptOutputDelta(stale, "stale")))

    @precondition(
        lambda self: self.coordinator.snapshot.current_attempt_epoch is not None
    )
    @rule()
    def provider_failure(self) -> None:
        epoch = self.coordinator.snapshot.current_attempt_epoch
        assert epoch is not None
        self._run(
            self.coordinator.submit(ProviderAttemptFailed(epoch, "ProviderError"))
        )

    @precondition(
        lambda self: self.coordinator.snapshot.current_attempt_epoch is not None
    )
    @rule()
    def text_only_completion(self) -> None:
        epoch = self.coordinator.snapshot.current_attempt_epoch
        assert epoch is not None
        self._run(self.coordinator.submit(AttemptTtsFailed(epoch, "speech_failed")))
        self._run(
            self.coordinator.submit(
                AttemptGenerationCompleted(epoch, "complete current answer")
            )
        )

    @precondition(
        lambda self: (
            self.coordinator.snapshot.current_attempt_epoch is not None
            and self.coordinator.snapshot.state is not SpeculativeVoiceState.SEALING
        )
    )
    @rule()
    def start_terminal_seal(self) -> None:
        epoch = self.coordinator.snapshot.current_attempt_epoch
        assert epoch is not None
        self.effects.pending_drain = self.loop.create_future()
        self._run(
            self.coordinator.submit(AttemptGenerationCompleted(epoch, "audible answer"))
        )
        self._run(
            self.coordinator.submit(
                AttemptPlaybackTerminal(epoch, self.scheduler.now_ns + 1_000_000)
            )
        )

    @precondition(lambda self: self.effects.pending_drain is not None)
    @rule()
    def finish_terminal_seal(self) -> None:
        pending = self.effects.pending_drain
        assert pending is not None
        if not pending.done():
            boundary = self.coordinator.snapshot.terminal_boundary_ns
            pending.set_result(
                DrainReceipt(
                    (boundary or 0) + 1, self.sequence, self.sequence, self.sequence, 0
                )
            )
        self.effects.pending_drain = None
        self._run(self.coordinator.flush())

    @rule()
    def explicit_stop(self) -> None:
        self._run(self.coordinator.submit(ControlAction(ControlKind.STOP)))

    @invariant()
    def at_most_one_current_epoch_and_no_stale_output(self) -> None:
        assert len(self.effects.active_epochs) <= 1
        current = self.coordinator.snapshot.current_attempt_epoch
        if current is None:
            assert self.effects.active_epochs == set()
        else:
            assert self.effects.active_epochs == {current}

    @invariant()
    def no_cancelled_or_pre_boundary_content_is_promoted(self) -> None:
        assert all(
            epoch not in self.effects.fenced_epochs
            and epoch not in self.effects.forbidden_promotion_epochs
            for epoch, _, _ in self.effects.promotions
        )

    def teardown(self) -> None:
        pending = self.effects.pending_drain
        if pending is not None and not pending.done():
            pending.cancel()
        self._run(self.coordinator.close())
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        self.loop.close()
        asyncio.set_event_loop(None)


SpeculativeCoordinatorStateMachine.TestCase.settings = settings(
    max_examples=40,
    stateful_step_count=30,
    deadline=None,
)
TestSpeculativeCoordinatorStateMachine = SpeculativeCoordinatorStateMachine.TestCase


@settings(max_examples=3, deadline=None)
@given(style=sampled_from(["case", "punctuation", "whitespace"]))
def test_cosmetic_terminal_revision_preserves_exact_attempt_snapshot(
    style: str,
) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    scheduler = _Scheduler()
    effects = _Effects()
    coordinator = SpeculativeTurnCoordinator(
        effects=effects,
        scheduler=scheduler,
        response_eagerness_ms=500,
        initial_clock_generation=0,
        initial_duplex_mode=DuplexMode.FULL_DUPLEX,
    )
    try:
        loop.run_until_complete(coordinator.start())
        loop.run_until_complete(
            coordinator.submit(
                AdmittedSpeechFrame(
                    sequence=0,
                    started_ns=0,
                    ended_ns=10_000_000,
                    clock_generation=0,
                )
            )
        )
        turn_id = coordinator.snapshot.turn_id
        assert turn_id is not None
        winning_prompt = "Winning Prompt"
        loop.run_until_complete(
            coordinator.submit(
                TranscriptRevision(
                    turn_id=turn_id,
                    revision_id=1,
                    stable_text="",
                    revisable_text=winning_prompt,
                    covered_through_ns=10_000_000,
                    mode="live",
                )
            )
        )
        scheduler.advance_ms(500)
        loop.run_until_complete(coordinator.flush())
        epoch = coordinator.snapshot.current_attempt_epoch
        assert epoch is not None

        if style == "case":
            effects.last_transcript = winning_prompt.swapcase()
        elif style == "punctuation":
            effects.last_transcript = f"{winning_prompt},!"
        else:
            effects.last_transcript = f"  {winning_prompt} \n"
        loop.run_until_complete(
            coordinator.submit(AttemptGenerationCompleted(epoch, "audible answer"))
        )
        loop.run_until_complete(
            coordinator.submit(AttemptPlaybackTerminal(epoch, 1_000_000_000))
        )
        loop.run_until_complete(coordinator.flush())

        assert effects.classification_drains == [(1_000_000_000, 0)]
        assert effects.promotions == [(epoch, winning_prompt, "audible answer")]
    finally:
        loop.run_until_complete(coordinator.close())
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        asyncio.set_event_loop(None)
