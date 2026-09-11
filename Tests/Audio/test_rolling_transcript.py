"""Revision and bounded rolling-window contracts for speculative STT."""

from __future__ import annotations

import asyncio
import gc
import threading
import weakref
from dataclasses import fields, replace

import pytest

import tldw_chatbook.Audio.rolling_transcript as rolling_transcript_module
from tldw_chatbook.Audio.duplex_contracts import AudioFrame
from tldw_chatbook.Audio.rolling_transcript import (
    TranscriptBackendFailure,
    TranscriptEngine,
    TranscriptHypothesis,
    TranscriptTiming,
    TranscriptToken,
    is_material_transcript_change,
)

pytestmark = pytest.mark.unit


def _frame(sequence: int, *, generation: int = 3) -> AudioFrame:
    started_ns = sequence * 10_000_000
    return AudioFrame(
        sequence=sequence,
        started_ns=started_ns,
        ended_ns=started_ns + 10_000_000,
        pcm16=bytes([sequence % 251]) * 960,
        clock_generation=generation,
    )


def _hypothesis(
    stable: str,
    tail: str,
    coverage_ns: int,
    *,
    final: bool = False,
) -> TranscriptHypothesis:
    return TranscriptHypothesis(
        stable_text=stable,
        revisable_text=tail,
        covered_through_ns=coverage_ns,
        is_final=final,
        provider_latency_ms=7,
        usage_units=2,
    )


class _Live:
    def __init__(self) -> None:
        self.frames: list[AudioFrame] = []
        self.started = 0
        self.close_calls = 0
        self._publish = None
        self._fail = None
        self._settle = None
        self._attempts = []
        self.submissions_by_attempt: list[list[AudioFrame]] = []

    def start(self, publish, fail, settle=None) -> None:
        self.started += 1
        self._publish = publish
        self._fail = fail
        self._settle = settle
        self._attempts.append((publish, fail, settle))
        self.submissions_by_attempt.append([])

    def submit(self, frame: AudioFrame) -> None:
        self.frames.append(frame)
        self.submissions_by_attempt[-1].append(frame)

    def publish(self, hypothesis: TranscriptHypothesis, *, attempt: int = -1) -> None:
        publish = self._attempts[attempt][0]
        publish(hypothesis)

    def fail(self, *, attempt: int = -1) -> None:
        fail = self._attempts[attempt][1]
        fail(RuntimeError("live failed"))

    def settle(self, sequence: int, *, attempt: int = -1) -> None:
        settle = self._attempts[attempt][2]
        assert settle is not None, "engine did not install a live settlement callback"
        settle(sequence)

    async def close(self) -> None:
        self.close_calls += 1


class _SlowRolling:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, int, int]] = []
        self.concurrent = 0
        self.max_concurrent = 0
        self._releases: asyncio.Queue[TranscriptHypothesis | BaseException] = (
            asyncio.Queue()
        )

    async def transcribe_window(
        self, *, pcm16: bytes, started_ns: int, ended_ns: int
    ) -> TranscriptHypothesis:
        self.calls.append((pcm16, started_ns, ended_ns))
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            outcome = await self._releases.get()
        finally:
            self.concurrent -= 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def release(self, outcome: TranscriptHypothesis | BaseException) -> None:
        self._releases.put_nowait(outcome)


class _LifecycleRolling(_SlowRolling):
    def __init__(self, *, abort_mode: str = "return") -> None:
        super().__init__()
        self.abort_mode = abort_mode
        self.transcribe_started = asyncio.Event()
        self.abort_started = asyncio.Event()
        self.abort_released = asyncio.Event()
        self.abort_completed = asyncio.Event()
        self.abort_calls = 0

    async def transcribe_window(
        self, *, pcm16: bytes, started_ns: int, ended_ns: int
    ) -> TranscriptHypothesis:
        self.transcribe_started.set()
        return await super().transcribe_window(
            pcm16=pcm16, started_ns=started_ns, ended_ns=ended_ns
        )

    async def abort(self) -> None:
        self.abort_calls += 1
        self.abort_started.set()
        try:
            if self.abort_mode == "raise":
                raise RuntimeError("abort failed")
            if self.abort_mode == "block":
                while not self.abort_released.is_set():
                    try:
                        await self.abort_released.wait()
                    except asyncio.CancelledError:
                        continue
        finally:
            self.abort_completed.set()


class _DetachedAbortState:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.completed = asyncio.Event()
        self.calls = 0


class _DetachedAbortRolling(_SlowRolling):
    def __init__(
        self, cleanup: asyncio.Future[None], state: _DetachedAbortState
    ) -> None:
        super().__init__()
        self._cleanup = cleanup
        self._abort_state = state

    async def abort(self) -> None:
        cleanup = self._cleanup
        state = self._abort_state
        state.calls += 1
        state.started.set()
        del self
        try:
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
        finally:
            state.completed.set()


class _SecretEqualRolling(_SlowRolling):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _SecretEqualRolling)

    def __repr__(self) -> str:
        return "SECRET_ADAPTER_REPR"


class _CancellationResistantRolling:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.released = asyncio.Event()
        self.abort_calls = 0

    async def transcribe_window(
        self, *, pcm16: bytes, started_ns: int, ended_ns: int
    ) -> TranscriptHypothesis:
        self.started.set()
        while not self.released.is_set():
            try:
                await self.released.wait()
            except asyncio.CancelledError:
                continue
        return _hypothesis("", "closed", ended_ns)

    async def abort(self) -> None:
        self.abort_calls += 1
        self.released.set()


class _SlowCooperativeAbortRolling:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.released = asyncio.Event()
        self.abort_completed = False

    async def transcribe_window(
        self, *, pcm16: bytes, started_ns: int, ended_ns: int
    ) -> TranscriptHypothesis:
        self.started.set()
        while not self.released.is_set():
            try:
                await self.released.wait()
            except asyncio.CancelledError:
                continue
        return _hypothesis("", "closed", ended_ns)

    async def abort(self) -> None:
        try:
            await asyncio.sleep(0.03)
            self.abort_completed = True
        finally:
            self.released.set()


class _UnresponsiveRolling:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.abort_started = asyncio.Event()
        self.released = asyncio.Event()

    async def transcribe_window(
        self, *, pcm16: bytes, started_ns: int, ended_ns: int
    ) -> TranscriptHypothesis:
        self.started.set()
        while not self.released.is_set():
            try:
                await self.released.wait()
            except asyncio.CancelledError:
                continue
        return _hypothesis("", "released", ended_ns)

    async def abort(self) -> None:
        self.abort_started.set()
        while not self.released.is_set():
            try:
                await self.released.wait()
            except asyncio.CancelledError:
                continue


class _ThrowingCloseLive(_Live):
    async def close(self) -> None:
        self.close_calls += 1
        raise RuntimeError("native close failed")


def test_revision_values_validate_monotonic_coverage_and_content_free_timing():
    assert [field.name for field in fields(TranscriptTiming)] == [
        "capture_duration_ms",
        "provider_latency_ms",
        "processed_duration_ms",
        "duplicated_duration_ms",
        "usage_units",
    ]
    with pytest.raises(ValueError, match="non-negative"):
        TranscriptTiming(provider_latency_ms=-1)

    async def scenario() -> None:
        live = _Live()
        revisions = []
        engine = TranscriptEngine(
            turn_id="turn-a", live_adapter=live, on_revision=revisions.append
        )
        engine.append_admitted_frame(_frame(0))
        live.publish(_hypothesis("Hello ", "wor", 10_000_000))
        await asyncio.sleep(0)
        engine.append_admitted_frame(_frame(1))
        live.publish(_hypothesis("Hello ", "world!", 20_000_000, final=True))
        await asyncio.sleep(0)

        assert [revision.revision_id for revision in revisions] == [1, 2]
        assert revisions[0].turn_id == "turn-a"
        assert revisions[1].stable_text == "Hello "
        assert revisions[1].revisable_text == "world!"
        assert revisions[1].timing == TranscriptTiming(
            capture_duration_ms=20,
            provider_latency_ms=7,
            processed_duration_ms=20,
            duplicated_duration_ms=0,
            usage_units=2,
        )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("Hello, WORLD!", " hello world "),
        ("wait... what?", "WAIT what"),
        ("two\nspaces", "two    spaces"),
        ("don't", "dont"),
        ("re-enter", "reenter"),
        ("don’t", "dont"),
        ("café", "cafe\N{COMBINING ACUTE ACCENT}"),
    ],
)
def test_punctuation_case_and_whitespace_are_not_material(before: str, after: str):
    assert is_material_transcript_change(before, after) is False


def test_word_edit_is_material():
    assert is_material_transcript_change("send it now", "end it now") is True


def test_hypothesis_rejects_any_future_or_overlapping_token_not_only_the_last():
    with pytest.raises(ValueError, match="coverage"):
        TranscriptHypothesis(
            tokens=(
                TranscriptToken("future ", 0, 1_000_000_000),
                TranscriptToken("ok", 10_000_000, 20_000_000),
            ),
            covered_through_ns=20_000_000,
        )
    with pytest.raises(ValueError, match="overlap"):
        TranscriptHypothesis(
            tokens=(
                TranscriptToken("first ", 0, 15_000_000),
                TranscriptToken("second", 10_000_000, 20_000_000),
            ),
            covered_through_ns=20_000_000,
        )


def test_native_live_is_preferred_and_freshness_uses_frame_tolerance():
    async def scenario() -> None:
        live = _Live()
        rolling = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-live",
            live_adapter=live,
            rolling_adapter=rolling,
            frame_tolerance_ns=10_000_000,
        )

        engine.append_admitted_frame(_frame(0))
        live.publish(_hypothesis("", "hello", 10_000_000))
        await asyncio.sleep(0)

        assert live.frames == [_frame(0)]
        assert rolling.calls == []
        assert engine.fresh_for(20_000_000) is True
        assert engine.fresh_for(20_000_001) is False

    asyncio.run(scenario())


def test_seal_waits_for_revision_covering_the_admitted_sequence():
    async def scenario() -> None:
        live = _Live()
        engine = TranscriptEngine(turn_id="turn-seal", live_adapter=live)
        engine.append_admitted_frame(_frame(0))
        engine.append_admitted_frame(_frame(1))

        seal = asyncio.create_task(engine.seal_through(1))
        await asyncio.sleep(0)
        assert seal.done() is False

        live.publish(_hypothesis("hello ", "there", 20_000_000))
        await asyncio.sleep(0)
        assert seal.done() is False

        live.settle(1)
        revision = await asyncio.wait_for(seal, timeout=0.2)
        assert revision.covered_through_ns == 20_000_000

    asyncio.run(scenario())


def test_rolling_window_has_one_in_flight_newest_pending_and_reconciles_overlap():
    async def scenario() -> None:
        rolling = _SlowRolling()
        revisions = []
        engine = TranscriptEngine(
            turn_id="turn-roll",
            rolling_adapter=rolling,
            rolling_window_ns=30_000_000,
            max_buffered_frames=3,
            on_revision=revisions.append,
        )

        engine.append_admitted_frame(_frame(0))
        await asyncio.sleep(0)
        assert len(rolling.calls) == 1

        # All three requests arrive while seq0 is in flight. Only the newest
        # desired window may survive as pending work.
        for sequence in (1, 2, 3):
            engine.append_admitted_frame(_frame(sequence))
        await asyncio.sleep(0)
        assert len(rolling.calls) == 1

        rolling.release(
            TranscriptHypothesis(
                tokens=(TranscriptToken("Hello, ", 0, 10_000_000),),
                covered_through_ns=10_000_000,
                provider_latency_ms=4,
                usage_units=1,
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert len(rolling.calls) == 2
        pcm, started_ns, ended_ns = rolling.calls[1]
        assert (started_ns, ended_ns) == (10_000_000, 40_000_000)
        assert len(pcm) == 3 * 960
        rolling.release(
            TranscriptHypothesis(
                tokens=(
                    TranscriptToken("world ", 10_000_000, 30_000_000),
                    TranscriptToken("friend!", 30_000_000, 40_000_000),
                ),
                covered_through_ns=40_000_000,
                provider_latency_ms=5,
                usage_units=3,
            )
        )
        await engine.wait_idle()

        assert rolling.max_concurrent == 1
        assert revisions[-1].stable_text == "Hello, "
        assert revisions[-1].revisable_text == "world friend!"
        assert revisions[-1].timing.duplicated_duration_ms == 0
        assert revisions[-1].timing.processed_duration_ms == 40

    asyncio.run(scenario())


def test_rolling_window_coalesces_undersized_audio_before_provider_inference():
    class ImmediateRolling:
        def __init__(self) -> None:
            self.calls: list[tuple[bytes, int, int]] = []

        async def transcribe_window(
            self, *, pcm16: bytes, started_ns: int, ended_ns: int
        ) -> TranscriptHypothesis:
            self.calls.append((pcm16, started_ns, ended_ns))
            return _hypothesis("", "complete phrase", ended_ns)

    async def scenario() -> None:
        rolling = ImmediateRolling()
        engine = TranscriptEngine(
            turn_id="turn-coalesced",
            rolling_adapter=rolling,
            rolling_min_window_ns=500_000_000,
            rolling_debounce_seconds=0.01,
        )

        engine.append_admitted_frame(_frame(0))
        await asyncio.sleep(0)
        for sequence in range(1, 50):
            engine.append_admitted_frame(_frame(sequence))
        await engine.wait_idle()

        assert len(rolling.calls) == 1
        assert rolling.calls[0][1:] == (0, 500_000_000)
        assert len(rolling.calls[0][0]) == 50 * 960

    asyncio.run(scenario())


def test_rolling_window_flushes_short_utterance_after_quiet_edge():
    class ImmediateRolling:
        def __init__(self) -> None:
            self.calls: list[tuple[bytes, int, int]] = []

        async def transcribe_window(
            self, *, pcm16: bytes, started_ns: int, ended_ns: int
        ) -> TranscriptHypothesis:
            self.calls.append((pcm16, started_ns, ended_ns))
            return _hypothesis("", "short phrase", ended_ns)

    async def scenario() -> None:
        rolling = ImmediateRolling()
        engine = TranscriptEngine(
            turn_id="turn-short-quiet-flush",
            rolling_adapter=rolling,
            rolling_min_window_ns=500_000_000,
            rolling_debounce_seconds=0.01,
        )

        engine.append_admitted_frame(_frame(0))
        await asyncio.sleep(0)
        for sequence in range(1, 10):
            engine.append_admitted_frame(_frame(sequence))
        await engine.wait_idle()

        assert len(rolling.calls) == 1
        assert rolling.calls[0][1:] == (0, 100_000_000)
        assert engine.editable_draft == "short phrase"

    asyncio.run(scenario())


def test_rolling_overrun_does_not_publish_or_seal_across_evicted_unproven_audio():
    async def scenario() -> None:
        rolling = _SlowRolling()
        revisions = []
        engine = TranscriptEngine(
            turn_id="turn-ordinary-rolling-overrun",
            rolling_adapter=rolling,
            rolling_window_ns=30_000_000,
            max_buffered_frames=3,
            on_revision=revisions.append,
        )
        try:
            engine.append_admitted_frame(_frame(0))
            await asyncio.sleep(0)
            for sequence in range(1, 11):
                engine.append_admitted_frame(_frame(sequence))
            seal = asyncio.create_task(engine.seal_through(1))

            rolling.release(
                TranscriptHypothesis(
                    tokens=(TranscriptToken("first ", 0, 10_000_000),),
                    covered_through_ns=10_000_000,
                )
            )
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert rolling.calls[1][1:] == (80_000_000, 110_000_000)

            rolling.release(
                TranscriptHypothesis(
                    tokens=(TranscriptToken("late", 80_000_000, 110_000_000),),
                    covered_through_ns=110_000_000,
                )
            )
            await engine.wait_idle()

            assert [
                revision.covered_through_ns
                for revision in revisions
                if revision.failure_code is None
            ] == [10_000_000]
            assert engine.latest_revision is not None
            assert engine.latest_revision.failure_code == "backend_failed"
            assert engine.latest_revision.covered_through_ns == 10_000_000
            with pytest.raises(TranscriptBackendFailure, match="backend_failed"):
                await asyncio.wait_for(seal, timeout=0.2)
        finally:
            await engine.close()

    asyncio.run(scenario())


def test_rolling_overlap_is_content_free_accounted_and_newest_display_wins():
    async def scenario() -> None:
        rolling = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-overlap",
            rolling_adapter=rolling,
            rolling_window_ns=30_000_000,
            max_buffered_frames=3,
        )
        for sequence in range(3):
            engine.append_admitted_frame(_frame(sequence))
        await asyncio.sleep(0)
        rolling.release(
            TranscriptHypothesis(
                tokens=(
                    TranscriptToken("HELLO ", 0, 10_000_000),
                    TranscriptToken("world", 10_000_000, 30_000_000),
                ),
                covered_through_ns=30_000_000,
            )
        )
        await engine.wait_idle()

        engine.append_admitted_frame(_frame(3))
        await asyncio.sleep(0)
        rolling.release(
            TranscriptHypothesis(
                tokens=(
                    TranscriptToken("world, ", 10_000_000, 30_000_000),
                    TranscriptToken("friend", 30_000_000, 40_000_000),
                ),
                covered_through_ns=40_000_000,
            )
        )
        await engine.wait_idle()

        revision = engine.latest_revision
        assert revision is not None
        assert revision.stable_text == "HELLO "
        assert revision.revisable_text == "world, friend"
        assert revision.timing.duplicated_duration_ms == 20
        assert revision.timing.processed_duration_ms == 60

    asyncio.run(scenario())


def test_one_fallback_then_terminal_failure_retains_draft_and_suspends_dispatch():
    async def scenario() -> None:
        live = _Live()
        fallback = _SlowRolling()
        revisions = []
        engine = TranscriptEngine(
            turn_id="turn-fail",
            live_adapter=live,
            fallback_adapter=fallback,
            on_revision=revisions.append,
        )
        engine.append_admitted_frame(_frame(0))
        live.publish(_hypothesis("keep ", "this draft", 10_000_000))

        live.fail()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(fallback.calls) == 1
        fallback.release(RuntimeError("fallback failed"))
        await engine.wait_idle()

        assert engine.editable_draft == "keep this draft"
        assert engine.automatic_dispatch_suspended is True
        assert engine.status == "failed"
        assert revisions[-1].failure_code == "fallback_failed"
        assert revisions[-1].is_final is True
        with pytest.raises(TranscriptBackendFailure, match="fallback_failed"):
            await engine.seal_through(0)

        # The same captured tail cannot trigger unbounded retries.
        engine.append_admitted_frame(_frame(0))
        await asyncio.sleep(0)
        assert len(fallback.calls) == 1

        # Fresh admitted speech is an explicit new attempt boundary.
        engine.append_admitted_frame(_frame(1))
        await asyncio.sleep(0)
        assert engine.automatic_dispatch_suspended is False
        assert len(fallback.calls) == 2
        fallback.release(_hypothesis("keep ", "this draft now", 20_000_000))
        await engine.wait_idle()

        engine.manual_retry()
        assert engine.automatic_dispatch_suspended is False

    asyncio.run(scenario())


@pytest.mark.parametrize("primary_abort_mode", ["return", "raise", "block"])
def test_failed_primary_rolling_is_retired_once_before_fallback_owns_work(
    monkeypatch, primary_abort_mode
):
    async def scenario() -> None:
        monkeypatch.setattr(
            rolling_transcript_module, "_ADAPTER_SHUTDOWN_TIMEOUT_SECONDS", 0.01
        )
        primary = _LifecycleRolling(abort_mode=primary_abort_mode)
        fallback = _LifecycleRolling()
        engine = TranscriptEngine(
            turn_id=f"turn-primary-retirement-{primary_abort_mode}",
            rolling_adapter=primary,
            fallback_adapter=fallback,
        )
        try:
            engine.append_admitted_frame(_frame(0))
            await asyncio.wait_for(primary.transcribe_started.wait(), timeout=0.2)
            primary.release(RuntimeError("primary failed"))

            await asyncio.wait_for(fallback.transcribe_started.wait(), timeout=0.2)
            assert primary.abort_started.is_set()
            assert primary.abort_calls == 1
            fallback.release(_hypothesis("", "fallback succeeded", 10_000_000))
            await engine.wait_idle()

            if primary_abort_mode == "block":
                assert engine.supervised_shutdown_task_count == 1
                primary.abort_released.set()
                await asyncio.wait_for(primary.abort_completed.wait(), timeout=0.2)
                await asyncio.sleep(0)
                assert engine.supervised_shutdown_task_count == 0

            await engine.close()
            assert primary.abort_calls == 1
            assert fallback.abort_calls == 1
        finally:
            primary.abort_released.set()
            fallback.abort_released.set()
            await engine.close()

    asyncio.run(scenario())


def test_concurrent_close_strongly_retains_primary_until_abort_task_completes(
    monkeypatch,
):
    async def scenario() -> None:
        monkeypatch.setattr(
            rolling_transcript_module, "_ADAPTER_SHUTDOWN_TIMEOUT_SECONDS", 0.01
        )
        cleanup = asyncio.get_running_loop().create_future()
        state = _DetachedAbortState()
        primary = _DetachedAbortRolling(cleanup, state)
        primary_ref = weakref.ref(primary)
        fallback = _LifecycleRolling()
        engine = TranscriptEngine(
            turn_id="turn-retirement-close-race",
            rolling_adapter=primary,
            fallback_adapter=fallback,
        )
        try:
            engine.append_admitted_frame(_frame(0))
            await asyncio.sleep(0)
            primary.release(RuntimeError("primary failed"))
            await asyncio.wait_for(state.started.wait(), timeout=0.2)

            await asyncio.wait_for(engine.close(), timeout=0.2)
            assert fallback.abort_calls == 1
            del primary
            gc.collect()

            assert primary_ref() is not None
            assert any(
                adapter is primary_ref()
                for adapter in engine._retiring_rolling_adapters.values()
            )

            cleanup.set_result(None)
            await asyncio.wait_for(state.completed.wait(), timeout=0.2)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            gc.collect()

            assert engine._retiring_rolling_adapters == {}
            assert primary_ref() is None
            assert state.calls == 1
            assert fallback.abort_calls == 1
        finally:
            if not cleanup.done():
                cleanup.set_result(None)
            await state.completed.wait()
            await engine.close()

    asyncio.run(scenario())


def test_backend_failure_without_fallback_can_be_retried_manually_once_requested():
    async def scenario() -> None:
        rolling = _SlowRolling()
        engine = TranscriptEngine(turn_id="turn-manual", rolling_adapter=rolling)
        engine.append_admitted_frame(_frame(0))
        await asyncio.sleep(0)
        rolling.release(RuntimeError("primary failed"))
        await engine.wait_idle()

        assert engine.latest_revision is not None
        assert engine.latest_revision.failure_code == "backend_failed"
        assert engine.automatic_dispatch_suspended is True

        engine.manual_retry()
        await asyncio.sleep(0)
        assert len(rolling.calls) == 2
        rolling.release(_hypothesis("", "retry worked", 10_000_000))
        await engine.wait_idle()

        assert engine.automatic_dispatch_suspended is False
        assert engine.editable_draft == "retry worked"

    asyncio.run(scenario())


def test_fresh_admitted_speech_restarts_a_failed_native_live_adapter():
    async def scenario() -> None:
        live = _Live()
        engine = TranscriptEngine(turn_id="turn-live-retry", live_adapter=live)
        engine.append_admitted_frame(_frame(0))
        live.publish(_hypothesis("", "editable draft", 10_000_000))
        await asyncio.sleep(0)
        live.fail()
        await asyncio.sleep(0)

        assert engine.automatic_dispatch_suspended is True
        assert live.started == 1

        engine.append_admitted_frame(_frame(1))

        assert engine.automatic_dispatch_suspended is False
        assert live.started == 2
        assert live.frames[-1] == _frame(1)

    asyncio.run(scenario())


def test_rolling_seal_waits_for_overlapping_pending_work_to_settle():
    async def scenario() -> None:
        rolling = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-causal-seal",
            rolling_adapter=rolling,
            rolling_window_ns=30_000_000,
        )
        engine.append_admitted_frame(_frame(0))
        await asyncio.sleep(0)
        engine.append_admitted_frame(_frame(1))
        seal = asyncio.create_task(engine.seal_through(0))
        await asyncio.sleep(0)

        rolling.release(
            TranscriptHypothesis(
                tokens=(TranscriptToken("hello ", 0, 10_000_000),),
                covered_through_ns=10_000_000,
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert len(rolling.calls) == 2
        assert seal.done() is False

        rolling.release(
            TranscriptHypothesis(
                tokens=(
                    TranscriptToken("hullo ", 0, 10_000_000),
                    TranscriptToken("world", 10_000_000, 20_000_000),
                ),
                covered_through_ns=20_000_000,
            )
        )
        await engine.wait_idle()

        revision = await asyncio.wait_for(seal, timeout=0.2)
        assert revision.stable_text + revision.revisable_text == "hullo world"

    asyncio.run(scenario())


def test_stale_native_callbacks_are_ignored_after_fallback_epoch_begins():
    async def scenario() -> None:
        live = _Live()
        fallback = _SlowRolling()
        revisions = []
        engine = TranscriptEngine(
            turn_id="turn-epoch",
            live_adapter=live,
            fallback_adapter=fallback,
            on_revision=revisions.append,
        )
        engine.append_admitted_frame(_frame(0))
        live.publish(_hypothesis("", "current draft", 10_000_000), attempt=0)
        await asyncio.sleep(0)
        live.fail(attempt=0)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        revision_count = len(revisions)

        live.publish(_hypothesis("", "stale overwrite", 10_000_000), attempt=0)
        live.fail(attempt=0)
        await asyncio.sleep(0)

        assert len(fallback.calls) == 1
        assert len(revisions) == revision_count
        assert engine.status == "transcribing"

        fallback.release(
            TranscriptHypothesis(
                tokens=(TranscriptToken("fallback draft", 0, 10_000_000),),
                covered_through_ns=10_000_000,
            )
        )
        await engine.wait_idle()
        assert engine.editable_draft == "fallback draft"

    asyncio.run(scenario())


def test_native_stable_prefix_survives_timestamped_rolling_fallback():
    async def scenario() -> None:
        live = _Live()
        fallback = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-native-token-fallback",
            live_adapter=live,
            fallback_adapter=fallback,
        )
        engine.append_admitted_frame(_frame(0))
        live.publish(_hypothesis("hello ", "wor", 10_000_000))
        await asyncio.sleep(0)
        live.fail()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        fallback.release(
            TranscriptHypothesis(
                tokens=(
                    TranscriptToken("hello ", 0, 5_000_000),
                    TranscriptToken("world", 5_000_000, 10_000_000),
                ),
                covered_through_ns=10_000_000,
            )
        )
        await engine.wait_idle()

        assert engine.latest_revision is not None
        assert engine.latest_revision.failure_code is None
        assert engine.latest_revision.stable_text == "hello "
        assert engine.latest_revision.revisable_text == "world"
        assert engine.editable_draft == "hello world"

    asyncio.run(scenario())


def test_every_overlapping_fallback_window_removes_the_native_prefix_once():
    async def scenario() -> None:
        live = _Live()
        fallback = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-overlapping-token-fallback",
            live_adapter=live,
            fallback_adapter=fallback,
            rolling_window_ns=30_000_000,
        )
        engine.append_admitted_frame(_frame(0))
        live.publish(_hypothesis("hello ", "wor", 10_000_000))
        await asyncio.sleep(0)
        live.fail()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        engine.append_admitted_frame(_frame(1))
        fallback.release(
            TranscriptHypothesis(
                tokens=(
                    TranscriptToken("hello ", 0, 5_000_000),
                    TranscriptToken("world ", 5_000_000, 10_000_000),
                ),
                covered_through_ns=10_000_000,
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(fallback.calls) == 2
        fallback.release(
            TranscriptHypothesis(
                tokens=(
                    TranscriptToken("hello ", 0, 5_000_000),
                    TranscriptToken("world ", 5_000_000, 10_000_000),
                    TranscriptToken("again", 10_000_000, 20_000_000),
                ),
                covered_through_ns=20_000_000,
            )
        )
        await engine.wait_idle()

        assert engine.latest_revision is not None
        assert engine.latest_revision.failure_code is None
        assert engine.latest_revision.stable_text == "hello "
        assert engine.latest_revision.revisable_text == "world again"
        assert engine.editable_draft == "hello world again"

    asyncio.run(scenario())


def test_overlapping_fallback_prefix_mismatch_remains_fail_closed():
    async def scenario() -> None:
        live = _Live()
        fallback = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-fallback-prefix-mismatch",
            live_adapter=live,
            fallback_adapter=fallback,
        )
        engine.append_admitted_frame(_frame(0))
        live.publish(_hypothesis("hello ", "wor", 10_000_000))
        await asyncio.sleep(0)
        live.fail()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        fallback.release(
            TranscriptHypothesis(
                tokens=(TranscriptToken("hullo world", 0, 10_000_000),),
                covered_through_ns=10_000_000,
            )
        )
        await engine.wait_idle()

        assert engine.latest_revision is not None
        assert engine.latest_revision.failure_code == "fallback_failed"
        assert engine.editable_draft == "hello wor"

    asyncio.run(scenario())


def test_nonoverlapping_fallback_window_keeps_a_literal_matching_prefix_token():
    async def scenario() -> None:
        live = _Live()
        fallback = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-later-literal-prefix",
            live_adapter=live,
            fallback_adapter=fallback,
            rolling_window_ns=10_000_000,
        )
        engine.append_admitted_frame(_frame(0))
        live.publish(_hypothesis("hello ", "wor", 10_000_000))
        await asyncio.sleep(0)
        live.fail()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        fallback.release(
            TranscriptHypothesis(
                tokens=(
                    TranscriptToken("hello ", 0, 5_000_000),
                    TranscriptToken("world ", 5_000_000, 10_000_000),
                ),
                covered_through_ns=10_000_000,
            )
        )
        await engine.wait_idle()

        engine.append_admitted_frame(_frame(1))
        await asyncio.sleep(0)
        fallback.release(
            TranscriptHypothesis(
                tokens=(
                    TranscriptToken("hello ", 10_000_000, 15_000_000),
                    TranscriptToken("again", 15_000_000, 20_000_000),
                ),
                covered_through_ns=20_000_000,
            )
        )
        await engine.wait_idle()

        assert engine.latest_revision is not None
        assert engine.latest_revision.failure_code is None
        assert engine.editable_draft == "hello world hello again"

    asyncio.run(scenario())


def test_first_nonoverlapping_fallback_window_freezes_full_native_editable_base():
    async def scenario() -> None:
        live = _Live()
        fallback = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-full-native-fallback-base",
            live_adapter=live,
            fallback_adapter=fallback,
            rolling_window_ns=20_000_000,
            max_buffered_frames=2,
        )
        engine.append_admitted_frame(_frame(0))
        live.publish(_hypothesis("hello ", "world ", 10_000_000))
        await asyncio.sleep(0)
        engine.append_admitted_frame(_frame(1))
        engine.append_admitted_frame(_frame(2))
        live.fail()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert fallback.calls[0][1:] == (10_000_000, 30_000_000)
        fallback.release(
            TranscriptHypothesis(
                tokens=(TranscriptToken("again", 10_000_000, 30_000_000),),
                covered_through_ns=30_000_000,
            )
        )
        await engine.wait_idle()

        assert engine.latest_revision is not None
        assert engine.latest_revision.failure_code is None
        assert engine.latest_revision.stable_text == "hello world "
        assert engine.latest_revision.revisable_text == "again"
        assert engine.editable_draft == "hello world again"

    asyncio.run(scenario())


def test_fallback_fails_closed_when_evicted_admitted_audio_exceeds_native_base():
    async def scenario() -> None:
        live = _Live()
        fallback = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-evicted-native-fallback-gap",
            live_adapter=live,
            fallback_adapter=fallback,
            rolling_window_ns=20_000_000,
            max_buffered_frames=2,
        )
        engine.append_admitted_frame(_frame(0))
        live.publish(_hypothesis("hello ", "world", 10_000_000))
        await asyncio.sleep(0)
        for sequence in (1, 2, 3):
            engine.append_admitted_frame(_frame(sequence))
        live.fail()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert fallback.calls[0][1:] == (20_000_000, 40_000_000)
        fallback.release(
            TranscriptHypothesis(
                tokens=(TranscriptToken("again", 20_000_000, 40_000_000),),
                covered_through_ns=40_000_000,
            )
        )
        await engine.wait_idle()

        assert engine.latest_revision is not None
        assert engine.latest_revision.failure_code == "fallback_failed"
        assert engine.latest_revision.covered_through_ns == 10_000_000
        assert engine.editable_draft == "hello world"
        with pytest.raises(TranscriptBackendFailure):
            await engine.seal_through(1)

    asyncio.run(scenario())


def test_live_overclaimed_coverage_fails_closed_without_becoming_fresh():
    async def scenario() -> None:
        live = _Live()
        engine = TranscriptEngine(turn_id="turn-live-coverage", live_adapter=live)
        engine.append_admitted_frame(_frame(0))

        live.publish(_hypothesis("", "unearned", 20_000_000))
        await asyncio.sleep(0)

        assert engine.fresh_for(20_000_000) is False
        assert engine.latest_revision is not None
        assert engine.latest_revision.failure_code == "backend_failed"
        with pytest.raises(TranscriptBackendFailure):
            await engine.seal_through(0)

    asyncio.run(scenario())


def test_rolling_overclaimed_coverage_fails_closed_at_request_boundary():
    async def scenario() -> None:
        rolling = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-rolling-coverage", rolling_adapter=rolling
        )
        engine.append_admitted_frame(_frame(0))
        await asyncio.sleep(0)
        rolling.release(_hypothesis("", "unearned", 20_000_000))
        await engine.wait_idle()

        assert engine.fresh_for(20_000_000) is False
        assert engine.latest_revision is not None
        assert engine.latest_revision.failure_code == "backend_failed"
        with pytest.raises(TranscriptBackendFailure):
            await engine.seal_through(0)

    asyncio.run(scenario())


def test_completed_rolling_call_must_cover_its_full_request_or_fail_seal():
    async def scenario() -> None:
        rolling = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-rolling-undercoverage", rolling_adapter=rolling
        )
        engine.append_admitted_frame(_frame(0))
        seal = asyncio.create_task(engine.seal_through(0))
        await asyncio.sleep(0)
        rolling.release(_hypothesis("", "", 0))
        await engine.wait_idle()

        assert engine.latest_revision is not None
        assert engine.latest_revision.failure_code == "backend_failed"
        assert engine.status == "failed"
        assert engine._runner_task is None
        with pytest.raises(TranscriptBackendFailure):
            await asyncio.wait_for(seal, timeout=0.05)

    asyncio.run(scenario())


def test_rolling_token_cannot_reach_before_request_lower_bound_beyond_tolerance():
    async def scenario() -> None:
        rolling = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-rolling-lower-bound",
            rolling_adapter=rolling,
            frame_tolerance_ns=1_000_000,
        )
        engine.append_admitted_frame(_frame(1))
        await asyncio.sleep(0)
        rolling.release(
            TranscriptHypothesis(
                tokens=(TranscriptToken("old correction", 0, 15_000_000),),
                covered_through_ns=20_000_000,
            )
        )
        await engine.wait_idle()

        assert engine.fresh_for(20_000_000) is False
        assert engine.latest_revision is not None
        assert engine.latest_revision.failure_code == "backend_failed"

    asyncio.run(scenario())


def test_normalized_timestamp_overlap_deduplicates_boundary_jitter():
    async def scenario() -> None:
        rolling = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-token-align",
            rolling_adapter=rolling,
            rolling_window_ns=20_000_000,
        )
        engine.append_admitted_frame(_frame(0))
        engine.append_admitted_frame(_frame(1))
        await asyncio.sleep(0)
        rolling.release(
            TranscriptHypothesis(
                tokens=(
                    TranscriptToken("hello ", 0, 10_000_000),
                    TranscriptToken("world", 10_000_000, 20_000_000),
                ),
                covered_through_ns=20_000_000,
            )
        )
        await engine.wait_idle()

        engine.append_admitted_frame(_frame(2))
        await asyncio.sleep(0)
        rolling.release(
            TranscriptHypothesis(
                tokens=(
                    TranscriptToken("Hello, ", 9_000_000, 11_000_000),
                    TranscriptToken("world ", 11_000_000, 20_000_000),
                    TranscriptToken("again", 20_000_000, 30_000_000),
                ),
                covered_through_ns=30_000_000,
            )
        )
        await engine.wait_idle()

        revision = engine.latest_revision
        assert revision is not None
        assert revision.stable_text == ""
        assert revision.revisable_text == "Hello, world again"

    asyncio.run(scenario())


def test_foreign_thread_live_callbacks_mutate_only_on_owner_loop_and_seal_causally():
    async def scenario() -> None:
        live = _Live()
        owner_thread = threading.get_ident()
        revision_threads = []
        callback_errors = []
        engine = TranscriptEngine(
            turn_id="turn-owner-loop",
            live_adapter=live,
            on_revision=lambda _revision: revision_threads.append(
                threading.get_ident()
            ),
        )
        engine.append_admitted_frame(_frame(0))
        seal = asyncio.create_task(engine.seal_through(0))

        def emit() -> None:
            try:
                live.publish(_hypothesis("", "threaded", 10_000_000))
                live.settle(0)
            except BaseException as exc:
                callback_errors.append(exc)

        thread = threading.Thread(target=emit)
        thread.start()
        thread.join()

        revision = await asyncio.wait_for(seal, timeout=0.2)
        assert callback_errors == []
        assert revision.revisable_text == "threaded"
        assert revision_threads == [owner_thread]

    asyncio.run(scenario())


def test_close_cancels_rolling_fails_waiters_and_clears_pcm_idempotently():
    async def scenario() -> None:
        rolling = _SlowRolling()
        engine = TranscriptEngine(turn_id="turn-close-roll", rolling_adapter=rolling)
        engine.append_admitted_frame(_frame(0))
        await asyncio.sleep(0)
        seal = asyncio.create_task(engine.seal_through(0))
        await asyncio.sleep(0)

        await engine.close()
        await engine.close()

        assert rolling.concurrent == 0
        assert engine._runner_task is None
        assert list(engine._frames) == []
        assert engine._pending_window is None
        with pytest.raises(TranscriptBackendFailure):
            await seal

    asyncio.run(scenario())


def test_close_fences_late_native_callbacks_and_closes_adapter_once():
    async def scenario() -> None:
        live = _Live()
        revisions = []
        engine = TranscriptEngine(
            turn_id="turn-close-live",
            live_adapter=live,
            on_revision=revisions.append,
        )
        engine.append_admitted_frame(_frame(0))
        await engine.close()
        await engine.close()

        live.publish(_hypothesis("", "late", 10_000_000), attempt=0)
        live.settle(0, attempt=0)
        live.fail(attempt=0)
        await asyncio.sleep(0)

        assert live.close_calls == 1
        assert revisions == []
        assert list(engine._frames) == []

    asyncio.run(scenario())


@pytest.mark.parametrize("jitter_ns", [-1, 1, -10_000, 10_000])
def test_rolling_accepts_subsample_device_timestamp_rounding(jitter_ns):
    async def scenario() -> None:
        rolling = _SlowRolling()
        engine = TranscriptEngine(turn_id="rounded-clock", rolling_adapter=rolling)
        first = _frame(0)
        second = replace(
            _frame(1),
            started_ns=10_000_000 + jitter_ns,
            ended_ns=20_000_000 + jitter_ns,
        )
        try:
            engine.append_admitted_frame(first)
            engine.append_admitted_frame(second)
            await asyncio.sleep(0)
            assert not engine.automatic_dispatch_suspended
            assert rolling.calls == [(first.pcm16 + second.pcm16, 0, 20_000_000)]
            rolling.release(_hypothesis("", "hello", 20_000_000))
            await engine.wait_idle()
            assert engine.fresh_for(second.ended_ns)
        finally:
            await engine.close()

    asyncio.run(scenario())


def test_rolling_rejects_overlap_larger_than_pcm_rounding():
    async def scenario() -> None:
        rolling = _SlowRolling()
        engine = TranscriptEngine(turn_id="overlapping-clock", rolling_adapter=rolling)
        try:
            engine.append_admitted_frame(_frame(0))
            engine.append_admitted_frame(
                replace(_frame(1), started_ns=9_000_000, ended_ns=19_000_000)
            )
            await asyncio.sleep(0)
            assert engine.automatic_dispatch_suspended
            assert rolling.calls == []
        finally:
            await engine.close()

    asyncio.run(scenario())


def test_discontinuous_frames_insert_bounded_silence_and_report_pcm_duration():
    async def scenario() -> None:
        rolling = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-gap",
            rolling_adapter=rolling,
            rolling_window_ns=100_000_000,
        )
        first = _frame(0)
        second = AudioFrame(
            sequence=1,
            started_ns=30_000_000,
            ended_ns=40_000_000,
            pcm16=b"\x02" * 960,
            clock_generation=first.clock_generation,
            discontinuity=True,
        )
        engine.append_admitted_frame(first)
        engine.append_admitted_frame(second)
        await asyncio.sleep(0)

        pcm, started_ns, ended_ns = rolling.calls[0]
        assert (started_ns, ended_ns) == (0, 40_000_000)
        assert pcm == first.pcm16 + bytes(1_920) + second.pcm16

        rolling.release(_hypothesis("", "gap", 40_000_000))
        await engine.wait_idle()
        assert engine.latest_revision is not None
        assert engine.latest_revision.timing.processed_duration_ms == 40

    asyncio.run(scenario())


def test_compacted_receipt_eviction_fails_closed_instead_of_guessing_membership():
    async def scenario() -> None:
        rolling = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-compact",
            rolling_adapter=rolling,
            rolling_window_ns=10_000_000,
            max_sequence_entries=2,
        )
        for sequence in range(4):
            engine.append_admitted_frame(_frame(sequence))
            await asyncio.sleep(0)
            rolling.release(
                _hypothesis("", f"revision-{sequence}", (sequence + 1) * 10_000_000)
            )
            await engine.wait_idle()

        assert len(engine._sequence_coverage) <= 2
        with pytest.raises(ValueError, match="unknown admitted sequence"):
            await engine.seal_through(0)
        revision = await engine.seal_through(3)
        assert revision.covered_through_ns == 40_000_000

    asyncio.run(scenario())


def test_compacted_sequence_waits_if_a_later_overlap_derives_from_it_again():
    async def scenario() -> None:
        rolling = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-reopened-overlap",
            rolling_adapter=rolling,
            rolling_window_ns=30_000_000,
        )
        engine.append_admitted_frame(_frame(0))
        await asyncio.sleep(0)
        rolling.release(_hypothesis("", "hello", 10_000_000))
        await engine.wait_idle()
        assert 0 not in engine._sequence_coverage

        engine.append_admitted_frame(_frame(1))
        await asyncio.sleep(0)
        seal = asyncio.create_task(engine.seal_through(0))
        await asyncio.sleep(0)
        assert seal.done() is False

        rolling.release(
            TranscriptHypothesis(
                tokens=(
                    TranscriptToken("hullo ", 0, 10_000_000),
                    TranscriptToken("world", 10_000_000, 20_000_000),
                ),
                covered_through_ns=20_000_000,
            )
        )
        await engine.wait_idle()
        revision = await asyncio.wait_for(seal, timeout=0.2)
        assert revision.revisable_text == "hullo world"

    asyncio.run(scenario())


def test_unsettled_sequence_metadata_capacity_fails_closed_instead_of_evicting():
    async def scenario() -> None:
        rolling = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-sequence-cap",
            rolling_adapter=rolling,
            max_buffered_frames=10,
            max_sequence_entries=2,
        )
        engine.append_admitted_frame(_frame(0))
        await asyncio.sleep(0)
        engine.append_admitted_frame(_frame(1))
        engine.append_admitted_frame(_frame(2))

        assert engine.latest_revision is not None
        assert engine.latest_revision.failure_code == "backend_failed"
        with pytest.raises(TranscriptBackendFailure):
            await engine.seal_through(0)
        await engine.close()

    asyncio.run(scenario())


def test_gapped_admitted_sequence_ids_compact_in_arrival_order_at_small_cap():
    async def scenario() -> None:
        rolling = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-gapped-sequences",
            rolling_adapter=rolling,
            rolling_window_ns=10_000_000,
            max_sequence_entries=2,
        )
        for ordinal, sequence in enumerate((0, 2, 4, 6)):
            started_ns = ordinal * 10_000_000
            engine.append_admitted_frame(
                AudioFrame(
                    sequence=sequence,
                    started_ns=started_ns,
                    ended_ns=started_ns + 10_000_000,
                    pcm16=bytes(960),
                    clock_generation=3,
                )
            )
            await asyncio.sleep(0)
            rolling.release(
                TranscriptHypothesis(
                    tokens=(
                        TranscriptToken(
                            f"revision-{sequence}",
                            started_ns,
                            started_ns + 10_000_000,
                        ),
                    ),
                    covered_through_ns=started_ns + 10_000_000,
                )
            )
            await engine.wait_idle()

        assert engine.latest_revision is not None
        assert engine.latest_revision.failure_code is None
        assert len(engine._sequence_coverage) <= 2
        with pytest.raises(ValueError, match="unknown admitted sequence"):
            await engine.seal_through(0)
        assert (await engine.seal_through(6)).covered_through_ns == 40_000_000

    asyncio.run(scenario())


def test_compacted_membership_never_infers_skipped_vad_sequence_ids():
    async def scenario() -> None:
        rolling = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-exact-compacted-membership",
            rolling_adapter=rolling,
        )
        for ordinal, sequence in enumerate((0, 2)):
            started_ns = ordinal * 10_000_000
            engine.append_admitted_frame(
                AudioFrame(
                    sequence=sequence,
                    started_ns=started_ns,
                    ended_ns=started_ns + 10_000_000,
                    pcm16=bytes(960),
                    clock_generation=3,
                )
            )
            await asyncio.sleep(0)
            rolling.release(
                TranscriptHypothesis(
                    tokens=(
                        TranscriptToken(
                            f"revision-{sequence}",
                            started_ns,
                            started_ns + 10_000_000,
                        ),
                    ),
                    covered_through_ns=started_ns + 10_000_000,
                )
            )
            await engine.wait_idle()

        assert (await engine.seal_through(0)).covered_through_ns == 20_000_000
        with pytest.raises(ValueError, match="unknown admitted sequence"):
            await engine.seal_through(1)

    asyncio.run(scenario())


def test_pause_beyond_rolling_horizon_excludes_stale_audio_before_gap_validation():
    async def scenario() -> None:
        rolling = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-normal-long-pause",
            rolling_adapter=rolling,
        )
        engine.append_admitted_frame(_frame(0))
        await asyncio.sleep(0)
        rolling.release(_hypothesis("", "before pause ", 10_000_000))
        await engine.wait_idle()

        engine.append_admitted_frame(
            AudioFrame(
                sequence=1,
                started_ns=5_000_000_000,
                ended_ns=5_010_000_000,
                pcm16=bytes(960),
                clock_generation=3,
                discontinuity=True,
            )
        )
        await asyncio.sleep(0)

        assert engine.latest_revision is not None
        assert engine.latest_revision.failure_code is None
        assert len(rolling.calls) == 2
        assert rolling.calls[-1][1:] == (5_000_000_000, 5_010_000_000)
        rolling.release(_hypothesis("", "after pause", 5_010_000_000))
        await engine.wait_idle()
        assert engine.latest_revision.failure_code is None
        assert engine.latest_revision.stable_text == "before pause "
        assert engine.latest_revision.revisable_text == "after pause"
        assert engine.editable_draft == "before pause after pause"

    asyncio.run(scenario())


def test_overlapping_tokenless_rolling_revision_fails_closed_as_ambiguous():
    async def scenario() -> None:
        rolling = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-ambiguous-tokenless-overlap",
            rolling_adapter=rolling,
            rolling_window_ns=30_000_000,
        )
        engine.append_admitted_frame(_frame(0))
        await asyncio.sleep(0)
        rolling.release(_hypothesis("", "hello", 10_000_000))
        await engine.wait_idle()

        engine.append_admitted_frame(_frame(1))
        await asyncio.sleep(0)
        rolling.release(_hypothesis("", "hello world", 20_000_000))
        await engine.wait_idle()

        assert engine.latest_revision is not None
        assert engine.latest_revision.failure_code == "backend_failed"
        assert engine.editable_draft == "hello"
        assert engine.fresh_for(20_000_000) is False
        with pytest.raises(TranscriptBackendFailure):
            await engine.seal_through(1)

    asyncio.run(scenario())


def test_temporal_overlap_replaces_materially_corrected_boundary_token():
    async def scenario() -> None:
        rolling = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-material-boundary",
            rolling_adapter=rolling,
            rolling_window_ns=20_000_000,
        )
        engine.append_admitted_frame(_frame(0))
        engine.append_admitted_frame(_frame(1))
        await asyncio.sleep(0)
        rolling.release(
            TranscriptHypothesis(
                tokens=(
                    TranscriptToken("cat ", 0, 10_000_000),
                    TranscriptToken("sat", 10_000_000, 20_000_000),
                ),
                covered_through_ns=20_000_000,
            )
        )
        await engine.wait_idle()

        engine.append_admitted_frame(_frame(2))
        await asyncio.sleep(0)
        rolling.release(
            TranscriptHypothesis(
                tokens=(
                    TranscriptToken("cap ", 9_000_000, 11_000_000),
                    TranscriptToken("sat ", 11_000_000, 20_000_000),
                    TranscriptToken("down", 20_000_000, 30_000_000),
                ),
                covered_through_ns=30_000_000,
            )
        )
        await engine.wait_idle()

        assert is_material_transcript_change("cat", "cap") is True
        assert engine.latest_revision is not None
        assert engine.latest_revision.stable_text == ""
        assert engine.latest_revision.revisable_text == "cap sat down"

    asyncio.run(scenario())


def test_boundary_retokenization_replaces_many_old_tokens_with_one_new_token():
    async def scenario() -> None:
        rolling = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-many-to-one-boundary",
            rolling_adapter=rolling,
            rolling_window_ns=20_000_000,
            frame_tolerance_ns=20_000_000,
        )
        engine.append_admitted_frame(_frame(1))
        engine.append_admitted_frame(_frame(2))
        await asyncio.sleep(0)
        rolling.release(
            TranscriptHypothesis(
                tokens=(
                    TranscriptToken("New ", 10_000_000, 20_000_000),
                    TranscriptToken("York", 20_000_000, 30_000_000),
                ),
                covered_through_ns=30_000_000,
            )
        )
        await engine.wait_idle()

        engine.append_admitted_frame(_frame(3))
        engine.append_admitted_frame(_frame(4))
        await asyncio.sleep(0)
        rolling.release(
            TranscriptHypothesis(
                tokens=(
                    TranscriptToken("Newark ", 10_000_000, 31_000_000),
                    TranscriptToken("again", 31_000_000, 50_000_000),
                ),
                covered_through_ns=50_000_000,
            )
        )
        await engine.wait_idle()

        assert engine.latest_revision is not None
        assert engine.latest_revision.stable_text == ""
        assert engine.latest_revision.revisable_text == "Newark again"

    asyncio.run(scenario())


def test_close_aborts_cancellation_resistant_rolling_within_bound():
    async def scenario() -> None:
        rolling = _CancellationResistantRolling()
        engine = TranscriptEngine(
            turn_id="turn-resistant-close", rolling_adapter=rolling
        )
        engine.append_admitted_frame(_frame(0))
        await rolling.started.wait()
        seal = asyncio.create_task(engine.seal_through(0))
        await asyncio.sleep(0)

        close_task = asyncio.create_task(engine.close())
        done, _pending = await asyncio.wait({close_task}, timeout=0.05)
        try:
            assert close_task in done
        finally:
            rolling.released.set()
            await asyncio.wait_for(close_task, timeout=0.2)

        assert rolling.abort_calls == 1
        assert list(engine._frames) == []
        with pytest.raises(TranscriptBackendFailure):
            await seal

    asyncio.run(scenario())


def test_close_allows_a_cooperative_30ms_abort_to_release_transcription_pcm():
    async def scenario() -> None:
        rolling = _SlowCooperativeAbortRolling()
        engine = TranscriptEngine(
            turn_id="turn-cooperative-slow-close", rolling_adapter=rolling
        )
        engine.append_admitted_frame(_frame(0))
        await rolling.started.wait()

        await engine.close()

        assert rolling.abort_completed is True
        assert engine.supervised_shutdown_task_count == 0

    asyncio.run(scenario())


def test_close_quarantines_and_supervises_a_surviving_pcm_task_until_completion():
    async def scenario() -> None:
        rolling = _UnresponsiveRolling()
        engine = TranscriptEngine(
            turn_id="turn-supervised-shutdown", rolling_adapter=rolling
        )
        engine.append_admitted_frame(_frame(0))
        await rolling.started.wait()

        started = asyncio.get_running_loop().time()
        await engine.close()
        elapsed = asyncio.get_running_loop().time() - started
        try:
            assert elapsed < 0.7
            assert engine.supervised_shutdown_task_count == 2
        finally:
            rolling.released.set()
            for _ in range(3):
                await asyncio.sleep(0)
        assert engine.supervised_shutdown_task_count == 0

    asyncio.run(scenario())


def test_cancelling_close_keeps_engine_owned_teardown_and_all_survivors_supervised():
    async def scenario() -> None:
        rolling = _UnresponsiveRolling()
        engine = TranscriptEngine(
            turn_id="turn-cancelled-close", rolling_adapter=rolling
        )
        engine.append_admitted_frame(_frame(0))
        await rolling.started.wait()

        close_task = asyncio.create_task(engine.close())
        await rolling.abort_started.wait()
        close_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_task
        try:
            assert engine.supervised_shutdown_task_count == 2
            assert engine._runner_task is not None
        finally:
            rolling.released.set()

        await engine.close()
        assert engine.supervised_shutdown_task_count == 0
        assert engine._runner_task is None
        assert list(engine._frames) == []
        await engine.close()

    asyncio.run(scenario())


def test_throwing_native_close_still_clears_state_and_is_idempotent():
    async def scenario() -> None:
        live = _ThrowingCloseLive()
        engine = TranscriptEngine(turn_id="turn-throwing-close", live_adapter=live)
        engine.append_admitted_frame(_frame(0))
        seal = asyncio.create_task(engine.seal_through(0))
        await asyncio.sleep(0)

        await engine.close()
        await engine.close()

        assert live.close_calls == 1
        assert list(engine._frames) == []
        assert engine._sequence_coverage == {}
        with pytest.raises(TranscriptBackendFailure):
            await seal

    asyncio.run(scenario())


def test_oversized_rolling_window_configuration_is_rejected_before_capture():
    async def scenario() -> None:
        with pytest.raises(ValueError, match="safe maximum"):
            TranscriptEngine(
                turn_id="turn-oversized-window",
                rolling_adapter=_SlowRolling(),
                rolling_window_ns=60_000_000_000,
            )

    asyncio.run(scenario())


def test_same_rolling_instance_is_rejected_without_exposing_adapter_details():
    async def scenario() -> None:
        adapter = _SecretEqualRolling()
        with pytest.raises(ValueError) as caught:
            TranscriptEngine(
                turn_id="turn-same-rolling-fallback",
                rolling_adapter=adapter,
                fallback_adapter=adapter,
            )

        message = str(caught.value)
        assert message == "rolling_adapter and fallback_adapter must be distinct"
        assert "SECRET_ADAPTER_REPR" not in message

        primary = _SecretEqualRolling()
        fallback = _SecretEqualRolling()
        assert primary == fallback
        engine = TranscriptEngine(
            turn_id="turn-equal-distinct-rolling-fallback",
            rolling_adapter=primary,
            fallback_adapter=fallback,
        )
        await engine.close()

    asyncio.run(scenario())


def test_oversized_admitted_pcm_fails_closed_before_window_copy():
    async def scenario() -> None:
        rolling = _SlowRolling()
        engine = TranscriptEngine(
            turn_id="turn-oversized-pcm",
            rolling_adapter=rolling,
            rolling_window_ns=10_000_000_000,
        )
        pcm16 = bytes(1_000_002)
        duration_ns = (len(pcm16) // 2) * 1_000_000_000 // 48_000
        engine.append_admitted_frame(
            AudioFrame(
                sequence=0,
                started_ns=0,
                ended_ns=duration_ns,
                pcm16=pcm16,
                clock_generation=3,
            )
        )

        assert engine.latest_revision is not None
        assert engine.latest_revision.failure_code == "backend_failed"
        assert rolling.calls == []
        await engine.close()

    asyncio.run(scenario())


def test_native_processed_duration_counts_pcm_not_wall_clock_gaps():
    async def scenario() -> None:
        live = _Live()
        engine = TranscriptEngine(turn_id="turn-native-duration", live_adapter=live)
        engine.append_admitted_frame(_frame(0))
        engine.append_admitted_frame(
            AudioFrame(
                sequence=1,
                started_ns=1_000_000_000,
                ended_ns=1_010_000_000,
                pcm16=bytes(960),
                clock_generation=3,
                discontinuity=True,
            )
        )
        live.publish(_hypothesis("", "two frames", 1_010_000_000))
        await asyncio.sleep(0)

        assert engine.latest_revision is not None
        assert engine.latest_revision.timing.processed_duration_ms == 20

    asyncio.run(scenario())


def test_live_observer_exception_cannot_strand_a_causal_seal():
    async def scenario() -> None:
        live = _Live()

        def observer(_revision) -> None:
            raise RuntimeError("observer failed")

        engine = TranscriptEngine(
            turn_id="turn-live-observer",
            live_adapter=live,
            on_revision=observer,
        )
        engine.append_admitted_frame(_frame(0))
        seal = asyncio.create_task(engine.seal_through(0))
        live.publish(_hypothesis("", "live survives", 10_000_000))
        live.settle(0)

        revision = await asyncio.wait_for(seal, timeout=0.2)
        assert revision.revisable_text == "live survives"
        assert revision.failure_code is None

    asyncio.run(scenario())


def test_rolling_observer_exception_cannot_trigger_backend_failure():
    async def scenario() -> None:
        rolling = _SlowRolling()

        def observer(_revision) -> None:
            raise RuntimeError("observer failed")

        engine = TranscriptEngine(
            turn_id="turn-rolling-observer",
            rolling_adapter=rolling,
            on_revision=observer,
        )
        engine.append_admitted_frame(_frame(0))
        seal = asyncio.create_task(engine.seal_through(0))
        await asyncio.sleep(0)
        rolling.release(_hypothesis("", "rolling survives", 10_000_000))
        await engine.wait_idle()

        revision = await asyncio.wait_for(seal, timeout=0.2)
        assert revision.revisable_text == "rolling survives"
        assert revision.failure_code is None

    asyncio.run(scenario())


def test_fresh_live_restart_replays_bounded_turn_audio_before_new_coverage():
    async def scenario() -> None:
        live = _Live()
        engine = TranscriptEngine(turn_id="turn-live-replay", live_adapter=live)
        engine.append_admitted_frame(_frame(0))
        live.fail(attempt=0)
        await asyncio.sleep(0)

        engine.append_admitted_frame(_frame(1))

        assert [frame.sequence for frame in live.submissions_by_attempt[1]] == [0, 1]
        live.publish(_hypothesis("", "replayed", 20_000_000), attempt=1)
        await asyncio.sleep(0)
        assert engine.latest_revision is not None
        assert engine.latest_revision.failure_code is None

    asyncio.run(scenario())


def test_manual_live_retry_replays_buffer_before_accepting_coverage():
    async def scenario() -> None:
        live = _Live()
        engine = TranscriptEngine(turn_id="turn-manual-live-replay", live_adapter=live)
        engine.append_admitted_frame(_frame(0))
        live.fail(attempt=0)
        await asyncio.sleep(0)

        engine.manual_retry()

        assert [frame.sequence for frame in live.submissions_by_attempt[1]] == [0]
        live.publish(_hypothesis("", "retried", 10_000_000), attempt=1)
        await asyncio.sleep(0)
        assert engine.latest_revision is not None
        assert engine.latest_revision.failure_code is None

    asyncio.run(scenario())


def test_live_retry_after_ring_eviction_preserves_covered_prefix_and_sends_suffix():
    async def scenario() -> None:
        live = _Live()
        engine = TranscriptEngine(
            turn_id="turn-live-evicted-prefix",
            live_adapter=live,
            max_buffered_frames=2,
        )
        for sequence in range(3):
            engine.append_admitted_frame(_frame(sequence))
        live.publish(_hypothesis("", "one two three ", 30_000_000), attempt=0)
        await asyncio.sleep(0)
        live.fail(attempt=0)
        await asyncio.sleep(0)

        engine.append_admitted_frame(_frame(3))

        assert [frame.sequence for frame in live.submissions_by_attempt[1]] == [3]
        live.publish(_hypothesis("", "four", 40_000_000), attempt=1)
        await asyncio.sleep(0)

        assert engine.latest_revision is not None
        assert engine.latest_revision.failure_code is None
        assert engine.latest_revision.stable_text == "one two three "
        assert engine.latest_revision.revisable_text == "four"
        assert engine.editable_draft == "one two three four"

    asyncio.run(scenario())


def test_manual_live_retry_rejects_evicted_untranscribed_audio():
    async def scenario() -> None:
        live = _Live()
        engine = TranscriptEngine(
            turn_id="turn-manual-evicted-gap",
            live_adapter=live,
            max_buffered_frames=2,
        )
        for sequence in range(3):
            engine.append_admitted_frame(_frame(sequence))
        live.fail(attempt=0)
        await asyncio.sleep(0)

        engine.manual_retry()

        assert live.started == 1
        assert engine.latest_revision is not None
        assert engine.latest_revision.failure_code == "backend_failed"
        assert engine.automatic_dispatch_suspended is True

    asyncio.run(scenario())


def test_manual_live_retry_does_not_open_an_empty_suffix_session():
    async def scenario() -> None:
        live = _Live()
        engine = TranscriptEngine(
            turn_id="turn-manual-empty-suffix",
            live_adapter=live,
            max_buffered_frames=2,
        )
        for sequence in range(3):
            engine.append_admitted_frame(_frame(sequence))
        live.publish(_hypothesis("", "complete", 30_000_000), attempt=0)
        await asyncio.sleep(0)
        live.fail(attempt=0)
        await asyncio.sleep(0)

        engine.manual_retry()

        assert live.started == 1
        assert engine.latest_revision is not None
        assert engine.latest_revision.failure_code == "backend_failed"
        assert engine.editable_draft == "complete"

    asyncio.run(scenario())
