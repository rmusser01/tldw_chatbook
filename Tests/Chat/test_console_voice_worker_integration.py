"""Real coordinator/effects with fake provider/audio; no device is opened."""

import asyncio
import threading
import time
import os
from dataclasses import replace

import pytest

from Tests.Chat.test_console_speculative_voice import _FakeScheduler, _revision, _speech
from Tests.integration.test_speculative_voice_pipeline import (
    _AudioStream,
    _RestartGateway,
    _Transport,
    _prepared_attempt,
)
from tldw_chatbook.Audio.duplex_contracts import (
    DrainReceipt,
    DuplexMode,
    RenderSubmission,
)
from tldw_chatbook.Chat.console_speculative_voice import (
    AttemptPlaybackBoundaryKnown,
    AttemptPlaybackTerminal,
    SpeculativeTurnCoordinator,
)
from tldw_chatbook.Chat.console_speculative_voice_session import (
    SpeculativeVoiceAttemptEffects,
)
from tldw_chatbook.Chat.console_voice_supervisor import VoiceDispatchSupervisor
from tldw_chatbook.Chat.console_voice_tts_bridge import OwnerLoopTtsSynthesizer
from tldw_chatbook.Chat.console_voice_worker import ConsoleVoiceWorker, VoiceUiBridge
from tldw_chatbook.TTS.adapter_types import TTSAudioResponse


@pytest.mark.asyncio
@pytest.mark.parametrize("close_during_provider", [False, True])
async def test_production_child_effects_use_original_parent_context_and_pcm_credits(
    monkeypatch,
    close_during_provider,
):
    from Tests.Audio.test_voice_process_core import core, eventually
    from Tests.Chat.test_console_voice_attempts import _request
    from Tests.Chat.test_console_voice_effect_barrier import _context
    from tldw_chatbook.Audio.voice_process_entry import _ChildEffects
    from tldw_chatbook.Audio.voice_process_lifetime import LifecyclePipe
    from tldw_chatbook.Audio.voice_process_types import ControlKind
    from tldw_chatbook.Chat.console_voice_process import (
        ConsoleVoiceProcess,
        DeviceLease,
    )
    from tldw_chatbook.Chat.console_speculative_voice_session import (
        PreparedSpeculativeVoiceAttempt,
        _LazyHandsFreeTts,
    )

    context = _context()
    prepared_calls, accepted_calls, faults = [], [], []
    source_entered = asyncio.Event()

    async def prepare(**kwargs):
        prepared_calls.append(kwargs)
        return PreparedSpeculativeVoiceAttempt(
            replace(_request(), attempt_epoch=kwargs["attempt_epoch"]), context
        )

    class Gateway:
        async def stream_chat(self, resolution, request, **kwargs):
            source_entered.set()
            yield "A controlled answer."
            if close_during_provider:
                await asyncio.Future()

    async def synthesize(self, *, text):
        assert text == "A controlled answer."

        async def frames():
            yield b"\x01\x00" * 480

        return TTSAudioResponse(
            provider_id="test",
            model_id="test",
            audio_format="pcm",
            content_type="audio/pcm",
            sample_rate=48000,
            metadata={"channels": 1},
            byte_stream=frames(),
        )

    monkeypatch.setattr(_LazyHandsFreeTts, "synthesize_hands_free", synthesize)
    parent = ConsoleVoiceProcess(
        DeviceLease(),
        current=lambda: True,
        prepare_attempt=prepare,
        gateway=Gateway(),
        dispatch_supervisor=VoiceDispatchSupervisor(),
        accepted_handoff=lambda text, frozen: (
            accepted_calls.append((text, frozen)) or "accepted-turn"
        ),
    )
    child_read, parent_write = os.pipe()
    parent_read, child_write = os.pipe()
    child = None
    parent.pipe = LifecyclePipe(
        parent_read,
        parent_write,
        generation=1,
        request_id="a" * 32,
        parent=True,
        consume=parent._receive,
        fault=parent._transport_fault,
    )
    parent._compose_parent()
    pipe = LifecyclePipe(
        child_read,
        child_write,
        generation=1,
        request_id="a" * 32,
        parent=False,
        consume=lambda record: child.receive(record),
        fault=faults.append,
    )
    audio, transport, _, scheduler = core(
        deferred_attempt_preparation=True, initial_duplex_mode=DuplexMode.FULL_DUPLEX
    )
    child = _ChildEffects(pipe, transport, lambda: audio, faults.append)
    audio._delegate_effects = child
    audio._effects._delegate = child
    try:
        await audio.coordinator.start()
        turn, epoch = await __import__(
            "Tests.Chat.test_console_speculative_voice", fromlist=["_start_attempt"]
        )._start_attempt(audio.coordinator, scheduler)
        if close_during_provider:
            await asyncio.wait_for(source_entered.wait(), 1)
            state = child.attempts[epoch]
            parent._effects.fence()
            closed = asyncio.create_task(child.close())
            try:
                await parent._effects.aclose()
                await asyncio.wait_for(asyncio.shield(closed), 0.5)
                assert state.cleanup.result().value == "clean"
                assert parent._effects.failure is None and faults == []
            finally:
                closed.cancel()
                await asyncio.gather(closed, return_exceptions=True)
            return
        await eventually(
            lambda: (
                bool(transport.rendered) or bool(faults) or parent._failure is not None
            )
        )
        assert faults == [] and parent._failure is None
        assert transport.rendered == [b"\x01\x00" * 480]
        await eventually(
            lambda: (
                parent._tts.outstanding == (0, 0)
                and parent._tts.cleanup_outstanding == (0, 0)
            )
        )
        assert prepared_calls[0]["transcript"] == "hello world"
        state = child.attempts[epoch]
        assert (
            parent._effects.context_for(
                state.key, state.prepared.header["context_handle"]
            )
            is context
        )
        child.fence_attempt(epoch)
        await eventually(lambda: state.cleanup.done())
        # An accepted revision retains the original issued context epoch.
        await audio.coordinator.submit(
            _revision(turn, 2, "hello world revised", 10**18)
        )
        child.record_revision(_revision(turn, 2, "hello world revised", 10**18))
        child.submit_accepted_voice_turn(
            "hello world revised", audio.coordinator._turn_context_handle
        )
        await eventually(
            lambda: bool(accepted_calls) or bool(faults) or parent._failure is not None
        )
        assert faults == [] and parent._failure is None
        assert accepted_calls == [("hello world revised", context)]
        await eventually(lambda: not parent._contexts)
        assert parent._effects.draft_for(turn) is None
    finally:
        child.fence_all()
        await parent._effects.aclose()
        parent._tts.transport_failed(
            __import__(
                "tldw_chatbook.Audio.voice_process_protocol", fromlist=["ProtocolError"]
            ).ProtocolError("voice_transport_closed")
        )
        for state in child.attempts.values():
            if not state.cleanup.done():
                from tldw_chatbook.Audio.voice_process_types import (
                    AttemptCleanupOutcome,
                )

                state.cleanup.set_result(AttemptCleanupOutcome.CLEAN)
        await audio.fence_and_close(ControlKind.TEARDOWN)
        for task in tuple(child.tasks):
            task.cancel()
        await asyncio.gather(*tuple(child.tasks), return_exceptions=True)
        for value in (pipe, parent.pipe):
            value.stop()
        os.close(parent_write)
        os.close(child_write)
        for value in (pipe, parent.pipe):
            assert await value.join()


@pytest.mark.asyncio
async def test_barge_in_cancels_before_ui_resumes_then_revises_and_starts_new_turn():
    ui_thread = threading.get_ident()
    worker = ConsoleVoiceWorker()
    ui = VoiceUiBridge(asyncio.get_running_loop())
    ui_blocked = threading.Event()
    cancelled_before_resume = threading.Event()
    prepared_turns = []
    promotions = []
    latencies = []

    async def prepare(**request):
        assert threading.get_ident() == ui_thread
        prepared_turns.append((request["turn_id"], request["transcript"]))
        return _prepared_attempt(request["attempt_epoch"])

    class Stream(_AudioStream):
        async def __anext__(self):
            assert threading.get_ident() == ui_thread
            return await super().__anext__()

        async def aclose(self):
            assert threading.get_ident() == ui_thread
            await super().aclose()

    async def synthesize(*, text):
        assert threading.get_ident() == ui_thread
        return TTSAudioResponse(
            provider_id="test",
            model_id="test",
            audio_format="pcm",
            content_type="audio/pcm",
            sample_rate=48_000,
            metadata={"channels": 1},
            byte_stream=Stream(),
        )

    async def build():
        gateway = _RestartGateway()
        scheduler = _FakeScheduler()

        class Transport(_Transport):
            def queue_render(self, pcm16):
                super().queue_render(pcm16)
                return RenderSubmission(0, 0, len(self.rendered) - 1)

            def fence_output(self):
                self.rendered.clear()

        transport = Transport()

        class Effects(SpeculativeVoiceAttemptEffects):
            async def drain_capture_through(self, boundary):
                return DrainReceipt(boundary + 10_000_000, 1, 1, 1, 0)

            async def drain_pending_classification_through(self, *_):
                pass

            async def seal_transcript_through(self, _sequence):
                return _revision(
                    coordinator.snapshot.turn_id, 99, prepared_turns[-1][1], 10**18
                )

        effects = Effects(
            submit_event=lambda event: coordinator.submit(event),
            prepare_attempt=lambda **kwargs: ui.prepare(lambda: prepare(**kwargs)),
            gateway=gateway,
            synthesizer=bridge,
            transport=transport,
            promotion=lambda **kwargs: ui.terminal(
                lambda: promotions.append((threading.get_ident(), kwargs["transcript"]))
            ),
            promotion_owner=object(),
            dispatch_supervisor=VoiceDispatchSupervisor(),
            project_preview=lambda _: None,
            clear_preview=lambda: None,
            submit_accepted_voice_turn=lambda *_: None,
        )
        coordinator = SpeculativeTurnCoordinator(
            effects=effects,
            scheduler=scheduler,
            initial_duplex_mode=DuplexMode.FULL_DUPLEX,
            deferred_attempt_preparation=True,
        )
        await coordinator.start()
        frame = _speech(0, started_ns=0)
        await coordinator.submit(frame)
        first_turn = coordinator.snapshot.turn_id
        await coordinator.submit(
            _revision(first_turn, 1, "Initial speech", frame.ended_ns)
        )
        scheduler.advance_ms(700)
        await coordinator.flush()
        await gateway.first_started.wait()
        while not transport.rendered:
            await asyncio.sleep(0.001)

        async def interrupt():
            while not ui_blocked.is_set():
                await asyncio.sleep(0.001)
            started = time.monotonic()
            frame = _speech(1, started_ns=scheduler.now_ns)
            await coordinator.submit(frame)
            await gateway.first_cancelled.wait()
            while transport.rendered:
                await asyncio.sleep(0.001)
            latencies.append(time.monotonic() - started)
            assert coordinator.snapshot.turn_id == first_turn
            cancelled_before_resume.set()

        interruption = asyncio.create_task(interrupt())
        return coordinator, scheduler, effects, first_turn, interruption

    bridge = OwnerLoopTtsSynthesizer(asyncio.get_running_loop(), synthesize)
    coordinator, scheduler, effects, first_turn, interruption = await worker.run(build)
    try:
        ui_blocked.set()
        assert cancelled_before_resume.wait(0.15), "barge-in depended on UI progress"
        assert latencies[0] < 0.15

        async def revised():
            await interruption
            await coordinator.submit(
                _revision(first_turn, 2, "Initial speech plus correction", 10**18)
            )
            scheduler.advance_ms(700)
            await coordinator.flush()
            epoch = coordinator.snapshot.current_attempt_epoch
            while effects.final_render_submission(epoch) is None:
                await asyncio.sleep(0.001)
            # This standalone worker/effects fixture has no session audio pump.
            # Supply its simulated actual DAC boundary and elapsed terminal;
            # native receipt/pump ordering is covered by the session tests.
            await coordinator.submit(
                AttemptPlaybackBoundaryKnown(epoch, scheduler.now_ns)
            )
            await coordinator.submit(AttemptPlaybackTerminal(epoch, scheduler.now_ns))
            while not promotions:
                await asyncio.sleep(0.001)
            await coordinator.flush()
            await coordinator.submit(
                _speech(2, started_ns=scheduler.now_ns + 1_000_000_000)
            )
            return coordinator.snapshot.turn_id

        async with asyncio.timeout(3):
            next_turn = await worker.run(revised)
        assert prepared_turns[:2] == [
            (first_turn, "Initial speech"),
            (first_turn, "Initial speech plus correction"),
        ]
        assert promotions == [(ui_thread, "Initial speech plus correction")]
        assert next_turn != first_turn
    finally:

        async def close():
            interruption.cancel()
            await asyncio.gather(interruption, return_exceptions=True)
            await coordinator.close()
            await effects.close()
            await bridge.aclose()

        await worker.run(close)
        await worker.aclose()
