"""Visible native voice control, callback continuity and categorical errors."""

from __future__ import annotations

import asyncio
import ctypes
from contextlib import asynccontextmanager
from dataclasses import replace
import gc
import struct
import time

import pytest
from textual.widgets import Switch

from Tests.Audio import test_native_duplex_stream as native_helpers
from Tests.Audio.fakes.native_duplex_helpers import load_native
from Tests.UI.test_console_dictation import _mounted_console
from Tests.UI.test_console_hands_free_wiring import _ready_host
from Tests.UI.test_console_native_chat_flow import (
    _ReadyResolutionGateway,
)
from Tests.UI.test_console_speculative_voice_wiring import _QualifiedSession
from Tests.Chat.test_console_speculative_voice import (
    _FakeEffects,
    _FakeScheduler,
    _revision,
    _start_attempt,
)
from Tests.Chat.test_console_speculative_voice_session import Transcript
from Tests.integration.test_speculative_voice_pipeline import _frame
from tldw_chatbook.Audio.voice_preprocessor import VoicePreprocessor
from tldw_chatbook.Audio.native_duplex_stream import (
    AudioShutdownUnconfirmed,
    NativeDuplexUnavailable,
)
from tldw_chatbook.UI.Console_Modules import hands_free as hands_free_module
from tldw_chatbook.Chat.console_speculative_voice import (
    AttemptPlaybackBoundaryKnown,
    AttemptPlaybackStarted,
    AttemptPlaybackTerminal,
)


driver = native_helpers.driver
portaudio = native_helpers.portaudio
native_transport = native_helpers.native_transport


@asynccontextmanager
async def _native_core(transport):
    from tldw_chatbook.Audio.voice_process_core import ConsoleSpeculativeVoiceSession
    from tldw_chatbook.Audio.voice_process_types import ControlKind

    effects = _FakeEffects()
    effects.assistant_rendering = False
    effects.final_render_submission = lambda epoch: None
    core = ConsoleSpeculativeVoiceSession(
        transport=transport,
        effects=effects,
        transcript_factory=Transcript,
        preprocessor_factory=lambda **callbacks: VoicePreprocessor.from_native(
            vad=lambda _: False, **callbacks
        ),
    )
    try:
        await core.enter(capture_live=False)
        yield core
    finally:
        await core.fence_and_close(ControlKind.TEARDOWN)


@pytest.mark.asyncio
async def test_direct_native_core_preserves_real_aec_records_through_gc(
    monkeypatch, native_transport, portaudio, driver
):
    """One bounded GIL hold at full-GC entry uses real native callback and AEC."""
    native = load_native()
    records = []
    process = VoicePreprocessor.process_capture

    async def observe_capture(self, frame, *, render_frames=(), **kwargs):
        references = tuple(render_frames)
        records.append((frame, references, time.monotonic_ns()))
        return await process(self, frame, render_frames=references, **kwargs)

    monkeypatch.setattr(VoicePreprocessor, "process_capture", observe_capture)
    async with _native_core(native_transport) as core:

        def prepare(core):
            assert isinstance(core._preprocessor._aec, native.AecProcessor)
            transport = core._transport
            for i in range(8):
                assert transport.queue_render(struct.pack("<h", 20 + i) * 480)
            return transport, transport._stream.bridge

        transport, bridge = prepare(core)
        ffi = portaudio.sd._ffi
        assert int(ffi.cast("uintptr_t", portaudio.callback)) == bridge.callback_address
        assert int(ffi.cast("uintptr_t", portaudio.userdata)) == bridge.userdata_address
        assert transport.buffer_capacities.capture_frames == 64
        paced_progress = driver.paced_progress_with_gil_held
        start_ns = ctypes.c_uint64()
        progress = []

        def controlled_gc_pause(phase, info):
            if phase == "start" and info["generation"] == 2:
                completed = paced_progress(
                    bridge.callback_address,
                    bridge.userdata_address,
                    8,
                    ctypes.byref(start_ns),
                )
                # The consumer cannot run while the native driver holds this GIL.
                # Keep assertions outside GC callbacks (exceptions are unraisable).
                progress.append(
                    (completed, len(records), bridge.snapshot()["callback_count"])
                )
                # Fence after output commit, before Python drains the backlog.
                transport.fence_audio_admission()

        gc.callbacks.append(controlled_gc_pause)
        try:
            gc.collect(2)
        finally:
            gc.callbacks.remove(controlled_gc_pause)
        assert progress == [(8, 0, 8)]
        async with asyncio.timeout(5):
            while len(records) != 8:
                await asyncio.sleep(0.01)
        await asyncio.sleep(0)
        assert [frame.sequence for frame, _, _ in records] == list(range(8))
        for i, (frame, references, drained_ns) in enumerate(records):
            assert frame.pcm16 == struct.pack("<h", i + 1) * 480
            assert not frame.discontinuity
            assert len(references) == 1
            assert references[0].sequence == i
            assert references[0].pcm16 == struct.pack("<h", i + 20) * 480
            if i:
                assert abs(frame.started_ns - records[i - 1][0].ended_ns) <= 1
            assert frame.ended_ns < drained_ns
        # Capture clock remains anchored at native observation, not delayed drain.
        assert records[0][2] - records[0][0].ended_ns >= 60_000_000
        assert transport.capture_overflows == transport.render_reference_overflows == 0
        assert transport.native_counters["callback_count"] == 8
        assert any(frame.assistant_rendering for frame, _, _ in records)
        assert core._preprocessor._last_render_sequence == 7
        assert not core._assistant_rendering() and not core._transcripts
    assert portaudio.calls[-2:] == ["stop", "close"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "postboundary,terminal_first", [(False, False), (True, False), (True, True)]
)
async def test_direct_native_core_routes_default_preroll_by_speech_onset(
    monkeypatch, native_transport, postboundary, terminal_first
):
    """Synthetic speech/provider events supplement the separate real-AEC proof."""
    async with _native_core(native_transport) as core:

        async def exercise(core):
            coordinator = core.coordinator
            scheduler = _FakeScheduler()
            effects = _FakeEffects()
            effects.assistant_rendering = False
            effects.final_render_submission = lambda epoch: None
            effects.abort_hook = core._transport.fence_audio_admission
            core._delegate_effects = effects
            core._effects._delegate = effects
            core._transcript_factory = Transcript
            coordinator._scheduler = scheduler
            coordinator._deferred_attempt_preparation = False
            turn_id, epoch = await _start_attempt(coordinator, scheduler)
            await core.submit(AttemptPlaybackStarted(epoch))
            boundary = 1_000_000_000
            await core.submit(AttemptPlaybackBoundaryKnown(epoch, boundary))
            # Synthetic admitted-event routing uses the real default-preroll
            # policy in idle half duplex. Native/AEC composition is proven above.
            core._preprocessor = VoicePreprocessor(
                aec=None,
                vad=lambda frame: frame.sequence == (2 if postboundary else 1),
                on_admitted_frame=core._on_admitted_frame,
                on_processed=core._on_processed,
                on_classification_changed=core._on_classification_changed,
            )
            silent = replace(
                _frame(1),
                started_ns=boundary,
                ended_ns=boundary + 10_000_000,
                pcm16=bytes(960),
                assistant_rendering=False,
            )
            positive = replace(
                _frame(2),
                started_ns=boundary + 10_000_000,
                ended_ns=boundary + 20_000_000,
                pcm16=b"\x02\x00" * 480,
                assistant_rendering=False,
            )
            if not postboundary:
                silent = replace(silent, pcm16=b"\x01\x00" * 480)

            async def admit(frame):
                core._admitted.clear()
                await core._preprocessor.process_capture(
                    frame, assistant_rendering=False
                )
                admitted = tuple(core._admitted)
                core._admitted.clear()
                for item in admitted:
                    await core._accept_admitted_frame(item, assistant_rendering=False)

            await admit(silent)
            if postboundary:
                if terminal_first:
                    scheduler.now_ns = boundary + 20_000_000
                    await core.submit(AttemptPlaybackTerminal(epoch, boundary))
                await admit(positive)
                if not terminal_first:
                    scheduler.now_ns = boundary + 20_000_000
                    await core.submit(AttemptPlaybackTerminal(epoch, boundary))
                snapshot = coordinator.snapshot
                assert snapshot.current_attempt_epoch == epoch
                assert snapshot.turn_id == turn_id
                assert snapshot.pending_next_turn_id != turn_id
                assert snapshot.pending_next_turn_id is not None
                transcript = core._transcripts[0]
                assert transcript.turn_id == snapshot.pending_next_turn_id
                assert [f.pcm16 for f in transcript.frames] == [
                    silent.pcm16,
                    positive.pcm16,
                ]
                assert [f.started_ns for f in transcript.frames] == [
                    boundary,
                    boundary + 10_000_000,
                ]
                assert [f.speech_started_ns for f in transcript.frames] == [
                    boundary + 10_000_000,
                    None,
                ]
            else:
                await coordinator.flush()
                assert coordinator.snapshot.turn_id == turn_id
                assert epoch in effects.fenced_epochs
                assert ("abort", epoch) in effects.operations
                assert core._transport.queue_render(bytes(960)) is None
                assert core._transcripts[0].turn_id == turn_id
                await core.submit(
                    _revision(turn_id, 2, "synthetic revised request", silent.ended_ns)
                )
                assert coordinator.snapshot.turn_id == turn_id
                assert (
                    coordinator.snapshot.transcript_text == "synthetic revised request"
                )
            assert effects.promotions == []

        await exercise(core)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "edge",
    [
        "startup",
        "startup-sync",
        "construction",
        "runtime",
        "cleanup",
        "cleanup-sync",
        "runtime-cleanup",
    ],
)
async def test_native_failure_copy_is_actionable_and_content_free(monkeypatch, edge):
    monkeypatch.setattr(hands_free_module, "speculative_voice_qualified", lambda: True)
    app, host = _ready_host()
    notices = []
    monkeypatch.setattr(app, "notify", lambda text, **_: notices.append(text))
    callbacks = {}
    failure = (
        NativeDuplexUnavailable("private backend device text")
        if edge in {"startup", "startup-sync", "construction"}
        else AudioShutdownUnconfirmed("private backend device text")
    )

    class Session(_QualifiedSession):
        def enter(self, *, capture_live):
            if edge == "startup-sync":
                raise failure
            if edge == "startup":

                async def start():
                    raise failure

                return start()

        def fence_and_close(self, reason):
            super().fence_and_close(reason)
            if edge == "cleanup-sync":
                raise failure
            if edge in {"cleanup", "runtime-cleanup"}:

                async def close():
                    raise failure

                return close()

    def factory(**kwargs):
        callbacks.update(kwargs)
        if edge == "construction":
            raise failure
        return Session()

    app.console_speculative_voice_session_factory = factory
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        monkeypatch.setattr(
            console._ensure_console_chat_controller().provider_gateway,
            "resolve_for_send",
            _ReadyResolutionGateway().resolve_for_send,
        )
        assert await pilot.click("#console-hands-free-switch")
        await pilot.pause()
        if edge in {"runtime", "runtime-cleanup"}:
            callbacks["on_runtime_failure"](failure)
        elif edge in {"cleanup", "cleanup-sync"}:
            assert await pilot.click("#console-hands-free-switch")
        await pilot.pause()
        assert console._console_hands_free is None
        assert not console.query_one("#console-hands-free-switch", Switch).value
        copy = " ".join(notices).lower()
        if edge in {"startup", "startup-sync", "construction"}:
            assert "native duplex unavailable" in copy
            assert "rebuild" in copy
        else:
            assert "audio shutdown unconfirmed" in copy
            assert "restart" in copy
        assert "private backend" not in copy
        assert "microphone closed" not in copy
        assert len(notices) == 1


@pytest.mark.asyncio
async def test_late_native_cleanup_cannot_notify_or_repaint_replacement(monkeypatch):
    monkeypatch.setattr(hands_free_module, "speculative_voice_qualified", lambda: True)
    app, host = _ready_host()
    notices = []
    monkeypatch.setattr(app, "notify", lambda text, **_: notices.append(text))
    release = asyncio.Event()
    sessions = []

    class Session(_QualifiedSession):
        def fence_and_close(self, reason):
            super().fence_and_close(reason)

            async def close():
                await release.wait()
                raise AudioShutdownUnconfirmed("private late failure")

            return close()

    def factory(**kwargs):
        session = Session() if not sessions else _QualifiedSession()
        sessions.append(session)
        return session

    app.console_speculative_voice_session_factory = factory
    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        monkeypatch.setattr(
            console._ensure_console_chat_controller().provider_gateway,
            "resolve_for_send",
            _ReadyResolutionGateway().resolve_for_send,
        )
        assert await pilot.click("#console-hands-free-switch")
        await pilot.pause()
        console.action_exit_console_hands_free()
        console.action_toggle_console_hands_free()
        await pilot.pause()
        replacement = console._console_hands_free
        notices.clear()
        release.set()
        await pilot.pause()
        assert console._console_hands_free is replacement
        assert console.query_one("#console-hands-free-switch", Switch).value
        assert notices == []
