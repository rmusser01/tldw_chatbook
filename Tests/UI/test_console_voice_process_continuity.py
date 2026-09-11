"""Mounted process/native continuity acceptance (software-only)."""

import asyncio
from contextlib import asynccontextmanager
import ctypes
import json
import os
from types import SimpleNamespace

import pytest

from Tests.Audio.fakes.native_duplex_helpers import (
    build_driver,
    driver_library,
    driver_clock_offset,
)
from Tests.Chat.test_console_voice_process import fake_popen, cleanup
from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from Tests.UI.test_console_native_chat_flow import _ReadyResolutionGateway


async def eventually(predicate, seconds=8):
    async with asyncio.timeout(seconds):
        while not predicate():
            await asyncio.sleep(0.005)


class ChildControl:
    """One bounded test command/response slot outside production IPC."""

    def __init__(self, directory):
        self.directory, self.sequence = directory, 0

    def read(self, name):
        path = self.directory / name
        return json.loads(path.read_text()) if path.exists() else {}

    async def command(self, op, **fields):
        self.sequence += 1
        temporary = self.directory / "command.tmp"
        temporary.write_text(json.dumps(dict(id=self.sequence, op=op, **fields)))
        temporary.replace(self.directory / "command.json")
        await eventually(
            lambda: self.read("response.json").get("id") == self.sequence, 30
        )
        response = self.read("response.json")
        assert "error" not in response, response
        return response["result"]

    async def wait(self, predicate):
        async with asyncio.timeout(8):
            while True:
                result = await self.command("snapshot")
                if predicate(result):
                    return result
                await asyncio.sleep(0.01)


@asynccontextmanager
async def mounted_native_process(monkeypatch, tmp_path, *, interruptions=3):
    from tldw_chatbook.Chat import console_voice_process, console_voice_input
    from tldw_chatbook.Chat import console_speculative_voice_session as session_module
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderResolution,
    )
    from tldw_chatbook.UI.Console_Modules import hands_free
    from tldw_chatbook.TTS.adapter_types import TTSAudioResponse
    from tldw_chatbook.Chat.console_voice_promotion import (
        VoicePromotionOwner,
        VoiceWinningPromotion,
    )

    driver = driver_library(build_driver(tmp_path))
    control = ChildControl(tmp_path)
    created, generations, cleanups, syntheses = [], [], [], []
    release_provider = asyncio.Event()
    promotions, promotion_requests, tts_cleanups = [], [], []
    promotion_gate = asyncio.Event()
    promotion_gate.set()
    original_promote = VoiceWinningPromotion.promote
    original_run_promotion = VoicePromotionOwner._run_promotion

    def promote(self, context, snapshot, **kwargs):
        promotion_requests.append(context)
        settlement = original_promote(self, context, snapshot, **kwargs)

        async def observe():
            outcome = await settlement
            promotions.append((context, outcome))
            return outcome

        return observe()

    async def run_promotion(self, claim):
        # Hold the actual publication owner, after the synchronous claim.
        await promotion_gate.wait()
        return await original_run_promotion(self, claim)

    original_init = console_voice_process.ConsoleVoiceProcess.__init__

    def initialize(self, *args, **kwargs):
        kwargs["popen"] = fake_popen("native", tmp_path, [])
        original_init(self, *args, **kwargs)
        created.append(self)

    async def ready(self, selection):
        result = await _ReadyResolutionGateway.resolve_for_send(self, selection)
        return ConsoleProviderResolution(**vars(result))

    async def stream(self, *args, **kwargs):
        index = len(generations) + 1
        generations.append(index)
        try:
            yield "You could make vegetable pasta. "
            if index <= interruptions:
                await release_provider.wait()
        finally:
            cleanups.append(index)

    async def synthesize(self, *, text):
        index = len(syntheses) + 1
        syntheses.append(index)

        async def pcm():
            try:
                for frame_index in range(500 if index <= interruptions else 12):
                    # Seed 320 ms before a 120 ms interruption, then follow the
                    # device cadence without filling the default 64-frame ring.
                    if frame_index >= 32:
                        await asyncio.sleep(0.01)
                    yield (100 + index).to_bytes(2, "little") * 480
            finally:
                tts_cleanups.append(index)

        return TTSAudioResponse(
            provider_id="test",
            model_id="test",
            audio_format="pcm",
            content_type="audio/pcm",
            byte_stream=pcm(),
            sample_rate=48000,
            metadata={"channels": 1},
        )

    monkeypatch.setattr(
        console_voice_process.ConsoleVoiceProcess, "__init__", initialize
    )
    monkeypatch.setattr(hands_free, "speculative_voice_qualified", lambda: True)
    monkeypatch.setattr(
        console_voice_input,
        "resolve",
        lambda: SimpleNamespace(provider="faster-whisper", model=None, language="en"),
    )
    monkeypatch.setattr(ConsoleProviderGateway, "resolve_for_send", ready)
    monkeypatch.setattr(ConsoleProviderGateway, "stream_chat", stream)
    monkeypatch.setattr(
        session_module._LazyHandsFreeTts, "synthesize_hands_free", synthesize
    )
    monkeypatch.setattr(VoiceWinningPromotion, "promote", promote)
    monkeypatch.setattr(VoicePromotionOwner, "_run_promotion", run_promotion)
    app, host = _ready_host()
    monkeypatch.delattr(app, "console_speculative_voice_session_factory", raising=False)
    try:
        async with host.run_test(size=(140, 42)) as pilot:
            console = await _mounted_console(host, pilot)
            assert await pilot.click("#console-hands-free-switch")
            await eventually(
                lambda: created and (created[-1].ready or created[-1]._failure)
            )
            owner = created[-1]
            if owner._failure is not None:
                failure = dict(
                    failure_code=owner._failure.code,
                    transport_failure_code=owner.transport_failure_code,
                    hello_verified=owner.hello_verified,
                    child_returncode=owner.process.poll() if owner.process else None,
                    startup=control.read("startup.json"),
                    diagnostics=control.read("diagnostics.json"),
                )
                (tmp_path / "readiness_failure.json").write_text(json.dumps(failure))
                pytest.fail(str(failure))
            await eventually(lambda: control.read("started.json"))
            yield SimpleNamespace(
                app=app,
                host=host,
                console=console,
                owner=owner,
                control=control,
                driver=driver,
                generations=generations,
                cleanups=cleanups,
                syntheses=syntheses,
                promotions=promotions,
                promotion_requests=promotion_requests,
                promotion_gate=promotion_gate,
                tts_cleanups=tts_cleanups,
            )
            await owner.close()
            await owner.wait_effects_closed()
    finally:
        promotion_gate.set()
        for owner in created:
            await cleanup(owner)


@pytest.mark.asyncio
async def test_mounted_native_child_progresses_during_parent_gil_holds(
    monkeypatch, tmp_path
):
    async with mounted_native_process(monkeypatch, tmp_path) as h:
        clock_offset = driver_clock_offset(h.driver)
        clock_uncertainty = h.driver.clock_uncertainty_ns
        await h.control.command("produce", count=1800)
        await h.control.wait(lambda result: len(result["dsp"]) >= 5)
        original = await h.control.command("speak")
        original_turn_id = original["turn_id"]
        assert original_turn_id
        measurements = []
        for index, duration in enumerate((720, 3000, 720), 1):
            before = await h.control.wait(
                lambda result: (
                    len(h.syntheses) >= index
                    and result["bridge"].get("render_occupancy", 0) >= 24
                )
            )
            await h.control.command("arm")
            token = h.control.sequence
            h.owner.pipe.send("lease")
            await eventually(lambda: h.control.read("lease.json").get("token") == token)
            lease = h.control.read("lease.json")
            prior_cleanup = tuple(h.cleanups)
            prior_tts_cleanup = tuple(h.tts_cleanups)
            bounds = (ctypes.c_uint64 * 2)()
            h.driver.hold_gil_ms(duration, bounds)
            bounds = [stamp + clock_offset for stamp in bounds]
            assert bounds[1] - bounds[0] >= duration * 1_000_000
            assert tuple(h.cleanups) == prior_cleanup
            assert tuple(h.tts_cleanups) == prior_tts_cleanup
            after = await h.control.command("snapshot")
            parent_drift = abs(driver_clock_offset(h.driver) - clock_offset)
            child_drift = abs(
                after["driver_offset_after_ns"] - after["driver_offset_ns"]
            )
            assert parent_drift < 1_000_000 and child_drift < 1_000_000
            calibration_bound = (
                clock_uncertainty
                + h.driver.clock_uncertainty_ns
                + after["driver_uncertainty_ns"]
                + after["driver_uncertainty_after_ns"]
                + parent_drift
                + child_drift
            )
            # Conservatively exclude endpoints by more than the full observed
            # bracketing uncertainty plus offset variation of both clocks.
            margin = 2_000_000
            assert calibration_bound < margin
            captures = sum(
                bounds[0] + margin <= stamp <= completed < bounds[1] - margin
                for stamp, completed, _ in after["callbacks"]
            )
            dsp = sum(
                bounds[0] + margin <= stamp < bounds[1] - margin and ok
                for stamp, ok, _ in after["dsp"]
            )
            admitted_ns, interrupted_turn = after["admissions"][-1]
            fence_ns, _, native_fence_ns = next(
                item for item in after["fences"] if item[0] >= admitted_ns
            )
            late_render = sum(
                native_fence_ns < stamp - after["driver_offset_ns"]
                and stamp < bounds[1] - margin
                and sample != 0
                for stamp, _, sample in after["callbacks"]
            )
            measurement = dict(
                duration_ms=duration,
                bounds=list(bounds),
                child_pid=after["pid"],
                captures=captures,
                dsp=dsp,
                admitted_ns=admitted_ns,
                fence_ns=fence_ns,
                native_fence_ns=native_fence_ns,
                late_render=late_render,
                native=after["native"],
                parent_offset_ns=clock_offset,
                child_offset_ns=after["driver_offset_ns"],
                calibration_bound_ns=calibration_bound,
                ring_overflows=[
                    after["capture_overflows"],
                    after["render_overflows"],
                    after["reference_overflows"],
                ],
                discontinuities=after["discontinuities"],
                drift_faults=after["drift_faults"],
            )
            measurements.append(measurement)
            (tmp_path / "measurements.json").write_text(json.dumps(measurements))
            print("PROCESS_GIL_HOLD " + json.dumps(measurement))
            assert after["pid"] != os.getpid()
            assert lease["render_occupancy"] >= 20
            assert after["native_abi"] == 1 and after["capacities"] == [64, 64, 64]
            assert after["preroll_ms"] == 240
            assert (
                lease["received_ns"]
                < bounds[0] + margin
                < admitted_ns
                < fence_ns
                < bounds[1] - margin
            )
            assert captures > 0 and dsp > 0
            assert any(
                bounds[0] + margin < stamp < bounds[1] - margin
                for _, stamp, _ in after["captures"]
            )
            assert (
                after["capture_overflows"]
                == after["render_overflows"]
                == after["reference_overflows"]
                == 0
            )
            invalid_native_records = (
                after["native"]["invalid_frames"] + after["native"]["invalid_timing"]
            )
            assert (
                after["discontinuities"]
                == after["drift_faults"]
                == invalid_native_records
                == 0
            )
            assert after["native"]["fatal_status_bits"] == 0
            assert 0 <= fence_ns - admitted_ns < 150_000_000
            assert interrupted_turn == original_turn_id
            assert late_render == 0
            assert not any(item[2] for item in after["captures"])
            assert before["aec"] == "AecProcessor"
        final = await h.control.wait(
            lambda result: (
                result["turn_id"] is None and not result["retained_attempt_epochs"]
            )
        )
        assert final["terminal_receipts"] == [
            ["terminal_claim", original_turn_id, None],
            ["terminal_result", original_turn_id, "promoted"],
        ]
        assert "playback_terminal" in final["events"]
        assert len(h.promotions) == 1
        assert (
            h.promotions[0][0].user_text
            == "Dinner request correction correction correction"
        )
        assert h.promotions[0][1].status.value == "promoted"
        followup = await h.control.command("speak")
        assert followup["turn_id"] != original_turn_id
        assert followup["turn_id"] is not None
        completed = await h.control.wait(
            lambda result: (
                result["turn_id"] is None and not result["retained_attempt_epochs"]
            )
        )
        assert completed["terminal_receipts"] == final["terminal_receipts"] + [
            ["terminal_claim", followup["turn_id"], None],
            ["terminal_result", followup["turn_id"], "promoted"],
        ]
        assert h.owner._failure is None, (
            h.owner._failure,
            h.owner.transport_failure_code,
            h.control.read("diagnostics.json"),
            h.control.read("failed_record.json"),
            h.control.read("failed_windows.json"),
        )
        assert not h.owner.snapshot.fenced, h.owner.snapshot
        final = await h.control.command("finish")
        assert len(final["callbacks"]) == 1800
        assert len(h.promotions) == 2
        assert h.promotions[1][1].status.value == "promoted"
        assert (
            final["capture_overflows"]
            == final["render_overflows"]
            == final["reference_overflows"]
            == 0
        )
        assert final["discontinuities"] == final["drift_faults"] == 0
        assert (
            final["native"]["invalid_frames"]
            == final["native"]["invalid_timing"]
            == final["native"]["fatal_status_bits"]
            == 0
        )
        for index, measurement in enumerate(measurements, 1):
            late_old_epoch_render_count = sum(
                stamp - final["driver_offset_ns"] > measurement["native_fence_ns"]
                and sample == 100 + index
                for stamp, _, sample in final["callbacks"]
            )
            assert late_old_epoch_render_count == 0
        assert h.owner._failure is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["status", "overflow"])
async def test_native_fault_switches_off_and_preserves_draft_once(
    monkeypatch, tmp_path, kind
):
    from textual.widgets import Switch

    async with mounted_native_process(monkeypatch, tmp_path) as h:
        await h.control.command("produce", count=100)
        await h.control.wait(lambda result: len(result["dsp"]) >= 5)
        await h.control.command("speak")
        await eventually(lambda: h.owner._draft_revisions)
        fault = await h.control.command("fault", kind=kind)
        if kind == "status":
            assert fault["native"]["fatal_status_bits"] == 4
        else:
            assert fault["capture_overflows"] > 0
        await eventually(
            lambda: not h.console.query_one("#console-hands-free-switch", Switch).value
        )
        composer = h.console.query_one("#console-native-composer")
        await eventually(lambda: "Dinner request" in composer.draft_text())
        await h.owner.close()
        await h.owner.wait_effects_closed()
        assert composer.draft_text().count("Dinner request") == 1
        assert h.owner.snapshot.fenced
        assert not h.promotions
