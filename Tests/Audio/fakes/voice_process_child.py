"""Executable no-device wrapper; factories enter only through direct Python calls."""

import asyncio
import ctypes
from dataclasses import replace
import importlib.abc
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


SIMULATED_TIME = [0.0]


class NoDevicesOrModels(importlib.abc.MetaPathFinder):
    def __init__(self, *, native=False):
        self.native = native

    def find_spec(self, fullname, path=None, target=None):
        if self.native and fullname.split(".")[0] == "tldw_voice_aec":
            return None
        if fullname.split(".")[0] in {
            "sounddevice",
            "tldw_voice_aec",
            "torch",
            "transformers",
            "faster_whisper",
            "mlx",
            "nemo",
        }:
            raise AssertionError("unexpected device or model import")


def forbidden_network(*args, **kwargs):
    raise AssertionError("unexpected network")


def install_native_mode(directory):
    """Private executable instrumentation, never selected by release IPC/config."""
    from Tests.Audio.fakes.native_duplex_helpers import (
        CallbackObservation,
        driver_library,
        driver_clock_offset,
        emit_callback,
        install_fake_portaudio,
        load_native,
    )

    install_fake_portaudio()
    os.environ["TLDW_NATIVE_DUPLEX_ROOT"] = str(
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    native = load_native()
    from tldw_chatbook.Audio import parakeet_voice_worker, voice_transcription
    from tldw_chatbook.Audio import voice_preprocessor, voice_process_entry
    from tldw_chatbook.Audio.voice_process_lifetime import ChildLease
    from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome
    from tldw_chatbook.Audio.rolling_transcript import TranscriptRevision

    class Stt:
        cleanup_outcome = AttemptCleanupOutcome.CLEAN

        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

    async def prepare_stt(*args, **kwargs):
        return voice_transcription._UNPREPARED_STREAMING_CANDIDATE, False

    class Transcript:
        def __init__(self, turn_id, publish):
            self.turn_id, self.publish = turn_id, publish
            self.revision = None
            self.sequences = set()

        def append_admitted_frame(self, frame):
            self.sequences.add(frame.sequence)
            revision = 1 if self.revision is None else self.revision.revision_id + 1
            self.revision = TranscriptRevision(
                self.turn_id,
                revision,
                "",
                "Dinner request" + " correction" * (revision - 1),
                frame.ended_ns,
                "live",
            )
            self.publish(self.revision)

        async def seal_through(self, sequence):
            if sequence not in self.sequences:
                raise ValueError("transcript does not own admitted sequence")
            return self.revision

        async def close(self):
            pass

    parakeet_voice_worker.LocalVoiceSttProcess = Stt
    voice_transcription._prepare_streaming_candidate = prepare_stt
    voice_preprocessor.create_webrtc_vad = lambda **kwargs: lambda _: False
    directory = Path(directory)
    library = directory / ("driver.dylib" if sys.platform == "darwin" else "driver.so")
    driver = driver_library(library, release_gil=True)
    driver_offset = driver_clock_offset(driver)
    driver_uncertainty = driver.clock_uncertainty_ns
    original_prepare = voice_process_entry._ProductionSession.prepare
    original_start = voice_process_entry._ProductionSession.start
    original_renew = ChildLease.renew
    state = dict(
        lease_count=0,
        armed=None,
        session=None,
        producer=None,
        latest=None,
        captures=[],
        dsp=[],
        fences=[],
        admissions=[],
        events=[],
        drained=[],
        terminal_order=None,
        boundary_speech=None,
        terminal_receipts=[],
    )
    first = ctypes.c_uint64()
    observations = (CallbackObservation * 2400)()

    def write(name, value):
        temporary = directory / (name + ".tmp")
        temporary.write_text(json.dumps(value))
        temporary.replace(directory / name)

    diagnostics = []

    def diagnose(error):
        locations = []
        tb = error.__traceback__
        while tb:
            locations.append(
                (
                    tb.tb_frame.f_code.co_filename,
                    tb.tb_frame.f_code.co_name,
                    tb.tb_lineno,
                )
            )
            tb = tb.tb_next
        diagnostics.append(
            (type(error).__name__, getattr(error, "code", None), locations)
        )
        del diagnostics[:-8]
        write("diagnostics.json", diagnostics)

    original_bind = voice_process_entry._ProductionSession.bind_pipe

    def bind(self, pipe, fatal):
        original_consume = pipe.reader._consume

        def consume(record):
            try:
                result = original_consume(record)
                header = record.header
                if header.get("op") in {"terminal_claim", "terminal_result"}:
                    state["terminal_receipts"].append(
                        [header["op"], header["turn_id"], header.get("disposition")]
                    )
                    del state["terminal_receipts"][:-4]
                return result
            except Exception as error:
                diagnose(error)
                write("failed_record.json", dict(record.header))
                write(
                    "failed_windows.json",
                    dict(
                        input_turns=pipe.input._turns,
                        output_turns=pipe.output._turns,
                        windows={
                            lane: dict(
                                turn=window.key.turn_id,
                                lane=window.key.lane,
                                ack_sequence=window._ack_sequence,
                                outstanding=window.outstanding,
                            )
                            for lane, window in pipe._windows.items()
                        },
                    ),
                )
                raise

        pipe.reader._consume = consume

        return original_bind(self, pipe, fatal)

    voice_process_entry._ProductionSession.bind_pipe = bind

    def renewal(self):
        renewed = original_renew(self)
        if renewed:
            state["lease_count"] += 1
            if state["armed"] is not None:
                token = state["armed"]
                state["armed"] = None
                asyncio.get_running_loop().call_later(
                    0.12, lambda: asyncio.create_task(speak())
                )
                write(
                    "lease.json",
                    dict(
                        token=token,
                        pid=os.getpid(),
                        received_ns=time.monotonic_ns(),
                        deadline=self.deadline,
                        render_occupancy=state[
                            "session"
                        ].core._transport._stream.bridge.snapshot()["render_occupancy"],
                    ),
                )
        return renewed

    ChildLease.renew = renewal

    async def prepare(self):
        write("startup.json", dict(stage="prepare", time_ns=time.monotonic_ns()))
        try:
            result = await original_prepare(self)
        except Exception as error:
            diagnose(error)
            raise
        state["session"] = self
        core = self.core
        assert isinstance(core._preprocessor._aec, native.AecProcessor)
        assert self.settings["vad_preroll_ms"] == 240
        core._transcript_factory = Transcript
        transport = core._transport
        original_ack = transport.acknowledge_capture
        original_fence = transport.fence_output
        original_process = core._preprocessor.process_capture
        original_pop = transport.pop_capture
        original_submit = core.coordinator.submit

        async def submit(event):
            from tldw_chatbook.Audio.voice_turn_coordinator import (
                AttemptPlaybackTerminal,
            )

            order = state["terminal_order"]
            if order is None or not isinstance(event, AttemptPlaybackTerminal):
                return await original_submit(event)
            state["terminal_order"] = None
            before_turn = core.coordinator.snapshot.turn_id
            if order == "terminal_first":
                await original_submit(event)
            # Synthetic post-AEC admission with the real default 240 ms preroll
            # policy. The independent native pump continues actual AEC above.
            admitted = []
            latest = state["latest"]
            silent = replace(
                latest,
                sequence=latest.sequence + 1,
                started_ns=event.render_boundary_ns,
                ended_ns=event.render_boundary_ns + 10_000_000,
                assistant_rendering=False,
                pcm16=bytes(960),
            )
            positive = replace(
                silent,
                sequence=silent.sequence + 1,
                started_ns=silent.ended_ns,
                ended_ns=silent.ended_ns + 10_000_000,
                pcm16=b"\x02\x00" * 480,
            )
            normalizer = voice_preprocessor.VoicePreprocessor(
                aec=None,
                vad=lambda frame: frame.sequence == positive.sequence,
                on_admitted_frame=admitted.append,
            )
            await normalizer.process_capture(silent, assistant_rendering=False)
            await normalizer.process_capture(positive, assistant_rendering=False)
            for frame in admitted:
                await core._accept_admitted_frame(frame, assistant_rendering=False)
            if order == "speech_first":
                await original_submit(event)
            snap = core.coordinator.snapshot
            state["boundary_speech"] = dict(
                order=order,
                original_turn_id=before_turn,
                turn_id=snap.turn_id,
                pending_next_turn_id=snap.pending_next_turn_id,
                boundary_ns=event.render_boundary_ns,
                frames=[
                    (frame.started_ns, frame.ended_ns, frame.speech_started_ns)
                    for frame in admitted
                ],
                preroll_frames=normalizer._vad_preroll.maxlen,
            )

        core.coordinator.submit = submit

        def pop():
            frame = original_pop()
            if frame is not None:
                state["drained"].append(
                    (frame.discontinuity, bool(frame.delay_evidence.clock_drift))
                )
            return frame

        async def process(frame, **kwargs):
            await original_process(frame, **kwargs)
            state["latest"] = frame
            state["captures"].append(
                (frame.sequence, time.monotonic_ns(), frame.discontinuity)
            )

        def acknowledge(*args, **kwargs):
            result = original_ack(*args, **kwargs)
            state["dsp"].append(
                (time.monotonic_ns(), kwargs["dsp_ok"], kwargs["vad_ok"])
            )
            return result

        def fence():
            result = original_fence()
            if transport._stream is not None:
                state["fences"].append(
                    (
                        time.monotonic_ns(),
                        transport._stream.bridge.output_epoch,
                        driver.driver_monotonic_ns(),
                    )
                )
            return result

        transport.acknowledge_capture = acknowledge
        transport.pop_capture = pop
        transport.fence_output = fence
        core._preprocessor.process_capture = process
        core._persist_voice_event = lambda name, **kwargs: state["events"].append(name)
        return result

    async def speak():
        core = state["session"].core
        frame = replace(
            state["latest"],
            assistant_rendering=False,
            speech_started_ns=state["latest"].started_ns,
        )
        admitted = time.monotonic_ns()
        await core._accept_admitted_frame(frame, assistant_rendering=False)
        state["admissions"].append((admitted, core.coordinator.snapshot.turn_id))

    def snapshot():
        core = state["session"].core
        transport = core._transport
        bridge = transport._stream.bridge if transport._stream else None
        return dict(
            pid=os.getpid(),
            lease_count=state["lease_count"],
            driver_offset_ns=driver_offset,
            driver_uncertainty_ns=driver_uncertainty,
            driver_offset_after_ns=driver_clock_offset(driver),
            driver_uncertainty_after_ns=driver.clock_uncertainty_ns,
            native_file=native.__file__,
            native_abi=native.DUPLEX_ABI_VERSION,
            capacities=[
                transport.buffer_capacities.capture_frames,
                transport.buffer_capacities.render_frames,
                transport.buffer_capacities.render_reference_frames,
            ],
            preroll_ms=state["session"].settings["vad_preroll_ms"],
            native=transport.native_counters,
            bridge=bridge.snapshot() if bridge else {},
            aec=type(core._preprocessor._aec).__name__,
            captures=state["captures"],
            dsp=state["dsp"],
            fences=state["fences"],
            admissions=state["admissions"],
            events=state["events"],
            terminal_receipts=state["terminal_receipts"],
            retained_attempt_epochs=list(state["session"].effects.attempts),
            turn_id=core.coordinator.snapshot.turn_id,
            phase=core.coordinator.snapshot.state.value,
            capture_overflows=transport.capture_overflows,
            render_overflows=transport.render_overflows,
            reference_overflows=transport.render_reference_overflows,
            discontinuities=sum(item[0] for item in state["drained"]),
            drift_faults=sum(item[1] for item in state["drained"]),
            boundary_speech=state["boundary_speech"],
            callbacks=[
                (
                    item.callback_ns + driver_offset,
                    item.completed_ns + driver_offset,
                    item.output_sample,
                )
                for item in observations[
                    : driver.completed_observations(observations, len(observations))
                ]
            ],
        )

    async def commands():
        previous = None
        while True:
            try:
                command = json.loads((directory / "command.json").read_text())
            except FileNotFoundError:
                await asyncio.sleep(0.002)
                continue
            if command["id"] != previous:
                previous = command["id"]
                try:
                    op = command["op"]
                    if op == "produce":
                        bridge = state["session"].core._transport._stream.bridge
                        count = command["count"]
                        assert (
                            0 < count <= len(observations) and state["producer"] is None
                        )
                        state["producer"] = asyncio.create_task(
                            asyncio.to_thread(
                                driver.observed_paced_progress,
                                bridge.callback_address,
                                bridge.userdata_address,
                                count,
                                ctypes.byref(first),
                                observations,
                            )
                        )
                    elif op == "speak":
                        await speak()
                    elif op == "arm":
                        state["armed"] = command["id"]
                    elif op == "terminal_order":
                        assert command["order"] in {"terminal_first", "speech_first"}
                        state["terminal_order"] = command["order"]
                    elif op == "finish":
                        await state["producer"]
                    elif op == "fault":
                        await state["producer"]
                        bridge = state["session"].core._transport._stream.bridge
                        if command["kind"] == "overflow":
                            holder = driver_library(library)
                            holder.progress_with_gil_held(
                                bridge.callback_address, bridge.userdata_address, 65
                            )
                        else:
                            assert command["kind"] == "status"
                            emit_callback(driver, bridge, status=4)
                    elif op != "snapshot":
                        raise AssertionError("unknown test command")
                    write("response.json", dict(id=previous, result=snapshot()))
                except Exception as error:
                    import traceback

                    write(
                        "response.json",
                        dict(
                            id=previous, error=repr(error), trace=traceback.format_exc()
                        ),
                    )
            await asyncio.sleep(0.002)

    async def start(self, capture_live):
        write("startup.json", dict(stage="start", time_ns=time.monotonic_ns()))
        await original_start(self, capture_live)
        self._test_commands = asyncio.create_task(commands())
        write("started.json", snapshot())

    voice_process_entry._ProductionSession.prepare = prepare
    voice_process_entry._ProductionSession.start = start
    write("startup.json", dict(stage="installed", time_ns=time.monotonic_ns()))


class FakeSession:
    def __init__(self, mode, marker):
        self.mode = mode
        self.marker = Path(marker)
        self.descendant = None
        self.closed = False
        self.event("constructed")

    def event(self, name):
        with self.marker.open("a") as stream:
            stream.write(name + "\n")

    async def prepare(self):
        self.event("prepare")
        print("ordinary library stdout", flush=True)
        print("ordinary library stderr", file=sys.stderr, flush=True)
        if self.mode in {"slow_prepare", "lease_expire"}:
            await asyncio.sleep(60)
        return False

    async def start(self, capture_live):
        assert capture_live is True
        self.event("start")
        if self.mode in {
            "descendant",
            "crash",
            "orphan_after_close",
            "graceful_descendant",
        }:
            model_code = (
                "import sys; sys.stdin.buffer.read()"
                if self.mode == "graceful_descendant"
                else "import time; time.sleep(60)"
            )
            self.descendant = subprocess.Popen(
                [sys.executable, "-c", model_code],
                stdin=subprocess.PIPE,
                close_fds=True,
            )
            self.event("pid=" + str(self.descendant.pid))
        if self.mode == "crash":
            os._exit(23)

    def begin_close(self):
        assert not self.closed
        self.closed = True
        self.event("close")

        async def native():
            if self.mode == "native_hang":
                await asyncio.sleep(60)
            if self.mode == "native_late":
                await asyncio.sleep(0.05)
                SIMULATED_TIME[0] = 2.001
            self.event("native_closed")
            return True

        async def resources():
            from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome

            if self.mode == "orphan_after_close":
                return
            if self.mode == "cleanup_unknown":
                return None
            if self.mode == "cleanup_string":
                return "clean"
            if self.mode == "cleanup_raise":
                raise RuntimeError("private cleanup failure")
            if self.mode == "cleanup_detached":
                return AttemptCleanupOutcome.DETACHED
            if self.mode in {"cleanup_late", "cleanup_exit"}:
                await asyncio.sleep(0.05)
                if self.mode == "cleanup_exit":
                    os._exit(0)
                SIMULATED_TIME[0] = 6.001
            if self.mode == "cleanup_hang":
                await asyncio.sleep(60)
            if self.descendant is not None:
                if self.mode != "graceful_descendant":
                    self.descendant.terminate()
                self.descendant.stdin.close()
                await asyncio.to_thread(self.descendant.wait, 2)
                self.event("model_exit=" + str(self.descendant.returncode))
                return (
                    AttemptCleanupOutcome.CLEAN
                    if self.mode == "graceful_descendant"
                    and self.descendant.returncode == 0
                    else AttemptCleanupOutcome.FORCE_CLOSED
                )
            self.event("resources_closed")
            return AttemptCleanupOutcome.CLEAN

        return native(), resources()


if __name__ == "__main__":
    mode, marker = sys.argv[1:]

    def write_startup(value):
        temporary = Path(marker) / "startup.tmp"
        temporary.write_text(json.dumps(value))
        temporary.replace(Path(marker) / "startup.json")

    if mode == "native":
        write_startup(dict(stage="wrapper", time_ns=time.monotonic_ns()))
    sys.meta_path.insert(0, NoDevicesOrModels(native=mode == "native"))
    socket.socket.connect = forbidden_network
    socket.create_connection = forbidden_network
    try:
        from tldw_chatbook.Audio.voice_process_entry import (
            isolate_standard_streams,
            run_child,
        )

        if mode == "native":
            write_startup(dict(stage="entry_imported", time_ns=time.monotonic_ns()))
        read_fd, write_fd = isolate_standard_streams()
        if mode == "native":
            write_startup(dict(stage="streams_isolated", time_ns=time.monotonic_ns()))
            install_native_mode(marker)
    except Exception as error:
        if mode == "native":
            import traceback

            write_startup(
                dict(
                    stage="wrapper_failed",
                    error=type(error).__name__,
                    locations=[
                        (item.filename, item.name, item.lineno)
                        for item in traceback.extract_tb(error.__traceback__)[-16:]
                    ],
                )
            )
        raise
    # Models use spawn; importing this module under __mp_main__ must do nothing.
    if mode == "queued_startup":
        from tldw_chatbook.Audio.voice_process_io import PipeReader

        original_reader_start = PipeReader.start

        def start_with_queued_control(self):
            original_reader_start(self)
            # Force real reader admission before the constructor continues.
            assert self.wait_received(1, 2)

        PipeReader.start = start_with_queued_control

    factory = (
        None
        if mode in {"release", "native"}
        else lambda bootstrap: FakeSession(mode, marker)
    )
    kwargs = {}
    if mode == "lease_expire":
        kwargs["clock"] = lambda: time.monotonic() * 20
    if mode in {"native_late", "cleanup_late"}:
        kwargs["clock"] = lambda: SIMULATED_TIME[0]
    if mode != "native":
        kwargs["native_abi_reader"] = lambda: 999 if mode == "wrong_native" else 1
    if mode == "late_hello":
        now = [0.0]
        kwargs["clock"] = lambda: now[0]

        def late_abi():
            now[0] = 5.0
            return 1

        kwargs["native_abi_reader"] = late_abi
    result = asyncio.run(
        run_child(read_fd, write_fd, session_factory=factory, **kwargs)
    )
    if mode == "queued_startup":
        raise SystemExit(result)
