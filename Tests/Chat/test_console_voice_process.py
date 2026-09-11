"""Real subprocess supervision with fake local models/devices only."""

import asyncio
import importlib
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

import pytest


ROOT = Path(__file__).resolve().parents[2]


def console():
    name = "tldw_chatbook.Chat.console_voice_process"
    assert importlib.util.find_spec(name), "parent process facade missing"
    return importlib.import_module(name)


def bootstrap(module):
    return module.bootstrap_record(
        generation=7,
        request_id="a" * 32,
        stt_provider="faster-whisper",
        stt_model=None,
        language="en",
        response_eagerness_ms=700,
        aec_enabled=True,
        vad_aggressiveness=2,
        vad_preroll_ms=300,
    )


def fake_popen(mode, marker, launches):
    def launch(args, **kwargs):
        assert args == [sys.executable, "-m", "tldw_chatbook.Audio.voice_process_entry"]
        assert kwargs["close_fds"] is True
        assert kwargs.get("shell", False) is False
        assert kwargs["start_new_session"] is True
        launches.append(True)
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "Tests.Audio.fakes.voice_process_child",
                mode,
                str(marker),
            ],
            **kwargs,
        )

    return launch


async def ready_true():
    return True


@pytest.mark.parametrize(
    "options",
    [
        {"stt_device": "cpu", "stt_compute_type": "float32", "stt_precision": "fp32"},
        {"stt_device": None, "stt_compute_type": None, "stt_precision": None},
    ],
)
def test_bootstrap_closed_stt_options_round_trip(options):
    from tldw_chatbook.Audio.voice_process_protocol import encode_record, read_record
    import io

    original = bootstrap(console())
    fields = {
        k: v
        for k, v in original.header.items()
        if k not in {"version", "op", "sequence", "root", "source", "native_abi"}
    }
    record = console().bootstrap_record(**(fields | options))
    decoded = read_record(
        io.BytesIO(encode_record(record, "parent_to_child")).read, "parent_to_child"
    )
    assert {key: decoded.header[key] for key in options} == options


@pytest.mark.parametrize(
    "options",
    [
        {"stt_device": "../../private"},
        {"stt_compute_type": True},
        {"stt_precision": "secret"},
    ],
)
def test_bootstrap_refuses_malformed_stt_options(options):
    from tldw_chatbook.Audio.voice_process_protocol import ProtocolError

    original = bootstrap(console())
    fields = {
        k: v
        for k, v in original.header.items()
        if k not in {"version", "op", "sequence", "root", "source", "native_abi"}
    }
    with pytest.raises(ProtocolError):
        console().bootstrap_record(**(fields | options))


async def cleanup(owner):
    try:
        await asyncio.wait_for(owner.close(), 12)
    finally:
        process = owner.process
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                await asyncio.to_thread(process.wait, 2)
            except subprocess.TimeoutExpired:
                pytest.fail("audio child survived final cleanup watchdog")


@pytest.mark.asyncio
async def test_preflight_rejection_never_launches(tmp_path):
    module = console()
    launches = []
    owner = module.ConsoleVoiceProcess(
        module.DeviceLease(), popen=fake_popen("normal", tmp_path / "events", launches)
    )

    async def no():
        return False

    with pytest.raises(module.VoiceProcessError):
        await owner.enter(bootstrap(module), preflight=no, current=lambda: True)
    assert not launches
    assert owner.process is None


@pytest.mark.asyncio
async def test_readiness_and_current_activation_precede_native_start(tmp_path):
    module = console()
    marker = tmp_path / "events"
    lease = module.DeviceLease()
    owner = module.ConsoleVoiceProcess(lease, popen=fake_popen("normal", marker, []))
    try:
        await asyncio.wait_for(
            owner.enter(bootstrap(module), preflight=ready_true, current=lambda: True),
            8,
        )
        assert marker.read_text().splitlines()[:3] == [
            "constructed",
            "prepare",
            "start",
        ]
        assert owner.ready
        evidence = await asyncio.wait_for(owner.close(), 10)
        assert evidence.native_closed and evidence.child_exited and evidence.tree_absent
        assert not lease.quarantined
    finally:
        await cleanup(owner)


@pytest.mark.asyncio
async def test_stale_activation_after_stt_prevents_native_start(tmp_path):
    module = console()
    marker = tmp_path / "events"
    owner = module.ConsoleVoiceProcess(
        module.DeviceLease(), popen=fake_popen("normal", marker, [])
    )

    def current():
        return not marker.exists()

    try:
        with pytest.raises(module.VoiceProcessError):
            await asyncio.wait_for(
                owner.enter(bootstrap(module), preflight=ready_true, current=current), 8
            )
        assert "start" not in marker.read_text().splitlines()
    finally:
        await cleanup(owner)


@pytest.mark.asyncio
async def test_cancelling_startup_retains_enter_and_close_observers(tmp_path):
    module = console()
    marker = tmp_path / "events"
    owner = module.ConsoleVoiceProcess(
        module.DeviceLease(), popen=fake_popen("slow_prepare", marker, [])
    )
    task = asyncio.create_task(
        owner.enter(bootstrap(module), preflight=ready_true, current=lambda: True)
    )
    try:
        async with asyncio.timeout(5):
            while not marker.exists():
                await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        evidence = await asyncio.wait_for(owner.close(), 12)
        assert evidence.tree_absent and evidence.child_exited
        assert "start" not in marker.read_text().splitlines()
    finally:
        task.cancel()
        await cleanup(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,drop_pipe", [("crash", False), ("descendant", True)])
async def test_model_descendant_dies_after_owner_crash_or_parent_eof(
    tmp_path, mode, drop_pipe
):
    module = console()
    marker = tmp_path / "events"
    lease = module.DeviceLease()
    owner = module.ConsoleVoiceProcess(lease, popen=fake_popen(mode, marker, []))
    descendant = None
    try:
        try:
            await asyncio.wait_for(
                owner.enter(
                    bootstrap(module), preflight=ready_true, current=lambda: True
                ),
                8,
            )
        except module.VoiceProcessError:
            assert mode == "crash"
        descendant = int(
            next(
                line[4:]
                for line in marker.read_text().splitlines()
                if line.startswith("pid=")
            )
        )
        if drop_pipe:
            # Stop and observe the borrowed-fd writer before the test closes its
            # owner endpoint; no writer can race descriptor reuse afterward.
            owner.pipe.writer.close()
            await asyncio.to_thread(owner.pipe.writer.join, 1)
            assert not owner.pipe.writer.alive
            owner.process.stdin.close()
        async with asyncio.timeout(12):
            while owner.close_task is None or not owner.close_task.done():
                await asyncio.sleep(0.02)
        evidence = owner.close_task.result()
        assert evidence.child_exited and evidence.tree_absent
        with pytest.raises(ProcessLookupError):
            os.kill(descendant, 0)
        if mode == "crash":
            assert not evidence.native_closed and lease.quarantined
    finally:
        await cleanup(owner)
        if descendant is not None:
            try:
                os.kill(descendant, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.asyncio
async def test_release_composition_currently_refuses_after_verified_hello(tmp_path):
    module = console()
    owner = module.ConsoleVoiceProcess(
        module.DeviceLease(), popen=fake_popen("release", tmp_path / "events", [])
    )
    try:
        with pytest.raises(module.VoiceProcessError, match="startup_failed"):
            await asyncio.wait_for(
                owner.enter(
                    bootstrap(module), preflight=ready_true, current=lambda: True
                ),
                8,
            )
        assert owner.hello_verified and not owner.ready
    finally:
        await cleanup(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,native_closed",
    [("native_hang", False), ("cleanup_hang", True), ("native_late", False)],
)
async def test_native_close_receipt_is_independent_of_model_cleanup(
    tmp_path, mode, native_closed
):
    module = console()
    lease = module.DeviceLease()
    terminations = []

    class ObservedTree(module.OwnedProcessTree):
        def terminate(self, *, force=False):
            terminations.append((force, owner.process.poll()))
            super().terminate(force=force)

    owner = module.ConsoleVoiceProcess(
        lease,
        popen=fake_popen(mode, tmp_path / "events", []),
        tree_factory=ObservedTree,
    )
    try:
        await asyncio.wait_for(
            owner.enter(bootstrap(module), preflight=ready_true, current=lambda: True),
            8,
        )
        close_caller = asyncio.create_task(owner.close())
        await asyncio.sleep(0.02)
        close_caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_caller
        evidence = await asyncio.wait_for(owner.close(), 12)
        assert evidence.native_closed is native_closed
        assert evidence.tree_absent and evidence.child_exited
        if mode == "cleanup_hang":
            assert (False, None) in terminations
            assert owner.process.returncode == -signal.SIGTERM
            assert "native_closed" in (tmp_path / "events").read_text().splitlines()
            assert (
                "resources_closed" not in (tmp_path / "events").read_text().splitlines()
            )
            assert evidence.forced_termination
        assert lease.quarantined
        with pytest.raises(module.VoiceProcessError, match="shutdown_unconfirmed"):
            lease.acquire(object())
    finally:
        await cleanup(owner)


@pytest.mark.asyncio
async def test_forced_descendant_cleanup_quarantines_after_graceful_audio_exit(
    tmp_path,
):
    module = console()
    lease = module.DeviceLease()
    marker = tmp_path / "events"
    terminations = []

    class ObservedTree(module.OwnedProcessTree):
        def terminate(self, *, force=False):
            terminations.append(owner.process.poll())
            super().terminate(force=force)

    owner = module.ConsoleVoiceProcess(
        lease,
        popen=fake_popen("orphan_after_close", marker, []),
        tree_factory=ObservedTree,
    )
    descendant = None
    try:
        await asyncio.wait_for(
            owner.enter(bootstrap(module), preflight=ready_true, current=lambda: True),
            8,
        )
        descendant = int(
            next(
                line[4:]
                for line in marker.read_text().splitlines()
                if line.startswith("pid=")
            )
        )
        evidence = await asyncio.wait_for(owner.close(), 12)
        assert owner.process.returncode == 0
        assert 0 in terminations
        assert evidence.forced_termination
        assert evidence.native_closed and evidence.tree_absent and evidence.child_exited
        with pytest.raises(ProcessLookupError):
            os.kill(descendant, 0)
        assert lease.quarantined
        with pytest.raises(module.VoiceProcessError, match="shutdown_unconfirmed"):
            lease.acquire(object())
    finally:
        await cleanup(owner)
        if descendant is not None:
            try:
                os.kill(descendant, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.asyncio
async def test_external_kill_after_native_receipt_keeps_device_quarantined(tmp_path):
    module = console()
    lease = module.DeviceLease()
    owner = module.ConsoleVoiceProcess(
        lease, popen=fake_popen("cleanup_hang", tmp_path / "events", [])
    )
    caller = None
    try:
        await asyncio.wait_for(
            owner.enter(bootstrap(module), preflight=ready_true, current=lambda: True),
            8,
        )
        caller = asyncio.create_task(owner.close())
        async with asyncio.timeout(2):
            while not owner.native_closed:
                await asyncio.sleep(0.01)
        os.kill(owner.process.pid, signal.SIGKILL)
        evidence = await asyncio.wait_for(caller, 8)
        assert owner.process.returncode == -signal.SIGKILL
        assert evidence.forced_termination
        assert evidence.native_closed and evidence.child_exited and evidence.tree_absent
        assert lease.quarantined
        with pytest.raises(module.VoiceProcessError, match="shutdown_unconfirmed"):
            lease.acquire(object())
    finally:
        if caller is not None:
            caller.cancel()
        await cleanup(owner)


@pytest.mark.asyncio
async def test_child_exit_preserves_unread_native_close_receipt(monkeypatch, tmp_path):
    from tldw_chatbook.Audio import voice_process_lifetime as lifetime

    paused = threading.Event()
    resume = threading.Event()

    class PausedAfterReady(lifetime.PipeReader):
        def _wake(self):
            super()._wake()
            if threading.current_thread() is self._thread and self._received == 2:
                paused.set()
                assert resume.wait(5), "receipt reader watchdog expired"

    monkeypatch.setattr(lifetime, "PipeReader", PausedAfterReady)
    module = console()
    lease = module.DeviceLease()
    marker = tmp_path / "events"
    owner = module.ConsoleVoiceProcess(lease, popen=fake_popen("normal", marker, []))
    caller = None
    try:
        await asyncio.wait_for(
            owner.enter(bootstrap(module), preflight=ready_true, current=lambda: True),
            5,
        )
        assert paused.is_set()
        caller = asyncio.create_task(owner.close())
        async with asyncio.timeout(4):
            while owner.process.poll() is None:
                await asyncio.sleep(0.005)
        # Give the parent process-exit observer a turn before the actual pipe
        # thread can read the already-written native receipt and following EOF.
        await asyncio.sleep(0.05)
        resume.set()
        evidence = await asyncio.wait_for(caller, 3)
        assert owner.process.returncode == 0
        assert "native_closed" in marker.read_text().splitlines()
        assert evidence.child_exited and evidence.tree_absent and evidence.pipes_closed
        assert evidence.native_closed
        assert not evidence.forced_termination
        assert not lease.quarantined
        lease.acquire(object())
    finally:
        resume.set()
        if caller is not None:
            await asyncio.wait_for(asyncio.shield(caller), 12)
        await cleanup(owner)
        if owner.pipe is not None:
            await asyncio.to_thread(owner.pipe.reader.join, 1)
            assert not owner.pipe.reader.alive


@pytest.mark.asyncio
async def test_child_owned_forced_model_exit_quarantines_on_normal_audio_close(
    tmp_path,
):
    module = console()
    lease = module.DeviceLease()
    marker = tmp_path / "events"
    owner = module.ConsoleVoiceProcess(
        lease, popen=fake_popen("descendant", marker, [])
    )
    try:
        await asyncio.wait_for(
            owner.enter(bootstrap(module), preflight=ready_true, current=lambda: True),
            8,
        )
        evidence = await asyncio.wait_for(owner.close(), 12)
        assert owner.process.returncode == 0
        assert "model_exit=" + str(-signal.SIGTERM) in marker.read_text().splitlines()
        assert evidence.native_closed and evidence.tree_absent and evidence.child_exited
        assert lease.quarantined
        with pytest.raises(module.VoiceProcessError, match="shutdown_unconfirmed"):
            lease.acquire(object())
    finally:
        await cleanup(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "cleanup_unknown",
        "cleanup_string",
        "cleanup_raise",
        "cleanup_detached",
        "cleanup_late",
        "cleanup_exit",
    ],
)
async def test_uncertain_model_cleanup_cannot_become_clean_on_audio_exit_zero(
    tmp_path, mode
):
    module = console()
    lease = module.DeviceLease()
    marker = tmp_path / "events"
    owner = module.ConsoleVoiceProcess(lease, popen=fake_popen(mode, marker, []))
    try:
        await asyncio.wait_for(
            owner.enter(bootstrap(module), preflight=ready_true, current=lambda: True),
            8,
        )
        evidence = await asyncio.wait_for(owner.close(), 12)
        assert owner.process.returncode == 0
        assert "native_closed" in marker.read_text().splitlines()
        assert evidence.native_closed and evidence.tree_absent and evidence.child_exited
        assert lease.quarantined
        with pytest.raises(module.VoiceProcessError, match="shutdown_unconfirmed"):
            lease.acquire(object())
    finally:
        await cleanup(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["force_closed", "detached", "failed", "clean"])
async def test_duplicate_resource_receipt_cannot_upgrade_cleanup(outcome):
    from tldw_chatbook.Audio.voice_process_protocol import Record
    from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome

    module = console()
    owner = module.ConsoleVoiceProcess(module.DeviceLease())
    first = Record(
        dict(
            version=1,
            op="resources_closed",
            generation=1,
            request_id="a" * 32,
            sequence=1,
            outcome=outcome,
        )
    )
    try:
        owner._receive(first)
        owner._receive(Record(dict(first.header, sequence=2, outcome="clean")))
        assert owner.resources_outcome == (
            None if outcome == "failed" else AttemptCleanupOutcome(outcome)
        )
        assert owner.transport_failure_code == "voice_protocol_invalid"
        if outcome == "force_closed":
            assert owner._forced_termination
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_duplicate_native_receipt_cannot_upgrade_failed_close():
    from tldw_chatbook.Audio.voice_process_protocol import Record

    module = console()
    owner = module.ConsoleVoiceProcess(module.DeviceLease())
    first = Record(
        dict(
            version=1,
            op="closed",
            generation=1,
            request_id="a" * 32,
            sequence=1,
            outcome="failed",
        )
    )
    try:
        owner._receive(first)
        owner._receive(Record(dict(first.header, sequence=2, outcome="clean")))
        assert not owner.native_closed
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("resources_first", [False, True])
async def test_graceful_model_exit_and_both_receipt_orders_release_device(
    tmp_path, resources_first
):
    from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome

    module = console()
    lease = module.DeviceLease()
    marker = tmp_path / "events"
    owner = module.ConsoleVoiceProcess(
        lease, popen=fake_popen("graceful_descendant", marker, [])
    )
    receive = owner._receive
    held = []

    def reorder_receipts(record):
        if record.header["op"] == "closed" and resources_first:
            held.append(record)
            return
        receive(record)
        if record.header["op"] == "resources_closed" and resources_first:
            receive(held.pop())

    owner._receive = reorder_receipts
    try:
        await asyncio.wait_for(
            owner.enter(bootstrap(module), preflight=ready_true, current=lambda: True),
            8,
        )
        evidence = await asyncio.wait_for(owner.close(), 12)
        assert "model_exit=0" in marker.read_text().splitlines()
        assert owner.process.returncode == 0
        assert evidence.native_closed and evidence.tree_absent and evidence.child_exited
        assert evidence.resources_outcome is AttemptCleanupOutcome.CLEAN
        assert not evidence.forced_termination
        assert not lease.quarantined
        lease.acquire(object())
    finally:
        await cleanup(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "trailer,expected",
    [
        ("duplicate", "voice_protocol_invalid"),
        ("malformed", "voice_protocol_invalid"),
        ("truncated", "voice_transport_truncated"),
        ("fault_protocol_invalid", "voice_protocol_invalid"),
        ("fault_capacity_exceeded", "voice_capacity_exceeded"),
        ("fault_transport_failed", "voice_transport_failed"),
        ("eof", None),
    ],
)
async def test_parser_receipt_integrity_survives_queued_clean_receipts(
    trailer, expected
):
    from types import SimpleNamespace
    from tldw_chatbook.Audio.voice_process_lifetime import LifecyclePipe
    from tldw_chatbook.Audio.voice_process_protocol import Record, encode_record
    from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome

    module = console()
    lease = module.DeviceLease()
    owner = module.ConsoleVoiceProcess(lease)
    child_read, parent_write = os.pipe()
    parent_read, child_write = os.pipe()
    # Only the already-absent PID/tree facts are synthetic. Both private pipes,
    # framing, parser admission, queued custody and owner callbacks are real.
    owner.process = SimpleNamespace(
        poll=lambda: 0,
        returncode=0,
        stdin=os.fdopen(parent_write, "wb", buffering=0),
        stdout=os.fdopen(parent_read, "rb", buffering=0),
    )
    owner.tree = SimpleNamespace(attached=True, observe_absence=lambda: True)
    lease.acquire(owner)
    owner.pipe = LifecyclePipe(
        os.dup(parent_read),
        parent_write,
        generation=7,
        request_id="a" * 32,
        parent=True,
        consume=owner._receive,
        fault=owner._transport_fault,
    )

    def receipt(op):
        return encode_record(
            Record(
                dict(
                    version=1,
                    op=op,
                    generation=7,
                    request_id="a" * 32,
                    sequence=1,
                    outcome="clean",
                )
            ),
            "child_to_parent",
        )

    frames = receipt("closed") + receipt("resources_closed")
    if trailer == "duplicate":
        frames += receipt("resources_closed")
    elif trailer == "malformed":
        frames += b"\0" * 8
    elif trailer == "truncated":
        frames += receipt("resources_closed")[:3]
    elif trailer.startswith("fault_"):
        frames += encode_record(
            Record(
                dict(
                    version=1,
                    op="fault",
                    generation=7,
                    request_id="a" * 32,
                    sequence=1,
                    code=trailer.removeprefix("fault_"),
                )
            ),
            "child_to_parent",
        )
    try:
        assert os.write(child_write, frames) == len(frames)
        os.close(child_write)
        child_write = None
        # Intentionally hold the owner loop while the pipe reader admits the
        # clean receipts and then rejects the trailer before any callback runs.
        owner.pipe.reader.join(1)
        assert not owner.pipe.reader.alive
        assert owner.pipe.reader.failure.code == (
            "voice_transport_eof"
            if trailer.startswith("fault_")
            else expected or "voice_transport_eof"
        )
        evidence = await asyncio.wait_for(owner.close(), 2)
        assert evidence.native_closed
        assert evidence.resources_outcome is AttemptCleanupOutcome.CLEAN
        assert evidence.child_exited and evidence.tree_absent and evidence.pipes_closed
        assert not evidence.forced_termination
        assert evidence.transport_failure_code == expected
        if expected is None:
            assert not lease.quarantined
            lease.acquire(object())
        else:
            assert lease.quarantined
            with pytest.raises(module.VoiceProcessError, match="shutdown_unconfirmed"):
                lease.acquire(object())
    finally:
        if child_write is not None:
            os.close(child_write)
        await asyncio.wait_for(owner.close(), 2)
        os.close(child_read)
        owner.pipe.stop()
        assert await owner.pipe.join()
        owner.process.stdin.close()
        owner.process.stdout.close()


@pytest.mark.asyncio
async def test_missing_containment_never_sends_bootstrap_or_starts_factory(tmp_path):
    module = console()
    marker = tmp_path / "events"

    class RefusedTree:
        attached = False

        def __init__(self, pid):
            pass

        def attach(self):
            raise module.VoiceProcessError()

        def observe_absence(self):
            return False

        def terminate(self, **kwargs):
            pass

    owner = module.ConsoleVoiceProcess(
        module.DeviceLease(),
        popen=fake_popen("normal", marker, []),
        tree_factory=RefusedTree,
    )
    try:
        with pytest.raises(module.VoiceProcessError):
            await asyncio.wait_for(
                owner.enter(
                    bootstrap(module), preflight=ready_true, current=lambda: True
                ),
                5,
            )
        assert not marker.exists()
        evidence = await asyncio.wait_for(owner.close(), 12)
        assert evidence.child_exited
    finally:
        await cleanup(owner)


@pytest.mark.asyncio
async def test_failed_popen_releases_unopened_device_lease():
    module = console()
    lease = module.DeviceLease()

    def failed_launch(*args, **kwargs):
        raise OSError("private startup detail")

    owner = module.ConsoleVoiceProcess(lease, popen=failed_launch)
    with pytest.raises(module.VoiceProcessError):
        await owner.enter(bootstrap(module), preflight=ready_true, current=lambda: True)
    await owner.close()
    lease.acquire(object())
    assert not lease.quarantined


@pytest.mark.asyncio
async def test_cancellation_while_preflight_pending_never_launches(tmp_path):
    module = console()
    pending = asyncio.Event()
    launches = []
    owner = module.ConsoleVoiceProcess(
        module.DeviceLease(), popen=fake_popen("normal", tmp_path / "events", launches)
    )

    async def slow_preflight():
        await pending.wait()
        return True

    caller = asyncio.create_task(
        owner.enter(bootstrap(module), preflight=slow_preflight, current=lambda: True)
    )
    try:
        await asyncio.sleep(0)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        await owner.close()
        pending.set()
        with pytest.raises(module.VoiceProcessError):
            await asyncio.wait_for(owner.enter_task, 1)
        assert not launches
    finally:
        pending.set()
        await cleanup(owner)
