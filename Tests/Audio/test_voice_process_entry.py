"""Release bootstrap cannot enable audio or inherit ordinary stdio."""

import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import io

import pytest


ROOT = Path(__file__).resolve().parents[2]


def entry():
    name = "tldw_chatbook.Audio.voice_process_entry"
    assert importlib.util.find_spec(name), "private child entry missing"
    return importlib.import_module(name)


def test_entry_import_and_spawn_reimport_do_not_start_audio():
    entry()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy,sys; runpy.run_module('tldw_chatbook.Audio.voice_process_entry', run_name='__mp_main__'); assert not any(n in sys.modules for n in ('sounddevice','tldw_voice_aec','tldw_chatbook.app','tldw_chatbook.STT'))",
        ],
        cwd=ROOT,
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == b""


@pytest.mark.parametrize("args", [[], ["--enable"], ["--factory=Fake"]])
def test_release_entry_closed_stdin_refuses_without_output(args):
    entry()
    result = subprocess.run(
        [sys.executable, "-m", "tldw_chatbook.Audio.voice_process_entry", *args],
        cwd=ROOT,
        input=b"",
        capture_output=True,
        timeout=7,
    )
    assert result.returncode != 0
    assert result.stdout == result.stderr == b""


def test_isolated_protocol_descriptors_are_noninheritable():
    entry()
    source = "from tldw_chatbook.Audio.voice_process_entry import isolate_standard_streams; import os; r,w=isolate_standard_streams(); assert not os.get_inheritable(r) and not os.get_inheritable(w); print('library'); os.write(2,b'noise'); os.write(w,b'private'); os.close(r); os.close(w)"
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        input=b"",
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 0
    assert result.stdout == b"private"
    assert result.stderr == b""


@pytest.mark.parametrize(
    "field,value",
    [("root", "/wrong"), ("source", "0" * 64), ("native_abi", 999), ("version", 99)],
)
def test_real_entry_rejects_mismatched_bootstrap_before_factory(tmp_path, field, value):
    entry()
    from Tests.Chat.test_console_voice_process import bootstrap, console
    from tldw_chatbook.Audio.voice_process_protocol import Record, encode_record

    marker = tmp_path / "events"
    record = bootstrap(console())
    fields = dict(record.header)
    if field != "version":
        fields[field] = value
    encoded = encode_record(Record(fields), "parent_to_child")
    if field == "version":
        encoded = encoded.replace(b'"version":1', b'"version":9')
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "Tests.Audio.fakes.voice_process_child",
            "normal",
            str(marker),
        ],
        cwd=ROOT,
        input=encoded,
        capture_output=True,
        timeout=7,
    )
    assert result.stdout == b""
    assert not marker.exists()


@pytest.mark.parametrize("mode", ["slow_prepare", "lease_expire"])
def test_real_child_stops_stalled_preparation_on_eof_or_lease_expiry(tmp_path, mode):
    entry()
    from Tests.Chat.test_console_voice_process import bootstrap, console
    from tldw_chatbook.Audio.voice_process_protocol import encode_record, read_record
    import time

    marker = tmp_path / "events"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "Tests.Audio.fakes.voice_process_child",
            mode,
            str(marker),
        ],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        process.stdin.write(encode_record(bootstrap(console()), "parent_to_child"))
        process.stdin.flush()
        deadline = time.monotonic() + 3
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        if mode == "lease_expire":
            # The direct injected clock expires the child's own lease, while
            # stdin stays open. EOF cannot accidentally satisfy this branch.
            process.wait(timeout=4)
        output, _ = process.communicate(timeout=7)
        assert "prepare" in marker.read_text().splitlines()
        assert "start" not in marker.read_text().splitlines()
        assert "close" in marker.read_text().splitlines()
        assert (
            read_record(io.BytesIO(output).read, "child_to_parent").header["op"]
            == "hello"
        )
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()


def test_native_abi_is_measured_before_factory(tmp_path):
    entry()
    from Tests.Chat.test_console_voice_process import bootstrap, console
    from tldw_chatbook.Audio.voice_process_protocol import encode_record

    marker = tmp_path / "events"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "Tests.Audio.fakes.voice_process_child",
            "wrong_native",
            str(marker),
        ],
        cwd=ROOT,
        input=encode_record(bootstrap(console()), "parent_to_child"),
        capture_output=True,
        timeout=7,
    )
    assert not marker.exists()
    assert result.stdout == b""
    assert result.stderr == b""


@pytest.mark.asyncio
@pytest.mark.parametrize("queued_op", ["lease", "close"])
async def test_queued_startup_control_cannot_displace_bootstrap(tmp_path, queued_op):
    import asyncio

    from Tests.Chat.test_console_voice_process import bootstrap, console
    from tldw_chatbook.Audio.voice_process_protocol import (
        Record,
        encode_record,
        read_record,
    )

    marker = tmp_path / "events"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "Tests.Audio.fakes.voice_process_child",
            "queued_startup",
            str(marker),
        ],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    replies = []

    def control(op, sequence, **fields):
        return Record(
            dict(
                version=1,
                generation=7,
                request_id="a" * 32,
                sequence=sequence,
                op=op,
                **fields,
            )
        )

    def send(record):
        process.stdin.write(encode_record(record, "parent_to_child"))
        process.stdin.flush()

    async def receive_until(*ops):
        async with asyncio.timeout(3):
            while not set(ops) <= {record.header["op"] for record in replies}:
                replies.append(
                    await asyncio.to_thread(
                        read_record, process.stdout.read, "child_to_parent"
                    )
                )

    try:
        # Already on the wire when the child switches from bootstrap to its
        # normal reader. The wrapper gates actual reader progress, not a sleep.
        queued = control(
            queued_op, 2, **({"reason": "teardown"} if queued_op == "close" else {})
        )
        process.stdin.write(
            encode_record(bootstrap(console()), "parent_to_child")
            + encode_record(queued, "parent_to_child")
        )
        process.stdin.flush()
        if queued_op == "lease":
            await receive_until("stt_ready")
            # Control sequence 2 requires bootstrap sequence 1 to be accounted
            # for; startup records must not manufacture ordinary credit.
            send(control("start", 2, capture_live=True))
            await receive_until("session_ready")
            assert not any(record.header["op"] == "credit" for record in replies)
            send(control("close", 3, reason="teardown"))
        await receive_until("closed", "resources_closed")
        assert await asyncio.to_thread(process.wait, 3) == 0
        events = marker.read_text().splitlines()
        assert ("start" in events) is (queued_op == "lease")
        assert [
            record.header["outcome"]
            for record in replies
            if record.header["op"] in {"closed", "resources_closed"}
        ] == ["clean", "clean"]
        assert not any(
            record.header["op"] == "fault"
            and record.header["code"] in {"protocol_invalid", "transport_failed"}
            for record in replies
        )
    finally:
        if process.poll() is None:
            process.kill()
        await asyncio.to_thread(process.wait, 2)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()


def test_queued_startup_wrapper_preserves_bootstrap_refusal_exit(tmp_path):
    from Tests.Chat.test_console_voice_process import bootstrap, console
    from tldw_chatbook.Audio.voice_process_protocol import Record, encode_record

    marker = tmp_path / "events"
    rejected = Record(dict(bootstrap(console()).header) | {"root": "/wrong"})
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "Tests.Audio.fakes.voice_process_child",
            "queued_startup",
            str(marker),
        ],
        cwd=ROOT,
        input=encode_record(rejected, "parent_to_child"),
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 2
    assert result.stdout == result.stderr == b""
    assert not marker.exists()


def test_late_identity_verification_cannot_revive_expired_handshake(tmp_path):
    entry()
    from Tests.Chat.test_console_voice_process import bootstrap, console
    from tldw_chatbook.Audio.voice_process_protocol import encode_record

    marker = tmp_path / "events"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "Tests.Audio.fakes.voice_process_child",
            "late_hello",
            str(marker),
        ],
        cwd=ROOT,
        input=encode_record(bootstrap(console()), "parent_to_child"),
        capture_output=True,
        timeout=7,
    )
    assert result.stderr == b""
    assert not marker.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage,expected",
    [
        ("malformed", "protocol_invalid"),
        ("truncated", "transport_failed"),
        ("eof", None),
        ("writer", "transport_failed"),
        ("writer_without_callback", "transport_failed"),
    ],
)
async def test_child_reports_pipe_integrity_without_erasing_cleanup(
    monkeypatch, damage, expected
):
    """Real pipes, with only entry's POSIX self-kill branch disabled (not Windows)."""
    import asyncio
    import os
    from types import SimpleNamespace

    from Tests.Chat.test_console_voice_process import bootstrap, console
    from tldw_chatbook.Audio import voice_process_lifetime as lifetime
    from tldw_chatbook.Audio.voice_process_protocol import (
        ProtocolError,
        Record,
        encode_record,
        read_record,
    )
    from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome

    module = entry()
    monkeypatch.setattr(
        module, "os", SimpleNamespace(name="nt", read=os.read, close=os.close)
    )
    pipes, replies, attempts, cleanup_facts = [], [], [], []
    real_pipe = lifetime.LifecyclePipe

    def capture_pipe(*args, **kwargs):
        pipe = real_pipe(*args, **kwargs)
        pipes.append(pipe)
        real_send = pipe.send

        def send(op, **fields):
            if op == "fault":
                attempts.append(fields["code"])
            return real_send(op, **fields)

        pipe.send = send
        return pipe

    monkeypatch.setattr(lifetime, "LifecyclePipe", capture_pipe)

    class Session:
        async def prepare(self):
            return False

        async def start(self, capture_live):
            assert capture_live

        def begin_close(self):
            async def native():
                cleanup_facts.append("native")
                return True

            async def resources():
                cleanup_facts.append("resources")
                return AttemptCleanupOutcome.CLEAN

            return native(), resources()

    child_read, parent_write = os.pipe()
    parent_read, child_write = os.pipe()

    def collect():
        try:
            while True:
                replies.append(
                    read_record(lambda n: os.read(parent_read, n), "child_to_parent")
                )
        except ProtocolError as error:
            assert error.code == "voice_transport_eof"
        finally:
            os.close(parent_read)

    output = asyncio.create_task(asyncio.to_thread(collect))
    child = asyncio.create_task(
        module.run_child(
            child_read,
            child_write,
            session_factory=lambda _: Session(),
            native_abi_reader=lambda: 1,
        )
    )

    async def wait_for(op):
        async with asyncio.timeout(3):
            while not any(record.header["op"] == op for record in replies):
                await asyncio.sleep(0.001)

    try:
        os.write(parent_write, encode_record(bootstrap(console()), "parent_to_child"))
        await wait_for("stt_ready")
        os.write(
            parent_write,
            encode_record(
                Record(
                    dict(
                        version=1,
                        generation=7,
                        request_id="a" * 32,
                        sequence=2,
                        op="start",
                        capture_live=True,
                    )
                ),
                "parent_to_child",
            ),
        )
        await wait_for("session_ready")
        if damage.startswith("writer"):

            def broken_write(_data):
                raise OSError("test write failure")

            pipes[0].writer._write = broken_write
            if damage == "writer_without_callback":
                pipes[0].writer._on_fault = lambda _error: None
        if damage == "malformed":
            os.write(parent_write, b"\0" * 8)
        elif damage == "truncated":
            os.write(parent_write, b"\0" * 3)
        os.close(parent_write)
        parent_write = None
        result = await asyncio.wait_for(asyncio.shield(child), 4)
        await asyncio.wait_for(asyncio.shield(output), 2)
        assert sorted(cleanup_facts) == ["native", "resources"]
        faults = [r.header["code"] for r in replies if r.header["op"] == "fault"]
        if not damage.startswith("writer"):
            assert [
                r.header["outcome"]
                for r in replies
                if r.header["op"] in {"closed", "resources_closed"}
            ] == ["clean", "clean"]
            assert faults == ([] if expected is None else [expected])
        else:
            assert pipes[0].writer.failure.code == "voice_transport_failed"
            assert attempts == [expected]
        assert (result == 0) is (expected is None)
    finally:
        if parent_write is not None:
            os.close(parent_write)
        await asyncio.wait_for(asyncio.shield(child), 4)
        await asyncio.wait_for(asyncio.shield(output), 2)
        for pipe in pipes:
            await asyncio.to_thread(pipe.reader.join, 1)
            await asyncio.to_thread(pipe.writer.join, 1)
            assert not pipe.reader.alive and not pipe.writer.alive
