"""Cross-process Windows profile writer custody and framed data limits."""

import os
import sqlite3
import struct
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows named-pipe writer")


def test_disconnected_client_cannot_replace_owner_during_physical_work():
    from tldw_chatbook.DB import _profile_writer_windows as pipe

    name = rf"\\.\pipe\tldw-writer-test-{uuid4().hex}"
    # The physical owner stays inside its command until stdin releases it.
    source = """
import sys
from tldw_chatbook.DB import _profile_writer_windows as pipe
with pipe.server(sys.argv[1]) as owner:
    print('elected', flush=True)
    pipe.accept(owner, 10)
    assert pipe.read_frame(owner, 10) == b'command'
    print('executing', flush=True)
    assert sys.stdin.readline().strip() == 'finish'
    pipe.disconnect(owner)
    pipe.accept(owner, 10)
    assert pipe.read_frame(owner, 10) == b'next'
    pipe.write_frame(owner, b'committed', 10)
    assert pipe.read_frame(owner, 10) == b'received'
"""
    worker = subprocess.Popen(
        [sys.executable, "-c", source, name],
        cwd=Path(__file__).resolve().parents[2],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert worker.stdout.readline().strip() == "elected"
        with pipe.client(name) as first:
            pipe.write_frame(first, b"command", 10)
            assert worker.stdout.readline().strip() == "executing"
        with pytest.raises(OSError):
            with pipe.server(name):
                pytest.fail("second server held the same election")
        with pytest.raises(OSError):
            with pipe.client(name):
                pytest.fail("second client entered before physical completion")
        worker.stdin.write("finish\n")
        worker.stdin.flush()
        pipe.wait_available(name, 10)
        with pipe.client(name) as second:
            pipe.write_frame(second, b"next", 10)
            assert pipe.read_frame(second, 10) == b"committed"
            pipe.write_frame(second, b"received", 10)
        assert worker.wait(timeout=10) == 0, worker.stderr.read()
        with pipe.server(name):
            pass
    finally:
        if worker.poll() is None:
            worker.kill()
        worker.communicate(timeout=10)


def test_pipe_rejects_oversized_declared_frame_before_body_and_bounds_idle_reader():
    import _winapi
    from tldw_chatbook.DB import _profile_writer_windows as pipe

    name = rf"\\.\pipe\tldw-writer-test-{uuid4().hex}"
    with pipe.server(name) as owner, pipe.client(name) as client:
        pipe.accept(owner, 1)
        write, _ = _winapi.WriteFile(client, struct.pack("!I", 2**32 - 1), True)
        assert write.GetOverlappedResult(True) == (4, 0)
        with pytest.raises(ValueError, match="frame"):
            pipe.read_frame(owner, 1)
        with pytest.raises(TimeoutError):
            pipe.read_frame(owner, .02)


def test_client_rejects_an_endpoint_with_the_default_public_read_dacl():
    import _winapi
    from tldw_chatbook.DB import _profile_writer_windows as pipe

    name = rf"\\.\pipe\tldw-writer-test-{uuid4().hex}"
    foreign = _winapi.CreateNamedPipe(
        name, _winapi.PIPE_ACCESS_DUPLEX | _winapi.FILE_FLAG_FIRST_PIPE_INSTANCE
        | _winapi.FILE_FLAG_OVERLAPPED, 0, 1, 4096, 4096, 0, 0,
    )
    try:
        with pytest.raises(PermissionError):
            with pipe.client(name):
                pytest.fail("sent a command to an endpoint with foreign rights")
    finally:
        _winapi.CloseHandle(foreign)


def test_two_clients_commit_to_one_profile_and_a_lost_commit_reply_is_not_replayed(monkeypatch):
    from tldw_chatbook import config
    from tldw_chatbook.DB import _profile_writer_windows as pipe
    from tldw_chatbook.DB import profile_writer as writer

    profile = config.get_user_data_dir()
    source = """
import sys
from pathlib import Path
from tldw_chatbook.DB.profile_writer import call_profile_writer
value = call_profile_writer(Path(sys.argv[1]), 'agent_runs.create_agent_definition',
    {'defn': {'name': sys.argv[2], 'description': 'check', 'instructions': 'check'}})
print(value, flush=True)
"""
    prefix = "writer-" + uuid4().hex
    names = [prefix + suffix for suffix in ("-one", "-two", "-lost", "-restarted")]

    def create(name):
        return writer.call_profile_writer(profile, "agent_runs.create_agent_definition", {
            "defn": {"name": name, "description": "check", "instructions": "check"},
        })

    children = []
    try:
        for name in names[:2]:
            children.append(subprocess.Popen(
                [sys.executable, "-c", source, str(profile), name],
                cwd=Path(__file__).resolve().parents[2],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ))
        receipts = []
        for child in children:
            out, err = child.communicate(timeout=50)
            assert child.returncode == 0, err
            receipts.append(out.strip())

        receive = pipe.read_frame

        def lose_reply(handle, timeout):
            receive(handle, timeout)  # Physical commit and resource retirement already happened.
            raise EOFError("connection lost before the caller observed the reply")

        with monkeypatch.context() as changes:
            changes.setattr(pipe, "read_frame", lose_reply)
            with pytest.raises(writer.ProfileWriterOutcomeUnknown):
                create(names[2])
        writer.drain_profile_writer(profile)
        restarted = create(names[3])
        with sqlite3.connect((profile / "agent_runs.db").as_uri() + "?mode=ro", uri=True) as db:
            rows = db.execute(
                "SELECT name, id FROM agent_definitions WHERE name IN (?, ?, ?, ?) ORDER BY name",
                names,
            ).fetchall()
        assert sorted(name for name, _ in rows) == sorted(names)
        assert [dict(rows)[name] for name in names[:2]] == receipts
        assert dict(rows)[names[3]] == restarted
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)
        writer.drain_profile_writer(profile)
