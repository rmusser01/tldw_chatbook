"""Cross-process Windows profile writer custody and framed data limits."""

import os
import sqlite3
import struct
import subprocess
import sys
import time
from contextlib import closing
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
        over = pipe._MAGIC + struct.pack("!Q", pipe.MAX_MESSAGE_BYTES + 1)
        write, _ = _winapi.WriteFile(client, over, True)
        assert write.GetOverlappedResult(True) == (len(over), 0)
        with pytest.raises(ValueError, match="frame"):
            pipe.read_frame(owner, 1)
        with pytest.raises(TimeoutError):
            pipe.read_frame(owner, .02)


def test_large_writer_message_transfers_in_order_and_incomplete_stream_is_refused():
    import _winapi
    from concurrent.futures import ThreadPoolExecutor
    from tldw_chatbook.DB import _profile_writer_windows as pipe

    name = rf"\\.\pipe\tldw-writer-test-{uuid4().hex}"
    # An ordinary 13 MiB attachment already expands past 16 MiB on the JSON
    # wire. The whole logical document arrives before either peer can adopt it.
    body = b"suffix" + bytes(range(256)) * (17 * 4096) + b"end"
    with pipe.server(name) as owner, pipe.client(name) as client, ThreadPoolExecutor(1) as executor:
        pipe.accept(owner, 1)
        sending = executor.submit(pipe.write_frame, client, body, 5)
        assert pipe.read_frame(owner, 5) == body
        sending.result(timeout=5)
    with pipe.server(name) as owner:
        with pipe.client(name) as client:
            pipe.accept(owner, 1)
            # One complete JSON object arrives, then EOF without the terminal
            # frame. A service must never dispatch this apparently valid prefix.
            prefix = pipe._MAGIC + struct.pack("!QI", 2, 2) + b"{}"
            write, _ = _winapi.WriteFile(client, prefix, True)
            assert write.GetOverlappedResult(True) == (len(prefix), 0)
        with pytest.raises((OSError, EOFError)):
            pipe.read_frame(owner, 1)


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
    writer.call_profile_writer(profile, "agent_runs.open", {})
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
        with closing(sqlite3.connect((profile / "agent_runs.db").as_uri() + "?mode=ro", uri=True)) as db:
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


def _wait_for_owned_server(profile, *, terminate=False):
    import _winapi
    import ctypes
    from ctypes import wintypes
    from tldw_chatbook.DB import _profile_writer_windows as pipe
    from tldw_chatbook.DB.profile_writer import _selection

    name, _ = _selection(profile)
    end = time.monotonic() + 10
    while True:
        try:
            with pipe.client(name) as handle:
                pid = wintypes.ULONG()
                assert pipe._server_pid(handle, ctypes.byref(pid))
                access = _winapi.SYNCHRONIZE | (1 if terminate else 0)
                # Retain the process behind this connected, authenticated test
                # endpoint. Termination never selects a PID from a process list.
                process = _winapi.OpenProcess(access, False, pid.value)
                return pid.value, process
        except OSError as error:
            if error.winerror not in (2, 231) or time.monotonic() >= end:
                raise
            time.sleep(.025)


def test_last_host_retires_writer_and_host_recovers_it_for_a_contained_client(tmp_path):
    import _winapi
    from tldw_chatbook import config
    from tldw_chatbook.DB import profile_writer as writer

    profile = config.get_user_data_dir()
    host = writer.start_profile_writer(profile)
    child_source = """
import sys
from pathlib import Path
from tldw_chatbook.DB.profile_writer import start_profile_writer
host = start_profile_writer(Path(sys.argv[1]))
Path(sys.argv[2]).write_text('registered')
assert sys.stdin.readline().strip() == 'close'
host.close()
"""
    ready = tmp_path / "registered.txt"
    diagnostics = tmp_path / "second-host.log"
    log = diagnostics.open("w", encoding="utf-8")
    second = subprocess.Popen(
        [sys.executable, "-c", child_source, str(profile), str(ready)], cwd=Path(__file__).resolve().parents[2],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log, text=True,
    )
    owned_processes = []
    try:
        deadline = time.monotonic() + 20
        while not ready.exists() and second.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        assert ready.exists(), diagnostics.read_text(encoding="utf-8", errors="replace")[-3000:]
        before_pid, before = _wait_for_owned_server(profile, terminate=True)
        owned_processes.append(before)
        host.close()
        assert _winapi.WaitForSingleObject(before, 0) == _winapi.WAIT_TIMEOUT
        _winapi.TerminateProcess(before, 77)
        assert _winapi.WaitForSingleObject(before, 5000) == _winapi.WAIT_OBJECT_0
        # No foreground caller can start the replacement: it must appear from
        # the surviving already-registered host's monitor.
        after_pid, after = _wait_for_owned_server(profile)
        owned_processes.append(after)
        assert after_pid != before_pid
        source = """
import sys
from pathlib import Path
from tldw_chatbook.DB.profile_writer import use_host_profile_writer, call_profile_writer
use_host_profile_writer()
print(call_profile_writer(Path(sys.argv[1]), 'agent_runs.open', {}))
"""
        contained = subprocess.run(
            [sys.executable, "-c", source, str(profile)], cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=30,
        )
        assert contained.returncode == 0, contained.stderr
        assert contained.stdout.strip() in {"True", "False"}
        second.communicate("close\n", timeout=15)
        assert second.returncode == 0, diagnostics.read_text(encoding="utf-8", errors="replace")[-3000:]
        assert _winapi.WaitForSingleObject(after, 0) == _winapi.WAIT_OBJECT_0
    finally:
        if second.poll() is None:
            second.kill()
        second.communicate(timeout=5)
        log.close()
        host.close()
        for process in owned_processes:
            _winapi.CloseHandle(process)


def _overlap_native_recovery():
    """Run with its own bootstrap root, independent of pytest's live startup lease."""
    from concurrent.futures import ThreadPoolExecutor

    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission
    from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
    from tldw_chatbook.Backup_Recovery.control_records import UNBOUND_NAMESPACE, admission_authority
    from tldw_chatbook.DB import profile_writer as writer
    from tldw_chatbook.DB.profile_writer_selection import writer_file_identity

    profile = config.get_user_data_dir()
    database = profile / "agent_runs.db"
    host = writer.start_profile_writer(profile)
    try:
        writer.call_profile_writer(profile, "agent_runs.open", {})
        authority = admission_authority(bootstrap.default_bootstrap_root())
        storage_admission._shutdown()  # This isolated coordinator has no app-owned repositories.
        payload = {
            "operation": "agent_runs.create_agent_definition",
            "arguments": {"defn": {"name": "maintenance-check", "instructions": "inspect"}},
            "database": list(writer_file_identity(database)),
        }
        with ThreadPoolExecutor(1) as executor:
            with authority.maintenance((UNBOUND_NAMESPACE,), 2):
                # A client may retain its own outer core operation while waiting
                # for the reply. Waiting behind maintenance in the service
                # would deadlock that caller's native drain.
                refused = executor.submit(writer._exchange, profile, "call", payload)
                with pytest.raises(bootstrap.RecoveryRequired, match="pause"):
                    refused.result(timeout=3)
        with ThreadPoolExecutor(1) as executor, closing(sqlite3.connect(database)) as blocker:
            blocker.execute("BEGIN IMMEDIATE")
            # Send directly over the real transport. A caller-side lease must
            # not be mistaken for evidence that the separate writer is enrolled.
            pending = executor.submit(writer._exchange, profile, "call", payload)
            try:
                until = time.monotonic() + 4
                while time.monotonic() < until:
                    try:
                        with authority.maintenance((UNBOUND_NAMESPACE,), .1):
                            pass
                    except AdmissionTimeout:
                        break
                    time.sleep(.02)
                else:
                    raise AssertionError("separate writer never held the native recovery lease")
                assert not pending.done()
            finally:
                blocker.rollback()
            identifier = pending.result(timeout=15)
        with closing(sqlite3.connect(":memory:")) as snapshot:
            with authority.maintenance((UNBOUND_NAMESPACE,), 2), closing(sqlite3.connect(database)) as store:
                store.backup(snapshot)
            writer.call_profile_writer(profile, "agent_runs.soft_delete_agent_definition", {
                "definition_id": identifier,
            })
            with authority.maintenance((UNBOUND_NAMESPACE,), 2), closing(sqlite3.connect(database)) as store:
                assert store.execute("SELECT deleted FROM agent_definitions WHERE id=?", (identifier,)).fetchone() == (1,)
                snapshot.backup(store)
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as observer:
            assert observer.execute("SELECT deleted FROM agent_definitions WHERE id=?", (identifier,)).fetchone() == (0,)
    finally:
        host.close()


def test_native_recovery_drains_writer_and_restores_committed_database(tmp_path):
    from tldw_chatbook.Utils.platform_files import os as files

    # No test process in the parent pytest session may hold this bootstrap
    # root open: real recovery must encounter the service's own leases alone.
    isolated = tmp_path / "writer-home"
    files.mkdir(isolated, 0o700)
    config_dir = isolated / "config"
    files.mkdir(config_dir, 0o700)
    env = dict(os.environ, HOME=str(isolated), USERPROFILE=str(isolated),
               XDG_CONFIG_HOME=str(config_dir), TLDW_CONFIG_PATH=str(config_dir / "config.toml"))
    source = "from Tests.DB.test_profile_writer_transport import _overlap_native_recovery; _overlap_native_recovery()"
    result = subprocess.run(
        [sys.executable, "-c", source], cwd=Path(__file__).resolve().parents[2],
        env=env, capture_output=True, text=True, timeout=50,
    )
    assert result.returncode == 0, result.stderr
