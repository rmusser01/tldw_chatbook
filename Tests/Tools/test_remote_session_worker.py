"""Laptop ``RemoteSessionWorker`` against the real bootstrap/loader/bundle.

No ssh: the ``spawn`` seam ignores the ssh argv and runs the fixed
bootstrap locally (``python -I -c <bootstrap>``), so the real stage-1
loader, bundle and fork-server ``serve`` loop answer every frame.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tldw_chatbook.Tools import remote_session_worker as worker_module
from tldw_chatbook.Tools.build_remote_worker_bundle import expected_bundle_stamp, loader_payload
from tldw_chatbook.Tools.remote_binding_locator import parse_remote_locator
from tldw_chatbook.Tools.remote_session_frames import HEADER, HELLO
from tldw_chatbook.Tools.remote_session_worker import RemoteSessionWorker, SessionStartError
from tldw_chatbook.Tools.remote_workspace_executor import (
    _bundle_payload,
    _encode_request,
    _session_request,
    bootstrap_source,
)
from tldw_chatbook.Tools.remote_workspace_transport import (
    RemoteWorkspaceTransport,
    SshMasterManager,
    TransportFailureKind,
)
from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain

_LOC = parse_remote_locator("ssh://ops@build-box.internal:2222/srv/work/workspace")
_HELLO_BODY = json.dumps({"max_children": 4, "idle_s": 30}).encode()


def HELLO_HEADER_PREFIX_BYTES() -> bytes:  # noqa: N802 - name fixed by the brief
    """The frame header the laptop's HELLO must start with."""
    return HEADER.pack(len(_HELLO_BODY), 0, HELLO)


def read_request(root: Path, path: str) -> bytes:
    return _encode_request(
        _session_request({"op": "fs_read", "path": path}, capture_directory_chain(root))
    )


def grep_request(root: Path, pattern: str, budget: int = 2) -> bytes:
    return _encode_request(
        _session_request(
            {"op": "fs_grep", "pattern": pattern, "budget": budget},
            capture_directory_chain(root),
        )
    )


def _local_argv() -> list[str]:
    return [sys.executable, "-I", "-c", bootstrap_source(len(loader_payload()))]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.txt").write_text("alpha\n")
    (root / "big.txt").write_text("a" * 60000 + "b\n")
    return root


@pytest.fixture
def worker_factory(monkeypatch):
    workers: list[RemoteSessionWorker] = []

    def make(
        *,
        spawn_argv: list[str] | None = None,
        expected_stamp: str | None = None,
        env: dict[str, str] | None = None,
        cache: bool = False,
        grace: float = 1.0,
    ):
        if expected_stamp is not None:
            monkeypatch.setattr(worker_module, "expected_bundle_stamp", lambda _data: expected_stamp)
        spawns: list[subprocess.Popen[bytes]] = []

        def spawn(ssh_argv: list[str]) -> subprocess.Popen[bytes]:
            assert ssh_argv[0] == "ssh"  # the real argv is built, then ignored
            proc = subprocess.Popen(
                spawn_argv or _local_argv(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            spawns.append(proc)
            return proc

        worker = RemoteSessionWorker(
            _LOC,
            transport=RemoteWorkspaceTransport(
                SshMasterManager(enabled=False), grace_seconds=grace
            ),
            python=sys.executable,
            max_children=4,
            idle_s=30,
            cache=cache,
            spawn=spawn,
        )
        workers.append(worker)
        return worker, spawns

    yield make
    for worker in workers:
        worker.close()


def test_many_calls_one_spawn(worker_factory, workspace):
    worker, spawns = worker_factory()
    worker.start()
    for _ in range(20):
        result = worker.call(read_request(workspace, "a.txt"), budget=10)
        assert result.failure is None and result.admitted
        assert json.loads(result.response)["outcome"] == "success"
    assert len(spawns) == 1


def test_concurrent_callers(worker_factory, workspace):
    worker, _ = worker_factory()
    worker.start()
    with ThreadPoolExecutor(8) as pool:
        results = list(
            pool.map(lambda _: worker.call(read_request(workspace, "a.txt"), budget=10), range(32))
        )
    assert all(r.failure is None and r.admitted for r in results)
    assert all("alpha" in json.loads(r.response)["result"] for r in results)
    assert worker.idle_since is not None


def test_watchdog_timeout_maps_to_op_timeout(worker_factory, workspace):
    worker, _ = worker_factory()
    worker.start()
    result = worker.call(grep_request(workspace, "(a+)+$"), budget=2)
    assert result.admitted and result.failure.kind is TransportFailureKind.OP_TIMEOUT
    assert worker.alive  # one op's death never takes the session down
    assert worker.call(read_request(workspace, "a.txt"), budget=10).failure is None


def test_stamp_mismatch_is_protocol_start_error(worker_factory):
    worker, _ = worker_factory(expected_stamp="0" * 64)
    with pytest.raises(SessionStartError) as err:
        worker.start()
    assert err.value.transport is False
    assert not worker.alive


def test_unreachable_is_transport_start_error(worker_factory):
    worker, _ = worker_factory(spawn_argv=["sh", "-c", "exit 255"])
    with pytest.raises(SessionStartError) as err:
        worker.start()
    assert err.value.transport is True
    assert err.value.failure.kind is TransportFailureKind.UNREACHABLE


def test_interpreter_missing_is_transport_start_error(worker_factory):
    worker, _ = worker_factory(spawn_argv=["sh", "-c", "exit 127"])
    with pytest.raises(SessionStartError) as err:
        worker.start()
    assert err.value.transport is True
    assert err.value.failure.kind is TransportFailureKind.INTERPRETER_MISSING


def test_loader_crash_is_protocol_start_error(worker_factory):
    worker, _ = worker_factory(spawn_argv=["sh", "-c", "head -c 10 >/dev/null; exit 3"])
    with pytest.raises(SessionStartError) as err:
        worker.start()
    assert err.value.transport is False
    assert err.value.failure.kind is TransportFailureKind.WORKER_FAILED_TO_START


def test_stdout_noise_is_transport_start_error(worker_factory):
    worker, _ = worker_factory(
        spawn_argv=["sh", "-c", "head -c 6000 /dev/zero | tr '\\0' x; sleep 30"]
    )
    started = time.monotonic()
    with pytest.raises(SessionStartError) as err:
        worker.start()
    assert err.value.transport is True
    assert err.value.failure.kind is TransportFailureKind.STDOUT_NOISE
    assert time.monotonic() - started < 10


def test_banner_before_magic_is_skipped(worker_factory, workspace):
    fake_host = f"echo 'Welcome to build-box'; exec {sys.executable} \"$@\""
    worker, _ = worker_factory(spawn_argv=["sh", "-c", fake_host, "sh", *_local_argv()[1:]])
    worker.start()
    assert worker.call(read_request(workspace, "a.txt"), budget=10).failure is None


#: A fake host that enforces the loader's LOCKSTEP contract: after the
#: header and after the bundle it checks that NOTHING else is readable
#: yet, then answers NEED/READY like the real loader, and finally checks
#: the next bytes are a HELLO frame.
_LOCKSTEP_HOST = textwrap.dedent(
    """
    import os, select, sys
    loader_len, bundle_hash, stamp, hello_prefix = int(sys.argv[1]), sys.argv[2], sys.argv[3], bytes.fromhex(sys.argv[4])
    magic = b"TLDW-REMOTE-0001"
    def read_exact(n):
        data = b""
        while len(data) < n:
            chunk = os.read(0, n - len(data))
            if not chunk:
                sys.exit(8)
            data += chunk
        return data
    def quiet():
        return not select.select([0], [], [], 0.3)[0]
    read_exact(loader_len)
    header = b""
    while not header.endswith(b"\\n"):
        header += read_exact(1)
    if not quiet():
        sys.exit(9)
    os.write(1, magic + b"NEED " + bundle_hash.encode() + b"\\n")
    read_exact(int.from_bytes(read_exact(4), "big"))
    if not quiet():
        sys.exit(9)
    os.write(1, magic + b"READY " + stamp.encode() + b"\\n")
    if read_exact(len(hello_prefix)) != hello_prefix:
        sys.exit(10)
    while os.read(0, 65536):
        pass
    """
)


def test_lockstep_sends_nothing_before_need_or_ready(worker_factory):
    artifact, compressed, _ = _bundle_payload()
    worker, spawns = worker_factory(
        spawn_argv=[
            sys.executable,
            "-c",
            _LOCKSTEP_HOST,
            str(len(loader_payload())),
            hashlib.sha256(compressed).hexdigest(),
            expected_bundle_stamp(artifact),
            HELLO_HEADER_PREFIX_BYTES().hex(),
        ]
    )
    worker.start()  # the fake host exits 8/9/10 (-> SessionStartError) on any violation
    time.sleep(0.2)
    assert spawns[0].poll() is None  # HELLO prefix matched; host still serving


def test_cache_hit_goes_straight_to_ready(worker_factory, workspace, tmp_path):
    runtime = tmp_path / "run"
    runtime.mkdir(mode=0o700)
    env = {**os.environ, "XDG_RUNTIME_DIR": str(runtime)}
    first, _ = worker_factory(env=env, cache=True)
    first.start()  # miss: NEED + bundle, then the loader caches it
    first.close()
    second, _ = worker_factory(env=env, cache=True)
    second.start()  # hit: READY first, no bundle sent
    assert second.call(read_request(workspace, "a.txt"), budget=10).failure is None


def test_hung_parent_kills_session(worker_factory, workspace):
    worker, spawns = worker_factory()
    worker.start()
    os.kill(spawns[0].pid, signal.SIGSTOP)
    started = time.monotonic()
    result = worker.call(read_request(workspace, "a.txt"), budget=1)
    assert result.failure is not None and not worker.alive
    assert time.monotonic() - started < 10


def test_session_death_fails_inflight_and_marks_dead(worker_factory, workspace):
    worker, spawns = worker_factory()
    worker.start()
    fut = ThreadPoolExecutor(1).submit(worker.call, grep_request(workspace, "(a+)+$", 5), budget=30)
    time.sleep(0.5)
    assert worker.idle_since is None  # a request is in flight
    spawns[0].kill()
    result = fut.result(10)
    assert result.failure is not None and not worker.alive
    after = worker.call(read_request(workspace, "a.txt"), budget=10)
    assert after.failure is not None and not after.admitted


def test_close_is_idempotent_and_never_raises(worker_factory):
    worker, spawns = worker_factory()
    worker.start()
    worker.close()
    worker.close()
    assert not worker.alive and spawns[0].poll() is not None


def test_malformed_status_is_session_death(worker_factory, workspace):
    """A STATUS body that fails to decode must kill the session, not the reader."""
    fake = textwrap.dedent(
        """
        import os, sys
        magic = b"TLDW-REMOTE-0001"
        os.read(0, 1 << 20)
        os.write(1, magic + b"READY " + sys.argv[1].encode() + b"\\n")
        import struct, time
        time.sleep(0.5)
        os.write(1, struct.pack(">IIB", 5, 1, 17) + b"nope!")
        time.sleep(30)
        """
    )
    artifact, _, _ = _bundle_payload()
    worker, _ = worker_factory(
        spawn_argv=[sys.executable, "-c", fake, expected_bundle_stamp(artifact)]
    )
    worker.start()
    result = worker.call(read_request(workspace, "a.txt"), budget=10)
    assert result.failure is not None and not worker.alive
