"""The in-process fixture must never change actual worker containment."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
import select
import signal
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from Tests.workspace_worker_test_support import run_in_process_workspace_worker
from tldw_chatbook.Tools import workspace_tool_worker
from tldw_chatbook.Tools.workspace_tool_executor import WorkspaceToolExecutor
from tldw_chatbook.Tools.workspace_tool_protocol import WorkspaceToolResponse


@pytest.mark.skipif(os.name != "posix", reason="POSIX CPU limit proof")
@pytest.mark.parametrize("backend", ["local", "bundle"])
def test_in_process_dispatch_preserves_cpu_limits_in_disposable_child(
    tmp_path, backend
):
    (tmp_path / "note.txt").write_text("real worker bytes", encoding="utf-8")
    request = WorkspaceToolExecutor(tmp_path)._build_request(
        "fs_read", {"path": "note.txt"}, intent="read"
    )
    source = """
import io, json, resource, sys
from Tests.workspace_worker_test_support import run_in_process_workspace_worker
from tldw_chatbook.Tools import remote_worker_bundle, workspace_tool_worker
module = workspace_tool_worker if sys.argv[1] == "local" else remote_worker_bundle
worker = module.run_workspace_worker
arm, disarm = module.arm_watchdog, module.disarm_watchdog
calls = []
def observe(frame, event, _arg):
    if event == "call" and frame.f_code is worker.__code__:
        calls.append(frame.f_globals is not worker.__globals__)
before = resource.getrlimit(resource.RLIMIT_CPU)
stdout = io.BytesIO()
sys.setprofile(observe)
try:
    code = run_in_process_workspace_worker(
        io.BytesIO(sys.stdin.buffer.read()), stdout, io.BytesIO(), worker=worker
    )
finally:
    sys.setprofile(None)
after = resource.getrlimit(resource.RLIMIT_CPU)
assert calls == [True], calls
assert module.arm_watchdog is arm and module.disarm_watchdog is disarm
assert before == after, (before, after)
print(json.dumps({"code": code, "frames": stdout.getvalue().decode()}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", source, backend],
        input=request.to_bytes(),
        capture_output=True,
        cwd=Path(__file__).resolve().parents[2],
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    result = json.loads(completed.stdout)
    assert result["code"] == 0
    frames = [
        WorkspaceToolResponse.from_bytes(line.encode())
        for line in result["frames"].splitlines()
    ]
    assert [frame.outcome for frame in frames] == ["admitted", "success"]
    expected_digest = hashlib.sha256(b"real worker bytes").hexdigest()
    assert (
        frames[-1].result
        == f"1\treal worker bytes\nsha256: {expected_digest}\nsize: 17"
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX fork containment proof")
def test_in_process_dispatch_keeps_real_watchdog_in_threads_and_fork(
    tmp_path, monkeypatch
):
    import resource

    original = workspace_tool_worker.run_workspace_worker
    arm = workspace_tool_worker.arm_watchdog
    disarm = workspace_tool_worker.disarm_watchdog
    request = WorkspaceToolExecutor(tmp_path)._build_request(
        "stat_path", {"path": "."}, intent="read"
    )
    observed = []

    def dispatch(stdout, decoded, started, *, bundle_sha256):
        frame = inspect.currentframe().f_back
        assert frame.f_code is original.__code__
        assert frame.f_globals is not original.__globals__
        assert frame.f_globals["_run_admitted_exchange"] is dispatch
        assert bundle_sha256 == "fixture-bundle"
        with ThreadPoolExecutor(max_workers=1) as executor:
            observed.append(
                executor.submit(
                    lambda: (
                        workspace_tool_worker.arm_watchdog is arm,
                        workspace_tool_worker.disarm_watchdog is disarm,
                    )
                ).result(timeout=5)
            )
        reader, writer = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(reader)
            try:
                before = resource.getrlimit(resource.RLIMIT_CPU)
                workspace_tool_worker.arm_watchdog(5, [])
                after = resource.getrlimit(resource.RLIMIT_CPU)
                workspace_tool_worker.disarm_watchdog()
                os.write(writer, json.dumps([before, after]).encode())
            finally:
                os.close(writer)
                os._exit(0)
        os.close(writer)
        try:
            ready, _, _ = select.select([reader], [], [], 5)
            if not ready:
                os.kill(pid, signal.SIGKILL)
            data = os.read(reader, 4096) if ready else b""
        finally:
            os.close(reader)
        _, status = os.waitpid(pid, 0)
        assert ready, "forked worker did not report its process limits"
        assert os.waitstatus_to_exitcode(status) == 0
        before, after = json.loads(data)
        assert after == [10, 10]
        assert after != before
        return 17

    monkeypatch.setattr(workspace_tool_worker, "_run_admitted_exchange", dispatch)
    assert (
        run_in_process_workspace_worker(
            io.BytesIO(request.to_bytes()),
            io.BytesIO(),
            io.BytesIO(),
            bundle_sha256="fixture-bundle",
        )
        == 17
    )
    assert observed == [(True, True)]
    assert workspace_tool_worker.run_workspace_worker is original
    assert workspace_tool_worker.arm_watchdog is arm
    assert workspace_tool_worker.disarm_watchdog is disarm
