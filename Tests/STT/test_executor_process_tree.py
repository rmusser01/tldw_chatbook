from __future__ import annotations

import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from Tests.STT.executor_test_support import (
    containment_crashed_leader_with_term_ignoring_descendant,
    containment_descendant,
    containment_probe,
)
from tldw_chatbook.STT.executor_process_tree import (
    ExecutorProcessTree,
    ProcessContainmentError,
    WorkerContainmentIdentity,
    _WindowsJobApi,
)


class _RecordingEvent:
    def __init__(self, calls: list[object]) -> None:
        self.calls = calls
        self.was_set = False

    def set(self) -> None:
        self.calls.append("admit")
        self.was_set = True


class _FakeProcess:
    def __init__(self, calls: list[object], *, dies: bool = True) -> None:
        self.pid = 12345
        self._alive = True
        self._dies = dies
        self.calls = calls

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.calls.append("terminate_process")
        if self._dies:
            self._alive = False

    def kill(self) -> None:
        self.calls.append("kill_process")
        if self._dies:
            self._alive = False

    def join(self, timeout: float | None = None) -> None:
        self.calls.append(("join", timeout))


class _FakeWindowsApi:
    def __init__(
        self,
        calls: list[object],
        *,
        assign_error: bool = False,
        job_exits: bool = True,
    ) -> None:
        self.calls = calls
        self.assign_error = assign_error
        self.job_exits = job_exits

    def create_kill_on_close_job(self) -> int:
        self.calls.append("create_job")
        return 99

    def assign_process(self, job_handle: int, pid: int) -> None:
        self.calls.append(("assign", job_handle, pid))
        if self.assign_error:
            raise OSError("assignment failed")

    def terminate_job(self, job_handle: int) -> None:
        self.calls.append(("terminate_job", job_handle))

    def wait_for_job_empty(self, job_handle: int, timeout: float) -> bool:
        self.calls.append(("wait_job", job_handle, timeout))
        return self.job_exits

    def close_handle(self, job_handle: int) -> None:
        self.calls.append(("close_job", job_handle))


def _receive(connection: object, timeout: float = 10.0) -> tuple[str, object]:
    assert connection.poll(timeout)  # type: ignore[attr-defined]
    return connection.recv()  # type: ignore[attr-defined,no-any-return]


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_posix_worker_enters_new_session_and_waits_for_parent_admission() -> None:
    if os.name != "posix":
        pytest.skip("POSIX containment evidence")
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    admission = context.Event()
    process = context.Process(target=containment_probe, args=(send, admission))
    process.start()
    tree: ExecutorProcessTree | None = None
    try:
        kind, identity = _receive(receive)
        assert kind == "identity"
        assert type(identity) is WorkerContainmentIdentity
        assert identity.pid == process.pid
        assert identity.process_group_id == process.pid
        assert receive.poll(0.1) is False

        tree = ExecutorProcessTree(process, admission, identity)
        tree.admit()

        assert _receive(receive) == ("admitted", True)
        process.join(10.0)
        assert process.is_alive() is False
    finally:
        if tree is not None and process.is_alive():
            tree.terminate_tree(term_timeout=1.0, kill_timeout=1.0)
        elif process.is_alive():
            process.terminate()
            process.join(5.0)


def test_windows_job_assignment_happens_before_admission() -> None:
    calls: list[object] = []
    process = _FakeProcess(calls)
    admission = _RecordingEvent(calls)
    api = _FakeWindowsApi(calls)
    identity = WorkerContainmentIdentity(pid=process.pid, process_group_id=None)

    tree = ExecutorProcessTree(
        process,
        admission,
        identity,
        platform_name="nt",
        windows_api=api,
    )
    tree.admit()

    assert calls == ["create_job", ("assign", 99, process.pid), "admit"]
    assert tree.admitted is True
    assert _WindowsJobApi.KILL_ON_JOB_CLOSE == 0x00002000


def test_failed_windows_assignment_never_admits_and_reaps_worker() -> None:
    calls: list[object] = []
    process = _FakeProcess(calls)
    admission = _RecordingEvent(calls)
    api = _FakeWindowsApi(calls, assign_error=True)
    identity = WorkerContainmentIdentity(pid=process.pid, process_group_id=None)
    tree = ExecutorProcessTree(
        process,
        admission,
        identity,
        platform_name="nt",
        windows_api=api,
    )

    with pytest.raises(ProcessContainmentError):
        tree.admit()

    assert admission.was_set is False
    assert "terminate_process" in calls
    assert ("close_job", 99) in calls
    assert process.is_alive() is False


def test_windows_dead_leader_still_terminates_and_proves_empty_job() -> None:
    calls: list[object] = []
    process = _FakeProcess(calls)
    admission = _RecordingEvent(calls)
    api = _FakeWindowsApi(calls)
    identity = WorkerContainmentIdentity(pid=process.pid, process_group_id=None)
    tree = ExecutorProcessTree(
        process,
        admission,
        identity,
        platform_name="nt",
        windows_api=api,
    )
    tree.admit()
    process._alive = False
    calls.clear()

    assert tree.terminate_tree(term_timeout=0.2, kill_timeout=0.3) is True
    assert calls == [
        ("terminate_job", 99),
        ("wait_job", 99, 0.2),
        ("join", 0.2),
        ("close_job", 99),
    ]


def test_unproven_tree_death_quarantines_containment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    process = _FakeProcess(calls, dies=False)
    admission = _RecordingEvent(calls)
    identity = WorkerContainmentIdentity(pid=process.pid, process_group_id=process.pid)
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    tree = ExecutorProcessTree(
        process,
        admission,
        identity,
        platform_name="posix",
    )

    assert tree.terminate_tree(term_timeout=0.0, kill_timeout=0.0) is False
    assert tree.quarantined is True
    assert tree.admitted is False
    assert (process.pid, signal.SIGTERM) in calls
    assert (process.pid, signal.SIGKILL) in calls


def test_posix_force_stop_removes_worker_and_descendant_before_scratch_cleanup(
    tmp_path: Path,
) -> None:
    if os.name != "posix":
        pytest.skip("POSIX containment evidence")
    scratch = tmp_path / "generation-scratch"
    scratch.mkdir(mode=0o700)
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    admission = context.Event()
    process = context.Process(
        target=containment_descendant,
        args=(send, admission, str(scratch)),
    )
    process.start()
    tree: ExecutorProcessTree | None = None
    child_pid: int | None = None
    try:
        kind, identity = _receive(receive)
        assert kind == "identity"
        assert type(identity) is WorkerContainmentIdentity
        tree = ExecutorProcessTree(process, admission, identity)
        tree.admit()
        child_kind, raw_child_pid = _receive(receive)
        assert child_kind == "child"
        child_pid = int(raw_child_pid)
        assert (scratch / "worker-admitted").is_file()

        assert tree.terminate_tree(term_timeout=2.0, kill_timeout=2.0) is True
        deadline = time.monotonic() + 5.0
        while _pid_exists(child_pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert process.is_alive() is False
        assert _pid_exists(child_pid) is False

        shutil.rmtree(scratch)
        assert scratch.exists() is False
    finally:
        if tree is not None and process.is_alive():
            tree.terminate_tree(term_timeout=1.0, kill_timeout=1.0)
        elif process.is_alive():
            process.terminate()
            process.join(5.0)
        if scratch.exists() and (child_pid is None or not _pid_exists(child_pid)):
            shutil.rmtree(scratch)


def test_posix_crashed_leader_still_reaps_term_ignoring_descendant(
    tmp_path: Path,
) -> None:
    if os.name != "posix":
        pytest.skip("POSIX containment evidence")
    scratch = tmp_path / "crashed-generation-scratch"
    scratch.mkdir(mode=0o700)
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    admission = context.Event()
    process = context.Process(
        target=containment_crashed_leader_with_term_ignoring_descendant,
        args=(send, admission, str(scratch)),
    )
    process.start()
    tree: ExecutorProcessTree | None = None
    identity: WorkerContainmentIdentity | None = None
    child_pid: int | None = None
    try:
        kind, identity = _receive(receive)
        assert kind == "identity"
        assert type(identity) is WorkerContainmentIdentity
        tree = ExecutorProcessTree(process, admission, identity)
        tree.admit()
        child_kind, raw_child_pid = _receive(receive)
        assert child_kind == "child"
        child_pid = int(raw_child_pid)
        process.join(10.0)
        assert process.is_alive() is False
        assert _pid_exists(child_pid) is True

        assert tree.terminate_tree(term_timeout=0.1, kill_timeout=2.0) is True
        assert _pid_exists(child_pid) is False

        shutil.rmtree(scratch)
        assert scratch.exists() is False
    finally:
        if identity is not None and child_pid is not None and _pid_exists(child_pid):
            try:
                os.killpg(identity.process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.join(2.0)
        if scratch.exists() and (child_pid is None or not _pid_exists(child_pid)):
            shutil.rmtree(scratch)


def test_module_import_does_not_touch_windows_runtime_on_posix() -> None:
    if os.name != "posix":
        pytest.skip("POSIX import boundary")
    script = """
import ctypes

def fail(*_args, **_kwargs):
    raise AssertionError("WinDLL loaded during module import")

ctypes.WinDLL = fail
import tldw_chatbook.STT.executor_process_tree
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
