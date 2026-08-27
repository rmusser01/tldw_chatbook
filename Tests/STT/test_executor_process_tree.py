from __future__ import annotations

import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import tldw_chatbook.STT.executor_process_tree as process_tree_module
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


def _pid_has_exited(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        synchronize = 0x00100000
        error_invalid_parameter = 87
        wait_object_0 = 0
        wait_timeout = 258
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
        ]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        ctypes.set_last_error(0)
        handle = kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            if error == error_invalid_parameter:
                return True
            raise OSError(error, "OpenProcess could not prove PID exit")
        try:
            result = kernel32.WaitForSingleObject(handle, 0)
            if result == wait_object_0:
                return True
            if result == wait_timeout:
                return False
            error = ctypes.get_last_error()
            raise OSError(error, "WaitForSingleObject failed")
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _wait_for_pid_exit(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not _pid_has_exited(pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    return _pid_has_exited(pid)


def _finalize_native_tree(
    tree: ExecutorProcessTree | None,
    process: object,
    identity: WorkerContainmentIdentity | None,
    child_pid: int | None,
) -> bool:
    tree_proven = False
    if tree is not None:
        try:
            tree_proven = tree.terminate_tree(term_timeout=1.0, kill_timeout=1.0)
        except Exception:
            pass
    worker_alive = process.is_alive()  # type: ignore[attr-defined]
    child_alive = child_pid is not None and not _pid_has_exited(child_pid)
    if os.name == "nt":
        if tree is not None and not tree_proven:
            tree._close_job_handle()
        if worker_alive:
            try:
                process.terminate()  # type: ignore[attr-defined]
            except OSError:
                pass
            process.join(1.0)  # type: ignore[attr-defined]
            if process.is_alive():  # type: ignore[attr-defined]
                try:
                    process.kill()  # type: ignore[attr-defined]
                except OSError:
                    pass
    elif (
        (worker_alive or child_alive)
        and identity is not None
        and identity.process_group_id is not None
        and identity.process_group_id != os.getpgrp()
    ):
        try:
            os.killpg(identity.process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.join(2.0)  # type: ignore[attr-defined]
    return tree_proven


def test_windows_finalizer_uses_only_the_owned_process_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    raw_pid_terminations: list[int] = []
    process = _FakeProcess(calls)

    monkeypatch.setattr(sys.modules[__name__], "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(sys.modules[__name__], "_pid_has_exited", lambda _pid: False)
    monkeypatch.setattr(
        sys.modules[__name__],
        "_terminate_captured_windows_pid",
        raw_pid_terminations.append,
        raising=False,
    )

    _finalize_native_tree(None, process, None, child_pid=67890)

    assert "terminate_process" in calls
    assert raw_pid_terminations == []


@pytest.mark.skipif(os.name != "nt", reason="native Windows PID probe")
def test_windows_pid_probe_does_not_terminate_a_live_process() -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert _pid_has_exited(child.pid) is False
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(10.0)


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


def test_cleanup_serializes_late_windows_admission_and_closes_job_once() -> None:
    calls: list[object] = []
    terminate_entered = threading.Event()
    release_terminate = threading.Event()

    class BlockingWindowsApi(_FakeWindowsApi):
        def terminate_job(self, job_handle: int) -> None:
            calls.append(("terminate_job", job_handle))
            terminate_entered.set()
            assert release_terminate.wait(5.0)
            process._alive = False

    process = _FakeProcess(calls)
    admission = _RecordingEvent(calls)
    api = BlockingWindowsApi(calls)
    identity = WorkerContainmentIdentity(pid=process.pid, process_group_id=None)
    tree = ExecutorProcessTree(
        process,
        admission,
        identity,
        platform_name="nt",
        windows_api=api,
    )
    tree.admit()
    cleanup_results: list[bool] = []
    late_errors: list[BaseException] = []
    cleanup_started = threading.Event()

    def run_cleanup() -> None:
        cleanup_started.set()
        cleanup_results.append(tree.close())

    tree._lock.acquire()
    cleanup = threading.Thread(target=run_cleanup)
    cleanup.start()
    assert cleanup_started.wait(5.0)
    tree._lock.release()
    assert terminate_entered.wait(5.0)

    def late_admit() -> None:
        try:
            tree.admit()
        except BaseException as error:
            late_errors.append(error)

    late = threading.Thread(target=late_admit)
    late.start()
    release_terminate.set()
    cleanup.join(5.0)
    late.join(5.0)

    assert cleanup.is_alive() is False
    assert late.is_alive() is False
    assert cleanup_results == [True]
    assert len(late_errors) == 1
    assert isinstance(late_errors[0], ProcessContainmentError)
    assert tree.admitted is False
    assert tree._closed is True
    assert tree._job_handle == 0
    assert calls.count("create_job") == 1
    assert calls.count(("close_job", 99)) == 1
    assert calls.count("admit") == 1


def test_cleanup_exception_permanently_closes_late_admission() -> None:
    calls: list[object] = []
    process = _FakeProcess(calls)
    admission = _RecordingEvent(calls)
    identity = WorkerContainmentIdentity(pid=process.pid, process_group_id=process.pid)
    tree = ExecutorProcessTree(
        process,
        admission,
        identity,
        platform_name="posix",
    )
    tree.admit()

    def fail_cleanup(**_kwargs: object) -> bool:
        raise RuntimeError("synthetic cleanup failure")

    tree._terminate_posix_group = fail_cleanup  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="synthetic cleanup failure"):
        tree.close()
    with pytest.raises(ProcessContainmentError):
        tree.admit()

    assert tree.admitted is False
    assert tree.quarantined is True
    assert tree._closed is True
    assert tree.close() is False


def test_unproven_tree_death_quarantines_containment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    process = _FakeProcess(calls, dies=False)
    admission = _RecordingEvent(calls)
    identity = WorkerContainmentIdentity(pid=process.pid, process_group_id=process.pid)
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(
        os, "killpg", lambda pgid, sig: calls.append((pgid, sig)), raising=False
    )
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


def test_native_force_stop_removes_worker_and_descendant_before_scratch_cleanup(
    tmp_path: Path,
) -> None:
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
    identity: WorkerContainmentIdentity | None = None
    child_pid: int | None = None
    try:
        kind, identity = _receive(receive)
        assert kind == "identity"
        assert type(identity) is WorkerContainmentIdentity
        assert identity == WorkerContainmentIdentity(
            pid=process.pid,
            process_group_id=process.pid if os.name == "posix" else None,
        )
        tree = ExecutorProcessTree(process, admission, identity)
        tree.admit()
        child_kind, raw_child_pid = _receive(receive)
        assert child_kind == "child"
        child_pid = int(raw_child_pid)
        assert (scratch / "worker-admitted").is_file()
        assert _pid_has_exited(child_pid) is False

        assert tree.terminate_tree(term_timeout=2.0, kill_timeout=2.0) is True
        assert process.is_alive() is False
        assert _pid_has_exited(process.pid) is True
        assert _wait_for_pid_exit(child_pid) is True

        shutil.rmtree(scratch)
        assert scratch.exists() is False
    finally:
        tree_proven = _finalize_native_tree(tree, process, identity, child_pid)
        if tree is None and process.is_alive():
            process.terminate()
            process.join(5.0)
        if (
            scratch.exists()
            and not process.is_alive()
            and (child_pid is None or _pid_has_exited(child_pid))
            and (tree is None or tree_proven)
        ):
            shutil.rmtree(scratch)


def test_native_crashed_leader_reaps_descendant_before_scratch_cleanup(
    tmp_path: Path,
) -> None:
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
        assert identity == WorkerContainmentIdentity(
            pid=process.pid,
            process_group_id=process.pid if os.name == "posix" else None,
        )
        tree = ExecutorProcessTree(process, admission, identity)
        tree.admit()
        child_kind, raw_child_pid = _receive(receive)
        assert child_kind == "child"
        child_pid = int(raw_child_pid)
        assert (scratch / "descendant-ready").is_file()
        process.join(10.0)
        assert process.is_alive() is False
        assert _pid_has_exited(child_pid) is False

        assert tree.terminate_tree(term_timeout=0.1, kill_timeout=2.0) is True
        assert process.is_alive() is False
        assert _pid_has_exited(process.pid) is True
        assert _wait_for_pid_exit(child_pid) is True

        shutil.rmtree(scratch)
        assert scratch.exists() is False
    finally:
        tree_proven = _finalize_native_tree(tree, process, identity, child_pid)
        if tree is None and process.is_alive():
            process.terminate()
            process.join(5.0)
        if (
            scratch.exists()
            and not process.is_alive()
            and (child_pid is None or _pid_has_exited(child_pid))
            and (tree is None or tree_proven)
        ):
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


def test_containment_documentation_is_shared_local_worker_wording() -> None:
    assert "STT" not in (process_tree_module.__doc__ or "")
    assert "local worker generation" in (process_tree_module.__doc__ or "")
    assert "worker generation" in (ExecutorProcessTree.__doc__ or "")


def test_close_is_idempotent_after_proven_tree_death() -> None:
    calls: list[object] = []
    process = _FakeProcess(calls)
    admission = _RecordingEvent(calls)
    identity = WorkerContainmentIdentity(pid=process.pid, process_group_id=process.pid)
    tree = ExecutorProcessTree(
        process,
        admission,
        identity,
        platform_name="posix",
    )
    process._alive = False
    tree._posix_group_exists = lambda _group_id: False  # type: ignore[method-assign]

    assert tree.close() is True
    first_calls = list(calls)
    assert tree.close() is True
    assert calls == first_calls


@pytest.mark.skipif(os.name != "nt", reason="native Windows Job Object evidence")
@pytest.mark.parametrize(
    "leader_exits",
    [False, True],
    ids=["cancellation", "ordinary-finalization"],
)
def test_windows_native_job_object_empties_on_cancel_and_finalization(
    tmp_path: Path,
    leader_exits: bool,
) -> None:
    scratch = tmp_path / "windows-job-object"
    scratch.mkdir(mode=0o700)
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    admission = context.Event()
    process = context.Process(
        target=(
            containment_crashed_leader_with_term_ignoring_descendant
            if leader_exits
            else containment_descendant
        ),
        args=(send, admission, str(scratch)),
    )
    process.start()
    tree: ExecutorProcessTree | None = None
    identity: WorkerContainmentIdentity | None = None
    child_pid: int | None = None
    try:
        kind, identity = _receive(receive)
        assert kind == "identity"
        tree = ExecutorProcessTree(process, admission, identity)
        tree.admit()
        assert tree._job_handle
        child_kind, raw_child_pid = _receive(receive)
        assert child_kind == "child"
        child_pid = int(raw_child_pid)
        if leader_exits:
            process.join(10.0)
            assert process.is_alive() is False

        assert tree.close() is True
        assert process.is_alive() is False
        assert _wait_for_pid_exit(child_pid) is True
    finally:
        _finalize_native_tree(tree, process, identity, child_pid)
        receive.close()
        send.close()
