from __future__ import annotations

import asyncio
import collections
import inspect
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

import tldw_chatbook.Notes.git_process_containment as containment
from tldw_chatbook.Notes.file_notes_git_service import (
    AsyncGitProcessRunner,
    GitRunCancelled,
    GitRunCancellationRejected,
)


HELPER = Path(__file__).with_name("git_process_tree_test_helper.py")


class _ControlledProcess:
    """Async child double whose exit and pipe EOF happen together."""

    pid = 4242

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.communicate_started = asyncio.Event()
        self._settled = asyncio.Event()

    async def communicate(self, stdin: bytes | None) -> tuple[bytes, bytes]:
        del stdin
        self.communicate_started.set()
        await self._settled.wait()
        return b"drained-output", b"drained-error"

    async def wait(self) -> int:
        await self._settled.wait()
        assert self.returncode is not None
        return self.returncode

    def settle(self, returncode: int) -> None:
        self.returncode = returncode
        self._settled.set()


@dataclass(slots=True)
class _FakeTree:
    process: _ControlledProcess


class _FakeProcessTreeController:
    """Deterministic containment boundary for runner lifecycle tests."""

    def __init__(
        self,
        process: _ControlledProcess,
        *,
        settle_on: str | None,
    ) -> None:
        self.process = process
        self.settle_on = settle_on
        self.admitted = False
        self.proved = False
        self.closed = False
        self.trace: list[str] = []
        self.wait_timeouts: list[float] = []

    async def spawn(self, *argv, **kwargs) -> _FakeTree:
        del argv, kwargs
        self.trace.append("spawn")
        self.admitted = True
        self.trace.append("admitted")
        return _FakeTree(self.process)

    def terminate(self, tree: _FakeTree) -> None:
        assert tree.process is self.process
        self.trace.append("terminate")
        if self.settle_on == "terminate":
            self.process.settle(-15)
            self.proved = True

    def kill(self, tree: _FakeTree) -> None:
        assert tree.process is self.process
        self.trace.append("kill")
        if self.settle_on == "kill":
            self.process.settle(-9)
            self.proved = True
        elif self.settle_on == "child_only":
            self.process.settle(-9)

    async def wait(self, tree: _FakeTree, *, timeout: float) -> bool:
        assert tree.process is self.process
        self.trace.append("wait")
        self.wait_timeouts.append(timeout)
        return self.proved

    def close(self, tree: _FakeTree) -> None:
        assert tree.process is self.process
        self.trace.append("close")
        self.closed = True


class _FakeWin32Calls:
    def __init__(
        self,
        trace: list[str],
        *,
        assignment_succeeds: bool = True,
        resume_result: int = 1,
    ) -> None:
        self.trace = trace
        self.assignment_succeeds = assignment_succeeds
        self.resume_result = resume_result

    def AssignProcessToJobObject(
        self,
        job_handle: int,
        process_handle: int,
    ) -> bool:
        assert (job_handle, process_handle) == (100, 500)
        self.trace.append("assign")
        return self.assignment_succeeds

    def ResumeThread(self, thread_handle: int) -> int:
        assert thread_handle == 501
        self.trace.append("resume")
        return self.resume_result


def _fake_windows_kernel(
    monkeypatch: pytest.MonkeyPatch,
    trace: list[str],
    *,
    assignment_succeeds: bool = True,
    resume_result: int = 1,
    wrapper_start_fails: bool = False,
    wrapper_construction_fails: bool = False,
    job_kill_proves: bool = True,
    job_query_fails: bool = False,
    job_creation_fails: bool = False,
    pipe_failure_at: int | None = None,
    create_process_fails: bool = False,
):
    kernel = object.__new__(containment._WindowsKernel)
    kernel.kernel32 = _FakeWin32Calls(
        trace,
        assignment_succeeds=assignment_succeeds,
        resume_result=resume_result,
    )
    kernel.closed_handles = []
    kernel.spawned_process = None
    kernel.process_exited = False
    kernel.job_query_fails = job_query_fails
    kernel.wrapper_failed = asyncio.Event()
    pipe_pairs = iter(((200, 201), (300, 301), (400, 401)))
    pipe_calls = 0

    def create_job() -> int:
        trace.append("job_kill_on_close_configured")
        if job_creation_fails:
            raise OSError("job creation failed")
        return 100

    def create_pipe(*, parent_reads: bool) -> tuple[int, int]:
        nonlocal pipe_calls
        pipe_calls += 1
        trace.append(f"pipe_parent_reads={parent_reads}")
        if pipe_failure_at == pipe_calls:
            raise OSError("pipe creation failed")
        return next(pipe_pairs)

    def create_process(
        argv: tuple[str, ...],
        *,
        cwd: str,
        environment,
        child_handles: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        assert argv == ("C:/Git/bin/git.exe", "push")
        assert cwd == "C:/repo"
        assert environment == {"PATH": "C:/Git/bin"}
        assert child_handles == (201, 301, 401)
        trace.append("create_suspended_with_exact_handle_list")
        if create_process_fails:
            raise OSError("process creation failed")
        return 500, 501, 4242

    def close_handle(handle: int) -> None:
        if handle:
            kernel.closed_handles.append(handle)
        if handle == 501:
            trace.append("close_primary_thread")
        elif handle == 100:
            trace.append("close_job")

    kernel._create_job = create_job
    kernel._pipe = create_pipe
    kernel._create_process = create_process
    kernel.close_handle = close_handle
    kernel._last_error = lambda operation: OSError(f"{operation} failed")
    kernel.generate_ctrl_break = lambda pid: trace.append("ctrl_break")

    def terminate_process(handle: int, exit_code: int) -> None:
        del exit_code
        assert handle == 500
        trace.append("runner_terminate_process")
        kernel.process_exited = True

    kernel.terminate_process = terminate_process

    def terminate_job(handle: int, exit_code: int) -> None:
        del exit_code
        assert handle == 100
        trace.append("runner_terminate_job")
        if job_kill_proves and kernel.spawned_process is not None:
            kernel.spawned_process.settle(1)
        if job_kill_proves:
            kernel.process_exited = True

    kernel.terminate_job = terminate_job
    def active_processes(handle: int) -> int:
        assert handle == 100
        trace.append("query_job")
        if kernel.job_query_fails:
            raise OSError("job query failed")
        return 0 if kernel.process_exited else 1

    kernel.active_processes = active_processes
    kernel.process_signaled = lambda handle: kernel.process_exited
    kernel.exit_code = lambda handle: 127

    class FakeWindowsProcess(_ControlledProcess):
        def __init__(self, *_args, **_kwargs) -> None:
            if wrapper_construction_fails:
                trace.append("wrapper_construction_failed")
                kernel.wrapper_failed.set()
                raise RuntimeError("wrapper construction failed")
            super().__init__()
            kernel.spawned_process = self

        def start_io(self) -> None:
            trace.append("wrapper_started_while_suspended")
            if wrapper_start_fails:
                raise RuntimeError("wrapper start failed")

        def close(self) -> None:
            trace.append("close_wrapper")

    monkeypatch.setattr(
        containment,
        "_WindowsAsyncChildProcess",
        FakeWindowsProcess,
    )
    return kernel


@pytest.mark.asyncio
async def test_windows_containment_assigns_suspended_child_before_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []
    controller = containment._WindowsJobObjectController()
    controller._kernel = _fake_windows_kernel(monkeypatch, trace)

    tree = await controller.spawn(
        "C:/Git/bin/git.exe",
        "push",
        cwd="C:/repo",
        environment={"PATH": "C:/Git/bin"},
        stdin=False,
    )
    trace.append("spawn_callback_boundary")

    assert trace.index("job_kill_on_close_configured") < trace.index(
        "create_suspended_with_exact_handle_list"
    )
    assert trace.index("create_suspended_with_exact_handle_list") < trace.index(
        "assign"
    )
    assert trace.index("assign") < trace.index("resume")
    assert trace.index("wrapper_started_while_suspended") < trace.index(
        "resume"
    )
    assert trace.index("resume") < trace.index("close_primary_thread")
    assert trace.index("close_primary_thread") < trace.index(
        "spawn_callback_boundary"
    )
    close_counts = collections.Counter(controller._kernel.closed_handles)
    assert close_counts[200] == 1
    assert close_counts[201] == 1
    assert close_counts[301] == 1
    assert close_counts[401] == 1
    assert close_counts[501] == 1
    assert close_counts[100] == 0
    assert max(close_counts.values()) == 1
    assert not tree.closed


@pytest.mark.asyncio
async def test_windows_assignment_failure_is_settled_by_async_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []
    controller = containment._WindowsJobObjectController()
    controller._kernel = _fake_windows_kernel(
        monkeypatch,
        trace,
        assignment_succeeds=False,
    )
    callback_called = False

    def mark_spawned() -> None:
        nonlocal callback_called
        callback_called = True

    runner = AsyncGitProcessRunner(process_tree_controller=controller)
    result = await runner.run(
        ("C:/Git/bin/git.exe", "push"),
        cwd="C:/repo",
        environment={"PATH": "C:/Git/bin"},
        owned_process_tree=True,
        on_spawn=mark_spawned,
    )

    assert not callback_called
    assert "resume" not in trace
    assert result.termination_uncertain
    assert result.containment_proved
    assert trace.index("assign") < trace.index("runner_terminate_process")
    assert trace.index("runner_terminate_process") < trace.index("close_job")
    assert collections.Counter(controller._kernel.closed_handles)[100] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resume_result",
    [0, 2, containment._WindowsKernel._RESUME_THREAD_FAILED],
)
async def test_windows_containment_invalid_resume_is_retained_without_callback(
    monkeypatch: pytest.MonkeyPatch,
    resume_result: int,
) -> None:
    trace: list[str] = []
    controller = containment._WindowsJobObjectController()
    controller._kernel = _fake_windows_kernel(
        monkeypatch,
        trace,
        resume_result=resume_result,
    )
    callback_called = False

    def mark_spawned() -> None:
        nonlocal callback_called
        callback_called = True

    runner = AsyncGitProcessRunner(
        process_tree_controller=controller,
        terminate_timeout=0.001,
        kill_timeout=0.01,
    )

    result = await runner.run(
        ("C:/Git/bin/git.exe", "push"),
        cwd="C:/repo",
        environment={"PATH": "C:/Git/bin"},
        owned_process_tree=True,
        on_spawn=mark_spawned,
    )

    assert not callback_called
    assert result.owned_process_tree
    assert result.containment_proved
    assert result.termination_uncertain
    assert trace.index("wrapper_started_while_suspended") < trace.index(
        "resume"
    )
    assert trace.index("resume") < trace.index("runner_terminate_job")
    assert "runner_terminate_job" in trace
    assert max(collections.Counter(controller._kernel.closed_handles).values()) == 1


@pytest.mark.asyncio
async def test_windows_wrapper_start_failure_is_settled_asynchronously(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []
    controller = containment._WindowsJobObjectController()
    controller._kernel = _fake_windows_kernel(
        monkeypatch,
        trace,
        wrapper_start_fails=True,
    )
    runner = AsyncGitProcessRunner(process_tree_controller=controller)

    result = await runner.run(
        ("C:/Git/bin/git.exe", "push"),
        cwd="C:/repo",
        environment={"PATH": "C:/Git/bin"},
        owned_process_tree=True,
    )

    assert "resume" not in trace
    assert result.termination_uncertain
    assert result.containment_proved
    assert trace.index("wrapper_started_while_suspended") < trace.index(
        "runner_terminate_job"
    )
    assert trace.index("runner_terminate_job") < trace.index("close_job")
    assert collections.Counter(controller._kernel.closed_handles)[100] == 1


@pytest.mark.asyncio
async def test_windows_wrapper_construction_failure_yields_to_async_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []
    controller = containment._WindowsJobObjectController()
    controller._kernel = _fake_windows_kernel(
        monkeypatch,
        trace,
        wrapper_construction_fails=True,
        job_kill_proves=False,
    )
    runner = AsyncGitProcessRunner(
        process_tree_controller=controller,
        terminate_timeout=0.05,
        kill_timeout=0.05,
    )

    async def prove_exit_after_scheduler_turn() -> None:
        await controller._kernel.wrapper_failed.wait()
        await asyncio.sleep(0)
        trace.append("event_loop_progress")
        controller._kernel.process_exited = True

    progress = asyncio.create_task(prove_exit_after_scheduler_turn())
    result = await runner.run(
        ("C:/Git/bin/git.exe", "push"),
        cwd="C:/repo",
        environment={"PATH": "C:/Git/bin"},
        owned_process_tree=True,
    )
    await progress

    assert "resume" not in trace
    assert result.containment_proved
    assert result.retained_child is None
    assert trace.index("wrapper_construction_failed") < trace.index(
        "event_loop_progress"
    )
    assert trace.index("event_loop_progress") < trace.index("close_job")
    assert max(collections.Counter(controller._kernel.closed_handles).values()) == 1


@pytest.mark.asyncio
async def test_windows_unproved_wrapper_construction_failure_is_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []
    controller = containment._WindowsJobObjectController()
    controller._kernel = _fake_windows_kernel(
        monkeypatch,
        trace,
        wrapper_construction_fails=True,
        job_kill_proves=False,
    )
    runner = AsyncGitProcessRunner(
        process_tree_controller=controller,
        terminate_timeout=0.001,
        kill_timeout=0.001,
    )

    result = await runner.run(
        ("C:/Git/bin/git.exe", "push"),
        cwd="C:/repo",
        environment={"PATH": "C:/Git/bin"},
        owned_process_tree=True,
    )

    assert result.termination_uncertain
    assert result.retained_child is not None
    assert not result.containment_proved
    assert "resume" not in trace
    assert not await runner.shutdown()

    controller._kernel.process_exited = True
    settlement = await runner.settle_retained_child(
        result.retained_child,
        timeout=0.01,
    )
    assert settlement.containment_proved
    assert runner.release_retained_child(result.retained_child)
    assert collections.Counter(controller._kernel.closed_handles)[100] == 1


@pytest.mark.asyncio
async def test_windows_unstarted_pipe_reader_close_publishes_eof() -> None:
    class FakeKernel:
        def __init__(self) -> None:
            self.closed: list[int] = []

        def close_handle(self, handle: int) -> None:
            self.closed.append(handle)

    kernel = FakeKernel()
    reader = containment._WindowsPipeReader(kernel, 700)

    reader.close()

    assert await asyncio.wait_for(reader.read(), timeout=0.05) == b""
    assert kernel.closed == [700]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "options"),
    [
        ("job", {"job_creation_fails": True}),
        ("pipe", {"pipe_failure_at": 2}),
        ("process", {"create_process_fails": True}),
    ],
)
async def test_windows_containment_early_launch_failure_never_reaches_callback(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    options: dict[str, object],
) -> None:
    trace: list[str] = []
    controller = containment._WindowsJobObjectController()
    controller._kernel = _fake_windows_kernel(
        monkeypatch,
        trace,
        **options,
    )
    callback_called = False

    def mark_spawned() -> None:
        nonlocal callback_called
        callback_called = True

    runner = AsyncGitProcessRunner(process_tree_controller=controller)
    with pytest.raises(OSError, match=f"{failure} creation failed"):
        await runner.run(
            ("C:/Git/bin/git.exe", "push"),
            cwd="C:/repo",
            environment={"PATH": "C:/Git/bin"},
            owned_process_tree=True,
            on_spawn=mark_spawned,
        )

    assert not callback_called
    assert "assign" not in trace
    assert "resume" not in trace
    close_counts = collections.Counter(controller._kernel.closed_handles)
    assert not close_counts or max(close_counts.values()) == 1


@pytest.mark.asyncio
async def test_windows_containment_unproved_wrapper_failure_retains_runner_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []
    controller = containment._WindowsJobObjectController()
    controller._kernel = _fake_windows_kernel(
        monkeypatch,
        trace,
        wrapper_start_fails=True,
        job_kill_proves=False,
    )
    runner = AsyncGitProcessRunner(
        process_tree_controller=controller,
        terminate_timeout=0.001,
        kill_timeout=0.001,
    )

    result = await runner.run(
        ("C:/Git/bin/git.exe", "push"),
        cwd="C:/repo",
        environment={"PATH": "C:/Git/bin"},
        owned_process_tree=True,
    )

    assert result.termination_uncertain
    assert not result.containment_proved
    assert result.retained_child is not None
    assert not await runner.shutdown()

    assert controller._kernel.spawned_process is not None
    controller._kernel.spawned_process.settle(127)
    controller._kernel.process_exited = True
    settlement = await runner.settle_retained_child(
        result.retained_child,
        timeout=0.01,
    )
    assert settlement.containment_proved
    assert runner.release_retained_child(result.retained_child)
    assert collections.Counter(controller._kernel.closed_handles)[100] == 1


@pytest.mark.asyncio
async def test_windows_failed_admission_job_query_error_closes_once_after_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []
    controller = containment._WindowsJobObjectController()
    controller._kernel = _fake_windows_kernel(
        monkeypatch,
        trace,
        wrapper_start_fails=True,
        job_query_fails=True,
    )
    runner = AsyncGitProcessRunner(
        process_tree_controller=controller,
        terminate_timeout=0.001,
        kill_timeout=0.001,
    )

    result = await runner.run(
        ("C:/Git/bin/git.exe", "push"),
        cwd="C:/repo",
        environment={"PATH": "C:/Git/bin"},
        owned_process_tree=True,
    )

    assert result.retained_child is not None
    assert not result.containment_proved
    assert runner.claim_retained_child(result.retained_child)
    controller._kernel.job_query_fails = False
    settlement = await runner.settle_retained_child(
        result.retained_child,
        timeout=0.01,
    )
    assert settlement.containment_proved
    assert runner.release_retained_child(result.retained_child)
    close_counts = collections.Counter(controller._kernel.closed_handles)
    assert close_counts[100] == 1
    assert max(close_counts.values()) == 1


class _FakeCreateProcessCalls:
    def __init__(self, kernel) -> None:
        self.kernel = kernel
        self.trace: list[str] = []
        self.created: dict[str, object] = {}

    def InitializeProcThreadAttributeList(
        self,
        attribute_list,
        count: int,
        flags: int,
        size_pointer,
    ) -> bool:
        assert count == 1
        assert flags == 0
        if not attribute_list:
            size_pointer._obj.value = 128
            self.trace.append("size_attribute_list")
            return False
        self.trace.append("initialize_attribute_list")
        return True

    def UpdateProcThreadAttribute(
        self,
        attribute_list,
        flags: int,
        attribute: int,
        value,
        size: int,
        previous,
        return_size,
    ) -> bool:
        del attribute_list, previous, return_size
        assert flags == 0
        assert attribute == self.kernel._PROC_THREAD_ATTRIBUTE_HANDLE_LIST
        handle_array_type = self.kernel.wintypes.HANDLE * 3
        handles = self.kernel.ctypes.cast(
            value,
            self.kernel.ctypes.POINTER(handle_array_type),
        ).contents
        self.created["handle_list"] = tuple(
            int(handle or 0) for handle in handles
        )
        assert size == self.kernel.ctypes.sizeof(handle_array_type)
        self.trace.append("install_exact_handle_list")
        return True

    def DeleteProcThreadAttributeList(self, attribute_list) -> None:
        assert attribute_list
        self.trace.append("delete_attribute_list")

    def CreateProcessW(
        self,
        application_name,
        command_line,
        process_attributes,
        thread_attributes,
        inherit_handles: bool,
        creation_flags: int,
        environment_block,
        cwd: str,
        startup_pointer,
        process_info_pointer,
    ) -> bool:
        del process_attributes, thread_attributes
        startup = startup_pointer._obj.StartupInfo
        self.created.update(
            application_name=application_name,
            command_line=command_line.value,
            inherit_handles=inherit_handles,
            creation_flags=creation_flags,
            environment_block="".join(environment_block),
            cwd=cwd,
            std_handles=(
                int(startup.hStdInput or 0),
                int(startup.hStdOutput or 0),
                int(startup.hStdError or 0),
            ),
            startup_flags=int(startup.dwFlags),
        )
        info = process_info_pointer._obj
        info.hProcess = 500
        info.hThread = 501
        info.dwProcessId = 4242
        info.dwThreadId = 4243
        self.trace.append("create_process")
        return True


def _ctypes_windows_kernel_for_create_process():
    import ctypes
    from ctypes import wintypes

    kernel = object.__new__(containment._WindowsKernel)
    kernel.ctypes = ctypes
    kernel.wintypes = wintypes
    kernel._define_types()
    calls = _FakeCreateProcessCalls(kernel)
    kernel.kernel32 = calls
    return kernel, calls


def test_windows_containment_createprocess_uses_exact_abi_contract() -> None:
    kernel, calls = _ctypes_windows_kernel_for_create_process()
    executable = "C:/Git/bin/git.exe"

    process_handle, thread_handle, pid = kernel._create_process(
        (executable, "push", "arg with space"),
        cwd="C:/repo",
        environment={"TEMP": "C:/Temp", "PATH": "C:/Git/bin"},
        child_handles=(201, 301, 401),
    )

    assert (process_handle, thread_handle, pid) == (500, 501, 4242)
    assert calls.created["application_name"] == executable
    assert calls.created["command_line"] == (
        'C:/Git/bin/git.exe push "arg with space"'
    )
    assert calls.created["inherit_handles"] is True
    assert calls.created["handle_list"] == (201, 301, 401)
    assert calls.created["std_handles"] == (201, 301, 401)
    assert calls.created["startup_flags"] == kernel._STARTF_USESTDHANDLES
    assert calls.created["creation_flags"] == (
        kernel._CREATE_SUSPENDED
        | kernel._CREATE_NEW_PROCESS_GROUP
        | kernel._CREATE_UNICODE_ENVIRONMENT
        | kernel._EXTENDED_STARTUPINFO_PRESENT
    )
    assert calls.created["environment_block"] == (
        "PATH=C:/Git/bin\0TEMP=C:/Temp\0\0"
    )
    assert calls.trace == [
        "size_attribute_list",
        "initialize_attribute_list",
        "install_exact_handle_list",
        "create_process",
        "delete_attribute_list",
    ]


@pytest.mark.parametrize(
    "environment",
    [
        {"PATH": "first", "Path": "second"},
        {"BAD=KEY": "value"},
        {"=C:": "C:/repo"},
        {"PATH": "bad\0value"},
    ],
)
def test_windows_containment_rejects_ambiguous_environment(environment) -> None:
    kernel, _calls = _ctypes_windows_kernel_for_create_process()

    with pytest.raises(ValueError, match="environment"):
        kernel._environment_block(environment)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "argv",
    [
        ("git.exe", "push"),
        (r"\git.exe", "push"),
        (r"C:git.exe", "push"),
        ("C:/Git/bin/git.exe\0other", "push"),
        ("C:/Git/bin/git.exe", "bad\0argument"),
    ],
)
async def test_windows_containment_rejects_unresolved_or_nul_argv(argv) -> None:
    class MustNotLaunch:
        called = False

        def spawn_suspended_assigned(self, *args, **kwargs):
            self.called = True
            raise AssertionError("invalid argv reached CreateProcess")

    api = MustNotLaunch()
    controller = containment._WindowsJobObjectController()
    controller._kernel = api

    with pytest.raises(ValueError, match="executable|NUL"):
        await controller.spawn(
            *argv,
            cwd="C:/repo",
            environment={"PATH": "C:/Git/bin"},
            stdin=False,
        )
    assert not api.called


@pytest.mark.asyncio
async def test_process_tree_child_spawn_callback_follows_containment_admission(
) -> None:
    child = _ControlledProcess()
    controller = _FakeProcessTreeController(child, settle_on="terminate")
    runner = AsyncGitProcessRunner(process_tree_controller=controller)
    callback_trace: list[str] = []

    command = asyncio.create_task(
        runner.run(
            ("git", "push"),
            cwd="/repo",
            environment={},
            timeout=0.001,
            owned_process_tree=True,
            on_spawn=lambda: callback_trace.append(
                "admitted" if controller.admitted else "too_early"
            ),
        )
    )
    await child.communicate_started.wait()
    result = await command

    assert callback_trace == ["admitted"]
    assert result.owned_process_tree
    assert result.containment_proved


@pytest.mark.asyncio
async def test_process_tree_timeout_uses_graceful_termination_and_drains_pipes(
) -> None:
    child = _ControlledProcess()
    controller = _FakeProcessTreeController(child, settle_on="terminate")
    runner = AsyncGitProcessRunner(
        process_tree_controller=controller,
        terminate_timeout=0.02,
        kill_timeout=0.03,
    )

    result = await runner.run(
        ("git", "ls-remote"),
        cwd="/repo",
        environment={},
        timeout=0.001,
        owned_process_tree=True,
    )

    assert controller.trace.count("terminate") == 1
    assert "kill" not in controller.trace
    assert result.stdout == b"drained-output"
    assert result.stderr == b"drained-error"
    assert result.stop_requested
    assert not result.force_stopped
    assert result.containment_proved


@pytest.mark.asyncio
async def test_process_tree_timeout_force_kills_after_one_bounded_grace_wait(
) -> None:
    child = _ControlledProcess()
    controller = _FakeProcessTreeController(child, settle_on="kill")
    runner = AsyncGitProcessRunner(
        process_tree_controller=controller,
        terminate_timeout=0.02,
        kill_timeout=0.03,
    )

    result = await runner.run(
        ("git", "push"),
        cwd="/repo",
        environment={},
        timeout=0.001,
        owned_process_tree=True,
    )

    assert controller.trace.index("terminate") < controller.trace.index("kill")
    bounded_waits = [timeout for timeout in controller.wait_timeouts if timeout]
    assert bounded_waits == [0.02, 0.03]
    assert result.force_stopped
    assert result.containment_proved


@pytest.mark.asyncio
async def test_process_tree_unproved_descendant_retains_uncertain_settlement(
) -> None:
    child = _ControlledProcess()
    controller = _FakeProcessTreeController(child, settle_on="child_only")
    runner = AsyncGitProcessRunner(
        process_tree_controller=controller,
        terminate_timeout=0.001,
        kill_timeout=0.001,
    )

    result = await runner.run(
        ("git", "push"),
        cwd="/repo",
        environment={},
        timeout=0.001,
        owned_process_tree=True,
    )

    assert result.termination_uncertain
    assert result.owned_process_tree
    assert not result.containment_proved
    assert result.retained_child is not None
    observation = runner.read_retained_child(result.retained_child)
    assert observation.state == "uncertain"
    assert not observation.containment_proved
    assert not runner.release_retained_child(result.retained_child)
    assert not await runner.shutdown()

    controller.proved = True
    settlement = await runner.settle_retained_child(
        result.retained_child,
        timeout=0.01,
    )
    assert settlement.containment_proved
    assert runner.release_retained_child(result.retained_child)
    assert controller.closed


@pytest.mark.asyncio
async def test_process_tree_callback_failure_settles_tree_before_raising() -> None:
    child = _ControlledProcess()
    controller = _FakeProcessTreeController(child, settle_on="kill")
    runner = AsyncGitProcessRunner(
        process_tree_controller=controller,
        terminate_timeout=0.001,
        kill_timeout=0.01,
    )

    def fail_callback() -> None:
        raise RuntimeError("spawn observer failed")

    with pytest.raises(RuntimeError, match="spawn observer failed"):
        await runner.run(
            ("git", "push"),
            cwd="/repo",
            environment={},
            owned_process_tree=True,
            on_spawn=fail_callback,
        )

    assert controller.trace.index("admitted") < controller.trace.index("terminate")
    assert controller.trace.index("terminate") < controller.trace.index("kill")
    assert controller.closed


class _AdmissionBarrierController(_FakeProcessTreeController):
    """Expose the post-create/pre-admission cancellation window."""

    def __init__(self, process: _ControlledProcess) -> None:
        super().__init__(process, settle_on="kill")
        self.child_created = asyncio.Event()
        self.release_admission = asyncio.Event()
        self.spawn_cancelled = False

    async def spawn(self, *argv, **kwargs) -> _FakeTree:
        del argv, kwargs
        self.trace.append("child_created")
        self.child_created.set()
        try:
            await self.release_admission.wait()
        except asyncio.CancelledError:
            self.spawn_cancelled = True
            raise
        self.admitted = True
        self.trace.append("admitted")
        return _FakeTree(self.process)


class _PreAdmissionBarrierRunner(AsyncGitProcessRunner):
    """Park the owned task before the runner commits to admission."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.inner_parked = asyncio.Event()
        self.release_inner = asyncio.Event()

    async def _run_owned_command(self, *args, **kwargs):
        self.inner_parked.set()
        await self.release_inner.wait()
        return await super()._run_owned_command(*args, **kwargs)


class _CancellationWaitBarrierRunner(AsyncGitProcessRunner):
    """Publish when the cancelled waiter starts shielding admission."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cancellation_waiting = asyncio.Event()

    async def _await_admission_outcome(self, record) -> bool:
        self.cancellation_waiting.set()
        return await super()._await_admission_outcome(record)


class _FailedAdmissionBarrierController(_AdmissionBarrierController):
    """Create a child but fail containment admission after a barrier."""

    async def spawn(self, *argv, **kwargs) -> _FakeTree:
        del argv, kwargs
        self.trace.append("child_created")
        self.child_created.set()
        await self.release_admission.wait()
        raise containment.ProcessTreeAdmissionError(
            "admission failed",
            _FakeTree(self.process),
        )


@pytest.mark.asyncio
async def test_process_tree_pre_admission_cancellation_never_starts_child(
) -> None:
    child = _ControlledProcess()
    controller = _FakeProcessTreeController(child, settle_on="kill")
    runner = _PreAdmissionBarrierRunner(
        process_tree_controller=controller,
        terminate_timeout=0.001,
        kill_timeout=0.01,
    )
    callback_trace: list[str] = []
    command = asyncio.create_task(
        runner.run(
            ("git", "push"),
            cwd="/repo",
            environment={},
            owned_process_tree=True,
            cancel_before_spawn=True,
            on_spawn=lambda: callback_trace.append("started"),
        )
    )
    await runner.inner_parked.wait()

    command.cancel()
    runner.release_inner.set()
    with pytest.raises(GitRunCancelled) as cancellation:
        await command
    assert cancellation.value.result is not None
    assert cancellation.value.retained_child is None
    assert controller.trace == []
    assert callback_trace == []
    assert await runner.shutdown()


@pytest.mark.asyncio
async def test_process_tree_pending_admission_rejects_cancellation_and_continues(
) -> None:
    child = _ControlledProcess()
    controller = _AdmissionBarrierController(child)
    runner = AsyncGitProcessRunner(
        process_tree_controller=controller,
        terminate_timeout=0.001,
        kill_timeout=0.01,
    )
    callback_trace: list[str] = []
    command = asyncio.create_task(
        runner.run(
            ("git", "push"),
            cwd="/repo",
            environment={},
            owned_process_tree=True,
            cancel_before_spawn=True,
            on_spawn=lambda: callback_trace.append("started"),
        )
    )
    await controller.child_created.wait()

    command.cancel()
    await asyncio.sleep(0)
    assert not command.done()
    assert not controller.spawn_cancelled
    controller.release_admission.set()

    with pytest.raises(GitRunCancelled) as cancellation:
        await command
    assert isinstance(cancellation.value, GitRunCancellationRejected)
    assert cancellation.value.cancellation_rejected
    assert cancellation.value.retained_child is not None
    assert cancellation.value.result is None
    assert callback_trace == ["started"]
    assert "terminate" not in controller.trace
    assert "kill" not in controller.trace
    token = cancellation.value.retained_child
    assert runner.claim_retained_child(token)

    assert await runner.shutdown()
    settlement = await runner.settle_retained_child(token, timeout=0.1)
    assert settlement.containment_proved
    assert runner.release_retained_child(token)
    assert controller.closed


@pytest.mark.asyncio
async def test_process_tree_repeated_cancellation_cannot_cancel_pending_admission(
) -> None:
    child = _ControlledProcess()
    controller = _AdmissionBarrierController(child)
    runner = _CancellationWaitBarrierRunner(
        process_tree_controller=controller,
        terminate_timeout=0.001,
        kill_timeout=0.01,
    )
    callback_trace: list[str] = []
    command = asyncio.create_task(
        runner.run(
            ("git", "push"),
            cwd="/repo",
            environment={},
            owned_process_tree=True,
            cancel_before_spawn=True,
            on_spawn=lambda: callback_trace.append("started"),
        )
    )
    await controller.child_created.wait()

    command.cancel()
    await runner.cancellation_waiting.wait()
    command.cancel()
    await asyncio.sleep(0)
    assert not command.done()
    assert not controller.spawn_cancelled
    controller.release_admission.set()

    with pytest.raises(GitRunCancelled) as cancellation:
        await command
    assert isinstance(cancellation.value, GitRunCancellationRejected)
    assert cancellation.value.cancellation_rejected
    assert cancellation.value.retained_child is not None
    assert cancellation.value.result is None
    assert callback_trace == ["started"]
    assert "terminate" not in controller.trace
    assert "kill" not in controller.trace

    token = cancellation.value.retained_child
    assert runner.claim_retained_child(token)
    assert await runner.shutdown()
    settlement = await runner.settle_retained_child(
        token,
        timeout=0.1,
    )
    assert settlement.containment_proved
    assert runner.release_retained_child(token)


@pytest.mark.asyncio
async def test_process_tree_failed_admission_is_not_active_cancellation(
) -> None:
    child = _ControlledProcess()
    controller = _FailedAdmissionBarrierController(child)
    runner = AsyncGitProcessRunner(
        process_tree_controller=controller,
        terminate_timeout=0.001,
        kill_timeout=0.01,
    )
    callback_trace: list[str] = []
    command = asyncio.create_task(
        runner.run(
            ("git", "push"),
            cwd="/repo",
            environment={},
            owned_process_tree=True,
            cancel_before_spawn=True,
            on_spawn=lambda: callback_trace.append("started"),
        )
    )
    await controller.child_created.wait()

    command.cancel()
    controller.release_admission.set()

    with pytest.raises(GitRunCancelled) as cancellation:
        await command
    assert type(cancellation.value) is GitRunCancelled
    assert cancellation.value.result is not None
    assert cancellation.value.retained_child is None
    assert cancellation.value.result.containment_proved
    assert callback_trace == []
    assert await runner.shutdown()


@pytest.mark.asyncio
async def test_process_tree_forced_sweep_never_resignals_retained_group() -> None:
    child = _ControlledProcess()
    controller = _FakeProcessTreeController(child, settle_on="child_only")
    runner = AsyncGitProcessRunner(
        process_tree_controller=controller,
        terminate_timeout=0.001,
        kill_timeout=0.001,
    )

    result = await runner.run(
        ("git", "push"),
        cwd="/repo",
        environment={},
        timeout=0.001,
        owned_process_tree=True,
    )
    assert result.retained_child is not None
    assert controller.trace.count("terminate") == 1
    assert controller.trace.count("kill") == 1

    assert not await runner.shutdown()
    assert controller.trace.count("terminate") == 1
    assert controller.trace.count("kill") == 1

    controller.proved = True
    settlement = await runner.settle_retained_child(
        result.retained_child,
        timeout=0.0,
    )
    assert settlement.containment_proved
    assert runner.release_retained_child(result.retained_child)


@pytest.mark.asyncio
async def test_process_tree_zero_timeout_refreshes_native_containment_proof() -> None:
    child = _ControlledProcess()
    controller = _FakeProcessTreeController(child, settle_on="child_only")
    runner = AsyncGitProcessRunner(
        process_tree_controller=controller,
        terminate_timeout=0.001,
        kill_timeout=0.001,
    )
    result = await runner.run(
        ("git", "push"),
        cwd="/repo",
        environment={},
        timeout=0.001,
        owned_process_tree=True,
    )
    assert result.retained_child is not None
    controller.proved = True

    settlement = await runner.settle_retained_child(
        result.retained_child,
        timeout=0.0,
    )

    assert settlement.containment_proved
    assert runner.release_retained_child(result.retained_child)


@pytest.mark.asyncio
async def test_process_tree_callback_failure_returns_token_when_cleanup_unproved(
) -> None:
    child = _ControlledProcess()
    controller = _FakeProcessTreeController(child, settle_on="child_only")
    runner = AsyncGitProcessRunner(
        process_tree_controller=controller,
        terminate_timeout=0.001,
        kill_timeout=0.001,
    )

    def fail_callback() -> None:
        raise RuntimeError("spawn observer failed")

    result = await runner.run(
        ("git", "push"),
        cwd="/repo",
        environment={},
        owned_process_tree=True,
        on_spawn=fail_callback,
    )

    assert result.termination_uncertain
    assert result.retained_child is not None
    assert b"spawn observer failed" in result.stderr
    controller.proved = True
    await runner.settle_retained_child(result.retained_child, timeout=0.01)
    assert runner.release_retained_child(result.retained_child)


@pytest.mark.asyncio
async def test_process_tree_controller_errors_return_explicit_uncertainty() -> None:
    class FailingController(_FakeProcessTreeController):
        fail = True

        def terminate(self, tree: _FakeTree) -> None:
            self.trace.append("terminate")
            if self.fail:
                raise RuntimeError("terminate failed")

        def kill(self, tree: _FakeTree) -> None:
            self.trace.append("kill")
            if self.fail:
                raise RuntimeError("kill failed")

        async def wait(self, tree: _FakeTree, *, timeout: float) -> bool:
            self.trace.append("wait")
            self.wait_timeouts.append(timeout)
            if self.fail:
                raise RuntimeError("wait failed")
            return self.proved

    child = _ControlledProcess()
    controller = FailingController(child, settle_on=None)
    runner = AsyncGitProcessRunner(
        process_tree_controller=controller,
        terminate_timeout=0.001,
        kill_timeout=0.001,
    )

    result = await runner.run(
        ("git", "push"),
        cwd="/repo",
        environment={},
        timeout=0.001,
        owned_process_tree=True,
    )

    assert result.termination_uncertain
    assert result.retained_child is not None
    controller.fail = False
    controller.proved = True
    child.settle(-9)
    await runner.settle_retained_child(result.retained_child, timeout=0.01)
    assert runner.release_retained_child(result.retained_child)


@pytest.mark.asyncio
async def test_process_tree_shutdown_rescans_children_admitted_during_shutdown(
) -> None:
    class LateProofController(_AdmissionBarrierController):
        async def wait(self, tree: _FakeTree, *, timeout: float) -> bool:
            self.trace.append("wait")
            self.wait_timeouts.append(timeout)
            if self.trace.count("wait") >= 4:
                self.proved = True
            return self.proved

        def kill(self, tree: _FakeTree) -> None:
            self.trace.append("kill")
            self.process.settle(-9)

    child = _ControlledProcess()
    controller = LateProofController(child)
    runner = AsyncGitProcessRunner(
        process_tree_controller=controller,
        terminate_timeout=0.001,
        kill_timeout=0.001,
    )
    command = asyncio.create_task(
        runner.run(
            ("git", "push"),
            cwd="/repo",
            environment={},
            owned_process_tree=True,
        )
    )
    await controller.child_created.wait()
    shutdown = asyncio.ensure_future(runner.shutdown())
    controller.release_admission.set()

    command_result = await command
    assert command_result.retained_child is not None
    assert await shutdown
    assert controller.trace.count("terminate") == 1
    assert controller.trace.count("kill") == 1
    assert controller.closed


@pytest.mark.asyncio
async def test_process_tree_output_failure_releases_proved_native_resources() -> None:
    class BrokenOutputProcess(_ControlledProcess):
        async def communicate(self, stdin: bytes | None) -> tuple[bytes, bytes]:
            del stdin
            self.communicate_started.set()
            self.returncode = 0
            self._settled.set()
            raise OSError("pipe reader failed")

    child = BrokenOutputProcess()
    controller = _FakeProcessTreeController(child, settle_on=None)
    controller.proved = True
    runner = AsyncGitProcessRunner(process_tree_controller=controller)

    result = await runner.run(
        ("git", "push"),
        cwd="/repo",
        environment={},
        owned_process_tree=True,
    )

    assert result.returncode == 0
    assert result.termination_uncertain
    assert result.containment_proved
    assert result.retained_child is None
    assert controller.closed
    assert await runner.shutdown()


def _records(stdout: bytes) -> list[dict[str, int | str]]:
    records: list[dict[str, int | str]] = []
    for line in stdout.splitlines():
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("event"), str):
            records.append(payload)
    return records


async def _wait_for_pid_exit(pid: int, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
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
                wait_result = kernel32.WaitForSingleObject(handle, 0)
                if wait_result == wait_object_0:
                    return True
                if wait_result != wait_timeout:
                    error = ctypes.get_last_error()
                    raise OSError(error, "WaitForSingleObject failed")
            finally:
                kernel32.CloseHandle(handle)
        else:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                return False
        await asyncio.sleep(0.02)
    return False


async def _wait_for_ready_file(
    path: Path,
    timeout: float = 3.0,
) -> dict[str, int]:
    deadline = time.monotonic() + timeout
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"process tree did not publish readiness: {path}")
        await asyncio.sleep(0.01)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert all(isinstance(value, int) for value in payload.values())
    return payload


def _cleanup_captured_posix_group(readiness: dict[str, int] | None) -> None:
    if os.name == "nt" or readiness is None:
        return
    pgid = readiness.get("pgid")
    if pgid is None or pgid <= 0 or pgid == os.getpgrp():
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _cleanup_captured_windows_processes(
    readiness: dict[str, int] | None,
) -> None:
    if os.name != "nt" or readiness is None:
        return
    import ctypes
    from ctypes import wintypes

    process_terminate = 0x0001
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    for key in ("grandchild_pid", "parent_pid"):
        pid = readiness.get(key)
        if pid is None or pid <= 0:
            continue
        handle = kernel32.OpenProcess(process_terminate, False, pid)
        if not handle:
            continue
        try:
            kernel32.TerminateProcess(handle, 127)
        finally:
            kernel32.CloseHandle(handle)


async def _finish_native_command_bounded(
    command: asyncio.Task[object],
    *,
    timeout: float = 1.0,
) -> None:
    if not command.done():
        done, _ = await asyncio.wait({command}, timeout=timeout)
        if not done:
            command.cancel()
            await asyncio.wait({command}, timeout=0.25)
    if command.done():
        await asyncio.gather(command, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group proof")
async def test_process_tree_posix_session_kills_and_drains_stubborn_descendant(
    tmp_path: Path,
) -> None:
    runner = AsyncGitProcessRunner(terminate_timeout=0.05, kill_timeout=1.0)
    ready_file = tmp_path / "tree-ready"
    command = asyncio.create_task(
        runner.run(
            (
                sys.executable,
                str(HELPER),
                "--ignore-termination",
                "--ready-file",
                str(ready_file),
            ),
            cwd=str(HELPER.parent),
            environment=dict(os.environ),
            timeout=None,
            stdout_limit=128 * 1024,
            stderr_limit=16 * 1024,
            owned_process_tree=True,
        )
    )
    shutdown = None
    readiness = None
    result = None
    shutdown_proved = False
    try:
        readiness = await _wait_for_ready_file(ready_file)
        shutdown = runner.shutdown()
        result = await asyncio.wait_for(asyncio.shield(command), timeout=3.0)
        shutdown_proved = await shutdown
        assert shutdown_proved
    finally:
        if shutdown is None:
            shutdown = runner.shutdown()
        try:
            shutdown_proved = await shutdown
        finally:
            if (
                not shutdown_proved
                or result is None
                or not result.containment_proved
            ):
                _cleanup_captured_posix_group(readiness)
            await _finish_native_command_bounded(command)
    assert result is not None
    records = _records(result.stdout)
    parent = next(record for record in records if record["event"] == "parent_spawned")
    grandchild = next(
        record for record in records if record["event"] == "grandchild_spawned"
    )

    assert parent["pid"] == parent["pgid"] == parent["sid"]
    assert grandchild["pgid"] == parent["pgid"]
    assert result.force_stopped
    assert result.containment_proved
    assert result.retained_child is None
    assert await _wait_for_pid_exit(int(parent["pid"]))
    assert await _wait_for_pid_exit(int(grandchild["pid"]))
    assert await runner.shutdown()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group proof")
@pytest.mark.parametrize(
    "close_grandchild_stdio",
    [False, True],
    ids=["descendant-holds-pipes", "descendant-closes-pipes"],
)
async def test_process_tree_posix_stops_descendant_after_parent_exit(
    tmp_path: Path,
    close_grandchild_stdio: bool,
) -> None:
    runner = AsyncGitProcessRunner(terminate_timeout=0.05, kill_timeout=1.0)
    ready_file = tmp_path / "early-parent-ready"
    helper_argv = [
        sys.executable,
        str(HELPER),
        "--ignore-termination",
        "--ready-file",
        str(ready_file),
        "--parent-exits-after-ready",
    ]
    if close_grandchild_stdio:
        helper_argv.append("--close-grandchild-stdio")
    command = asyncio.create_task(
        runner.run(
            tuple(helper_argv),
            cwd=str(HELPER.parent),
            environment=dict(os.environ),
            timeout=None,
            stdout_limit=128 * 1024,
            stderr_limit=16 * 1024,
            owned_process_tree=True,
        )
    )
    readiness = None
    shutdown = None
    result = None
    shutdown_proved = False
    try:
        readiness = await _wait_for_ready_file(ready_file)
        if not close_grandchild_stdio:
            shutdown = runner.shutdown()
        result = await asyncio.wait_for(
            asyncio.shield(command),
            timeout=3.0,
        )
        if shutdown is not None:
            shutdown_proved = await shutdown
            assert shutdown_proved
    finally:
        if shutdown is None:
            shutdown = runner.shutdown()
        try:
            shutdown_proved = await shutdown
        finally:
            if (
                not shutdown_proved
                or result is None
                or not result.containment_proved
            ):
                _cleanup_captured_posix_group(readiness)
            await _finish_native_command_bounded(command)
    assert result is not None
    records = _records(result.stdout)
    parent = next(record for record in records if record["event"] == "parent_spawned")
    grandchild = next(
        record for record in records if record["event"] == "grandchild_spawned"
    )

    assert result.stop_requested
    assert result.force_stopped
    assert result.containment_proved
    assert result.retained_child is None
    assert await _wait_for_pid_exit(int(parent["pid"]))
    assert await _wait_for_pid_exit(int(grandchild["pid"]))


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object proof")
async def test_process_tree_windows_job_contains_immediate_descendant_spawn(
    tmp_path: Path,
) -> None:
    runner = AsyncGitProcessRunner(terminate_timeout=0.05, kill_timeout=1.0)
    ready_file = tmp_path / "tree-ready"
    command = asyncio.create_task(
        runner.run(
            (
                sys.executable,
                str(HELPER),
                "--ignore-termination",
                "--ready-file",
                str(ready_file),
            ),
            cwd=str(HELPER.parent),
            environment=dict(os.environ),
            timeout=None,
            stdout_limit=128 * 1024,
            stderr_limit=16 * 1024,
            owned_process_tree=True,
        )
    )
    shutdown = None
    readiness = None
    result = None
    shutdown_proved = False
    try:
        readiness = await _wait_for_ready_file(ready_file)
        shutdown = runner.shutdown()
        result = await asyncio.wait_for(asyncio.shield(command), timeout=3.0)
        shutdown_proved = await shutdown
        assert shutdown_proved
    finally:
        if shutdown is None:
            shutdown = runner.shutdown()
        try:
            shutdown_proved = await shutdown
        finally:
            if (
                not shutdown_proved
                or result is None
                or not result.containment_proved
            ):
                _cleanup_captured_windows_processes(readiness)
            await _finish_native_command_bounded(command)
    assert result is not None
    records = _records(result.stdout)
    parent = next(record for record in records if record["event"] == "parent_spawned")
    grandchild = next(
        record for record in records if record["event"] == "grandchild_spawned"
    )

    assert result.force_stopped
    assert result.containment_proved
    assert result.retained_child is None
    assert await _wait_for_pid_exit(int(parent["pid"]))
    assert await _wait_for_pid_exit(int(grandchild["pid"]))
    assert await runner.shutdown()


def test_process_tree_opt_in_is_explicit_on_public_runner_signature() -> None:
    parameter = inspect.signature(AsyncGitProcessRunner.run).parameters[
        "owned_process_tree"
    ]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False


@pytest.mark.parametrize(
    ("probe_error", "absent"),
    [
        (None, False),
        (PermissionError("unobservable group"), False),
        (OSError("indeterminate group"), False),
        (ProcessLookupError("group is absent"), True),
    ],
)
def test_posix_group_probe_requires_explicit_absence_proof(
    monkeypatch: pytest.MonkeyPatch,
    probe_error: OSError | None,
    absent: bool,
) -> None:
    def probe_group(pgid: int, signal_number: int) -> None:
        assert pgid == 4242
        assert signal_number == 0
        if probe_error is not None:
            raise probe_error

    monkeypatch.setattr(os, "killpg", probe_group)

    assert containment._PosixProcessGroupController._group_absent(4242) is absent
