"""Platform process-tree ownership for retained network Git children."""

from __future__ import annotations

import asyncio
import ntpath
import os
import signal
import subprocess
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

ProcessArg = str | bytes
_POLL_INTERVAL_SECONDS = 0.01


class AsyncChildProcess(Protocol):
    """Subset of an asynchronous child used by the Git process runner."""

    pid: int
    returncode: int | None
    stdin: Any
    stdout: Any
    stderr: Any

    async def communicate(self, stdin: bytes | None) -> tuple[bytes, bytes]: ...

    async def wait(self) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


@dataclass(slots=True)
class OwnedProcessTree:
    """One created child plus its retained native containment identity."""

    process: AsyncChildProcess
    native_identity: object
    closed: bool = False


class ProcessTreeAdmissionError(OSError):
    """A native child exists but containment admission did not complete."""

    def __init__(self, message: str, tree: OwnedProcessTree) -> None:
        super().__init__(message)
        self.tree = tree


class ProcessTreeControl(Protocol):
    """Injectable lifecycle boundary used by ``AsyncGitProcessRunner``."""

    async def spawn(
        self,
        *argv: ProcessArg,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bool,
    ) -> OwnedProcessTree:
        """Spawn and admit a direct child before returning."""

    def terminate(self, tree: OwnedProcessTree) -> None:
        """Request graceful termination for the admitted tree."""

    def kill(self, tree: OwnedProcessTree) -> None:
        """Force-stop the admitted tree."""

    async def wait(self, tree: OwnedProcessTree, *, timeout: float) -> bool:
        """Return true only when the direct child and containment are empty."""

    def close(self, tree: OwnedProcessTree) -> None:
        """Release a proved-empty containment identity."""


class ProcessTreeController:
    """Select the native POSIX process-group or Windows Job Object adapter."""

    def __init__(self) -> None:
        self._backend: ProcessTreeControl
        if os.name == "nt":
            self._backend = _WindowsJobObjectController()
        else:
            self._backend = _PosixProcessGroupController()

    async def spawn(
        self,
        *argv: ProcessArg,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bool,
    ) -> OwnedProcessTree:
        """Spawn one child and establish native containment before returning."""
        return await self._backend.spawn(
            *argv,
            cwd=cwd,
            environment=environment,
            stdin=stdin,
        )

    def terminate(self, tree: OwnedProcessTree) -> None:
        """Request native graceful tree termination."""
        self._backend.terminate(tree)

    def kill(self, tree: OwnedProcessTree) -> None:
        """Request native forced tree termination."""
        self._backend.kill(tree)

    async def wait(self, tree: OwnedProcessTree, *, timeout: float) -> bool:
        """Wait boundedly for direct-child and descendant settlement proof."""
        return await self._backend.wait(tree, timeout=timeout)

    def close(self, tree: OwnedProcessTree) -> None:
        """Release native handles only after settlement was proved."""
        self._backend.close(tree)


class _PosixProcessGroupController:
    """Own one new POSIX session through its retained process-group ID."""

    async def spawn(
        self,
        *argv: ProcessArg,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bool,
    ) -> OwnedProcessTree:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=dict(environment),
            stdin=(
                asyncio.subprocess.PIPE
                if stdin
                else asyncio.subprocess.DEVNULL
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        return OwnedProcessTree(process, process.pid)

    def terminate(self, tree: OwnedProcessTree) -> None:
        self._signal_group(tree, signal.SIGTERM)

    def kill(self, tree: OwnedProcessTree) -> None:
        self._signal_group(tree, signal.SIGKILL)

    async def wait(self, tree: OwnedProcessTree, *, timeout: float) -> bool:
        process = tree.process
        pgid = self._pgid(tree)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout)
        while True:
            direct_terminal = process.returncode is not None
            group_absent = self._group_absent(pgid)
            if direct_terminal and group_absent:
                return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(_POLL_INTERVAL_SECONDS, remaining))

    def close(self, tree: OwnedProcessTree) -> None:
        tree.closed = True

    @staticmethod
    def _pgid(tree: OwnedProcessTree) -> int:
        pgid = tree.native_identity
        if not isinstance(pgid, int) or pgid <= 0:
            raise ValueError("Invalid retained POSIX process-group identity")
        return pgid

    @staticmethod
    def _group_absent(pgid: int) -> bool:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except (PermissionError, OSError):
            return False
        return False

    def _signal_group(self, tree: OwnedProcessTree, signum: int) -> None:
        pgid = self._pgid(tree)
        if pgid == os.getpgrp():
            raise RuntimeError("Refusing to signal the application process group")
        try:
            os.killpg(pgid, signum)
        except ProcessLookupError:
            return


class _WindowsJobObjectController:
    """Own descendants in a kill-on-close Job before resuming the child."""

    def __init__(self) -> None:
        self._kernel: _WindowsKernel | None = None

    @property
    def _api(self) -> _WindowsKernel:
        if self._kernel is None:
            self._kernel = _WindowsKernel()
        return self._kernel

    async def spawn(
        self,
        *argv: ProcessArg,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bool,
    ) -> OwnedProcessTree:
        if not argv:
            raise ValueError("A process tree requires a non-empty argv")
        arguments = tuple(os.fsdecode(argument) for argument in argv)
        if any("\0" in argument for argument in arguments):
            raise ValueError("Windows process argv cannot contain NUL")
        executable = arguments[0]
        drive, tail = ntpath.splitdrive(executable)
        if not drive or not tail.startswith(("\\", "/")):
            raise ValueError(
                "Windows process executable must be fully qualified"
            )
        if os.name == "nt" and not os.path.isfile(executable):
            raise ValueError("Windows process executable must resolve to a file")
        return self._api.spawn_suspended_assigned(
            arguments,
            cwd=cwd,
            environment=environment,
            stdin=stdin,
        )

    def terminate(self, tree: OwnedProcessTree) -> None:
        identity = self._identity(tree)
        if identity.assigned:
            self._api.generate_ctrl_break(identity.pid)
        else:
            tree.process.terminate()

    def kill(self, tree: OwnedProcessTree) -> None:
        identity = self._identity(tree)
        if identity.assigned:
            self._api.terminate_job(identity.job_handle, 1)
        else:
            tree.process.kill()

    async def wait(self, tree: OwnedProcessTree, *, timeout: float) -> bool:
        identity = self._identity(tree)
        process = tree.process
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout)
        while True:
            poll = getattr(process, "poll", None)
            if callable(poll):
                poll()
            direct_terminal = process.returncode is not None
            job_empty = not identity.assigned
            if identity.assigned:
                try:
                    job_empty = (
                        self._api.active_processes(identity.job_handle) == 0
                    )
                except OSError:
                    return False
            if direct_terminal and job_empty:
                return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(_POLL_INTERVAL_SECONDS, remaining))

    def close(self, tree: OwnedProcessTree) -> None:
        if tree.closed:
            return
        identity = self._identity(tree)
        if isinstance(
            tree.process,
            (_WindowsAsyncChildProcess, _WindowsFailedAdmissionProcess),
        ):
            tree.process.close()
        self._api.close_handle(identity.thread_handle)
        identity.thread_handle = 0
        self._api.close_handle(identity.job_handle)
        tree.closed = True

    @staticmethod
    def _identity(tree: OwnedProcessTree) -> _WindowsJobIdentity:
        identity = tree.native_identity
        if not isinstance(identity, _WindowsJobIdentity):
            raise ValueError("Invalid retained Windows Job Object identity")
        return identity


@dataclass(slots=True)
class _WindowsJobIdentity:
    job_handle: int
    pid: int
    assigned: bool = False
    thread_handle: int = 0


class _WindowsPipeReader:
    """Daemon-drained blocking Win32 pipe exposed as an async reader."""

    _MAX_BUFFERED_BYTES = 256 * 1024

    def __init__(self, kernel: _WindowsKernel, handle: int) -> None:
        self._kernel = kernel
        self._handle = handle
        self._loop = asyncio.get_running_loop()
        self._ready = asyncio.Event()
        self._condition = threading.Condition()
        self._chunks: deque[bytes] = deque()
        self._buffered = 0
        self._eof = False
        self._closed = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._drain,
            name="git-process-tree-pipe-reader",
            daemon=True,
        )
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._thread.start()
        self._started = True

    async def read(self, size: int = -1) -> bytes:
        requested = size if size > 0 else 64 * 1024
        while True:
            with self._condition:
                if self._chunks:
                    chunk = self._chunks.popleft()
                    if len(chunk) > requested:
                        result = chunk[:requested]
                        self._chunks.appendleft(chunk[requested:])
                    else:
                        result = chunk
                    self._buffered -= len(result)
                    self._condition.notify_all()
                    return result
                if self._error is not None:
                    raise self._error
                if self._eof:
                    return b""
                self._ready.clear()
            await self._ready.wait()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            if not self._started:
                self._eof = True
            handle = self._handle
            self._handle = 0
            self._condition.notify_all()
        if handle:
            self._kernel.close_handle(handle)
        self._wake_reader()

    def _drain(self) -> None:
        try:
            while True:
                with self._condition:
                    while (
                        not self._closed
                        and self._buffered >= self._MAX_BUFFERED_BYTES
                    ):
                        self._condition.wait()
                    if self._closed:
                        return
                    handle = self._handle
                chunk = self._kernel.read_file(handle, 64 * 1024)
                if not chunk:
                    return
                with self._condition:
                    self._chunks.append(chunk)
                    self._buffered += len(chunk)
                self._wake_reader()
        except BaseException as error:
            with self._condition:
                if not self._closed:
                    self._error = error
        finally:
            with self._condition:
                handle = self._handle
                self._handle = 0
                self._eof = True
            if handle:
                self._kernel.close_handle(handle)
            self._wake_reader()

    def _wake_reader(self) -> None:
        try:
            self._loop.call_soon_threadsafe(self._ready.set)
        except RuntimeError:
            pass


class _WindowsPipeWriter:
    """Daemon-written stdin pipe that cannot pin interpreter shutdown."""

    def __init__(self, kernel: _WindowsKernel, handle: int) -> None:
        self._kernel = kernel
        self._handle = handle
        self._payload = bytearray()
        self._future: asyncio.Future[None] | None = None

    def write(self, payload: bytes) -> None:
        if self._future is not None:
            raise RuntimeError("Windows child stdin write already started")
        self._payload.extend(payload)

    async def drain(self) -> None:
        if self._future is None:
            loop = asyncio.get_running_loop()
            self._future = loop.create_future()
            threading.Thread(
                target=self._write,
                args=(loop, self._future, bytes(self._payload)),
                name="git-process-tree-pipe-writer",
                daemon=True,
            ).start()
        await asyncio.shield(self._future)

    def close(self) -> None:
        if self._future is None:
            self._close_handle()

    def _write(
        self,
        loop: asyncio.AbstractEventLoop,
        future: asyncio.Future[None],
        payload: bytes,
    ) -> None:
        error: BaseException | None = None
        try:
            self._kernel.write_file(self._handle, payload)
        except BaseException as caught:
            error = caught
        finally:
            self._close_handle()

        def publish() -> None:
            if future.done():
                return
            if error is None:
                future.set_result(None)
            else:
                future.set_exception(error)

        try:
            loop.call_soon_threadsafe(publish)
        except RuntimeError:
            pass

    def _close_handle(self) -> None:
        handle = self._handle
        self._handle = 0
        if handle:
            self._kernel.close_handle(handle)


class _ClosedWindowsPipeReader:
    """EOF-only stream used when suspended admission already failed."""

    async def read(self, size: int = -1) -> bytes:
        del size
        return b""


class _ClosingWindowsPipeWriter:
    """Input facade that closes, but never writes to, a failed admission."""

    def __init__(self, process: _WindowsFailedAdmissionProcess) -> None:
        self._process = process

    def write(self, payload: bytes) -> None:
        del payload

    async def drain(self) -> None:
        self.close()

    def close(self) -> None:
        self._process.close_stdin()


class _WindowsFailedAdmissionProcess:
    """Raw-handle owner retained across fallible Win32 wrapper creation."""

    def __init__(
        self,
        kernel: _WindowsKernel,
        *,
        stdin: bool,
    ) -> None:
        self._kernel = kernel
        self._process_handle = 0
        self._stdin_handle = 0
        self._stdout_handle = 0
        self._stderr_handle = 0
        self.pid = 0
        self.returncode: int | None = None
        self.stdin = _ClosingWindowsPipeWriter(self) if stdin else None
        self.stdout = _ClosedWindowsPipeReader()
        self.stderr = _ClosedWindowsPipeReader()

    def adopt(
        self,
        process_handle: int,
        pid: int,
    ) -> None:
        """Adopt one newly created suspended process without allocating."""
        self._process_handle = process_handle
        self.pid = pid

    def adopt_pipes(
        self,
        stdin_handle: int,
        stdout_handle: int,
        stderr_handle: int,
    ) -> None:
        """Adopt parent pipe handles after native process ownership."""
        self._stdin_handle = stdin_handle
        self._stdout_handle = stdout_handle
        self._stderr_handle = stderr_handle

    def clear_process(self) -> None:
        """Forget raw process state after failed adoption cleanup."""
        self._process_handle = 0
        self.pid = 0

    def wrapper_handles(self) -> tuple[int, int | None, int, int]:
        """Return raw handles while this fallback remains their owner."""
        return (
            self._process_handle,
            self._stdin_handle or None,
            self._stdout_handle,
            self._stderr_handle,
        )

    def transfer_to_wrapper(self) -> None:
        """Relinquish raw handles after full wrapper construction succeeds."""
        self._process_handle = 0
        self._stdin_handle = 0
        self._stdout_handle = 0
        self._stderr_handle = 0

    async def communicate(self, stdin: bytes | None) -> tuple[bytes, bytes]:
        del stdin
        self._close_pipe_handles()
        await self.wait()
        return b"", b""

    async def wait(self) -> int:
        self._close_pipe_handles()
        while self.returncode is None:
            if not self._process_handle:
                raise OSError("Windows failed-admission process handle was lost")
            if self.poll() is not None:
                break
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        assert self.returncode is not None
        return self.returncode

    def poll(self) -> int | None:
        handle = self._process_handle
        if self.returncode is not None or not handle:
            return self.returncode
        if not self._kernel.process_signaled(handle):
            return None
        self.returncode = self._kernel.exit_code(handle)
        self._process_handle = 0
        self._kernel.close_handle(handle)
        return self.returncode

    def terminate(self) -> None:
        self._terminate_direct(127)

    def kill(self) -> None:
        self._terminate_direct(127)

    def close_stdin(self) -> None:
        handle = self._stdin_handle
        self._stdin_handle = 0
        if handle:
            self._kernel.close_handle(handle)

    def close(self) -> None:
        self._close_pipe_handles()
        handle = self._process_handle
        self._process_handle = 0
        if handle:
            self._kernel.close_handle(handle)

    def _close_pipe_handles(self) -> None:
        self.close_stdin()
        for attribute in ("_stdout_handle", "_stderr_handle"):
            handle = getattr(self, attribute)
            setattr(self, attribute, 0)
            if handle:
                self._kernel.close_handle(handle)

    def _terminate_direct(self, exit_code: int) -> None:
        if self._process_handle:
            self._kernel.terminate_process(self._process_handle, exit_code)


class _WindowsAsyncChildProcess:
    """Minimal asynchronous view of one directly launched Win32 process."""

    def __init__(
        self,
        kernel: _WindowsKernel,
        process_handle: int,
        pid: int,
        stdin_handle: int | None,
        stdout_handle: int,
        stderr_handle: int,
    ) -> None:
        self._kernel = kernel
        self._process_handle = process_handle
        self._handle_lock = threading.Lock()
        self.pid = pid
        self.returncode: int | None = None
        self.stdin = (
            _WindowsPipeWriter(kernel, stdin_handle)
            if stdin_handle is not None
            else None
        )
        self.stdout = _WindowsPipeReader(kernel, stdout_handle)
        self.stderr = _WindowsPipeReader(kernel, stderr_handle)

    def start_io(self) -> None:
        """Start daemon drains while the direct child is still suspended."""
        try:
            self.stdout.start()
            self.stderr.start()
        except BaseException:
            self.stdout.close()
            self.stderr.close()
            raise

    async def communicate(self, stdin: bytes | None) -> tuple[bytes, bytes]:
        async def read_all(reader: _WindowsPipeReader) -> bytes:
            chunks: list[bytes] = []
            while chunk := await reader.read(64 * 1024):
                chunks.append(chunk)
            return b"".join(chunks)

        async def write_input() -> None:
            writer = self.stdin
            if writer is None:
                return
            if stdin is not None:
                writer.write(stdin)
                await writer.drain()
            writer.close()

        stdout, stderr, _ = await asyncio.gather(
            read_all(self.stdout),
            read_all(self.stderr),
            write_input(),
        )
        await self.wait()
        return stdout, stderr

    async def wait(self) -> int:
        while self.returncode is None:
            if self.poll() is not None:
                break
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        assert self.returncode is not None
        return self.returncode

    def poll(self) -> int | None:
        """Publish direct-child exit without blocking the event loop."""
        with self._handle_lock:
            handle = self._process_handle
        if self.returncode is not None:
            return self.returncode
        if not handle:
            raise OSError("Windows child process handle was lost")
        if not self._kernel.process_signaled(handle):
            return None
        returncode = self._kernel.exit_code(handle)
        with self._handle_lock:
            if self._process_handle == handle:
                self._process_handle = 0
        self._kernel.close_handle(handle)
        self.returncode = returncode
        return returncode

    def terminate(self) -> None:
        self._terminate_direct(1)

    def kill(self) -> None:
        self._terminate_direct(1)

    def close(self) -> None:
        if self.stdin is not None:
            self.stdin.close()
        self.stdout.close()
        self.stderr.close()
        with self._handle_lock:
            handle = self._process_handle
            self._process_handle = 0
        if handle:
            self._kernel.close_handle(handle)

    def _terminate_direct(self, exit_code: int) -> None:
        with self._handle_lock:
            handle = self._process_handle
        if handle:
            self._kernel.terminate_process(handle, exit_code)

class _WindowsKernel:
    """Typed ctypes calls needed for suspended Job Object admission."""

    _CREATE_SUSPENDED = 0x00000004
    _CREATE_NEW_PROCESS_GROUP = 0x00000200
    _CREATE_UNICODE_ENVIRONMENT = 0x00000400
    _EXTENDED_STARTUPINFO_PRESENT = 0x00080000
    _STARTF_USESTDHANDLES = 0x00000100
    _PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
    _HANDLE_FLAG_INHERIT = 0x00000001
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _CTRL_BREAK_EVENT = 1
    _RESUME_THREAD_FAILED = 0xFFFFFFFF
    _WAIT_OBJECT_0 = 0
    _WAIT_TIMEOUT = 258
    _ERROR_BROKEN_PIPE = 109
    _ERROR_OPERATION_ABORTED = 995

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("Windows Job Objects require Windows")
        import ctypes
        from ctypes import wintypes

        self.ctypes = ctypes
        self.wintypes = wintypes
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._define_types()
        self._bind_functions()

    def spawn_suspended_assigned(
        self,
        argv: tuple[str, ...],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bool,
    ) -> OwnedProcessTree:
        job_handle = self._create_job()
        parent_stdin = parent_stdout = parent_stderr = 0
        child_stdin = child_stdout = child_stderr = 0
        process: _WindowsAsyncChildProcess | None = None
        fallback: _WindowsFailedAdmissionProcess | None = None
        identity: _WindowsJobIdentity | None = None
        tree: OwnedProcessTree | None = None
        try:
            fallback = _WindowsFailedAdmissionProcess(self, stdin=stdin)
            identity = _WindowsJobIdentity(job_handle, 0)
            tree = OwnedProcessTree(fallback, identity)
            job_handle = 0
            parent_stdin, child_stdin = self._pipe(parent_reads=False)
            parent_stdout, child_stdout = self._pipe(parent_reads=True)
            parent_stderr, child_stderr = self._pipe(parent_reads=True)
            self._create_process(
                argv,
                cwd=cwd,
                environment=environment,
                child_handles=(child_stdin, child_stdout, child_stderr),
                fallback=fallback,
                identity=identity,
            )
            fallback.adopt_pipes(
                parent_stdin,
                parent_stdout,
                parent_stderr,
            )
            parent_stdin = parent_stdout = parent_stderr = 0
            self.close_handle(child_stdin)
            child_stdin = 0
            self.close_handle(child_stdout)
            child_stdout = 0
            self.close_handle(child_stderr)
            child_stderr = 0
            if not stdin:
                fallback.close_stdin()
            (
                wrapper_process_handle,
                wrapper_stdin_handle,
                wrapper_stdout_handle,
                wrapper_stderr_handle,
            ) = fallback.wrapper_handles()
            if not self.kernel32.AssignProcessToJobObject(
                identity.job_handle,
                wrapper_process_handle,
            ):
                raise self._last_error("AssignProcessToJobObject")
            identity.assigned = True
            process = _WindowsAsyncChildProcess(
                self,
                wrapper_process_handle,
                identity.pid,
                wrapper_stdin_handle,
                wrapper_stdout_handle,
                wrapper_stderr_handle,
            )
            fallback.transfer_to_wrapper()
            tree.process = process
            process.start_io()
            resumed = self.kernel32.ResumeThread(identity.thread_handle)
            self.close_handle(identity.thread_handle)
            identity.thread_handle = 0
            if resumed != 1:
                raise ProcessTreeAdmissionError(
                    "ResumeThread did not release exactly one suspension",
                    tree,
                )
            return tree
        except ProcessTreeAdmissionError:
            raise
        except BaseException as error:
            if tree is not None and identity is not None and identity.pid > 0:
                raise ProcessTreeAdmissionError(
                    "Windows process-tree admission failed",
                    tree,
                ) from error
            if tree is not None and identity is not None:
                self.close_handle(identity.job_handle)
                tree.closed = True
            raise
        finally:
            for handle in (
                job_handle,
                child_stdin,
                child_stdout,
                child_stderr,
                parent_stdin,
                parent_stdout,
                parent_stderr,
            ):
                self.close_handle(handle)

    def generate_ctrl_break(self, pid: int) -> None:
        if not self.kernel32.GenerateConsoleCtrlEvent(
            self._CTRL_BREAK_EVENT,
            pid,
        ):
            raise self._last_error("GenerateConsoleCtrlEvent")

    def terminate_job(self, job_handle: int, exit_code: int) -> None:
        if not self.kernel32.TerminateJobObject(job_handle, exit_code):
            raise self._last_error("TerminateJobObject")

    def terminate_process(self, process_handle: int, exit_code: int) -> None:
        if not self.kernel32.TerminateProcess(process_handle, exit_code):
            raise self._last_error("TerminateProcess")

    def active_processes(self, job_handle: int) -> int:
        accounting = self.JobAccounting()
        if not self.kernel32.QueryInformationJobObject(
            job_handle,
            self._JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            self.ctypes.byref(accounting),
            self.ctypes.sizeof(accounting),
            None,
        ):
            raise self._last_error("QueryInformationJobObject")
        return int(accounting.ActiveProcesses)

    def process_signaled(self, process_handle: int) -> bool:
        result = self.kernel32.WaitForSingleObject(process_handle, 0)
        if result == self._WAIT_OBJECT_0:
            return True
        if result == self._WAIT_TIMEOUT:
            return False
        raise self._last_error("WaitForSingleObject")

    def exit_code(self, process_handle: int) -> int:
        code = self.wintypes.DWORD()
        if not self.kernel32.GetExitCodeProcess(
            process_handle,
            self.ctypes.byref(code),
        ):
            raise self._last_error("GetExitCodeProcess")
        return int(code.value)

    def read_file(self, handle: int, size: int) -> bytes:
        buffer = self.ctypes.create_string_buffer(size)
        read = self.wintypes.DWORD()
        if not self.kernel32.ReadFile(
            handle,
            buffer,
            size,
            self.ctypes.byref(read),
            None,
        ):
            error = self.ctypes.get_last_error()
            if error in {self._ERROR_BROKEN_PIPE, self._ERROR_OPERATION_ABORTED}:
                return b""
            raise OSError(error, "ReadFile failed")
        return buffer.raw[: read.value]

    def write_file(self, handle: int, payload: bytes) -> None:
        offset = 0
        while offset < len(payload):
            chunk = payload[offset : offset + 64 * 1024]
            buffer = self.ctypes.create_string_buffer(chunk)
            written = self.wintypes.DWORD()
            if not self.kernel32.WriteFile(
                handle,
                buffer,
                len(chunk),
                self.ctypes.byref(written),
                None,
            ):
                error = self.ctypes.get_last_error()
                if error == self._ERROR_BROKEN_PIPE:
                    raise BrokenPipeError(error, "WriteFile pipe closed")
                raise OSError(error, "WriteFile failed")
            if written.value == 0:
                raise OSError("WriteFile made no progress")
            offset += int(written.value)

    def close_handle(self, handle: int) -> None:
        if handle:
            self.kernel32.CloseHandle(handle)

    def _create_job(self) -> int:
        job_handle = self._handle_value(
            self.kernel32.CreateJobObjectW(None, None)
        )
        if not job_handle:
            raise self._last_error("CreateJobObjectW")
        limits = self.JobExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = (
            self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not self.kernel32.SetInformationJobObject(
            job_handle,
            self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            self.ctypes.byref(limits),
            self.ctypes.sizeof(limits),
        ):
            error = self._last_error("SetInformationJobObject")
            self.close_handle(job_handle)
            raise error
        return job_handle

    def _pipe(self, *, parent_reads: bool) -> tuple[int, int]:
        attributes = self.SecurityAttributes()
        attributes.nLength = self.ctypes.sizeof(attributes)
        attributes.bInheritHandle = True
        read_handle = self.wintypes.HANDLE()
        write_handle = self.wintypes.HANDLE()
        if not self.kernel32.CreatePipe(
            self.ctypes.byref(read_handle),
            self.ctypes.byref(write_handle),
            self.ctypes.byref(attributes),
            0,
        ):
            raise self._last_error("CreatePipe")
        read_value = self._handle_value(read_handle)
        write_value = self._handle_value(write_handle)
        parent_handle = read_value if parent_reads else write_value
        child_handle = write_value if parent_reads else read_value
        if not self.kernel32.SetHandleInformation(
            parent_handle,
            self._HANDLE_FLAG_INHERIT,
            0,
        ):
            error = self._last_error("SetHandleInformation")
            self.close_handle(read_value)
            self.close_handle(write_value)
            raise error
        return parent_handle, child_handle

    def _create_process(
        self,
        argv: tuple[str, ...],
        *,
        cwd: str,
        environment: Mapping[str, str],
        child_handles: tuple[int, int, int],
        fallback: _WindowsFailedAdmissionProcess,
        identity: _WindowsJobIdentity,
    ) -> None:
        attribute_size = self.ctypes.c_size_t()
        self.kernel32.InitializeProcThreadAttributeList(
            None,
            1,
            0,
            self.ctypes.byref(attribute_size),
        )
        attribute_buffer = self.ctypes.create_string_buffer(
            attribute_size.value
        )
        attribute_list = self.ctypes.cast(
            attribute_buffer,
            self.ctypes.c_void_p,
        )
        if not self.kernel32.InitializeProcThreadAttributeList(
            attribute_list,
            1,
            0,
            self.ctypes.byref(attribute_size),
        ):
            raise self._last_error("InitializeProcThreadAttributeList")
        handle_array = (self.wintypes.HANDLE * len(child_handles))(
            *child_handles
        )
        try:
            if not self.kernel32.UpdateProcThreadAttribute(
                attribute_list,
                0,
                self._PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                self.ctypes.byref(handle_array),
                self.ctypes.sizeof(handle_array),
                None,
                None,
            ):
                raise self._last_error("UpdateProcThreadAttribute")
            startup = self.StartupInfoEx()
            startup.StartupInfo.cb = self.ctypes.sizeof(startup)
            startup.StartupInfo.dwFlags = self._STARTF_USESTDHANDLES
            startup.StartupInfo.hStdInput = child_handles[0]
            startup.StartupInfo.hStdOutput = child_handles[1]
            startup.StartupInfo.hStdError = child_handles[2]
            startup.lpAttributeList = attribute_list
            process_info = self.ProcessInformation()
            command_line = self.ctypes.create_unicode_buffer(
                subprocess.list2cmdline(argv)
            )
            environment_block = self._environment_block(environment)
            flags = (
                self._CREATE_SUSPENDED
                | self._CREATE_NEW_PROCESS_GROUP
                | self._CREATE_UNICODE_ENVIRONMENT
                | self._EXTENDED_STARTUPINFO_PRESENT
            )
            if not self.kernel32.CreateProcessW(
                argv[0],
                command_line,
                None,
                None,
                True,
                flags,
                environment_block,
                cwd,
                self.ctypes.byref(startup),
                self.ctypes.byref(process_info),
            ):
                raise self._last_error("CreateProcessW")
            process_handle = process_info.hProcess
            thread_handle = process_info.hThread
            pid = process_info.dwProcessId
            try:
                fallback.adopt(process_handle, pid)
                identity.thread_handle = thread_handle
                identity.pid = pid
            except BaseException:
                try:
                    self.terminate_process(process_handle, 127)
                except BaseException:
                    fallback._process_handle = process_handle
                    fallback.pid = pid
                    identity.thread_handle = thread_handle
                    identity.pid = pid
                    raise
                fallback.clear_process()
                identity.thread_handle = 0
                identity.pid = 0
                self.close_handle(thread_handle)
                self.close_handle(process_handle)
                raise
        finally:
            self.kernel32.DeleteProcThreadAttributeList(attribute_list)

    def _environment_block(self, environment: Mapping[str, str]) -> Any:
        entries: list[str] = []
        seen_keys: set[str] = set()
        for key, value in sorted(
            environment.items(),
            key=lambda item: item[0].casefold(),
        ):
            folded = key.casefold()
            if (
                "\0" in key
                or "\0" in value
                or not key
                or "=" in key
                or folded in seen_keys
            ):
                raise ValueError("Invalid Windows child environment")
            seen_keys.add(folded)
            entries.append(f"{key}={value}")
        return self.ctypes.create_unicode_buffer("\0".join(entries) + "\0")

    def _last_error(self, operation: str) -> OSError:
        error = self.ctypes.get_last_error()
        return OSError(error, f"{operation} failed")

    def _handle_value(self, handle: Any) -> int:
        return int(self.ctypes.cast(handle, self.ctypes.c_void_p).value or 0)

    def _define_types(self) -> None:
        ctypes = self.ctypes
        wintypes = self.wintypes

        class SecurityAttributes(ctypes.Structure):
            _fields_ = [
                ("nLength", wintypes.DWORD),
                ("lpSecurityDescriptor", wintypes.LPVOID),
                ("bInheritHandle", wintypes.BOOL),
            ]

        class StartupInfo(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("lpReserved", wintypes.LPWSTR),
                ("lpDesktop", wintypes.LPWSTR),
                ("lpTitle", wintypes.LPWSTR),
                ("dwX", wintypes.DWORD),
                ("dwY", wintypes.DWORD),
                ("dwXSize", wintypes.DWORD),
                ("dwYSize", wintypes.DWORD),
                ("dwXCountChars", wintypes.DWORD),
                ("dwYCountChars", wintypes.DWORD),
                ("dwFillAttribute", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("wShowWindow", wintypes.WORD),
                ("cbReserved2", wintypes.WORD),
                ("lpReserved2", ctypes.POINTER(wintypes.BYTE)),
                ("hStdInput", wintypes.HANDLE),
                ("hStdOutput", wintypes.HANDLE),
                ("hStdError", wintypes.HANDLE),
            ]

        class StartupInfoEx(ctypes.Structure):
            _fields_ = [
                ("StartupInfo", StartupInfo),
                ("lpAttributeList", wintypes.LPVOID),
            ]

        class ProcessInformation(ctypes.Structure):
            _fields_ = [
                ("hProcess", wintypes.HANDLE),
                ("hThread", wintypes.HANDLE),
                ("dwProcessId", wintypes.DWORD),
                ("dwThreadId", wintypes.DWORD),
            ]

        class BasicLimitInformation(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JobExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        class JobAccounting(ctypes.Structure):
            _fields_ = [
                ("TotalUserTime", ctypes.c_longlong),
                ("TotalKernelTime", ctypes.c_longlong),
                ("ThisPeriodTotalUserTime", ctypes.c_longlong),
                ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                ("TotalPageFaultCount", wintypes.DWORD),
                ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD),
                ("TotalTerminatedProcesses", wintypes.DWORD),
            ]

        self.SecurityAttributes = SecurityAttributes
        self.StartupInfoEx = StartupInfoEx
        self.ProcessInformation = ProcessInformation
        self.JobExtendedLimits = JobExtendedLimits
        self.JobAccounting = JobAccounting

    def _bind_functions(self) -> None:
        ctypes = self.ctypes
        wintypes = self.wintypes
        kernel32 = self.kernel32
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.CreatePipe.argtypes = [
            ctypes.POINTER(wintypes.HANDLE),
            ctypes.POINTER(wintypes.HANDLE),
            ctypes.POINTER(self.SecurityAttributes),
            wintypes.DWORD,
        ]
        kernel32.CreatePipe.restype = wintypes.BOOL
        kernel32.SetHandleInformation.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        kernel32.SetHandleInformation.restype = wintypes.BOOL
        kernel32.InitializeProcThreadAttributeList.argtypes = [
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        kernel32.InitializeProcThreadAttributeList.restype = wintypes.BOOL
        kernel32.UpdateProcThreadAttribute.argtypes = [
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.c_size_t,
            wintypes.LPVOID,
            ctypes.c_size_t,
            wintypes.LPVOID,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        kernel32.UpdateProcThreadAttribute.restype = wintypes.BOOL
        kernel32.DeleteProcThreadAttributeList.argtypes = [wintypes.LPVOID]
        kernel32.DeleteProcThreadAttributeList.restype = None
        kernel32.CreateProcessW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            wintypes.LPVOID,
            wintypes.LPVOID,
            wintypes.BOOL,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.LPCWSTR,
            ctypes.POINTER(self.StartupInfoEx),
            ctypes.POINTER(self.ProcessInformation),
        ]
        kernel32.CreateProcessW.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
        ]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
        kernel32.ResumeThread.restype = wintypes.DWORD
        kernel32.GenerateConsoleCtrlEvent.argtypes = [
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        kernel32.GenerateConsoleCtrlEvent.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.UINT,
        ]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateProcess.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.WaitForSingleObject.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
        ]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.ReadFile.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPVOID,
        ]
        kernel32.ReadFile.restype = wintypes.BOOL
        kernel32.WriteFile.argtypes = [
            wintypes.HANDLE,
            wintypes.LPCVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPVOID,
        ]
        kernel32.WriteFile.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
