"""Private CreateProcessW boundary for the one qualified ffplay command."""

from __future__ import annotations

import ctypes as C
from collections.abc import Callable
from ctypes import wintypes as W
from functools import cache
import math
import os
import subprocess

from tldw_chatbook.Utils.windows_job import process_creation_time as _creation_time

_close_fd = os.close
_HANDLE_LIST = 0x00020002  # PROC_THREAD_ATTRIBUTE_HANDLE_LIST
_JOB_LIST = 0x0002000D  # PROC_THREAD_ATTRIBUTE_JOB_LIST
_SUSPENDED = 0x00000004
_EXTENDED_UNICODE = 0x00080000 | 0x00000400


class _StartupInfo(C.Structure):
    _fields_ = [
        ("cb", W.DWORD),
        ("reserved", W.LPWSTR),
        ("desktop", W.LPWSTR),
        ("title", W.LPWSTR),
        ("window_geometry_and_fill", W.DWORD * 7),
        ("flags", W.DWORD),
        ("show_window", W.WORD),
        ("reserved_size", W.WORD),
        ("reserved_bytes", W.LPVOID),
        ("stdin", W.HANDLE),
        ("stdout", W.HANDLE),
        ("stderr", W.HANDLE),
    ]


class _StartupInfoEx(C.Structure):
    _fields_ = [("base", _StartupInfo), ("attributes", W.LPVOID)]


class _ProcessInfo(C.Structure):
    _fields_ = [
        ("process", W.HANDLE),
        ("thread", W.HANDLE),
        ("pid", W.DWORD),
        ("tid", W.DWORD),
    ]


@cache
def _kernel():
    kernel = C.WinDLL("kernel32", use_last_error=True)
    size_pointer = C.POINTER(C.c_size_t)
    for name, args, result in (
        (
            "InitializeProcThreadAttributeList",
            [W.LPVOID, W.DWORD, W.DWORD, size_pointer],
            W.BOOL,
        ),
        (
            "UpdateProcThreadAttribute",
            [
                W.LPVOID,
                W.DWORD,
                C.c_size_t,
                W.LPVOID,
                C.c_size_t,
                W.LPVOID,
                size_pointer,
            ],
            W.BOOL,
        ),
        ("DeleteProcThreadAttributeList", [W.LPVOID], None),
        ("ResumeThread", [W.HANDLE], W.DWORD),
        (
            "CreateProcessW",
            [
                W.LPCWSTR,
                W.LPWSTR,
                W.LPVOID,
                W.LPVOID,
                W.BOOL,
                W.DWORD,
                W.LPVOID,
                W.LPCWSTR,
                C.POINTER(_StartupInfoEx),
                C.POINTER(_ProcessInfo),
            ],
            W.BOOL,
        ),
    ):
        function = getattr(kernel, name)
        function.argtypes, function.restype = args, result
    return kernel


class SpawnRefused(Exception):
    """No native birth occurred and all setup resources retired successfully."""


class UncertainBirth(Exception):
    """Keep the native storage and every resource lacking a positive retirement."""

    def __init__(self, child: WindowsPlayer):
        self.child = child


class _Reader:
    def __init__(self):
        self.fd = -1
        self.failed_close = None

    def read(self, size: int) -> bytes:
        return os.read(self.fd, size)

    def close(self) -> None:
        if self.fd != -1:
            fd, self.fd = self.fd, -1
            try:
                _close_fd(fd)
            except BaseException:
                self.failed_close = fd
                raise


class WindowsPlayer:
    """Own process output storage before CreateProcessW can populate it."""

    def __init__(self):
        self.info = _ProcessInfo()
        self.stderr = _Reader()
        self.returncode: int | None = None
        self.uncertain_closes = []
        self.birth_confirmed = False
        self.job = None
        self.predecessor = None

    @property
    def has_process(self) -> bool:
        return bool(self.info.process)

    def poll(self) -> int | None:
        try:
            return self.wait(timeout=0)
        except subprocess.TimeoutExpired:
            return None

    def wait(self, timeout: float) -> int:
        import _winapi

        if self.returncode is None:
            status = _winapi.WaitForSingleObject(
                self.info.process, math.ceil(timeout * 1000)
            )
            if status == _winapi.WAIT_TIMEOUT:
                raise subprocess.TimeoutExpired("controlled_ffplay", timeout)
            if status != _winapi.WAIT_OBJECT_0:
                raise OSError("player_wait_unconfirmed")
            self.returncode = _winapi.GetExitCodeProcess(self.info.process)
        return self.returncode

    def terminate(self) -> None:
        import _winapi

        _winapi.TerminateProcess(self.info.process, 1)

    kill = terminate

    def close_process_handle(self) -> None:
        import _winapi

        if self.job is not None:
            job, self.job = self.job, None
            try:
                _winapi.CloseHandle(job)
            except BaseException:
                self.uncertain_closes.append(job)
                raise
        handle, self.info.process = self.info.process, None
        if handle is not None:
            try:
                _winapi.CloseHandle(handle)
            except BaseException:
                self.uncertain_closes.append(handle)
                raise


def _environment_block(environment: dict[str, str]):
    entries = []
    for key, value in sorted(environment.items(), key=lambda pair: pair[0].casefold()):
        if not key or "=" in key or "\0" in key or "\0" in value:
            raise ValueError("invalid_environment")
        entries.append(f"{key}={value}")
    return C.create_unicode_buffer("\0".join(entries) + "\0")


_unconfirmed_version_probes = []


def check_contained_version(executable):
    """Prove this process can create the qualified binary in a birth-assigned job."""
    if _unconfirmed_version_probes:
        raise OSError("player_containment_unconfirmed")
    env = dict(os.environ)
    env.pop("FFREPORT", None)
    child = None
    try:
        try:
            child = spawn_player([str(executable), "-version"], env=env)
        except UncertainBirth as error:
            child = error.child
            raise OSError("player_containment_unconfirmed") from None
        except SpawnRefused:
            raise OSError("player_containment_unavailable") from None
        try:
            code = child.wait(5)
        except subprocess.TimeoutExpired:
            raise OSError("player_containment_unavailable") from None
        if code != 0:
            raise OSError("player_containment_unavailable")
    finally:
        if child is not None:
            try:
                if child.has_process and child.poll() is None:
                    child.terminate()
                    child.wait(5)
                child.stderr.close()
                child.close_process_handle()
                if child.uncertain_closes:
                    raise OSError("player_containment_unconfirmed")
            except BaseException:
                _unconfirmed_version_probes.append(child)
                raise


def spawn_player(
    argv: list[str],
    *,
    env: dict[str, str],
    start_allowed: Callable[[], bool] | None = None,
    custody=None,
) -> WindowsPlayer:
    """Launch with only DEVNULL and the stderr writer inheritable by this child."""
    try:
        import _winapi
        import msvcrt

        child = WindowsPlayer()
        kernel = _kernel()
        from tldw_chatbook.Utils.windows_job import WindowsJobApi
    except BaseException:
        # No process or per-attempt file/pipe has been allocated.
        raise SpawnRefused from None
    null_fd = writer_fd = -1
    attributes = None
    attr_initialized = False
    state = "preparing"
    try:
        child.job = WindowsJobApi().create_kill_on_close_job()
        null_fd = os.open(os.devnull, os.O_RDWR | os.O_BINARY)
        child.stderr.fd, writer_fd = os.pipe()
        handles = (W.HANDLE * 2)(
            msvcrt.get_osfhandle(null_fd), msvcrt.get_osfhandle(writer_fd)
        )
        for handle in handles:
            os.set_handle_inheritable(handle, True)
        size = C.c_size_t()
        kernel.InitializeProcThreadAttributeList(None, 2, 0, C.byref(size))
        if not size.value:
            raise OSError("handle_list_size_unavailable")
        attributes = C.create_string_buffer(size.value)
        if not kernel.InitializeProcThreadAttributeList(
            attributes, 2, 0, C.byref(size)
        ):
            raise OSError("handle_list_setup_failed")
        attr_initialized = True
        if not kernel.UpdateProcThreadAttribute(
            attributes, 0, _HANDLE_LIST, handles, C.sizeof(handles), None, None
        ):
            raise OSError("handle_allowlist_failed")
        jobs = (W.HANDLE * 1)(child.job)
        if not kernel.UpdateProcThreadAttribute(
            attributes, 0, _JOB_LIST, jobs, C.sizeof(jobs), None, None
        ):
            raise OSError("player_containment_unavailable")
        startup = _StartupInfoEx()
        startup.base.cb = C.sizeof(startup)
        startup.base.flags = 0x100  # STARTF_USESTDHANDLES
        startup.base.stdin, startup.base.stdout, startup.base.stderr = (
            handles[0],
            handles[0],
            handles[1],
        )
        startup.attributes = C.cast(attributes, W.LPVOID)
        command = C.create_unicode_buffer(subprocess.list2cmdline(argv))
        environment = _environment_block(env)
        if start_allowed is not None and not start_allowed():
            raise SpawnRefused
        state = "entering"
        created = kernel.CreateProcessW(
            argv[0],
            command,
            None,
            None,
            True,
            _EXTENDED_UNICODE | _SUSPENDED,
            environment,
            None,
            C.byref(startup),
            C.byref(child.info),
        )
        if created:
            child.birth_confirmed = True
            if custody is not None:
                predecessor = child.info.pid, _creation_time(child.info.process)
                custody.seal_predecessor(*predecessor)
                child.predecessor = predecessor
            if start_allowed is not None and not start_allowed():
                raise OSError("player_start_revoked")
            if kernel.ResumeThread(child.info.thread) != 1:
                raise OSError("player_resume_unconfirmed")
            state = "created"
        else:
            state = "refused"
    except BaseException:
        # Native interruptions can occur after the kernel populates child.info.
        # Classify the last observed state after all setup closes have run.
        pass
    finally:
        if attr_initialized:
            try:
                kernel.DeleteProcThreadAttributeList(attributes)
            except BaseException:
                child.uncertain_closes.append(attributes)
        thread, child.info.thread = child.info.thread, None
        if thread:
            try:
                _winapi.CloseHandle(thread)
            except BaseException:
                child.uncertain_closes.append(thread)
        for fd in (writer_fd, null_fd):
            if fd != -1:
                try:
                    _close_fd(fd)
                except BaseException:
                    child.uncertain_closes.append(fd)
    if state == "created" and child.has_process and not child.uncertain_closes:
        return child
    if not child.has_process:
        if state in {"preparing", "refused"}:
            try:
                child.close_process_handle()
            except BaseException:
                pass  # The child retains its uncertain close evidence.
        try:
            child.stderr.close()
        except BaseException:
            child.uncertain_closes.append(child.stderr)
        if state in {"preparing", "refused"} and not child.uncertain_closes:
            raise SpawnRefused
    raise UncertainBirth(child)
