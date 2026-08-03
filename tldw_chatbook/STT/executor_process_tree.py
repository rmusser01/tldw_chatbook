"""Contain one local STT worker generation and all of its descendants."""

from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
from typing import Any


class ProcessContainmentError(RuntimeError):
    """Raised when a worker cannot be safely admitted to containment."""


@dataclass(frozen=True, slots=True)
class WorkerContainmentIdentity:
    """Worker-reported identity needed for parent-side containment."""

    pid: int
    process_group_id: int | None

    def __post_init__(self) -> None:
        if type(self.pid) is not int or self.pid <= 0:
            raise ValueError("pid must be a positive integer")
        if self.process_group_id is not None and (
            type(self.process_group_id) is not int or self.process_group_id <= 0
        ):
            raise ValueError("process_group_id must be a positive integer")


def enter_worker_containment() -> WorkerContainmentIdentity:
    """Enter the worker-side containment boundary before admission."""

    if os.name == "posix":
        os.setsid()
        return WorkerContainmentIdentity(
            pid=os.getpid(),
            process_group_id=os.getpgrp(),
        )
    return WorkerContainmentIdentity(pid=os.getpid(), process_group_id=None)


class _WindowsJobApi:
    """Lazy ctypes wrapper for the four Job Object calls TASK-601 needs."""

    KILL_ON_JOB_CLOSE = 0x00002000
    _EXTENDED_LIMIT_INFORMATION = 9
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _WAIT_OBJECT_0 = 0
    _WAIT_TIMEOUT = 258

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("Windows Job Objects require Windows")
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        self._wintypes = wintypes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

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

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self._ExtendedLimitInformation = ExtendedLimitInformation
        kernel32 = self._kernel32
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
        ]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD

    def _last_error(self, operation: str) -> OSError:
        error = self._ctypes.get_last_error()
        return OSError(error, f"{operation} failed")

    def _handle_value(self, handle: object) -> int:
        return int(self._ctypes.cast(handle, self._ctypes.c_void_p).value or 0)

    def create_kill_on_close_job(self) -> int:
        """Create one Job Object configured to kill members on close."""

        handle = self._handle_value(self._kernel32.CreateJobObjectW(None, None))
        if not handle:
            raise self._last_error("CreateJobObjectW")
        limits = self._ExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = self.KILL_ON_JOB_CLOSE
        if not self._kernel32.SetInformationJobObject(
            handle,
            self._EXTENDED_LIMIT_INFORMATION,
            self._ctypes.byref(limits),
            self._ctypes.sizeof(limits),
        ):
            error = self._last_error("SetInformationJobObject")
            self.close_handle(handle)
            raise error
        return handle

    def assign_process(self, job_handle: int, pid: int) -> None:
        """Open and assign one already-spawned, not-yet-admitted worker."""

        access = (
            self._PROCESS_TERMINATE
            | self._PROCESS_SET_QUOTA
            | self._PROCESS_QUERY_LIMITED_INFORMATION
        )
        process_handle = self._handle_value(
            self._kernel32.OpenProcess(access, False, pid)
        )
        if not process_handle:
            raise self._last_error("OpenProcess")
        try:
            if not self._kernel32.AssignProcessToJobObject(job_handle, process_handle):
                raise self._last_error("AssignProcessToJobObject")
        finally:
            self._kernel32.CloseHandle(process_handle)

    def terminate_job(self, job_handle: int) -> None:
        """Terminate every process currently assigned to the job."""

        if not self._kernel32.TerminateJobObject(job_handle, 1):
            raise self._last_error("TerminateJobObject")

    def wait_for_job_empty(self, job_handle: int, timeout: float) -> bool:
        """Wait until every process assigned to one job has exited."""

        milliseconds = min(max(int(max(0.0, timeout) * 1000), 0), 0xFFFFFFFE)
        result = int(self._kernel32.WaitForSingleObject(job_handle, milliseconds))
        if result == self._WAIT_OBJECT_0:
            return True
        if result == self._WAIT_TIMEOUT:
            return False
        raise self._last_error("WaitForSingleObject")

    def close_handle(self, job_handle: int) -> None:
        """Close a Job Object handle if present."""

        if job_handle:
            self._kernel32.CloseHandle(job_handle)


class ExecutorProcessTree:
    """Parent-owned containment lifecycle for one worker generation."""

    def __init__(
        self,
        process: Any,
        admission_event: Any,
        identity: WorkerContainmentIdentity,
        *,
        platform_name: str | None = None,
        windows_api: Any | None = None,
    ) -> None:
        if type(identity) is not WorkerContainmentIdentity:
            raise TypeError("identity must be a WorkerContainmentIdentity")
        if getattr(process, "pid", None) != identity.pid:
            raise ValueError("worker identity does not match process")
        self._process = process
        self._admission_event = admission_event
        self._identity = identity
        self._platform_name = platform_name or os.name
        if self._platform_name not in {"posix", "nt"}:
            raise ValueError("unsupported process containment platform")
        self._windows_api = windows_api
        self._job_handle = 0
        self._admitted = False
        self._quarantined = False
        self._closed = False

    @property
    def admitted(self) -> bool:
        """Return whether this generation passed parent admission."""

        return self._admitted

    @property
    def quarantined(self) -> bool:
        """Return whether worker-tree death could not be proven."""

        return self._quarantined

    def admit(self) -> None:
        """Establish platform containment before releasing the worker."""

        if self._closed or self._quarantined:
            raise ProcessContainmentError("worker containment is unavailable")
        if self._admitted:
            return
        if not self._process.is_alive():
            raise ProcessContainmentError("worker exited before containment admission")
        try:
            if self._platform_name == "nt":
                api = self._windows_api or _WindowsJobApi()
                self._windows_api = api
                self._job_handle = api.create_kill_on_close_job()
                api.assign_process(self._job_handle, self._identity.pid)
            elif self._identity.process_group_id != self._identity.pid:
                raise ProcessContainmentError(
                    "POSIX worker did not establish its own process group"
                )
            self._admission_event.set()
            self._admitted = True
        except BaseException as error:
            self._admitted = False
            self._terminate_unadmitted()
            self._close_job_handle()
            self._quarantined = self._process.is_alive()
            if isinstance(error, ProcessContainmentError):
                raise
            raise ProcessContainmentError(
                "worker process-tree admission failed"
            ) from error

    def terminate_tree(
        self,
        *,
        term_timeout: float = 2.0,
        kill_timeout: float = 2.0,
    ) -> bool:
        """Terminate the contained tree and return only after proven death."""

        self._admitted = False
        if self._platform_name == "posix":
            return self._terminate_posix_group(
                term_timeout=term_timeout,
                kill_timeout=kill_timeout,
            )

        job_proven_dead = not self._job_handle
        if self._job_handle:
            try:
                self._windows_api.terminate_job(self._job_handle)
                job_proven_dead = self._windows_api.wait_for_job_empty(
                    self._job_handle,
                    max(0.0, term_timeout),
                )
            except OSError:
                job_proven_dead = False
        if self._process.is_alive() and not self._job_handle:
            self._process.terminate()
        self._process.join(max(0.0, term_timeout))

        if self._process.is_alive() or not job_proven_dead:
            if self._job_handle:
                try:
                    self._windows_api.terminate_job(self._job_handle)
                    job_proven_dead = self._windows_api.wait_for_job_empty(
                        self._job_handle,
                        max(0.0, kill_timeout),
                    )
                except OSError:
                    job_proven_dead = False
            elif self._process.is_alive():
                try:
                    self._process.kill()
                except OSError:
                    pass
            self._process.join(max(0.0, kill_timeout))

        if self._process.is_alive() or not job_proven_dead:
            self._quarantined = True
            return False
        self._close_job_handle()
        self._closed = True
        return True

    def _terminate_posix_group(
        self,
        *,
        term_timeout: float,
        kill_timeout: float,
    ) -> bool:
        """Signal and prove death of the group independently of its leader."""

        group_id = self._identity.process_group_id
        if group_id is None:
            self._quarantined = True
            return False
        if self._posix_group_exists(group_id):
            try:
                os.killpg(group_id, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self._process.join(max(0.0, term_timeout))
        group_dead = self._wait_for_posix_group_exit(group_id, term_timeout)
        if self._process.is_alive() or not group_dead:
            try:
                os.killpg(group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._process.join(max(0.0, kill_timeout))
            group_dead = self._wait_for_posix_group_exit(group_id, kill_timeout)
        if self._process.is_alive() or not group_dead:
            self._quarantined = True
            return False
        self._closed = True
        return True

    @staticmethod
    def _posix_group_exists(group_id: int) -> bool:
        try:
            os.killpg(group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @classmethod
    def _wait_for_posix_group_exit(cls, group_id: int, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while cls._posix_group_exists(group_id) and time.monotonic() < deadline:
            time.sleep(0.01)
        return not cls._posix_group_exists(group_id)

    def close(self) -> bool:
        """Idempotently close containment, terminating a live worker if needed."""

        if self._closed:
            return not self._quarantined
        return self.terminate_tree()

    def _terminate_unadmitted(self) -> None:
        if not self._process.is_alive():
            return
        self._process.terminate()
        self._process.join(2.0)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(2.0)

    def _close_job_handle(self) -> None:
        if not self._job_handle:
            return
        self._windows_api.close_handle(self._job_handle)
        self._job_handle = 0


__all__ = [
    "ExecutorProcessTree",
    "ProcessContainmentError",
    "WorkerContainmentIdentity",
    "enter_worker_containment",
]
