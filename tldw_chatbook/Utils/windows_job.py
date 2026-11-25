"""Checked Windows Job Object primitives for owned subprocesses."""

from __future__ import annotations

import ctypes
import os
import time


class _JobObjectBasicAccountingInformation(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", ctypes.c_uint32),
        ("TotalProcesses", ctypes.c_uint32),
        ("ActiveProcesses", ctypes.c_uint32),
        ("TotalTerminatedProcesses", ctypes.c_uint32),
    ]


class WindowsJobApi:
    """Lazy ctypes wrapper for the Job Object calls TASK-601 needs."""

    KILL_ON_JOB_CLOSE = 0x00002000
    _BASIC_ACCOUNTING_INFORMATION = 1
    _EXTENDED_LIMIT_INFORMATION = 9
    _EMPTY_POLL_SECONDS = 0.01
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("Windows Job Objects require Windows")
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
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

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

        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            accounting = _JobObjectBasicAccountingInformation()
            if not self._kernel32.QueryInformationJobObject(
                job_handle,
                self._BASIC_ACCOUNTING_INFORMATION,
                self._ctypes.byref(accounting),
                self._ctypes.sizeof(accounting),
                None,
            ):
                raise self._last_error("QueryInformationJobObject")
            if accounting.ActiveProcesses == 0:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(self._EMPTY_POLL_SECONDS, remaining))

    def close_handle(self, job_handle: int) -> None:
        """Close a Job Object handle if present."""

        if job_handle and not self._kernel32.CloseHandle(job_handle):
            raise self._last_error("CloseHandle")


def process_creation_time(handle: int) -> int:
    from ctypes import wintypes as W

    get_times = ctypes.WinDLL("kernel32", use_last_error=True).GetProcessTimes
    get_times.argtypes = [W.HANDLE, *([ctypes.POINTER(W.FILETIME)] * 4)]
    get_times.restype = W.BOOL
    times = (W.FILETIME * 4)()
    if not get_times(handle, *(ctypes.byref(value) for value in times)):
        raise OSError("player_identity_unavailable")
    return times[0].dwHighDateTime << 32 | times[0].dwLowDateTime


def predecessor_retired(pid: int, created: int) -> bool:
    """Observe the original process without requesting any mutation right."""
    import _winapi

    try:
        handle = _winapi.OpenProcess(0x00100000 | 0x1000, False, pid)
    except OSError as error:
        if error.winerror == 87:  # ERROR_INVALID_PARAMETER for a nonexistent PID
            return True
        raise OSError("previous_player_unconfirmed") from None
    try:
        return process_creation_time(handle) != created or (
            _winapi.WaitForSingleObject(handle, 0) == _winapi.WAIT_OBJECT_0
        )
    finally:
        _winapi.CloseHandle(handle)
