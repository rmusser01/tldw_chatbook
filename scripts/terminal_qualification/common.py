#!/usr/bin/env python3
"""Prepare and collect content-free TASK-22512 qualification rows."""

from __future__ import annotations

import argparse
import base64
import binascii
import contextlib
import hashlib
import json
import ntpath
import os
import platform
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from email.parser import BytesParser
from importlib import metadata
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Sequence


SCHEMA_VERSION = 1
ALLOWED_STATUSES = {"PASS", "FAIL", "UNAVAILABLE", "UNSUPPORTED_FAIL_CLOSED"}
ALLOWED_REASON_CATEGORIES = {
    "mandatory-shell-unavailable",
    "native-windows-host-required",
    "optional-shell-unavailable",
}
ALLOWED_CLASSIFICATIONS = {"adapter-capped", "static", "viewport-bounded"}
POSIX_COLLECTED_PROBES = frozenset(
    {
        "artifacts",
        "environment-default",
        "environment-bash",
        "environment-zsh",
        "pyte",
        "pywinpty",
    }
)
WINDOWS_COLLECTED_PROBES = frozenset(
    {
        "artifacts",
        "environment-default",
        "environment-powershell",
        "environment-cmd",
        "pyte",
        "pywinpty",
    }
)
# Retained for callers that mean the already-published POSIX qualification rows.
REQUIRED_COLLECTED_PROBES = POSIX_COLLECTED_PROBES
ALL_COLLECTED_PROBES = POSIX_COLLECTED_PROBES | WINDOWS_COLLECTED_PROBES
CURRENT_GENERATION_MARKER = ".current-generation"
PENDING_PUBLICATION_MARKER = ".publication-pending"
RECOVERY_DIRECTORY_PREFIX = ".publication-recovery-"
PUBLICATION_SCHEMA_VERSION = 1
OUTPUT_READ_CHUNK = 8 * 1024
WINDOWS_BOOTSTRAP_RELEASE = b"TLDW_TASK22512_JOB_ADMITTED\n"
SECRET_LIKE_RE = re.compile(
    r"(?:(?<![A-Za-z])sk-[a-z0-9_-]{8,}|AKIA[0-9A-Z]{16}|"
    r"ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\."
    r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])|"
    r"(?:authorization\s*:\s*)?bearer\s+[A-Za-z0-9._~+/=-]{16,}|"
    r"-----BEGIN [A-Z ]+PRIVATE KEY-----|"
    r"://[^/\s:@]+:[^@\s]+@|"
    r"(?:credential|password|passwd|secret|token|api[_-]?key|access[_-]?key)"
    r"\s*(?:=|:)\s*[^\s,;]{8,}|"
    r"--(?:credential|password|secret|token|api[_-]?key)"
    r"(?:=|\s+)[^\s,;]{8,})",
    re.IGNORECASE,
)
SENSITIVE_ARG_RE = re.compile(
    r"^--(?:credential|password|secret|token|api[_-]?key)$",
    re.IGNORECASE,
)
DIST_NAME_RE = re.compile(r"[-_.]+")

ROOT_REQUIRED_KEYS = {
    "schema_version",
    "row_id",
    "probe",
    "status",
    "mandatory",
    "started_at_utc",
    "completed_at_utc",
    "elapsed_seconds",
    "command",
    "platform",
    "measurements",
    "runtime",
    "rows",
}
ROOT_PROBE_KEYS = {
    "artifacts": {
        "artifacts",
        "failure_category",
        "requirements",
        "resolved_distributions",
    },
    "pyte": {"term"},
    "pywinpty": {"reason_category"},
}
ROOT_ENVIRONMENT_KEYS = {
    "account_profile_candidate_count",
    "actual_startup",
    "initial_key_count",
    "initial_keys",
    "reason_category",
    "selected_shell_family",
    "sensitive_initial_key_count",
    "synthetic_profile",
}
COMMAND_KEYS = {"argv", "working_directory"}
PLATFORM_KEYS = {
    "architecture",
    "os",
    "os_release",
    "os_version",
    "python_executable_name",
    "python_implementation",
    "python_version",
}
MEASUREMENT_KEYS = {"current_rss_bytes", "peak_rss_bytes"}
RUNTIME_HOST_KEYS = {"kind"}
RUNTIME_DOCKER_KEYS = {"container_id", "image", "image_id", "kind"}
ARTIFACT_KEYS = {
    "filename",
    "kind",
    "license",
    "license_classifiers",
    "license_expression",
    "license_files",
    "name",
    "sha256",
    "sha256_after_install",
    "sha256_before_install",
    "size_bytes",
    "tags",
    "version",
}
ARTIFACT_REQUIRED_KEYS = {
    "filename",
    "kind",
    "sha256",
    "sha256_after_install",
    "sha256_before_install",
    "size_bytes",
}
LICENSE_FILE_KEYS = {"name", "sha256"}
RESOLVED_DISTRIBUTION_KEYS = {
    "name",
    "primary_file",
    "primary_file_sha256",
    "record_file",
    "record_file_sha256",
    "version",
}
SHELL_RESULT_KEYS = {
    "capture_within_bound",
    "captured_byte_count",
    "command_discovery",
    "module_discovery",
    "output_overflowed",
    "default_module_discovery",
    "profile_extended_module_discovery",
    "profile_contract_applicable",
    "profile_marker_present",
    "sensitive_key_repopulated_by_profile",
    "startup_completed",
}
ROW_BASE_KEYS = {"id", "mandatory", "status"}
ENVIRONMENT_ROW_FACT_KEYS = {
    "available",
    "initial_key_count",
    "reason_category",
    "sensitive_initial_key_count",
}
ROW_SCHEMAS = {
    "artifacts": {
        "artifact-download-hash-offline-install": {"artifact_count"},
    },
    "environment-default": {
        "environment-default-shell": ENVIRONMENT_ROW_FACT_KEYS,
    },
    "environment-bash": {
        "environment-bash": ENVIRONMENT_ROW_FACT_KEYS,
    },
    "environment-zsh": {
        "environment-zsh": ENVIRONMENT_ROW_FACT_KEYS,
    },
    "environment-powershell": {
        "environment-powershell": ENVIRONMENT_ROW_FACT_KEYS,
    },
    "environment-cmd": {
        "environment-cmd": ENVIRONMENT_ROW_FACT_KEYS,
    },
    "pyte": {
        "package-pyte-0.8.2": {
            "artifact_filename",
            "artifact_sha256",
            "artifact_size_bytes",
            "artifact_verified_during_probe",
            "distribution_version",
            "primary_file_name",
            "primary_file_sha256",
            "record_file_name",
            "record_file_sha256",
        },
        "parser-shell-captures": {
            "available_count",
            "captured_byte_count",
            "captured_count",
        },
        "parser-powershell-cmd-fixtures": {"fixture_byte_count", "fixture_count"},
        "parser-full-screen-programs": {
            "captured_byte_count",
            "class_available_counts",
            "class_clean_exit",
            "class_interactive_markers",
            "class_pass",
            "fixture_count",
            "real_program_count",
        },
        "parser-unicode-cells": {
            "combining_normalized",
            "cursor_column",
            "fixture_count",
            "wide_placeholder_count",
        },
        "parser-alternate-screen": {
            "alternate_isolated",
            "control_sequence_count",
            "entered",
            "entry_count",
            "exit_count",
            "exited",
            "primary_restored",
        },
        "parser-resize": {"columns", "lines"},
        "parser-bracketed-paste": {"fixture_byte_count"},
        "parser-terminal-queries": {"fixture_byte_count"},
        "parser-malformed-controls": {"fixture_byte_count"},
        "parser-incomplete-sequence-bounds": {
            "accepted_fixture_count",
            "control_sequence_byte_limit",
            "csi_parameter_digit_limit",
            "csi_parameter_limit",
            "csi_parameter_value_limit",
            "csi_private_intermediate_byte_limit",
            "non_csi_control_byte_limit",
            "rejected_fixture_count",
            "string_control_byte_limit",
        },
        "parser-mutable-collections": {
            "classifications",
            "observed_mutable_names",
            "unknown_mutable_count",
        },
        "parser-memory-bound": {
            "feed_row_count",
            "limit_bytes",
            "tracemalloc_peak_bytes",
            "viewport_columns",
            "viewport_rows",
        },
    },
    "pywinpty": {
        "package-pywinpty-3.0.5": {
            "artifact_filename",
            "artifact_sha256",
            "artifact_size_bytes",
            "artifact_verified_during_probe",
            "distribution_version",
            "native_execution",
            "primary_file_name",
            "primary_file_sha256",
            "record_file_name",
            "record_file_sha256",
        },
        "windows-platform-floor": {
            "minimum_windows_build",
            "native_execution",
            "windows_build",
        },
        "windows-low-level-api": {
            "fresh_worker",
            "low_level_api",
            "native_execution",
            "worker_std_streams_fd_backed",
        },
        "windows-conpty-only": {"constructed", "native_execution"},
        "windows-job-admission-membership": {
            "job_admitted_before_conpty",
            "job_member_count",
            "job_membership_complete",
            "native_execution",
            "normal_cleanup_all_wait_object_0",
            "normal_cleanup_expected_process_count",
            "normal_cleanup_retained_handle_count",
            "normal_cleanup_wait_object_0_count",
            "terminal_child_member_before_crash",
            "terminal_grandchild_member_before_crash",
        },
        "windows-handle-inheritance": {
            "job_handle_non_inheritable",
            "native_execution",
        },
        "windows-one-credit-bounded-read": {
            "credit_limit_bytes",
            "max_concurrent_readers",
            "max_unacknowledged_credits",
            "measured_chunk_count",
            "native_execution",
            "one_credit_max_bytes",
            "read_api_accepts_size",
            "upstream_read_buffer_bytes",
        },
        "windows-concurrent-io-close": {
            "cancel_completed",
            "cancel_completed_at_handoff",
            "cancel_completed_post_close",
            "cancel_entered",
            "concurrent_operation_count",
            "inflight_operation_category_count",
            "io_inflight_at_handoff",
            "max_concurrent_readers",
            "native_execution",
            "priority_close_completed",
            "priority_close_preempted_inflight",
            "quiet_terminal_quiescent_before_handoff",
            "quiet_terminal_startup_drained",
            "read_completed_at_handoff",
            "read_completed_post_close",
            "read_entered",
            "read_returned_after_close",
            "resize_completed",
            "resize_completed_at_handoff",
            "resize_completed_post_close",
            "resize_entered",
            "resize_returned_after_close",
            "cancel_returned_after_close",
            "write_completed_at_handoff",
            "write_completed_post_close",
            "write_entered",
            "write_returned_after_close",
            "write_completed",
        },
        "windows-profile-module-discovery": {
            "default_module_discovery",
            "native_execution",
            "profile_extended_module_discovery",
            "profile_module_discovery",
        },
        "windows-unicode-alternate-screen": {
            "alternate_isolated",
            "alternate_screen",
            "native_execution",
            "primary_restored",
            "unicode_roundtrip",
        },
        "windows-app-crash-descendant-cleanup": {
            "app_crash_observed",
            "crash_all_descendants_wait_object_0",
            "crash_app_process_separate",
            "crash_app_sole_job_handle_owner",
            "crash_descendant_set_stable",
            "crash_descendants_ready_before_abort",
            "crash_job_handle_non_inheritable",
            "crash_known_descendant_count",
            "crash_supervisor_job_handle_count",
            "crash_supervisor_synchronize_handle_count",
            "crash_wait_object_0_count",
            "crash_worker_admitted_before_conpty",
            "native_execution",
        },
        "windows-eof-output-integrity": {
            "eof_observed",
            "captured_byte_count",
            "digest_equal",
            "missing_eof_bounded",
            "native_execution",
            "output_integrity",
            "post_exit_drain_bounded",
            "sequence_complete",
            "terminal_child_crash_observed",
            "terminal_child_eof_observed",
        },
        "four-session-managed-rss": {
            "four_session_count",
            "four_session_rss_delta_bytes",
            "four_session_rss_limit_bytes",
            "native_execution",
            "rss_controller_process_count",
            "rss_crash_session_present",
            "rss_fixture_process_count",
            "rss_fixture_processes_excluded",
            "rss_helper_process_count",
            "rss_ipc_included_in_worker",
            "rss_measurement_complete",
            "rss_sample_live_session_count",
            "rss_worker_process_count",
        },
    },
}
ROW_KEYS = ROW_BASE_KEYS | set().union(
    *(fields for schemas in ROW_SCHEMAS.values() for fields in schemas.values())
)
ROW_BOOL_KEYS = {
    "alternate_isolated",
    "alternate_screen",
    "app_crash_observed",
    "available",
    "artifact_verified_during_probe",
    "cancel_completed",
    "cancel_completed_at_handoff",
    "cancel_completed_post_close",
    "cancel_entered",
    "combining_normalized",
    "constructed",
    "crash_all_descendants_wait_object_0",
    "crash_app_process_separate",
    "crash_app_sole_job_handle_owner",
    "crash_descendant_set_stable",
    "crash_descendants_ready_before_abort",
    "crash_job_handle_non_inheritable",
    "crash_worker_admitted_before_conpty",
    "entered",
    "eof_observed",
    "exited",
    "fresh_worker",
    "io_inflight_at_handoff",
    "job_admitted_before_conpty",
    "job_handle_non_inheritable",
    "job_membership_complete",
    "low_level_api",
    "mandatory",
    "missing_eof_bounded",
    "native_execution",
    "normal_cleanup_all_wait_object_0",
    "output_integrity",
    "primary_restored",
    "priority_close_completed",
    "priority_close_preempted_inflight",
    "quiet_terminal_quiescent_before_handoff",
    "quiet_terminal_startup_drained",
    "profile_module_discovery",
    "default_module_discovery",
    "profile_extended_module_discovery",
    "read_api_accepts_size",
    "read_completed_at_handoff",
    "read_completed_post_close",
    "read_entered",
    "read_returned_after_close",
    "resize_completed",
    "resize_completed_at_handoff",
    "resize_completed_post_close",
    "resize_entered",
    "resize_returned_after_close",
    "cancel_returned_after_close",
    "write_returned_after_close",
    "digest_equal",
    "post_exit_drain_bounded",
    "sequence_complete",
    "terminal_child_crash_observed",
    "terminal_child_eof_observed",
    "terminal_child_member_before_crash",
    "terminal_grandchild_member_before_crash",
    "rss_crash_session_present",
    "rss_fixture_processes_excluded",
    "rss_ipc_included_in_worker",
    "rss_measurement_complete",
    "unicode_roundtrip",
    "worker_std_streams_fd_backed",
    "write_completed",
    "write_completed_at_handoff",
    "write_completed_post_close",
    "write_entered",
}
ROW_INT_KEYS = (
    ROW_KEYS
    - ROW_BOOL_KEYS
    - {
        "artifact_filename",
        "artifact_sha256",
        "class_available_counts",
        "class_clean_exit",
        "class_interactive_markers",
        "class_pass",
        "classifications",
        "distribution_version",
        "id",
        "observed_mutable_names",
        "primary_file_name",
        "primary_file_sha256",
        "record_file_name",
        "record_file_sha256",
        "reason_category",
        "status",
    }
)


class QualificationError(RuntimeError):
    """Raised when evidence cannot be collected without weakening the contract."""


def required_collected_probes(platform_os: str) -> frozenset[str]:
    """Return the one exact six-probe family approved for a platform."""
    if platform_os in {"Darwin", "Linux"}:
        return POSIX_COLLECTED_PROBES
    if platform_os == "Windows":
        return WINDOWS_COLLECTED_PROBES
    raise QualificationError("raw evidence platform is unsupported")


def _required_probes_for_payloads(
    payloads: Sequence[dict[str, Any]],
) -> frozenset[str]:
    if not payloads:
        raise QualificationError("raw evidence sibling set is empty")
    platform_value = payloads[0].get("platform")
    platform_os = platform_value.get("os") if isinstance(platform_value, dict) else None
    if not isinstance(platform_os, str):
        raise QualificationError("raw evidence sibling platform is invalid")
    return required_collected_probes(platform_os)


def _validate_exact_probe_set(payloads: Sequence[dict[str, Any]]) -> None:
    required = _required_probes_for_payloads(payloads)
    probes = {str(payload.get("probe")) for payload in payloads}
    if probes != required or len(payloads) != len(required):
        raise QualificationError("raw evidence sibling probe set is incomplete")


@dataclass(frozen=True)
class BoundedResult:
    """Bounded, content-capped result from one owned subprocess group."""

    args: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool
    terminated: bool
    killed: bool
    overflowed: bool
    stored_output_bytes: int


@dataclass(frozen=True)
class WindowsBootstrapSetup:
    """Fixture writes performed only after disposable-user admission."""

    profile_files: tuple[tuple[str, str], ...] = ()
    registry_values: tuple[tuple[str, str, str], ...] = ()

    def as_json(self) -> dict[str, list[list[str]]]:
        return {
            "profile_files": [list(item) for item in self.profile_files],
            "registry_values": [list(item) for item in self.registry_values],
        }


@dataclass(frozen=True)
class DisposableProfileIdentity:
    """Verified identity and profile created for one Windows probe."""

    username: str
    sid: str
    profile_path: Path


class OwnedProcessJob:
    """Own a Windows process tree and kill it when the Job handle closes."""

    def __init__(self) -> None:
        self.handle: object | None = None
        if os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes

        class IoCounters(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_uint64)
                for name in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            ]

        class BasicLimit(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class ExtendedLimit(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimit),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise QualificationError("bounded subprocess Job creation failed")
        limits = ExtendedLimit()
        limits.BasicLimitInformation.LimitFlags = 0x00002000
        if not kernel32.SetInformationJobObject(
            handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            kernel32.CloseHandle(handle)
            raise QualificationError("bounded subprocess Job setup failed")
        self.handle = handle
        self._kernel32 = kernel32

    def assign(self, process: subprocess.Popen[bytes]) -> None:
        if self.handle is None:
            return
        process_handle = getattr(process, "_handle", None)
        if process_handle is None or not self._kernel32.AssignProcessToJobObject(
            self.handle, process_handle
        ):
            self.close()
            raise QualificationError("bounded subprocess Job admission failed")

    def contains(self, process: subprocess.Popen[bytes]) -> bool:
        """Independently verify that a process belongs to this exact Job."""
        if self.handle is None:
            return os.name != "nt"
        import ctypes
        from ctypes import wintypes

        process_handle = getattr(process, "_handle", None)
        if process_handle is None:
            raise QualificationError(
                "bounded subprocess Job membership verification failed"
            )
        result = wintypes.BOOL()
        self._kernel32.IsProcessInJob.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.BOOL),
        ]
        if not self._kernel32.IsProcessInJob(
            process_handle, self.handle, ctypes.byref(result)
        ):
            raise QualificationError(
                "bounded subprocess Job membership verification failed"
            )
        return bool(result.value)

    def close(self) -> None:
        if self.handle is not None:
            self._kernel32.CloseHandle(self.handle)
            self.handle = None


class _WindowsHandleProcess:
    """The small Popen-compatible surface needed for alternate-user launch."""

    def __init__(
        self,
        *,
        args: Sequence[str],
        process_handle: object,
        pid: int,
        stdin: BinaryIO,
        stdout: BinaryIO,
        stderr: BinaryIO,
    ) -> None:
        import ctypes
        from ctypes import wintypes

        self.args = tuple(args)
        self._handle = process_handle
        self.pid = pid
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.returncode: int | None = None
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.WaitForSingleObject.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
        ]
        self._kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        if self._kernel32.WaitForSingleObject(self._handle, 0) == 0:
            code = self._exit_code()
            self.returncode = code
            return code
        return None

    def wait(self, timeout: float | None = None) -> int:
        milliseconds = 0xFFFFFFFF if timeout is None else max(0, int(timeout * 1000))
        result = self._kernel32.WaitForSingleObject(self._handle, milliseconds)
        if result == 0x00000102:
            raise subprocess.TimeoutExpired(self.args, timeout)
        if result != 0:
            raise OSError("alternate-user process wait failed")
        self.returncode = self._exit_code()
        return self.returncode

    def _exit_code(self) -> int:
        import ctypes
        from ctypes import wintypes

        code = wintypes.DWORD()
        if not self._kernel32.GetExitCodeProcess(self._handle, ctypes.byref(code)):
            raise OSError("alternate-user process exit status failed")
        return int(code.value)

    def terminate(self) -> None:
        if not self._kernel32.TerminateProcess(self._handle, 121):
            raise OSError("alternate-user process termination failed")

    def send_signal(self, requested_signal: int) -> None:
        if requested_signal == getattr(subprocess, "CTRL_BREAK_EVENT", -1):
            if not self._kernel32.GenerateConsoleCtrlEvent(requested_signal, self.pid):
                raise OSError("alternate-user CTRL_BREAK_EVENT failed")
            return
        self.terminate()


class _NativeWindowsRegistryApi:
    """Write the current process's user hive without a predefined HKCU handle."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise QualificationError("native Windows registry API is unavailable")
        import ctypes

        self.ctypes = ctypes
        self.advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

    def verify_disposable_user(self, username: str) -> bool:
        from ctypes import wintypes

        size = wintypes.DWORD(257)
        buffer = self.ctypes.create_unicode_buffer(size.value)
        if not self.advapi32.GetUserNameW(buffer, self.ctypes.byref(size)):
            return False
        return buffer.value.casefold() == username.casefold()

    def open_current_user(self) -> object:
        from ctypes import wintypes

        handle = wintypes.HANDLE()
        status = self.advapi32.RegOpenCurrentUser(0x0002001F, self.ctypes.byref(handle))
        if status != 0:
            raise QualificationError("disposable user registry hive is unavailable")
        return handle

    def set_string(self, root: object, subkey: str, name: str, value: str) -> None:
        from ctypes import wintypes

        key = wintypes.HANDLE()
        disposition = wintypes.DWORD()
        status = self.advapi32.RegCreateKeyExW(
            root,
            subkey,
            0,
            None,
            0,
            0x0002,
            None,
            self.ctypes.byref(key),
            self.ctypes.byref(disposition),
        )
        if status != 0:
            raise QualificationError("disposable user registry key creation failed")
        try:
            encoded = (value + "\x00").encode("utf-16-le")
            buffer = self.ctypes.create_string_buffer(encoded)
            status = self.advapi32.RegSetValueExW(
                key,
                name,
                0,
                1,
                self.ctypes.cast(buffer, self.ctypes.POINTER(self.ctypes.c_ubyte)),
                len(encoded),
            )
            if status != 0:
                raise QualificationError("disposable user registry value write failed")
        finally:
            self.advapi32.RegCloseKey(key)

    def close_key(self, root: object) -> None:
        self.advapi32.RegCloseKey(root)


class _NativeWindowsProfileApi:
    """Create, launch as, verify, and remove one disposable local account."""

    LOGON_WITH_PROFILE = 0x00000001

    def __init__(self) -> None:
        if os.name != "nt":
            raise QualificationError(
                "disposable Windows account/profile precondition unavailable"
            )
        import ctypes

        self.ctypes = ctypes
        self.advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.netapi32 = ctypes.WinDLL("netapi32", use_last_error=True)
        self.userenv = ctypes.WinDLL("userenv", use_last_error=True)

    def create_account(self, username: str, password: str) -> None:
        from ctypes import wintypes

        class UserInfo1(self.ctypes.Structure):
            _fields_ = [
                ("name", wintypes.LPWSTR),
                ("password", wintypes.LPWSTR),
                ("password_age", wintypes.DWORD),
                ("privilege", wintypes.DWORD),
                ("home_dir", wintypes.LPWSTR),
                ("comment", wintypes.LPWSTR),
                ("flags", wintypes.DWORD),
                ("script_path", wintypes.LPWSTR),
            ]

        info = UserInfo1(
            username,
            password,
            0,
            1,
            None,
            "TASK-22512 disposable qualification account",
            0x00000001 | 0x00010000,
            None,
        )
        parameter_error = wintypes.DWORD()
        status = self.netapi32.NetUserAdd(
            None, 1, self.ctypes.byref(info), self.ctypes.byref(parameter_error)
        )
        if status != 0:
            raise PermissionError(f"NetUserAdd failed: {status}")

    def _account_sid(self, username: str) -> str:
        from ctypes import wintypes

        sid_size = wintypes.DWORD()
        domain_size = wintypes.DWORD()
        sid_type = wintypes.DWORD()
        self.advapi32.LookupAccountNameW(
            None,
            username,
            None,
            self.ctypes.byref(sid_size),
            None,
            self.ctypes.byref(domain_size),
            self.ctypes.byref(sid_type),
        )
        if not sid_size.value:
            raise OSError("LookupAccountNameW sizing failed")
        sid = self.ctypes.create_string_buffer(sid_size.value)
        domain = self.ctypes.create_unicode_buffer(max(1, domain_size.value))
        if not self.advapi32.LookupAccountNameW(
            None,
            username,
            sid,
            self.ctypes.byref(sid_size),
            domain,
            self.ctypes.byref(domain_size),
            self.ctypes.byref(sid_type),
        ):
            raise OSError("LookupAccountNameW failed")
        text = wintypes.LPWSTR()
        if not self.advapi32.ConvertSidToStringSidW(sid, self.ctypes.byref(text)):
            raise OSError("ConvertSidToStringSidW failed")
        try:
            return str(text.value)
        finally:
            self.kernel32.LocalFree(text)

    def create_profile(self, username: str) -> DisposableProfileIdentity:
        sid = self._account_sid(username)
        path = self.ctypes.create_unicode_buffer(32768)
        result = self.userenv.CreateProfile(sid, username, path, len(path))
        if result != 0:
            raise PermissionError(f"CreateProfile failed: {result}")
        return DisposableProfileIdentity(username, sid, Path(path.value))

    def delete_profile(self, identity: DisposableProfileIdentity) -> None:
        if not self.userenv.DeleteProfileW(
            identity.sid, str(identity.profile_path), None
        ):
            raise OSError("DeleteProfileW failed")

    def delete_account(self, username: str) -> None:
        status = self.netapi32.NetUserDel(None, username)
        if status not in {0, 2221}:
            raise OSError(f"NetUserDel failed: {status}")

    def launch_waiting_bootstrap(
        self,
        argv: Sequence[str],
        *,
        username: str,
        password: str,
        identity: DisposableProfileIdentity,
        **kwargs: object,
    ) -> _WindowsHandleProcess:
        import msvcrt
        from ctypes import wintypes

        bootstrap_copy = identity.profile_path / "task22512_bounded_bootstrap.py"
        shutil.copy2(Path(__file__).resolve(), bootstrap_copy)
        launch_argv = list(argv)
        if len(launch_argv) < 2:
            raise QualificationError("bounded bootstrap command is invalid")
        launch_argv[1] = str(bootstrap_copy)
        pipe_fds = [*os.pipe(), *os.pipe(), *os.pipe()]
        (
            stdin_read,
            stdin_write,
            stdout_read,
            stdout_write,
            stderr_read,
            stderr_write,
        ) = pipe_fds
        child_fds = (stdin_read, stdout_write, stderr_write)
        parent_fds = (stdin_write, stdout_read, stderr_read)
        for descriptor in child_fds:
            os.set_inheritable(descriptor, True)
        for descriptor in parent_fds:
            os.set_inheritable(descriptor, False)

        class StartupInfo(self.ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("reserved", wintypes.LPWSTR),
                ("desktop", wintypes.LPWSTR),
                ("title", wintypes.LPWSTR),
                ("x", wintypes.DWORD),
                ("y", wintypes.DWORD),
                ("x_size", wintypes.DWORD),
                ("y_size", wintypes.DWORD),
                ("x_count_chars", wintypes.DWORD),
                ("y_count_chars", wintypes.DWORD),
                ("fill_attribute", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("show_window", wintypes.WORD),
                ("reserved2_size", wintypes.WORD),
                ("reserved2", self.ctypes.POINTER(self.ctypes.c_ubyte)),
                ("stdin", wintypes.HANDLE),
                ("stdout", wintypes.HANDLE),
                ("stderr", wintypes.HANDLE),
            ]

        class ProcessInformation(self.ctypes.Structure):
            _fields_ = [
                ("process", wintypes.HANDLE),
                ("thread", wintypes.HANDLE),
                ("process_id", wintypes.DWORD),
                ("thread_id", wintypes.DWORD),
            ]

        startup = StartupInfo()
        startup.cb = self.ctypes.sizeof(startup)
        startup.flags = 0x00000100
        startup.stdin = msvcrt.get_osfhandle(stdin_read)
        startup.stdout = msvcrt.get_osfhandle(stdout_write)
        startup.stderr = msvcrt.get_osfhandle(stderr_write)
        process_info = ProcessInformation()
        command_line = self.ctypes.create_unicode_buffer(
            subprocess.list2cmdline(launch_argv)
        )
        try:
            created = self.advapi32.CreateProcessWithLogonW(
                username,
                ".",
                password,
                self.LOGON_WITH_PROFILE,
                None,
                command_line,
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200),
                None,
                str(identity.profile_path),
                self.ctypes.byref(startup),
                self.ctypes.byref(process_info),
            )
        finally:
            for descriptor in child_fds:
                os.close(descriptor)
        if not created:
            for descriptor in parent_fds:
                os.close(descriptor)
            raise PermissionError(
                f"CreateProcessWithLogonW failed: {self.ctypes.get_last_error()}"
            )
        self.kernel32.CloseHandle(process_info.thread)
        return _WindowsHandleProcess(
            args=launch_argv,
            process_handle=process_info.process,
            pid=int(process_info.process_id),
            stdin=os.fdopen(stdin_write, "wb", buffering=0),
            stdout=os.fdopen(stdout_read, "rb", buffering=0),
            stderr=os.fdopen(stderr_read, "rb", buffering=0),
        )

    def verify_process_identity_profile(
        self,
        process: object,
        identity: DisposableProfileIdentity,
    ) -> bool:
        from ctypes import wintypes

        process_handle = getattr(process, "_handle", None)
        if process_handle is None:
            return False
        token = wintypes.HANDLE()
        if not self.advapi32.OpenProcessToken(
            process_handle, 0x0008, self.ctypes.byref(token)
        ):
            return False
        try:
            required = wintypes.DWORD()
            self.advapi32.GetTokenInformation(
                token, 1, None, 0, self.ctypes.byref(required)
            )
            if not required.value:
                return False
            token_user = self.ctypes.create_string_buffer(required.value)
            if not self.advapi32.GetTokenInformation(
                token,
                1,
                token_user,
                required,
                self.ctypes.byref(required),
            ):
                return False
            sid_pointer = self.ctypes.cast(
                token_user, self.ctypes.POINTER(self.ctypes.c_void_p)
            )[0]
            sid_text = wintypes.LPWSTR()
            if not self.advapi32.ConvertSidToStringSidW(
                sid_pointer, self.ctypes.byref(sid_text)
            ):
                return False
            try:
                if str(sid_text.value).casefold() != identity.sid.casefold():
                    return False
            finally:
                self.kernel32.LocalFree(sid_text)
            size = wintypes.DWORD(32768)
            profile_path = self.ctypes.create_unicode_buffer(size.value)
            if not self.userenv.GetUserProfileDirectoryW(
                token, profile_path, self.ctypes.byref(size)
            ):
                return False
            if os.path.normcase(
                os.path.normpath(profile_path.value)
            ) != os.path.normcase(os.path.normpath(str(identity.profile_path))):
                return False
            hive = wintypes.HANDLE()
            status = self.advapi32.RegOpenKeyExW(
                wintypes.HANDLE(0x80000003),
                identity.sid,
                0,
                0x00020019,
                self.ctypes.byref(hive),
            )
            if status != 0:
                return False
            self.advapi32.RegCloseKey(hive)
            return True
        finally:
            self.kernel32.CloseHandle(token)


class _DisposableWindowsProfile:
    """Own one local account and its profile for exactly one bounded probe."""

    def __init__(
        self,
        api: object,
        *,
        username: str,
        password: str,
    ) -> None:
        self.api = api
        self.username = username
        self._password = password
        self.identity: DisposableProfileIdentity | None = None
        self._account_created = False

    @classmethod
    def native(cls) -> _DisposableWindowsProfile:
        if os.name != "nt":
            raise QualificationError(
                "disposable Windows account/profile precondition unavailable"
            )
        suffix = secrets.token_hex(5)
        username = f"tldwq{suffix}"
        password = f"Tldw!{secrets.token_urlsafe(24)}9a"
        return cls(_NativeWindowsProfileApi(), username=username, password=password)

    def __enter__(self) -> _DisposableWindowsProfile:
        try:
            self.api.create_account(self.username, self._password)
            self._account_created = True
            identity = self.api.create_profile(self.username)
            if (
                not isinstance(identity, DisposableProfileIdentity)
                or identity.username != self.username
                or not identity.sid.startswith("S-")
                or not ntpath.isabs(str(identity.profile_path))
            ):
                raise QualificationError(
                    "disposable Windows profile identity is invalid"
                )
            self.identity = identity
            return self
        except Exception as exc:
            cleanup_failures = self._cleanup()
            if cleanup_failures:
                raise ExceptionGroup(
                    "disposable Windows profile setup and cleanup failed",
                    [exc, *cleanup_failures],
                ) from None
            raise QualificationError(
                "disposable Windows account/profile precondition unavailable"
            ) from exc

    def __exit__(
        self,
        exc_type: object,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        del exc_type, traceback
        cleanup_failures = self._cleanup()
        if cleanup_failures:
            failures: list[BaseException] = [*cleanup_failures]
            message = "disposable Windows profile cleanup failed"
            if exc is not None:
                failures.insert(0, exc)
                message = "disposable Windows profile body and cleanup failed"
            if all(isinstance(failure, Exception) for failure in failures):
                raise ExceptionGroup(message, failures) from None
            raise BaseExceptionGroup(message, failures) from None
        return False

    def _cleanup(self) -> list[Exception]:
        failures: list[Exception] = []
        if self.identity is not None:
            try:
                self.api.delete_profile(self.identity)
            except Exception as exc:  # cleanup must continue to account deletion
                failures.append(exc)
            self.identity = None
        if self._account_created:
            try:
                self.api.delete_account(self.username)
            except Exception as exc:  # both cleanup operations are always attempted
                failures.append(exc)
            self._account_created = False
        return failures

    def launch_waiting_bootstrap(
        self,
        argv: Sequence[str],
        **kwargs: object,
    ) -> object:
        if self.identity is None:
            raise QualificationError("disposable Windows profile is not active")
        return self.api.launch_waiting_bootstrap(
            argv,
            username=self.username,
            password=self._password,
            identity=self.identity,
            **kwargs,
        )

    def verify_process_identity_profile(self, process: object) -> None:
        if self.identity is None:
            raise QualificationError("disposable Windows profile is not active")
        if not self.api.verify_process_identity_profile(process, self.identity):
            raise QualificationError(
                "disposable Windows process identity/profile verification failed"
            )


def _install_disposable_registry_values(
    registry: object,
    *,
    username: str,
    values: Sequence[tuple[str, str, str]],
) -> None:
    """Write only through the verified disposable process's current-user handle."""
    if not registry.verify_disposable_user(username):
        raise QualificationError(
            "disposable user registry identity verification failed"
        )
    root = registry.open_current_user()
    if root is None:
        raise QualificationError("disposable user registry hive is unavailable")
    try:
        for subkey, name, value in values:
            registry.set_string(root, subkey, name, value)
    finally:
        registry.close_key(root)


def _current_process_is_in_job() -> bool:
    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    result = wintypes.BOOL()
    if not kernel32.IsProcessInJob(
        kernel32.GetCurrentProcess(), None, ctypes.byref(result)
    ):
        return False
    return bool(result.value)


def _bootstrap_environment(overrides: dict[str, str]) -> dict[str, str]:
    """Build a scrubbed environment from the disposable process's own profile."""
    allowed = {
        "APPDATA",
        "COMSPEC",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "PATH",
        "PATHEXT",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PROGRAMW6432",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERDOMAIN",
        "USERNAME",
        "USERPROFILE",
        "WINDIR",
    }
    ambient = {key.upper(): value for key, value in os.environ.items()}
    result = {key: ambient[key] for key in allowed if ambient.get(key)}
    for key, value in overrides.items():
        if key in {"LANG", "LC_ALL", "TERM"} and value and "\x00" not in value:
            result[key] = value
    return result


def _apply_windows_bootstrap_setup(
    setup: WindowsBootstrapSetup,
    *,
    expected_username: str,
    registry_factory: Callable[[], object] | None = None,
) -> None:
    profile_value = os.environ.get("USERPROFILE")
    if not profile_value:
        raise QualificationError("disposable Windows profile path is unavailable")
    profile_root = Path(profile_value).resolve()
    for relative_text, content in setup.profile_files:
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise QualificationError(
                "disposable Windows profile fixture path is unsafe"
            )
        destination = (profile_root / relative).resolve()
        if profile_root not in destination.parents:
            raise QualificationError("disposable Windows profile fixture escaped")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    if setup.registry_values:
        factory = registry_factory or _NativeWindowsRegistryApi
        _install_disposable_registry_values(
            factory(), username=expected_username, values=setup.registry_values
        )


def _decode_bootstrap_payload(encoded: str) -> dict[str, Any]:
    try:
        padding = "=" * (-len(encoded) % 4)
        raw = base64.urlsafe_b64decode((encoded + padding).encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except (binascii.Error, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise QualificationError("bounded bootstrap payload is invalid") from exc
    if not isinstance(payload, dict):
        raise QualificationError("bounded bootstrap payload is invalid")
    return payload


def _bootstrap_setup_from_json(value: object) -> WindowsBootstrapSetup | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "profile_files",
        "registry_values",
    }:
        raise QualificationError("bounded bootstrap setup is invalid")

    def tuples(key: str, width: int) -> tuple[tuple[str, ...], ...]:
        items = value[key]
        if not isinstance(items, list):
            raise QualificationError("bounded bootstrap setup is invalid")
        result: list[tuple[str, ...]] = []
        for item in items:
            if (
                not isinstance(item, list)
                or len(item) != width
                or not all(isinstance(part, str) for part in item)
            ):
                raise QualificationError("bounded bootstrap setup is invalid")
            result.append(tuple(item))
        return tuple(result)

    profile_files = tuples("profile_files", 2)
    registry_values = tuples("registry_values", 3)
    return WindowsBootstrapSetup(
        profile_files=profile_files,  # type: ignore[arg-type]
        registry_values=registry_values,  # type: ignore[arg-type]
    )


def _run_bounded_bootstrap_payload(encoded: str) -> int:
    payload = _decode_bootstrap_payload(encoded)
    if set(payload) != {
        "command",
        "cwd",
        "environment",
        "expected_username",
        "setup",
    }:
        raise QualificationError("bounded bootstrap payload is invalid")
    command = payload["command"]
    cwd = payload["cwd"]
    environment = payload["environment"]
    expected_username = payload["expected_username"]
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(item, str) and item for item in command)
        or not isinstance(cwd, str)
        or not cwd
        or (environment is not None and not isinstance(environment, dict))
        or (
            isinstance(environment, dict)
            and not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in environment.items()
            )
        )
        or (expected_username is not None and not isinstance(expected_username, str))
    ):
        raise QualificationError("bounded bootstrap payload is invalid")
    selected_environment = environment
    if expected_username is not None:
        selected_environment = _bootstrap_environment(environment or {})
    return _bounded_bootstrap_main(
        command,
        cwd=Path(cwd),
        environment=selected_environment,
        setup=_bootstrap_setup_from_json(payload["setup"]),
        expected_username=expected_username,
    )


def _bounded_bootstrap_main(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
    setup: WindowsBootstrapSetup | None = None,
    expected_username: str | None = None,
    input_stream: BinaryIO | None = None,
    output_stream: BinaryIO | None = None,
    error_stream: BinaryIO | None = None,
    is_in_job: Callable[[], bool] = _current_process_is_in_job,
    popen_factory: Callable[..., object] = subprocess.Popen,
) -> int:
    """Wait for admission, verify it in-process, then and only then spawn."""
    source = input_stream or sys.stdin.buffer
    output = output_stream or sys.stdout.buffer
    error = error_stream or sys.stderr.buffer
    release = source.read(len(WINDOWS_BOOTSTRAP_RELEASE))
    if release != WINDOWS_BOOTSTRAP_RELEASE or not is_in_job():
        return 120
    try:
        if setup is not None:
            if not expected_username:
                raise QualificationError("disposable Windows username is unavailable")
            _apply_windows_bootstrap_setup(setup, expected_username=expected_username)
        child_input = source.read()
        process = popen_factory(
            tuple(command),
            cwd=cwd,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=output,
            stderr=error,
            close_fds=False,
        )
        _, _ = process.communicate(input=child_input)
        return int(process.returncode)
    except (OSError, subprocess.SubprocessError, QualificationError):
        return 121


def _bootstrap_argv(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    setup: WindowsBootstrapSetup | None,
    expected_username: str | None,
) -> tuple[str, ...]:
    payload = {
        "command": [str(item) for item in command],
        "cwd": str(cwd),
        "environment": env,
        "expected_username": expected_username,
        "setup": None if setup is None else setup.as_json(),
    }
    encoded = (
        base64.urlsafe_b64encode(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        .decode("ascii")
        .rstrip("=")
    )
    return (
        sys.executable,
        str(Path(__file__).resolve()),
        "_bounded-bootstrap",
        encoded,
    )


def _abort_unreleased_bootstrap(process: object) -> None:
    stdin = getattr(process, "stdin", None)
    if stdin is not None:
        try:
            stdin.close()
        except OSError:
            pass
    if process.poll() is None:
        try:
            process.terminate()
        except OSError:
            pass
    try:
        process.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _write_bootstrap_release(
    stream: BinaryIO,
    input_bytes: bytes | None,
) -> None:
    try:
        stream.write(WINDOWS_BOOTSTRAP_RELEASE + (input_bytes or b""))
        stream.flush()
    except (BrokenPipeError, OSError):
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass


def _launch_admitted_windows_bootstrap(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    input_bytes: bytes | None,
    job: object,
    setup: WindowsBootstrapSetup | None = None,
    profile: object | None = None,
    popen_factory: Callable[..., object] = subprocess.Popen,
) -> tuple[object, threading.Thread]:
    """Launch only a waiter, prove admission/identity, then release it."""
    expected_username = (
        profile.identity.username
        if profile is not None and getattr(profile, "identity", None) is not None
        else None
    )
    bootstrap_environment = (
        {
            key: value
            for key, value in (env or {}).items()
            if key in {"LANG", "LC_ALL", "TERM"}
        }
        if profile is not None
        else env
    )
    argv = _bootstrap_argv(
        command,
        cwd=cwd,
        env=bootstrap_environment,
        setup=setup,
        expected_username=expected_username,
    )
    kwargs: dict[str, object] = {
        "cwd": cwd,
        "env": env,
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "close_fds": True,
        "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    }
    process = (
        profile.launch_waiting_bootstrap(argv, **kwargs)
        if profile is not None
        else popen_factory(argv, **kwargs)
    )
    try:
        job.assign(process)
        if not job.contains(process):
            raise QualificationError(
                "bounded subprocess Job membership verification failed"
            )
        if profile is not None:
            profile.verify_process_identity_profile(process)
    except Exception:
        _abort_unreleased_bootstrap(process)
        raise
    stdin = getattr(process, "stdin", None)
    if stdin is None:
        _abort_unreleased_bootstrap(process)
        raise QualificationError("bounded bootstrap release pipe is unavailable")
    writer = threading.Thread(
        target=_write_bootstrap_release,
        args=(stdin, input_bytes),
        name="task22512-bootstrap-release",
        daemon=True,
    )
    writer.start()
    return process, writer


class _BoundedOutputCollector:
    """Drain two pipes while retaining at most one combined byte ceiling."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.stdout = bytearray()
        self.stderr = bytearray()
        self.overflowed = threading.Event()
        self.stop_requested = threading.Event()
        self._lock = threading.Lock()
        self._first_error: Exception | None = None

    @property
    def stored_output_bytes(self) -> int:
        with self._lock:
            return len(self.stdout) + len(self.stderr)

    @property
    def first_error(self) -> Exception | None:
        with self._lock:
            return self._first_error

    def drain(self, stream: BinaryIO, *, stderr: bool) -> None:
        destination = self.stderr if stderr else self.stdout
        try:
            while True:
                chunk = stream.read(OUTPUT_READ_CHUNK)
                if not chunk:
                    return
                with self._lock:
                    remaining = self.limit - len(self.stdout) - len(self.stderr)
                    if remaining > 0:
                        destination.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        self.overflowed.set()
                        self.stop_requested.set()
        except Exception as exc:
            with self._lock:
                if self._first_error is None:
                    self._first_error = exc
            self.stop_requested.set()
        finally:
            try:
                stream.close()
            except OSError:
                pass


def _pipe_input_writer(stream: BinaryIO, data: bytes) -> None:
    try:
        stream.write(data)
        stream.flush()
    except (BrokenPipeError, OSError):
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass


def terminate_owned_group(
    process: subprocess.Popen[bytes],
    job: OwnedProcessJob,
    *,
    grace_seconds: float,
) -> tuple[bool, bool]:
    """Terminate then kill a whole owned process group, returning actions taken."""
    terminated = False
    killed = False
    if process.poll() is not None:
        return terminated, killed
    if os.name == "nt":
        terminated = True
        try:
            process.send_signal(subprocess.CTRL_BREAK_EVENT)
            process.wait(timeout=grace_seconds)
        except (OSError, subprocess.TimeoutExpired):
            killed = True
            job.close()
    else:
        terminated = True
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            killed = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired as exc:
        raise QualificationError("owned subprocess group did not reap") from exc
    return terminated, killed


def run_bounded(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    output_limit: int,
    operation: str,
    env: dict[str, str] | None = None,
    input_bytes: bytes | None = None,
    windows_profile_setup: WindowsBootstrapSetup | None = None,
) -> BoundedResult:
    """Run one command with capped pipe drains and whole-group cleanup."""
    if timeout_seconds <= 0 or output_limit <= 0 or not operation:
        raise QualificationError("bounded subprocess contract is invalid")
    command = tuple(str(item) for item in argv)
    timed_out = False
    terminated = False
    killed = False
    process: object | None = None
    input_writer: threading.Thread | None = None
    collector = _BoundedOutputCollector(output_limit)
    readers: list[threading.Thread] = []
    profile_context: contextlib.AbstractContextManager[object | None]
    if windows_profile_setup is not None:
        if os.name != "nt":
            raise QualificationError(
                "disposable Windows account/profile precondition unavailable"
            )
        profile_context = _DisposableWindowsProfile.native()
    else:
        profile_context = contextlib.nullcontext(None)
    with profile_context as profile:
        job = OwnedProcessJob()
        try:
            if os.name == "nt":
                process, input_writer = _launch_admitted_windows_bootstrap(
                    command,
                    cwd=cwd,
                    env=env,
                    input_bytes=input_bytes,
                    job=job,
                    setup=windows_profile_setup,
                    profile=profile,
                )
            else:
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=env,
                    stdin=subprocess.PIPE
                    if input_bytes is not None
                    else subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    close_fds=True,
                    start_new_session=True,
                )
                job.assign(process)
                if not job.contains(process):
                    raise QualificationError(
                        "bounded subprocess Job membership verification failed"
                    )
                if input_bytes is not None:
                    if process.stdin is None:
                        raise QualificationError(
                            "bounded subprocess input pipe is unavailable"
                        )
                    input_writer = threading.Thread(
                        target=_pipe_input_writer,
                        args=(process.stdin, input_bytes),
                        name="task22512-bounded-input",
                        daemon=True,
                    )
                    input_writer.start()
            stdout = getattr(process, "stdout", None)
            stderr = getattr(process, "stderr", None)
            if stdout is None or stderr is None:
                raise QualificationError(
                    "bounded subprocess output pipe is unavailable"
                )
            readers = [
                threading.Thread(
                    target=collector.drain,
                    kwargs={"stream": stdout, "stderr": False},
                    name="task22512-stdout-drain",
                    daemon=True,
                ),
                threading.Thread(
                    target=collector.drain,
                    kwargs={"stream": stderr, "stderr": True},
                    name="task22512-stderr-drain",
                    daemon=True,
                ),
            ]
            for reader in readers:
                reader.start()
            deadline = time.monotonic() + timeout_seconds
            while process.poll() is None:
                if collector.first_error is not None:
                    terminated, killed = terminate_owned_group(
                        process, job, grace_seconds=min(2.0, timeout_seconds)
                    )
                    break
                if collector.overflowed.is_set():
                    terminated, killed = terminate_owned_group(
                        process, job, grace_seconds=min(2.0, timeout_seconds)
                    )
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    terminated, killed = terminate_owned_group(
                        process, job, grace_seconds=min(2.0, timeout_seconds)
                    )
                    break
                collector.stop_requested.wait(0.01)
            if process.poll() is None:
                terminated, killed = terminate_owned_group(
                    process, job, grace_seconds=min(2.0, timeout_seconds)
                )
        except OSError as exc:
            raise QualificationError(f"{operation}_launch_failed") from exc
        finally:
            if process is not None and process.poll() is None:
                extra_terminated, extra_killed = terminate_owned_group(
                    process, job, grace_seconds=1.0
                )
                terminated = terminated or extra_terminated
                killed = killed or extra_killed
            job.close()
            if input_writer is not None:
                input_writer.join(timeout=1.0)
            for reader in readers:
                reader.join(timeout=1.0)
    if process is None:
        raise QualificationError(f"{operation}_launch_failed")
    drain_error = collector.first_error
    if any(reader.is_alive() for reader in readers):
        if drain_error is not None:
            raise QualificationError(
                f"{operation}_output_drain_failed"
            ) from drain_error
        raise QualificationError("bounded subprocess output drain did not stop")
    if drain_error is not None:
        raise QualificationError(f"{operation}_output_drain_failed") from drain_error
    stdout_bytes = bytes(collector.stdout)
    stderr_bytes = bytes(collector.stderr)
    return BoundedResult(
        args=command,
        returncode=int(process.returncode),
        stdout=stdout_bytes,
        stderr=stderr_bytes,
        timed_out=timed_out,
        terminated=terminated,
        killed=killed,
        overflowed=collector.overflowed.is_set(),
        stored_output_bytes=len(stdout_bytes) + len(stderr_bytes),
    )


def utc_now() -> str:
    """Return an RFC 3339 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    """Hash a file without retaining its content."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_distribution(name: str) -> str:
    return DIST_NAME_RE.sub("-", name).lower()


def platform_facts() -> dict[str, object]:
    """Return platform facts that contain no environment or profile values."""
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_executable_name": Path(sys.executable).name,
    }


def memory_facts() -> dict[str, int | None]:
    """Return current and peak RSS where the host exposes them."""
    current: int | None = None
    peak: int | None = None
    if sys.platform.startswith("linux"):
        try:
            fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
            current = int(fields[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            current = None
    try:
        import resource

        raw_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak = raw_peak if sys.platform == "darwin" else raw_peak * 1024
    except (ImportError, OSError, ValueError):
        peak = None
    return {"current_rss_bytes": current, "peak_rss_bytes": peak}


def _atomic_json(path: Path, payload: dict[str, object], *, replace: bool) -> None:
    if path.exists() and not replace:
        raise QualificationError(f"refusing to replace existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_probe_result(
    path: Path, payload: dict[str, object], *, replace: bool
) -> None:
    """Validate and atomically write one content-free probe result."""
    validate_content_free(payload)
    _atomic_json(path, payload, replace=replace)


def command_facts() -> dict[str, object]:
    """Record the invoked interpreter and process argv without launcher rewriting."""
    original = getattr(sys, "orig_argv", None)
    invoked_interpreter = os.path.abspath(sys.executable)
    suffix = list(original[1:]) if isinstance(original, list) and original else sys.argv
    argv = [invoked_interpreter, *suffix]
    return {
        "argv": [str(item) for item in argv],
        "working_directory": str(Path.cwd().resolve()),
    }


def runtime_facts(
    kind: str,
    *,
    image: str | None = None,
    image_id: str | None = None,
    container_id: str | None = None,
) -> dict[str, object]:
    """Build an allowlisted host or Docker runtime identity."""
    if kind == "host":
        return {"kind": "host"}
    if kind != "docker" or not all((image, image_id, container_id)):
        raise QualificationError("docker runtime identity is incomplete")
    facts: dict[str, object] = {
        "kind": "docker",
        "image": image,
        "image_id": image_id,
        "container_id": container_id,
    }
    _validate_runtime(facts, path=("runtime",))
    return facts


def _validate_string(value: str, *, path: tuple[str, ...]) -> None:
    key = path[-1] if path else ""
    if not value or len(value) > 8192 or any(ord(char) < 32 for char in value):
        raise QualificationError(f"unsafe string in raw-evidence field: {key}")
    if SECRET_LIKE_RE.search(value):
        raise QualificationError(f"secret-like string in raw-evidence field: {key}")
    if key == "argv":
        return
    if key == "initial_keys" and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_()]*", value):
        return
    if key in {"row_id", "probe"} and not re.fullmatch(r"[a-z0-9][a-z0-9-]*", value):
        raise QualificationError(f"invalid identifier in raw-evidence field: {key}")
    if key == "id" and not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", value):
        raise QualificationError("invalid result-row identifier")
    if key == "status" and value not in ALLOWED_STATUSES:
        raise QualificationError("raw evidence status is invalid")
    if key == "reason_category" and value not in ALLOWED_REASON_CATEGORIES:
        raise QualificationError("raw evidence reason category is invalid")
    if key in {"buffer", "charset", "dirty", "mode", "savepoints", "tabstops"}:
        if value not in ALLOWED_CLASSIFICATIONS:
            raise QualificationError(
                "raw evidence collection classification is invalid"
            )
    if key in {
        "artifact_sha256",
        "primary_file_sha256",
        "record_file_sha256",
        "sha256",
        "sha256_after_install",
        "sha256_before_install",
    } and not re.fullmatch(r"[0-9a-f]{64}", value):
        raise QualificationError(f"invalid SHA-256 in raw-evidence field: {key}")
    if key == "generation_id" and not re.fullmatch(r"[0-9a-f]{32}", value):
        raise QualificationError("raw evidence generation id is invalid")
    if key == "image_id" and not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise QualificationError("raw evidence image id is invalid")
    if key == "container_id" and not re.fullmatch(r"[0-9a-f]{12,64}", value):
        raise QualificationError("raw evidence container id is invalid")
    if key == "kind" and value not in {
        "docker",
        "host",
        "source-distribution",
        "wheel",
    }:
        raise QualificationError("raw evidence kind is invalid")
    if key == "term" and value != "linux":
        raise QualificationError("raw evidence TERM identity is invalid")
    if key == "working_directory" and not (
        Path(value).is_absolute() or re.fullmatch(r"[A-Za-z]:[\\/].+", value)
    ):
        raise QualificationError("raw evidence working directory is not absolute")


def _expect_mapping(value: object, *, path: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QualificationError(f"raw evidence object expected at: {'/'.join(path)}")
    if not all(isinstance(key, str) for key in value):
        raise QualificationError(f"non-string raw-evidence key at: {'/'.join(path)}")
    return value


def _expect_keys(
    value: dict[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    path: tuple[str, ...],
) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise QualificationError(
            f"unknown raw-evidence field at {'/'.join(path) or '<root>'}: "
            f"{sorted(unknown)[0]}"
        )
    if missing:
        raise QualificationError(
            f"missing raw-evidence field at {'/'.join(path) or '<root>'}: "
            f"{sorted(missing)[0]}"
        )


def _expect_bool(value: object, *, path: tuple[str, ...]) -> None:
    if type(value) is not bool:
        raise QualificationError(f"raw evidence boolean expected at: {'/'.join(path)}")


def _expect_int(value: object, *, path: tuple[str, ...]) -> None:
    if type(value) is not int or value < 0:
        raise QualificationError(
            f"raw evidence non-negative integer expected at: {'/'.join(path)}"
        )


def _expect_number(value: object, *, path: tuple[str, ...]) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise QualificationError(
            f"raw evidence non-negative number expected at: {'/'.join(path)}"
        )


def _expect_string(value: object, *, path: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise QualificationError(f"raw evidence string expected at: {'/'.join(path)}")
    _validate_string(value, path=path)
    return value


def _validate_string_list(
    value: object,
    *,
    path: tuple[str, ...],
    environment_keys: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        raise QualificationError(f"raw evidence list expected at: {'/'.join(path)}")
    result: list[str] = []
    for item in value:
        text = _expect_string(item, path=path)
        if environment_keys and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_()]*", text):
            raise QualificationError("invalid environment key name in raw evidence")
        result.append(text)
    return result


def _validate_argv(value: object, *, path: tuple[str, ...]) -> None:
    argv = _validate_string_list(value, path=path)
    if not argv:
        raise QualificationError("raw evidence argv must not be empty")
    for index, item in enumerate(argv[:-1]):
        if SENSITIVE_ARG_RE.fullmatch(item):
            raise QualificationError(
                f"sensitive option value in raw-evidence field: {'/'.join(path)}"
            )


def _validate_command(value: object, *, path: tuple[str, ...]) -> None:
    command = _expect_mapping(value, path=path)
    _expect_keys(
        command,
        allowed=COMMAND_KEYS,
        required=COMMAND_KEYS,
        path=path,
    )
    _validate_argv(command["argv"], path=(*path, "argv"))
    _expect_string(command["working_directory"], path=(*path, "working_directory"))


def _validate_platform(value: object, *, path: tuple[str, ...]) -> None:
    platform_value = _expect_mapping(value, path=path)
    _expect_keys(
        platform_value,
        allowed=PLATFORM_KEYS,
        required=PLATFORM_KEYS,
        path=path,
    )
    for key, child in platform_value.items():
        _expect_string(child, path=(*path, key))


def _validate_measurements(value: object, *, path: tuple[str, ...]) -> None:
    measurements = _expect_mapping(value, path=path)
    _expect_keys(
        measurements,
        allowed=MEASUREMENT_KEYS,
        required=MEASUREMENT_KEYS,
        path=path,
    )
    for key, child in measurements.items():
        if child is not None:
            _expect_int(child, path=(*path, key))


def _validate_runtime(value: object, *, path: tuple[str, ...]) -> None:
    runtime = _expect_mapping(value, path=path)
    kind = _expect_string(runtime.get("kind"), path=(*path, "kind"))
    expected = RUNTIME_HOST_KEYS if kind == "host" else RUNTIME_DOCKER_KEYS
    _expect_keys(runtime, allowed=expected, required=expected, path=path)
    for key, child in runtime.items():
        _expect_string(child, path=(*path, key))


def _validate_artifact(value: object, *, path: tuple[str, ...]) -> None:
    artifact = _expect_mapping(value, path=path)
    _expect_keys(
        artifact,
        allowed=ARTIFACT_KEYS,
        required=ARTIFACT_REQUIRED_KEYS,
        path=path,
    )
    for key in (
        "filename",
        "kind",
        "sha256",
        "sha256_after_install",
        "sha256_before_install",
    ):
        _expect_string(artifact[key], path=(*path, key))
    _expect_int(artifact["size_bytes"], path=(*path, "size_bytes"))
    for key in ("license", "license_expression", "name", "version"):
        if key in artifact and artifact[key] is not None:
            _expect_string(artifact[key], path=(*path, key))
    for key in ("license_classifiers", "tags"):
        if key in artifact:
            _validate_string_list(artifact[key], path=(*path, key))
    if "license_files" in artifact:
        files = artifact["license_files"]
        if not isinstance(files, list):
            raise QualificationError("raw evidence license_files must be a list")
        for index, child in enumerate(files):
            file_value = _expect_mapping(
                child, path=(*path, "license_files", str(index))
            )
            _expect_keys(
                file_value,
                allowed=LICENSE_FILE_KEYS,
                required=LICENSE_FILE_KEYS,
                path=(*path, "license_files", str(index)),
            )
            for key, item in file_value.items():
                _expect_string(item, path=(*path, "license_files", str(index), key))


def _validate_resolved_distribution(value: object, *, path: tuple[str, ...]) -> None:
    distribution = _expect_mapping(value, path=path)
    _expect_keys(
        distribution,
        allowed=RESOLVED_DISTRIBUTION_KEYS,
        required=RESOLVED_DISTRIBUTION_KEYS,
        path=path,
    )
    for key, child in distribution.items():
        _expect_string(child, path=(*path, key))


def _validate_shell_result(value: object, *, path: tuple[str, ...]) -> None:
    result = _expect_mapping(value, path=path)
    required = SHELL_RESULT_KEYS - {"module_discovery", "output_overflowed"}
    _expect_keys(result, allowed=SHELL_RESULT_KEYS, required=required, path=path)
    for key, child in result.items():
        if key == "captured_byte_count":
            _expect_int(child, path=(*path, key))
        else:
            _expect_bool(child, path=(*path, key))


def _validate_row(value: object, *, probe: str, path: tuple[str, ...]) -> None:
    row = _expect_mapping(value, path=path)
    row_id = row.get("id")
    if not isinstance(row_id, str):
        raise QualificationError(
            f"raw evidence result-row id expected at: {'/'.join(path)}"
        )
    _validate_string(row_id, path=(*path, "id"))
    row_schema = ROW_SCHEMAS[probe].get(row_id)
    if row_schema is None:
        raise QualificationError(f"unsupported result row for {probe}: {row_id}")
    required = set(ROW_BASE_KEYS)
    status = row.get("status")
    if probe.startswith("environment-"):
        required.add("available")
        if row.get("available") is True:
            required |= {"initial_key_count", "sensitive_initial_key_count"}
        else:
            required.add("reason_category")
    elif probe == "pywinpty" and status == "UNSUPPORTED_FAIL_CLOSED":
        required.add("native_execution")
    else:
        required |= row_schema
    _expect_keys(
        row,
        allowed=ROW_BASE_KEYS | row_schema,
        required=required,
        path=path,
    )
    for key, child in row.items():
        child_path = (*path, key)
        if key in ROW_BOOL_KEYS:
            if child is not None:
                _expect_bool(child, path=child_path)
        elif key in ROW_INT_KEYS:
            if child is not None:
                _expect_int(child, path=child_path)
        elif key in {
            "id",
            "status",
            "artifact_filename",
            "artifact_sha256",
            "distribution_version",
            "primary_file_name",
            "primary_file_sha256",
            "record_file_name",
            "record_file_sha256",
            "reason_category",
        }:
            if child is not None:
                _expect_string(child, path=child_path)
        elif key == "observed_mutable_names":
            _validate_string_list(child, path=child_path)
        elif key in {
            "class_available_counts",
            "class_clean_exit",
            "class_interactive_markers",
            "class_pass",
            "classifications",
        }:
            nested = _expect_mapping(child, path=child_path)
            _expect_keys(
                nested,
                allowed={"editor", "monitor", "pager"}
                if key != "classifications"
                else {"buffer", "charset", "dirty", "mode", "savepoints", "tabstops"},
                required={"editor", "monitor", "pager"}
                if key != "classifications"
                else {"buffer", "charset", "dirty", "mode", "savepoints", "tabstops"},
                path=child_path,
            )
            for nested_key, nested_value in nested.items():
                if key == "class_available_counts":
                    _expect_int(nested_value, path=(*child_path, nested_key))
                elif key in {
                    "class_clean_exit",
                    "class_interactive_markers",
                    "class_pass",
                }:
                    _expect_bool(nested_value, path=(*child_path, nested_key))
                else:
                    _expect_string(nested_value, path=(*child_path, nested_key))


def _validate_row_semantics(payload: dict[str, Any], *, probe: str) -> None:
    rows = payload["rows"]
    expected_ids = set(ROW_SCHEMAS[probe])
    actual_ids = [row["id"] for row in rows]
    if len(actual_ids) != len(set(actual_ids)):
        raise QualificationError("raw evidence contains duplicate result rows")
    if set(actual_ids) != expected_ids or len(actual_ids) != len(expected_ids):
        raise QualificationError("raw evidence result-row set is incomplete")

    root_status = payload["status"]
    root_mandatory = payload["mandatory"]
    if any(row["mandatory"] is not root_mandatory for row in rows):
        raise QualificationError("raw evidence root/child mandatory status differs")
    child_statuses = [row["status"] for row in rows]
    if root_status == "PASS" and any(status != "PASS" for status in child_statuses):
        raise QualificationError("raw evidence PASS root has non-PASS child")
    if root_status == "FAIL" and (
        "FAIL" not in child_statuses
        or any(status not in {"PASS", "FAIL"} for status in child_statuses)
    ):
        raise QualificationError("raw evidence FAIL root is inconsistent with children")
    if root_status == "UNSUPPORTED_FAIL_CLOSED" and (
        probe != "pywinpty" or any(status != root_status for status in child_statuses)
    ):
        raise QualificationError("raw evidence unsupported root/child status differs")
    if root_status == "UNAVAILABLE" and (
        not probe.startswith("environment-")
        or any(status != root_status for status in child_statuses)
    ):
        raise QualificationError("raw evidence unavailable root/child status differs")

    if probe == "pywinpty":
        expected_native = root_status != "UNSUPPORTED_FAIL_CLOSED"
        if any(row["native_execution"] is not expected_native for row in rows):
            raise QualificationError("raw evidence native execution semantics differ")
    if probe == "artifacts":
        artifact_row = rows[0]
        if artifact_row["artifact_count"] != len(payload["artifacts"]):
            raise QualificationError("raw evidence artifact cardinality differs")
        if root_status == "PASS" and (
            not payload["artifacts"] or not payload["resolved_distributions"]
        ):
            raise QualificationError("passing artifact evidence is incomplete")


def validate_sibling_identity(
    payloads: Sequence[dict[str, Any]], *, require_generation: bool = True
) -> None:
    """Require sibling probes to share one row/platform/runtime/generation identity."""
    if not payloads:
        raise QualificationError("raw evidence sibling set is empty")
    for payload in payloads:
        validate_content_free(payload)
    first = payloads[0]
    probes: set[str] = set()
    for payload in payloads:
        probe = str(payload["probe"])
        if probe in probes:
            raise QualificationError("raw evidence sibling probe is duplicated")
        probes.add(probe)
        if payload["row_id"] != first["row_id"]:
            raise QualificationError("raw evidence sibling row identity differs")
        if payload["platform"] != first["platform"]:
            raise QualificationError("raw evidence sibling platform identity differs")
        if payload["runtime"] != first["runtime"]:
            raise QualificationError("raw evidence sibling runtime identity differs")
    if require_generation:
        generations = [payload.get("generation_id") for payload in payloads]
        if any(not isinstance(value, str) for value in generations):
            raise QualificationError("raw evidence sibling generation is missing")
        if len(set(generations)) != 1:
            raise QualificationError("raw evidence sibling generation differs")


def validate_content_free(value: object) -> None:
    """Validate the exact object shapes and content-free scalar types of one row."""
    payload = _expect_mapping(value, path=())
    probe = payload.get("probe")
    if not isinstance(probe, str):
        raise QualificationError("raw evidence probe id is invalid")
    if probe in ROW_SCHEMAS and probe.startswith("environment-"):
        probe_keys = ROOT_ENVIRONMENT_KEYS
    elif probe in ROOT_PROBE_KEYS and probe in ROW_SCHEMAS:
        probe_keys = ROOT_PROBE_KEYS[probe]
    else:
        raise QualificationError(f"unsupported raw-evidence probe: {probe}")
    allowed = ROOT_REQUIRED_KEYS | probe_keys | {"collection_command", "generation_id"}
    required = set(ROOT_REQUIRED_KEYS)
    if probe == "artifacts":
        required |= {"artifacts", "requirements", "resolved_distributions"}
    elif probe == "pyte":
        required.add("term")
    elif probe.startswith("environment-"):
        required.add("initial_keys")
    _expect_keys(payload, allowed=allowed, required=required, path=())

    _expect_int(payload["schema_version"], path=("schema_version",))
    for key in ("row_id", "probe", "status", "started_at_utc", "completed_at_utc"):
        _expect_string(payload[key], path=(key,))
    _expect_bool(payload["mandatory"], path=("mandatory",))
    _expect_number(payload["elapsed_seconds"], path=("elapsed_seconds",))
    _validate_command(payload["command"], path=("command",))
    if "collection_command" in payload:
        _validate_command(payload["collection_command"], path=("collection_command",))
    _validate_platform(payload["platform"], path=("platform",))
    _validate_measurements(payload["measurements"], path=("measurements",))
    _validate_runtime(payload["runtime"], path=("runtime",))
    if "generation_id" in payload:
        _expect_string(payload["generation_id"], path=("generation_id",))
    platform_os = payload["platform"]["os"]
    selected_shell_family = payload.get("selected_shell_family")
    if probe == "environment-default" and platform_os == "Windows":
        if selected_shell_family not in {"powershell", "cmd", "unavailable"}:
            raise QualificationError(
                "Windows environment-default selected shell is missing or invalid"
            )
    elif selected_shell_family is not None:
        raise QualificationError(
            "selected shell family is valid only for Windows environment-default"
        )

    rows = payload["rows"]
    if not isinstance(rows, list):
        raise QualificationError("raw evidence rows must be a list")
    for index, row in enumerate(rows):
        _validate_row(row, probe=probe, path=("rows", str(index)))
    _validate_row_semantics(payload, probe=probe)

    if "reason_category" in payload:
        _expect_string(payload["reason_category"], path=("reason_category",))
    if "failure_category" in payload:
        _expect_string(payload["failure_category"], path=("failure_category",))
    if "term" in payload:
        _expect_string(payload["term"], path=("term",))
    if "requirements" in payload:
        _validate_string_list(payload["requirements"], path=("requirements",))
    if "initial_keys" in payload:
        _validate_string_list(
            payload["initial_keys"], path=("initial_keys",), environment_keys=True
        )
    for key in (
        "initial_key_count",
        "sensitive_initial_key_count",
        "account_profile_candidate_count",
    ):
        if key in payload:
            _expect_int(payload[key], path=(key,))
    for key in ("actual_startup", "synthetic_profile"):
        if key in payload:
            _validate_shell_result(payload[key], path=(key,))
    if "artifacts" in payload:
        artifacts = payload["artifacts"]
        if not isinstance(artifacts, list):
            raise QualificationError("raw evidence artifacts must be a list")
        for index, artifact in enumerate(artifacts):
            _validate_artifact(artifact, path=("artifacts", str(index)))
    if "resolved_distributions" in payload:
        distributions = payload["resolved_distributions"]
        if not isinstance(distributions, list):
            raise QualificationError(
                "raw evidence resolved_distributions must be a list"
            )
        for index, distribution in enumerate(distributions):
            _validate_resolved_distribution(
                distribution, path=("resolved_distributions", str(index))
            )


def _run(
    argv: Sequence[str],
    *,
    cwd: Path,
    operation: str,
    timeout_seconds: float,
) -> BoundedResult:
    completed = run_bounded(
        argv,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        output_limit=1024 * 1024,
        operation=operation,
    )
    if completed.overflowed:
        raise QualificationError(f"{operation}_output_overflow")
    return completed


def _venv_python(row_dir: Path) -> Path:
    if os.name == "nt":
        return row_dir / "venv" / "Scripts" / "python.exe"
    return row_dir / "venv" / "bin" / "python"


def _selected_base_interpreter() -> Path:
    executable_name = (
        "python.exe"
        if os.name == "nt"
        else f"python{sys.version_info.major}.{sys.version_info.minor}"
    )
    binary_dir = "Scripts" if os.name == "nt" else "bin"
    candidate = Path(sys.base_prefix) / binary_dir / executable_name
    if candidate.is_file():
        return candidate
    fallback = Path(sys.executable)
    if fallback.is_file():
        return fallback
    raise QualificationError("selected interpreter executable is unavailable")


def create_isolated_venv(venv_dir: Path) -> None:
    """Create a venv from the selected runtime's real base-prefix executable."""
    if venv_dir.exists():
        raise QualificationError("qualification venv path already exists")
    try:
        completed = _run(
            (str(_selected_base_interpreter()), "-m", "venv", str(venv_dir)),
            cwd=venv_dir.parent,
            operation="venv-creation",
            timeout_seconds=120.0,
        )
    except OSError as exc:
        raise QualificationError("venv_creation_failed") from exc
    if completed.returncode != 0:
        print(
            "venv creation diagnostic: "
            f"returncode={completed.returncode} "
            f"timed_out={completed.timed_out} "
            f"terminated={completed.terminated} "
            f"killed={completed.killed} "
            f"overflowed={completed.overflowed}",
            file=sys.stderr,
        )
        if completed.stderr:
            print(completed.stderr.decode("utf-8", "replace"), file=sys.stderr)
        raise QualificationError("venv_creation_failed")


def _wheel_facts(path: Path) -> dict[str, object]:
    facts: dict[str, object] = {"kind": "wheel", "tags": [], "license_files": []}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        wheel_name = next(
            (name for name in names if name.endswith(".dist-info/WHEEL")), None
        )
        metadata_name = next(
            (name for name in names if name.endswith(".dist-info/METADATA")),
            None,
        )
        if wheel_name:
            wheel_text = archive.read(wheel_name).decode("utf-8", "replace")
            facts["tags"] = sorted(
                line.split(":", 1)[1].strip()
                for line in wheel_text.splitlines()
                if line.startswith("Tag:")
            )
        if metadata_name:
            message = BytesParser().parsebytes(archive.read(metadata_name))
            facts.update(
                {
                    "name": message.get("Name"),
                    "version": message.get("Version"),
                    "license": message.get("License"),
                    "license_expression": message.get("License-Expression"),
                    "license_classifiers": sorted(
                        classifier
                        for classifier in message.get_all("Classifier", [])
                        if classifier.startswith("License ::")
                    ),
                }
            )
        license_names = sorted(
            name
            for name in names
            if "/licenses/" in name.lower()
            or Path(name).name.lower().startswith(("license", "copying", "notice"))
        )
        facts["license_files"] = [
            {
                "name": Path(name).name,
                "sha256": hashlib.sha256(archive.read(name)).hexdigest(),
            }
            for name in license_names
            if not name.endswith("/")
        ]
    return facts


def _artifact_facts(path: Path) -> dict[str, object]:
    facts: dict[str, object] = {
        "filename": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if path.suffix == ".whl":
        facts.update(_wheel_facts(path))
    else:
        facts["kind"] = "source-distribution"
    return facts


def _site_packages(venv_python: Path, row_dir: Path) -> Path:
    completed = _run(
        (
            str(venv_python),
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ),
        cwd=row_dir,
        operation="site-packages-discovery",
        timeout_seconds=30.0,
    )
    if completed.returncode != 0:
        raise QualificationError("cannot locate prepared environment site-packages")
    return Path(completed.stdout.decode().strip())


def _prepared_platform_facts(venv_python: Path, row_dir: Path) -> dict[str, object]:
    source = (
        "import json,platform,sys; from pathlib import Path; "
        "print(json.dumps({"
        "'os':platform.system(),'os_release':platform.release(),"
        "'os_version':platform.version(),'architecture':platform.machine(),"
        "'python_implementation':platform.python_implementation(),"
        "'python_version':platform.python_version(),"
        "'python_executable_name':Path(sys.executable).name}))"
    )
    completed = _run(
        (str(venv_python), "-c", source),
        cwd=row_dir,
        operation="prepared-platform-discovery",
        timeout_seconds=30.0,
    )
    if completed.returncode != 0:
        raise QualificationError("cannot identify prepared environment platform")
    try:
        facts = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationError(
            "cannot identify prepared environment platform"
        ) from exc
    _validate_platform(facts, path=("platform",))
    return facts


def _resolved_distributions(site_packages: Path) -> list[dict[str, object]]:
    resolved: list[dict[str, object]] = []
    for distribution in metadata.distributions(path=[str(site_packages)]):
        name = distribution.metadata.get("Name")
        if not name:
            continue
        primary: Path | None = None
        primary_relative: str | None = None
        record: Path | None = None
        record_relative: str | None = None
        for relative in sorted(distribution.files or (), key=lambda item: str(item)):
            candidate = distribution.locate_file(relative)
            if candidate.is_file() and str(relative).replace("\\", "/").endswith(
                ".dist-info/RECORD"
            ):
                record = candidate
                record_relative = str(relative)
            if (
                candidate.is_file()
                and candidate.suffix.lower()
                in {
                    ".py",
                    ".pyd",
                    ".so",
                }
                and primary is None
            ):
                primary = candidate
                primary_relative = str(relative)
        if (
            primary is None
            or primary_relative is None
            or record is None
            or record_relative is None
        ):
            raise QualificationError(
                f"installed distribution inventory is incomplete: {name}"
            )
        resolved.append(
            {
                "name": name,
                "version": distribution.version,
                "primary_file": primary_relative,
                "primary_file_sha256": sha256_file(primary),
                "record_file": record_relative,
                "record_file_sha256": sha256_file(record),
            }
        )
    return sorted(resolved, key=lambda item: _normalize_distribution(str(item["name"])))


def _failure_manifest(
    *,
    row_id: str,
    requirements: Sequence[str],
    started: str,
    started_monotonic: float,
    category: str,
    runtime: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "row_id": row_id,
        "probe": "artifacts",
        "status": "FAIL",
        "mandatory": True,
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 6),
        "command": command_facts(),
        "platform": platform_facts(),
        "measurements": memory_facts(),
        "runtime": runtime,
        "requirements": list(requirements),
        "failure_category": category,
        "artifacts": [],
        "resolved_distributions": [],
        "rows": [
            {
                "id": "artifact-download-hash-offline-install",
                "mandatory": True,
                "status": "FAIL",
                "artifact_count": 0,
            }
        ],
    }


def prepare_row(
    *,
    row_id: str,
    row_dir: Path,
    requirements: Sequence[str],
    json_out: Path,
    replace: bool,
    runtime: dict[str, object] | None = None,
) -> bool:
    """Download, hash, offline-install, and inventory one isolated row."""
    started = utc_now()
    started_monotonic = time.monotonic()
    selected_runtime = runtime or runtime_facts("host")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", row_id):
        raise QualificationError(
            "row id must use lowercase ASCII letters, digits, and hyphens"
        )
    if json_out.exists() and not replace:
        raise QualificationError(f"refusing to replace existing output: {json_out}")
    row_dir = row_dir.resolve()
    json_out = json_out.resolve()
    if row_dir not in json_out.parents:
        raise QualificationError(
            "artifact manifest must be stored inside its unique row directory"
        )
    row_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = row_dir / "artifacts"
    venv_dir = row_dir / "venv"
    if artifact_dir.exists() or venv_dir.exists():
        raise QualificationError("row directory already contains qualification state")
    artifact_dir.mkdir()
    try:
        create_isolated_venv(venv_dir)
        venv_python = _venv_python(row_dir)
        downloaded = _run(
            (
                str(venv_python),
                "-m",
                "pip",
                "download",
                "--disable-pip-version-check",
                "--dest",
                str(artifact_dir),
                *requirements,
            ),
            cwd=row_dir,
            operation="artifact-download",
            timeout_seconds=300.0,
        )
        if downloaded.returncode != 0:
            raise QualificationError("artifact_download_failed")
        artifacts = sorted(path for path in artifact_dir.iterdir() if path.is_file())
        if not artifacts:
            raise QualificationError("artifact_download_empty")
        artifact_facts = [_artifact_facts(path) for path in artifacts]
        for facts in artifact_facts:
            facts["sha256_before_install"] = facts["sha256"]
        installed = _run(
            (
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                "--find-links",
                str(artifact_dir),
                *requirements,
            ),
            cwd=row_dir,
            operation="offline-install",
            timeout_seconds=300.0,
        )
        if installed.returncode != 0:
            raise QualificationError("offline_install_failed")
        for path, facts in zip(artifacts, artifact_facts, strict=True):
            after = sha256_file(path)
            facts["sha256_after_install"] = after
            if after != facts["sha256_before_install"]:
                raise QualificationError("artifact_changed_during_offline_install")
        site_packages = _site_packages(venv_python, row_dir)
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "row_id": row_id,
            "probe": "artifacts",
            "status": "PASS",
            "mandatory": True,
            "started_at_utc": started,
            "completed_at_utc": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started_monotonic, 6),
            "command": command_facts(),
            "platform": _prepared_platform_facts(venv_python, row_dir),
            "measurements": memory_facts(),
            "runtime": selected_runtime,
            "requirements": list(requirements),
            "artifacts": artifact_facts,
            "resolved_distributions": _resolved_distributions(site_packages),
            "rows": [
                {
                    "id": "artifact-download-hash-offline-install",
                    "mandatory": True,
                    "status": "PASS",
                    "artifact_count": len(artifacts),
                }
            ],
        }
        write_probe_result(json_out, payload, replace=replace)
        return True
    except QualificationError as exc:
        category = str(exc)
        payload = _failure_manifest(
            row_id=row_id,
            requirements=requirements,
            started=started,
            started_monotonic=started_monotonic,
            category=category,
            runtime=selected_runtime,
        )
        write_probe_result(json_out, payload, replace=replace)
        return False


def _raw_json_files(row_dir: Path) -> Iterable[Path]:
    for path in sorted(row_dir.glob("*.json")):
        if path.is_file():
            yield path


def _validate_raw_payload(payload: dict[str, Any]) -> tuple[str, str]:
    validate_content_free(payload)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise QualificationError("raw evidence has unsupported schema")
    row_id = payload.get("row_id")
    probe = payload.get("probe")
    status = payload.get("status")
    if not isinstance(row_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", row_id):
        raise QualificationError("raw evidence row id is invalid")
    if not isinstance(probe, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", probe):
        raise QualificationError("raw evidence probe id is invalid")
    if status not in ALLOWED_STATUSES:
        raise QualificationError("raw evidence status is invalid")
    command = payload.get("command")
    if not isinstance(command, dict) or not isinstance(command.get("argv"), list):
        raise QualificationError("raw evidence generation command is missing")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise QualificationError("raw evidence rows are missing")
    for row in rows:
        if not isinstance(row, dict) or row.get("status") not in ALLOWED_STATUSES:
            raise QualificationError("raw evidence contains an invalid result row")
    return row_id, probe


def _canonical_json_bytes(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _stage_bytes(destination: Path, content: bytes) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if temp_path.exists():
            temp_path.unlink()
        raise
    return temp_path


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_metadata(path: Path, payload: dict[str, object], *, replace: bool) -> None:
    if path.exists() and not replace:
        raise QualificationError(f"refusing to replace publication metadata: {path}")
    temporary = _stage_bytes(path, _canonical_json_bytes(payload))
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError(f"{label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise QualificationError(f"{label} root must be an object")
    return payload


def _expected_published_names(row_id: str, probes: Iterable[str]) -> set[str]:
    return {f"{row_id}-{probe}.json" for probe in probes}


def _validate_manifest_probe_names(names: Iterable[str], *, row_id: str) -> None:
    visible = set(names)
    expected_sets = (
        _expected_published_names(row_id, POSIX_COLLECTED_PROBES),
        _expected_published_names(row_id, WINDOWS_COLLECTED_PROBES),
    )
    if visible not in expected_sets:
        raise QualificationError("published generation file set is incomplete")


def _generation_manifest(
    *, row_id: str, generation_id: str, files: Sequence[Path]
) -> dict[str, object]:
    return {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "row_id": row_id,
        "generation_id": generation_id,
        "files": [
            {"name": path.name, "sha256": sha256_file(path)}
            for path in sorted(files, key=lambda item: item.name)
        ],
    }


def _validate_generation_manifest(
    marker: dict[str, Any], *, row_directory: Path
) -> tuple[str, str, dict[str, str]]:
    if set(marker) != {"schema_version", "row_id", "generation_id", "files"}:
        raise QualificationError("published generation marker shape is invalid")
    if marker["schema_version"] != PUBLICATION_SCHEMA_VERSION:
        raise QualificationError("published generation marker schema is invalid")
    row_id = marker["row_id"]
    generation_id = marker["generation_id"]
    if not isinstance(row_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", row_id):
        raise QualificationError("published generation row id is invalid")
    if not isinstance(generation_id, str) or not re.fullmatch(
        r"[0-9a-f]{32}", generation_id
    ):
        raise QualificationError("published generation id is invalid")
    entries = marker["files"]
    if not isinstance(entries, list):
        raise QualificationError("published generation file manifest is invalid")
    hashes: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"name", "sha256"}:
            raise QualificationError("published generation file entry is invalid")
        name = entry["name"]
        digest = entry["sha256"]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in hashes
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise QualificationError("published generation file identity is invalid")
        hashes[name] = digest
    _validate_manifest_probe_names(hashes, row_id=row_id)
    visible = {path.name for path in row_directory.glob("*.json") if path.is_file()}
    if visible != set(hashes):
        raise QualificationError("published generation visible file set differs")
    return row_id, generation_id, hashes


def _validate_published_generation(
    row_directory: Path,
) -> list[dict[str, Any]]:
    marker = _read_json_object(
        row_directory / CURRENT_GENERATION_MARKER,
        label="published generation marker",
    )
    row_id, generation_id, hashes = _validate_generation_manifest(
        marker, row_directory=row_directory
    )
    payloads: list[dict[str, Any]] = []
    for name, digest in sorted(hashes.items()):
        path = row_directory / name
        if sha256_file(path) != digest:
            raise QualificationError("published generation file hash differs")
        payload = _read_json_object(path, label="published generation evidence")
        _validate_raw_payload(payload)
        if payload.get("row_id") != row_id:
            raise QualificationError("published generation row identity differs")
        if payload.get("generation_id") != generation_id:
            raise QualificationError("published generation payload marker differs")
        payloads.append(payload)
    validate_sibling_identity(payloads)
    _validate_exact_probe_set(payloads)
    return payloads


def _load_legacy_generation(
    row_directory: Path,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    """Load a complete pre-marker generation without accepting its row schema."""
    files = sorted(path for path in row_directory.glob("*.json") if path.is_file())
    row_id = row_directory.name
    try:
        _validate_manifest_probe_names((path.name for path in files), row_id=row_id)
    except QualificationError as exc:
        raise QualificationError(
            "legacy published generation file set is incomplete"
        ) from exc
    payloads: list[dict[str, Any]] = []
    generations: set[str] = set()
    probes: set[str] = set()
    for path in files:
        payload = _read_json_object(path, label="legacy published evidence")
        probe = payload.get("probe")
        generation_id = payload.get("generation_id")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise QualificationError("legacy published schema is invalid")
        if payload.get("row_id") != row_id:
            raise QualificationError("legacy published row identity differs")
        if not isinstance(probe, str) or path.name != f"{row_id}-{probe}.json":
            raise QualificationError("legacy published probe identity differs")
        if not isinstance(generation_id, str) or not re.fullmatch(
            r"[0-9a-f]{32}", generation_id
        ):
            raise QualificationError("legacy published generation id is invalid")
        probes.add(probe)
        generations.add(generation_id)
        payloads.append(payload)
    try:
        _validate_exact_probe_set(payloads)
    except QualificationError:
        raise QualificationError("legacy published sibling set is incomplete")

    if len(generations) != 1:
        raise QualificationError("legacy published generation differs")
    first = payloads[0]
    if not isinstance(first.get("platform"), dict) or not isinstance(
        first.get("runtime"), dict
    ):
        raise QualificationError("legacy published sibling identity is invalid")
    for payload in payloads[1:]:
        if payload.get("platform") != first["platform"]:
            raise QualificationError("legacy published platform identity differs")
        if payload.get("runtime") != first["runtime"]:
            raise QualificationError("legacy published runtime identity differs")
    marker = _generation_manifest(
        row_id=row_id,
        generation_id=next(iter(generations)),
        files=files,
    )
    return payloads, marker


def _validate_legacy_recovery_generation(
    row_directory: Path,
) -> list[dict[str, Any]]:
    marker = _read_json_object(
        row_directory / CURRENT_GENERATION_MARKER,
        label="legacy recovery generation marker",
    )
    row_id, generation_id, hashes = _validate_generation_manifest(
        marker, row_directory=row_directory
    )
    payloads: list[dict[str, Any]] = []
    for name, digest in sorted(hashes.items()):
        path = row_directory / name
        if sha256_file(path) != digest:
            raise QualificationError("legacy recovery generation file hash differs")
        payload = _read_json_object(path, label="legacy recovery evidence")
        if payload.get("row_id") != row_id:
            raise QualificationError("legacy recovery row identity differs")
        if payload.get("generation_id") != generation_id:
            raise QualificationError("legacy recovery generation differs")
        if name != f"{row_id}-{payload.get('probe')}.json":
            raise QualificationError("legacy recovery probe identity differs")
        payloads.append(payload)
    try:
        _validate_exact_probe_set(payloads)
    except QualificationError:
        raise QualificationError("legacy recovery sibling set is incomplete")
    return payloads


def _remove_recovery_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if not child.is_file():
            raise QualificationError("publication recovery directory is invalid")
        child.unlink()
    path.rmdir()


def _cleanup_publication_transaction(
    row_directory: Path, recovery_directory: Path
) -> None:
    _remove_recovery_directory(recovery_directory)
    pending = row_directory / PENDING_PUBLICATION_MARKER
    if pending.exists():
        pending.unlink()
    _fsync_directory(row_directory)


def _load_pending_publication(row_directory: Path) -> dict[str, Any]:
    pending = _read_json_object(
        row_directory / PENDING_PUBLICATION_MARKER,
        label="pending publication marker",
    )
    required = {
        "schema_version",
        "transaction_id",
        "row_id",
        "new_generation_id",
        "previous_generation_id",
        "previous_generation_kind",
        "recovery_directory",
    }
    if set(pending) != required or pending["schema_version"] != (
        PUBLICATION_SCHEMA_VERSION
    ):
        raise QualificationError("pending publication marker shape is invalid")
    transaction_id = pending["transaction_id"]
    row_id = pending["row_id"]
    new_generation_id = pending["new_generation_id"]
    previous_generation_id = pending["previous_generation_id"]
    previous_generation_kind = pending["previous_generation_kind"]
    recovery_name = pending["recovery_directory"]
    if not isinstance(transaction_id, str) or not re.fullmatch(
        r"[0-9a-f]{32}", transaction_id
    ):
        raise QualificationError("pending publication transaction id is invalid")
    if not isinstance(row_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", row_id):
        raise QualificationError("pending publication row id is invalid")
    if not isinstance(new_generation_id, str) or not re.fullmatch(
        r"[0-9a-f]{32}", new_generation_id
    ):
        raise QualificationError("pending publication generation id is invalid")
    if previous_generation_id is not None and (
        not isinstance(previous_generation_id, str)
        or not re.fullmatch(r"[0-9a-f]{32}", previous_generation_id)
    ):
        raise QualificationError("pending previous generation id is invalid")
    if previous_generation_kind not in {"none", "current", "legacy"}:
        raise QualificationError("pending previous generation kind is invalid")
    if (previous_generation_id is None) != (previous_generation_kind == "none"):
        raise QualificationError("pending previous generation state is invalid")
    if recovery_name != f"{RECOVERY_DIRECTORY_PREFIX}{transaction_id}":
        raise QualificationError("pending recovery directory identity is invalid")
    return pending


def _recover_pending_publication(
    row_directory: Path, *, allow_legacy_result: bool = False
) -> list[dict[str, Any]]:
    pending = _load_pending_publication(row_directory)
    recovery_directory = row_directory / pending["recovery_directory"]
    current: list[dict[str, Any]] | None = None
    try:
        current = _validate_published_generation(row_directory)
    except QualificationError:
        pass
    if current:
        current_generation = str(current[0]["generation_id"])
        if current_generation in {
            pending["new_generation_id"],
            pending["previous_generation_id"],
        }:
            _cleanup_publication_transaction(row_directory, recovery_directory)
            return current

    previous_generation = pending["previous_generation_id"]
    if previous_generation is None:
        for name in _expected_published_names(
            str(pending["row_id"]), ALL_COLLECTED_PROBES
        ):
            destination = row_directory / name
            if destination.exists():
                destination.unlink()
        marker = row_directory / CURRENT_GENERATION_MARKER
        if marker.exists():
            marker.unlink()
        _cleanup_publication_transaction(row_directory, recovery_directory)
        raise QualificationError("interrupted initial publication has no current row")

    previous_kind = str(pending["previous_generation_kind"])
    recovered = (
        _validate_legacy_recovery_generation(recovery_directory)
        if previous_kind == "legacy"
        else _validate_published_generation(recovery_directory)
    )
    if recovered[0]["generation_id"] != previous_generation:
        raise QualificationError("publication recovery generation differs")
    staged: list[Path] = []
    try:
        for source in sorted(recovery_directory.glob("*.json")):
            destination = row_directory / source.name
            temporary = _stage_bytes(destination, source.read_bytes())
            staged.append(temporary)
            os.replace(temporary, destination)
        marker_destination = row_directory / CURRENT_GENERATION_MARKER
        if previous_kind == "legacy":
            if marker_destination.exists():
                marker_destination.unlink()
        else:
            marker_source = recovery_directory / CURRENT_GENERATION_MARKER
            marker_temporary = _stage_bytes(
                marker_destination, marker_source.read_bytes()
            )
            staged.append(marker_temporary)
            os.replace(marker_temporary, marker_destination)
        _fsync_directory(row_directory)
        if previous_kind == "legacy":
            restored, _ = _load_legacy_generation(row_directory)
        else:
            restored = _validate_published_generation(row_directory)
    finally:
        for temporary in staged:
            if temporary.exists():
                temporary.unlink()
    _cleanup_publication_transaction(row_directory, recovery_directory)
    if previous_kind == "legacy" and not allow_legacy_result:
        raise QualificationError(
            "recovered legacy generation is complete but not current"
        )
    return restored


def validate_published_row(
    row_directory: Path, *, recover: bool = False
) -> list[dict[str, Any]]:
    """Validate the manifest-authoritative current generation for one row."""
    pending = row_directory / PENDING_PUBLICATION_MARKER
    if pending.exists():
        if not recover:
            raise QualificationError(
                "published generation has an unfinished transaction"
            )
        return _recover_pending_publication(row_directory)
    return _validate_published_generation(row_directory)


def _prepare_publication_transaction(
    row_directory: Path,
    *,
    row_id: str,
    new_generation_id: str,
    previous_payloads: Sequence[dict[str, Any]],
    previous_generation_kind: str,
) -> Path:
    transaction_id = uuid.uuid4().hex
    recovery_directory = row_directory / (
        f"{RECOVERY_DIRECTORY_PREFIX}{transaction_id}"
    )
    recovery_directory.mkdir()
    previous_generation_id: str | None = None
    try:
        if previous_payloads:
            previous_generation_id = str(previous_payloads[0]["generation_id"])
            for source in sorted(row_directory.glob("*.json")):
                destination = recovery_directory / source.name
                shutil.copyfile(source, destination)
                with destination.open("rb") as stream:
                    os.fsync(stream.fileno())
            marker_destination = recovery_directory / CURRENT_GENERATION_MARKER
            if previous_generation_kind == "legacy":
                recovery_files = sorted(recovery_directory.glob("*.json"))
                recovery_marker = _generation_manifest(
                    row_id=row_id,
                    generation_id=previous_generation_id,
                    files=recovery_files,
                )
                _atomic_metadata(
                    marker_destination,
                    recovery_marker,
                    replace=False,
                )
            else:
                marker_source = row_directory / CURRENT_GENERATION_MARKER
                shutil.copyfile(marker_source, marker_destination)
                with marker_destination.open("rb") as stream:
                    os.fsync(stream.fileno())
            _fsync_directory(recovery_directory)
            if previous_generation_kind == "legacy":
                _validate_legacy_recovery_generation(recovery_directory)
            else:
                _validate_published_generation(recovery_directory)
        pending: dict[str, object] = {
            "schema_version": PUBLICATION_SCHEMA_VERSION,
            "transaction_id": transaction_id,
            "row_id": row_id,
            "new_generation_id": new_generation_id,
            "previous_generation_id": previous_generation_id,
            "previous_generation_kind": previous_generation_kind,
            "recovery_directory": recovery_directory.name,
        }
        _atomic_metadata(
            row_directory / PENDING_PUBLICATION_MARKER,
            pending,
            replace=False,
        )
    except BaseException:
        _remove_recovery_directory(recovery_directory)
        raise
    return recovery_directory


def collect_row(*, row_dir: Path, evidence_root: Path, replace: bool) -> int:
    """Publish one complete validated generation with recoverable replacement."""
    source_files = list(_raw_json_files(row_dir))
    if not source_files:
        raise QualificationError("row directory contains no JSON evidence")
    validated: list[tuple[dict[str, Any], Path]] = []
    generation_id = uuid.uuid4().hex
    seen: set[tuple[str, str]] = set()
    for source in source_files:
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise QualificationError(
                f"invalid JSON evidence file: {source.name}"
            ) from exc
        if not isinstance(payload, dict):
            raise QualificationError(
                f"raw evidence root must be an object: {source.name}"
            )
        row_id, probe = _validate_raw_payload(payload)
        identity = (row_id, probe)
        if identity in seen:
            raise QualificationError(
                f"duplicate raw evidence identity: {row_id}/{probe}"
            )
        seen.add(identity)
        destination = evidence_root / row_id / f"{row_id}-{probe}.json"
        if destination.exists() and not replace:
            raise QualificationError(
                f"refusing to replace collected evidence: {destination}"
            )
        if "collection_command" in payload:
            raise QualificationError("source evidence already has collection metadata")
        if "generation_id" in payload:
            raise QualificationError("source evidence already has a generation id")
        payload["generation_id"] = generation_id
        payload["collection_command"] = command_facts()
        validate_content_free(payload)
        validated.append((payload, destination))

    payloads = [payload for payload, _ in validated]
    validate_sibling_identity(payloads)
    _validate_exact_probe_set(payloads)

    row_ids = {str(payload["row_id"]) for payload in payloads}
    if len(row_ids) != 1:
        raise QualificationError("raw evidence sibling row identity differs")
    row_id = next(iter(row_ids))
    row_directory = evidence_root / row_id
    visible_destinations = sorted(row_directory.glob("*.json"))
    if visible_destinations and not replace:
        raise QualificationError(
            f"refusing to replace collected evidence: {visible_destinations[0]}"
        )
    if (row_directory / PENDING_PUBLICATION_MARKER).exists():
        validate_published_row(row_directory, recover=True)
    previous_payloads: list[dict[str, Any]] = []
    previous_generation_kind = "none"
    if visible_destinations:
        if (row_directory / CURRENT_GENERATION_MARKER).exists():
            previous_payloads = validate_published_row(row_directory, recover=False)
            previous_generation_kind = "current"
        else:
            previous_payloads, _ = _load_legacy_generation(row_directory)
            previous_generation_kind = "legacy"

    staged: list[tuple[Path, Path]] = []
    marker_temporary: Path | None = None
    recovery_directory: Path | None = None
    try:
        for payload, destination in validated:
            temp_path = _stage_bytes(destination, _canonical_json_bytes(payload))
            staged.append((temp_path, destination))
        marker = {
            "schema_version": PUBLICATION_SCHEMA_VERSION,
            "row_id": row_id,
            "generation_id": generation_id,
            "files": [
                {
                    "name": destination.name,
                    "sha256": hashlib.sha256(temp_path.read_bytes()).hexdigest(),
                }
                for temp_path, destination in sorted(
                    staged, key=lambda item: item[1].name
                )
            ],
        }
        marker_destination = row_directory / CURRENT_GENERATION_MARKER
        marker_temporary = _stage_bytes(
            marker_destination, _canonical_json_bytes(marker)
        )
        recovery_directory = _prepare_publication_transaction(
            row_directory,
            row_id=row_id,
            new_generation_id=generation_id,
            previous_payloads=previous_payloads,
            previous_generation_kind=previous_generation_kind,
        )
        for temp_path, destination in staged:
            os.replace(temp_path, destination)
        os.replace(marker_temporary, marker_destination)
        _fsync_directory(row_directory)
        _validate_published_generation(row_directory)
        _cleanup_publication_transaction(row_directory, recovery_directory)
    except Exception:
        if (row_directory / PENDING_PUBLICATION_MARKER).exists():
            _recover_pending_publication(row_directory, allow_legacy_result=True)
        raise
    finally:
        for temp_path, _ in staged:
            if temp_path.exists():
                temp_path.unlink()
        if marker_temporary is not None and marker_temporary.exists():
            marker_temporary.unlink()
        if (
            recovery_directory is not None
            and recovery_directory.exists()
            and not (row_directory / PENDING_PUBLICATION_MARKER).exists()
        ):
            _remove_recovery_directory(recovery_directory)
    return len(validated)


def artifact_manifest(path: Path, *, required_distribution: str) -> dict[str, Any]:
    """Load a prepared manifest and verify an exact installed distribution/artifact pair."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError("artifact manifest is unreadable") from exc
    if not isinstance(payload, dict):
        raise QualificationError("artifact manifest root must be an object")
    _validate_raw_payload(payload)
    if payload.get("status") != "PASS":
        raise QualificationError("artifact manifest did not pass preparation")
    target = _normalize_distribution(required_distribution)
    distributions = payload.get("resolved_distributions", [])
    matches = [
        item
        for item in distributions
        if isinstance(item, dict)
        and _normalize_distribution(str(item.get("name", ""))) == target
    ]
    if len(matches) != 1:
        raise QualificationError(
            f"prepared distribution is not unique: {required_distribution}"
        )
    artifacts = payload.get("artifacts", [])
    artifact_matches = [
        item
        for item in artifacts
        if isinstance(item, dict)
        and _normalize_distribution(str(item.get("name", ""))) == target
    ]
    if len(artifact_matches) != 1:
        raise QualificationError(
            f"prepared artifact is not unique: {required_distribution}"
        )
    distribution = matches[0]
    artifact = artifact_matches[0]
    filename = artifact["filename"]
    if (
        not isinstance(filename, str)
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
    ):
        raise QualificationError("prepared artifact filename is unsafe")
    artifact_dir = path.parent / "artifacts"
    artifact_path = artifact_dir / filename
    try:
        resolved_artifact = artifact_path.resolve(strict=True)
    except OSError as exc:
        raise QualificationError("prepared artifact is unavailable") from exc
    if resolved_artifact.parent != artifact_dir.resolve():
        raise QualificationError("prepared artifact path escaped its row")
    digest = sha256_file(resolved_artifact)
    recorded_hashes = {
        artifact["sha256"],
        artifact["sha256_before_install"],
        artifact["sha256_after_install"],
    }
    if recorded_hashes != {digest}:
        raise QualificationError("prepared artifact changed after preparation")
    if resolved_artifact.stat().st_size != artifact["size_bytes"]:
        raise QualificationError("prepared artifact size changed after preparation")
    if _normalize_distribution(str(artifact.get("name", ""))) != target:
        raise QualificationError("prepared artifact name is not bound")
    if artifact.get("version") != distribution.get("version"):
        raise QualificationError("prepared artifact/distribution version differs")

    venv_python = _venv_python(path.parent)
    if not venv_python.is_file():
        raise QualificationError("prepared environment is unavailable for rehash")
    current_distributions = _resolved_distributions(
        _site_packages(venv_python, path.parent)
    )
    current_matches = [
        item
        for item in current_distributions
        if _normalize_distribution(str(item.get("name", ""))) == target
    ]
    if len(current_matches) != 1 or current_matches[0] != distribution:
        raise QualificationError(
            "installed distribution facts changed after preparation"
        )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-row")
    prepare.add_argument("--row-id", required=True)
    prepare.add_argument("--row-dir", required=True, type=Path)
    prepare.add_argument("--requirement", action="append", required=True)
    prepare.add_argument("--json-out", required=True, type=Path)
    prepare.add_argument("--replace", action="store_true")
    prepare.add_argument("--runtime-kind", choices=("host", "docker"), default="host")
    prepare.add_argument("--runtime-image")
    prepare.add_argument("--runtime-image-id")
    prepare.add_argument("--runtime-container-id")

    collect = subparsers.add_parser("collect-row")
    collect.add_argument("--row-dir", required=True, type=Path)
    collect.add_argument("--evidence-root", required=True, type=Path)
    collect.add_argument("--replace", action="store_true")

    validate = subparsers.add_parser("validate-row")
    validate.add_argument("--row-dir", required=True, type=Path)
    validate.add_argument("--recover", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == "_bounded-bootstrap":
        if len(raw_argv) != 2:
            return 122
        try:
            return _run_bounded_bootstrap_payload(raw_argv[1])
        except QualificationError:
            return 122
    args = _parser().parse_args(raw_argv)
    try:
        if args.command == "prepare-row":
            passed = prepare_row(
                row_id=args.row_id,
                row_dir=args.row_dir,
                requirements=args.requirement,
                json_out=args.json_out,
                replace=args.replace,
                runtime=runtime_facts(
                    args.runtime_kind,
                    image=args.runtime_image,
                    image_id=args.runtime_image_id,
                    container_id=args.runtime_container_id,
                ),
            )
            return 0 if passed else 1
        if args.command == "collect-row":
            collect_row(
                row_dir=args.row_dir,
                evidence_root=args.evidence_root,
                replace=args.replace,
            )
        else:
            validate_published_row(args.row_dir, recover=args.recover)
        return 0
    except QualificationError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
