#!/usr/bin/env python3
"""Qualify low-level ConPTY behavior through an owned Windows Job boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from common import (
    SCHEMA_VERSION,
    QualificationError,
    artifact_manifest,
    command_facts,
    memory_facts,
    platform_facts,
    utc_now,
    validate_content_free,
    validate_sibling_identity,
    write_probe_result,
)


WINDOWS_ROWS = (
    "package-pywinpty-3.0.5",
    "windows-platform-floor",
    "windows-low-level-api",
    "windows-conpty-only",
    "windows-job-admission-membership",
    "windows-handle-inheritance",
    "windows-one-credit-bounded-read",
    "windows-concurrent-io-close",
    "windows-profile-module-discovery",
    "windows-unicode-alternate-screen",
    "windows-app-crash-descendant-cleanup",
    "windows-eof-output-integrity",
    "four-session-managed-rss",
)
CREDIT_LIMIT_BYTES = 64 * 1024
UPSTREAM_READ_BUFFER_BYTES = 32 * 1024
RSS_LIMIT_BYTES = 256 * 1024 * 1024
WINDOWS_BUILD_FLOOR = 17763
WORKER_TIMEOUT_SECONDS = 30.0
CLOSE_TIMEOUT_SECONDS = 5.0
SYNCHRONIZE_ACCESS = 0x00100000
WAIT_OBJECT_0 = 0
STABLE_JOB_SAMPLE_COUNT = 3
DESCENDANT_RE = re.compile(rb"TLDW22512-DESCENDANT=(\d+)")
MARKER = "TLDW22512-UNICODE-\u754c"
INTEGRITY_FRAME_COUNT = 4
INTEGRITY_PAYLOAD_BYTES = 40_000


def _manifest_context(path: Path) -> tuple[str, dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError("artifact manifest is unreadable") from exc
    row_id = payload.get("row_id") if isinstance(payload, dict) else None
    if not isinstance(row_id, str):
        raise QualificationError("artifact manifest row identity is invalid")
    runtime = payload.get("runtime", {"kind": "host"})
    if not isinstance(runtime, dict):
        raise QualificationError("artifact manifest runtime identity is invalid")
    return row_id, runtime


def _unsupported_payload(
    row_id: str,
    runtime: dict[str, object],
    started_at: str,
    elapsed: float,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "row_id": row_id,
        "probe": "pywinpty",
        "status": "UNSUPPORTED_FAIL_CLOSED",
        "mandatory": True,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "elapsed_seconds": round(elapsed, 6),
        "command": command_facts(),
        "platform": platform_facts(),
        "measurements": memory_facts(),
        "runtime": runtime,
        "reason_category": "native-windows-host-required",
        "rows": [
            {
                "id": row_id_value,
                "mandatory": True,
                "status": "UNSUPPORTED_FAIL_CLOSED",
                "native_execution": False,
            }
            for row_id_value in WINDOWS_ROWS
        ],
    }


def _default_observations(
    *, distribution_version: object = None, windows_build: object = None
) -> dict[str, object]:
    return {
        "artifact_filename": None,
        "artifact_sha256": None,
        "artifact_size_bytes": 0,
        "artifact_verified_during_probe": False,
        "distribution_version": distribution_version,
        "primary_file_name": None,
        "primary_file_sha256": None,
        "record_file_name": None,
        "record_file_sha256": None,
        "windows_build": windows_build,
        "fresh_worker": False,
        "worker_std_streams_fd_backed": False,
        "low_level_api": False,
        "conpty_constructed": False,
        "job_admitted_before_conpty": False,
        "job_membership_complete": False,
        "job_member_count": 0,
        "job_handle_non_inheritable": False,
        "one_credit_max_bytes": 0,
        "measured_chunk_count": 0,
        "read_api_accepts_size": False,
        "upstream_read_buffer_bytes": UPSTREAM_READ_BUFFER_BYTES,
        "max_unacknowledged_credits": 0,
        "max_concurrent_readers": 0,
        "concurrent_operation_count": 0,
        "inflight_operation_category_count": 0,
        "io_inflight_at_handoff": False,
        "priority_close_preempted_inflight": False,
        "quiet_terminal_startup_drained": False,
        "quiet_terminal_quiescent_before_handoff": False,
        "read_entered": False,
        "write_entered": False,
        "resize_entered": False,
        "cancel_entered": False,
        "read_completed_at_handoff": False,
        "write_completed_at_handoff": False,
        "resize_completed_at_handoff": False,
        "cancel_completed_at_handoff": False,
        "read_completed_post_close": False,
        "write_completed_post_close": False,
        "resize_completed_post_close": False,
        "cancel_completed_post_close": False,
        "write_completed": False,
        "resize_completed": False,
        "cancel_completed": False,
        "priority_close_completed": False,
        "read_returned_after_close": False,
        "write_returned_after_close": False,
        "resize_returned_after_close": False,
        "cancel_returned_after_close": False,
        "normal_cleanup_expected_process_count": 0,
        "normal_cleanup_retained_handle_count": 0,
        "normal_cleanup_wait_object_0_count": 0,
        "normal_cleanup_all_wait_object_0": False,
        "profile_module_discovery": False,
        "default_module_discovery": False,
        "profile_extended_module_discovery": False,
        "unicode_roundtrip": False,
        "alternate_screen": False,
        "alternate_isolated": False,
        "primary_restored": False,
        "app_crash_observed": False,
        "crash_app_process_separate": False,
        "crash_app_sole_job_handle_owner": False,
        "crash_job_handle_non_inheritable": False,
        "crash_worker_admitted_before_conpty": False,
        "crash_descendant_set_stable": False,
        "crash_descendants_ready_before_abort": False,
        "crash_supervisor_job_handle_count": 0,
        "crash_known_descendant_count": 0,
        "crash_supervisor_synchronize_handle_count": 0,
        "crash_wait_object_0_count": 0,
        "crash_all_descendants_wait_object_0": False,
        "terminal_child_crash_observed": False,
        "terminal_child_eof_observed": False,
        "terminal_child_member_before_crash": False,
        "terminal_grandchild_member_before_crash": False,
        "eof_observed": False,
        "output_integrity": False,
        "captured_byte_count": 0,
        "sequence_complete": False,
        "digest_equal": False,
        "post_exit_drain_bounded": False,
        "missing_eof_bounded": False,
        "four_session_count": 0,
        "rss_measurement_complete": False,
        "four_session_rss_delta_bytes": RSS_LIMIT_BYTES + 1,
        "rss_controller_process_count": 0,
        "rss_worker_process_count": 0,
        "rss_helper_process_count": 0,
        "rss_fixture_process_count": 0,
        "rss_fixture_processes_excluded": False,
        "rss_ipc_included_in_worker": False,
        "rss_sample_live_session_count": 0,
        "rss_crash_session_present": False,
    }


def _build_native_rows(
    observations: dict[str, object],
) -> list[dict[str, object]]:
    """Map independently measured native observations to mandatory rows."""

    def row(row_id: str, passed: bool, **facts: object) -> dict[str, object]:
        return {
            "id": row_id,
            "mandatory": True,
            "native_execution": True,
            "status": "PASS" if passed else "FAIL",
            **facts,
        }

    version = observations.get("distribution_version")
    build = observations.get("windows_build")
    credit_bytes = observations.get("one_credit_max_bytes")
    measured_chunk_count = observations.get("measured_chunk_count")
    unacknowledged = observations.get("max_unacknowledged_credits")
    readers = observations.get("max_concurrent_readers")
    member_count = observations.get("job_member_count")
    session_count = observations.get("four_session_count")
    rss_delta = observations.get("four_session_rss_delta_bytes")
    crash_known_count = observations.get("crash_known_descendant_count")
    crash_sync_count = observations.get("crash_supervisor_synchronize_handle_count")
    crash_wait_count = observations.get("crash_wait_object_0_count")
    cleanup_expected_count = observations.get("normal_cleanup_expected_process_count")
    cleanup_retained_count = observations.get("normal_cleanup_retained_handle_count")
    cleanup_wait_count = observations.get("normal_cleanup_wait_object_0_count")
    normal_cleanup_passed = bool(
        observations.get("normal_cleanup_all_wait_object_0") is True
        and isinstance(cleanup_expected_count, int)
        and cleanup_expected_count > 0
        and cleanup_retained_count == cleanup_expected_count
        and cleanup_wait_count == cleanup_expected_count
    )
    low_level_passed = (
        observations.get("fresh_worker") is True
        and observations.get("worker_std_streams_fd_backed") is True
        and observations.get("low_level_api") is True
    )
    job_passed = (
        observations.get("job_admitted_before_conpty") is True
        and observations.get("job_membership_complete") is True
        and observations.get("terminal_child_member_before_crash") is True
        and observations.get("terminal_grandchild_member_before_crash") is True
        and isinstance(member_count, int)
        and member_count >= 6
        and normal_cleanup_passed
    )
    credit_passed = (
        isinstance(credit_bytes, int)
        and 0 < credit_bytes <= CREDIT_LIMIT_BYTES
        and isinstance(measured_chunk_count, int)
        and measured_chunk_count > 0
        and observations.get("read_api_accepts_size") is False
        and observations.get("upstream_read_buffer_bytes") == UPSTREAM_READ_BUFFER_BYTES
        and unacknowledged == 1
        and readers == 1
    )
    io_passed = (
        all(
            observations.get(name) is True
            for name in (
                "quiet_terminal_startup_drained",
                "quiet_terminal_quiescent_before_handoff",
                "read_entered",
                "write_entered",
                "resize_entered",
                "cancel_entered",
                "write_completed_at_handoff",
                "resize_completed_at_handoff",
                "cancel_completed_at_handoff",
                "read_completed_post_close",
                "write_completed",
                "resize_completed",
                "cancel_completed",
                "priority_close_completed",
            )
        )
        and all(
            observations.get(name) is False
            for name in (
                "read_completed_at_handoff",
                "write_completed_post_close",
                "resize_completed_post_close",
                "cancel_completed_post_close",
            )
        )
        and observations.get("concurrent_operation_count") == 4
        and observations.get("inflight_operation_category_count") == 1
        and observations.get("io_inflight_at_handoff") is True
        and observations.get("priority_close_preempted_inflight") is True
        and readers == 1
    )
    alternate_passed = all(
        observations.get(name) is True
        for name in (
            "unicode_roundtrip",
            "alternate_screen",
            "alternate_isolated",
            "primary_restored",
        )
    )
    crash_passed = all(
        observations.get(name) is True
        for name in (
            "app_crash_observed",
            "crash_app_process_separate",
            "crash_app_sole_job_handle_owner",
            "crash_job_handle_non_inheritable",
            "crash_worker_admitted_before_conpty",
            "crash_descendant_set_stable",
            "crash_descendants_ready_before_abort",
            "crash_all_descendants_wait_object_0",
        )
    ) and (
        observations.get("crash_supervisor_job_handle_count") == 0
        and isinstance(crash_known_count, int)
        and crash_known_count >= 3
        and crash_sync_count == crash_known_count
        and crash_wait_count == crash_known_count
    )
    eof_passed = (
        observations.get("terminal_child_crash_observed") is True
        and observations.get("terminal_child_eof_observed") is True
        and observations.get("output_integrity") is True
        and observations.get("sequence_complete") is True
        and observations.get("digest_equal") is True
        and observations.get("post_exit_drain_bounded") is True
        and (
            observations.get("eof_observed") is True
            or observations.get("missing_eof_bounded") is True
        )
    )
    rss_passed = (
        session_count == 4
        and observations.get("rss_measurement_complete") is True
        and observations.get("rss_controller_process_count") == 1
        and observations.get("rss_worker_process_count") == 1
        and isinstance(observations.get("rss_helper_process_count"), int)
        and observations.get("rss_fixture_process_count") == 4
        and observations.get("rss_fixture_processes_excluded") is True
        and observations.get("rss_ipc_included_in_worker") is True
        and observations.get("rss_sample_live_session_count") == 4
        and observations.get("rss_crash_session_present") is False
        and isinstance(rss_delta, int)
        and 0 <= rss_delta <= RSS_LIMIT_BYTES
    )
    return [
        row(
            "package-pywinpty-3.0.5",
            version == "3.0.5"
            and observations.get("artifact_verified_during_probe") is True
            and all(
                isinstance(observations.get(name), str)
                for name in (
                    "artifact_filename",
                    "artifact_sha256",
                    "primary_file_name",
                    "primary_file_sha256",
                    "record_file_name",
                    "record_file_sha256",
                )
            ),
            artifact_filename=observations.get("artifact_filename"),
            artifact_sha256=observations.get("artifact_sha256"),
            artifact_size_bytes=observations.get("artifact_size_bytes"),
            artifact_verified_during_probe=observations.get(
                "artifact_verified_during_probe"
            ),
            distribution_version=version,
            primary_file_name=observations.get("primary_file_name"),
            primary_file_sha256=observations.get("primary_file_sha256"),
            record_file_name=observations.get("record_file_name"),
            record_file_sha256=observations.get("record_file_sha256"),
        ),
        row(
            "windows-platform-floor",
            isinstance(build, int) and build >= WINDOWS_BUILD_FLOOR,
            windows_build=build,
            minimum_windows_build=WINDOWS_BUILD_FLOOR,
        ),
        row(
            "windows-low-level-api",
            low_level_passed,
            fresh_worker=observations.get("fresh_worker"),
            worker_std_streams_fd_backed=observations.get(
                "worker_std_streams_fd_backed"
            ),
            low_level_api=observations.get("low_level_api"),
        ),
        row(
            "windows-conpty-only",
            observations.get("conpty_constructed") is True,
            constructed=observations.get("conpty_constructed"),
        ),
        row(
            "windows-job-admission-membership",
            job_passed,
            job_admitted_before_conpty=observations.get("job_admitted_before_conpty"),
            job_membership_complete=observations.get("job_membership_complete"),
            job_member_count=member_count,
            terminal_child_member_before_crash=observations.get(
                "terminal_child_member_before_crash"
            ),
            terminal_grandchild_member_before_crash=observations.get(
                "terminal_grandchild_member_before_crash"
            ),
            normal_cleanup_expected_process_count=cleanup_expected_count,
            normal_cleanup_retained_handle_count=cleanup_retained_count,
            normal_cleanup_wait_object_0_count=cleanup_wait_count,
            normal_cleanup_all_wait_object_0=observations.get(
                "normal_cleanup_all_wait_object_0"
            ),
        ),
        row(
            "windows-handle-inheritance",
            observations.get("job_handle_non_inheritable") is True,
            job_handle_non_inheritable=observations.get("job_handle_non_inheritable"),
        ),
        row(
            "windows-one-credit-bounded-read",
            credit_passed,
            one_credit_max_bytes=credit_bytes,
            credit_limit_bytes=CREDIT_LIMIT_BYTES,
            measured_chunk_count=measured_chunk_count,
            read_api_accepts_size=observations.get("read_api_accepts_size"),
            upstream_read_buffer_bytes=observations.get("upstream_read_buffer_bytes"),
            max_unacknowledged_credits=unacknowledged,
            max_concurrent_readers=readers,
        ),
        row(
            "windows-concurrent-io-close",
            io_passed,
            max_concurrent_readers=readers,
            concurrent_operation_count=observations.get("concurrent_operation_count"),
            inflight_operation_category_count=observations.get(
                "inflight_operation_category_count"
            ),
            io_inflight_at_handoff=observations.get("io_inflight_at_handoff"),
            priority_close_preempted_inflight=observations.get(
                "priority_close_preempted_inflight"
            ),
            quiet_terminal_startup_drained=observations.get(
                "quiet_terminal_startup_drained"
            ),
            quiet_terminal_quiescent_before_handoff=observations.get(
                "quiet_terminal_quiescent_before_handoff"
            ),
            read_entered=observations.get("read_entered"),
            write_entered=observations.get("write_entered"),
            resize_entered=observations.get("resize_entered"),
            cancel_entered=observations.get("cancel_entered"),
            read_completed_at_handoff=observations.get("read_completed_at_handoff"),
            write_completed_at_handoff=observations.get("write_completed_at_handoff"),
            resize_completed_at_handoff=observations.get("resize_completed_at_handoff"),
            cancel_completed_at_handoff=observations.get("cancel_completed_at_handoff"),
            read_completed_post_close=observations.get("read_completed_post_close"),
            write_completed_post_close=observations.get("write_completed_post_close"),
            resize_completed_post_close=observations.get("resize_completed_post_close"),
            cancel_completed_post_close=observations.get("cancel_completed_post_close"),
            write_completed=observations.get("write_completed"),
            resize_completed=observations.get("resize_completed"),
            cancel_completed=observations.get("cancel_completed"),
            priority_close_completed=observations.get("priority_close_completed"),
            read_returned_after_close=observations.get("read_returned_after_close"),
            write_returned_after_close=observations.get("write_returned_after_close"),
            resize_returned_after_close=observations.get("resize_returned_after_close"),
            cancel_returned_after_close=observations.get("cancel_returned_after_close"),
        ),
        row(
            "windows-profile-module-discovery",
            observations.get("profile_module_discovery") is True
            and observations.get("default_module_discovery") is True
            and observations.get("profile_extended_module_discovery") is True,
            profile_module_discovery=observations.get("profile_module_discovery"),
            default_module_discovery=observations.get("default_module_discovery"),
            profile_extended_module_discovery=observations.get(
                "profile_extended_module_discovery"
            ),
        ),
        row(
            "windows-unicode-alternate-screen",
            alternate_passed,
            unicode_roundtrip=observations.get("unicode_roundtrip"),
            alternate_screen=observations.get("alternate_screen"),
            alternate_isolated=observations.get("alternate_isolated"),
            primary_restored=observations.get("primary_restored"),
        ),
        row(
            "windows-app-crash-descendant-cleanup",
            crash_passed,
            app_crash_observed=observations.get("app_crash_observed"),
            crash_app_process_separate=observations.get("crash_app_process_separate"),
            crash_app_sole_job_handle_owner=observations.get(
                "crash_app_sole_job_handle_owner"
            ),
            crash_job_handle_non_inheritable=observations.get(
                "crash_job_handle_non_inheritable"
            ),
            crash_worker_admitted_before_conpty=observations.get(
                "crash_worker_admitted_before_conpty"
            ),
            crash_descendant_set_stable=observations.get("crash_descendant_set_stable"),
            crash_descendants_ready_before_abort=observations.get(
                "crash_descendants_ready_before_abort"
            ),
            crash_supervisor_job_handle_count=observations.get(
                "crash_supervisor_job_handle_count"
            ),
            crash_known_descendant_count=crash_known_count,
            crash_supervisor_synchronize_handle_count=crash_sync_count,
            crash_wait_object_0_count=crash_wait_count,
            crash_all_descendants_wait_object_0=observations.get(
                "crash_all_descendants_wait_object_0"
            ),
        ),
        row(
            "windows-eof-output-integrity",
            eof_passed,
            terminal_child_crash_observed=observations.get(
                "terminal_child_crash_observed"
            ),
            terminal_child_eof_observed=observations.get("terminal_child_eof_observed"),
            eof_observed=observations.get("eof_observed"),
            output_integrity=observations.get("output_integrity"),
            captured_byte_count=observations.get("captured_byte_count"),
            sequence_complete=observations.get("sequence_complete"),
            digest_equal=observations.get("digest_equal"),
            post_exit_drain_bounded=observations.get("post_exit_drain_bounded"),
            missing_eof_bounded=observations.get("missing_eof_bounded"),
        ),
        row(
            "four-session-managed-rss",
            rss_passed,
            four_session_count=session_count,
            rss_measurement_complete=observations.get("rss_measurement_complete"),
            four_session_rss_delta_bytes=rss_delta,
            four_session_rss_limit_bytes=RSS_LIMIT_BYTES,
            rss_controller_process_count=observations.get(
                "rss_controller_process_count"
            ),
            rss_worker_process_count=observations.get("rss_worker_process_count"),
            rss_helper_process_count=observations.get("rss_helper_process_count"),
            rss_fixture_process_count=observations.get("rss_fixture_process_count"),
            rss_fixture_processes_excluded=observations.get(
                "rss_fixture_processes_excluded"
            ),
            rss_ipc_included_in_worker=observations.get("rss_ipc_included_in_worker"),
            rss_sample_live_session_count=observations.get(
                "rss_sample_live_session_count"
            ),
            rss_crash_session_present=observations.get("rss_crash_session_present"),
        ),
    ]


def _windows_api() -> tuple[Any, Any, Any]:
    import ctypes
    from ctypes import wintypes

    if os.name != "nt":
        raise QualificationError("native Windows API requested on another host")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.IsProcessInJob.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.BOOL),
    ]
    kernel32.IsProcessInJob.restype = wintypes.BOOL
    kernel32.OpenProcess.argtypes = [
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.SetEvent.argtypes = [wintypes.HANDLE]
    kernel32.SetEvent.restype = wintypes.BOOL
    kernel32.CreateEventW.argtypes = [
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    kernel32.CreateEventW.restype = wintypes.HANDLE
    kernel32.OpenEventW.argtypes = [
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    kernel32.OpenEventW.restype = wintypes.HANDLE
    kernel32.SetHandleInformation.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    kernel32.SetHandleInformation.restype = wintypes.BOOL
    return ctypes, wintypes, kernel32


def _raise_windows_error(name: str) -> None:
    import ctypes

    raise QualificationError(f"{name}_failed_{ctypes.get_last_error()}")


class _WindowsJob:
    """Own a kill-on-close Job whose handle is never inherited."""

    def __init__(self) -> None:
        ctypes, wintypes, kernel32 = _windows_api()

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
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

        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetHandleInformation.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        kernel32.SetHandleInformation.restype = wintypes.BOOL
        kernel32.GetHandleInformation.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetHandleInformation.restype = wintypes.BOOL
        self._ctypes = ctypes
        self._wintypes = wintypes
        self._kernel32 = kernel32
        self.handle = kernel32.CreateJobObjectW(None, None)
        if not self.handle:
            _raise_windows_error("CreateJobObjectW")
        limits = ExtendedLimit()
        limits.BasicLimitInformation.LimitFlags = 0x00002000 | 0x00000400
        if not kernel32.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            self.close()
            _raise_windows_error("SetInformationJobObject")
        if not kernel32.SetHandleInformation(self.handle, 1, 0):
            self.close()
            _raise_windows_error("SetHandleInformation")

    def non_inheritable(self) -> bool:
        flags = self._wintypes.DWORD()
        if not self._kernel32.GetHandleInformation(
            self.handle, self._ctypes.byref(flags)
        ):
            _raise_windows_error("GetHandleInformation")
        return not bool(flags.value & 1)

    def _open_process(self, process_id: int, access: int) -> Any:
        self._kernel32.OpenProcess.argtypes = [
            self._wintypes.DWORD,
            self._wintypes.BOOL,
            self._wintypes.DWORD,
        ]
        self._kernel32.OpenProcess.restype = self._wintypes.HANDLE
        handle = self._kernel32.OpenProcess(access, False, process_id)
        if not handle:
            _raise_windows_error("OpenProcess")
        return handle

    def assign(self, process_id: int) -> None:
        self._kernel32.AssignProcessToJobObject.argtypes = [
            self._wintypes.HANDLE,
            self._wintypes.HANDLE,
        ]
        self._kernel32.AssignProcessToJobObject.restype = self._wintypes.BOOL
        handle = self._open_process(process_id, 0x0100 | 0x0001 | 0x1000)
        try:
            if not self._kernel32.AssignProcessToJobObject(self.handle, handle):
                _raise_windows_error("AssignProcessToJobObject")
        finally:
            self._kernel32.CloseHandle(handle)

    def contains(self, process_id: int) -> bool:
        self._kernel32.IsProcessInJob.argtypes = [
            self._wintypes.HANDLE,
            self._wintypes.HANDLE,
            self._ctypes.POINTER(self._wintypes.BOOL),
        ]
        self._kernel32.IsProcessInJob.restype = self._wintypes.BOOL
        try:
            handle = self._open_process(process_id, 0x1000)
        except QualificationError:
            return False
        result = self._wintypes.BOOL()
        try:
            return bool(
                self._kernel32.IsProcessInJob(
                    handle, self.handle, self._ctypes.byref(result)
                )
            ) and bool(result.value)
        finally:
            self._kernel32.CloseHandle(handle)

    def process_ids(self) -> list[int]:
        self._kernel32.QueryInformationJobObject.argtypes = [
            self._wintypes.HANDLE,
            self._ctypes.c_int,
            self._ctypes.c_void_p,
            self._wintypes.DWORD,
            self._ctypes.POINTER(self._wintypes.DWORD),
        ]
        self._kernel32.QueryInformationJobObject.restype = self._wintypes.BOOL
        for capacity in (16, 64, 256):

            class ProcessIdList(self._ctypes.Structure):
                _fields_ = [
                    ("NumberOfAssignedProcesses", self._wintypes.DWORD),
                    ("NumberOfProcessIdsInList", self._wintypes.DWORD),
                    ("ProcessIdList", self._ctypes.c_size_t * capacity),
                ]

            value = ProcessIdList()
            returned = self._wintypes.DWORD()
            if self._kernel32.QueryInformationJobObject(
                self.handle,
                3,
                self._ctypes.byref(value),
                self._ctypes.sizeof(value),
                self._ctypes.byref(returned),
            ):
                count = min(value.NumberOfProcessIdsInList, capacity)
                return [int(value.ProcessIdList[index]) for index in range(count)]
        _raise_windows_error("QueryInformationJobObject")

    def retain_process_handles(self, process_ids: Sequence[int]) -> list[Any]:
        """Retain SYNCHRONIZE authority before the Job can terminate its members."""
        handles: list[Any] = []
        try:
            for process_id in process_ids:
                handles.append(self._open_process(process_id, 0x00100000))
        except QualificationError:
            for handle in handles:
                self._kernel32.CloseHandle(handle)
            raise
        return handles

    def close(self) -> None:
        if getattr(self, "handle", None):
            self._kernel32.CloseHandle(self.handle)
            self.handle = None


def _create_event(name: str) -> Any:
    ctypes, wintypes, kernel32 = _windows_api()
    kernel32.CreateEventW.argtypes = [
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    kernel32.CreateEventW.restype = wintypes.HANDLE
    handle = kernel32.CreateEventW(None, True, False, name)
    if not handle:
        _raise_windows_error("CreateEventW")
    if not kernel32.SetHandleInformation(handle, 1, 0):
        kernel32.CloseHandle(handle)
        _raise_windows_error("SetHandleInformation")
    return handle


def _set_event(handle: Any) -> None:
    _, _, kernel32 = _windows_api()
    if not kernel32.SetEvent(handle):
        _raise_windows_error("SetEvent")


def _wait_event(handle: Any, timeout_seconds: float) -> bool:
    _, _, kernel32 = _windows_api()
    return kernel32.WaitForSingleObject(handle, int(timeout_seconds * 1000)) == 0


def _wait_named_event(name: str, timeout_seconds: float) -> bool:
    _, _, kernel32 = _windows_api()
    handle = kernel32.OpenEventW(0x00100000 | 0x0002, False, name)
    if not handle:
        _raise_windows_error("OpenEventW")
    try:
        return _wait_event(handle, timeout_seconds)
    finally:
        kernel32.CloseHandle(handle)


def _signal_named_event(name: str) -> None:
    _, _, kernel32 = _windows_api()
    handle = kernel32.OpenEventW(0x0002, False, name)
    if not handle:
        _raise_windows_error("OpenEventW")
    try:
        _set_event(handle)
    finally:
        kernel32.CloseHandle(handle)


def _close_handle(handle: Any) -> None:
    if handle:
        _, _, kernel32 = _windows_api()
        kernel32.CloseHandle(handle)


def _process_is_in_job(process_id: int | None = None) -> bool:
    """Query current Job membership without requiring the parent Job handle."""
    ctypes, wintypes, kernel32 = _windows_api()
    process = kernel32.GetCurrentProcess()
    opened = None
    if process_id is not None:
        opened = kernel32.OpenProcess(0x1000, False, process_id)
        if not opened:
            return False
        process = opened
    result = wintypes.BOOL()
    try:
        return bool(
            kernel32.IsProcessInJob(process, None, ctypes.byref(result))
        ) and bool(result.value)
    finally:
        if opened:
            kernel32.CloseHandle(opened)


def _open_synchronize_process_handles(
    process_ids: Sequence[int],
) -> tuple[Any, list[Any]]:
    """Open only wait authority for each stable crash-Job member."""
    if (
        not process_ids
        or len(set(process_ids)) != len(process_ids)
        or any(
            type(process_id) is not int or process_id <= 0 for process_id in process_ids
        )
    ):
        raise QualificationError("crash descendant identity set is invalid")
    _, wintypes, kernel32 = _windows_api()
    kernel32.OpenProcess.argtypes = [
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handles: list[Any] = []
    try:
        for process_id in process_ids:
            handle = kernel32.OpenProcess(SYNCHRONIZE_ACCESS, False, process_id)
            if not handle:
                _raise_windows_error("OpenProcess")
            handles.append(handle)
    except (OSError, QualificationError):
        for handle in handles:
            kernel32.CloseHandle(handle)
        raise
    return kernel32, handles


def _wait_retained_process_handles(
    kernel32: Any,
    handles: Sequence[Any],
    *,
    timeout_seconds: float,
) -> tuple[bool, int]:
    """Wait on retained identities and fail closed on timeout or API failure."""
    if not handles or timeout_seconds <= 0:
        for handle in handles:
            kernel32.CloseHandle(handle)
        return False, 0
    deadline = time.monotonic() + timeout_seconds
    wait_object_0_count = 0
    all_closed = True
    try:
        for handle in handles:
            remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
            if kernel32.WaitForSingleObject(handle, remaining_ms) == WAIT_OBJECT_0:
                wait_object_0_count += 1
    finally:
        for handle in handles:
            if not kernel32.CloseHandle(handle):
                all_closed = False
    return (
        all_closed and wait_object_0_count == len(handles),
        wait_object_0_count,
    )


_NORMAL_CLEANUP_FACT_KEYS = {
    "normal_cleanup_expected_process_count",
    "normal_cleanup_retained_handle_count",
    "normal_cleanup_wait_object_0_count",
    "normal_cleanup_all_wait_object_0",
}


def _normal_cleanup_complete(facts: dict[str, object]) -> bool:
    if set(facts) != _NORMAL_CLEANUP_FACT_KEYS:
        return False
    expected = facts["normal_cleanup_expected_process_count"]
    retained = facts["normal_cleanup_retained_handle_count"]
    waited = facts["normal_cleanup_wait_object_0_count"]
    return bool(
        type(expected) is int
        and expected > 0
        and type(retained) is int
        and retained == expected
        and type(waited) is int
        and waited == expected
        and facts["normal_cleanup_all_wait_object_0"] is True
    )


def _normal_cleanup_facts(
    job: _WindowsJob,
    process: Any,
    *,
    timeout_seconds: float,
) -> dict[str, object]:
    """Retain exact process identities, close the Job, wait, and reap."""
    if timeout_seconds <= 0:
        raise QualificationError("normal cleanup timeout is invalid")
    process_ids = job.process_ids()
    if (
        not process_ids
        or len(set(process_ids)) != len(process_ids)
        or any(
            type(process_id) is not int or process_id <= 0 for process_id in process_ids
        )
    ):
        raise QualificationError("normal cleanup process identity set is invalid")
    expected_count = len(process_ids)
    retained_handles = job.retain_process_handles(process_ids)
    retained_count = len(retained_handles)
    if retained_count != expected_count:
        close_results = [
            bool(job._kernel32.CloseHandle(handle)) for handle in retained_handles
        ]
        if not all(close_results):
            raise QualificationError("normal cleanup retained handle close failed")
        raise QualificationError("normal cleanup retained handle count is partial")
    job.close()
    all_waited, wait_count = _wait_retained_process_handles(
        job._kernel32,
        retained_handles,
        timeout_seconds=timeout_seconds,
    )
    process.wait(timeout=timeout_seconds)
    facts: dict[str, object] = {
        "normal_cleanup_expected_process_count": expected_count,
        "normal_cleanup_retained_handle_count": retained_count,
        "normal_cleanup_wait_object_0_count": wait_count,
        "normal_cleanup_all_wait_object_0": bool(
            all_waited and wait_count == expected_count
        ),
    }
    if not _normal_cleanup_complete(facts):
        raise QualificationError("normal cleanup did not reach WAIT_OBJECT_0")
    return facts


def _commit_native_observations(
    published: dict[str, object],
    candidate: dict[str, object],
    *,
    cleanup_action: Callable[[], dict[str, object]],
) -> None:
    """Publish candidate observations only after exact normal cleanup succeeds."""
    cleanup_facts = cleanup_action()
    if not _normal_cleanup_complete(cleanup_facts):
        raise QualificationError("normal cleanup facts are incomplete")
    completed = dict(candidate)
    completed.update(cleanup_facts)
    published.clear()
    published.update(completed)


def _working_set_bytes(process_id: int) -> int | None:
    ctypes, wintypes, kernel32 = _windows_api()
    psapi = ctypes.WinDLL("psapi", use_last_error=True)

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    handle = kernel32.OpenProcess(0x1000 | 0x0010, False, process_id)
    if not handle:
        return None
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    try:
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.WorkingSetSize)
    finally:
        kernel32.CloseHandle(handle)


def _aggregate_working_set(process_ids: Sequence[int]) -> int | None:
    values = [_working_set_bytes(process_id) for process_id in process_ids]
    if not values or any(value is None for value in values):
        return None
    return sum(int(value) for value in values)


def _managed_rss_population(
    *,
    controller_pid: int,
    worker_pid: int,
    job_member_ids: Sequence[int],
    fixture_process_ids: Sequence[int],
) -> tuple[list[int], dict[str, object]]:
    """Classify the exactly-four-session managed RSS population."""
    members = set(job_member_ids)
    fixtures = set(fixture_process_ids)
    if (
        len(fixture_process_ids) != 4
        or len(fixtures) != 4
        or len(members) != len(job_member_ids)
        or worker_pid not in members
        or not fixtures <= members
        or controller_pid in members
        or worker_pid in fixtures
    ):
        raise QualificationError("four-session RSS population is ambiguous")
    helpers = sorted(members - fixtures - {worker_pid})
    managed = [controller_pid, worker_pid, *helpers]
    return managed, {
        "rss_controller_process_count": 1,
        "rss_worker_process_count": 1,
        "rss_helper_process_count": len(helpers),
        "rss_fixture_process_count": len(fixtures),
        "rss_fixture_processes_excluded": not bool(fixtures & set(managed)),
        "rss_ipc_included_in_worker": True,
        "rss_sample_live_session_count": len(fixtures),
        "rss_crash_session_present": False,
    }


@dataclass(frozen=True)
class _OutputChunk:
    """One measured pywinpty output chunk that requires acknowledgement."""

    data: bytes
    sequence: int


class _OutputCredit:
    """Allow one measured, explicitly acknowledged ``PTY.read`` at a time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._outstanding = 0
        self._current: _OutputChunk | None = None
        self._next_sequence = 1
        self.max_unacknowledged = 0
        self.max_chunk_bytes = 0
        self.measured_chunk_count = 0
        self.read_api_accepts_size = False
        self.upstream_read_buffer_bytes = UPSTREAM_READ_BUFFER_BYTES

    @property
    def outstanding(self) -> int:
        with self._lock:
            return self._outstanding

    def read(self, terminal: Any, *, blocking: bool = True) -> _OutputChunk:
        with self._lock:
            if self._outstanding:
                raise QualificationError("output credit already outstanding")
            self._outstanding = 1
            self.max_unacknowledged = max(self.max_unacknowledged, self._outstanding)
        try:
            value = terminal.read(blocking=blocking)
            if not isinstance(value, str):
                raise QualificationError("ConPTY PTY.read returned a non-string chunk")
            data = value.encode("utf-8")
            if len(data) > CREDIT_LIMIT_BYTES:
                raise QualificationError("bounded ConPTY read exceeded its credit")
            with self._lock:
                chunk = _OutputChunk(data=data, sequence=self._next_sequence)
                self._next_sequence += 1
                self._current = chunk
                self.measured_chunk_count += 1
                self.max_chunk_bytes = max(self.max_chunk_bytes, len(data))
            return chunk
        except BaseException:
            with self._lock:
                self._outstanding = 0
                self._current = None
            raise

    def acknowledge(self, chunk: _OutputChunk) -> None:
        """Release exactly the currently outstanding measured chunk."""
        with self._lock:
            if self._outstanding != 1 or self._current is not chunk:
                raise QualificationError("output acknowledgement does not match credit")
            self._outstanding = 0
            self._current = None


def _sequenced_output(
    *, frame_count: int, payload_bytes: int
) -> tuple[list[bytes], bytes]:
    """Build deterministic, independently sequence-checkable output frames."""
    if frame_count <= 1 or payload_bytes <= UPSTREAM_READ_BUFFER_BYTES:
        raise QualificationError("integrity fixture must span multiple read buffers")
    frames: list[bytes] = []
    for index in range(frame_count):
        payload = bytes((65 + index % 26,)) * payload_bytes
        frames.append(
            f"<TLDW22512:{index:04d}:BEGIN>".encode()
            + payload
            + f"<TLDW22512:{index:04d}:END>".encode()
        )
    return frames, b"".join(frames)


def _sequence_complete(captured: bytes, frames: Sequence[bytes]) -> bool:
    cursor = 0
    for frame in frames:
        offset = captured.find(frame, cursor)
        if offset < 0:
            return False
        cursor = offset + len(frame)
    return True


def _drain_after_exit(
    terminal: Any,
    credit: _OutputCredit,
    *,
    frames: Sequence[bytes],
    expected: bytes,
    deadline_seconds: float,
    post_exit_seconds: float,
) -> dict[str, object]:
    """Drain through process exit until observed EOF or a bounded missing-EOF stop."""
    if deadline_seconds <= 0 or post_exit_seconds <= 0:
        raise QualificationError("post-exit drain deadline is invalid")
    captured = bytearray()
    capture_limit = len(expected) + CREDIT_LIMIT_BYTES
    deadline = time.monotonic() + deadline_seconds
    post_exit_deadline: float | None = None
    eof_observed = False
    while time.monotonic() < deadline and len(captured) < capture_limit:
        chunk: _OutputChunk | None = None
        try:
            chunk = credit.read(terminal, blocking=False)
            captured.extend(chunk.data)
        except (OSError, QualificationError):
            pass
        finally:
            if chunk is not None:
                credit.acknowledge(chunk)
        try:
            alive = bool(terminal.isalive())
        except OSError:
            alive = False
        if not alive and post_exit_deadline is None:
            post_exit_deadline = time.monotonic() + post_exit_seconds
        if post_exit_deadline is not None:
            try:
                eof_observed = bool(terminal.iseof())
            except OSError:
                eof_observed = False
            if eof_observed and (chunk is None or not chunk.data):
                break
            if time.monotonic() >= post_exit_deadline:
                break
        if chunk is None or not chunk.data:
            time.sleep(0.01)
    bounded_missing_eof = bool(post_exit_deadline is not None and not eof_observed)
    actual = bytes(captured)
    return {
        "captured_byte_count": len(actual),
        "sequence_complete": _sequence_complete(actual, frames),
        "digest_equal": hashlib.sha256(actual).digest()
        == hashlib.sha256(expected).digest(),
        "eof_observed": eof_observed,
        "missing_eof_bounded": bounded_missing_eof,
        "post_exit_drain_bounded": post_exit_deadline is not None
        and (eof_observed or bounded_missing_eof),
    }


def _fixture_source() -> str:
    return (
        "import os,subprocess,sys,time\n"
        f"marker={MARKER!r}\n"
        "mode=sys.argv[1]\n"
        "if mode == 'integrity':\n"
        f" for index in range({INTEGRITY_FRAME_COUNT}):\n"
        f"  payload=bytes((65 + index % 26,))*{INTEGRITY_PAYLOAD_BYTES}\n"
        "  begin=('<TLDW22512:%04d:BEGIN>' % index).encode()\n"
        "  end=('<TLDW22512:%04d:END>' % index).encode()\n"
        "  sys.stdout.buffer.write(begin+payload+end)\n"
        " sys.stdout.buffer.flush()\n"
        " os.abort()\n"
        "sys.stdout.write('PRIMARY:'+marker+'\\x1b[?1049hALT\\x1b[?1049l\\n')\n"
        "sys.stdout.flush()\n"
        "if mode in ('terminal-crash','crash-live'):\n"
        " child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'],close_fds=True)\n"
        " sys.stdout.write('TLDW22512-DESCENDANT=%d\\n' % child.pid)\n"
        " sys.stdout.flush()\n"
        " if mode == 'terminal-crash':\n"
        "  sys.stdin.buffer.read(1)\n"
        "  os.abort()\n"
        "while True:\n"
        " value=sys.stdin.buffer.read(1)\n"
        " if value:\n"
        "  sys.stdout.buffer.write(value); sys.stdout.buffer.flush()\n"
        " else:\n"
        "  time.sleep(0.05)\n"
    )


def _spawn_session(winpty: Any, mode: str) -> Any:
    terminal = winpty.PTY(80, 24, backend=winpty.Backend.ConPTY)
    argv = ["-u", "-c", _fixture_source(), mode]
    spawned = terminal.spawn(
        sys.executable,
        subprocess.list2cmdline(argv),
        cwd=tempfile.gettempdir(),
        env=None,
    )
    if not spawned or not isinstance(terminal.pid, int):
        raise QualificationError("low-level ConPTY spawn failed")
    return terminal


def _terminate_terminal_processes(terminals: Sequence[Any]) -> None:
    """Terminate live fixture processes through pywinpty's process handles."""
    _, wintypes, kernel32 = _windows_api()
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateProcess.restype = wintypes.BOOL
    handles = [terminal.fd for terminal in terminals if terminal.isalive()]
    if not handles:
        return
    if any(type(handle) is not int or handle <= 0 for handle in handles):
        raise QualificationError("ConPTY process handle is unavailable for close")
    for handle in handles:
        if not kernel32.TerminateProcess(handle, 1):
            _raise_windows_error("TerminateProcess")


def _read_until_pattern(
    terminal: Any,
    credit: _OutputCredit,
    pattern: re.Pattern[bytes],
    *,
    timeout: float,
) -> tuple[bytes, bool]:
    """Read measured chunks until a content-free fixture marker is complete."""
    captured = bytearray()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and len(captured) < CREDIT_LIMIT_BYTES * 4:
        chunk: _OutputChunk | None = None
        try:
            chunk = credit.read(terminal, blocking=False)
            captured.extend(chunk.data)
        except (OSError, QualificationError):
            pass
        finally:
            if chunk is not None:
                credit.acknowledge(chunk)
        if pattern.search(captured):
            return bytes(captured), True
        time.sleep(0.01)
    return bytes(captured), False


def _read_in_thread(
    terminal: Any, credit: _OutputCredit, *, timeout: float
) -> tuple[bytes, bool, bool]:
    state: dict[str, object] = {"value": b"", "completed": False}

    def read_once() -> None:
        chunk: _OutputChunk | None = None
        try:
            chunk = credit.read(terminal, blocking=True)
            state["value"] = chunk.data
        except (OSError, QualificationError):
            state["value"] = b""
        finally:
            if chunk is not None:
                credit.acknowledge(chunk)
            state["completed"] = True

    reader = threading.Thread(target=read_once, daemon=True)
    reader.start()
    reader.join(timeout=timeout)
    cancellation_required = reader.is_alive()
    if reader.is_alive():
        try:
            terminal.cancel_io()
        except OSError:
            pass
        reader.join(timeout=1.0)
    value = state["value"]
    return (
        bytes(value) if isinstance(value, bytes) else b"",
        not reader.is_alive(),
        cancellation_required,
    )


def _alternate_facts(captured: bytes) -> dict[str, bool]:
    try:
        import pyte
        from pyte_probe import _AlternateScreenAdapter

        adapter = _AlternateScreenAdapter(pyte, columns=80, lines=24)
        stream = pyte.Stream(adapter)
        stream.feed(captured.decode("utf-8", "replace"))
        primary = "".join(adapter.primary.display)
        alternate = "".join(adapter.alternate.display)
        return {
            "unicode_roundtrip": MARKER in primary,
            "alternate_screen": adapter.entry_count >= 1 and adapter.exit_count >= 1,
            "alternate_isolated": "ALT" in alternate and "ALT" not in primary,
            "primary_restored": not adapter.in_alternate
            and adapter.active is adapter.primary
            and "PRIMARY" in primary,
        }
    except (ImportError, UnicodeError, ValueError):
        return {
            "unicode_roundtrip": False,
            "alternate_screen": False,
            "alternate_isolated": False,
            "primary_restored": False,
        }


def _drain_terminal_until_quiet(
    terminal: Any,
    credit: _OutputCredit,
    *,
    deadline: float,
    quiet_seconds: float,
) -> tuple[bool, bool]:
    """Drain known startup output and require a bounded quiet interval."""
    if quiet_seconds <= 0:
        raise QualificationError("quiet ConPTY interval is invalid")
    startup_drained = False
    last_output_at: float | None = None
    while time.monotonic() < deadline:
        chunk: _OutputChunk | None = None
        try:
            chunk = credit.read(terminal, blocking=False)
            if chunk.data:
                startup_drained = True
                last_output_at = time.monotonic()
        except (OSError, QualificationError):
            pass
        finally:
            if chunk is not None:
                credit.acknowledge(chunk)
        now = time.monotonic()
        if (
            startup_drained
            and last_output_at is not None
            and now - last_output_at >= quiet_seconds
        ):
            return True, True
        time.sleep(min(0.01, quiet_seconds))
    return startup_drained, False


def _concurrent_operations(
    terminals: Sequence[Any],
    credit: _OutputCredit,
    *,
    close_action: Callable[[], None] | None = None,
    timeout: float = 5.0,
    startup_quiet_seconds: float = 0.1,
) -> dict[str, object]:
    """Close after synchronous calls return and one quiet-terminal read blocks."""
    if len(terminals) < 4 or timeout <= 0 or startup_quiet_seconds <= 0:
        raise QualificationError("concurrent ConPTY operation setup is invalid")
    names = ("read", "write", "resize", "cancel")
    ready = {name: threading.Event() for name in names}
    entered = {name: threading.Event() for name in names}
    returned = {name: threading.Event() for name in names}
    returned_after_close = {name: False for name in names}
    succeeded = {name: False for name in names}
    start = threading.Event()
    close_started = threading.Event()
    deadline = time.monotonic() + timeout
    startup_drained, quiet_before_handoff = _drain_terminal_until_quiet(
        terminals[3],
        credit,
        deadline=deadline,
        quiet_seconds=startup_quiet_seconds,
    )
    if not startup_drained or not quiet_before_handoff:
        raise QualificationError("dedicated ConPTY read terminal did not become quiet")

    def read_action() -> None:
        chunk = credit.read(terminals[3], blocking=True)
        try:
            _ = chunk.data
        finally:
            credit.acknowledge(chunk)

    def operation(name: str, action: Callable[[], object]) -> None:
        ready[name].set()
        if not start.wait(timeout=timeout):
            returned[name].set()
            return
        entered[name].set()
        try:
            action()
            succeeded[name] = True
        except (OSError, QualificationError):
            pass
        finally:
            returned_after_close[name] = close_started.is_set()
            returned[name].set()

    threads = [
        threading.Thread(
            target=operation,
            args=("read", read_action),
            daemon=True,
        ),
        threading.Thread(
            target=operation,
            args=("write", lambda: terminals[0].write("W")),
            daemon=True,
        ),
        threading.Thread(
            target=operation,
            args=("resize", lambda: terminals[2].set_size(100, 30)),
            daemon=True,
        ),
        threading.Thread(
            target=operation,
            args=("cancel", lambda: terminals[1].cancel_io()),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    all_ready = all(
        event.wait(timeout=max(0.0, deadline - time.monotonic()))
        for event in ready.values()
    )
    start.set()
    all_entered = all(
        event.wait(timeout=max(0.0, deadline - time.monotonic()))
        for event in entered.values()
    )
    synchronous_completed = all(
        returned[name].wait(timeout=max(0.0, deadline - time.monotonic()))
        for name in ("write", "resize", "cancel")
    )
    completed_at_handoff = {name: returned[name].is_set() for name in names}
    blocking_read_unresolved = bool(
        all_entered and synchronous_completed and not completed_at_handoff["read"]
    )
    inflight_count = sum(
        event.is_set() and not returned[name].is_set()
        for name, event in entered.items()
    )
    close_completed = False
    if all_ready and blocking_read_unresolved and close_action is not None:
        close_started.set()
        try:
            close_action()
            close_completed = True
        except (OSError, QualificationError):
            close_completed = False
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    completed_post_close = {
        name: bool(not completed_at_handoff[name] and returned_after_close[name])
        for name in names
    }
    return {
        "quiet_terminal_startup_drained": startup_drained,
        "quiet_terminal_quiescent_before_handoff": quiet_before_handoff,
        "write_completed": succeeded["write"],
        "resize_completed": succeeded["resize"],
        "cancel_completed": succeeded["cancel"],
        "priority_close_completed": close_completed,
        "concurrent_operation_count": sum(event.is_set() for event in entered.values()),
        "inflight_operation_category_count": inflight_count,
        "io_inflight_at_handoff": blocking_read_unresolved,
        "priority_close_preempted_inflight": blocking_read_unresolved
        and close_completed
        and completed_post_close["read"],
        **{f"{name}_entered": entered[name].is_set() for name in names},
        **{
            f"{name}_completed_at_handoff": completed_at_handoff[name] for name in names
        },
        **{
            f"{name}_completed_post_close": completed_post_close[name] for name in names
        },
        **{
            f"{name}_returned_after_close": returned_after_close[name] for name in names
        },
    }


def _worker_observations(
    *,
    report_rss_ready: Callable[[Sequence[int]], None] | None = None,
    await_rss_continue: Callable[[], bool] | None = None,
) -> dict[str, object]:
    import winpty

    observations = _default_observations()
    observations["fresh_worker"] = True
    observations["worker_std_streams_fd_backed"] = all(
        _fd_is_backed(descriptor) for descriptor in (0, 1, 2)
    )
    observations["low_level_api"] = all(
        (
            hasattr(winpty, "PTY"),
            hasattr(winpty, "Backend"),
            hasattr(winpty.Backend, "ConPTY"),
        )
    )
    observations["job_admitted_before_conpty"] = _process_is_in_job()
    if (
        not observations["low_level_api"]
        or not observations["job_admitted_before_conpty"]
    ):
        return observations

    live_terminals = [_spawn_session(winpty, "live") for _ in range(4)]
    observations["four_session_count"] = len(live_terminals)
    observations["conpty_constructed"] = len(live_terminals) == 4
    live_members = [_process_is_in_job(terminal.pid) for terminal in live_terminals]
    fixture_process_ids = [terminal.pid for terminal in live_terminals]
    if not all(type(process_id) is int for process_id in fixture_process_ids):
        raise QualificationError("live ConPTY fixture PID is unavailable")
    if report_rss_ready is not None:
        report_rss_ready(fixture_process_ids)
        if await_rss_continue is None or not await_rss_continue():
            raise QualificationError("four-session RSS sampling did not complete")

    credit = _OutputCredit()
    captured, first_read_bounded, _ = _read_in_thread(
        live_terminals[0], credit, timeout=5.0
    )
    observations.update(_alternate_facts(captured))
    observations["output_integrity"] = bool(
        first_read_bounded and MARKER.encode("utf-8") in captured
    )

    concurrent = _concurrent_operations(
        live_terminals,
        credit,
        close_action=lambda: _terminate_terminal_processes(live_terminals),
    )
    observations.update(concurrent)
    try:
        _terminate_terminal_processes(live_terminals)
    except (OSError, QualificationError):
        pass

    crash_terminal = _spawn_session(winpty, "terminal-crash")
    crash_output, descendant_marker_found = _read_until_pattern(
        crash_terminal, credit, DESCENDANT_RE, timeout=5.0
    )
    descendant_match = DESCENDANT_RE.search(crash_output)
    descendant_id = int(descendant_match.group(1)) if descendant_match else None
    observations["terminal_child_member_before_crash"] = _process_is_in_job(
        crash_terminal.pid
    )
    crash_terminal.write("!")
    crash_deadline = time.monotonic() + 5.0
    while crash_terminal.isalive() and time.monotonic() < crash_deadline:
        time.sleep(0.05)
    observations["terminal_child_crash_observed"] = not crash_terminal.isalive()
    observations["terminal_child_eof_observed"] = bool(crash_terminal.iseof())
    observations["terminal_grandchild_member_before_crash"] = bool(
        descendant_id is not None and _process_is_in_job(descendant_id)
    )
    observations["output_integrity"] = bool(
        observations["output_integrity"] and descendant_marker_found
    )

    frames, expected = _sequenced_output(
        frame_count=INTEGRITY_FRAME_COUNT,
        payload_bytes=INTEGRITY_PAYLOAD_BYTES,
    )
    integrity_terminal = _spawn_session(winpty, "integrity")
    integrity_member = _process_is_in_job(integrity_terminal.pid)
    integrity_facts = _drain_after_exit(
        integrity_terminal,
        credit,
        frames=frames,
        expected=expected,
        deadline_seconds=10.0,
        post_exit_seconds=1.0,
    )
    observations.update(integrity_facts)
    observations["output_integrity"] = bool(
        observations["output_integrity"]
        and integrity_facts["sequence_complete"]
        and integrity_facts["digest_equal"]
    )
    observations["job_membership_complete"] = bool(
        all(live_members)
        and observations["terminal_child_member_before_crash"]
        and observations["terminal_grandchild_member_before_crash"]
        and integrity_member
    )

    observations["max_concurrent_readers"] = 1
    observations["one_credit_max_bytes"] = credit.max_chunk_bytes
    observations["measured_chunk_count"] = credit.measured_chunk_count
    observations["read_api_accepts_size"] = credit.read_api_accepts_size
    observations["upstream_read_buffer_bytes"] = credit.upstream_read_buffer_bytes
    observations["max_unacknowledged_credits"] = credit.max_unacknowledged
    return observations


def _fd_is_backed(descriptor: int) -> bool:
    try:
        os.fstat(descriptor)
    except OSError:
        return False
    return True


def _write_worker_result(path: Path, observations: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(observations, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_rss_fixture_ids(path: Path, process_ids: Sequence[int]) -> None:
    if (
        len(process_ids) != 4
        or len(set(process_ids)) != 4
        or any(
            type(process_id) is not int or process_id <= 0 for process_id in process_ids
        )
    ):
        raise QualificationError("four-session RSS fixture identity is invalid")
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="ascii") as stream:
        json.dump(list(process_ids), stream)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_rss_fixture_ids(path: Path) -> list[int]:
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError(
            "four-session RSS fixture identity is unreadable"
        ) from exc
    if (
        not isinstance(payload, list)
        or len(payload) != 4
        or len(set(payload)) != 4
        or any(type(process_id) is not int or process_id <= 0 for process_id in payload)
    ):
        raise QualificationError("four-session RSS fixture identity is invalid")
    return payload


def _native_worker(
    result_path: Path,
    start_event: str,
    ready_event: str,
    rss_fixture_path: Path,
    rss_ready_event: str,
    rss_continue_event: str,
) -> int:
    if not _wait_named_event(start_event, WORKER_TIMEOUT_SECONDS):
        return 3

    def report_rss_ready(process_ids: Sequence[int]) -> None:
        _write_rss_fixture_ids(rss_fixture_path, process_ids)
        _signal_named_event(rss_ready_event)

    try:
        observations = _worker_observations(
            report_rss_ready=report_rss_ready,
            await_rss_continue=lambda: _wait_named_event(
                rss_continue_event, WORKER_TIMEOUT_SECONDS
            ),
        )
    except (
        ImportError,
        OSError,
        QualificationError,
        subprocess.SubprocessError,
    ) as exc:
        if isinstance(exc, QualificationError):
            category = re.sub(r"[^A-Za-z0-9_.-]", "_", str(exc))[:160]
        elif isinstance(exc, OSError):
            category = (
                f"winerror-{getattr(exc, 'winerror', None)}-"
                f"errno-{getattr(exc, 'errno', None)}"
            )
        else:
            category = type(exc).__name__
        print(
            f"TASK22512_WORKER_FAILURE:{type(exc).__name__}:{category}",
            file=sys.stderr,
        )
        observations = _default_observations()
        observations["fresh_worker"] = True
        observations["worker_std_streams_fd_backed"] = all(
            _fd_is_backed(descriptor) for descriptor in (0, 1, 2)
        )
    _write_worker_result(result_path, observations)
    _signal_named_event(ready_event)
    while True:
        time.sleep(60.0)


def _load_worker_result(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError("native worker result is unreadable") from exc
    expected = _default_observations()
    if not isinstance(payload, dict) or set(payload) != set(expected):
        raise QualificationError("native worker result shape is invalid")
    for key, value in payload.items():
        template = expected[key]
        if template is None:
            if value is not None:
                raise QualificationError("native worker nullable fact is invalid")
        elif type(value) is not type(template):
            raise QualificationError("native worker fact type is invalid")
    return payload


def _write_crash_result(path: Path, payload: dict[str, object]) -> None:
    """Durably hand content-free process identities to the local supervisor."""
    _write_worker_result(path, payload)


def _load_crash_worker_result(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError("native crash worker result is unreadable") from exc
    expected = {
        "worker_process_id": int,
        "terminal_process_id": int,
        "terminal_descendant_process_id": int,
        "worker_admitted_before_conpty": bool,
        "terminal_member": bool,
        "terminal_descendant_member": bool,
    }
    if not isinstance(payload, dict) or set(payload) != set(expected):
        raise QualificationError("native crash worker result shape is invalid")
    for key, expected_type in expected.items():
        value = payload[key]
        if type(value) is not expected_type:
            raise QualificationError("native crash worker fact type is invalid")
        if expected_type is int and value <= 0:
            raise QualificationError("native crash worker identity is invalid")
    return payload


def _load_crash_app_result(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError("native crash app result is unreadable") from exc
    expected = {
        "app_process_id",
        "job_handle_owner_count",
        "job_handle_non_inheritable",
        "worker_admitted_before_conpty",
        "descendant_set_stable",
        "known_descendant_process_ids",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise QualificationError("native crash app result shape is invalid")
    app_process_id = payload["app_process_id"]
    owner_count = payload["job_handle_owner_count"]
    process_ids = payload["known_descendant_process_ids"]
    if (
        type(app_process_id) is not int
        or app_process_id <= 0
        or type(owner_count) is not int
        or owner_count != 1
        or type(payload["job_handle_non_inheritable"]) is not bool
        or type(payload["worker_admitted_before_conpty"]) is not bool
        or type(payload["descendant_set_stable"]) is not bool
        or not isinstance(process_ids, list)
        or len(process_ids) < 3
        or len(set(process_ids)) != len(process_ids)
        or any(
            type(process_id) is not int or process_id <= 0 for process_id in process_ids
        )
        or app_process_id in process_ids
    ):
        raise QualificationError("native crash app result is invalid")
    return payload


def _stable_job_process_ids(
    job: _WindowsJob,
    required_process_ids: Sequence[int],
    *,
    timeout_seconds: float,
) -> list[int]:
    """Require repeated identical complete Job membership observations."""
    required = set(required_process_ids)
    deadline = time.monotonic() + timeout_seconds
    previous: list[int] | None = None
    matching_samples = 0
    while time.monotonic() < deadline:
        process_ids = sorted(job.process_ids())
        if required <= set(process_ids) and process_ids == previous:
            matching_samples += 1
            if matching_samples >= STABLE_JOB_SAMPLE_COUNT:
                return process_ids
        else:
            matching_samples = 1 if required <= set(process_ids) else 0
        previous = process_ids
        time.sleep(0.05)
    raise QualificationError("native crash Job membership did not stabilize")


def _native_crash_worker(
    result_path: Path,
    start_event: str,
    ready_event: str,
) -> int:
    """Create crash descendants only after the app-owned Job admits this worker."""
    if not _wait_named_event(start_event, WORKER_TIMEOUT_SECONDS):
        return 3
    import winpty

    worker_admitted = _process_is_in_job()
    if not worker_admitted:
        return 4
    terminal = _spawn_session(winpty, "crash-live")
    credit = _OutputCredit()
    captured, marker_found = _read_until_pattern(
        terminal, credit, DESCENDANT_RE, timeout=5.0
    )
    descendant_match = DESCENDANT_RE.search(captured)
    descendant_id = int(descendant_match.group(1)) if descendant_match else None
    if not marker_found or descendant_id is None:
        try:
            terminal_alive = bool(terminal.isalive())
        except OSError:
            terminal_alive = False
        try:
            terminal_eof = bool(terminal.iseof())
        except OSError:
            terminal_eof = False
        print(
            "TASK22512_CRASH_WORKER_CAPTURE:"
            + json.dumps(
                {
                    "captured_byte_count": len(captured),
                    "descendant_prefix_present": b"TLDW22512-DESCENDANT="
                    in captured,
                    "max_chunk_bytes": credit.max_chunk_bytes,
                    "measured_chunk_count": credit.measured_chunk_count,
                    "primary_prefix_present": b"PRIMARY:" in captured,
                    "terminal_alive": terminal_alive,
                    "terminal_eof": terminal_eof,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 5
    payload: dict[str, object] = {
        "worker_process_id": os.getpid(),
        "terminal_process_id": terminal.pid,
        "terminal_descendant_process_id": descendant_id,
        "worker_admitted_before_conpty": worker_admitted,
        "terminal_member": _process_is_in_job(terminal.pid),
        "terminal_descendant_member": _process_is_in_job(descendant_id),
    }
    _write_crash_result(result_path, payload)
    _signal_named_event(ready_event)
    while True:
        time.sleep(60.0)


def _native_crash_app_controller(
    app_result_path: Path,
    worker_result_path: Path,
    supervisor_ready_event: str,
    supervisor_retained_event: str,
) -> int:
    """Sole-own a kill-on-close Job and abort after supervisor handoff."""
    worker_start_name = f"Local\\tldw-task22512-crash-start-{uuid.uuid4().hex}"
    worker_ready_name = f"Local\\tldw-task22512-crash-worker-{uuid.uuid4().hex}"
    worker_start_handle = _create_event(worker_start_name)
    worker_ready_handle = _create_event(worker_ready_name)
    job = _WindowsJob()
    process: subprocess.Popen[bytes] | None = None
    try:
        with (
            tempfile.TemporaryFile() as worker_stdin,
            tempfile.TemporaryFile() as worker_stdout,
            tempfile.TemporaryFile() as worker_stderr,
        ):
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--native-crash-worker",
                    "--crash-worker-result",
                    str(worker_result_path),
                    "--crash-start-event",
                    worker_start_name,
                    "--crash-worker-ready-event",
                    worker_ready_name,
                ],
                stdin=worker_stdin,
                stdout=worker_stdout,
                stderr=worker_stderr,
                close_fds=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            job.assign(process.pid)
            admitted = job.contains(process.pid)
            _set_event(worker_start_handle)
            if not _wait_event(worker_ready_handle, WORKER_TIMEOUT_SECONDS):
                worker_stderr.seek(0)
                for line in worker_stderr.read(4096).decode(
                    "utf-8", "replace"
                ).splitlines():
                    if line.startswith("TASK22512_CRASH_WORKER_CAPTURE:"):
                        print(line, file=sys.stderr)
                print(
                    f"TASK22512_CRASH_WORKER_EXIT:{process.poll()}",
                    file=sys.stderr,
                )
                raise QualificationError("native crash worker readiness timeout")
            worker = _load_crash_worker_result(worker_result_path)
            required_process_ids = [
                worker["worker_process_id"],
                worker["terminal_process_id"],
                worker["terminal_descendant_process_id"],
            ]
            if (
                worker["worker_process_id"] != process.pid
                or not admitted
                or worker["worker_admitted_before_conpty"] is not True
                or worker["terminal_member"] is not True
                or worker["terminal_descendant_member"] is not True
            ):
                raise QualificationError("native crash worker admission is invalid")
            process_ids = _stable_job_process_ids(
                job,
                required_process_ids,
                timeout_seconds=WORKER_TIMEOUT_SECONDS,
            )
            if not all(job.contains(process_id) for process_id in process_ids):
                raise QualificationError("native crash Job membership is incomplete")
            _write_crash_result(
                app_result_path,
                {
                    "app_process_id": os.getpid(),
                    "job_handle_owner_count": 1,
                    "job_handle_non_inheritable": job.non_inheritable(),
                    "worker_admitted_before_conpty": True,
                    "descendant_set_stable": True,
                    "known_descendant_process_ids": process_ids,
                },
            )
            _signal_named_event(supervisor_ready_event)
            if not _wait_named_event(supervisor_retained_event, WORKER_TIMEOUT_SECONDS):
                raise QualificationError("native crash supervisor handoff timeout")
            os.abort()
    finally:
        if job.handle:
            job.close()
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=CLOSE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=CLOSE_TIMEOUT_SECONDS)
        _close_handle(worker_start_handle)
        _close_handle(worker_ready_handle)


def _run_app_crash_supervisor() -> dict[str, object]:
    """Observe app-owned Job teardown using only descendant wait handles."""
    ready_name = f"Local\\tldw-task22512-crash-ready-{uuid.uuid4().hex}"
    retained_name = f"Local\\tldw-task22512-crash-retained-{uuid.uuid4().hex}"
    ready_handle = _create_event(ready_name)
    retained_handle = _create_event(retained_name)
    process: subprocess.Popen[bytes] | None = None
    kernel32: Any = None
    retained_handles: list[Any] = []
    facts: dict[str, object] = {
        "app_crash_observed": False,
        "crash_app_process_separate": False,
        "crash_app_sole_job_handle_owner": False,
        "crash_job_handle_non_inheritable": False,
        "crash_worker_admitted_before_conpty": False,
        "crash_descendant_set_stable": False,
        "crash_descendants_ready_before_abort": False,
        "crash_supervisor_job_handle_count": 0,
        "crash_known_descendant_count": 0,
        "crash_supervisor_synchronize_handle_count": 0,
        "crash_wait_object_0_count": 0,
        "crash_all_descendants_wait_object_0": False,
    }
    try:
        with tempfile.TemporaryDirectory(
            prefix="tldw-task22512-crash-supervisor-"
        ) as raw:
            app_result_path = Path(raw) / "app.json"
            worker_result_path = Path(raw) / "worker.json"
            with (
                tempfile.TemporaryFile() as app_stdin,
                tempfile.TemporaryFile() as app_stdout,
                tempfile.TemporaryFile() as app_stderr,
            ):
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--native-crash-app-controller",
                        "--crash-app-result",
                        str(app_result_path),
                        "--crash-worker-result",
                        str(worker_result_path),
                        "--crash-ready-event",
                        ready_name,
                        "--crash-retained-event",
                        retained_name,
                    ],
                    stdin=app_stdin,
                    stdout=app_stdout,
                    stderr=app_stderr,
                    close_fds=True,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
                if not _wait_event(ready_handle, WORKER_TIMEOUT_SECONDS):
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        pass
                    app_stderr.seek(0)
                    for line in app_stderr.read(4096).decode(
                        "ascii", "ignore"
                    ).splitlines():
                        if line.startswith(
                            (
                                "TASK22512_TOP_LEVEL_FAILURE:",
                                "TASK22512_CRASH_WORKER_CAPTURE:",
                                "TASK22512_CRASH_WORKER_EXIT:",
                            )
                        ):
                            print(line, file=sys.stderr)
                    raise QualificationError("native crash app readiness timeout")
                app = _load_crash_app_result(app_result_path)
                process_ids = app["known_descendant_process_ids"]
                if app["app_process_id"] != process.pid:
                    raise QualificationError("native crash app identity is invalid")
                kernel32, retained_handles = _open_synchronize_process_handles(
                    process_ids
                )
                facts.update(
                    {
                        "crash_app_process_separate": bool(
                            process.pid != os.getpid()
                            and process.pid not in process_ids
                        ),
                        "crash_app_sole_job_handle_owner": app["job_handle_owner_count"]
                        == 1,
                        "crash_job_handle_non_inheritable": app[
                            "job_handle_non_inheritable"
                        ],
                        "crash_worker_admitted_before_conpty": app[
                            "worker_admitted_before_conpty"
                        ],
                        "crash_descendant_set_stable": app["descendant_set_stable"],
                        "crash_known_descendant_count": len(process_ids),
                        "crash_supervisor_synchronize_handle_count": len(
                            retained_handles
                        ),
                    }
                )
                facts["crash_descendants_ready_before_abort"] = bool(
                    facts["crash_app_process_separate"]
                    and facts["crash_app_sole_job_handle_owner"]
                    and facts["crash_job_handle_non_inheritable"]
                    and facts["crash_worker_admitted_before_conpty"]
                    and facts["crash_descendant_set_stable"]
                    and len(retained_handles) == len(process_ids)
                )
                _set_event(retained_handle)
                exit_code = process.wait(timeout=CLOSE_TIMEOUT_SECONDS)
                facts["app_crash_observed"] = exit_code != 0
                all_waited, wait_count = _wait_retained_process_handles(
                    kernel32,
                    retained_handles,
                    timeout_seconds=CLOSE_TIMEOUT_SECONDS,
                )
                retained_handles = []
                facts["crash_wait_object_0_count"] = wait_count
                facts["crash_all_descendants_wait_object_0"] = bool(
                    all_waited and wait_count == len(process_ids)
                )
                return facts
    finally:
        if kernel32 is not None:
            for handle in retained_handles:
                kernel32.CloseHandle(handle)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=CLOSE_TIMEOUT_SECONDS)
        _close_handle(ready_handle)
        _close_handle(retained_handle)


def _environment_row_facts(manifest_path: Path) -> dict[str, bool]:
    default_discovery: list[bool] = []
    extended_discovery: list[bool] = []
    rows_passed: list[bool] = []
    payloads: list[dict[str, Any]] = []
    probes = (
        "environment-default",
        "environment-powershell",
        "environment-cmd",
    )
    for probe in probes:
        try:
            payload = json.loads(
                (manifest_path.parent / f"{probe}.json").read_text(encoding="utf-8")
            )
            if not isinstance(payload, dict):
                raise QualificationError("environment evidence root is invalid")
            validate_content_free(payload)
            if payload.get("probe") != probe:
                raise QualificationError("environment evidence filename differs")
            payloads.append(payload)
        except (OSError, json.JSONDecodeError, QualificationError):
            return {
                "profile_module_discovery": False,
                "default_module_discovery": False,
                "profile_extended_module_discovery": False,
            }
        actual = payload.get("actual_startup")
        if not isinstance(actual, dict):
            rows_passed.append(False)
            default_discovery.append(False)
            extended_discovery.append(False)
            continue
        default = actual.get("default_module_discovery") is True
        extended = actual.get("profile_extended_module_discovery") is True
        startup_passed = all(
            actual.get(key) is True
            for key in (
                "startup_completed",
                "command_discovery",
                "profile_marker_present",
                "sensitive_key_repopulated_by_profile",
                "capture_within_bound",
            )
        )
        rows_passed.append(payload.get("status") == "PASS" and startup_passed)
        default_discovery.append(default)
        extended_discovery.append(extended)
    try:
        validate_sibling_identity(payloads, require_generation=False)
    except QualificationError:
        return {
            "profile_module_discovery": False,
            "default_module_discovery": False,
            "profile_extended_module_discovery": False,
        }
    default_passed = len(default_discovery) == len(probes) and all(default_discovery)
    extended_passed = len(extended_discovery) == len(probes) and all(extended_discovery)
    return {
        "profile_module_discovery": len(rows_passed) == len(probes)
        and all(rows_passed)
        and default_passed
        and extended_passed,
        "default_module_discovery": default_passed,
        "profile_extended_module_discovery": extended_passed,
    }


def _run_native_controller(
    manifest_path: Path,
    observations: dict[str, object],
) -> None:
    del manifest_path
    candidate = dict(observations)
    start_name = f"Local\\tldw-task22512-start-{uuid.uuid4().hex}"
    rss_ready_name = f"Local\\tldw-task22512-rss-ready-{uuid.uuid4().hex}"
    rss_continue_name = f"Local\\tldw-task22512-rss-continue-{uuid.uuid4().hex}"
    ready_name = f"Local\\tldw-task22512-ready-{uuid.uuid4().hex}"
    start_handle = _create_event(start_name)
    rss_ready_handle = _create_event(rss_ready_name)
    rss_continue_handle = _create_event(rss_continue_name)
    ready_handle = _create_event(ready_name)
    job = _WindowsJob()
    process: subprocess.Popen[bytes] | None = None
    try:
        with tempfile.TemporaryDirectory(
            prefix="tldw-task22512-winpty-controller-"
        ) as raw:
            result_path = Path(raw) / "worker.json"
            rss_fixture_path = Path(raw) / "rss-fixtures.json"
            with (
                tempfile.TemporaryFile() as worker_stdin,
                tempfile.TemporaryFile() as worker_stdout,
                tempfile.TemporaryFile() as worker_stderr,
            ):
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--native-worker",
                        "--worker-result",
                        str(result_path),
                        "--start-event",
                        start_name,
                        "--ready-event",
                        ready_name,
                        "--rss-fixture-result",
                        str(rss_fixture_path),
                        "--rss-ready-event",
                        rss_ready_name,
                        "--rss-continue-event",
                        rss_continue_name,
                    ],
                    stdin=worker_stdin,
                    stdout=worker_stdout,
                    stderr=worker_stderr,
                    close_fds=True,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
                candidate["fresh_worker"] = True
                candidate["worker_std_streams_fd_backed"] = True
                candidate["job_handle_non_inheritable"] = job.non_inheritable()
                job.assign(process.pid)
                assigned = job.contains(process.pid)
                baseline_population = [os.getpid(), process.pid]
                baseline = _aggregate_working_set(baseline_population)
                _set_event(start_handle)
                if not _wait_event(rss_ready_handle, WORKER_TIMEOUT_SECONDS):
                    worker_stderr.seek(0)
                    for line in worker_stderr.read(4096).decode(
                        "ascii", "ignore"
                    ).splitlines():
                        if line.startswith("TASK22512_WORKER_FAILURE:"):
                            print(line, file=sys.stderr)
                    raise QualificationError("four-session RSS readiness timeout")
                fixture_process_ids = _load_rss_fixture_ids(rss_fixture_path)
                sample_member_ids = job.process_ids()
                managed_population, rss_facts = _managed_rss_population(
                    controller_pid=os.getpid(),
                    worker_pid=process.pid,
                    job_member_ids=sample_member_ids,
                    fixture_process_ids=fixture_process_ids,
                )
                sample_membership_complete = bool(
                    assigned
                    and len(sample_member_ids) >= 6
                    and all(
                        job.contains(process_id) for process_id in sample_member_ids
                    )
                )
                current = _aggregate_working_set(managed_population)
                candidate.update(rss_facts)
                candidate["four_session_count"] = len(fixture_process_ids)
                candidate["rss_measurement_complete"] = bool(
                    sample_membership_complete
                    and baseline is not None
                    and current is not None
                )
                if baseline is not None and current is not None:
                    candidate["four_session_rss_delta_bytes"] = max(
                        0, current - baseline
                    )
                _set_event(rss_continue_handle)
                if not _wait_event(ready_handle, WORKER_TIMEOUT_SECONDS):
                    raise QualificationError("native worker readiness timeout")
                worker = _load_worker_result(result_path)
                if worker.get("low_level_api") is not True:
                    worker_stderr.seek(0)
                    for line in worker_stderr.read(4096).decode(
                        "ascii", "ignore"
                    ).splitlines():
                        if line.startswith("TASK22512_WORKER_FAILURE:"):
                            print(line, file=sys.stderr)
                parent_owned = {
                    "artifact_filename",
                    "artifact_sha256",
                    "artifact_size_bytes",
                    "artifact_verified_during_probe",
                    "distribution_version",
                    "primary_file_name",
                    "primary_file_sha256",
                    "record_file_name",
                    "record_file_sha256",
                    "windows_build",
                    "profile_module_discovery",
                    "default_module_discovery",
                    "profile_extended_module_discovery",
                    "job_handle_non_inheritable",
                    "job_member_count",
                    "normal_cleanup_expected_process_count",
                    "normal_cleanup_retained_handle_count",
                    "normal_cleanup_wait_object_0_count",
                    "normal_cleanup_all_wait_object_0",
                    "app_crash_observed",
                    "crash_app_process_separate",
                    "crash_app_sole_job_handle_owner",
                    "crash_job_handle_non_inheritable",
                    "crash_worker_admitted_before_conpty",
                    "crash_descendant_set_stable",
                    "crash_descendants_ready_before_abort",
                    "crash_supervisor_job_handle_count",
                    "crash_known_descendant_count",
                    "crash_supervisor_synchronize_handle_count",
                    "crash_wait_object_0_count",
                    "crash_all_descendants_wait_object_0",
                    "rss_measurement_complete",
                    "four_session_rss_delta_bytes",
                    "rss_controller_process_count",
                    "rss_worker_process_count",
                    "rss_helper_process_count",
                    "rss_fixture_process_count",
                    "rss_fixture_processes_excluded",
                    "rss_ipc_included_in_worker",
                    "rss_sample_live_session_count",
                    "rss_crash_session_present",
                }
                for key in worker:
                    if key not in parent_owned:
                        candidate[key] = worker[key]
                candidate["job_admitted_before_conpty"] = bool(
                    assigned and worker["job_admitted_before_conpty"]
                )
                candidate["job_member_count"] = len(sample_member_ids)
                candidate["job_membership_complete"] = bool(
                    worker["job_membership_complete"] and sample_membership_complete
                )
                candidate.update(_run_app_crash_supervisor())
                _commit_native_observations(
                    observations,
                    candidate,
                    cleanup_action=lambda: _normal_cleanup_facts(
                        job,
                        process,
                        timeout_seconds=CLOSE_TIMEOUT_SECONDS,
                    ),
                )
    finally:
        if job.handle:
            job.close()
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=CLOSE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=CLOSE_TIMEOUT_SECONDS)
        _close_handle(start_handle)
        _close_handle(rss_ready_handle)
        _close_handle(rss_continue_handle)
        _close_handle(ready_handle)


def _native_probe(manifest_path: Path) -> tuple[bool, list[dict[str, object]]]:
    """Run the native controller and convert every unavailable fact to FAIL."""
    manifest = artifact_manifest(manifest_path, required_distribution="pywinpty")
    distribution = next(
        item
        for item in manifest["resolved_distributions"]
        if str(item.get("name", "")).lower() == "pywinpty"
    )
    artifact = next(
        item
        for item in manifest["artifacts"]
        if str(item.get("name", "")).lower() == "pywinpty"
    )
    version = distribution["version"]
    release = sys.getwindowsversion()
    observations = _default_observations(
        distribution_version=version,
        windows_build=release.build,
    )
    observations.update(
        {
            "artifact_filename": artifact["filename"],
            "artifact_sha256": artifact["sha256"],
            "artifact_size_bytes": artifact["size_bytes"],
            "artifact_verified_during_probe": True,
            "primary_file_name": distribution["primary_file"],
            "primary_file_sha256": distribution["primary_file_sha256"],
            "record_file_name": distribution["record_file"],
            "record_file_sha256": distribution["record_file_sha256"],
        }
    )
    observations.update(_environment_row_facts(manifest_path))
    try:
        _run_native_controller(manifest_path, observations)
    except (OSError, QualificationError, subprocess.SubprocessError) as exc:
        if isinstance(exc, QualificationError):
            category = re.sub(r"[^A-Za-z0-9_.-]", "_", str(exc))[:160]
        elif isinstance(exc, OSError):
            category = (
                f"winerror-{getattr(exc, 'winerror', None)}-"
                f"errno-{getattr(exc, 'errno', None)}"
            )
        else:
            category = type(exc).__name__
        print(
            f"TASK22512_CONTROLLER_FAILURE:{type(exc).__name__}:{category}",
            file=sys.stderr,
        )
    rows = _build_native_rows(observations)
    return all(row["status"] == "PASS" for row in rows), rows


def probe(manifest_path: Path, json_out: Path, *, replace: bool) -> bool:
    """Run native Windows rows or record a complete fail-closed host refusal."""
    started_at = utc_now()
    started = time.monotonic()
    row_id, runtime = _manifest_context(manifest_path)
    if os.name != "nt":
        write_probe_result(
            json_out,
            _unsupported_payload(
                row_id, runtime, started_at, time.monotonic() - started
            ),
            replace=replace,
        )
        return False
    passed, rows = _native_probe(manifest_path)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "row_id": row_id,
        "probe": "pywinpty",
        "status": "PASS" if passed else "FAIL",
        "mandatory": True,
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "command": command_facts(),
        "platform": platform_facts(),
        "measurements": memory_facts(),
        "runtime": runtime,
        "rows": rows,
    }
    write_probe_result(json_out, payload, replace=replace)
    return passed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-manifest", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--native-worker", action="store_true")
    parser.add_argument("--native-crash-app-controller", action="store_true")
    parser.add_argument("--native-crash-worker", action="store_true")
    parser.add_argument("--worker-result", type=Path)
    parser.add_argument("--start-event")
    parser.add_argument("--ready-event")
    parser.add_argument("--rss-fixture-result", type=Path)
    parser.add_argument("--rss-ready-event")
    parser.add_argument("--rss-continue-event")
    parser.add_argument("--crash-app-result", type=Path)
    parser.add_argument("--crash-worker-result", type=Path)
    parser.add_argument("--crash-ready-event")
    parser.add_argument("--crash-retained-event")
    parser.add_argument("--crash-start-event")
    parser.add_argument("--crash-worker-ready-event")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.native_crash_app_controller:
            if not all(
                (
                    args.crash_app_result,
                    args.crash_worker_result,
                    args.crash_ready_event,
                    args.crash_retained_event,
                )
            ):
                raise QualificationError(
                    "native crash app/controller arguments are incomplete"
                )
            return _native_crash_app_controller(
                args.crash_app_result,
                args.crash_worker_result,
                args.crash_ready_event,
                args.crash_retained_event,
            )
        if args.native_crash_worker:
            if not all(
                (
                    args.crash_worker_result,
                    args.crash_start_event,
                    args.crash_worker_ready_event,
                )
            ):
                raise QualificationError("native crash worker arguments are incomplete")
            return _native_crash_worker(
                args.crash_worker_result,
                args.crash_start_event,
                args.crash_worker_ready_event,
            )
        if args.native_worker:
            if not all(
                (
                    args.worker_result,
                    args.start_event,
                    args.ready_event,
                    args.rss_fixture_result,
                    args.rss_ready_event,
                    args.rss_continue_event,
                )
            ):
                raise QualificationError("native worker arguments are incomplete")
            return _native_worker(
                args.worker_result,
                args.start_event,
                args.ready_event,
                args.rss_fixture_result,
                args.rss_ready_event,
                args.rss_continue_event,
            )
        if args.artifact_manifest is None or args.json_out is None:
            raise QualificationError("--artifact-manifest and --json-out are required")
        return (
            0
            if probe(args.artifact_manifest, args.json_out, replace=args.replace)
            else 1
        )
    except (
        ImportError,
        QualificationError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        if isinstance(exc, QualificationError):
            category = re.sub(r"[^A-Za-z0-9_.-]", "_", str(exc))[:160]
        elif isinstance(exc, OSError):
            category = (
                f"winerror-{getattr(exc, 'winerror', None)}-"
                f"errno-{getattr(exc, 'errno', None)}"
            )
        else:
            category = type(exc).__name__
        print(
            f"TASK22512_TOP_LEVEL_FAILURE:{type(exc).__name__}:{category}",
            file=sys.stderr,
        )
        print(f"pywinpty qualification failed: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
