"""Measured, privacy-safe physical-device qualification for speculative voice."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import secrets
import stat
import statistics
import sys

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from Packaging.compute_voice_source_digest import compute_voice_source_digest
from Packaging.physical_voice_runner import (
    PhysicalVoiceRunnerError,
    collect_live_observations,
)
from tldw_chatbook import __version__


_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA_PATH = Path(__file__).with_name("voice_physical_report.schema.json")
_SOURCE_PATH_LIST = _ROOT / "Packaging/speculative_voice_source_paths.txt"
_FIXTURE_ROOT = _ROOT / "Tests/Packaging/fixtures/speculative_voice_physical"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_PLATFORM = re.compile(
    r"(?:macos-(?:arm64|x86_64)|windows-x86_64|linux-(?:x86_64|aarch64))"
)
_FORBIDDEN_REPORT_KEYS = {
    "audio",
    "capture_body",
    "credential",
    "device_id",
    "device_identifier",
    "device_name",
    "pcm",
    "request_body",
    "response",
    "response_text",
    "transcript",
}
_OBSERVATION_KEYS = {
    "aec_delay_estimate_available",
    "aec_delay_estimate_refined",
    "aec_health_path",
    "aec_processor_operational",
    "audio_ring_growth",
    "capture_overflows",
    "channels",
    "control_overflows",
    "correlation_samples",
    "demotion_reason_counts",
    "device_handle_leaks",
    "device_identifier",
    "double_talk_detected",
    "double_talk_trials",
    "erle_samples_db",
    "frame_duration_ms",
    "isolation_warmup_duration_ms",
    "leakage_db_samples",
    "manual_interruption_available",
    "operator_checklist",
    "playback_speech_admission_closed",
    "post_fence_callbacks",
    "process_leaks",
    "reference_overflows",
    "render_only_vad_events",
    "render_overflows",
    "rendered_false_barge_events",
    "rendered_speech_minutes",
    "route_generation_trials",
    "safety_path",
    "sample_rate_hz",
    "saturation_events",
    "silence_false_barge_events",
    "silence_minutes",
    "soak_minutes",
    "stop_latency_samples_ms",
    "transport",
    "unbounded_task_growth",
    "unsuppressed_interruption_observed",
}
_CHECKLIST_KEYS = {
    "device_switch_completed",
    "double_talk_completed",
    "interruption_completed",
    "rendered_speech_completed",
    "silence_completed",
    "soak_completed",
}
_THRESHOLDS = {
    "median_erle_db_min": 20.0,
    "p10_erle_db_min": 10.0,
    "false_barges_per_30_minutes_max": 1.0,
    "double_talk_recall_min": 0.95,
    "isolation_correlation_p95_max": 0.12,
    "isolation_eligible_windows_min": 5,
    "isolation_leakage_p95_db_max": -30.0,
    "isolation_render_only_vad_events_max": 0,
    "isolation_warmup_duration_ms_min": 5_000.0,
    "route_round_trips_required": 3,
    "stop_latency_p95_ms_max": 150.0,
    "soak_minutes_min": 30.0,
}
_MAX_ACOUSTIC_SAMPLES = 3_600
_MAX_INTERRUPTION_SAMPLES = 20
_MAX_ISOLATION_WARMUP_MS = 1_800_000.0
_MAX_ROUTE_ROUND_TRIPS = 3
_MAX_EVENT_COUNT = 1_000_000
_MAX_ROUTE_GENERATION = (1 << 63) - 1
_MAX_TREE_DEPTH = 32
_MAX_TREE_NODES = 50_000
_MAX_REPORT_BYTES = 2 * 1024 * 1024
_NATIVE_DIRECTORY_FD_OPERATIONS = (
    os.name == "posix"
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and all(
        operation in os.supports_dir_fd
        for operation in (os.open, os.stat, os.link, os.unlink)
    )
)
_DEMOTION_REASONS = {
    "ambiguous-correlation",
    "capture-sequence-gap",
    "correlated-render",
    "inaudible-render",
    "invalid-capture-pcm",
    "invalid-render-pcm",
    "missing-render-reference",
    "missing-timing",
    "native-processor-failure",
    "render-reference-failure",
    "render-reference-gap",
    "render-reference-overflow",
    "render-sequence-gap",
    "route-mismatch",
    "saturation",
    "timing-discontinuity",
    "unbounded-timing",
}
_ROUTE_TRIAL_KEYS = {
    "alternate_admission_closed",
    "alternate_generation",
    "return_admission_closed",
    "return_generation",
    "returned_to_target",
    "start_generation",
}
_REPORT_INTEGER_FIELDS = (
    (("schema_version",), 2, 2),
    (("device", "sample_rate_hz"), 8_000, 192_000),
    (("device", "channels"), 1, 2),
    (("device", "frame_duration_ms"), 1, 100),
    (("isolation", "eligible_windows"), 0, _MAX_ACOUSTIC_SAMPLES),
    (("isolation", "required_windows"), 5, 5),
    (("isolation", "render_only_vad_events"), 0, _MAX_EVENT_COUNT),
    (("isolation", "demotions"), 0, _MAX_EVENT_COUNT),
    (("transport_health", "capture_overflows"), 0, _MAX_EVENT_COUNT),
    (("transport_health", "render_overflows"), 0, _MAX_EVENT_COUNT),
    (("transport_health", "reference_overflows"), 0, _MAX_EVENT_COUNT),
    (("transport_health", "control_overflows"), 0, _MAX_EVENT_COUNT),
    (("transport_health", "saturation_events"), 0, _MAX_EVENT_COUNT),
    (("trials", "rendered_speech", "false_barge_events"), 0, _MAX_EVENT_COUNT),
    (("trials", "double_talk", "trials"), 0, _MAX_EVENT_COUNT),
    (("trials", "double_talk", "detected"), 0, _MAX_EVENT_COUNT),
    (("trials", "interruption", "trials"), 1, _MAX_INTERRUPTION_SAMPLES),
    (("trials", "silence", "false_barge_events"), 0, _MAX_EVENT_COUNT),
    (("trials", "device_switch", "trials"), 0, _MAX_ROUTE_ROUND_TRIPS),
    (("trials", "device_switch", "safe_fallbacks"), 0, _MAX_ROUTE_ROUND_TRIPS),
    (("trials", "soak", "process_leaks"), 0, _MAX_EVENT_COUNT),
    (("trials", "soak", "device_handle_leaks"), 0, _MAX_EVENT_COUNT),
    (("trials", "soak", "post_fence_callbacks"), 0, _MAX_EVENT_COUNT),
    (("thresholds", "isolation_eligible_windows_min"), 5, 5),
    (("thresholds", "isolation_render_only_vad_events_max"), 0, 0),
    (("thresholds", "route_round_trips_required"), 3, 3),
)


class PhysicalVoiceReportError(RuntimeError):
    """Raised when physical voice evidence is unsafe or malformed."""


@dataclass(frozen=True, slots=True)
class AutomatedPrerequisiteEvidence:
    """Exact passing automated evidence consumed by one physical report."""

    source_tree_digest: str
    companion_version: str
    upstream_commit: str
    report_hashes: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class _WindowsFileIdentity:
    """Exact Windows identity returned by FileIdInfo."""

    volume_serial_number: int
    file_id: bytes

    def __post_init__(self) -> None:
        if (
            type(self.volume_serial_number) is not int
            or not 0 <= self.volume_serial_number <= (1 << 64) - 1
            or type(self.file_id) is not bytes
            or len(self.file_id) != 16
        ):
            raise PhysicalVoiceReportError("Windows directory identity is invalid")


def _supports_pinned_directory_operations() -> bool:
    """Return whether this runtime supports descriptor-relative publication."""

    return _NATIVE_DIRECTORY_FD_OPERATIONS


def _directory_backend() -> str:
    if _supports_pinned_directory_operations():
        return "posix"
    if os.name == "nt":
        return "windows"
    return "unsupported"


def _runtime_platform_key(
    runtime: str | None = None,
    machine: str | None = None,
) -> str:
    """Return the report platform key for the current supported runtime."""

    runtime_name = (sys.platform if runtime is None else runtime).casefold()
    machine_name = (platform.machine() if machine is None else machine).casefold()
    aliases = {
        ("darwin", "arm64"): "macos-arm64",
        ("darwin", "x86_64"): "macos-x86_64",
        ("win32", "amd64"): "windows-x86_64",
        ("win32", "x86_64"): "windows-x86_64",
        ("linux", "x86_64"): "linux-x86_64",
        ("linux", "aarch64"): "linux-aarch64",
        ("linux", "arm64"): "linux-aarch64",
    }
    try:
        return aliases[(runtime_name, machine_name)]
    except KeyError as exc:
        raise PhysicalVoiceReportError(
            "physical qualification runtime platform is unsupported"
        ) from exc


@dataclass(slots=True)
class _WindowsDirectoryPin:
    """One Windows directory handle plus write-through move operations."""

    path: Path
    identity: _WindowsFileIdentity
    handle: int
    _open_and_identify: Callable[[Path], tuple[int, _WindowsFileIdentity]]
    _close_handle: Callable[[int], int]
    _move_file_ex: Callable[[str, str, int], int]
    _get_last_error: Callable[[], int]

    def revalidate(self) -> None:
        """Compare the path using a second handle and the same Win32 identity."""

        verifier = 0
        try:
            verifier, identity = self._open_and_identify(self.path)
            if identity != self.identity:
                raise PhysicalVoiceReportError(
                    "qualification output parent identity changed"
                )
        except BaseException as primary:
            if verifier:
                handle = verifier
                verifier = 0
                try:
                    if not self._close_handle(handle):
                        error = self._get_last_error()
                        raise OSError(
                            error,
                            "Windows verifier directory handle close failed",
                        )
                except BaseException as cleanup_error:
                    primary.add_note(
                        f"Windows verifier handle cleanup failed: {cleanup_error}"
                    )
            raise
        handle = verifier
        verifier = 0
        if handle and not self._close_handle(handle):
            error = self._get_last_error()
            raise OSError(error, "Windows verifier directory handle close failed")

    def close(self) -> None:
        handle = self.handle
        self.handle = 0
        if handle and not self._close_handle(handle):
            error = self._get_last_error()
            raise OSError(error, "Windows directory handle close failed")

    def move_write_through(
        self,
        source: Path,
        destination: Path,
        *,
        replace: bool,
    ) -> None:
        movefile_replace_existing = 0x00000001
        movefile_write_through = 0x00000008
        flags = movefile_write_through
        if replace:
            flags |= movefile_replace_existing
        if not self._move_file_ex(os.fspath(source), os.fspath(destination), flags):
            error = self._get_last_error()
            raise OSError(error, "Windows write-through publication failed")


def _open_windows_directory(path: Path) -> _WindowsDirectoryPin:
    """Open and validate one non-reparse Windows directory without delete sharing."""

    if os.name != "nt":
        raise PhysicalVoiceReportError("Windows directory pinning is unavailable")
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError as exc:  # pragma: no cover - Windows stdlib invariant
        raise PhysicalVoiceReportError(
            "Windows directory pinning is unavailable"
        ) from exc

    class _FileId128(ctypes.Structure):
        _fields_ = (("identifier", wintypes.BYTE * 16),)

    class _FileIdInfo(ctypes.Structure):
        _fields_ = (
            ("volume_serial_number", ctypes.c_ulonglong),
            ("file_id", _FileId128),
        )

    class _FileAttributeTagInfo(ctypes.Structure):
        _fields_ = (
            ("file_attributes", wintypes.DWORD),
            ("reparse_tag", wintypes.DWORD),
        )

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    get_information = kernel32.GetFileInformationByHandleEx
    get_information.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    get_information.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    move_file_ex = kernel32.MoveFileExW
    move_file_ex.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)
    move_file_ex.restype = wintypes.BOOL

    file_read_attributes = 0x00000080
    file_share_read = 0x00000001
    file_share_write = 0x00000002
    open_existing = 3
    file_flag_open_reparse_point = 0x00200000
    file_flag_backup_semantics = 0x02000000
    invalid_handle = ctypes.c_void_p(-1).value

    def open_and_identify(
        candidate: Path,
    ) -> tuple[int, _WindowsFileIdentity]:
        handle = create_file(
            os.fspath(candidate),
            file_read_attributes,
            file_share_read | file_share_write,
            None,
            open_existing,
            file_flag_open_reparse_point | file_flag_backup_semantics,
            None,
        )
        if handle in (None, invalid_handle):
            error = ctypes.get_last_error()
            raise OSError(error, "Windows directory pin could not be opened")
        descriptor = int(handle)
        try:
            file_attribute_tag_info = 9
            attribute_information = _FileAttributeTagInfo()
            if not get_information(
                handle,
                file_attribute_tag_info,
                ctypes.byref(attribute_information),
                ctypes.sizeof(attribute_information),
            ):
                error = ctypes.get_last_error()
                raise OSError(error, "Windows directory pin could not be inspected")
            file_attribute_directory = 0x00000010
            file_attribute_reparse_point = 0x00000400
            if not attribute_information.file_attributes & file_attribute_directory:
                raise PhysicalVoiceReportError(
                    "qualification output parent must be a regular directory"
                )
            if attribute_information.file_attributes & file_attribute_reparse_point:
                raise PhysicalVoiceReportError(
                    "qualification output parent cannot be a reparse point"
                )
            file_id_info = 18
            identity_information = _FileIdInfo()
            if not get_information(
                handle,
                file_id_info,
                ctypes.byref(identity_information),
                ctypes.sizeof(identity_information),
            ):
                error = ctypes.get_last_error()
                raise OSError(
                    error,
                    "Windows directory identity could not be inspected",
                )
            identity = _WindowsFileIdentity(
                int(identity_information.volume_serial_number),
                bytes(identity_information.file_id.identifier),
            )
        except BaseException as primary:
            released_handle = descriptor
            descriptor = 0
            try:
                if not close_handle(released_handle):
                    error = ctypes.get_last_error()
                    raise OSError(
                        error,
                        "Windows directory handle cleanup failed",
                    )
            except BaseException as cleanup_error:
                primary.add_note(
                    f"Windows directory handle cleanup failed: {cleanup_error}"
                )
            raise
        return descriptor, identity

    handle, identity = open_and_identify(path)
    return _WindowsDirectoryPin(
        path=path,
        identity=identity,
        handle=handle,
        _open_and_identify=open_and_identify,
        _close_handle=close_handle,
        _move_file_ex=move_file_ex,
        _get_last_error=ctypes.get_last_error,
    )


@dataclass(slots=True)
class _OutputDirectory:
    """One output parent pinned for the complete qualification transaction."""

    path: Path
    identity: _FileIdentity | None
    descriptor: int | None
    windows_pin: _WindowsDirectoryPin | None

    @classmethod
    def acquire(cls, path: Path) -> _OutputDirectory:
        parent = Path(path)
        descriptor: int | None = None
        windows_pin: _WindowsDirectoryPin | None = None
        try:
            parent.mkdir(parents=True, exist_ok=True)
            path_metadata = parent.lstat()
            if not stat.S_ISDIR(path_metadata.st_mode):
                raise PhysicalVoiceReportError(
                    "qualification output parent must be a regular directory"
                )
            identity = _identity(path_metadata)
            backend = _directory_backend()
            if backend == "posix":
                flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                descriptor = os.open(parent, flags)
                descriptor_metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISDIR(descriptor_metadata.st_mode)
                    or _identity(descriptor_metadata) != identity
                ):
                    raise PhysicalVoiceReportError(
                        "qualification output parent identity changed"
                    )
            elif backend == "windows":
                windows_pin = _open_windows_directory(parent)
                windows_pin.revalidate()
                identity = None
            else:
                raise PhysicalVoiceReportError(
                    "qualification output directory pinning is unsupported"
                )
            return cls(parent, identity, descriptor, windows_pin)
        except BaseException as primary:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except BaseException as cleanup_error:
                    primary.add_note(
                        f"output directory descriptor cleanup failed: {cleanup_error}"
                    )
            if windows_pin is not None:
                try:
                    windows_pin.close()
                except BaseException as cleanup_error:
                    primary.add_note(
                        f"Windows directory handle cleanup failed: {cleanup_error}"
                    )
            if isinstance(primary, PhysicalVoiceReportError):
                raise
            if not isinstance(primary, (OSError, ValueError)):
                raise
            raise PhysicalVoiceReportError(
                "qualification output parent is unavailable"
            ) from primary

    def __enter__(self) -> _OutputDirectory:
        return self

    def __exit__(
        self,
        _exc_type: object,
        primary: BaseException | None,
        _traceback: object,
    ) -> bool:
        try:
            self.close()
        except BaseException as cleanup_error:
            if primary is None:
                raise
            primary.add_note(f"output directory close failed: {cleanup_error}")
        return False

    def close(self) -> None:
        failure: OSError | None = None
        if self.descriptor is not None:
            descriptor = self.descriptor
            self.descriptor = None
            try:
                os.close(descriptor)
            except OSError as exc:
                failure = exc
        if self.windows_pin is not None:
            pin = self.windows_pin
            self.windows_pin = None
            try:
                pin.close()
            except OSError as exc:
                failure = failure or exc
        if failure is not None:
            raise PhysicalVoiceReportError(
                "qualification output parent could not be closed"
            ) from failure

    def revalidate(self) -> None:
        if self.windows_pin is not None:
            try:
                self.windows_pin.revalidate()
            except PhysicalVoiceReportError:
                raise
            except (OSError, ValueError) as exc:
                raise PhysicalVoiceReportError(
                    "qualification output parent identity changed"
                ) from exc
            return
        try:
            metadata = self.path.lstat()
        except (OSError, ValueError) as exc:
            raise PhysicalVoiceReportError(
                "qualification output parent identity changed"
            ) from exc
        if (
            self.identity is None
            or not stat.S_ISDIR(metadata.st_mode)
            or _identity(metadata) != self.identity
        ):
            raise PhysicalVoiceReportError(
                "qualification output parent identity changed"
            )

    def _annotate_revalidation_failure(self, primary: BaseException) -> None:
        try:
            self.revalidate()
        except PhysicalVoiceReportError as cleanup_error:
            primary.add_note(
                f"output directory identity revalidation failed: {cleanup_error}"
            )

    def stat(self, name: str) -> os.stat_result:
        if self.descriptor is not None:
            return os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
        self.revalidate()
        try:
            metadata = (self.path / name).lstat()
        except BaseException as primary:
            self._annotate_revalidation_failure(primary)
            raise
        self.revalidate()
        return metadata

    def open_exclusive(self, name: str, flags: int, mode: int) -> int:
        if self.descriptor is not None:
            return os.open(name, flags, mode, dir_fd=self.descriptor)
        self.revalidate()
        descriptor = os.open(self.path / name, flags, mode)
        try:
            self.revalidate()
        except BaseException as primary:
            try:
                os.close(descriptor)
            except BaseException as cleanup_error:
                primary.add_note(f"staging descriptor cleanup failed: {cleanup_error}")
            raise
        return descriptor

    def link(self, source: str, destination: str) -> None:
        if self.descriptor is not None:
            os.link(
                source,
                destination,
                src_dir_fd=self.descriptor,
                dst_dir_fd=self.descriptor,
                follow_symlinks=False,
            )
            return
        self.revalidate()
        try:
            os.link(
                self.path / source,
                self.path / destination,
                follow_symlinks=False,
            )
        except BaseException as primary:
            self._annotate_revalidation_failure(primary)
            raise
        self.revalidate()

    def unlink(self, name: str) -> None:
        if self.descriptor is not None:
            os.unlink(name, dir_fd=self.descriptor)
            return
        self.revalidate()
        try:
            os.unlink(self.path / name)
        except BaseException as primary:
            self._annotate_revalidation_failure(primary)
            raise
        self.revalidate()

    def replace(self, source: str, destination: str) -> None:
        self.revalidate()
        try:
            if self.descriptor is not None:
                os.replace(
                    source,
                    destination,
                    src_dir_fd=self.descriptor,
                    dst_dir_fd=self.descriptor,
                )
            elif self.windows_pin is not None:
                self.windows_pin.move_write_through(
                    self.path / source,
                    self.path / destination,
                    replace=True,
                )
            else:  # pragma: no cover - construction rejects unsupported backends
                raise PhysicalVoiceReportError(
                    "qualification output replacement is unavailable"
                )
        except BaseException as primary:
            self._annotate_revalidation_failure(primary)
            raise
        self.revalidate()

    def sync(self) -> None:
        if self.descriptor is not None:
            os.fsync(self.descriptor)
            return
        raise PhysicalVoiceReportError(
            "qualification output directory durability barrier is unavailable"
        )

    def move_write_through(
        self,
        source: str,
        destination: str,
        *,
        replace: bool,
    ) -> None:
        if self.windows_pin is None:
            raise PhysicalVoiceReportError(
                "Windows write-through publication is unavailable"
            )
        self.revalidate()
        try:
            self.windows_pin.move_write_through(
                self.path / source,
                self.path / destination,
                replace=replace,
            )
        except BaseException as primary:
            self._annotate_revalidation_failure(primary)
            raise
        self.revalidate()


def _canonical_bytes(value: object) -> bytes:
    try:
        _walk_bounded_tree(value)
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (
        TypeError,
        ValueError,
        OverflowError,
        UnicodeEncodeError,
        RecursionError,
        PhysicalVoiceReportError,
    ) as exc:
        raise PhysicalVoiceReportError(
            "canonical qualification serialization failed"
        ) from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_json_report(path: Path) -> tuple[dict[str, object], bytes]:
    report_path = Path(path)
    if not report_path.is_file() or report_path.is_symlink():
        raise PhysicalVoiceReportError("qualification report is missing or unsafe")
    try:
        raw = report_path.read_bytes()
        if len(raw) > _MAX_REPORT_BYTES:
            raise PhysicalVoiceReportError("qualification report is too large")
        value = json.loads(raw)
    except PhysicalVoiceReportError:
        raise
    except (OSError, UnicodeError, ValueError, OverflowError, RecursionError) as exc:
        raise PhysicalVoiceReportError("qualification report is invalid JSON") from exc
    if not isinstance(value, dict):
        raise PhysicalVoiceReportError("qualification report must be an object")
    return value, raw


def _walk_bounded_tree(
    value: object,
    *,
    path: str = "report",
) -> Sequence[tuple[object, str]]:
    """Return a bounded iterative walk of an untrusted JSON-like tree."""

    stack: list[tuple[object, str, int]] = [(value, path, 0)]
    walked: list[tuple[object, str]] = []
    while stack:
        item, item_path, depth = stack.pop()
        if depth > _MAX_TREE_DEPTH:
            raise PhysicalVoiceReportError(
                "qualification report exceeds the maximum tree depth"
            )
        walked.append((item, item_path))
        if len(walked) > _MAX_TREE_NODES:
            raise PhysicalVoiceReportError(
                "qualification report nodes exceed the complexity limit"
            )
        if isinstance(item, Mapping):
            if len(walked) + len(stack) + len(item) > _MAX_TREE_NODES:
                raise PhysicalVoiceReportError(
                    "qualification report nodes exceed the complexity limit"
                )
            children: list[tuple[object, str, int]] = []
            for key, child in item.items():
                if not isinstance(key, str):
                    raise PhysicalVoiceReportError(
                        "qualification report keys must be strings"
                    )
                children.append((child, f"{item_path}.{key}", depth + 1))
            stack.extend(reversed(children))
        elif isinstance(item, (list, tuple)):
            if len(walked) + len(stack) + len(item) > _MAX_TREE_NODES:
                raise PhysicalVoiceReportError(
                    "qualification report nodes exceed the complexity limit"
                )
            stack.extend(
                (child, f"{item_path}[{index}]", depth + 1)
                for index, child in reversed(list(enumerate(item)))
            )
    return walked


def _reject_forbidden_report_fields(value: object, *, path: str = "report") -> None:
    for item, item_path in _walk_bounded_tree(value, path=path):
        if not isinstance(item, Mapping):
            continue
        for key in item:
            if key.casefold() in _FORBIDDEN_REPORT_KEYS:
                raise PhysicalVoiceReportError(
                    f"privacy-forbidden qualification field: {item_path}.{key}"
                )


def _reject_non_finite_numbers(value: object, *, path: str = "report") -> None:
    for item, item_path in _walk_bounded_tree(value, path=path):
        if not isinstance(item, float):
            continue
        if not math.isfinite(item):
            raise PhysicalVoiceReportError(
                f"qualification report number must be finite: {item_path}"
            )
        if item == 0.0 and math.copysign(1.0, item) < 0.0:
            raise PhysicalVoiceReportError(
                f"qualification report number must not be negative zero: {item_path}"
            )


def _require_passing_report(
    report: Mapping[str, object],
    *,
    scenario: str,
    source_tree_digest: str,
) -> None:
    _reject_non_finite_numbers(report)
    _reject_forbidden_report_fields(report)
    if report.get("scenario") != scenario:
        raise PhysicalVoiceReportError("automated prerequisite scenario is invalid")
    if report.get("source_tree_digest") != source_tree_digest:
        raise PhysicalVoiceReportError(
            "automated prerequisite source-tree digest does not match"
        )
    if report.get("passed") is not True:
        raise PhysicalVoiceReportError("automated prerequisite did not pass")


def _find_sibling_report(
    directory: Path,
    *,
    scenario: str,
    source_tree_digest: str,
    platform_key: str | None,
) -> tuple[dict[str, object], bytes]:
    if platform_key is not None:
        exact = directory / f"{platform_key}-{scenario}.json"
        if exact.is_symlink() or not exact.is_file():
            raise PhysicalVoiceReportError(
                f"exact {platform_key} {scenario} prerequisite report is required"
            )
        report, raw = _read_json_report(exact)
        _require_passing_report(
            report,
            scenario=scenario,
            source_tree_digest=source_tree_digest,
        )
        soak = report.get("soak")
        if not isinstance(soak, Mapping) or soak.get("passed") is not True:
            raise PhysicalVoiceReportError("soak prerequisite did not pass")
        return report, raw
    matches: list[tuple[dict[str, object], bytes]] = []
    for candidate in sorted(directory.glob("*.json")):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        try:
            report, raw = _read_json_report(candidate)
        except PhysicalVoiceReportError:
            continue
        if report.get("scenario") == scenario:
            matches.append((report, raw))
    if len(matches) != 1:
        raise PhysicalVoiceReportError(
            "exactly one matching soak prerequisite report is required"
        )
    report, raw = matches[0]
    _require_passing_report(
        report,
        scenario=scenario,
        source_tree_digest=source_tree_digest,
    )
    soak = report.get("soak")
    if not isinstance(soak, Mapping) or soak.get("passed") is not True:
        raise PhysicalVoiceReportError("soak prerequisite did not pass")
    return report, raw


def load_automated_prerequisites(
    automated_report_path: Path,
    *,
    expected_source_tree_digest: str,
    platform_key: str | None = None,
) -> AutomatedPrerequisiteEvidence:
    """Load one automated report plus its exact peer soak reports."""

    if not _SHA256.fullmatch(expected_source_tree_digest):
        raise PhysicalVoiceReportError("source-tree digest must be lowercase SHA-256")
    automated, automated_raw = _read_json_report(automated_report_path)
    _require_passing_report(
        automated,
        scenario="automated",
        source_tree_digest=expected_source_tree_digest,
    )
    companion = automated.get("companion")
    corpus = automated.get("corpus")
    latency = automated.get("latency_distributions")
    if (
        not isinstance(companion, Mapping)
        or companion.get("passed") is not True
        or not isinstance(corpus, Mapping)
        or corpus.get("passed") is not True
        or not isinstance(latency, Mapping)
        or latency.get("passed") is not True
    ):
        raise PhysicalVoiceReportError("automated prerequisite sections did not pass")
    version = companion.get("version")
    upstream_commit = companion.get("upstream_commit")
    if (
        not isinstance(version, str)
        or not version
        or not isinstance(upstream_commit, str)
        or not _COMMIT.fullmatch(upstream_commit)
    ):
        raise PhysicalVoiceReportError("automated companion identity is invalid")
    directory = Path(automated_report_path).parent
    _, duplex_raw = _find_sibling_report(
        directory,
        scenario="duplex-soak",
        source_tree_digest=expected_source_tree_digest,
        platform_key=platform_key,
    )
    _, cancellation_raw = _find_sibling_report(
        directory,
        scenario="cancellation-soak",
        source_tree_digest=expected_source_tree_digest,
        platform_key=platform_key,
    )
    hashes = {
        "automated_report_sha256": hashlib.sha256(automated_raw).hexdigest(),
        "corpus_section_sha256": _canonical_sha256(corpus),
        "latency_section_sha256": _canonical_sha256(latency),
        "duplex_soak_report_sha256": hashlib.sha256(duplex_raw).hexdigest(),
        "cancellation_soak_report_sha256": hashlib.sha256(cancellation_raw).hexdigest(),
    }
    return AutomatedPrerequisiteEvidence(
        source_tree_digest=expected_source_tree_digest,
        companion_version=version,
        upstream_commit=upstream_commit,
        report_hashes=hashes,
    )


def load_fixture_observations(path: Path) -> dict[str, object]:
    """Load checked-in synthetic observations used only to test the harness."""

    observations, _ = _read_json_report(path)
    return observations


def _number(
    observations: Mapping[str, object],
    key: str,
    *,
    minimum: float | None = None,
) -> float:
    value = observations.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhysicalVoiceReportError(f"observation {key} must be numeric")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise PhysicalVoiceReportError(f"observation {key} is out of bounds") from exc
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise PhysicalVoiceReportError(f"observation {key} is out of bounds")
    return number


def _integer(
    observations: Mapping[str, object],
    key: str,
    *,
    minimum: int = 0,
    maximum: int = _MAX_EVENT_COUNT,
) -> int:
    value = observations.get(key)
    if type(value) is not int or not minimum <= value <= maximum:
        raise PhysicalVoiceReportError(
            f"observation {key} must be an integer within bounds"
        )
    return value


def _boolean(observations: Mapping[str, object], key: str) -> bool:
    value = observations.get(key)
    if type(value) is not bool:
        raise PhysicalVoiceReportError(f"observation {key} must be boolean")
    return value


def _choice(
    observations: Mapping[str, object],
    key: str,
    choices: set[str],
) -> str:
    value = observations.get(key)
    if not isinstance(value, str) or value not in choices:
        raise PhysicalVoiceReportError(f"observation {key} is invalid")
    return value


def _round_canonical(number: float) -> float:
    rounded = round(number, 6)
    return 0.0 if rounded == 0.0 else rounded


def _sample_array(
    observations: Mapping[str, object],
    key: str,
    *,
    maximum_items: int,
    minimum: float,
    maximum: float,
    require_samples: bool = False,
) -> list[float]:
    """Return one bounded, six-decimal canonical observation array."""

    value = observations.get(key)
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) > maximum_items
        or (require_samples and not value)
    ):
        raise PhysicalVoiceReportError(f"observation {key} samples are out of bounds")
    samples: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise PhysicalVoiceReportError(f"observation {key} samples must be numeric")
        try:
            number = float(item)
        except (OverflowError, ValueError) as exc:
            raise PhysicalVoiceReportError(
                f"observation {key} samples are out of bounds"
            ) from exc
        if not math.isfinite(number) or not minimum <= number <= maximum:
            raise PhysicalVoiceReportError(
                f"observation {key} samples are out of bounds"
            )
        samples.append(_round_canonical(number))
    return samples


def _canonical_number(
    observations: Mapping[str, object],
    key: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    number = _number(observations, key, minimum=minimum)
    if number > maximum:
        raise PhysicalVoiceReportError(f"observation {key} is out of bounds")
    return _round_canonical(number)


def _demotion_records(
    observations: Mapping[str, object],
) -> list[dict[str, object]]:
    value = observations.get("demotion_reason_counts")
    if not isinstance(value, Mapping) or len(value) > len(_DEMOTION_REASONS):
        raise PhysicalVoiceReportError("demotion reason counts have an invalid shape")
    if any(not isinstance(reason, str) for reason in value):
        raise PhysicalVoiceReportError("demotion reason counts are invalid")
    records: list[dict[str, object]] = []
    total = 0
    for reason in sorted(value):
        count = value[reason]
        if (
            reason not in _DEMOTION_REASONS
            or type(count) is not int
            or not 1 <= count <= _MAX_EVENT_COUNT
        ):
            raise PhysicalVoiceReportError("demotion reason counts are invalid")
        total += count
        if total > _MAX_EVENT_COUNT:
            raise PhysicalVoiceReportError("demotion reason counts exceed bounds")
        records.append({"count": count, "reason": reason})
    return records


def _route_generation_trials(
    observations: Mapping[str, object],
) -> list[dict[str, object]]:
    value = observations.get("route_generation_trials")
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) > _MAX_ROUTE_ROUND_TRIPS
    ):
        raise PhysicalVoiceReportError("route generation trials have an invalid shape")
    trials: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != _ROUTE_TRIAL_KEYS:
            raise PhysicalVoiceReportError(
                "route generation trial has an invalid shape"
            )
        for key in (
            "start_generation",
            "alternate_generation",
            "return_generation",
        ):
            if (
                type(item[key]) is not int
                or not 0 <= item[key] <= _MAX_ROUTE_GENERATION
            ):
                raise PhysicalVoiceReportError(
                    "route generation trial generations are invalid"
                )
        for key in (
            "alternate_admission_closed",
            "return_admission_closed",
            "returned_to_target",
        ):
            if type(item[key]) is not bool:
                raise PhysicalVoiceReportError(
                    "route generation trial flags are invalid"
                )
        trials.append(dict(item))
    return trials


def _safe_route_trial_count(trials: Sequence[Mapping[str, object]]) -> int:
    safe = 0
    previous_return: object = None
    for trial in trials:
        start = trial["start_generation"]
        alternate = trial["alternate_generation"]
        returned = trial["return_generation"]
        chained = previous_return is None or start == previous_return
        if all(
            (
                chained,
                start < alternate,
                alternate < returned,
                trial["alternate_admission_closed"] is True,
                trial["return_admission_closed"] is True,
                trial["returned_to_target"] is True,
            )
        ):
            safe += 1
        previous_return = returned
    return safe


def _report_path_value(
    report: Mapping[str, object],
    path: tuple[str, ...],
) -> object:
    value: object = report
    for part in path:
        assert isinstance(value, Mapping)
        value = value[part]
    return value


def _validate_integer_type_image(report: Mapping[str, object]) -> None:
    """Enforce the canonical integer image that JSON Schema cannot express."""

    for path, minimum, maximum in _REPORT_INTEGER_FIELDS:
        value = _report_path_value(report, path)
        if type(value) is not int:
            raise PhysicalVoiceReportError(
                f"physical report integer type is invalid: {'.'.join(path)}"
            )
        if not minimum <= value <= maximum:
            raise PhysicalVoiceReportError(
                f"physical report integer bounds are invalid: {'.'.join(path)}"
            )

    isolation = report["isolation"]
    trials = report["trials"]
    assert isinstance(isolation, Mapping)
    assert isinstance(trials, Mapping)
    demotion_records = isolation["demotion_reasons"]
    device_switch = trials["device_switch"]
    assert isinstance(demotion_records, list)
    assert isinstance(device_switch, Mapping)
    route_trials = device_switch["route_generation_trials"]
    assert isinstance(route_trials, list)
    for record in demotion_records:
        assert isinstance(record, Mapping)
        count = record["count"]
        if type(count) is not int:
            raise PhysicalVoiceReportError(
                "physical report demotion count integer type is invalid"
            )
        if not 1 <= count <= _MAX_EVENT_COUNT:
            raise PhysicalVoiceReportError(
                "physical report demotion count integer bounds are invalid"
            )
    for trial in route_trials:
        assert isinstance(trial, Mapping)
        for key in (
            "start_generation",
            "alternate_generation",
            "return_generation",
        ):
            generation = trial[key]
            if type(generation) is not int:
                raise PhysicalVoiceReportError(
                    "physical report route generation integer type is invalid"
                )
            if not 0 <= generation <= _MAX_ROUTE_GENERATION:
                raise PhysicalVoiceReportError(
                    "physical report route generation integer bounds are invalid"
                )


def _percentile(samples: Sequence[float], percentile: float) -> float | None:
    if not samples:
        return None
    ordered = sorted(samples)
    return _round_canonical(ordered[max(0, math.ceil(len(ordered) * percentile) - 1)])


def _median(samples: Sequence[float]) -> float | None:
    if not samples:
        return None
    return _round_canonical(statistics.median(samples))


def _physical_safety_passes(report: Mapping[str, object]) -> bool:
    aec = report["aec"]
    isolation = report["isolation"]
    transport_health = report["transport_health"]
    trials = report["trials"]
    degradation = report["degradation"]
    checklist = report["operator_checklist"]
    assert isinstance(aec, Mapping)
    assert isinstance(isolation, Mapping)
    assert isinstance(transport_health, Mapping)
    assert isinstance(trials, Mapping)
    assert isinstance(degradation, Mapping)
    assert isinstance(checklist, Mapping)
    rendered = trials["rendered_speech"]
    double_talk = trials["double_talk"]
    interruption = trials["interruption"]
    silence = trials["silence"]
    device_switch = trials["device_switch"]
    soak = trials["soak"]
    assert all(
        isinstance(section, Mapping)
        for section in (
            rendered,
            double_talk,
            interruption,
            silence,
            device_switch,
            soak,
        )
    )
    common_safe = all(
        (
            all(value is True for value in checklist.values()),
            rendered["false_barges_per_30_minutes"] <= 1.0,
            interruption["manual_available"] is True,
            silence["false_barge_events"] == 0,
            device_switch["trials"] == 3,
            device_switch["safe_fallbacks"] == 3,
            soak["minutes"] >= 30.0,
            soak["unbounded_task_growth"] is False,
            soak["process_leaks"] == 0,
            soak["audio_ring_growth"] is False,
            soak["device_handle_leaks"] == 0,
            soak["post_fence_callbacks"] == 0,
            all(value == 0 for value in transport_health.values()),
            isolation["demotions"] == 0,
            degradation["manual_interruption_available"] is True,
            degradation["unsuppressed_interruption_observed"] is False,
        )
    )
    if not common_safe:
        return False
    safety_path = report["safety_path"]
    if safety_path == "half-duplex":
        return degradation["playback_speech_admission_closed"] is True
    full_duplex_safe = all(
        (
            aec["processor_operational"] is True,
            double_talk["trials"] > 0,
            double_talk["recall"] >= 0.95,
            interruption["p95_stop_latency_ms"] <= 150.0,
            degradation["playback_speech_admission_closed"] is False,
        )
    )
    if safety_path == "aec":
        return full_duplex_safe and all(
            (
                aec["health_path"] == "healthy",
                aec["delay_estimate_available"] is True,
                aec["delay_estimate_refined"] is True,
                aec["median_erle_db"] is not None,
                aec["median_erle_db"] >= 20.0,
                aec["p10_erle_db"] is not None,
                aec["p10_erle_db"] >= 10.0,
            )
        )
    if safety_path == "acoustic-isolation":
        return full_duplex_safe and all(
            (
                aec["health_path"] in {"warming", "degraded"},
                isolation["eligible_windows"] >= isolation["required_windows"],
                isolation["required_windows"] == 5,
                isolation["warmup_duration_ms"] >= 5_000.0,
                isolation["p95_correlation"] is not None,
                isolation["p95_correlation"] <= 0.12,
                isolation["p95_leakage_db"] is not None,
                isolation["p95_leakage_db"] <= -30.0,
                isolation["render_only_vad_events"] == 0,
            )
        )
    return False


def build_physical_report(
    *,
    evidence_kind: str,
    source_tree_digest: str,
    platform_key: str,
    device_class: str,
    app_version: str,
    prerequisites: AutomatedPrerequisiteEvidence,
    observations: Mapping[str, object],
    salt: bytes | None = None,
) -> dict[str, object]:
    """Build one report while retaining the raw device identity only in memory."""

    if set(observations) != _OBSERVATION_KEYS:
        raise PhysicalVoiceReportError("physical observations have an unknown shape")
    if evidence_kind not in {"physical", "synthetic-fixture"}:
        raise PhysicalVoiceReportError("physical evidence kind is invalid")
    if (
        not _SHA256.fullmatch(source_tree_digest)
        or source_tree_digest != prerequisites.source_tree_digest
    ):
        raise PhysicalVoiceReportError(
            "source-tree digest does not match prerequisites"
        )
    if not _PLATFORM.fullmatch(platform_key):
        raise PhysicalVoiceReportError("physical platform key is invalid")
    if device_class not in {"builtin", "usb", "bluetooth"}:
        raise PhysicalVoiceReportError("physical device class is invalid")
    raw_identifier = observations.get("device_identifier")
    if not isinstance(raw_identifier, str) or not raw_identifier.strip():
        raise PhysicalVoiceReportError("device identifier is invalid")
    try:
        raw_identifier_bytes = raw_identifier.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise PhysicalVoiceReportError("device identifier is invalid") from exc
    if len(raw_identifier_bytes) > 512:
        raise PhysicalVoiceReportError("device identifier is invalid")
    identity_salt = secrets.token_bytes(16) if salt is None else bytes(salt)
    if len(identity_salt) != 16:
        raise PhysicalVoiceReportError("device identity salt must be 16 bytes")
    transport = _choice(
        observations,
        "transport",
        {"builtin", "usb", "wired", "bluetooth"},
    )
    allowed_transport = {
        "builtin": {"builtin"},
        "usb": {"usb", "wired"},
        "bluetooth": {"bluetooth"},
    }
    if transport not in allowed_transport[device_class]:
        raise PhysicalVoiceReportError("transport does not match the device class")

    rendered_minutes = _number(
        observations,
        "rendered_speech_minutes",
        minimum=0.000001,
    )
    false_barges = _integer(observations, "rendered_false_barge_events")
    double_talk_trials = _integer(observations, "double_talk_trials")
    double_talk_detected = _integer(observations, "double_talk_detected")
    if double_talk_detected > double_talk_trials:
        raise PhysicalVoiceReportError("double-talk detections exceed trial count")
    checklist = observations.get("operator_checklist")
    if not isinstance(checklist, Mapping) or set(checklist) != _CHECKLIST_KEYS:
        raise PhysicalVoiceReportError("operator checklist has an unknown shape")
    if any(type(value) is not bool for value in checklist.values()):
        raise PhysicalVoiceReportError("operator checklist values must be boolean")
    safety_path = _choice(
        observations,
        "safety_path",
        {"aec", "acoustic-isolation", "half-duplex"},
    )
    erle_samples = _sample_array(
        observations,
        "erle_samples_db",
        maximum_items=_MAX_ACOUSTIC_SAMPLES,
        minimum=-100.0,
        maximum=100.0,
        require_samples=safety_path == "aec",
    )
    correlation_samples = _sample_array(
        observations,
        "correlation_samples",
        maximum_items=_MAX_ACOUSTIC_SAMPLES,
        minimum=0.0,
        maximum=1.0,
        require_samples=safety_path == "acoustic-isolation",
    )
    leakage_samples = _sample_array(
        observations,
        "leakage_db_samples",
        maximum_items=_MAX_ACOUSTIC_SAMPLES,
        minimum=-200.0,
        maximum=100.0,
        require_samples=safety_path == "acoustic-isolation",
    )
    if len(correlation_samples) != len(leakage_samples):
        raise PhysicalVoiceReportError(
            "isolation correlation and leakage samples must be paired"
        )
    stop_latency_samples = _sample_array(
        observations,
        "stop_latency_samples_ms",
        maximum_items=_MAX_INTERRUPTION_SAMPLES,
        minimum=0.0,
        maximum=10_000.0,
        require_samples=True,
    )
    demotion_records = _demotion_records(observations)
    route_trials = _route_generation_trials(observations)

    report: dict[str, object] = {
        "schema_version": 2,
        "evidence_kind": evidence_kind,
        "source_tree_digest": source_tree_digest,
        "platform": platform_key,
        "device_class": device_class,
        "app_version": app_version,
        "safety_path": safety_path,
        "companion": {
            "name": "tldw-voice-aec",
            "version": prerequisites.companion_version,
            "upstream_commit": prerequisites.upstream_commit,
        },
        "automated_prerequisites": {
            "source_tree_digest": prerequisites.source_tree_digest,
            **dict(prerequisites.report_hashes),
        },
        "device": {
            "identity_salt": identity_salt.hex(),
            "identity_sha256": hashlib.sha256(
                identity_salt + raw_identifier_bytes
            ).hexdigest(),
            "transport": transport,
            "sample_rate_hz": _integer(
                observations,
                "sample_rate_hz",
                minimum=8_000,
                maximum=192_000,
            ),
            "channels": _integer(
                observations,
                "channels",
                minimum=1,
                maximum=2,
            ),
            "frame_duration_ms": _integer(
                observations,
                "frame_duration_ms",
                minimum=1,
                maximum=100,
            ),
        },
        "aec": {
            "delay_estimate_available": _boolean(
                observations,
                "aec_delay_estimate_available",
            ),
            "delay_estimate_refined": _boolean(
                observations,
                "aec_delay_estimate_refined",
            ),
            "health_path": _choice(
                observations,
                "aec_health_path",
                {"healthy", "warming", "degraded"},
            ),
            "processor_operational": _boolean(
                observations,
                "aec_processor_operational",
            ),
            "erle_samples_db": erle_samples,
            "median_erle_db": _median(erle_samples),
            "p10_erle_db": _percentile(erle_samples, 0.10),
        },
        "isolation": {
            "eligible_windows": len(correlation_samples),
            "required_windows": 5,
            "warmup_duration_ms": _canonical_number(
                observations,
                "isolation_warmup_duration_ms",
                minimum=0.0,
                maximum=_MAX_ISOLATION_WARMUP_MS,
            ),
            "correlation_samples": correlation_samples,
            "p95_correlation": _percentile(correlation_samples, 0.95),
            "leakage_db_samples": leakage_samples,
            "p95_leakage_db": _percentile(leakage_samples, 0.95),
            "render_only_vad_events": _integer(
                observations,
                "render_only_vad_events",
            ),
            "demotions": sum(record["count"] for record in demotion_records),
            "demotion_reasons": demotion_records,
        },
        "transport_health": {
            "capture_overflows": _integer(observations, "capture_overflows"),
            "render_overflows": _integer(observations, "render_overflows"),
            "reference_overflows": _integer(observations, "reference_overflows"),
            "control_overflows": _integer(observations, "control_overflows"),
            "saturation_events": _integer(observations, "saturation_events"),
        },
        "trials": {
            "rendered_speech": {
                "minutes": rendered_minutes,
                "false_barge_events": false_barges,
                "false_barges_per_30_minutes": _round_canonical(
                    false_barges * 30.0 / rendered_minutes,
                ),
            },
            "double_talk": {
                "trials": double_talk_trials,
                "detected": double_talk_detected,
                "recall": _round_canonical(
                    double_talk_detected / double_talk_trials,
                )
                if double_talk_trials
                else 0.0,
            },
            "interruption": {
                "trials": len(stop_latency_samples),
                "stop_latency_samples_ms": stop_latency_samples,
                "p95_stop_latency_ms": _percentile(stop_latency_samples, 0.95),
                "manual_available": _boolean(
                    observations,
                    "manual_interruption_available",
                ),
            },
            "silence": {
                "minutes": _number(
                    observations,
                    "silence_minutes",
                    minimum=0.000001,
                ),
                "false_barge_events": _integer(
                    observations,
                    "silence_false_barge_events",
                ),
            },
            "device_switch": {
                "trials": len(route_trials),
                "safe_fallbacks": _safe_route_trial_count(route_trials),
                "route_generation_trials": route_trials,
            },
            "soak": {
                "minutes": _number(
                    observations,
                    "soak_minutes",
                    minimum=0.000001,
                ),
                "unbounded_task_growth": _boolean(
                    observations,
                    "unbounded_task_growth",
                ),
                "process_leaks": _integer(observations, "process_leaks"),
                "audio_ring_growth": _boolean(observations, "audio_ring_growth"),
                "device_handle_leaks": _integer(
                    observations,
                    "device_handle_leaks",
                ),
                "post_fence_callbacks": _integer(
                    observations,
                    "post_fence_callbacks",
                ),
            },
        },
        "degradation": {
            "playback_speech_admission_closed": _boolean(
                observations,
                "playback_speech_admission_closed",
            ),
            "manual_interruption_available": _boolean(
                observations,
                "manual_interruption_available",
            ),
            "unsuppressed_interruption_observed": _boolean(
                observations,
                "unsuppressed_interruption_observed",
            ),
        },
        "operator_checklist": dict(checklist),
        "thresholds": dict(_THRESHOLDS),
    }
    report["passed"] = _physical_safety_passes(report)
    validate_physical_report(report)
    return report


def validate_physical_report(
    report: Mapping[str, object],
    *,
    schema_path: Path = _SCHEMA_PATH,
) -> None:
    """Validate strict shape, privacy rules, derivations, and safety outcome."""

    _reject_non_finite_numbers(report)
    _reject_forbidden_report_fields(report)
    if len(_canonical_bytes(report)) + 1 > _MAX_REPORT_BYTES:
        raise PhysicalVoiceReportError("physical report canonical size is too large")
    try:
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        error = next(iter(validator.iter_errors(report)), None)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhysicalVoiceReportError("physical report schema is unavailable") from exc
    except SchemaError as exc:
        raise PhysicalVoiceReportError("physical report schema is invalid") from exc
    if error is not None:
        raise PhysicalVoiceReportError("physical report schema validation failed")
    _validate_integer_type_image(report)
    prerequisites = report["automated_prerequisites"]
    aec = report["aec"]
    isolation = report["isolation"]
    trials = report["trials"]
    assert isinstance(prerequisites, Mapping)
    assert isinstance(aec, Mapping)
    assert isinstance(isolation, Mapping)
    assert isinstance(trials, Mapping)
    if prerequisites["source_tree_digest"] != report["source_tree_digest"]:
        raise PhysicalVoiceReportError("physical report source-tree digest is mixed")
    device = report["device"]
    assert isinstance(device, Mapping)
    allowed_transport = {
        "builtin": {"builtin"},
        "usb": {"usb", "wired"},
        "bluetooth": {"bluetooth"},
    }
    if device["transport"] not in allowed_transport[report["device_class"]]:
        raise PhysicalVoiceReportError(
            "physical report transport does not match device class"
        )
    if aec["delay_estimate_refined"] is True and (
        aec["delay_estimate_available"] is not True
    ):
        raise PhysicalVoiceReportError(
            "physical report refined delay estimate is unavailable"
        )
    if report["safety_path"] == "aec" and (
        aec["delay_estimate_available"] is not True
        or aec["delay_estimate_refined"] is not True
    ):
        raise PhysicalVoiceReportError(
            "physical report AEC path lacks refined delay evidence"
        )
    if report["safety_path"] == "acoustic-isolation" and aec["health_path"] not in {
        "warming",
        "degraded",
    }:
        raise PhysicalVoiceReportError(
            "physical report isolation path conflicts with healthy AEC"
        )
    rendered = trials["rendered_speech"]
    double_talk = trials["double_talk"]
    interruption = trials["interruption"]
    degradation = report["degradation"]
    assert all(
        isinstance(section, Mapping)
        for section in (rendered, double_talk, interruption, degradation)
    )
    if (
        interruption["manual_available"]
        is not degradation["manual_interruption_available"]
    ):
        raise PhysicalVoiceReportError(
            "physical report manual interruption derivation is invalid"
        )
    if double_talk["detected"] > double_talk["trials"]:
        raise PhysicalVoiceReportError(
            "physical report double-talk detections exceed trials"
        )
    expected_false_rate = _round_canonical(
        rendered["false_barge_events"] * 30.0 / rendered["minutes"],
    )
    expected_recall = (
        _round_canonical(double_talk["detected"] / double_talk["trials"])
        if double_talk["trials"]
        else 0.0
    )
    erle_samples = aec["erle_samples_db"]
    correlation_samples = isolation["correlation_samples"]
    leakage_samples = isolation["leakage_db_samples"]
    stop_latency_samples = interruption["stop_latency_samples_ms"]
    assert all(
        isinstance(samples, list)
        for samples in (
            erle_samples,
            correlation_samples,
            leakage_samples,
            stop_latency_samples,
        )
    )
    if any(
        float(sample) != round(float(sample), 6)
        for samples in (
            erle_samples,
            correlation_samples,
            leakage_samples,
            stop_latency_samples,
        )
        for sample in samples
    ):
        raise PhysicalVoiceReportError(
            "physical report sample derivation is not canonical"
        )
    if len(correlation_samples) != len(leakage_samples):
        raise PhysicalVoiceReportError(
            "physical report isolation samples are not paired"
        )
    warmup_duration_ms = isolation["warmup_duration_ms"]
    if warmup_duration_ms != round(float(warmup_duration_ms), 6):
        raise PhysicalVoiceReportError(
            "physical report warmup derivation is not canonical"
        )
    demotion_records = isolation["demotion_reasons"]
    assert isinstance(demotion_records, list)
    demotion_reasons = [record["reason"] for record in demotion_records]
    if demotion_reasons != sorted(set(demotion_reasons)):
        raise PhysicalVoiceReportError(
            "physical report demotion reason records are not canonical"
        )
    expected_demotions = sum(record["count"] for record in demotion_records)
    if expected_demotions > _MAX_EVENT_COUNT:
        raise PhysicalVoiceReportError(
            "physical report demotion count exceeds integer bounds"
        )
    device_switch = trials["device_switch"]
    assert isinstance(device_switch, Mapping)
    route_trials = device_switch["route_generation_trials"]
    assert isinstance(route_trials, list)
    assert all(isinstance(trial, Mapping) for trial in route_trials)
    expected_safe_fallbacks = _safe_route_trial_count(route_trials)
    if (
        rendered["false_barges_per_30_minutes"] != expected_false_rate
        or double_talk["recall"] != expected_recall
        or aec["median_erle_db"] != _median(erle_samples)
        or aec["p10_erle_db"] != _percentile(erle_samples, 0.10)
        or isolation["p95_correlation"] != _percentile(correlation_samples, 0.95)
        or isolation["p95_leakage_db"] != _percentile(leakage_samples, 0.95)
        or isolation["eligible_windows"] != len(correlation_samples)
        or interruption["trials"] != len(stop_latency_samples)
        or interruption["p95_stop_latency_ms"]
        != _percentile(stop_latency_samples, 0.95)
    ):
        raise PhysicalVoiceReportError("physical report metric derivation is invalid")
    if isolation["demotions"] != expected_demotions:
        raise PhysicalVoiceReportError(
            "physical report demotion count derivation is invalid"
        )
    if (
        device_switch["trials"] != len(route_trials)
        or device_switch["safe_fallbacks"] != expected_safe_fallbacks
    ):
        raise PhysicalVoiceReportError(
            "physical report route trial derivation is invalid"
        )
    if report["passed"] is not _physical_safety_passes(report):
        raise PhysicalVoiceReportError("physical report safety outcome is invalid")


def read_physical_report(path: Path) -> tuple[dict[str, object], bytes]:
    """Read and strictly validate one bounded physical qualification report."""

    report, raw = _read_json_report(path)
    validate_physical_report(report)
    return report, raw


def write_physical_report(path: Path, report: Mapping[str, object]) -> None:
    """Atomically write canonical JSON without ever following an output symlink."""

    validate_physical_report(report)
    output = Path(path)
    if not output.name:
        raise PhysicalVoiceReportError("physical report output name is invalid")
    with _OutputDirectory.acquire(output.parent) as directory:
        old_identity: _FileIdentity | None = None
        try:
            metadata = directory.stat(output.name)
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(metadata.st_mode):
                raise PhysicalVoiceReportError(
                    "physical report output cannot be a symlink"
                )
            if not stat.S_ISREG(metadata.st_mode):
                raise PhysicalVoiceReportError(
                    "physical report output must be a regular file"
                )
            old_identity = _identity(metadata)

        staged_name: str | None = None
        staged_identity: _FileIdentity | None = None
        backup_name: str | None = None
        backup_identity: _FileIdentity | None = None
        replacement_attempted = False
        publication_durable = False
        try:
            staged_name, staged_identity = _stage_bytes(
                directory,
                output.name,
                _serialized_report(report),
            )
            if old_identity is not None:
                backup_name = f".{output.name}.voice-backup-{secrets.token_hex(12)}"
                directory.link(output.name, backup_name)
                backup_metadata = directory.stat(backup_name)
                backup_identity = _identity(backup_metadata)
                if (
                    not stat.S_ISREG(backup_metadata.st_mode)
                    or backup_identity != old_identity
                ):
                    raise PhysicalVoiceReportError(
                        "physical report backup identity changed"
                    )
                if directory.descriptor is not None:
                    directory.sync()

            replacement_attempted = True
            directory.replace(staged_name, output.name)
            staged_name = None
            published_metadata = directory.stat(output.name)
            if (
                not stat.S_ISREG(published_metadata.st_mode)
                or _identity(published_metadata) != staged_identity
            ):
                raise PhysicalVoiceReportError(
                    "physical report publication identity changed"
                )
            if directory.descriptor is not None:
                directory.sync()
            publication_durable = True
        except BaseException as primary:
            cleanup_failures: list[str] = []
            if backup_name is not None and backup_identity is None:
                try:
                    backup_identity = _identity(directory.stat(backup_name))
                except FileNotFoundError:
                    backup_name = None
                except BaseException as exc:
                    cleanup_failures.append(
                        f"backup inspection: {type(exc).__name__}: {exc}"
                    )
            replacement_present = False
            unowned_replacement = False
            if replacement_attempted and staged_identity is not None:
                try:
                    replacement_metadata = directory.stat(output.name)
                    replacement_present = (
                        stat.S_ISREG(replacement_metadata.st_mode)
                        and _identity(replacement_metadata) == staged_identity
                    )
                    unowned_replacement = not replacement_present
                except FileNotFoundError:
                    pass
                except BaseException as exc:
                    unowned_replacement = True
                    cleanup_failures.append(
                        f"replacement inspection: {type(exc).__name__}: {exc}"
                    )

            retain_backup = unowned_replacement and backup_name is not None
            if replacement_present and not publication_durable:
                if backup_name is not None and old_identity is not None:
                    try:
                        directory.replace(backup_name, output.name)
                        backup_name = None
                        backup_identity = None
                        restored_metadata = directory.stat(output.name)
                        if (
                            not stat.S_ISREG(restored_metadata.st_mode)
                            or _identity(restored_metadata) != old_identity
                        ):
                            raise PhysicalVoiceReportError(
                                "physical report restore identity changed"
                            )
                        if directory.descriptor is not None:
                            directory.sync()
                    except BaseException as exc:
                        retain_backup = True
                        cleanup_failures.append(
                            f"report restore: {type(exc).__name__}: {exc}"
                        )
                elif staged_identity is not None:
                    try:
                        _unlink_if_owned(directory, output.name, staged_identity)
                        if directory.descriptor is not None:
                            directory.sync()
                    except BaseException as exc:
                        cleanup_failures.append(
                            f"fresh report rollback: {type(exc).__name__}: {exc}"
                        )

            if staged_name is not None and staged_identity is not None:
                try:
                    _unlink_if_owned(directory, staged_name, staged_identity)
                except BaseException as exc:
                    cleanup_failures.append(
                        f"report staging cleanup: {type(exc).__name__}: {exc}"
                    )
            if (
                backup_name is not None
                and backup_identity is not None
                and not retain_backup
            ):
                try:
                    _unlink_if_owned(directory, backup_name, backup_identity)
                except BaseException as exc:
                    cleanup_failures.append(
                        f"report backup cleanup: {type(exc).__name__}: {exc}"
                    )

            if not isinstance(
                primary,
                (OSError, ValueError, PhysicalVoiceReportError),
            ):
                for cleanup_failure in cleanup_failures:
                    primary.add_note(cleanup_failure)
                raise
            if cleanup_failures:
                raise PhysicalVoiceReportError(
                    "physical report publication failed; restore or rollback failed: "
                    + "; ".join(cleanup_failures)
                ) from primary
            if isinstance(primary, PhysicalVoiceReportError):
                raise
            raise PhysicalVoiceReportError(
                "physical report publication failed"
            ) from primary

        if backup_name is not None and backup_identity is not None:
            try:
                _unlink_if_owned(directory, backup_name, backup_identity)
                if directory.descriptor is not None:
                    directory.sync()
            except (OSError, ValueError, PhysicalVoiceReportError) as exc:
                raise PhysicalVoiceReportError(
                    "physical report published but backup cleanup failed"
                ) from exc


def physical_report_summary(report: Mapping[str, object]) -> str:
    """Return a content-free operator summary for one report."""

    validate_physical_report(report)
    aec = report["aec"]
    assert isinstance(aec, Mapping)
    return "\n".join(
        (
            "Speculative voice physical qualification",
            f"Evidence: {report['evidence_kind']}",
            f"Platform: {report['platform']}",
            f"Device class: {report['device_class']}",
            f"Safety path: {report['safety_path']}",
            "Effective mode: "
            + (
                "half-duplex"
                if report["safety_path"] == "half-duplex"
                else "full-duplex"
            ),
            f"AEC health: {aec['health_path']}",
            f"Result: {'PASS' if report['passed'] else 'FAIL'}",
            "",
        )
    )


def _serialized_report(report: Mapping[str, object]) -> bytes:
    serialized = _canonical_bytes(report) + b"\n"
    if len(serialized) > _MAX_REPORT_BYTES:
        raise PhysicalVoiceReportError("physical report output is too large")
    return serialized


def _require_fresh_output_target(
    directory: _OutputDirectory,
    name: str,
    *,
    label: str,
) -> None:
    try:
        directory.stat(name)
    except FileNotFoundError:
        return
    except (OSError, ValueError) as exc:
        raise PhysicalVoiceReportError(f"{label} output is unavailable") from exc
    raise PhysicalVoiceReportError(f"{label} output already exists")


def _identity(metadata: os.stat_result) -> _FileIdentity:
    return _FileIdentity(metadata.st_dev, metadata.st_ino)


def _unlink_if_owned(
    directory: _OutputDirectory,
    name: str,
    identity: _FileIdentity,
) -> bool:
    try:
        metadata = directory.stat(name)
    except FileNotFoundError:
        return True
    if not stat.S_ISREG(metadata.st_mode) or _identity(metadata) != identity:
        return True
    directory.unlink(name)
    return True


def _stage_bytes(
    directory: _OutputDirectory,
    output_name: str,
    serialized: bytes,
) -> tuple[str, _FileIdentity]:
    if len(serialized) > _MAX_REPORT_BYTES:
        raise PhysicalVoiceReportError("qualification output is too large")
    directory.revalidate()
    stage_name = f".{output_name}.voice-stage-{secrets.token_hex(12)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    identity: _FileIdentity | None = None
    failure: BaseException | None = None
    cleanup_failures: list[str] = []
    try:
        descriptor = directory.open_exclusive(stage_name, flags, 0o600)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise PhysicalVoiceReportError(
                "qualification staging output must be a regular file"
            )
        identity = _identity(metadata)
        view = memoryview(serialized)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("qualification staging write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        staged_descriptor = descriptor
        descriptor = None
        os.close(staged_descriptor)
        if _identity(directory.stat(stage_name)) != identity:
            raise PhysicalVoiceReportError("qualification staging identity changed")
        directory.revalidate()
    except BaseException as exc:
        failure = exc
    if descriptor is not None:
        staged_descriptor = descriptor
        descriptor = None
        try:
            os.close(staged_descriptor)
        except OSError as exc:
            if failure is None:
                failure = exc
            else:
                cleanup_failures.append(f"staging descriptor close: {exc}")
    if failure is not None:
        if identity is not None:
            try:
                _unlink_if_owned(directory, stage_name, identity)
            except BaseException as exc:
                cleanup_failures.append(
                    f"staging output cleanup: {type(exc).__name__}: {exc}"
                )
        if not isinstance(
            failure,
            (OSError, ValueError, PhysicalVoiceReportError),
        ):
            for cleanup_failure in cleanup_failures:
                failure.add_note(cleanup_failure)
            raise failure
        if cleanup_failures:
            raise PhysicalVoiceReportError(
                "qualification staging and cleanup failed: "
                + "; ".join(cleanup_failures)
            ) from failure
        if isinstance(failure, PhysicalVoiceReportError):
            raise failure
        raise PhysicalVoiceReportError("qualification staging failed") from failure
    assert identity is not None
    return stage_name, identity


def _publish_exclusive(
    directory: _OutputDirectory,
    stage_name: str,
    output_name: str,
    identity: _FileIdentity,
) -> bool:
    stage_consumed = directory.windows_pin is not None
    if stage_consumed:
        directory.move_write_through(
            stage_name,
            output_name,
            replace=False,
        )
    else:
        directory.link(stage_name, output_name)
    metadata = directory.stat(output_name)
    if not stat.S_ISREG(metadata.st_mode) or _identity(metadata) != identity:
        raise PhysicalVoiceReportError("qualification publication identity changed")
    return stage_consumed


def _cleanup_owned(
    directory: _OutputDirectory,
    name: str | None,
    identity: _FileIdentity | None,
    *,
    label: str,
    failures: list[str],
) -> bool:
    if name is None or identity is None:
        return True
    try:
        return _unlink_if_owned(directory, name, identity)
    except BaseException as exc:
        failures.append(f"{label}: {type(exc).__name__}: {exc}")
        return False


def _publish_report_pair(
    directory: _OutputDirectory,
    report_path: Path,
    report: Mapping[str, object],
    summary_path: Path,
    summary: str,
) -> None:
    validate_physical_report(report)
    report_bytes = _serialized_report(report)
    try:
        summary_bytes = summary.encode("utf-8")
    except UnicodeError as exc:
        raise PhysicalVoiceReportError("physical summary serialization failed") from exc
    if len(summary_bytes) > _MAX_REPORT_BYTES:
        raise PhysicalVoiceReportError("physical summary output is too large")

    report_stage: str | None = None
    summary_stage: str | None = None
    report_identity: _FileIdentity | None = None
    summary_identity: _FileIdentity | None = None
    failure: BaseException | None = None
    try:
        report_stage, report_identity = _stage_bytes(
            directory,
            report_path.name,
            report_bytes,
        )
        summary_stage, summary_identity = _stage_bytes(
            directory,
            summary_path.name,
            summary_bytes,
        )
        directory.revalidate()
        summary_stage_consumed = _publish_exclusive(
            directory,
            summary_stage,
            summary_path.name,
            summary_identity,
        )
        if not summary_stage_consumed:
            _unlink_if_owned(directory, summary_stage, summary_identity)
        summary_stage = None
        if directory.descriptor is not None:
            directory.sync()
        directory.revalidate()
        report_stage_consumed = _publish_exclusive(
            directory,
            report_stage,
            report_path.name,
            report_identity,
        )
        if not report_stage_consumed:
            _unlink_if_owned(directory, report_stage, report_identity)
        report_stage = None
        if directory.descriptor is not None:
            directory.sync()
        directory.revalidate()
    except BaseException as exc:
        failure = exc

    if failure is None:
        return

    cleanup_failures: list[str] = []
    report_absent = _cleanup_owned(
        directory,
        report_path.name,
        report_identity,
        label="report rollback",
        failures=cleanup_failures,
    )
    if report_absent:
        _cleanup_owned(
            directory,
            summary_path.name,
            summary_identity,
            label="summary rollback",
            failures=cleanup_failures,
        )
    _cleanup_owned(
        directory,
        report_stage,
        report_identity,
        label="report staging cleanup",
        failures=cleanup_failures,
    )
    _cleanup_owned(
        directory,
        summary_stage,
        summary_identity,
        label="summary staging cleanup",
        failures=cleanup_failures,
    )
    if directory.descriptor is not None:
        try:
            directory.sync()
        except BaseException as exc:
            cleanup_failures.append(
                f"rollback directory sync: {type(exc).__name__}: {exc}"
            )
    if cleanup_failures:
        if not isinstance(
            failure,
            (OSError, ValueError, PhysicalVoiceReportError),
        ):
            for cleanup_failure in cleanup_failures:
                failure.add_note(cleanup_failure)
            raise failure
        detail = "; ".join(cleanup_failures)
        raise PhysicalVoiceReportError(
            f"physical evidence publication failed; rollback failed: {detail}"
        ) from failure
    if not isinstance(
        failure,
        (OSError, ValueError, PhysicalVoiceReportError),
    ):
        raise failure
    raise PhysicalVoiceReportError("physical evidence publication failed") from failure


def main(argv: list[str] | None = None) -> int:
    """Run a synthetic harness check or collect real physical evidence."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-tree-digest", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument(
        "--device-class",
        choices=("builtin", "usb", "bluetooth"),
        required=True,
    )
    parser.add_argument("--automated-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fixture",
        choices=(
            "safe-full-duplex",
            "safe-isolated-full-duplex",
            "safe-half-duplex",
            "unsafe-unsuppressed",
        ),
    )
    args = parser.parse_args(argv)
    try:
        try:
            summary_path = args.output.with_suffix(".summary.txt")
        except (OSError, ValueError) as exc:
            raise PhysicalVoiceReportError(
                "physical qualification output path is invalid"
            ) from exc
        if summary_path == args.output:
            raise PhysicalVoiceReportError(
                "physical report and summary outputs must be distinct"
            )
        if args.fixture is None and args.platform != _runtime_platform_key():
            raise PhysicalVoiceReportError(
                "physical qualification platform does not match this runtime"
            )
        with _OutputDirectory.acquire(args.output.parent) as directory:
            _require_fresh_output_target(
                directory,
                args.output.name,
                label="physical report",
            )
            _require_fresh_output_target(
                directory,
                summary_path.name,
                label="physical summary",
            )
            if args.fixture is None:
                actual_digest = compute_voice_source_digest(
                    root=_ROOT,
                    path_list=_SOURCE_PATH_LIST,
                )
                if actual_digest != args.source_tree_digest:
                    raise PhysicalVoiceReportError(
                        "source-tree digest does not match listed source"
                    )
            prerequisites = load_automated_prerequisites(
                args.automated_report,
                expected_source_tree_digest=args.source_tree_digest,
                platform_key=args.platform,
            )
            if args.fixture is None:
                try:
                    observations = asyncio.run(
                        collect_live_observations(device_class=args.device_class)
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    raise PhysicalVoiceReportError(
                        "live physical voice collection failed"
                    ) from exc
                evidence_kind = "physical"
            else:
                fixture_path = _FIXTURE_ROOT / f"{args.fixture.replace('-', '_')}.json"
                observations = load_fixture_observations(fixture_path)
                evidence_kind = "synthetic-fixture"
            report = build_physical_report(
                evidence_kind=evidence_kind,
                source_tree_digest=args.source_tree_digest,
                platform_key=args.platform,
                device_class=args.device_class,
                app_version=__version__,
                prerequisites=prerequisites,
                observations=observations,
            )
            summary = physical_report_summary(report)
            _publish_report_pair(
                directory,
                args.output,
                report,
                summary_path,
                summary,
            )
            print(summary, end="")
    except (PhysicalVoiceReportError, PhysicalVoiceRunnerError) as exc:
        message = str(exc)
        notes = getattr(exc, "__notes__", ())
        if notes:
            message += "; " + "; ".join(notes)
        parser.error(message)
    return 0 if report["passed"] is True else 1


__all__ = [
    "AutomatedPrerequisiteEvidence",
    "PhysicalVoiceReportError",
    "build_physical_report",
    "load_automated_prerequisites",
    "load_fixture_observations",
    "main",
    "physical_report_summary",
    "read_physical_report",
    "validate_physical_report",
    "write_physical_report",
]
