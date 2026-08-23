"""Bounded, privacy-safe local observation of RAM and optional accelerator memory."""

from __future__ import annotations

import os
import platform
import secrets
import stat
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import islice
from pathlib import Path
from typing import Protocol

import psutil

from .machine_memory import (
    AcceleratorMemoryObservation,
    AcceleratorSource,
    AcceleratorState,
    MAX_ACCELERATORS,
    MAX_INPUT_BYTES,
    MIB,
    MachineMemorySnapshot,
    MemoryKind,
    ProbeReason,
    SystemMemoryState,
)


LINUX_NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
WINDOWS_NVIDIA_SMI = (
    Path(r"C:\Windows\System32\nvidia-smi.exe"),
    Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
)
NVIDIA_ARGV = (
    "--query-gpu=index,name,memory.total",
    "--format=csv,noheader,nounits",
)
COMMAND_TIMEOUT_SECONDS = 2.0
TERMINATE_GRACE_SECONDS = 0.25
MAX_COMMAND_OUTPUT_BYTES = 64 * 1024
MAX_SYSFS_READ_BYTES = 64
_PIPE_POLL_INTERVAL_SECONDS = 0.005
_WINDOWS_PIPE_ACCESS_INBOUND = 0x00000001
_WINDOWS_FILE_FLAG_FIRST_PIPE_INSTANCE = 0x00080000
_WINDOWS_PIPE_NOWAIT = 0x00000001
_WINDOWS_PIPE_REJECT_REMOTE_CLIENTS = 0x00000008
_WINDOWS_ERROR_BROKEN_PIPE = 109
_WINDOWS_ERROR_NO_DATA = 232
_WINDOWS_ERROR_PIPE_NOT_CONNECTED = 233
_WINDOWS_ERROR_PIPE_CONNECTED = 535
_WINDOWS_ERROR_PIPE_LISTENING = 536
_WINDOWS_PIPE_BUFFER_BYTES = 8192


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Bounded command outcome without raw error text."""

    return_code: int | None
    output: bytes
    reason: ProbeReason | None


class _CommandOutput(Protocol):
    """Owned command-output channel with deadline-safe reads."""

    def read_nowait(self, maximum_bytes: int) -> tuple[bytes, bool]: ...

    def close(self) -> None: ...


class _WindowsPipeApi(Protocol):
    """Small injectable boundary around the required Win32 pipe calls."""

    def create_named_pipe(
        self,
        name: str,
        open_mode: int,
        pipe_mode: int,
        input_buffer_size: int,
    ) -> int: ...

    def connect_named_pipe(self, handle: int) -> int | None: ...

    def open_writer(self, name: str) -> int: ...

    def wrap_writer(self, handle: int) -> object: ...

    def read_file(
        self, handle: int, maximum_bytes: int
    ) -> tuple[bytes, int | None]: ...

    def close_handle(self, handle: int) -> None: ...


@dataclass(slots=True)
class _PosixCommandOutput:
    stream: object
    descriptor: int

    def read_nowait(self, maximum_bytes: int) -> tuple[bytes, bool]:
        return _read_command_stdout_nowait(self.descriptor, maximum_bytes)

    def close(self) -> None:
        _close_command_stdout(self.stream)


@dataclass(slots=True)
class _WindowsNamedPipeOutput:
    handle: int
    api: _WindowsPipeApi
    closed: bool = False

    def read_nowait(self, maximum_bytes: int) -> tuple[bytes, bool]:
        if maximum_bytes <= 0:
            return b"", False
        chunk, error_code = self.api.read_file(self.handle, maximum_bytes)
        if error_code is None:
            return chunk, False
        if error_code == _WINDOWS_ERROR_NO_DATA:
            return b"", False
        if error_code in {
            _WINDOWS_ERROR_BROKEN_PIPE,
            _WINDOWS_ERROR_PIPE_NOT_CONNECTED,
        }:
            return b"", True
        raise OSError(error_code, "ReadFile failed")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.api.close_handle(self.handle)
        except Exception:
            pass


@dataclass(frozen=True, slots=True)
class MachineProbeSources:
    """Injected operating-system observation seams."""

    platform_name: Callable[[], str]
    architecture: Callable[[], str]
    virtual_memory: Callable[[], object]
    lstat_path: Callable[[Path], os.stat_result]
    resolve_path: Callable[[Path], Path]
    read_bounded: Callable[[Path, int], bytes]
    drm_cards: Callable[[], tuple[Path, ...]]
    run_command: Callable[[Path, tuple[str, ...], float, int], CommandResult]


def _read_bounded(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        return stream.read(limit + 1)


def _drm_cards() -> tuple[Path, ...]:
    candidates = Path("/sys/class/drm").glob("card[0-9]*")
    exact_cards = (
        path
        for path in candidates
        if path.name[4:].isascii() and path.name[4:].isdigit()
    )
    return tuple(islice(exact_cards, MAX_ACCELERATORS))


def production_probe_sources() -> MachineProbeSources:
    """Build production sources while keeping all observations injectable.

    Returns:
        Trusted, bounded platform adapters for the production memory probe.
    """

    return MachineProbeSources(
        platform_name=platform.system,
        architecture=platform.machine,
        virtual_memory=psutil.virtual_memory,
        lstat_path=lambda path: path.lstat(),
        resolve_path=lambda path: path.resolve(strict=True),
        read_bounded=_read_bounded,
        drm_cards=_drm_cards,
        run_command=_run_bounded_command,
    )


def _normalize_platform(value: object) -> str:
    if type(value) is not str:
        return "other"
    normalized = value.casefold()
    return normalized if normalized in {"darwin", "linux", "windows"} else "other"


def _sanitize_identifier(value: object) -> str:
    if type(value) is not str:
        return "unknown"
    sanitized = "".join(
        character
        for character in value[:32]
        if character.isascii() and (character.isalnum() or character in "_.-")
    )
    return sanitized or "unknown"


def _observe_system_memory(
    virtual_memory: Callable[[], object],
) -> tuple[int | None, int | None, SystemMemoryState, ProbeReason | None]:
    try:
        memory = virtual_memory()
        total = getattr(memory, "total")
    except PermissionError:
        return (
            None,
            None,
            SystemMemoryState.PERMISSION_DENIED,
            ProbeReason.PERMISSION_DENIED,
        )
    except Exception:
        return None, None, SystemMemoryState.UNAVAILABLE, ProbeReason.MEMORY_UNAVAILABLE
    if type(total) is not int or not 1 <= total <= MAX_INPUT_BYTES:
        return None, None, SystemMemoryState.UNAVAILABLE, ProbeReason.MEMORY_UNAVAILABLE
    try:
        available = getattr(memory, "available")
    except Exception:
        return total, None, SystemMemoryState.PARTIAL, ProbeReason.MEMORY_UNAVAILABLE
    if type(available) is not int or not 0 <= available <= total:
        return total, None, SystemMemoryState.PARTIAL, ProbeReason.INVALID_MEMORY_VALUE
    return total, available, SystemMemoryState.OBSERVED, None


def _unsupported_snapshot(architecture: str) -> MachineMemorySnapshot:
    return MachineMemorySnapshot(
        platform="other",
        architecture=architecture,
        system_state=SystemMemoryState.UNSUPPORTED,
        accelerator_state=AcceleratorState.UNSUPPORTED,
        total_bytes=None,
        available_bytes=None,
        memory_kind=MemoryKind.UNKNOWN,
        accelerators=(),
        system_reason=ProbeReason.UNSUPPORTED_PLATFORM,
        accelerator_reason=ProbeReason.UNSUPPORTED_PLATFORM,
    )


def _snapshot_without_capacity(
    platform_name: str,
    architecture: str,
    state: SystemMemoryState,
    reason: ProbeReason,
) -> MachineMemorySnapshot:
    return MachineMemorySnapshot(
        platform=platform_name,
        architecture=architecture,
        system_state=state,
        accelerator_state=(
            AcceleratorState.UNSUPPORTED
            if platform_name == "darwin"
            else AcceleratorState.NOT_OBSERVED
        ),
        total_bytes=None,
        available_bytes=None,
        memory_kind=MemoryKind.UNKNOWN,
        accelerators=(),
        system_reason=reason,
        accelerator_reason=(
            ProbeReason.UNSUPPORTED_PLATFORM if platform_name == "darwin" else None
        ),
    )


def _apple_unified_snapshot(
    total: int,
    available: int | None,
    state: SystemMemoryState,
    reason: ProbeReason | None,
    architecture: str,
) -> MachineMemorySnapshot:
    return MachineMemorySnapshot(
        platform="darwin",
        architecture=architecture,
        system_state=state,
        accelerator_state=AcceleratorState.OBSERVED,
        total_bytes=total,
        available_bytes=available,
        memory_kind=MemoryKind.UNIFIED,
        accelerators=(
            AcceleratorMemoryObservation(
                vendor="apple",
                label="Apple unified memory",
                total_bytes=None,
                shared=True,
                source=AcceleratorSource.APPLE_UNIFIED,
            ),
        ),
        system_reason=reason,
        accelerator_reason=None,
    )


def _is_trusted_executable(
    path: Path,
    platform_name: str,
    sources: MachineProbeSources,
) -> tuple[bool, ProbeReason]:
    try:
        path_stat = sources.lstat_path(path)
        parent_stat = sources.lstat_path(path.parent)
        resolved = sources.resolve_path(path)
    except (FileNotFoundError, NotADirectoryError):
        return False, ProbeReason.EXECUTABLE_NOT_FOUND
    except PermissionError:
        return False, ProbeReason.PERMISSION_DENIED
    except Exception:
        return False, ProbeReason.UNTRUSTED_EXECUTABLE
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        return False, ProbeReason.UNTRUSTED_EXECUTABLE
    if resolved != path:
        return False, ProbeReason.UNTRUSTED_EXECUTABLE
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if getattr(path_stat, "st_file_attributes", 0) & reparse:
        return False, ProbeReason.UNTRUSTED_EXECUTABLE
    if platform_name == "linux":
        if path_stat.st_uid != 0 or parent_stat.st_uid != 0:
            return False, ProbeReason.UNTRUSTED_EXECUTABLE
        if (path_stat.st_mode | parent_stat.st_mode) & (stat.S_IWGRP | stat.S_IWOTH):
            return False, ProbeReason.UNTRUSTED_EXECUTABLE
    return True, ProbeReason.EXECUTABLE_NOT_FOUND


def _parse_nvidia_output(
    output: bytes,
) -> tuple[tuple[AcceleratorMemoryObservation, ...], ProbeReason | None]:
    try:
        text = output.decode("utf-8")
    except UnicodeDecodeError:
        return (), ProbeReason.MALFORMED_OUTPUT
    observations: list[AcceleratorMemoryObservation] = []
    indexes: set[int] = set()
    labels: set[str] = set()
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if (
            len(parts) != 3
            or not parts[0].isascii()
            or not parts[0].isdigit()
            or not parts[2].isascii()
            or not parts[2].isdigit()
            or len(parts[0]) > 20
            or len(parts[2]) > 20
        ):
            return (), ProbeReason.MALFORMED_OUTPUT
        try:
            index = int(parts[0])
            memory_mib = int(parts[2])
        except ValueError:
            return (), ProbeReason.MALFORMED_OUTPUT
        if index in indexes:
            return (), ProbeReason.DUPLICATE_DEVICE
        indexes.add(index)
        if len(indexes) > MAX_ACCELERATORS:
            return (), ProbeReason.TOO_MANY_DEVICES
        name = "".join(character for character in parts[1] if character.isprintable())[
            :96
        ]
        total_bytes = memory_mib * MIB
        if not name or memory_mib <= 0 or total_bytes > MAX_INPUT_BYTES:
            return (), ProbeReason.MALFORMED_OUTPUT
        label = name
        duplicate_number = 2
        while label.casefold() in labels:
            suffix = f" #{duplicate_number}"
            label = f"{name[: 96 - len(suffix)]}{suffix}"
            duplicate_number += 1
        labels.add(label.casefold())
        observations.append(
            AcceleratorMemoryObservation(
                vendor="nvidia",
                label=label,
                total_bytes=total_bytes,
                shared=False,
                source=AcceleratorSource.NVIDIA_SMI,
            )
        )
    if not observations:
        return (), ProbeReason.MALFORMED_OUTPUT
    return tuple(observations), None


def _observe_accelerators(
    platform_name: str,
    sources: MachineProbeSources,
) -> tuple[
    tuple[AcceleratorMemoryObservation, ...], AcceleratorState, ProbeReason | None
]:
    nvidia_observations, nvidia_reason = _observe_nvidia(platform_name, sources)
    if platform_name != "linux":
        if nvidia_observations:
            return nvidia_observations, AcceleratorState.OBSERVED, None
        if nvidia_reason is ProbeReason.PERMISSION_DENIED:
            return (), AcceleratorState.PERMISSION_DENIED, nvidia_reason
        return (), AcceleratorState.NOT_OBSERVED, nvidia_reason

    drm_observations, drm_reason = _observe_linux_drm(sources)
    combined = nvidia_observations + drm_observations
    aggregate_reason: ProbeReason | None = None
    if len(combined) > MAX_ACCELERATORS:
        combined = combined[:MAX_ACCELERATORS]
        aggregate_reason = ProbeReason.TOO_MANY_DEVICES
    combined = _unique_accelerator_labels(combined)
    branch_reason = aggregate_reason or drm_reason
    if branch_reason is None and nvidia_reason not in {
        None,
        ProbeReason.EXECUTABLE_NOT_FOUND,
    }:
        branch_reason = nvidia_reason
    if combined:
        return (
            combined,
            (
                AcceleratorState.PARTIAL
                if branch_reason is not None
                else AcceleratorState.OBSERVED
            ),
            branch_reason,
        )
    final_reason = drm_reason or nvidia_reason
    if final_reason in {
        ProbeReason.PERMISSION_DENIED,
        ProbeReason.SYSFS_PERMISSION_DENIED,
    }:
        return (), AcceleratorState.PERMISSION_DENIED, final_reason
    return (), AcceleratorState.NOT_OBSERVED, final_reason


def _unique_accelerator_labels(
    observations: tuple[AcceleratorMemoryObservation, ...],
) -> tuple[AcceleratorMemoryObservation, ...]:
    unique: list[AcceleratorMemoryObservation] = []
    labels: set[str] = set()
    for observation in observations:
        label = observation.label
        duplicate_number = 2
        while label.casefold() in labels:
            suffix = f" #{duplicate_number}"
            label = f"{observation.label[: 96 - len(suffix)]}{suffix}"
            duplicate_number += 1
        labels.add(label.casefold())
        unique.append(
            observation
            if label == observation.label
            else replace(observation, label=label)
        )
    return tuple(unique)


def _observe_nvidia(
    platform_name: str,
    sources: MachineProbeSources,
) -> tuple[tuple[AcceleratorMemoryObservation, ...], ProbeReason | None]:
    paths = (LINUX_NVIDIA_SMI,) if platform_name == "linux" else WINDOWS_NVIDIA_SMI
    trust_reason = ProbeReason.EXECUTABLE_NOT_FOUND
    for executable in paths:
        trusted, candidate_reason = _is_trusted_executable(
            executable, platform_name, sources
        )
        if not trusted:
            if candidate_reason is not ProbeReason.EXECUTABLE_NOT_FOUND:
                trust_reason = candidate_reason
            continue
        try:
            result = sources.run_command(
                executable,
                NVIDIA_ARGV,
                COMMAND_TIMEOUT_SECONDS,
                MAX_COMMAND_OUTPUT_BYTES,
            )
        except Exception:
            return (), ProbeReason.COMMAND_FAILED
        if (
            type(result) is not CommandResult
            or type(result.output) is not bytes
            or (result.return_code is not None and type(result.return_code) is not int)
            or (result.reason is not None and type(result.reason) is not ProbeReason)
        ):
            return (), ProbeReason.COMMAND_FAILED
        if len(result.output) > MAX_COMMAND_OUTPUT_BYTES:
            return (), ProbeReason.OUTPUT_TOO_LARGE
        if result.reason is not None:
            return (), result.reason
        if result.return_code != 0:
            return (), ProbeReason.COMMAND_FAILED
        observations, parse_reason = _parse_nvidia_output(result.output)
        if parse_reason is not None:
            return (), parse_reason
        return observations, None
    return (), trust_reason


def _observe_linux_drm(
    sources: MachineProbeSources,
) -> tuple[tuple[AcceleratorMemoryObservation, ...], ProbeReason | None]:
    try:
        cards = sources.drm_cards()[:MAX_ACCELERATORS]
    except PermissionError:
        return (), ProbeReason.SYSFS_PERMISSION_DENIED
    except Exception:
        return (), ProbeReason.SYSFS_MALFORMED
    observations: list[AcceleratorMemoryObservation] = []
    seen_targets: set[Path] = set()
    failure: ProbeReason | None = None
    sys_devices = Path("/sys/devices")
    for card in cards:
        try:
            target = sources.resolve_path(card / "device")
        except PermissionError:
            failure = ProbeReason.SYSFS_PERMISSION_DENIED
            continue
        except Exception:
            failure = ProbeReason.SYSFS_MALFORMED
            continue
        try:
            target.relative_to(sys_devices)
        except ValueError:
            failure = ProbeReason.SYSFS_UNTRUSTED_PATH
            continue
        if target in seen_targets:
            failure = ProbeReason.DUPLICATE_DEVICE
            continue
        seen_targets.add(target)
        try:
            vendor_bytes = sources.read_bounded(target / "vendor", MAX_SYSFS_READ_BYTES)
            if len(vendor_bytes) > MAX_SYSFS_READ_BYTES:
                failure = ProbeReason.SYSFS_MALFORMED
                continue
            vendor = vendor_bytes.decode("ascii").strip().casefold()
        except PermissionError:
            failure = ProbeReason.SYSFS_PERMISSION_DENIED
            continue
        except Exception:
            failure = ProbeReason.SYSFS_MALFORMED
            continue
        if vendor != "0x1002":
            continue
        try:
            total_bytes_raw = sources.read_bounded(
                target / "mem_info_vram_total", MAX_SYSFS_READ_BYTES
            )
            if len(total_bytes_raw) > MAX_SYSFS_READ_BYTES:
                raise ValueError
            total_text = total_bytes_raw.decode("ascii").strip()
            if not total_text.isdigit():
                raise ValueError
            total_bytes = int(total_text)
            if not 1 <= total_bytes <= MAX_INPUT_BYTES:
                raise ValueError
        except PermissionError:
            failure = ProbeReason.SYSFS_PERMISSION_DENIED
            continue
        except Exception:
            failure = ProbeReason.SYSFS_MALFORMED
            continue
        observations.append(
            AcceleratorMemoryObservation(
                vendor="amd",
                label=f"AMD DRM-reported VRAM {len(observations) + 1}",
                total_bytes=total_bytes,
                shared=False,
                source=AcceleratorSource.LINUX_DRM,
            )
        )
    return tuple(observations), failure


def observe_machine_memory(
    *, sources: MachineProbeSources | None = None
) -> MachineMemorySnapshot:
    """Observe bounded local memory facts without propagating raw errors.

    Args:
        sources: Optional injected platform adapters; production adapters are
            used when omitted.

    Returns:
        An immutable snapshot containing bounded evidence and fixed reasons.
    """

    active = sources or production_probe_sources()
    try:
        platform_name = _normalize_platform(active.platform_name())
    except Exception:
        platform_name = "other"
    try:
        architecture = _sanitize_identifier(active.architecture())
    except Exception:
        architecture = "unknown"
    if platform_name == "other":
        return _unsupported_snapshot(architecture)
    total, available, state, reason = _observe_system_memory(active.virtual_memory)
    if total is None and platform_name == "darwin":
        return _snapshot_without_capacity(
            platform_name,
            architecture,
            state,
            reason or ProbeReason.MEMORY_UNAVAILABLE,
        )
    if platform_name == "darwin" and architecture in {"arm64", "aarch64"}:
        return _apple_unified_snapshot(total, available, state, reason, architecture)
    if platform_name == "darwin":
        return MachineMemorySnapshot(
            platform=platform_name,
            architecture=architecture,
            system_state=state,
            accelerator_state=AcceleratorState.PARTIAL,
            total_bytes=total,
            available_bytes=available,
            memory_kind=MemoryKind.SYSTEM,
            accelerators=(),
            system_reason=reason,
            accelerator_reason=ProbeReason.UNSUPPORTED_PLATFORM,
        )
    accelerators, accelerator_state, accelerator_reason = _observe_accelerators(
        platform_name, active
    )
    return MachineMemorySnapshot(
        platform=platform_name,
        architecture=architecture,
        system_state=state,
        accelerator_state=accelerator_state,
        total_bytes=total,
        available_bytes=available,
        memory_kind=MemoryKind.SYSTEM if total is not None else MemoryKind.UNKNOWN,
        accelerators=accelerators,
        system_reason=reason,
        accelerator_reason=accelerator_reason,
    )


class _CtypesWindowsPipeApi:
    """ctypes-backed Win32 calls, loaded only on the Windows command path."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        self._wintypes = wintypes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._invalid_handle = ctypes.c_void_p(-1).value

        self._create_named_pipe = self._kernel32.CreateNamedPipeW
        self._create_named_pipe.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
        )
        self._create_named_pipe.restype = wintypes.HANDLE

        self._connect_named_pipe = self._kernel32.ConnectNamedPipe
        self._connect_named_pipe.argtypes = (wintypes.HANDLE, wintypes.LPVOID)
        self._connect_named_pipe.restype = wintypes.BOOL

        self._create_file = self._kernel32.CreateFileW
        self._create_file.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        self._create_file.restype = wintypes.HANDLE

        self._read_file = self._kernel32.ReadFile
        self._read_file.argtypes = (
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPVOID,
        )
        self._read_file.restype = wintypes.BOOL

        self._close_handle = self._kernel32.CloseHandle
        self._close_handle.argtypes = (wintypes.HANDLE,)
        self._close_handle.restype = wintypes.BOOL

    def create_named_pipe(
        self,
        name: str,
        open_mode: int,
        pipe_mode: int,
        input_buffer_size: int,
    ) -> int:
        handle = self._create_named_pipe(
            name,
            open_mode,
            pipe_mode,
            1,
            0,
            input_buffer_size,
            0,
            None,
        )
        if handle == self._invalid_handle:
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        return int(handle)

    def connect_named_pipe(self, handle: int) -> int | None:
        if self._connect_named_pipe(handle, None):
            return None
        return self._ctypes.get_last_error()

    def open_writer(self, name: str) -> int:
        generic_write = 0x40000000
        open_existing = 3
        file_attribute_normal = 0x00000080
        handle = self._create_file(
            name,
            generic_write,
            0,
            None,
            open_existing,
            file_attribute_normal,
            None,
        )
        if handle == self._invalid_handle:
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        return int(handle)

    def wrap_writer(self, handle: int) -> object:
        import msvcrt

        try:
            descriptor = msvcrt.open_osfhandle(
                handle,
                os.O_WRONLY | getattr(os, "O_BINARY", 0),
            )
        except Exception:
            self.close_handle(handle)
            raise
        try:
            return os.fdopen(descriptor, "wb", buffering=0)
        except Exception:
            os.close(descriptor)
            raise

    def read_file(self, handle: int, maximum_bytes: int) -> tuple[bytes, int | None]:
        buffer = self._ctypes.create_string_buffer(maximum_bytes)
        bytes_read = self._wintypes.DWORD()
        if self._read_file(
            handle,
            buffer,
            maximum_bytes,
            self._ctypes.byref(bytes_read),
            None,
        ):
            return buffer.raw[: bytes_read.value], None
        return b"", self._ctypes.get_last_error()

    def close_handle(self, handle: int) -> None:
        self._close_handle(handle)


def _open_windows_output_pipe(
    *,
    api: _WindowsPipeApi | None = None,
    pipe_name: str | None = None,
) -> tuple[_WindowsNamedPipeOutput, object]:
    """Open a local nonblocking byte pipe for one child command's output."""

    active_api = api or _CtypesWindowsPipeApi()
    name = pipe_name or (
        rf"\\.\pipe\tldw-chatbook-machine-memory-{os.getpid()}-"
        f"{secrets.token_hex(16)}"
    )
    server_handle = active_api.create_named_pipe(
        name,
        _WINDOWS_PIPE_ACCESS_INBOUND | _WINDOWS_FILE_FLAG_FIRST_PIPE_INSTANCE,
        _WINDOWS_PIPE_NOWAIT | _WINDOWS_PIPE_REJECT_REMOTE_CLIENTS,
        _WINDOWS_PIPE_BUFFER_BYTES,
    )
    output = _WindowsNamedPipeOutput(server_handle, active_api)
    writer_handle: int | None = None
    try:
        listening_result = active_api.connect_named_pipe(server_handle)
        if listening_result not in {None, _WINDOWS_ERROR_PIPE_LISTENING}:
            raise OSError(listening_result, "ConnectNamedPipe failed")
        writer_handle = active_api.open_writer(name)
        connected_result = active_api.connect_named_pipe(server_handle)
        if connected_result not in {None, _WINDOWS_ERROR_PIPE_CONNECTED}:
            raise OSError(connected_result, "ConnectNamedPipe failed")
        owned_writer_handle = writer_handle
        writer_handle = None
        writer = active_api.wrap_writer(owned_writer_handle)
        return output, writer
    except Exception:
        if writer_handle is not None:
            try:
                active_api.close_handle(writer_handle)
            except Exception:
                pass
        output.close()
        raise


def _start_bounded_command(
    executable: Path,
    argv: tuple[str, ...],
) -> tuple[subprocess.Popen[bytes], _CommandOutput]:
    """Start a command with a platform-safe owned output channel."""

    if os.name == "nt":
        output, writer = _open_windows_output_pipe()
        try:
            process = subprocess.Popen(
                [str(executable), *argv],
                stdout=writer,
                stderr=subprocess.STDOUT,
                shell=False,
            )
        except Exception:
            output.close()
            _close_command_stdout(writer)
            raise
        try:
            writer.close()  # type: ignore[attr-defined]
        except Exception:
            _terminate_and_reap(process)
            output.close()
            raise
        return process, output

    process = subprocess.Popen(
        [str(executable), *argv],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
    )
    if process.stdout is None:
        _terminate_and_reap(process)
        raise RuntimeError("stdout pipe unavailable")
    stdout = process.stdout
    try:
        descriptor = _prepare_nonblocking_stdout(stdout)
    except Exception:
        _terminate_and_reap(process)
        _close_command_stdout(stdout)
        raise
    return process, _PosixCommandOutput(stdout, descriptor)


def _run_bounded_command(
    executable: Path,
    argv: tuple[str, ...],
    timeout: float,
    output_limit: int,
) -> CommandResult:
    """Run one trusted executable with a bounded combined-output buffer."""

    try:
        process, command_output = _start_bounded_command(executable, argv)
    except FileNotFoundError:
        return CommandResult(None, b"", ProbeReason.EXECUTABLE_NOT_FOUND)
    except PermissionError:
        return CommandResult(None, b"", ProbeReason.PERMISSION_DENIED)
    except Exception:
        return CommandResult(None, b"", ProbeReason.COMMAND_FAILED)

    def abort(reason: ProbeReason) -> CommandResult:
        _terminate_and_reap(process)
        command_output.close()
        return CommandResult(None, b"", reason)

    output = bytearray()
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return abort(ProbeReason.COMMAND_TIMEOUT)
        try:
            read_limit = min(8192, output_limit - len(output) + 1)
            chunk, reached_eof = command_output.read_nowait(read_limit)
        except Exception:
            return abort(ProbeReason.COMMAND_FAILED)
        if chunk:
            if len(output) + len(chunk) > output_limit:
                return abort(ProbeReason.OUTPUT_TOO_LARGE)
            output.extend(chunk)
            continue
        try:
            return_code = process.poll()
        except Exception:
            return abort(ProbeReason.COMMAND_FAILED)
        if return_code is not None:
            break
        if reached_eof:
            try:
                return_code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                return abort(ProbeReason.COMMAND_TIMEOUT)
            except Exception:
                return abort(ProbeReason.COMMAND_FAILED)
            break
        time.sleep(min(_PIPE_POLL_INTERVAL_SECONDS, remaining))

    command_output.close()
    return CommandResult(return_code, bytes(output), None)


def _prepare_nonblocking_stdout(stdout: object) -> int:
    """Return a descriptor that the collector can poll without blocking."""

    descriptor = stdout.fileno()  # type: ignore[attr-defined]
    if type(descriptor) is not int or descriptor < 0:
        raise ValueError("stdout descriptor is invalid")
    if os.name != "nt":
        os.set_blocking(descriptor, False)
    return descriptor


def _read_command_stdout_nowait(
    descriptor: int, maximum_bytes: int
) -> tuple[bytes, bool]:
    """Read a prepared POSIX descriptor without blocking."""

    if maximum_bytes <= 0:
        return b"", False
    try:
        chunk = os.read(descriptor, maximum_bytes)
    except BlockingIOError:
        return b"", False
    except InterruptedError:
        return b"", False
    return chunk, chunk == b""


def _close_command_stdout(stdout: object) -> None:
    try:
        stdout.close()  # type: ignore[attr-defined]
    except Exception:
        pass


def _terminate_and_reap(process: subprocess.Popen[bytes]) -> None:
    try:
        process.terminate()
    except Exception:
        pass
    try:
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass
    try:
        process.kill()
    except Exception:
        pass
    try:
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass
