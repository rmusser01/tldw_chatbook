"""Bounded, privacy-safe local observation of RAM and optional accelerator memory."""

from __future__ import annotations

import os
import platform
import queue
import stat
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

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


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Bounded command outcome without raw error text."""

    return_code: int | None
    output: bytes
    reason: ProbeReason | None


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
    """Build production sources while keeping all observations injectable."""

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
        available = getattr(memory, "available")
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
            nvidia_result = (
                (),
                AcceleratorState.NOT_OBSERVED,
                ProbeReason.COMMAND_FAILED,
            )
            break
        if (
            type(result) is not CommandResult
            or type(result.output) is not bytes
            or (result.return_code is not None and type(result.return_code) is not int)
            or (result.reason is not None and type(result.reason) is not ProbeReason)
        ):
            nvidia_result = (
                (),
                AcceleratorState.NOT_OBSERVED,
                ProbeReason.COMMAND_FAILED,
            )
            break
        if len(result.output) > MAX_COMMAND_OUTPUT_BYTES:
            nvidia_result = (
                (),
                AcceleratorState.NOT_OBSERVED,
                ProbeReason.OUTPUT_TOO_LARGE,
            )
            break
        if result.reason is not None:
            nvidia_result = ((), AcceleratorState.NOT_OBSERVED, result.reason)
            break
        if result.return_code != 0:
            nvidia_result = (
                (),
                AcceleratorState.NOT_OBSERVED,
                ProbeReason.COMMAND_FAILED,
            )
            break
        observations, parse_reason = _parse_nvidia_output(result.output)
        if parse_reason is not None:
            nvidia_result = ((), AcceleratorState.NOT_OBSERVED, parse_reason)
            break
        return observations, AcceleratorState.OBSERVED, None
    else:
        nvidia_result = ((), AcceleratorState.NOT_OBSERVED, trust_reason)

    if platform_name != "linux":
        if nvidia_result[2] is ProbeReason.PERMISSION_DENIED:
            return (), AcceleratorState.PERMISSION_DENIED, ProbeReason.PERMISSION_DENIED
        return nvidia_result
    drm_observations, drm_reason = _observe_linux_drm(sources)
    if drm_observations:
        incomplete_reason = drm_reason
        if incomplete_reason is None and nvidia_result[2] not in {
            None,
            ProbeReason.EXECUTABLE_NOT_FOUND,
        }:
            incomplete_reason = nvidia_result[2]
        return (
            drm_observations,
            (
                AcceleratorState.PARTIAL
                if incomplete_reason is not None
                else AcceleratorState.OBSERVED
            ),
            incomplete_reason,
        )
    if drm_reason is ProbeReason.SYSFS_PERMISSION_DENIED:
        return (), AcceleratorState.PERMISSION_DENIED, drm_reason
    if drm_reason is not None:
        return (), AcceleratorState.NOT_OBSERVED, drm_reason
    if nvidia_result[2] is ProbeReason.PERMISSION_DENIED:
        return (), AcceleratorState.PERMISSION_DENIED, ProbeReason.PERMISSION_DENIED
    return nvidia_result


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
    """Observe bounded local memory facts without propagating raw errors."""

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
    if total is None:
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
        memory_kind=MemoryKind.SYSTEM,
        accelerators=accelerators,
        system_reason=reason,
        accelerator_reason=accelerator_reason,
    )


def _run_bounded_command(
    executable: Path,
    argv: tuple[str, ...],
    timeout: float,
    output_limit: int,
) -> CommandResult:
    """Run one trusted executable with a bounded combined-output buffer."""

    try:
        process = subprocess.Popen(
            [str(executable), *argv],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
        )
    except FileNotFoundError:
        return CommandResult(None, b"", ProbeReason.EXECUTABLE_NOT_FOUND)
    except PermissionError:
        return CommandResult(None, b"", ProbeReason.PERMISSION_DENIED)
    except Exception:
        return CommandResult(None, b"", ProbeReason.COMMAND_FAILED)
    if process.stdout is None:
        _terminate_and_reap(process)
        return CommandResult(None, b"", ProbeReason.COMMAND_FAILED)

    messages: queue.Queue[tuple[bool, bytes]] = queue.Queue(maxsize=1)

    def read_stdout() -> None:
        try:
            while True:
                chunk = process.stdout.read(min(8192, output_limit + 1))
                messages.put((True, chunk))
                if not chunk:
                    return
        except Exception:
            messages.put((False, b""))

    threading.Thread(target=read_stdout, daemon=True).start()
    output = bytearray()
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_and_reap(process)
            return CommandResult(None, b"", ProbeReason.COMMAND_TIMEOUT)
        try:
            succeeded, chunk = messages.get(timeout=remaining)
        except queue.Empty:
            _terminate_and_reap(process)
            return CommandResult(None, b"", ProbeReason.COMMAND_TIMEOUT)
        if not succeeded:
            _terminate_and_reap(process)
            return CommandResult(None, b"", ProbeReason.COMMAND_FAILED)
        if not chunk:
            break
        if len(output) + len(chunk) > output_limit:
            _terminate_and_reap(process)
            return CommandResult(None, b"", ProbeReason.OUTPUT_TOO_LARGE)
        output.extend(chunk)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        _terminate_and_reap(process)
        return CommandResult(None, b"", ProbeReason.COMMAND_TIMEOUT)
    try:
        return_code = process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        _terminate_and_reap(process)
        return CommandResult(None, b"", ProbeReason.COMMAND_TIMEOUT)
    except Exception:
        _terminate_and_reap(process)
        return CommandResult(None, b"", ProbeReason.COMMAND_FAILED)
    return CommandResult(return_code, bytes(output), None)


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
        process.wait()
    except Exception:
        pass
