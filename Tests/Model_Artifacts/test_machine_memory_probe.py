"""Deterministic tests for bounded, privacy-safe machine-memory observation."""

from __future__ import annotations

import stat
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tldw_chatbook.Model_Artifacts.machine_memory import (
    AcceleratorState,
    AcceleratorSource,
    GIB,
    MAX_INPUT_BYTES,
    MIB,
    MemoryKind,
    ProbeReason,
    SystemMemoryState,
)
from tldw_chatbook.Model_Artifacts import machine_memory_probe as probe
from tldw_chatbook.Model_Artifacts.machine_memory_probe import (
    COMMAND_TIMEOUT_SECONDS,
    LINUX_NVIDIA_SMI,
    MAX_COMMAND_OUTPUT_BYTES,
    NVIDIA_ARGV,
    TERMINATE_GRACE_SECONDS,
    WINDOWS_NVIDIA_SMI,
    CommandResult,
    MachineProbeSources,
    observe_machine_memory,
)


def _sources(
    *,
    platform_name: str,
    architecture: str,
    total: int,
    available: int,
    run_command: Callable[..., CommandResult],
    virtual_memory: Callable[[], object] | None = None,
    lstat_path: Callable[[Path], object] | None = None,
    resolve_path: Callable[[Path], Path] | None = None,
    read_bounded: Callable[[Path, int], bytes] | None = None,
    drm_cards: Callable[[], tuple[Path, ...]] | None = None,
) -> MachineProbeSources:
    trusted_stat = SimpleNamespace(
        st_mode=stat.S_IFREG | 0o755,
        st_uid=0,
        st_file_attributes=0,
    )
    return MachineProbeSources(
        platform_name=lambda: platform_name,
        architecture=lambda: architecture,
        virtual_memory=virtual_memory
        or (lambda: SimpleNamespace(total=total, available=available)),
        lstat_path=lstat_path or (lambda _path: trusted_stat),
        resolve_path=resolve_path or (lambda path: path),
        read_bounded=read_bounded or (lambda _path, _limit: b""),
        drm_cards=drm_cards or (lambda: ()),
        run_command=run_command,
    )


def test_darwin_arm64_reports_one_unified_pool_without_accelerator_command() -> None:
    """Entering the discrete probe would double-count Apple unified memory."""
    runner = Mock()

    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="darwin",
            architecture="arm64",
            total=32 * GIB,
            available=18 * GIB,
            run_command=runner,
        )
    )

    assert snapshot.memory_kind is MemoryKind.UNIFIED
    assert snapshot.total_bytes == 32 * GIB
    assert len(snapshot.accelerators) == 1
    assert snapshot.accelerators[0].shared is True
    runner.assert_not_called()


def test_linux_keeps_valid_ram_when_nvidia_output_is_malformed() -> None:
    """A malformed optional probe must not discard independently valid RAM."""
    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=64 * GIB,
            available=40 * GIB,
            run_command=lambda *_args: CommandResult(0, b"bad,row\n", None),
        )
    )

    assert snapshot.system_state is SystemMemoryState.OBSERVED
    assert snapshot.accelerator_state is AcceleratorState.NOT_OBSERVED
    assert snapshot.total_bytes == 64 * GIB


def test_darwin_non_arm_reports_explicit_empty_partial_accelerator_state() -> None:
    """A native probe must not be invented merely to satisfy a partial-state shape."""
    runner = Mock()

    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="darwin",
            architecture="x86_64",
            total=32 * GIB,
            available=18 * GIB,
            run_command=runner,
        )
    )

    assert snapshot.system_state is SystemMemoryState.OBSERVED
    assert snapshot.accelerator_state is AcceleratorState.PARTIAL
    assert snapshot.accelerators == ()
    assert snapshot.accelerator_reason is ProbeReason.UNSUPPORTED_PLATFORM
    runner.assert_not_called()


@pytest.mark.parametrize(
    ("virtual_memory", "expected_state", "expected_reason"),
    [
        (
            lambda: (_ for _ in ()).throw(PermissionError("private-host")),
            SystemMemoryState.PERMISSION_DENIED,
            ProbeReason.PERMISSION_DENIED,
        ),
        (
            lambda: (_ for _ in ()).throw(RuntimeError("private-host")),
            SystemMemoryState.UNAVAILABLE,
            ProbeReason.MEMORY_UNAVAILABLE,
        ),
    ],
)
def test_system_probe_maps_raw_failures_to_fixed_reasons(
    virtual_memory: Callable[[], object],
    expected_state: SystemMemoryState,
    expected_reason: ProbeReason,
) -> None:
    """Propagating exception text would leak machine-specific information."""
    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=1,
            available=1,
            run_command=Mock(),
            virtual_memory=virtual_memory,
        )
    )

    assert snapshot.system_state is expected_state
    assert snapshot.system_reason is expected_reason
    assert "private-host" not in repr(snapshot)


def test_unsupported_platform_never_reads_system_or_accelerator_sources() -> None:
    """Adding permissive fallback probing would widen the supported trust boundary."""
    memory = Mock()
    runner = Mock()

    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="freebsd",
            architecture="amd64",
            total=1,
            available=1,
            run_command=runner,
            virtual_memory=memory,
        )
    )

    assert snapshot.platform == "other"
    assert snapshot.system_state is SystemMemoryState.UNSUPPORTED
    assert snapshot.accelerator_state is AcceleratorState.UNSUPPORTED
    memory.assert_not_called()
    runner.assert_not_called()


def test_invalid_available_memory_preserves_valid_total_as_partial() -> None:
    """Discarding total RAM would unnecessarily remove stable capacity evidence."""
    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=16 * GIB,
            available=17 * GIB,
            run_command=lambda *_args: CommandResult(1, b"", None),
        )
    )

    assert snapshot.system_state is SystemMemoryState.PARTIAL
    assert snapshot.total_bytes == 16 * GIB
    assert snapshot.available_bytes is None
    assert snapshot.system_reason is ProbeReason.INVALID_MEMORY_VALUE


def test_missing_available_memory_preserves_valid_total_as_partial() -> None:
    """Reading a missing available field must not discard an already valid total."""
    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=1,
            available=1,
            virtual_memory=lambda: SimpleNamespace(total=16 * GIB),
            run_command=lambda *_args: CommandResult(1, b"", None),
        )
    )

    assert snapshot.system_state is SystemMemoryState.PARTIAL
    assert snapshot.total_bytes == 16 * GIB
    assert snapshot.available_bytes is None
    assert snapshot.system_reason is ProbeReason.MEMORY_UNAVAILABLE


def test_raising_available_memory_preserves_valid_total_as_partial() -> None:
    """An available-property failure must not erase independently valid total RAM."""

    class Memory:
        total = 16 * GIB

        @property
        def available(self) -> int:
            raise RuntimeError("private available failure")

    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=1,
            available=1,
            virtual_memory=Memory,
            run_command=lambda *_args: CommandResult(1, b"", None),
        )
    )

    assert snapshot.system_state is SystemMemoryState.PARTIAL
    assert snapshot.total_bytes == 16 * GIB
    assert snapshot.available_bytes is None
    assert snapshot.system_reason is ProbeReason.MEMORY_UNAVAILABLE


@pytest.mark.parametrize(
    ("memory_error", "expected_state", "expected_reason"),
    [
        (
            PermissionError("private RAM denial"),
            SystemMemoryState.PERMISSION_DENIED,
            ProbeReason.PERMISSION_DENIED,
        ),
        (
            RuntimeError("private RAM failure"),
            SystemMemoryState.UNAVAILABLE,
            ProbeReason.MEMORY_UNAVAILABLE,
        ),
    ],
)
def test_unavailable_ram_retains_independently_observed_nvidia(
    memory_error: Exception,
    expected_state: SystemMemoryState,
    expected_reason: ProbeReason,
) -> None:
    """A system-memory failure must not suppress valid optional accelerator facts."""
    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=1,
            available=1,
            virtual_memory=lambda: (_ for _ in ()).throw(memory_error),
            run_command=lambda *_args: CommandResult(0, b"0, GPU, 8192\n", None),
        )
    )

    assert snapshot.system_state is expected_state
    assert snapshot.system_reason is expected_reason
    assert snapshot.total_bytes is None
    assert snapshot.memory_kind is MemoryKind.UNKNOWN
    assert snapshot.accelerator_state is AcceleratorState.OBSERVED
    assert snapshot.accelerators[0].total_bytes == 8 * GIB


def test_linux_retains_separate_nvidia_and_amd_observations() -> None:
    """Returning after NVIDIA would omit supported AMD DRM evidence on mixed systems."""
    card = Path("/sys/class/drm/card0")

    def read(path: Path, _limit: int) -> bytes:
        return b"0x1002\n" if path.name == "vendor" else b"8589934592\n"

    def resolve(path: Path) -> Path:
        if path == LINUX_NVIDIA_SMI:
            return path
        return Path("/sys/devices/pci/card0")

    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=64 * GIB,
            available=40 * GIB,
            run_command=lambda *_args: CommandResult(0, b"0, NVIDIA GPU, 8192\n", None),
            resolve_path=resolve,
            read_bounded=read,
            drm_cards=lambda: (card,),
        )
    )

    assert snapshot.accelerator_state is AcceleratorState.OBSERVED
    assert [item.source for item in snapshot.accelerators] == [
        AcceleratorSource.NVIDIA_SMI,
        AcceleratorSource.LINUX_DRM,
    ]
    assert [item.total_bytes for item in snapshot.accelerators] == [8 * GIB, 8 * GIB]


def test_linux_nvidia_fact_survives_failed_amd_branch_as_partial() -> None:
    """An enabled DRM failure must mark retained NVIDIA evidence as partial."""

    def resolve(path: Path) -> Path:
        return path if path == LINUX_NVIDIA_SMI else Path("/tmp/escaped")

    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=64 * GIB,
            available=40 * GIB,
            run_command=lambda *_args: CommandResult(0, b"0, NVIDIA GPU, 8192\n", None),
            resolve_path=resolve,
            drm_cards=lambda: (Path("/sys/class/drm/card0"),),
        )
    )

    assert snapshot.accelerator_state is AcceleratorState.PARTIAL
    assert len(snapshot.accelerators) == 1
    assert snapshot.accelerators[0].source is AcceleratorSource.NVIDIA_SMI
    assert snapshot.accelerator_reason is ProbeReason.SYSFS_UNTRUSTED_PATH


def test_linux_mixed_accelerators_enforce_aggregate_sixteen_device_cap() -> None:
    """Independent branch caps must not permit more than 16 aggregate observations."""
    nvidia_rows = b"".join(
        f"{index}, NVIDIA-{index}, 1024\n".encode() for index in range(10)
    )
    cards = tuple(Path(f"/sys/class/drm/card{index}") for index in range(7))

    def resolve(path: Path) -> Path:
        if path == LINUX_NVIDIA_SMI:
            return path
        return Path("/sys/devices/pci") / path.parent.name

    def read(path: Path, _limit: int) -> bytes:
        return b"0x1002\n" if path.name == "vendor" else b"1073741824\n"

    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=64 * GIB,
            available=40 * GIB,
            run_command=lambda *_args: CommandResult(0, nvidia_rows, None),
            resolve_path=resolve,
            read_bounded=read,
            drm_cards=lambda: cards,
        )
    )

    assert len(snapshot.accelerators) == 16
    assert snapshot.accelerator_state is AcceleratorState.PARTIAL
    assert snapshot.accelerator_reason is ProbeReason.TOO_MANY_DEVICES


def test_linux_mixed_accelerator_labels_remain_unique_and_nonthrowing() -> None:
    """A product name matching the fixed DRM label must not invalidate both facts."""

    def resolve(path: Path) -> Path:
        return path if path == LINUX_NVIDIA_SMI else Path("/sys/devices/pci/card0")

    def read(path: Path, _limit: int) -> bytes:
        return b"0x1002\n" if path.name == "vendor" else b"1073741824\n"

    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=64 * GIB,
            available=40 * GIB,
            run_command=lambda *_args: CommandResult(
                0, b"0, AMD DRM-reported VRAM 1, 1024\n", None
            ),
            resolve_path=resolve,
            read_bounded=read,
            drm_cards=lambda: (Path("/sys/class/drm/card0"),),
        )
    )

    assert len(snapshot.accelerators) == 2
    assert [item.label for item in snapshot.accelerators] == [
        "AMD DRM-reported VRAM 1",
        "AMD DRM-reported VRAM 1 #2",
    ]


def test_linux_nvidia_uses_only_fixed_path_argv_and_limits() -> None:
    """Allowing PATH/config/remote input would execute an untrusted program."""
    calls: list[tuple[Path, tuple[str, ...], float, int]] = []

    def runner(
        executable: Path, argv: tuple[str, ...], timeout: float, limit: int
    ) -> CommandResult:
        calls.append((executable, argv, timeout, limit))
        return CommandResult(0, b"0, NVIDIA RTX 4090, 24576\n", None)

    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=64 * GIB,
            available=40 * GIB,
            run_command=runner,
        )
    )

    assert calls == [
        (LINUX_NVIDIA_SMI, NVIDIA_ARGV, COMMAND_TIMEOUT_SECONDS, 64 * 1024)
    ]
    assert snapshot.accelerator_state is AcceleratorState.OBSERVED
    assert snapshot.accelerators[0].total_bytes == 24 * GIB


def test_windows_uses_second_fixed_nvidia_path_when_first_is_absent() -> None:
    """Windows discovery must not fall through to PATH after fixed candidates."""
    trusted_stat = SimpleNamespace(
        st_mode=stat.S_IFREG | 0o755,
        st_uid=0,
        st_file_attributes=0,
    )
    calls: list[Path] = []

    def lstat(path: Path) -> object:
        if path == WINDOWS_NVIDIA_SMI[0]:
            raise FileNotFoundError
        return trusted_stat

    def runner(
        executable: Path, _argv: tuple[str, ...], _timeout: float, _limit: int
    ) -> CommandResult:
        calls.append(executable)
        return CommandResult(0, b"0, NVIDIA GPU, 8192\n", None)

    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="windows",
            architecture="amd64",
            total=32 * GIB,
            available=16 * GIB,
            run_command=runner,
            lstat_path=lstat,
        )
    )

    assert calls == [WINDOWS_NVIDIA_SMI[1]]
    assert snapshot.accelerator_state is AcceleratorState.OBSERVED
    assert snapshot.accelerators[0].total_bytes == 8 * GIB


def test_nvidia_duplicate_names_are_disambiguated_without_device_identifiers() -> None:
    """Two identical board names must not crash the bounded snapshot validator."""
    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=64 * GIB,
            available=40 * GIB,
            run_command=lambda *_args: CommandResult(
                0,
                "0, 同型 GPU, 8192\n1, 同型 GPU, 8192\n".encode(),
                None,
            ),
        )
    )

    assert [item.label for item in snapshot.accelerators] == [
        "同型 GPU",
        "同型 GPU #2",
    ]


def test_accelerator_runner_exception_is_fixed_and_nonthrowing() -> None:
    """A future runner failure must not propagate host-specific exception text."""
    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=64 * GIB,
            available=40 * GIB,
            run_command=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("private GPU error")
            ),
        )
    )

    assert snapshot.system_state is SystemMemoryState.OBSERVED
    assert snapshot.accelerator_state is AcceleratorState.NOT_OBSERVED
    assert snapshot.accelerator_reason is ProbeReason.COMMAND_FAILED
    assert "private GPU error" not in repr(snapshot)


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(return_code=0, output=b"0, GPU, 1\n", reason=None),
        CommandResult(0, "not-bytes", None),
        CommandResult(0, b"0, GPU, 1\n", "not-a-reason"),
    ],
)
def test_malformed_injected_command_result_fails_closed(result: object) -> None:
    """A broken injected runner must not make observation throw or trust bad values."""
    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=64 * GIB,
            available=40 * GIB,
            run_command=lambda *_args: result,  # type: ignore[return-value]
        )
    )

    assert snapshot.accelerators == ()
    assert snapshot.accelerator_reason is ProbeReason.COMMAND_FAILED


@pytest.mark.parametrize(
    ("output", "expected_reason"),
    [
        (b"0, A, 1\n0, B, 2\n", ProbeReason.DUPLICATE_DEVICE),
        (
            b"".join(f"{index}, GPU-{index}, 1\n".encode() for index in range(17)),
            ProbeReason.TOO_MANY_DEVICES,
        ),
        (b"0, GPU, 0\n", ProbeReason.MALFORMED_OUTPUT),
        (
            f"0, GPU, {MAX_INPUT_BYTES // MIB + 1}\n".encode(),
            ProbeReason.MALFORMED_OUTPUT,
        ),
        (b"0, GPU\n", ProbeReason.MALFORMED_OUTPUT),
        (b"0, GPU, " + b"9" * 5000 + b"\n", ProbeReason.MALFORMED_OUTPUT),
    ],
)
def test_nvidia_parser_rejects_untrusted_device_rows(
    output: bytes, expected_reason: ProbeReason
) -> None:
    """Weak row validation could inject duplicate, excessive, or invalid facts."""
    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=64 * GIB,
            available=40 * GIB,
            run_command=lambda *_args: CommandResult(0, output, None),
        )
    )

    assert snapshot.accelerators == ()
    assert snapshot.accelerator_state is AcceleratorState.NOT_OBSERVED
    assert snapshot.accelerator_reason is expected_reason


def test_injected_runner_output_is_rebounded_before_parsing() -> None:
    """An injected or future runner must not bypass the 64-KiB boundary."""
    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=64 * GIB,
            available=40 * GIB,
            run_command=lambda *_args: CommandResult(
                0, b"x" * (MAX_COMMAND_OUTPUT_BYTES + 1), None
            ),
        )
    )

    assert snapshot.accelerators == ()
    assert snapshot.accelerator_reason is ProbeReason.OUTPUT_TOO_LARGE


@pytest.mark.parametrize(
    ("mode", "uid", "attributes"),
    [
        (stat.S_IFLNK | 0o777, 0, 0),
        (stat.S_IFREG | 0o775, 0, 0),
        (stat.S_IFREG | 0o755, 501, 0),
        (stat.S_IFREG | 0o755, 0, 0x400),
    ],
)
def test_untrusted_nvidia_path_is_never_executed(
    mode: int, uid: int, attributes: int
) -> None:
    """Removing any path trust check could execute a replaced binary."""
    runner = Mock()
    path_stat = SimpleNamespace(st_mode=mode, st_uid=uid, st_file_attributes=attributes)
    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=64 * GIB,
            available=40 * GIB,
            run_command=runner,
            lstat_path=lambda _path: path_stat,
        )
    )

    assert snapshot.accelerator_reason is ProbeReason.UNTRUSTED_EXECUTABLE
    runner.assert_not_called()


def test_linux_untrusted_parent_or_resolved_target_is_never_executed() -> None:
    """Trusting only the file metadata leaves its directory or link target replaceable."""
    runner = Mock()
    trusted = SimpleNamespace(
        st_mode=stat.S_IFREG | 0o755,
        st_uid=0,
        st_file_attributes=0,
    )
    untrusted_parent = SimpleNamespace(
        st_mode=stat.S_IFDIR | 0o775,
        st_uid=0,
        st_file_attributes=0,
    )

    parent_snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=64 * GIB,
            available=40 * GIB,
            run_command=runner,
            lstat_path=lambda path: (
                trusted if path == LINUX_NVIDIA_SMI else untrusted_parent
            ),
            resolve_path=lambda path: path,
        )
    )
    resolved_snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=64 * GIB,
            available=40 * GIB,
            run_command=runner,
            lstat_path=lambda _path: trusted,
            resolve_path=lambda _path: Path("/opt/replaced-nvidia-smi"),
        )
    )

    assert parent_snapshot.accelerator_reason is ProbeReason.UNTRUSTED_EXECUTABLE
    assert resolved_snapshot.accelerator_reason is ProbeReason.UNTRUSTED_EXECUTABLE
    runner.assert_not_called()


def test_permission_denied_nvidia_and_sysfs_returns_accelerator_denied() -> None:
    """Access denial must remain independent from observed system RAM."""
    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=64 * GIB,
            available=40 * GIB,
            run_command=Mock(),
            lstat_path=lambda _path: (_ for _ in ()).throw(PermissionError()),
            drm_cards=lambda: (_ for _ in ()).throw(PermissionError()),
        )
    )

    assert snapshot.system_state is SystemMemoryState.OBSERVED
    assert snapshot.accelerator_state is AcceleratorState.PERMISSION_DENIED
    assert snapshot.accelerator_reason is ProbeReason.SYSFS_PERMISSION_DENIED


def test_windows_nvidia_permission_denied_has_independent_denied_state() -> None:
    """A denied optional Windows probe must not be mislabeled as merely absent."""
    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="windows",
            architecture="amd64",
            total=32 * GIB,
            available=16 * GIB,
            run_command=Mock(),
            lstat_path=lambda _path: (_ for _ in ()).throw(PermissionError()),
        )
    )

    assert snapshot.system_state is SystemMemoryState.OBSERVED
    assert snapshot.accelerator_state is AcceleratorState.PERMISSION_DENIED
    assert snapshot.accelerator_reason is ProbeReason.PERMISSION_DENIED


class _ChunkStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self._buffer = b"".join(chunks)
        self.max_requested = 0
        self.closed = False
        self.reader: threading.Thread | None = None

    def read(self, size: int) -> bytes:
        self.reader = threading.current_thread()
        self.max_requested = max(self.max_requested, size)
        chunk, self._buffer = self._buffer[:size], self._buffer[size:]
        return chunk

    def close(self) -> None:
        self.closed = True


class _BlockingStream:
    def __init__(self, release: threading.Event) -> None:
        self._release = release
        self.closed = False
        self.reader: threading.Thread | None = None

    def read(self, _size: int) -> bytes:
        self.reader = threading.current_thread()
        self._release.wait(1)
        return b""

    def close(self) -> None:
        self.closed = True
        self._release.set()


class _FakeProcess:
    def __init__(
        self,
        stdout: object,
        *,
        survive_terminate: bool,
        return_code: int = 0,
    ) -> None:
        self.stdout = stdout
        self.returncode = return_code
        self.survive_terminate = survive_terminate
        self.events: list[object] = []
        self.release = threading.Event()

    def terminate(self) -> None:
        self.events.append("terminate")

    def kill(self) -> None:
        self.events.append("kill")
        self.release.set()

    def wait(self, timeout: float | None = None) -> int:
        self.events.append(("wait", timeout))
        if timeout == TERMINATE_GRACE_SECONDS and self.survive_terminate:
            raise subprocess.TimeoutExpired("private-command", timeout)
        self.release.set()
        return self.returncode


class _LateExitProcess(_FakeProcess):
    def wait(self, timeout: float | None = None) -> int:
        self.events.append(("wait", timeout))
        if timeout is not None:
            raise subprocess.TimeoutExpired("private-command", timeout)
        self.release.set()
        return self.returncode


class _UnstartableThread:
    def start(self) -> None:
        raise RuntimeError("private thread failure")


def test_bounded_runner_rejects_oversize_before_full_accumulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A whole-stream read would permit unbounded child output in memory."""
    stream = _ChunkStream([b"x" * (MAX_COMMAND_OUTPUT_BYTES + 4 * 8192)])
    process = _FakeProcess(stream, survive_terminate=False)
    popen = Mock(return_value=process)
    monkeypatch.setattr(probe.subprocess, "Popen", popen)

    result = probe._run_bounded_command(
        LINUX_NVIDIA_SMI, NVIDIA_ARGV, 2.0, MAX_COMMAND_OUTPUT_BYTES
    )

    assert result == CommandResult(None, b"", ProbeReason.OUTPUT_TOO_LARGE)
    assert stream.max_requested <= 8192
    assert process.events == ["terminate", ("wait", TERMINATE_GRACE_SECONDS)]
    assert stream.closed is True
    assert stream.reader is not None
    stream.reader.join(timeout=0.1)
    assert stream.reader.is_alive() is False
    popen.assert_called_once_with(
        [str(LINUX_NVIDIA_SMI), *NVIDIA_ARGV],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
    )


def test_bounded_runner_reaps_child_when_reader_cannot_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local thread-start failure must not leak the already-created child or pipe."""
    stream = _ChunkStream([b"output"])
    process = _FakeProcess(stream, survive_terminate=False)
    monkeypatch.setattr(probe.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(
        probe.threading,
        "Thread",
        lambda **_kwargs: _UnstartableThread(),
    )

    result = probe._run_bounded_command(
        LINUX_NVIDIA_SMI, NVIDIA_ARGV, 2.0, MAX_COMMAND_OUTPUT_BYTES
    )

    assert result == CommandResult(None, b"", ProbeReason.COMMAND_FAILED)
    assert process.events == ["terminate", ("wait", TERMINATE_GRACE_SECONDS)]
    assert stream.closed is True


def test_bounded_runner_timeout_terminates_kills_and_reaps_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting the final kill/wait can leak a timed-out child process."""
    process = _FakeProcess(object(), survive_terminate=True)
    stream = _BlockingStream(process.release)
    process.stdout = stream
    monkeypatch.setattr(probe.subprocess, "Popen", Mock(return_value=process))

    result = probe._run_bounded_command(
        LINUX_NVIDIA_SMI, NVIDIA_ARGV, 0.01, MAX_COMMAND_OUTPUT_BYTES
    )

    assert result == CommandResult(None, b"", ProbeReason.COMMAND_TIMEOUT)
    assert process.events == [
        "terminate",
        ("wait", TERMINATE_GRACE_SECONDS),
        "kill",
        ("wait", None),
    ]
    assert stream.closed is True
    assert stream.reader is not None
    stream.reader.join(timeout=0.1)
    assert stream.reader.is_alive() is False


def test_bounded_runner_returns_nonzero_without_raw_output_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nonzero child must be reaped and represented by a fixed code upstream."""
    process = _FakeProcess(
        _ChunkStream([b"private stderr"]),
        survive_terminate=False,
        return_code=7,
    )
    monkeypatch.setattr(probe.subprocess, "Popen", Mock(return_value=process))

    result = probe._run_bounded_command(
        LINUX_NVIDIA_SMI, NVIDIA_ARGV, 2.0, MAX_COMMAND_OUTPUT_BYTES
    )

    assert result.return_code == 7
    assert result.output == b"private stderr"
    assert result.reason is None
    assert len(process.events) == 1
    assert process.events[0][0] == "wait"
    assert 0 < process.events[0][1] <= 2.0


def test_bounded_runner_times_out_if_stdout_closes_before_process_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Waiting without the remaining deadline could block after bounded output ends."""
    process = _LateExitProcess(_ChunkStream([]), survive_terminate=True)
    monkeypatch.setattr(probe.subprocess, "Popen", Mock(return_value=process))

    result = probe._run_bounded_command(
        LINUX_NVIDIA_SMI, NVIDIA_ARGV, 0.01, MAX_COMMAND_OUTPUT_BYTES
    )

    assert result.reason is ProbeReason.COMMAND_TIMEOUT
    assert process.events[-4:] == [
        "terminate",
        ("wait", TERMINATE_GRACE_SECONDS),
        "kill",
        ("wait", None),
    ]


def test_linux_amd_drm_observation_is_bounded_and_does_not_use_intel() -> None:
    """Widening accepted vendors would claim unsupported kernel contracts."""
    cards = tuple(Path(f"/sys/class/drm/card{index}") for index in range(18))
    reads: list[tuple[Path, int]] = []

    def resolve(path: Path) -> Path:
        return Path("/sys/devices/pci/drm") / path.parent.name

    def read(path: Path, limit: int) -> bytes:
        reads.append((path, limit))
        if path.name == "vendor":
            return b"0x1002\n" if path.parent.name == "card0" else b"0x8086\n"
        return b"8589934592\n"

    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=64 * GIB,
            available=40 * GIB,
            run_command=lambda *_args: CommandResult(
                None, b"", ProbeReason.EXECUTABLE_NOT_FOUND
            ),
            lstat_path=lambda _path: (_ for _ in ()).throw(FileNotFoundError()),
            resolve_path=resolve,
            read_bounded=read,
            drm_cards=lambda: cards,
        )
    )

    assert snapshot.accelerator_state is AcceleratorState.OBSERVED
    assert len(snapshot.accelerators) == 1
    assert snapshot.accelerators[0].vendor == "amd"
    assert snapshot.accelerators[0].label == "AMD DRM-reported VRAM 1"
    assert snapshot.accelerators[0].total_bytes == 8 * GIB
    assert snapshot.accelerators[0].source is AcceleratorSource.LINUX_DRM
    assert all(limit == 64 for _path, limit in reads)
    assert len({path.parent.name for path, _limit in reads}) <= 16


def test_linux_drm_rejects_resolution_outside_sys_devices() -> None:
    """Following a class-device link outside sysfs would cross the trust boundary."""
    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=64 * GIB,
            available=40 * GIB,
            run_command=lambda *_args: CommandResult(
                None, b"", ProbeReason.EXECUTABLE_NOT_FOUND
            ),
            lstat_path=lambda _path: (_ for _ in ()).throw(FileNotFoundError()),
            resolve_path=lambda _path: Path("/tmp/escaped"),
            drm_cards=lambda: (Path("/sys/class/drm/card0"),),
        )
    )

    assert snapshot.accelerators == ()
    assert snapshot.accelerator_reason is ProbeReason.SYSFS_UNTRUSTED_PATH


def test_valid_amd_with_malformed_sibling_is_partial() -> None:
    """A valid fact must survive while exposing that another enabled branch failed."""
    cards = (Path("/sys/class/drm/card0"), Path("/sys/class/drm/card1"))

    def resolve(path: Path) -> Path:
        return Path("/sys/devices/pci") / path.parent.name

    def read(path: Path, _limit: int) -> bytes:
        if path.name == "vendor":
            return b"0x1002\n"
        return b"8589934592\n" if path.parent.name == "card0" else b"not-digits\n"

    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=64 * GIB,
            available=40 * GIB,
            run_command=lambda *_args: CommandResult(
                None, b"", ProbeReason.EXECUTABLE_NOT_FOUND
            ),
            lstat_path=lambda _path: (_ for _ in ()).throw(FileNotFoundError()),
            resolve_path=resolve,
            read_bounded=read,
            drm_cards=lambda: cards,
        )
    )

    assert snapshot.accelerator_state is AcceleratorState.PARTIAL
    assert len(snapshot.accelerators) == 1
    assert snapshot.accelerator_reason is ProbeReason.SYSFS_MALFORMED


def test_production_drm_discovery_accepts_only_exact_card_number_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connector entries matching a loose glob must never become device probes."""
    candidates = [
        Path("/sys/class/drm/card0"),
        Path("/sys/class/drm/card0-HDMI-A-1"),
        Path("/sys/class/drm/cardx"),
        Path("/sys/class/drm/card12"),
    ]
    monkeypatch.setattr(
        probe.Path,
        "glob",
        lambda _self, _pattern: iter(candidates),
    )

    assert probe._drm_cards() == (candidates[0], candidates[3])


def test_production_drm_discovery_stops_after_sixteen_exact_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Materializing every glob result would violate bounded discovery."""

    def candidates() -> object:
        for index in range(16):
            yield Path(f"/sys/class/drm/card{index}")
        raise AssertionError("discovery read beyond the 16-card cap")

    monkeypatch.setattr(
        probe.Path,
        "glob",
        lambda _self, _pattern: candidates(),
    )

    assert len(probe._drm_cards()) == 16


def test_probe_never_logs_or_exposes_raw_machine_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Logging a source exception would persist private host details."""
    secret = "host=private UUID=abc PCI=0000:01:00.0"

    snapshot = observe_machine_memory(
        sources=_sources(
            platform_name="linux",
            architecture="x86_64",
            total=1,
            available=1,
            run_command=Mock(),
            virtual_memory=lambda: (_ for _ in ()).throw(RuntimeError(secret)),
        )
    )

    assert secret not in repr(snapshot)
    assert secret not in caplog.text
