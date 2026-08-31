"""Admission-gated POSIX controlling-PTY terminal backend."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import contextlib
from dataclasses import dataclass
import errno
import fcntl
import json
import os
from pathlib import Path
import secrets
import select
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
from typing import Any

import psutil

from .contracts import (
    AdmissionGate,
    BackendIdentity,
    CleanupAttempt,
    CleanupProof,
    CleanupSchedule,
    MAX_COLUMNS,
    MAX_IO_CHUNK_BYTES,
    MAX_PENDING_INPUT_BYTES,
    MAX_PENDING_OUTPUT_BYTES,
    MAX_ROWS,
    MIN_COLUMNS,
    MIN_ROWS,
    TerminalLaunchRequest,
)
from .launch import (
    ShellChoice,
    build_terminal_environment,
    discover_shell_choices,
    resolve_shell_choice,
)


_LAUNCH_TIMEOUT_SECONDS = 5.0
_PROCESS_POLL_SECONDS = 0.01
_ZERO_SCAN_INTERVAL_SECONDS = 0.05
_SPAWN_LOCK = threading.Lock()
_STDERR_FALLBACK: Any | None = None


@dataclass(frozen=True, slots=True)
class PosixProcessIdentity:
    """PID/birth plus session and current process-group identity.

    Attributes:
        pid: Numeric process identifier.
        birth_time: Kernel-backed process creation time.
        sid: Current POSIX session identifier.
        initial_pgid: Process group observed for this snapshot. For the admitted
            launcher this is the immutable initial process group.
    """

    pid: int
    birth_time: float
    sid: int
    initial_pgid: int

    def __post_init__(self) -> None:
        if type(self.pid) is not int or self.pid <= 0:
            raise ValueError("pid must be a positive integer")
        if not isinstance(self.birth_time, (int, float)) or isinstance(
            self.birth_time, bool
        ):
            raise TypeError("birth_time must be numeric")
        if type(self.sid) is not int or self.sid <= 0:
            raise ValueError("sid must be a positive integer")
        if type(self.initial_pgid) is not int or self.initial_pgid <= 0:
            raise ValueError("process group must be a positive integer")


@dataclass(frozen=True, slots=True)
class OwnershipScan:
    """One complete or incomplete process ownership observation.

    Attributes:
        owned: Revalidated same-session or tracked descendants.
        observed: Candidate-group population used to prove exclusivity.
        complete: Whether admitted ownership enumeration and identity reads succeeded.
        group_membership_complete: Whether global group-membership enumeration
            was complete enough to allow broad group signalling.
    """

    owned: tuple[PosixProcessIdentity, ...]
    observed: tuple[PosixProcessIdentity, ...]
    complete: bool
    group_membership_complete: bool = True


@dataclass(frozen=True, slots=True)
class SignalPlan:
    """Validated process groups and individual identities to signal.

    Attributes:
        group_ids: Exclusively owned groups with a same-birth live leader.
        individuals: Processes requiring immediate PID/birth revalidation.
    """

    group_ids: tuple[int, ...] = ()
    individuals: tuple[PosixProcessIdentity, ...] = ()


def _plan_signals(scan: OwnershipScan) -> SignalPlan:
    """Plan broad group signals only from complete exclusive membership.

    Args:
        scan: Current ownership and full process-group observation.

    Returns:
        Safe group IDs plus individual fallbacks for every other owned process.
    """
    owned = tuple(sorted(scan.owned, key=lambda item: item.pid))
    if not scan.complete or not scan.group_membership_complete:
        return SignalPlan(individuals=owned)

    owned_keys = {(item.pid, item.birth_time) for item in owned}
    observed_by_pid = {item.pid: item for item in scan.observed}
    safe_groups: list[int] = []
    covered: set[tuple[int, float]] = set()
    for group_id in sorted({item.initial_pgid for item in owned}):
        leader = next((item for item in owned if item.pid == group_id), None)
        current_leader = observed_by_pid.get(group_id)
        members = tuple(item for item in scan.observed if item.initial_pgid == group_id)
        if leader is None or current_leader is None or not members:
            continue
        if current_leader.birth_time != leader.birth_time:
            continue
        member_keys = {(item.pid, item.birth_time) for item in members}
        if not member_keys <= owned_keys:
            continue
        safe_groups.append(group_id)
        covered.update(member_keys)
    individuals = tuple(
        item for item in owned if (item.pid, item.birth_time) not in covered
    )
    return SignalPlan(tuple(safe_groups), individuals)


def _stream_fileno(stream: Any) -> int:
    try:
        descriptor = stream.fileno()
    except Exception:
        return -1
    return descriptor if isinstance(descriptor, int) and descriptor >= 0 else -1


def _fd_backed_stderr() -> Any:
    real = sys.__stderr__
    if real is not None and _stream_fileno(real) >= 0:
        return real
    global _STDERR_FALLBACK
    if _STDERR_FALLBACK is None:
        _STDERR_FALLBACK = open(os.devnull, "w")
    return _STDERR_FALLBACK


def _stderr_context() -> Any:
    if _stream_fileno(sys.stderr) >= 0:
        return contextlib.nullcontext()
    return contextlib.redirect_stderr(_fd_backed_stderr())


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _read_pipe(fd: int, *, deadline: float, maximum: int) -> bytes:
    payload = bytearray()
    while len(payload) < maximum:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("POSIX terminal startup failed")
        readable, _, _ = select.select([fd], [], [], remaining)
        if not readable:
            raise RuntimeError("POSIX terminal startup failed")
        chunk = os.read(fd, min(4096, maximum - len(payload)))
        if not chunk:
            return bytes(payload)
        payload.extend(chunk)
        if b"\n" in payload:
            line, _, remainder = bytes(payload).partition(b"\n")
            if remainder:
                raise RuntimeError("POSIX terminal startup failed")
            return line
    raise RuntimeError("POSIX terminal startup failed")


def _safe_close(fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def _set_window_size(fd: int, columns: int, rows: int) -> None:
    packed = struct.pack("HHHH", rows, columns, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, packed)


def _validate_dimensions(columns: int, rows: int) -> None:
    if type(columns) is not int or not MIN_COLUMNS <= columns <= MAX_COLUMNS:
        raise ValueError("terminal columns are invalid")
    if type(rows) is not int or not MIN_ROWS <= rows <= MAX_ROWS:
        raise ValueError("terminal rows are invalid")


def _validate_environment(source: Mapping[str, str]) -> dict[str, str]:
    if len(source) > 128:
        raise ValueError("terminal environment is invalid")
    environment = dict(source)
    if not all(
        isinstance(key, str)
        and key
        and "=" not in key
        and "\0" not in key
        and isinstance(value, str)
        and "\0" not in value
        for key, value in environment.items()
    ):
        raise ValueError("terminal environment is invalid")
    return environment


class PosixTerminalBackend:
    """Own one admitted POSIX terminal generation and its cleanup proof."""

    def __init__(
        self,
        *,
        environment_factory: Callable[[], Mapping[str, str]] | None = None,
        scan_wrapper: Callable[..., OwnershipScan] | None = None,
        shell_choices_factory: Callable[[], tuple[ShellChoice, ...]] | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialize an unstarted POSIX backend.

        Args:
            environment_factory: Terminal-specific scrubbed environment reader.
            scan_wrapper: Test seam that may narrow or deny a real ownership scan.
            shell_choices_factory: Code-owned POSIX shell discovery seam.
            monotonic_clock: Monotonic cleanup clock.
            sleep: Bounded cleanup wait primitive.
        """
        if os.name != "posix":
            raise OSError("POSIX terminal backend unavailable")
        self._environment_factory = environment_factory or build_terminal_environment
        self._scan_wrapper = scan_wrapper
        self._shell_choices_factory = shell_choices_factory or discover_shell_choices
        self._clock = monotonic_clock
        self._sleep = sleep
        self._state_lock = threading.RLock()
        self._io_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._scan_lock = threading.Lock()
        self._close_requested = threading.Event()
        self._input_pending = threading.Event()
        self._shell_reaped = threading.Event()
        self._monitor_stop = threading.Event()
        self._process: subprocess.Popen[bytes] | None = None
        self._master_fd: int | None = None
        self._identity: PosixProcessIdentity | None = None
        self._tracked: dict[int, float] = {}
        self._exit_code: int | None = None
        self._reap_count = 0
        self._pty_eof = False
        self._input_closed = False
        self._input_buffer = bytearray()
        self._output_buffer = bytearray()
        self._output_complete = True
        self._process_only_dead = False
        self._zero_scan_times: list[float] = []
        self._last_attempt_t0: float | None = None
        self._last_proof: CleanupProof | None = None

    def start(
        self,
        request: TerminalLaunchRequest,
        admission: AdmissionGate,
    ) -> BackendIdentity:
        """Launch a gated helper that becomes the admitted interactive shell.

        Args:
            request: Validated terminal launch values.
            admission: Parent authority decision and opaque session token.

        Returns:
            Backend identity carrying exactly the admission token.

        Raises:
            RuntimeError: If admission, helper identity, or exec cannot be proven.
            ValueError: If launch dimensions or directory are invalid.
        """
        with self._state_lock:
            if self._process is not None or self._master_fd is not None:
                raise RuntimeError("POSIX terminal startup failed")
        _validate_dimensions(request.columns, request.rows)
        start_directory = Path(request.start_directory)
        if not start_directory.is_absolute() or not start_directory.is_dir():
            raise ValueError("terminal start directory is invalid")
        choices = self._shell_choices_factory()
        shell = resolve_shell_choice(request.shell or "default", choices)
        environment = _validate_environment(self._environment_factory())
        admitted = (
            type(admission) is AdmissionGate
            and admission.admitted is True
            and isinstance(admission.token, str)
            and bool(admission.token)
            and len(admission.token) <= 1024
        )
        gate_token = admission.token if admitted else secrets.token_hex(16)
        config = (
            json.dumps(
                {
                    "argv": list(shell.argv),
                    "cwd": str(start_directory),
                    "environment": environment,
                    "executable": str(shell.executable),
                    "token": gate_token,
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        if len(config) > 64 * 1024:
            raise ValueError("terminal launch values are invalid")

        process: subprocess.Popen[bytes] | None = None
        descriptors: dict[str, int | None] = {
            "master": None,
            "slave": None,
            "config_read": None,
            "config_write": None,
            "gate_read": None,
            "gate_write": None,
            "report_read": None,
            "report_write": None,
            "status_read": None,
            "status_write": None,
        }
        reaper_started = False
        try:
            with _SPAWN_LOCK:
                with _stderr_context():
                    descriptors["master"], descriptors["slave"] = os.openpty()
                    _set_window_size(
                        int(descriptors["slave"]),
                        request.columns,
                        request.rows,
                    )
                    (
                        descriptors["config_read"],
                        descriptors["config_write"],
                    ) = os.pipe()
                    descriptors["gate_read"], descriptors["gate_write"] = os.pipe()
                    (
                        descriptors["report_read"],
                        descriptors["report_write"],
                    ) = os.pipe()
                    (
                        descriptors["status_read"],
                        descriptors["status_write"],
                    ) = os.pipe()
                    pass_fds = (
                        int(descriptors["slave"]),
                        int(descriptors["config_read"]),
                        int(descriptors["gate_read"]),
                        int(descriptors["report_write"]),
                        int(descriptors["status_write"]),
                    )
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "tldw_chatbook.Terminal.posix_launcher",
                            "--slave-fd",
                            str(descriptors["slave"]),
                            "--config-fd",
                            str(descriptors["config_read"]),
                            "--admission-fd",
                            str(descriptors["gate_read"]),
                            "--report-fd",
                            str(descriptors["report_write"]),
                            "--exec-status-fd",
                            str(descriptors["status_write"]),
                        ],
                        pass_fds=pass_fds,
                        close_fds=True,
                    )
            for name in (
                "slave",
                "config_read",
                "gate_read",
                "report_write",
                "status_write",
            ):
                _safe_close(descriptors[name])
                descriptors[name] = None
            _write_all(int(descriptors["config_write"]), config)
            _safe_close(descriptors["config_write"])
            descriptors["config_write"] = None
            deadline = time.monotonic() + _LAUNCH_TIMEOUT_SECONDS
            report = _read_pipe(
                int(descriptors["report_read"]),
                deadline=deadline,
                maximum=4096,
            )
            identity = self._validate_report(process, report)
            master_fd: int | None = None
            if admitted:
                self._revalidate_identity(identity)
                master_fd = int(descriptors["master"])
                flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
                fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                with self._state_lock:
                    self._process = process
                    self._master_fd = master_fd
                    self._identity = identity
                    self._tracked[identity.pid] = identity.birth_time
                try:
                    self._start_shell_reaper(process)
                except Exception:
                    with self._state_lock:
                        self._process = None
                        self._master_fd = None
                        self._identity = None
                        self._tracked.pop(identity.pid, None)
                    raise
                reaper_started = True
                descriptors["master"] = None
            decision = (
                json.dumps(
                    {
                        "admitted": admitted,
                        "token": gate_token if admitted else "",
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
                + b"\n"
            )
            _write_all(int(descriptors["gate_write"]), decision)
            _safe_close(descriptors["gate_write"])
            descriptors["gate_write"] = None
            status = _read_pipe(
                int(descriptors["status_read"]),
                deadline=deadline,
                maximum=1,
            )
            if not admitted:
                self._reap_unadmitted(process)
                raise RuntimeError("POSIX terminal admission failed")
            if status:
                raise RuntimeError("POSIX terminal startup failed")
            if master_fd is None:
                raise RuntimeError("POSIX terminal startup failed")
            self._start_owner_workers(process)
            return BackendIdentity(session_id=admission.token)
        except Exception:
            if process is not None:
                if reaper_started:
                    try:
                        self.request_priority_close()
                        self.cleanup(CleanupAttempt(time.monotonic()))
                    except Exception:
                        pass
                elif self._process is None:
                    self._reap_unadmitted(process)
            raise
        finally:
            for descriptor in descriptors.values():
                _safe_close(descriptor)

    def read(self, maximum: int = 64 * 1024) -> bytes | None:
        """Read one nonblocking bounded PTY chunk.

        Args:
            maximum: Maximum bytes to return, capped by the terminal contract.

        Returns:
            Bytes when available, ``None`` on backpressure, and ``b""`` at EOF.
        """
        if type(maximum) is not int or not 1 <= maximum <= MAX_IO_CHUNK_BYTES:
            raise ValueError("terminal read size is invalid")
        return self._read_master(maximum)

    def take_preserved_cleanup_output(self, maximum: int) -> bytes:
        """Transfer one bounded healthy-cleanup chunk after process proof and EOF.

        Args:
            maximum: Maximum bytes to transfer to the manager-owned output actor.

        Returns:
            Preserved bytes, or ``b""`` when none are available or transfer is not
            yet safe.
        """
        if type(maximum) is not int or not 1 <= maximum <= MAX_IO_CHUNK_BYTES:
            raise ValueError("terminal read size is invalid")
        with self._io_lock:
            with self._state_lock:
                transfer_allowed = (
                    self._process_only_dead and self._pty_eof and self._output_complete
                )
            if not transfer_allowed or not self._output_buffer:
                return b""
            payload = bytes(self._output_buffer[:maximum])
            del self._output_buffer[: len(payload)]
            return payload

    def write(self, data: bytes) -> None:
        """Accept one bounded input event to the PTY or ordered input queue.

        Args:
            data: Input bytes admitted by the bounded input actor.

        Raises:
            BlockingIOError: If bounded capacity cannot accept the complete event.
            OSError: If input is closed.
        """
        if not isinstance(data, bytes) or not data or len(data) > MAX_IO_CHUNK_BYTES:
            raise ValueError("terminal input is invalid")
        with self._io_lock:
            with self._state_lock:
                fd = self._master_fd
                if fd is None or self._input_closed:
                    raise OSError("terminal input is closed")
            self._flush_input_locked(fd)
            if len(self._input_buffer) + len(data) > MAX_PENDING_INPUT_BYTES:
                raise BlockingIOError("terminal input backpressure")
            if self._input_buffer:
                self._input_buffer.extend(data)
                self._input_pending.set()
                return
            try:
                written = os.write(fd, data)
            except BlockingIOError:
                written = 0
            if written < 0 or written > len(data):
                raise OSError("terminal input failed")
            if written < len(data):
                self._input_buffer.extend(data[written:])
                self._input_pending.set()

    def resize(self, columns: int, rows: int) -> None:
        """Resize the PTY and let the kernel notify its foreground process group.

        Args:
            columns: Validated terminal width.
            rows: Validated terminal height.
        """
        _validate_dimensions(columns, rows)
        with self._io_lock:
            with self._state_lock:
                fd = self._master_fd
                if fd is None:
                    raise OSError("terminal resize is unavailable")
            _set_window_size(fd, columns, rows)

    def request_priority_close(self) -> None:
        """Disable input and set the idempotent out-of-band cleanup signal."""
        with self._state_lock:
            self._input_closed = True
        self._close_requested.set()
        with self._io_lock:
            self._input_buffer.clear()
            self._input_pending.clear()
        self._input_pending.set()

    def cleanup(self, attempt: CleanupAttempt) -> CleanupProof:
        """Run identity-safe cleanup under absolute ADR-099 boundaries.

        Args:
            attempt: Monotonic attempt start shared with the manager.

        Returns:
            Separate process-death, PTY-EOF, and output-completeness evidence.
        """
        if type(attempt) is not CleanupAttempt:
            raise TypeError("attempt must be CleanupAttempt")
        with self._cleanup_lock:
            if self._last_attempt_t0 == attempt.t0 and self._last_proof is not None:
                return self._last_proof
            proof = self._run_cleanup(attempt)
            self._last_attempt_t0 = attempt.t0
            self._last_proof = proof
            return proof

    def cleanup_parser_failure(self, attempt: CleanupAttempt) -> CleanupProof:
        """Clean after parser failure without exposing untrusted PTY bytes.

        Args:
            attempt: Original manager cleanup attempt and absolute time origin.

        Returns:
            Process and EOF evidence with output completeness always false.
        """
        if type(attempt) is not CleanupAttempt:
            raise TypeError("attempt must be CleanupAttempt")
        with self._cleanup_lock:
            if self._last_attempt_t0 == attempt.t0 and self._last_proof is not None:
                return self._last_proof
            self._output_complete = False
            proof = self._run_cleanup(attempt, parser_failed=True)
            self._last_attempt_t0 = attempt.t0
            self._last_proof = proof
            return proof

    def cleanup_raw_drain(self, attempt: CleanupAttempt) -> CleanupProof:
        """Discard bounded bytes after parser failure and process-only proof.

        Args:
            attempt: Cleanup attempt whose absolute deadline bounds raw draining.

        Returns:
            Cleanup evidence with output completeness always false.
        """
        if type(attempt) is not CleanupAttempt:
            raise TypeError("attempt must be CleanupAttempt")
        with self._state_lock:
            process_only_dead = self._process_only_dead
        if not process_only_dead:
            return CleanupProof(
                process_dead=False,
                stream_closed=self._pty_eof,
                output_complete=False,
            )
        self._output_complete = False
        with self._io_lock:
            self._output_buffer.clear()
        deadline = attempt.t0 + CleanupSchedule().deadline_seconds
        if self._deadline_expired(deadline):
            return CleanupProof(
                process_dead=process_only_dead,
                stream_closed=self._pty_eof,
                output_complete=False,
            )
        while not self._pty_eof:
            if self._deadline_expired(deadline):
                break
            self._discard_cleanup_turn()
            if self._deadline_expired(deadline):
                break
            if not self._pty_eof:
                remaining = max(0.0, deadline - self._clock())
                self._sleep(min(_PROCESS_POLL_SECONDS, remaining))
        proof = CleanupProof(
            process_dead=process_only_dead,
            stream_closed=self._pty_eof,
            output_complete=False,
        )
        if proof.process_dead and proof.stream_closed:
            self._close_proven()
        return proof

    @property
    def identity_for_tests(self) -> PosixProcessIdentity:
        """Return the immutable admitted shell identity for focused tests."""
        with self._state_lock:
            identity = self._identity
        if identity is None:
            raise RuntimeError("POSIX terminal is not started")
        return identity

    @property
    def launcher_pid_for_tests(self) -> int:
        """Return the launcher PID, which remains the shell PID after exec."""
        return self.identity_for_tests.pid

    @property
    def master_is_nonblocking_for_tests(self) -> bool:
        """Return whether the retained PTY master has ``O_NONBLOCK``."""
        with self._state_lock:
            fd = self._master_fd
        if fd is None:
            return False
        return bool(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_NONBLOCK)

    @property
    def shell_reaped(self) -> bool:
        """Return whether the sole reaper observed exact shell exit."""
        return self._shell_reaped.is_set()

    @property
    def shell_reap_count_for_tests(self) -> int:
        """Return the number of completed authoritative shell waits."""
        with self._state_lock:
            return self._reap_count

    @property
    def zero_scan_times_for_tests(self) -> tuple[float, ...]:
        """Return monotonic timestamps used by stable zero-process proof."""
        with self._state_lock:
            return tuple(self._zero_scan_times)

    @property
    def default_scan_for_tests(self) -> Callable[..., OwnershipScan]:
        """Return the real ownership scanner for denial wrappers."""
        return self._default_scan

    def owned_processes_for_tests(self) -> tuple[PosixProcessIdentity, ...]:
        """Return only current PID/birth-owned identities for focused tests."""
        return self._scan().owned

    def managed_process_inventory_for_tests(self) -> tuple[Any, ...]:
        """Return manager-compatible PID/birth identities for RSS accounting."""
        from .session_manager import ManagedProcessIdentity

        return tuple(
            ManagedProcessIdentity(
                pid=identity.pid,
                birth_identity=repr(identity.birth_time),
            )
            for identity in self._scan().owned
        )

    def wait_for_shell_exit(self, *, timeout_seconds: float) -> int | None:
        """Wait boundedly for the sole reaper and return exact shell status.

        Args:
            timeout_seconds: Maximum wait duration.

        Returns:
            Exact shell exit code, or ``None`` if the wait expires.
        """
        if not isinstance(timeout_seconds, (int, float)) or isinstance(
            timeout_seconds, bool
        ):
            raise TypeError("timeout_seconds must be numeric")
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must not be negative")
        if not self._shell_reaped.wait(float(timeout_seconds)):
            return None
        with self._state_lock:
            return self._exit_code

    def _validate_report(
        self,
        process: subprocess.Popen[bytes],
        payload: bytes,
    ) -> PosixProcessIdentity:
        try:
            value = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise RuntimeError("POSIX terminal startup failed") from None
        if type(value) is not dict or set(value) != {
            "birth_time",
            "pgid",
            "pid",
            "sid",
        }:
            raise RuntimeError("POSIX terminal startup failed")
        try:
            identity = PosixProcessIdentity(
                pid=value["pid"],
                birth_time=value["birth_time"],
                sid=value["sid"],
                initial_pgid=value["pgid"],
            )
        except (TypeError, ValueError):
            raise RuntimeError("POSIX terminal startup failed") from None
        if (
            identity.pid != process.pid
            or identity.sid != identity.pid
            or identity.initial_pgid != identity.pid
        ):
            raise RuntimeError("POSIX terminal startup failed")
        self._revalidate_identity(identity)
        return identity

    @staticmethod
    def _revalidate_identity(identity: PosixProcessIdentity) -> None:
        try:
            birth_time = psutil.Process(identity.pid).create_time()
            sid = os.getsid(identity.pid)
        except (OSError, psutil.Error):
            raise RuntimeError("POSIX terminal startup failed") from None
        if birth_time != identity.birth_time or sid != identity.sid:
            raise RuntimeError("POSIX terminal startup failed")

    @staticmethod
    def _reap_unadmitted(process: subprocess.Popen[bytes]) -> None:
        try:
            process.wait(timeout=0.25)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=0.5)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass

    def _start_shell_reaper(self, process: subprocess.Popen[bytes]) -> None:
        threading.Thread(
            target=self._reap_shell,
            args=(process,),
            name=f"terminal-reaper-{process.pid}",
            daemon=True,
        ).start()

    def _start_owner_workers(self, process: subprocess.Popen[bytes]) -> None:
        threading.Thread(
            target=self._monitor_owned_processes,
            name=f"terminal-monitor-{process.pid}",
            daemon=True,
        ).start()
        threading.Thread(
            target=self._flush_pending_input,
            name=f"terminal-input-{process.pid}",
            daemon=True,
        ).start()

    def _reap_shell(self, process: subprocess.Popen[bytes]) -> None:
        exit_code = process.wait()
        with self._state_lock:
            self._exit_code = exit_code
            self._reap_count += 1
        self._shell_reaped.set()

    def _monitor_owned_processes(self) -> None:
        while not self._monitor_stop.wait(_PROCESS_POLL_SECONDS * 2):
            try:
                self._default_scan()
            except Exception:
                continue

    def _flush_pending_input(self) -> None:
        while not self._close_requested.is_set():
            if not self._input_pending.wait(_PROCESS_POLL_SECONDS):
                continue
            try:
                with self._io_lock:
                    with self._state_lock:
                        fd = self._master_fd
                        closed = self._input_closed
                    if fd is None or closed:
                        return
                    self._flush_input_locked(fd)
            except OSError:
                with self._state_lock:
                    self._input_closed = True
                with self._io_lock:
                    self._input_buffer.clear()
                    self._input_pending.clear()
                return
            if self._input_pending.is_set():
                self._close_requested.wait(_PROCESS_POLL_SECONDS)

    def _flush_input_locked(self, fd: int) -> None:
        if not self._input_buffer:
            self._input_pending.clear()
            return
        try:
            written = os.write(fd, self._input_buffer[:MAX_IO_CHUNK_BYTES])
        except BlockingIOError:
            self._input_pending.set()
            return
        if written < 0 or written > min(len(self._input_buffer), MAX_IO_CHUNK_BYTES):
            raise OSError("terminal input failed")
        if written:
            del self._input_buffer[:written]
        if self._input_buffer:
            self._input_pending.set()
        else:
            self._input_pending.clear()

    def _read_master(self, maximum: int) -> bytes | None:
        with self._io_lock:
            if self._output_buffer:
                payload = bytes(self._output_buffer[:maximum])
                del self._output_buffer[: len(payload)]
                return payload
            with self._state_lock:
                if self._pty_eof:
                    return b""
                fd = self._master_fd
                if fd is None:
                    return b""
            try:
                payload = os.read(fd, maximum)
            except BlockingIOError:
                return None
            except OSError as exc:
                if exc.errno != errno.EIO:
                    raise OSError("terminal output failed") from None
                payload = b""
            if not payload:
                with self._state_lock:
                    self._pty_eof = True
            return payload

    def _buffer_cleanup_turn(self) -> None:
        """Preserve at most one bounded PTY read for the healthy output path."""
        payload: bytes | None = None
        with self._io_lock:
            with self._state_lock:
                if self._pty_eof:
                    return
                fd = self._master_fd
                if fd is None:
                    return
            capacity = MAX_PENDING_OUTPUT_BYTES - len(self._output_buffer)
            if capacity <= 0:
                return
            try:
                payload = os.read(fd, min(MAX_IO_CHUNK_BYTES, capacity))
            except BlockingIOError:
                return
            except OSError as exc:
                if exc.errno != errno.EIO:
                    raise OSError("terminal output failed") from None
                payload = b""
            if payload:
                self._output_buffer.extend(payload)
                return
            with self._state_lock:
                self._pty_eof = True

    def _discard_cleanup_turn(self) -> None:
        """Discard at most one PTY read after explicit parser-failure fallback."""
        with self._io_lock:
            with self._state_lock:
                if self._pty_eof:
                    return
                fd = self._master_fd
                if fd is None:
                    return
            try:
                payload = os.read(fd, MAX_IO_CHUNK_BYTES)
            except BlockingIOError:
                return
            except OSError as exc:
                if exc.errno != errno.EIO:
                    raise OSError("terminal output failed") from None
                payload = b""
            if not payload:
                with self._state_lock:
                    self._pty_eof = True

    def _scan(self) -> OwnershipScan:
        try:
            if self._scan_wrapper is None:
                return self._default_scan()
            scan = self._scan_wrapper()
        except Exception:
            return OwnershipScan((), (), False)
        if type(scan) is not OwnershipScan:
            return OwnershipScan((), (), False)
        return scan

    def _deadline_expired(self, deadline: float) -> bool:
        return self._clock() >= deadline

    def _scan_before_deadline(self, deadline: float) -> tuple[OwnershipScan, bool]:
        if self._deadline_expired(deadline):
            return OwnershipScan((), (), False), False
        scan = self._scan()
        return scan, not self._deadline_expired(deadline)

    def _default_scan(self) -> OwnershipScan:
        with self._scan_lock:
            return self._default_scan_locked()

    def _default_scan_locked(self) -> OwnershipScan:
        with self._state_lock:
            identity = self._identity
            tracked = dict(self._tracked)
        if identity is None:
            return OwnershipScan((), (), False)

        complete = True
        group_membership_complete = True
        retired: dict[int, float] = {}

        def retire_gone_identity(pid: int) -> None:
            birth_time = tracked.pop(pid, None)
            if birth_time is not None:
                retired[pid] = birth_time

        try:
            if psutil.Process(identity.pid).create_time() != identity.birth_time:
                complete = False
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
        except psutil.AccessDenied:
            complete = False

        try:
            pids = set(psutil.pids())
        except psutil.Error:
            pids = set()
            complete = False
            group_membership_complete = False
        pids.update(tracked)

        membership: dict[int, tuple[int, int]] = {}
        candidate_pids = set(tracked)
        for pid in pids:
            try:
                sid = os.getsid(pid)
                pgid = os.getpgid(pid)
            except ProcessLookupError:
                retire_gone_identity(pid)
                continue
            except PermissionError:
                complete = False
                group_membership_complete = False
                continue
            membership[pid] = (sid, pgid)
            if sid == identity.sid:
                candidate_pids.add(pid)

        owned: dict[int, PosixProcessIdentity] = {}
        for pid in candidate_pids:
            current_membership = membership.get(pid)
            if current_membership is None:
                continue
            try:
                birth_time = psutil.Process(pid).create_time()
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                retire_gone_identity(pid)
                continue
            except psutil.AccessDenied:
                complete = False
                continue
            expected_birth = tracked.get(pid)
            if expected_birth is not None and birth_time != expected_birth:
                complete = False
                continue
            sid, pgid = current_membership
            current = PosixProcessIdentity(pid, birth_time, sid, pgid)
            owned[pid] = current
            tracked[pid] = birth_time

        owned_groups = {item.initial_pgid for item in owned.values()}
        observed = list(owned.values())
        for pid, (sid, pgid) in membership.items():
            if pid in owned or pgid not in owned_groups:
                continue
            try:
                birth_time = psutil.Process(pid).create_time()
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except psutil.AccessDenied:
                complete = False
                group_membership_complete = False
                continue
            observed.append(PosixProcessIdentity(pid, birth_time, sid, pgid))

        with self._state_lock:
            for pid, birth_time in retired.items():
                if self._tracked.get(pid) == birth_time:
                    self._tracked.pop(pid, None)
            self._tracked.update(tracked)
        return OwnershipScan(
            tuple(sorted(owned.values(), key=lambda item: item.pid)),
            tuple(sorted(observed, key=lambda item: item.pid)),
            complete,
            group_membership_complete,
        )

    @staticmethod
    def _identity_alive(identity: PosixProcessIdentity) -> bool | None:
        try:
            birth_time = psutil.Process(identity.pid).create_time()
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return False
        except psutil.AccessDenied:
            return None
        if birth_time != identity.birth_time:
            return None
        return True

    def _signal_owned(
        self,
        scan: OwnershipScan,
        signum: int,
        deadline: float,
    ) -> bool:
        if self._deadline_expired(deadline):
            return scan.complete
        complete = scan.complete
        plan = _plan_signals(scan)
        signalled: set[tuple[int, float]] = set()
        for group_id in plan.group_ids:
            current, within_deadline = self._scan_before_deadline(deadline)
            complete = complete and current.complete
            if not within_deadline:
                return complete
            current_plan = _plan_signals(current)
            if group_id not in current_plan.group_ids:
                continue
            members = tuple(
                item for item in current.owned if item.initial_pgid == group_id
            )
            if self._deadline_expired(deadline):
                return complete
            try:
                os.killpg(group_id, signum)
            except ProcessLookupError:
                continue
            except PermissionError:
                complete = False
                continue
            signalled.update((item.pid, item.birth_time) for item in members)

        individuals = tuple(
            item for item in scan.owned if (item.pid, item.birth_time) not in signalled
        )
        for identity in individuals:
            if self._deadline_expired(deadline):
                return complete
            alive = self._identity_alive(identity)
            if alive is None:
                complete = False
            if self._deadline_expired(deadline):
                return complete
            if alive is None:
                continue
            if not alive:
                continue
            if self._deadline_expired(deadline):
                return complete
            try:
                os.kill(identity.pid, signum)
            except ProcessLookupError:
                continue
            except PermissionError:
                complete = False
        return complete

    def _wait_stage(
        self,
        deadline: float,
        *,
        preserve_output: bool = True,
    ) -> tuple[OwnershipScan, bool]:
        scan, within_deadline = self._scan_before_deadline(deadline)
        if not within_deadline:
            return scan, scan.complete
        complete = scan.complete
        while True:
            if preserve_output:
                if self._deadline_expired(deadline):
                    return scan, complete
                self._buffer_cleanup_turn()
                if self._deadline_expired(deadline):
                    return scan, complete
            if self._shell_reaped.is_set() and not scan.owned:
                return scan, complete
            remaining = deadline - self._clock()
            if remaining <= 0:
                return scan, complete
            self._sleep(min(_PROCESS_POLL_SECONDS, max(0.0, remaining)))
            if self._deadline_expired(deadline):
                return scan, complete
            scan, within_deadline = self._scan_before_deadline(deadline)
            if not within_deadline:
                return scan, complete and scan.complete
            complete = complete and scan.complete

    def _run_cleanup(
        self,
        attempt: CleanupAttempt,
        *,
        parser_failed: bool = False,
    ) -> CleanupProof:
        self.request_priority_close()
        schedule = CleanupSchedule()
        hangup_at = attempt.t0 + schedule.hangup_no_later_than
        terminate_at = attempt.t0 + schedule.terminate_no_later_than
        force_kill_at = attempt.t0 + schedule.force_kill_no_later_than
        deadline = attempt.t0 + schedule.deadline_seconds
        complete = True

        with self._state_lock:
            self._process_only_dead = False
        if self._deadline_expired(deadline):
            return CleanupProof(
                process_dead=False,
                stream_closed=self._pty_eof,
                output_complete=(
                    not parser_failed and self._output_complete and self._pty_eof
                ),
            )

        scan, within_deadline = self._scan_before_deadline(deadline)
        if not within_deadline:
            return CleanupProof(
                process_dead=False,
                stream_closed=self._pty_eof,
                output_complete=(
                    not parser_failed and self._output_complete and self._pty_eof
                ),
            )
        complete = complete and scan.complete
        if not self._deadline_expired(hangup_at):
            signal_complete = self._signal_owned(scan, signal.SIGHUP, hangup_at)
            complete = complete and signal_complete
            if not self._deadline_expired(hangup_at):
                scan, stage_complete = self._wait_stage(
                    hangup_at,
                    preserve_output=not parser_failed,
                )
                complete = complete and stage_complete
        if scan.owned and not self._deadline_expired(terminate_at):
            signal_complete = self._signal_owned(scan, signal.SIGTERM, terminate_at)
            complete = complete and signal_complete
            if not self._deadline_expired(terminate_at):
                scan, stage_complete = self._wait_stage(
                    terminate_at,
                    preserve_output=not parser_failed,
                )
                complete = complete and stage_complete
        if scan.owned and not self._deadline_expired(force_kill_at):
            signal_complete = self._signal_owned(
                scan,
                signal.SIGKILL,
                force_kill_at,
            )
            complete = complete and signal_complete
            if not self._deadline_expired(force_kill_at):
                _, stage_complete = self._wait_stage(
                    force_kill_at,
                    preserve_output=not parser_failed,
                )
                complete = complete and stage_complete

        process_dead, proof_complete = self._prove_zero_owned(
            deadline,
            preserve_output=not parser_failed,
        )
        complete = complete and proof_complete
        with self._state_lock:
            self._process_only_dead = process_dead and complete
        if parser_failed and process_dead and complete:
            with self._io_lock:
                self._output_buffer.clear()
        while process_dead and complete and not self._pty_eof:
            if self._deadline_expired(deadline):
                break
            if parser_failed:
                self._discard_cleanup_turn()
            else:
                self._buffer_cleanup_turn()
            if self._deadline_expired(deadline):
                break
            if not self._pty_eof:
                remaining = max(0.0, deadline - self._clock())
                self._sleep(min(_PROCESS_POLL_SECONDS, remaining))
        process_dead = process_dead and complete
        proof = CleanupProof(
            process_dead=process_dead,
            stream_closed=self._pty_eof,
            output_complete=(
                not parser_failed and self._output_complete and self._pty_eof
            ),
        )
        if proof.process_dead and proof.stream_closed:
            self._close_proven()
        return proof

    def _prove_zero_owned(
        self,
        deadline: float,
        *,
        preserve_output: bool = True,
    ) -> tuple[bool, bool]:
        first_zero: float | None = None
        complete = True
        while not self._deadline_expired(deadline):
            if preserve_output:
                self._buffer_cleanup_turn()
                if self._deadline_expired(deadline):
                    return False, False
            scan, within_deadline = self._scan_before_deadline(deadline)
            if not within_deadline:
                return False, False
            complete = complete and scan.complete
            now = self._clock()
            if now >= deadline:
                return False, False
            if self._shell_reaped.is_set() and not scan.owned:
                if first_zero is None:
                    first_zero = now
                    with self._state_lock:
                        self._zero_scan_times.append(now)
                elif now - first_zero >= _ZERO_SCAN_INTERVAL_SECONDS:
                    with self._state_lock:
                        self._zero_scan_times.append(now)
                    return complete, complete
            else:
                first_zero = None
            remaining = deadline - now
            self._sleep(min(_PROCESS_POLL_SECONDS, max(0.0, remaining)))
        return False, complete

    def _close_proven(self) -> None:
        self._monitor_stop.set()
        with self._io_lock:
            with self._state_lock:
                fd = self._master_fd
                self._master_fd = None
            _safe_close(fd)
