"""Real-process qualification for the admitted POSIX terminal backend."""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator
import errno
import json
import os
from pathlib import Path
import select
import shlex
import shutil
import signal
import subprocess
import sys
import threading
from threading import Event, Thread
import time

import psutil
from pydantic import BaseModel
import pytest

fcntl = pytest.importorskip(
    "fcntl",
    reason="POSIX PTY backend requires POSIX",
    exc_type=ImportError,
)

from tldw_chatbook.Terminal import posix_backend as posix_backend_module  # noqa: E402
from tldw_chatbook.Terminal.contracts import (  # noqa: E402
    AdmissionGate,
    CleanupAttempt,
    CleanupProof,
    CleanupSchedule,
    MAX_IO_CHUNK_BYTES,
    MAX_PENDING_INPUT_BYTES,
    MAX_PENDING_OUTPUT_BYTES,
    TerminalLaunchRequest,
    TerminalLifecycle,
)
from tldw_chatbook.Terminal.launch import _new_shell_choice  # noqa: E402
from tldw_chatbook.Terminal.io_actors import MAX_PARSER_SLICE_BYTES  # noqa: E402
from tldw_chatbook.Terminal.posix_backend import (  # noqa: E402
    OwnershipScan,
    PosixProcessIdentity,
    PosixTerminalBackend,
    _plan_signals,
    _read_pipe,
    _stderr_context,
)
from tldw_chatbook.Terminal.posix_launcher import _validated_config  # noqa: E402
from tldw_chatbook.Terminal.session_manager import (  # noqa: E402
    TerminalSessionManager,
    TerminalViewToken,
)


REPOSITORY_ROOT = Path(__file__).parents[2]
FIXTURES = REPOSITORY_ROOT / "Tests" / "fixtures" / "terminal"
TERMINAL_CHILD = FIXTURES / "terminal_child.py"
DESCENDANT_HOLDS_TTY = FIXTURES / "descendant_holds_tty.py"
JOB_CONTROL_TREE = FIXTURES / "job_control_tree.py"


def _read_fd(fd: int, *, timeout: float, maximum: int = 64 * 1024) -> bytes:
    deadline = time.monotonic() + timeout
    result = bytearray()
    while len(result) < maximum:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        readable, _, _ = select.select([fd], [], [], remaining)
        if not readable:
            break
        try:
            chunk = os.read(fd, min(4096, maximum - len(result)))
        except OSError as exc:
            if exc.errno == errno.EIO:
                break
            raise
        if not chunk:
            break
        result.extend(chunk)
        if b"\n" in result:
            break
    return bytes(result)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _direct_launcher(
    tmp_path: Path,
    *,
    admitted: bool,
    extra_fd: int,
) -> tuple[subprocess.Popen[bytes], int, dict[str, object], Path]:
    master_fd, slave_fd = os.openpty()
    config_read, config_write = os.pipe()
    gate_read, gate_write = os.pipe()
    report_read, report_write = os.pipe()
    status_read, status_write = os.pipe()
    sentinel = tmp_path / "launcher-sentinel.json"
    config = {
        "argv": [
            sys.executable,
            str(TERMINAL_CHILD),
            "sentinel",
            str(sentinel),
            "--check-fd",
            str(extra_fd),
        ],
        "cwd": str(tmp_path),
        "environment": {
            "HOME": str(tmp_path),
            "PATH": os.environ["PATH"],
            "TERM": "linux",
        },
        "executable": sys.executable,
        "token": "launcher-token",
    }
    process: subprocess.Popen[bytes] | None = None
    process_birth: float | None = None
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tldw_chatbook.Terminal.posix_launcher",
                "--slave-fd",
                str(slave_fd),
                "--config-fd",
                str(config_read),
                "--admission-fd",
                str(gate_read),
                "--report-fd",
                str(report_write),
                "--exec-status-fd",
                str(status_write),
            ],
            pass_fds=(
                slave_fd,
                config_read,
                gate_read,
                report_write,
                status_write,
                extra_fd,
            ),
            close_fds=True,
        )
        process_birth = psutil.Process(process.pid).create_time()
        for descriptor in (
            slave_fd,
            config_read,
            gate_read,
            report_write,
            status_write,
        ):
            os.close(descriptor)
        _write_all(config_write, json.dumps(config).encode("utf-8") + b"\n")
        os.close(config_write)
        report = _read_fd(report_read, timeout=3.0, maximum=4096)
        os.close(report_read)
        assert report, "launcher did not report its gated identity"
        identity = json.loads(report)
        assert process.poll() is None
        assert not sentinel.exists()
        decision = {
            "admitted": admitted,
            "token": "launcher-token" if admitted else "",
        }
        _write_all(gate_write, json.dumps(decision).encode("utf-8") + b"\n")
        os.close(gate_write)
        status = _read_fd(status_read, timeout=3.0, maximum=1)
        os.close(status_read)
        assert status == b""
        return process, master_fd, identity, sentinel
    except BaseException:
        if process is not None and process_birth is not None:
            _terminate_exact(process.pid, process_birth)
        for descriptor in (
            master_fd,
            slave_fd,
            config_read,
            config_write,
            gate_read,
            gate_write,
            report_read,
            report_write,
            status_read,
            status_write,
        ):
            _safe_test_close(descriptor)
        raise


def _environment(home: Path) -> dict[str, str]:
    return {
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LOGNAME": "terminal-test",
        "PATH": os.environ["PATH"],
        "SHELL": shutil.which("bash") or "/bin/sh",
        "TERM": "linux",
        "USER": "terminal-test",
    }


def _write_profile_that_spawns_child(
    home: Path,
    tmp_path: Path,
    *,
    stem: str,
) -> tuple[Path, Path]:
    profile_sentinel = tmp_path / f"{stem}-profile-executed"
    child_pid_file = tmp_path / f"{stem}-profile-child.pid"
    child_command = shlex.join(
        [sys.executable, str(TERMINAL_CHILD), "sighup", str(child_pid_file)]
    )
    (home / ".bash_profile").write_text(
        f"{child_command} &\n"
        f"printf profile-executed > {shlex.quote(str(profile_sentinel))}\n",
        encoding="utf-8",
    )
    return profile_sentinel, child_pid_file


def _backend(tmp_path: Path) -> PosixTerminalBackend:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / ".bash_profile").write_text(
        "export TERMINAL_PROFILE_SENTINEL=loaded\n",
        encoding="utf-8",
    )
    backend = PosixTerminalBackend(environment_factory=lambda: _environment(home))
    identity = backend.start(
        TerminalLaunchRequest(
            name="test-terminal",
            shell="bash",
            start_directory=str(tmp_path),
            columns=80,
            rows=24,
        ),
        AdmissionGate(admitted=True, token="test-session"),
    )
    assert identity.session_id == "test-session"
    return backend


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[PosixTerminalBackend]:
    instance = _backend(tmp_path)
    _read_until(instance, b"$", timeout=3.0)
    try:
        yield instance
    finally:
        _cleanup_backend_exact(instance)


def _read_until(
    backend: PosixTerminalBackend,
    needle: bytes,
    *,
    timeout: float = 5.0,
) -> bytes:
    deadline = time.monotonic() + timeout
    result = bytearray()
    while time.monotonic() < deadline:
        chunk = backend.read()
        if chunk is None:
            time.sleep(0.005)
            continue
        if chunk == b"":
            break
        result.extend(chunk)
        if needle in result:
            return bytes(result)
    raise AssertionError(f"terminal output did not contain {needle!r}")


def _read_to_eof(backend: PosixTerminalBackend, *, timeout: float = 5.0) -> bytes:
    deadline = time.monotonic() + timeout
    result = bytearray()
    while time.monotonic() < deadline:
        chunk = backend.read()
        if chunk is None:
            time.sleep(0.005)
            continue
        if chunk == b"":
            return bytes(result)
        result.extend(chunk)
    raise AssertionError("PTY EOF was not observed")


def _json_line(output: bytes, key: str) -> dict[str, object]:
    for raw_line in output.replace(b"\r", b"").splitlines():
        for start, byte in enumerate(raw_line):
            if byte != ord("{"):
                continue
            try:
                value = json.loads(raw_line[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and key in value:
                return value
    raise AssertionError(f"JSON terminal output did not contain {key!r}")


def test_json_line_accepts_prompt_prefixed_payload() -> None:
    output = b'\x1b[?2004luser@linux:/tmp$ {"stdin_tty": true, "cwd": "/tmp"}\r\n'

    assert _json_line(output, "stdin_tty") == {
        "stdin_tty": True,
        "cwd": "/tmp",
    }


def _pid_matches(pid: int, birth_time: float) -> bool:
    try:
        return psutil.Process(pid).create_time() == birth_time
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False


def _safe_test_close(descriptor: int | None) -> None:
    if descriptor is None:
        return
    try:
        os.close(descriptor)
    except OSError:
        pass


def _capture_exact(pid: int) -> tuple[int, float] | None:
    try:
        return pid, psutil.Process(pid).create_time()
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
        return None


def _terminate_exact(pid: int, birth_time: float) -> None:
    if not _pid_matches(pid, birth_time):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if _wait_for(lambda: not _pid_matches(pid, birth_time), timeout=0.75):
        return
    if not _pid_matches(pid, birth_time):
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    _wait_for(lambda: not _pid_matches(pid, birth_time), timeout=0.75)


def _cleanup_backend_exact(
    instance: PosixTerminalBackend,
    extra_identities: tuple[tuple[int, float], ...] = (),
    *,
    require_proven: bool = True,
) -> None:
    known = {pid: birth for pid, birth in extra_identities}
    cleanup_attempted = False
    cleanup_error: Exception | None = None
    cleanup_proof = None
    try:
        identity = instance.identity_for_tests
    except RuntimeError:
        identity = None
    if identity is not None:
        known[identity.pid] = identity.birth_time
    try:
        for owned in instance.owned_processes_for_tests():
            known[owned.pid] = owned.birth_time
    except Exception:
        pass
    if identity is not None and _pid_matches(identity.pid, identity.birth_time):
        cleanup_attempted = True
        try:
            instance.request_priority_close()
            cleanup_proof = instance.cleanup(CleanupAttempt(time.monotonic()))
        except Exception as exc:
            cleanup_error = exc
    for pid, birth_time in known.items():
        _terminate_exact(pid, birth_time)
    if identity is not None:
        instance.wait_for_shell_exit(timeout_seconds=1.0)
    if require_proven and cleanup_attempted:
        if cleanup_error is not None:
            raise cleanup_error
        assert cleanup_proof is not None
        assert cleanup_proof.process_dead is True
        assert cleanup_proof.stream_closed is True


def _wait_for(
    predicate: Callable[[], bool],
    *,
    timeout: float = 5.0,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _wait_for_positive_pid(path: Path, *, timeout: float = 5.0) -> int | None:
    observed: int | None = None

    def ready() -> bool:
        nonlocal observed
        try:
            candidate = int(path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, ValueError):
            return False
        if candidate <= 0:
            return False
        observed = candidate
        return True

    if not _wait_for(ready, timeout=timeout):
        return None
    return observed


def _manager_screen_contains(
    terminal: TerminalSessionManager,
    view: TerminalViewToken,
    needle: str,
) -> bool:
    state = terminal.view_state(view)
    if state is None or not state.sessions:
        return False
    screen = state.sessions[0].screen
    return needle in "\n".join(
        line.text for line in (*screen.scrollback, *screen.lines)
    )


def test_wait_for_positive_pid_ignores_incomplete_file_contents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "fixture.pid"
    responses = iter(("", "0", "not-a-pid", "321"))

    def scripted_read_text(
        _path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        del encoding, errors
        return next(responses)

    monkeypatch.setattr(Path, "read_text", scripted_read_text)

    assert _wait_for_positive_pid(pid_file, timeout=0.25) == 321


def test_posix_only_imports_follow_module_platform_gate() -> None:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    fcntl_guards = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and (node.func.value.id, node.func.attr) == ("pytest", "importorskip")
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "fcntl"
    ]
    local_imports = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and isinstance(node.module, str)
        and node.module.startswith("tldw_chatbook.")
    ]

    assert len(fcntl_guards) == 1
    assert local_imports
    assert fcntl_guards[0] < min(local_imports)


def test_launcher_config_is_a_strict_pydantic_boundary(tmp_path: Path) -> None:
    executable = sys.executable
    value = {
        "argv": [executable, "-V"],
        "cwd": str(tmp_path),
        "environment": {"PATH": os.environ["PATH"]},
        "executable": executable,
        "token": "launcher-token",
    }

    config = _validated_config(value)

    assert isinstance(config, BaseModel)
    assert config.executable == executable
    assert config.argv == [executable, "-V"]
    assert config.cwd == str(tmp_path)
    assert config.environment == {"PATH": os.environ["PATH"]}
    assert config.token == "launcher-token"
    with pytest.raises(ValueError, match="launcher config is invalid"):
        _validated_config({**value, "unexpected": True})
    without_executable = dict(value)
    without_executable.pop("executable")
    with pytest.raises(ValueError, match="launcher config is invalid"):
        _validated_config(without_executable)


def test_spawn_stderr_fallback_is_closed_after_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidStderr:
        @staticmethod
        def fileno() -> int:
            return -1

    invalid = InvalidStderr()
    monkeypatch.setattr(sys, "stderr", invalid)
    monkeypatch.setattr(sys, "__stderr__", invalid)

    with _stderr_context():
        fallback = sys.stderr
        descriptor = fallback.fileno()
        os.fstat(descriptor)

    with pytest.raises(OSError) as error:
        os.fstat(descriptor)
    assert error.value.errno == errno.EBADF


def test_backend_uses_central_normalized_start_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requested = tmp_path / "requested"
    normalized = tmp_path / "normalized"
    home = tmp_path / "home"
    requested.mkdir()
    normalized.mkdir()
    home.mkdir()
    validated: list[object] = []

    def validate_start_directory(value: object) -> Path:
        validated.append(value)
        return normalized

    monkeypatch.setattr(
        posix_backend_module,
        "validate_existing_absolute_directory",
        validate_start_directory,
        raising=False,
    )
    backend = PosixTerminalBackend(environment_factory=lambda: _environment(home))
    try:
        backend.start(
            TerminalLaunchRequest(
                name="normalized-directory",
                shell="bash",
                start_directory=str(requested),
                columns=80,
                rows=24,
            ),
            AdmissionGate(admitted=True, token="normalized-directory"),
        )
        _read_until(backend, b"$", timeout=3.0)
        command = shlex.join([sys.executable, str(TERMINAL_CHILD), "probe"])
        backend.write((command + "\n").encode())
        probe = _json_line(_read_until(backend, b'"stdin_tty": true'), "stdin_tty")

        assert validated == [str(requested)]
        assert probe["cwd"] == str(normalized)
    finally:
        _cleanup_backend_exact(backend)


def test_pre_admission_failure_proves_disposed_backend(tmp_path: Path) -> None:
    now = 1_000.0

    def clock() -> float:
        return now

    def advance(duration: float) -> None:
        nonlocal now
        now += duration

    def fail_environment() -> dict[str, str]:
        raise RuntimeError("environment unavailable")

    backend = PosixTerminalBackend(
        environment_factory=fail_environment,
        monotonic_clock=clock,
        sleep=advance,
    )
    with pytest.raises(RuntimeError, match="environment unavailable"):
        backend.start(
            TerminalLaunchRequest(
                name="pre-admission-failure",
                shell="bash",
                start_directory=str(tmp_path),
                columns=80,
                rows=24,
            ),
            AdmissionGate(admitted=True, token="pre-admission-failure"),
        )

    proof = backend.cleanup(CleanupAttempt(now))

    assert proof == CleanupProof(True, True, True)
    assert now == 1_000.0


def test_parent_source_forbids_unsafe_fork_paths_and_uses_fresh_launcher() -> None:
    backend_path = Path("tldw_chatbook/Terminal/posix_backend.py")
    launcher_path = Path("tldw_chatbook/Terminal/posix_launcher.py")
    backend_source = backend_path.read_text(encoding="utf-8")
    launcher_source = launcher_path.read_text(encoding="utf-8")
    tree = ast.parse(backend_source)

    forbidden_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            if isinstance(owner, ast.Name) and (owner.id, node.func.attr) in {
                ("os", "fork"),
                ("pty", "fork"),
            }:
                forbidden_calls.append(f"{owner.id}.{node.func.attr}")
            if any(keyword.arg == "preexec_fn" for keyword in node.keywords):
                forbidden_calls.append("preexec_fn")

    assert forbidden_calls == []
    assert "os.setsid(" not in backend_source
    assert launcher_source.count("os.setsid()") == 1
    popen_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and (node.func.value.id, node.func.attr) == ("subprocess", "Popen")
    ]
    assert len(popen_calls) == 1
    argv = popen_calls[0].args[0]
    assert isinstance(argv, ast.List)
    assert ast.unparse(argv.elts[0]) == "sys.executable"
    assert [ast.literal_eval(item) for item in argv.elts[1:3]] == [
        "-m",
        "tldw_chatbook.Terminal.posix_launcher",
    ]
    assert {keyword.arg for keyword in popen_calls[0].keywords} >= {
        "close_fds",
        "pass_fds",
    }


def test_fresh_launcher_reports_identity_then_refuses_before_exec(
    tmp_path: Path,
) -> None:
    extra_read, extra_write = os.pipe()
    process: subprocess.Popen[bytes] | None = None
    master_fd: int | None = None
    process_birth: float | None = None
    try:
        process, master_fd, identity, sentinel = _direct_launcher(
            tmp_path,
            admitted=False,
            extra_fd=extra_write,
        )
        process_birth = float(identity["birth_time"])
        assert identity == {
            "birth_time": psutil.Process(process.pid).create_time(),
            "pgid": process.pid,
            "pid": process.pid,
            "sid": process.pid,
        }
        assert psutil.Process(process.pid).terminal() is None
        assert process.wait(timeout=3.0) == 125
        assert not sentinel.exists()
        assert _read_fd(master_fd, timeout=0.5) == b""
    finally:
        if process is not None and process_birth is not None:
            _terminate_exact(process.pid, process_birth)
        _safe_test_close(master_fd)
        _safe_test_close(extra_read)
        _safe_test_close(extra_write)


def test_fresh_launcher_admission_execs_in_place_with_controlling_tty(
    tmp_path: Path,
) -> None:
    extra_read, extra_write = os.pipe()
    process: subprocess.Popen[bytes] | None = None
    master_fd: int | None = None
    process_birth: float | None = None
    try:
        process, master_fd, identity, sentinel = _direct_launcher(
            tmp_path,
            admitted=True,
            extra_fd=extra_write,
        )
        process_birth = float(identity["birth_time"])
        output = _read_fd(master_fd, timeout=3.0)
        assert process.wait(timeout=3.0) == 0
        payload = json.loads(sentinel.read_text(encoding="utf-8"))
        assert payload == {
            "descriptor_closed": True,
            "pid": identity["pid"],
            "sid": identity["sid"],
            "stdin_tty": True,
        }
        assert b'"sentinel": true' in output
    finally:
        if process is not None and process_birth is not None:
            _terminate_exact(process.pid, process_birth)
        _safe_test_close(master_fd)
        _safe_test_close(extra_read)
        _safe_test_close(extra_write)


def test_backend_parent_launches_recorded_fresh_python_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[object, dict[str, object]]] = []
    real_popen = subprocess.Popen

    def recording_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        calls.append((args[0], dict(kwargs)))
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    backend = _backend(tmp_path)
    try:
        argv, kwargs = calls[0]
        assert isinstance(argv, list)
        assert argv[:3] == [
            sys.executable,
            "-m",
            "tldw_chatbook.Terminal.posix_launcher",
        ]
        assert kwargs["close_fds"] is True
        assert isinstance(kwargs["pass_fds"], tuple)
        identity = backend.identity_for_tests
        assert identity.pid == backend.launcher_pid_for_tests
        assert identity.sid == identity.pid
        assert identity.initial_pgid == identity.pid
    finally:
        _cleanup_backend_exact(backend)


def test_parent_setup_failure_while_gated_cannot_exec_profile_or_leak_processes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    profile_sentinel, child_pid_file = _write_profile_that_spawns_child(
        home,
        tmp_path,
        stem="setup-failure",
    )
    launched: list[tuple[int, float]] = []
    real_popen = subprocess.Popen

    def recording_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        process = real_popen(*args, **kwargs)
        launched.append((process.pid, psutil.Process(process.pid).create_time()))
        return process

    def fail_parent_setup(
        _fd: int,
        _operation: int,
        _argument: object = 0,
    ) -> int:
        child_pid = _wait_for_positive_pid(child_pid_file, timeout=1.0)
        if child_pid is not None:
            _wait_for(profile_sentinel.exists, timeout=1.0)
        raise OSError(errno.EIO, "injected setup failure")

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    monkeypatch.setattr(fcntl, "fcntl", fail_parent_setup)
    backend = PosixTerminalBackend(environment_factory=lambda: _environment(home))
    child_identity: tuple[int, float] | None = None
    try:
        with pytest.raises(OSError):
            backend.start(
                TerminalLaunchRequest(
                    name="setup-failure",
                    shell="bash",
                    start_directory=str(tmp_path),
                    columns=80,
                    rows=24,
                ),
                AdmissionGate(admitted=True, token="setup-failure"),
            )
        if child_pid_file.exists():
            child_identity = _capture_exact(
                int(child_pid_file.read_text(encoding="ascii"))
            )

        assert launched
        assert profile_sentinel.exists() is False
        assert child_pid_file.exists() is False
        assert all(not _pid_matches(pid, birth_time) for pid, birth_time in launched)
        assert backend.cleanup(CleanupAttempt(time.monotonic())) == CleanupProof(
            True,
            True,
            True,
        )
    finally:
        monkeypatch.undo()
        if child_identity is None and child_pid_file.exists():
            try:
                child_identity = _capture_exact(
                    int(child_pid_file.read_text(encoding="ascii"))
                )
            except (OSError, ValueError):
                pass
        exact = tuple(launched)
        if child_identity is not None:
            exact = (*exact, child_identity)
        _cleanup_backend_exact(backend, exact, require_proven=False)


def test_exec_status_read_failure_after_admission_cleans_owned_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    profile_sentinel, child_pid_file = _write_profile_that_spawns_child(
        home,
        tmp_path,
        stem="status-failure",
    )
    launched: list[tuple[int, float]] = []
    unadmitted_reaps: list[int] = []
    cleanup_proofs: list[CleanupProof] = []
    observed_children: list[tuple[int, float]] = []
    real_popen = subprocess.Popen

    def recording_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        process = real_popen(*args, **kwargs)
        launched.append((process.pid, psutil.Process(process.pid).create_time()))
        return process

    def fail_exec_status_read(
        fd: int,
        *,
        deadline: float,
        maximum: int,
    ) -> bytes:
        if maximum != 1:
            return _read_pipe(fd, deadline=deadline, maximum=maximum)
        child_pid = _wait_for_positive_pid(child_pid_file, timeout=2.0)
        assert child_pid is not None
        assert _wait_for(profile_sentinel.exists, timeout=2.0)
        child_identity = _capture_exact(child_pid)
        assert child_identity is not None
        observed_children.append(child_identity)
        raise RuntimeError("POSIX terminal startup failed")

    class ObservedBackend(PosixTerminalBackend):
        def _reap_unadmitted(
            self,
            process: subprocess.Popen[bytes],
            *,
            deadline: float | None = None,
        ) -> bool:
            unadmitted_reaps.append(process.pid)
            return super()._reap_unadmitted(process, deadline=deadline)

        def cleanup(self, attempt: CleanupAttempt) -> CleanupProof:
            proof = super().cleanup(attempt)
            cleanup_proofs.append(proof)
            return proof

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    monkeypatch.setattr(
        "tldw_chatbook.Terminal.posix_backend._read_pipe",
        fail_exec_status_read,
    )
    backend = ObservedBackend(environment_factory=lambda: _environment(home))
    child_identity: tuple[int, float] | None = None
    try:
        with pytest.raises(RuntimeError, match="POSIX terminal startup failed"):
            backend.start(
                TerminalLaunchRequest(
                    name="status-failure",
                    shell="bash",
                    start_directory=str(tmp_path),
                    columns=80,
                    rows=24,
                ),
                AdmissionGate(admitted=True, token="status-failure"),
            )
        child_identity = observed_children[0]

        assert profile_sentinel.exists() is True
        assert observed_children == [child_identity]
        assert unadmitted_reaps == []
        assert len(cleanup_proofs) == 1
        assert cleanup_proofs[0].process_dead is True
        assert cleanup_proofs[0].stream_closed is True
        assert backend.shell_reap_count_for_tests == 1
        assert all(not _pid_matches(pid, birth_time) for pid, birth_time in launched)
        assert not _pid_matches(*child_identity)
    finally:
        monkeypatch.undo()
        if child_identity is None and child_pid_file.exists():
            try:
                child_identity = _capture_exact(
                    int(child_pid_file.read_text(encoding="ascii"))
                )
            except (OSError, ValueError):
                pass
        exact = tuple(launched)
        if child_identity is not None:
            exact = (*exact, child_identity)
        _cleanup_backend_exact(backend, exact, require_proven=False)


def test_reaper_thread_start_failure_rolls_back_before_admission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    profile_sentinel, child_pid_file = _write_profile_that_spawns_child(
        home,
        tmp_path,
        stem="reaper-failure",
    )
    launched: list[tuple[subprocess.Popen[bytes], float]] = []
    real_popen = subprocess.Popen

    def recording_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        process = real_popen(*args, **kwargs)
        launched.append((process, psutil.Process(process.pid).create_time()))
        return process

    def fail_thread_start(_thread: threading.Thread) -> None:
        child_pid = _wait_for_positive_pid(child_pid_file, timeout=1.0)
        if child_pid is not None:
            _wait_for(profile_sentinel.exists, timeout=1.0)
        raise RuntimeError("injected thread start failure")

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    monkeypatch.setattr(threading.Thread, "start", fail_thread_start)
    backend = PosixTerminalBackend(environment_factory=lambda: _environment(home))
    child_identity: tuple[int, float] | None = None
    try:
        with pytest.raises(RuntimeError, match="injected thread start failure"):
            backend.start(
                TerminalLaunchRequest(
                    name="reaper-failure",
                    shell="bash",
                    start_directory=str(tmp_path),
                    columns=80,
                    rows=24,
                ),
                AdmissionGate(admitted=True, token="reaper-failure"),
            )

        assert profile_sentinel.exists() is False
        assert child_pid_file.exists() is False
        assert backend.shell_reap_count_for_tests == 0
        with pytest.raises(RuntimeError, match="POSIX terminal is not started"):
            _ = backend.identity_for_tests
        assert all(
            not _pid_matches(process.pid, birth_time)
            for process, birth_time in launched
        )
        assert backend.cleanup(CleanupAttempt(time.monotonic())) == CleanupProof(
            True,
            True,
            True,
        )
    finally:
        monkeypatch.undo()
        if child_pid_file.exists():
            try:
                child_identity = _capture_exact(
                    int(child_pid_file.read_text(encoding="ascii"))
                )
            except (OSError, ValueError):
                pass
        exact = [(process.pid, birth_time) for process, birth_time in launched]
        if child_identity is not None:
            exact.append(child_identity)
        for pid, birth_time in exact:
            _terminate_exact(pid, birth_time)
        for process, _birth_time in launched:
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        with backend._io_lock:
            with backend._state_lock:
                master_fd = backend._master_fd
                backend._master_fd = None
        _safe_test_close(master_fd)


def test_interactive_pty_retains_state_and_round_trips_unicode(
    backend: PosixTerminalBackend,
    tmp_path: Path,
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    backend.write(("cd " + shlex.quote(str(nested)) + "\n").encode())
    backend.write("export TERMINAL_CHILD_VALUE='Zażółć 🦊'\n".encode())
    command = shlex.join([sys.executable, str(TERMINAL_CHILD), "probe"])
    backend.write((command + "\n").encode())
    output = _read_until(backend, b'"stdin_tty": true')
    probe = _json_line(output, "stdin_tty")
    assert probe["cwd"] == str(nested)
    assert probe["value"] == "Zażółć 🦊"
    assert probe["stdin_tty"] is True
    assert probe["stdout_tty"] is True
    assert probe["stderr_tty"] is True
    assert probe["sid"] == backend.identity_for_tests.sid

    unicode_command = shlex.join([sys.executable, str(TERMINAL_CHILD), "unicode"])
    backend.write((unicode_command + "\n").encode())
    _read_until(backend, b"UNICODE_READY")
    backend.write("γειά σου 🌍\n".encode("utf-8"))
    assert "UNICODE:γειά σου 🌍".encode() in _read_until(
        backend,
        "UNICODE:γειά σου 🌍".encode(),
    )


def test_short_write_accepts_suffix_once_and_flushes_it_in_order(
    backend: PosixTerminalBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master_fd = backend._master_fd
    assert master_fd is not None
    real_write = os.write
    first = True

    def short_once(fd: int, data: bytes | memoryview) -> int:
        nonlocal first
        if fd != master_fd or not first:
            return real_write(fd, data)
        first = False
        prefix = bytes(data[:5])
        return real_write(fd, prefix)

    monkeypatch.setattr(os, "write", short_once)
    try:
        backend.write(b"printf 'SHORT_WRITE_OK\\n'\n")
        assert b"SHORT_WRITE_OK" in _read_until(backend, b"SHORT_WRITE_OK")
    finally:
        monkeypatch.undo()


def test_eagain_write_is_buffered_and_flushed_without_caller_retry(
    backend: PosixTerminalBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master_fd = backend._master_fd
    assert master_fd is not None
    real_write = os.write
    first = True

    def eagain_once(fd: int, data: bytes | memoryview) -> int:
        nonlocal first
        if fd == master_fd and first:
            first = False
            raise BlockingIOError(errno.EAGAIN, "backpressure")
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", eagain_once)
    try:
        backend.write(b"printf 'EAGAIN_WRITE_OK\\n'\n")
        assert b"EAGAIN_WRITE_OK" in _read_until(backend, b"EAGAIN_WRITE_OK")
    finally:
        monkeypatch.undo()


def test_pending_input_bound_rejects_only_before_accepting_new_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_fd, write_fd = os.pipe()
    backend = PosixTerminalBackend()
    backend._master_fd = write_fd
    real_write = os.write

    def always_block(fd: int, data: bytes | memoryview) -> int:
        if fd == write_fd:
            raise BlockingIOError(errno.EAGAIN, "backpressure")
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", always_block)
    try:
        chunk = b"x" * MAX_IO_CHUNK_BYTES
        for _ in range(MAX_PENDING_INPUT_BYTES // len(chunk)):
            backend.write(chunk)
        with pytest.raises(BlockingIOError, match="terminal input backpressure"):
            backend.write(b"x")
    finally:
        monkeypatch.undo()
        backend.request_priority_close()
        _safe_test_close(read_fd)
        _safe_test_close(write_fd)


@pytest.mark.parametrize("operation", ["read", "write", "resize"])
def test_close_proven_serializes_master_syscalls_before_fd_reuse(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    read_fd, write_fd = os.pipe()
    master_fd = read_fd if operation == "read" else write_fd
    peer_fd = write_fd if operation == "read" else read_fd
    reuse_source = os.open(os.devnull, os.O_RDWR)
    original_identity = (os.fstat(master_fd).st_dev, os.fstat(master_fd).st_ino)
    entered = Event()
    release = Event()
    close_started = Event()
    close_reached = Event()
    observed_identities: list[tuple[int, int]] = []
    operation_errors: list[BaseException] = []
    backend = PosixTerminalBackend()
    backend._master_fd = master_fd
    real_read = os.read
    real_write = os.write
    real_ioctl = fcntl.ioctl

    def wait_inside_syscall(fd: int) -> None:
        stat = os.fstat(fd)
        observed_identities.append((stat.st_dev, stat.st_ino))
        entered.set()
        if not release.wait(1.0):
            raise RuntimeError("syscall barrier failed")
        stat = os.fstat(fd)
        observed_identities.append((stat.st_dev, stat.st_ino))

    def blocked_read(fd: int, maximum: int) -> bytes:
        if fd != master_fd:
            return real_read(fd, maximum)
        wait_inside_syscall(fd)
        return b"x"

    def blocked_write(fd: int, data: bytes | memoryview) -> int:
        if fd != master_fd:
            return real_write(fd, data)
        wait_inside_syscall(fd)
        return len(data)

    def blocked_ioctl(fd: int, request: int, argument: object = 0) -> object:
        if fd != master_fd:
            return real_ioctl(fd, request, argument)
        wait_inside_syscall(fd)
        return 0

    def close_and_reuse(fd: int | None) -> None:
        assert fd == master_fd
        close_reached.set()
        os.close(master_fd)
        os.dup2(reuse_source, master_fd)

    def run_operation() -> None:
        try:
            if operation == "read":
                assert backend.read(1) == b"x"
            elif operation == "write":
                backend.write(b"x")
            else:
                backend.resize(80, 24)
        except BaseException as exc:
            operation_errors.append(exc)

    def run_close() -> None:
        close_started.set()
        backend._close_proven()

    if operation == "read":
        monkeypatch.setattr(os, "read", blocked_read)
    elif operation == "write":
        monkeypatch.setattr(os, "write", blocked_write)
    else:
        monkeypatch.setattr(fcntl, "ioctl", blocked_ioctl)
    monkeypatch.setattr(
        "tldw_chatbook.Terminal.posix_backend._safe_close",
        close_and_reuse,
    )
    operation_thread = Thread(target=run_operation)
    close_thread = Thread(target=run_close)
    close_reached_while_blocked = False
    try:
        operation_thread.start()
        assert entered.wait(1.0)
        close_thread.start()
        assert close_started.wait(1.0)
        close_reached_while_blocked = close_reached.wait(0.2)
    finally:
        release.set()
        operation_thread.join(1.0)
        close_thread.join(1.0)
        monkeypatch.undo()
        _safe_test_close(master_fd)
        _safe_test_close(peer_fd)
        _safe_test_close(reuse_source)

    assert operation_thread.is_alive() is False
    assert close_thread.is_alive() is False
    assert operation_errors == []
    assert close_reached_while_blocked is False
    assert observed_identities == [original_identity, original_identity]


def test_nonblocking_read_resize_winch_and_alternate_screen(
    backend: PosixTerminalBackend,
) -> None:
    started = time.monotonic()
    assert backend.read() is None
    assert time.monotonic() - started < 0.05
    assert backend.master_is_nonblocking_for_tests is True

    winch_command = shlex.join([sys.executable, str(TERMINAL_CHILD), "winch"])
    backend.write((winch_command + "\n").encode())
    _read_until(backend, b"WINCH_READY")
    backend.resize(101, 37)
    assert b"WINCH:101x37" in _read_until(backend, b"WINCH:101x37")
    backend.write(b"\n")

    alternate_command = shlex.join([sys.executable, str(TERMINAL_CHILD), "alternate"])
    backend.write((alternate_command + "\n").encode())
    output = _read_until(backend, b"ALT_SCREEN")
    assert b"\x1b[?1049h" in output
    assert b"\x1b[?1049l" in output


def test_runtime_eio_waits_for_complete_zero_owned_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master_fd, peer_fd = os.pipe()
    os.set_blocking(master_fd, False)
    descendant = PosixProcessIdentity(44001, 4.4, 44001, 44001)
    owned = True

    def scan() -> OwnershipScan:
        members = (descendant,) if owned else ()
        return OwnershipScan(members, members, complete=True)

    def terminal_eio(fd: int, _maximum: int) -> bytes:
        assert fd == master_fd
        raise OSError(errno.EIO, "transient PTY EIO")

    backend = PosixTerminalBackend(scan_wrapper=scan)
    backend._master_fd = master_fd
    backend._shell_reaped.set()
    monkeypatch.setattr(os, "read", terminal_eio)
    try:
        assert backend.read() is None
        assert backend._pty_eof is False

        owned = False
        assert backend.read() == b""
        assert backend._pty_eof is True
    finally:
        _safe_test_close(master_fd)
        _safe_test_close(peer_fd)


def test_finalize_shutdown_closes_the_master_without_waiting(
    backend: PosixTerminalBackend,
) -> None:
    master_fd = backend._master_fd
    assert master_fd is not None

    backend.finalize_shutdown()
    backend.finalize_shutdown()

    assert backend._master_fd is None
    with pytest.raises(OSError):
        os.fstat(master_fd)


def test_finalize_shutdown_fences_master_publication_during_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = PosixTerminalBackend(environment_factory=lambda: _environment(tmp_path))
    openpty_entered = Event()
    release_openpty = Event()
    real_openpty = os.openpty
    opened_descriptors: list[int] = []
    outcome: dict[str, object] = {}

    def blocked_openpty() -> tuple[int, int]:
        descriptors = real_openpty()
        opened_descriptors.extend(descriptors)
        openpty_entered.set()
        assert release_openpty.wait(1)
        return descriptors

    def start_backend() -> None:
        try:
            outcome["identity"] = backend.start(
                TerminalLaunchRequest(
                    name="finalized-during-start",
                    shell="bash",
                    start_directory=str(tmp_path),
                    columns=80,
                    rows=24,
                ),
                AdmissionGate(admitted=True, token="finalized-during-start"),
            )
        except Exception as error:
            outcome["error"] = error

    monkeypatch.setattr(os, "openpty", blocked_openpty)
    launch_thread = Thread(target=start_backend)
    try:
        launch_thread.start()
        assert openpty_entered.wait(1)
        backend.finalize_shutdown()
        release_openpty.set()
        launch_thread.join(3)

        assert not launch_thread.is_alive()
        assert isinstance(outcome.get("error"), RuntimeError)
        assert "identity" not in outcome
        assert backend._master_fd is None
        for descriptor in opened_descriptors:
            with pytest.raises(OSError):
                os.fstat(descriptor)
    finally:
        release_openpty.set()
        launch_thread.join(3)
        monkeypatch.undo()
        _cleanup_backend_exact(backend, require_proven=False)


def test_exact_shell_exit_is_singly_reaped_and_pty_reaches_eof(
    backend: PosixTerminalBackend,
) -> None:
    backend.write(b"exit 23\n")
    _read_until(backend, b"exit 23", timeout=3.0)
    assert backend.wait_for_shell_exit(timeout_seconds=5.0) == 23
    assert backend.wait_for_shell_exit(timeout_seconds=0.0) == 23
    _read_to_eof(backend)
    attempt = CleanupAttempt(time.monotonic())
    proof = backend.cleanup(attempt)
    assert proof.process_dead is True
    assert proof.stream_closed is True
    assert backend.shell_reap_count_for_tests == 1
    first, second = backend.zero_scan_times_for_tests[-2:]
    assert second - first >= 0.05
    assert second <= attempt.t0 + 5.0


def test_cleanup_expired_at_entry_does_not_inspect_or_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 105.0
    calls = {"scan": 0, "revalidate": 0, "kill": 0, "killpg": 0, "read": 0}

    def scan() -> OwnershipScan:
        calls["scan"] += 1
        return OwnershipScan((), (), True)

    backend = PosixTerminalBackend(
        monotonic_clock=lambda: now,
        scan_wrapper=scan,
        sleep=lambda _duration: pytest.fail("expired cleanup must not sleep"),
    )

    def revalidate(_identity: PosixProcessIdentity) -> bool:
        calls["revalidate"] += 1
        return True

    def kill(_pid: int, _signum: int) -> None:
        calls["kill"] += 1

    def killpg(_pgid: int, _signum: int) -> None:
        calls["killpg"] += 1

    def read_turn() -> bool:
        calls["read"] += 1
        return False

    monkeypatch.setattr(backend, "_identity_alive", revalidate)
    monkeypatch.setattr(backend, "_buffer_cleanup_turn", read_turn)
    monkeypatch.setattr(backend, "_discard_cleanup_turn", read_turn)
    monkeypatch.setattr(os, "kill", kill)
    monkeypatch.setattr(os, "killpg", killpg)

    proof = backend.cleanup(CleanupAttempt(100.0))

    assert proof == CleanupProof(False, False, False)
    assert calls == {"scan": 0, "revalidate": 0, "kill": 0, "killpg": 0, "read": 0}


def test_cleanup_scan_crossing_deadline_cannot_revalidate_or_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 204.99
    identity = PosixProcessIdentity(43210, 1.0, 43210, 43210)
    calls = {"scan": 0, "revalidate": 0, "kill": 0, "killpg": 0, "read": 0}

    def clock() -> float:
        return now

    def delayed_scan() -> OwnershipScan:
        nonlocal now
        calls["scan"] += 1
        now = 205.0
        return OwnershipScan((identity,), (identity,), True)

    backend = PosixTerminalBackend(
        monotonic_clock=clock,
        scan_wrapper=delayed_scan,
        sleep=lambda _duration: pytest.fail("expired cleanup must not sleep"),
    )

    def revalidate(_identity: PosixProcessIdentity) -> bool:
        calls["revalidate"] += 1
        return True

    def kill(_pid: int, _signum: int) -> None:
        calls["kill"] += 1

    def killpg(_pgid: int, _signum: int) -> None:
        calls["killpg"] += 1

    def read_turn() -> bool:
        calls["read"] += 1
        return False

    monkeypatch.setattr(backend, "_identity_alive", revalidate)
    monkeypatch.setattr(backend, "_buffer_cleanup_turn", read_turn)
    monkeypatch.setattr(backend, "_discard_cleanup_turn", read_turn)
    monkeypatch.setattr(os, "kill", kill)
    monkeypatch.setattr(os, "killpg", killpg)

    proof = backend.cleanup(CleanupAttempt(200.0))

    assert proof == CleanupProof(False, False, False)
    assert calls == {"scan": 1, "revalidate": 0, "kill": 0, "killpg": 0, "read": 0}


def test_wait_stage_expired_at_entry_does_not_scan() -> None:
    scan_calls = 0

    def scan() -> OwnershipScan:
        nonlocal scan_calls
        scan_calls += 1
        return OwnershipScan((), (), True)

    backend = PosixTerminalBackend(
        monotonic_clock=lambda: 300.0,
        scan_wrapper=scan,
        sleep=lambda _duration: pytest.fail("expired stage must not sleep"),
    )

    observed, complete = backend._wait_stage(300.0)

    assert scan_calls == 0
    assert observed == OwnershipScan((), (), False)
    assert complete is False


@pytest.mark.parametrize(
    ("entry_offset", "expected_signals"),
    [
        (CleanupSchedule().hangup_no_later_than, [signal.SIGTERM]),
        (CleanupSchedule().terminate_no_later_than, [signal.SIGKILL]),
        (CleanupSchedule().force_kill_no_later_than, []),
    ],
)
def test_cleanup_skips_signal_stages_expired_at_entry(
    monkeypatch: pytest.MonkeyPatch,
    entry_offset: float,
    expected_signals: list[int],
) -> None:
    t0 = 400.0
    now = t0 + entry_offset
    identity = PosixProcessIdentity(44001, 4.4, 44001, 44001)
    owned = bool(expected_signals)
    signals: list[int] = []

    def clock() -> float:
        return now

    def sleep(duration: float) -> None:
        nonlocal now
        now += max(0.0, duration)

    def scan() -> OwnershipScan:
        members = (identity,) if owned else ()
        return OwnershipScan(members, members, True)

    def killpg(_pgid: int, signum: int) -> None:
        nonlocal owned
        signals.append(signum)
        owned = False

    backend = PosixTerminalBackend(
        monotonic_clock=clock,
        scan_wrapper=scan,
        sleep=sleep,
    )
    backend._shell_reaped.set()
    backend._pty_eof = True
    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(
        os,
        "kill",
        lambda _pid, _signum: pytest.fail("safe group unexpectedly fell back to PID"),
    )

    proof = backend.cleanup(CleanupAttempt(t0))

    assert signals == expected_signals
    assert proof == CleanupProof(True, True, True)


@pytest.mark.parametrize(
    ("entry_offset", "crossed_offset", "expected_signals"),
    [
        (0.74, CleanupSchedule().hangup_no_later_than, [signal.SIGTERM]),
        (
            CleanupSchedule().hangup_no_later_than,
            CleanupSchedule().terminate_no_later_than,
            [signal.SIGKILL],
        ),
        (
            CleanupSchedule().terminate_no_later_than,
            CleanupSchedule().force_kill_no_later_than,
            [],
        ),
    ],
)
def test_cleanup_scan_crossing_stage_boundary_skips_that_signal(
    monkeypatch: pytest.MonkeyPatch,
    entry_offset: float,
    crossed_offset: float,
    expected_signals: list[int],
) -> None:
    t0 = 500.0
    now = t0 + entry_offset
    identity = PosixProcessIdentity(55001, 5.5, 55001, 55001)
    owned = True
    scan_calls = 0
    signals: list[int] = []

    def clock() -> float:
        return now

    def sleep(duration: float) -> None:
        nonlocal now
        now += max(0.0, duration)

    def scan() -> OwnershipScan:
        nonlocal now, owned, scan_calls
        scan_calls += 1
        members = (identity,) if owned else ()
        if scan_calls == 2:
            now = t0 + crossed_offset
            if not expected_signals:
                owned = False
        return OwnershipScan(members, members, True)

    def killpg(_pgid: int, signum: int) -> None:
        nonlocal owned
        signals.append(signum)
        owned = False

    backend = PosixTerminalBackend(
        monotonic_clock=clock,
        scan_wrapper=scan,
        sleep=sleep,
    )
    backend._shell_reaped.set()
    backend._pty_eof = True
    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(
        os,
        "kill",
        lambda _pid, _signum: pytest.fail("safe group unexpectedly fell back to PID"),
    )

    proof = backend.cleanup(CleanupAttempt(t0))

    assert signals == expected_signals
    assert proof == CleanupProof(True, True, True)


def test_cleanup_continuous_output_is_turn_bounded_and_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    now = 100.0
    read_allowed = False
    mode = "continuous"
    read_calls = 0
    real_read = os.read

    def clock() -> float:
        nonlocal now, read_allowed
        read_allowed = True
        now += 0.01
        return now

    def sleep(duration: float) -> None:
        nonlocal now, read_allowed
        read_allowed = True
        now += max(0.0, duration)

    def controlled_read(fd: int, maximum: int) -> bytes:
        nonlocal read_allowed, read_calls
        if fd != read_fd:
            return real_read(fd, maximum)
        assert read_allowed, "cleanup performed more than one read in one turn"
        read_allowed = False
        read_calls += 1
        if mode == "continuous":
            return b"z" * maximum
        return b""

    backend = PosixTerminalBackend(
        monotonic_clock=clock,
        scan_wrapper=lambda: OwnershipScan((), (), True),
        sleep=sleep,
    )
    backend._master_fd = read_fd
    backend._shell_reaped.set()
    monkeypatch.setattr(os, "read", controlled_read)
    try:
        attempt = CleanupAttempt(clock())
        proof = backend.cleanup(attempt)
        assert now <= attempt.t0 + 5.05
        assert read_calls <= 8
        assert proof.process_dead is True
        assert proof.stream_closed is False
        assert proof.output_complete is False

        preserved = backend.read()
        assert preserved == b"z" * MAX_IO_CHUNK_BYTES

        mode = "eof"
        while backend.read() not in (None, b""):
            pass
        retry = CleanupAttempt(clock())
        proof = backend.cleanup(retry)
        assert proof.process_dead is True
        assert proof.stream_closed is True
        assert proof.output_complete is True
    finally:
        monkeypatch.undo()
        _safe_test_close(read_fd)
        _safe_test_close(write_fd)


def test_process_dead_is_independent_of_pty_eof_after_reap_and_two_zero_scans() -> None:
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    now = 200.0

    def clock() -> float:
        nonlocal now
        now += 0.01
        return now

    def sleep(duration: float) -> None:
        nonlocal now
        now += max(0.0, duration)

    backend = PosixTerminalBackend(
        monotonic_clock=clock,
        scan_wrapper=lambda: OwnershipScan((), (), True),
        sleep=sleep,
    )
    backend._master_fd = read_fd
    backend._shell_reaped.set()
    try:
        proof = backend.cleanup(CleanupAttempt(clock()))
        assert len(backend.zero_scan_times_for_tests) >= 2
        assert proof.process_dead is True
        assert proof.stream_closed is False
        assert proof.output_complete is False
    finally:
        _safe_test_close(read_fd)
        _safe_test_close(write_fd)


def test_parser_failure_retains_buffer_without_process_only_proof() -> None:
    now = 250.0
    backend = PosixTerminalBackend(monotonic_clock=lambda: now)
    backend._output_buffer.extend(b"untrusted-buffered-output")

    proof = backend.cleanup_parser_failure(CleanupAttempt(now - 6.0))

    assert proof == CleanupProof(False, False, False)
    assert bytes(backend._output_buffer) == b"untrusted-buffered-output"


def test_parser_failure_discards_buffer_only_after_process_only_proof() -> None:
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    _safe_test_close(write_fd)
    now = 275.0
    scan_buffers: list[bytes] = []
    discard_states: list[tuple[bool, bytes]] = []
    backend: PosixTerminalBackend

    def clock() -> float:
        nonlocal now
        now += 0.01
        return now

    def sleep(duration: float) -> None:
        nonlocal now
        now += max(0.0, duration)

    def scan() -> OwnershipScan:
        scan_buffers.append(bytes(backend._output_buffer))
        return OwnershipScan((), (), True)

    class ObservedBackend(PosixTerminalBackend):
        def _discard_cleanup_turn(self) -> None:
            discard_states.append((self._process_only_dead, bytes(self._output_buffer)))
            super()._discard_cleanup_turn()

    backend = ObservedBackend(
        monotonic_clock=clock,
        scan_wrapper=scan,
        sleep=sleep,
    )
    backend._master_fd = read_fd
    backend._output_buffer.extend(b"untrusted-buffered-output")
    backend._shell_reaped.set()
    attempt = CleanupAttempt(clock())
    try:
        proof = backend.cleanup_parser_failure(attempt)

        assert proof == CleanupProof(True, True, False)
        assert scan_buffers
        assert set(scan_buffers) == {b"untrusted-buffered-output"}
        assert discard_states == [(True, b"")]
        assert backend._last_attempt_t0 == attempt.t0
    finally:
        _safe_test_close(read_fd)


def test_explicit_raw_cleanup_discard_is_one_read_per_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    now = 300.0
    read_allowed = False
    raw_reads = 0
    real_read = os.read

    def clock() -> float:
        nonlocal now, read_allowed
        read_allowed = True
        now += 0.01
        return now

    def sleep(duration: float) -> None:
        nonlocal now, read_allowed
        read_allowed = True
        now += max(0.0, duration)

    def raw_then_eof(fd: int, maximum: int) -> bytes:
        nonlocal read_allowed, raw_reads
        if fd != read_fd:
            return real_read(fd, maximum)
        assert read_allowed, "raw cleanup performed more than one read in one turn"
        read_allowed = False
        raw_reads += 1
        if raw_reads == 1:
            return b"discard-after-parser-failure"
        return b""

    backend = PosixTerminalBackend(
        monotonic_clock=clock,
        scan_wrapper=lambda: OwnershipScan((), (), True),
        sleep=sleep,
    )
    backend._master_fd = read_fd
    backend._shell_reaped.set()
    try:
        first_attempt = CleanupAttempt(clock())
        first_proof = backend.cleanup(first_attempt)
        assert first_proof.process_dead is True
        assert first_proof.stream_closed is False
        monkeypatch.setattr(os, "read", raw_then_eof)
        raw_attempt = CleanupAttempt(clock())
        proof = backend.cleanup_raw_drain(raw_attempt)
        assert raw_reads == 2
        assert proof.process_dead is True
        assert proof.stream_closed is True
        assert proof.output_complete is False
    finally:
        monkeypatch.undo()
        _safe_test_close(read_fd)
        _safe_test_close(write_fd)


def test_manager_parser_failure_closes_direct_flood_under_original_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    flood_bytes = MAX_PENDING_OUTPUT_BYTES + (64 * 1024)
    ready_file = tmp_path / "parser-flood.ready"
    feed_sizes: list[int] = []
    snapshot_calls = 0
    parser_attempts: list[CleanupAttempt] = []
    parser_proofs: list[CleanupProof] = []
    parser_cleanup_entered = Event()
    allow_parser_cleanup = Event()
    healthy_cleanup_turns = 0
    raw_turn_process_states: list[tuple[bool, bool]] = []
    raw_read_sizes: list[int] = []
    backend: PosixTerminalBackend | None = None
    real_read = os.read

    class FailImmediately:
        failure_reason = None

        def feed(self, data: bytes) -> None:
            feed_sizes.append(len(data))
            raise RuntimeError("terminal parser failed")

        def resize(self, *, columns: int, rows: int) -> None:
            del columns, rows

        def snapshot(self) -> object:
            nonlocal snapshot_calls
            snapshot_calls += 1
            return object()

        def take_pending_replies(self) -> tuple[bytes, ...]:
            return ()

    class ObservedPosixBackend(PosixTerminalBackend):
        recording_raw_turn = False

        def _buffer_cleanup_turn(self) -> None:
            nonlocal healthy_cleanup_turns
            healthy_cleanup_turns += 1
            super()._buffer_cleanup_turn()

        def _discard_cleanup_turn(self) -> None:
            identity = self.identity_for_tests
            raw_turn_process_states.append(
                (
                    self._process_only_dead,
                    _pid_matches(identity.pid, identity.birth_time),
                )
            )
            self.recording_raw_turn = True
            try:
                super()._discard_cleanup_turn()
            finally:
                self.recording_raw_turn = False

        def cleanup_parser_failure(self, attempt: CleanupAttempt) -> CleanupProof:
            parser_attempts.append(attempt)
            parser_cleanup_entered.set()
            if not allow_parser_cleanup.wait(1.0):
                raise RuntimeError("parser cleanup barrier failed")
            proof = super().cleanup_parser_failure(attempt)
            parser_proofs.append(proof)
            return proof

    def backend_factory() -> PosixTerminalBackend:
        nonlocal backend
        fixture_choice = _new_shell_choice(
            key="fixture",
            label="Parser failure fixture",
            family="test",
            executable=Path(sys.executable),
            argv=(
                sys.executable,
                str(TERMINAL_CHILD),
                "parser-flood",
                str(flood_bytes),
                str(ready_file),
            ),
        )
        backend = ObservedPosixBackend(
            environment_factory=lambda: _environment(tmp_path),
            shell_choices_factory=lambda: (fixture_choice,),
        )
        return backend

    terminal = TerminalSessionManager(
        lambda: True,
        backend_factory,
        screen_model_factory=lambda _columns, _rows: FailImmediately(),
    )
    terminal.arm(acknowledge_disclosure=True)
    result = terminal.create_session(
        TerminalLaunchRequest(
            name="large-parser-failure",
            shell="fixture",
            start_directory=str(tmp_path),
            columns=80,
            rows=24,
        )
    )
    assert result.admitted is True
    assert result.projection is not None
    session_id = result.projection.session_id
    assert backend is not None
    identity = backend.identity_for_tests

    def record_raw_read(fd: int, maximum: int) -> bytes:
        if backend is not None and backend.recording_raw_turn:
            raw_read_sizes.append(maximum)
        return real_read(fd, maximum)

    monkeypatch.setattr(os, "read", record_raw_read)
    try:
        assert parser_cleanup_entered.wait(2.0)
        assert ready_file.exists()

        receipt = terminal.cleanup_receipt(session_id)
        assert receipt is not None
        assert parser_attempts == [receipt.attempt]
        pending_bytes, next_read_size = terminal.output_actor_accounting_for_tests(
            session_id
        )
        assert 0 <= pending_bytes <= MAX_PENDING_OUTPUT_BYTES
        assert next_read_size == min(
            MAX_IO_CHUNK_BYTES,
            MAX_PENDING_OUTPUT_BYTES - pending_bytes,
        )
        assert feed_sizes == [MAX_PARSER_SLICE_BYTES]
        assert snapshot_calls == 0
        assert _pid_matches(identity.pid, identity.birth_time)

        allow_parser_cleanup.set()
        assert terminal.wait_for_cleanup(session_id, timeout_seconds=6.0)
        assert parser_attempts == [receipt.attempt]
        assert parser_proofs == [CleanupProof(True, True, False)]
        assert healthy_cleanup_turns == 0
        assert raw_turn_process_states
        assert all(
            process_only and not process_alive
            for process_only, process_alive in raw_turn_process_states
        )
        assert raw_read_sizes
        assert len(raw_read_sizes) == len(raw_turn_process_states)
        assert all(0 < maximum <= MAX_IO_CHUNK_BYTES for maximum in raw_read_sizes)
        assert backend.wait_for_shell_exit(timeout_seconds=1.0) is not None
        assert backend.shell_reap_count_for_tests == 1
        assert not _pid_matches(identity.pid, identity.birth_time)
        assert backend.read() == b""
        assert terminal.projection(session_id) is None
    finally:
        allow_parser_cleanup.set()
        monkeypatch.undo()
        _cleanup_backend_exact(backend)


def test_manager_hands_cleanup_tail_to_screen_before_output_is_complete(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    holder_pid_file = tmp_path / "cleanup-tail-holder.pid"
    after_shell = tmp_path / "cleanup-tail-after-shell.json"
    release_tail = tmp_path / "cleanup-tail-release"
    tail_written = tmp_path / "cleanup-tail-written"
    handed_off: list[bytes] = []
    cleanup_started = Event()
    allow_cleanup = Event()
    backend: PosixTerminalBackend | None = None
    holder_identity: tuple[int, float] | None = None

    class ObservedBackend(PosixTerminalBackend):
        def cleanup(self, attempt: CleanupAttempt) -> CleanupProof:
            cleanup_started.set()
            if not allow_cleanup.wait(2.0):
                return CleanupProof()
            return super().cleanup(attempt)

        def take_preserved_cleanup_output(self, maximum: int) -> bytes:
            chunk = super().take_preserved_cleanup_output(maximum)
            if chunk:
                handed_off.append(chunk)
            return chunk

    def backend_factory() -> PosixTerminalBackend:
        nonlocal backend
        backend = ObservedBackend(
            environment_factory=lambda: _environment(home),
        )
        return backend

    terminal = TerminalSessionManager(lambda: True, backend_factory)
    terminal.arm(acknowledge_disclosure=True)
    result = terminal.create_session(
        TerminalLaunchRequest(
            name="cleanup-tail",
            shell="bash",
            start_directory=str(tmp_path),
            columns=80,
            rows=24,
        )
    )
    assert result.admitted is True
    assert result.projection is not None
    session_id = result.projection.session_id
    assert backend is not None
    identity = backend.identity_for_tests
    view = terminal.attach_view()
    try:
        assert _wait_for(
            lambda: _manager_screen_contains(terminal, view, "$"),
            timeout=3.0,
        )
        command = shlex.join(
            [
                sys.executable,
                str(DESCENDANT_HOLDS_TTY),
                str(holder_pid_file),
                str(after_shell),
                "cleanup-tail-marker",
                str(release_tail),
                str(tail_written),
            ]
        )
        assert terminal.send_paste(
            session_id,
            command,
            bracketed=False,
            view=view,
        ).accepted
        assert terminal.send_key(session_id, b"\r", view=view).accepted
        holder_pid = _wait_for_positive_pid(holder_pid_file, timeout=3.0)
        assert holder_pid is not None
        assert _wait_for(
            lambda: _manager_screen_contains(terminal, view, "$"),
            timeout=3.0,
        )
        assert terminal.send_paste(
            session_id,
            "exit 19",
            bracketed=False,
            view=view,
        ).accepted
        assert terminal.send_key(session_id, b"\r", view=view).accepted
        assert _wait_for(
            lambda: (
                terminal.projection(session_id) is not None
                and terminal.projection(session_id).exit_code == 19
            ),
            timeout=3.0,
        )
        holder_identity = _capture_exact(holder_pid)
        assert holder_identity is not None
        assert _wait_for(after_shell.exists, timeout=3.0)
        descriptor_state = json.loads(after_shell.read_text(encoding="utf-8"))
        assert descriptor_state["held_slave_open"] is True
        assert descriptor_state["held_slave_tty"] is True

        receipt = terminal.cleanup_receipt(session_id)
        assert receipt is not None
        assert cleanup_started.wait(1.0)

        def runtime_stopped() -> bool:
            with terminal._lock:
                record = terminal._sessions.get(session_id)
                thread = None if record is None else record.runtime_thread
            return thread is not None and not thread.is_alive()

        assert _wait_for(runtime_stopped, timeout=1.0)
        assert handed_off == []
        assert _pid_matches(*holder_identity)
        release_tail.touch()
        assert _wait_for(tail_written.exists, timeout=2.0)
        allow_cleanup.set()
        assert terminal.wait_for_cleanup(session_id, timeout_seconds=6.0)

        projection = terminal.projection(session_id)
        assert projection is not None
        assert projection.lifecycle is TerminalLifecycle.EXITED
        assert projection.stream_closed is True
        assert projection.output_complete is True
        state = terminal.view_state(view)
        assert state is not None
        assert len(state.sessions) == 1
        visible = "\n".join(line.text for line in state.sessions[0].screen.lines)
        assert "cleanup-tail-marker" in visible
        assert b"cleanup-tail-marker" in b"".join(handed_off)
        assert backend._output_buffer == b""
        assert backend.shell_reap_count_for_tests == 1
        assert not _pid_matches(identity.pid, identity.birth_time)
        assert not _pid_matches(*holder_identity)
    finally:
        release_tail.touch(exist_ok=True)
        allow_cleanup.set()
        extra = () if holder_identity is None else (holder_identity,)
        _cleanup_backend_exact(backend, extra)


def test_unrelated_darwin_enumeration_denial_fails_process_proof_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = _backend(tmp_path)
    _read_until(backend, b"$", timeout=3.0)
    shell_pid = backend.identity_for_tests.pid
    real_pids = psutil.pids
    real_getsid = os.getsid
    real_killpg = os.killpg
    synthetic_pid = max(real_pids(), default=1) + 1_000_000
    denial_calls = 0
    killpg_calls: list[tuple[int, int]] = []

    def pids_with_unrelated_denied_process() -> list[int]:
        pids = real_pids()
        assert synthetic_pid not in pids
        return [*pids, synthetic_pid]

    def deny_only_unrelated_sid(pid: int) -> int:
        nonlocal denial_calls
        if pid == synthetic_pid:
            denial_calls += 1
            raise PermissionError
        return real_getsid(pid)

    def record_killpg(pgid: int, signum: int) -> None:
        killpg_calls.append((pgid, signum))
        real_killpg(pgid, signum)

    monkeypatch.setattr(psutil, "pids", pids_with_unrelated_denied_process)
    monkeypatch.setattr(os, "getsid", deny_only_unrelated_sid)
    monkeypatch.setattr(os, "killpg", record_killpg)
    try:
        denied_scan = backend.default_scan_for_tests()
        denied_plan = _plan_signals(denied_scan)
        assert denial_calls >= 1
        assert denied_scan.complete is False
        assert denied_scan.group_membership_complete is False
        assert shell_pid in {identity.pid for identity in denied_scan.owned}
        assert denied_plan.group_ids == ()
        assert shell_pid in {identity.pid for identity in denied_plan.individuals}

        started = time.monotonic()
        backend.request_priority_close()
        proof = backend.cleanup(CleanupAttempt(started))
        elapsed = time.monotonic() - started

        assert backend.wait_for_shell_exit(timeout_seconds=1.0) is not None
        assert proof.stream_closed is True
        assert backend.owned_processes_for_tests() == ()
        assert proof.process_dead is False
        assert elapsed < 0.75
        assert denial_calls >= 2
        assert killpg_calls == []
    finally:
        monkeypatch.undo()
        _cleanup_backend_exact(backend)


def test_gone_tracked_descendant_pid_reuse_does_not_poison_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 650.0
    admitted = PosixProcessIdentity(66001, 6.6, 66001, 66001)
    exited_child = PosixProcessIdentity(66002, 6.7, 66001, 66002)
    reused_birth = 9.9
    phase = "gone"
    shell_alive = True
    reused_candidate_reads = 0

    class FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def create_time(self) -> float:
            nonlocal reused_candidate_reads
            if self.pid == admitted.pid and shell_alive:
                return admitted.birth_time
            if self.pid == exited_child.pid and phase == "reused":
                reused_candidate_reads += 1
                return reused_birth
            raise psutil.NoSuchProcess(self.pid)

    def pids() -> list[int]:
        values = [exited_child.pid] if phase == "reused" else []
        if shell_alive:
            values.append(admitted.pid)
        return values

    def getsid(pid: int) -> int:
        if pid == admitted.pid and shell_alive:
            return admitted.sid
        if pid == exited_child.pid and phase == "reused":
            return 99001
        raise ProcessLookupError

    def getpgid(pid: int) -> int:
        if pid == admitted.pid and shell_alive:
            return admitted.initial_pgid
        if pid == exited_child.pid and phase == "reused":
            return 99002
        raise ProcessLookupError

    def terminate_group(_pgid: int, _signum: int) -> None:
        nonlocal shell_alive
        shell_alive = False

    def terminate_pid(_pid: int, _signum: int) -> None:
        nonlocal shell_alive
        shell_alive = False

    def sleep(duration: float) -> None:
        nonlocal now
        now += max(0.0, duration)

    backend = PosixTerminalBackend(
        monotonic_clock=lambda: now,
        sleep=sleep,
    )
    backend._identity = admitted
    backend._tracked = {
        admitted.pid: admitted.birth_time,
        exited_child.pid: exited_child.birth_time,
    }
    backend._shell_reaped.set()
    backend._pty_eof = True
    monkeypatch.setattr(psutil, "Process", FakeProcess)
    monkeypatch.setattr(psutil, "pids", pids)
    monkeypatch.setattr(os, "getsid", getsid)
    monkeypatch.setattr(os, "getpgid", getpgid)
    monkeypatch.setattr(os, "killpg", terminate_group)
    monkeypatch.setattr(os, "kill", terminate_pid)

    gone_scan = backend.default_scan_for_tests()
    phase = "reused"
    proof = backend.cleanup(CleanupAttempt(now))

    assert gone_scan.complete is True
    assert proof == CleanupProof(True, True, True)
    assert exited_child.pid not in backend._tracked
    assert reused_candidate_reads == 0


def test_signal_plan_requires_same_birth_leader_and_exclusive_membership() -> None:
    leader = PosixProcessIdentity(401, 4.01, 401, 401)
    member = PosixProcessIdentity(402, 4.02, 401, 401)
    unrelated = PosixProcessIdentity(499, 4.99, 499, 401)
    safe = OwnershipScan((leader, member), (leader, member), complete=True)
    assert _plan_signals(safe).group_ids == (401,)
    assert _plan_signals(safe).individuals == ()

    leader_exited = OwnershipScan((member,), (member,), complete=True)
    assert _plan_signals(leader_exited).group_ids == ()
    assert _plan_signals(leader_exited).individuals == (member,)

    reused_leader = PosixProcessIdentity(401, 9.99, 999, 401)
    numeric_reuse = OwnershipScan(
        (leader, member),
        (reused_leader, member),
        complete=True,
    )
    assert _plan_signals(numeric_reuse).group_ids == ()
    assert _plan_signals(numeric_reuse).individuals == (leader, member)

    mixed = OwnershipScan(
        (leader, member),
        (leader, member, unrelated),
        complete=True,
    )
    assert _plan_signals(mixed).group_ids == ()
    assert _plan_signals(mixed).individuals == (leader, member)

    denied = OwnershipScan((leader, member), (leader, member), complete=False)
    assert _plan_signals(denied).group_ids == ()
    assert _plan_signals(denied).individuals == (leader, member)

    unrelated_denied = OwnershipScan(
        (leader, member),
        (leader, member),
        complete=True,
        group_membership_complete=False,
    )
    assert _plan_signals(unrelated_denied).group_ids == ()
    assert _plan_signals(unrelated_denied).individuals == (leader, member)


def test_real_foreground_background_groups_are_owned_and_cleaned(
    backend: PosixTerminalBackend,
    tmp_path: Path,
) -> None:
    report = tmp_path / "job-tree.json"
    transition_before = tmp_path / "transition-before.json"
    transition_go = tmp_path / "transition-go"
    transition_after = tmp_path / "transition-after.json"
    known: dict[int, float] = {}
    try:
        command = shlex.join(
            [
                sys.executable,
                str(JOB_CONTROL_TREE),
                str(report),
                str(TERMINAL_CHILD),
                str(transition_before),
                str(transition_go),
                str(transition_after),
            ]
        )
        backend.write((command + "\n").encode())
        _read_until(backend, b"JOB_TREE_READY")
        data = json.loads(report.read_text(encoding="utf-8"))
        assert _wait_for(transition_before.exists)
        before = json.loads(transition_before.read_text(encoding="utf-8"))
        assert data["sid"] == backend.identity_for_tests.sid
        assert data["foreground_pgid"] != data["background_pgid"]
        assert os.getpgid(data["background_member"]) == data["background_pgid"]
        assert before["pid"] == data["transition_member"]
        assert before["birth_time"] == data["transition_birth"]
        assert before["pgid"] == data["background_pgid"]
        expected = {
            data["pid"],
            data["background_leader"],
            data["background_member"],
            data["transition_member"],
        }
        for pid in expected:
            captured = _capture_exact(pid)
            assert captured is not None
            known[captured[0]] = captured[1]
        assert _wait_for(
            lambda: (
                expected
                <= {identity.pid for identity in backend.owned_processes_for_tests()}
            ),
        )
        transition_identity = next(
            item
            for item in backend.owned_processes_for_tests()
            if item.pid == before["pid"]
        )
        assert transition_identity.birth_time == before["birth_time"]
        assert transition_identity.initial_pgid == before["pgid"]

        transition_go.touch()
        assert _wait_for(transition_after.exists)
        after = json.loads(transition_after.read_text(encoding="utf-8"))
        assert after["pid"] == before["pid"]
        assert after["birth_time"] == before["birth_time"]
        assert after["sid"] == before["sid"]
        assert after["pgid"] != before["pgid"]
        assert _wait_for(
            lambda: any(
                item.pid == after["pid"]
                and item.birth_time == after["birth_time"]
                and item.initial_pgid == after["pgid"]
                for item in backend.owned_processes_for_tests()
            )
        )

        backend.request_priority_close()
        proof = backend.cleanup(CleanupAttempt(time.monotonic()))
        assert proof.process_dead is True
        assert proof.stream_closed is True
        assert all(not _pid_matches(pid, known[pid]) for pid in expected)
    finally:
        if report.exists():
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
            for key in (
                "pid",
                "background_leader",
                "background_member",
                "transition_member",
            ):
                pid = data.get(key)
                if type(pid) is int and pid not in known:
                    captured = _capture_exact(pid)
                    if captured is not None:
                        known[captured[0]] = captured[1]
        for pid, birth_time in known.items():
            _terminate_exact(pid, birth_time)


def test_manager_runtime_bridge_disarm_cleans_idle_real_shell(
    tmp_path: Path,
) -> None:
    backend: PosixTerminalBackend | None = None

    def backend_factory() -> PosixTerminalBackend:
        nonlocal backend
        backend = PosixTerminalBackend(
            environment_factory=lambda: _environment(tmp_path),
        )
        return backend

    terminal = TerminalSessionManager(lambda: True, backend_factory)
    terminal.arm(acknowledge_disclosure=True)
    result = terminal.create_session(
        TerminalLaunchRequest(
            name="runtime-disarm",
            shell="default",
            start_directory=str(tmp_path),
            columns=80,
            rows=24,
        )
    )
    assert result.admitted is True
    assert result.projection is not None

    try:
        terminal.disarm()
        assert terminal.wait_for_cleanup(
            result.projection.session_id,
            timeout_seconds=6.0,
        )
        assert terminal.projection(result.projection.session_id) is None
    finally:
        terminal.finalize_shutdown()
        _cleanup_backend_exact(backend, require_proven=False)


def test_shell_exit_with_descendant_holding_slave_is_not_mistaken_for_eof(
    backend: PosixTerminalBackend,
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "holder.pid"
    after_shell = tmp_path / "holder-after-shell.json"
    child_identity: tuple[int, float] | None = None
    try:
        command = shlex.join(
            [
                sys.executable,
                str(DESCENDANT_HOLDS_TTY),
                str(pid_file),
                str(after_shell),
            ]
        )
        backend.write((command + "\n").encode())
        child_pid = _wait_for_positive_pid(pid_file)
        assert child_pid is not None
        child_identity = _capture_exact(child_pid)
        assert child_identity is not None
        child_birth = child_identity[1]
        _read_until(backend, b"$", timeout=3.0)
        backend.write(b"exit 17\n")
        _read_until(backend, b"exit 17", timeout=3.0)
        assert backend.wait_for_shell_exit(timeout_seconds=5.0) == 17
        assert _wait_for(after_shell.exists)
        descriptor_state = json.loads(after_shell.read_text(encoding="utf-8"))
        assert descriptor_state["pid"] == child_pid
        assert all(
            descriptor_state[key]
            for key in (
                "held_slave_open",
                "held_slave_tty",
                "stdin_tty_before",
                "stdout_tty_before",
                "stderr_tty_before",
            )
        )
        assert _pid_matches(child_pid, child_birth)
        backend.read()
        assert child_pid in {
            identity.pid for identity in backend.owned_processes_for_tests()
        }

        proof = backend.cleanup(CleanupAttempt(time.monotonic()))
        assert proof.process_dead is True
        assert proof.stream_closed is True
        assert not _pid_matches(child_pid, child_birth)
    finally:
        if child_identity is None and pid_file.exists():
            try:
                child_pid = int(pid_file.read_text(encoding="ascii"))
            except (OSError, ValueError):
                child_pid = 0
            if child_pid > 0:
                child_identity = _capture_exact(child_pid)
        if child_identity is not None:
            _terminate_exact(*child_identity)


def test_incomplete_enumeration_returns_cleanup_unproven(
    tmp_path: Path,
) -> None:
    real_scan: Callable[..., OwnershipScan] | None = None

    def incomplete_scan(*args: object, **kwargs: object) -> OwnershipScan:
        assert real_scan is not None
        scan = real_scan(*args, **kwargs)
        return OwnershipScan(scan.owned, scan.observed, complete=False)

    backend = PosixTerminalBackend(
        environment_factory=lambda: _environment(tmp_path),
        scan_wrapper=incomplete_scan,
    )
    real_scan = backend.default_scan_for_tests
    try:
        backend.start(
            TerminalLaunchRequest(
                name="denied",
                shell="bash",
                start_directory=str(tmp_path),
                columns=80,
                rows=24,
            ),
            AdmissionGate(admitted=True, token="denied"),
        )
        backend.request_priority_close()
        proof = backend.cleanup(CleanupAttempt(time.monotonic()))
        assert proof.process_dead is False
        assert proof.stream_closed in {False, True}
    finally:
        _cleanup_backend_exact(backend, require_proven=False)


@pytest.mark.parametrize("failure_mode", ["access_denied", "identity_mismatch"])
def test_denied_or_mismatched_identity_returns_cleanup_unproven(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    real_scan: Callable[..., OwnershipScan] | None = None
    calls = 0

    def failed_once(*args: object, **kwargs: object) -> OwnershipScan:
        nonlocal calls
        calls += 1
        assert real_scan is not None
        if calls > 1:
            return real_scan(*args, **kwargs)
        if failure_mode == "access_denied":
            raise psutil.AccessDenied(pid=os.getpid())
        scan = real_scan(*args, **kwargs)
        assert scan.owned
        target, *remainder = scan.owned
        mismatch = PosixProcessIdentity(
            target.pid,
            target.birth_time + 1.0,
            target.sid,
            target.initial_pgid,
        )
        return OwnershipScan((mismatch, *remainder), scan.observed, complete=True)

    backend = PosixTerminalBackend(
        environment_factory=lambda: _environment(tmp_path),
        scan_wrapper=failed_once,
    )
    real_scan = backend.default_scan_for_tests
    try:
        backend.start(
            TerminalLaunchRequest(
                name=failure_mode,
                shell="bash",
                start_directory=str(tmp_path),
                columns=80,
                rows=24,
            ),
            AdmissionGate(admitted=True, token=failure_mode),
        )
        backend.request_priority_close()
        proof = backend.cleanup(CleanupAttempt(time.monotonic()))
        assert proof.process_dead is False
        assert backend.wait_for_shell_exit(timeout_seconds=1.0) is not None
    finally:
        _cleanup_backend_exact(backend, require_proven=False)


def test_app_crash_master_close_hangs_up_ordinary_child_but_not_detached_limit(
    tmp_path: Path,
) -> None:
    report = tmp_path / "crash-report.json"
    ordinary = tmp_path / "ordinary.pid"
    detached = tmp_path / "detached.pid"
    probe_environment = dict(os.environ)
    probe_environment.pop("PYTHONPATH", None)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "Tests.fixtures.terminal.posix_app_crash_probe",
            "--report",
            str(report),
            "--ordinary",
            str(ordinary),
            "--detached",
            str(detached),
            "--fixture",
            str(TERMINAL_CHILD),
        ],
        close_fds=True,
        cwd=REPOSITORY_ROOT,
        env=probe_environment,
    )
    process_birth = psutil.Process(process.pid).create_time()
    known: dict[int, float] = {}
    try:
        assert _wait_for(report.exists, timeout=8.0)
        data = json.loads(report.read_text(encoding="utf-8"))
        shell_pid = data["shell_pid"]
        shell_birth = data["shell_birth"]
        known[shell_pid] = shell_birth
        ordinary_pid = data["ordinary_pid"]
        detached_pid = data["detached_pid"]
        ordinary_birth = data["ordinary_birth"]
        detached_birth = data["detached_birth"]
        known[ordinary_pid] = ordinary_birth
        known[detached_pid] = detached_birth
        assert process.wait(timeout=5.0) == 73
        assert _wait_for(lambda: not _pid_matches(shell_pid, shell_birth))
        assert _wait_for(lambda: not _pid_matches(ordinary_pid, ordinary_birth))
        assert _pid_matches(detached_pid, detached_birth)
    finally:
        _terminate_exact(process.pid, process_birth)
        if process.poll() is None:
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
        if report.exists():
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
            for pid_key, birth_key in (
                ("shell_pid", "shell_birth"),
                ("ordinary_pid", "ordinary_birth"),
                ("detached_pid", "detached_birth"),
            ):
                pid = data.get(pid_key)
                birth_time = data.get(birth_key)
                if type(pid) is int and isinstance(birth_time, (int, float)):
                    known[pid] = float(birth_time)
        for path in (ordinary, detached):
            if not path.exists():
                continue
            try:
                pid = int(path.read_text(encoding="ascii"))
            except (OSError, ValueError):
                continue
            if pid not in known:
                captured = _capture_exact(pid)
                if captured is not None:
                    known[captured[0]] = captured[1]
        for pid, birth_time in known.items():
            _terminate_exact(pid, birth_time)
        if process.poll() is None:
            _terminate_exact(process.pid, process_birth)
            process.wait(timeout=3.0)
