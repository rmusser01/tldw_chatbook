"""Runner unit tests: real subprocesses, real limits, no mocks."""

import os
import shutil
import sys
import textwrap
import time
from pathlib import Path

import pytest

from tldw_chatbook.Skills_Interop.skill_script_runner import (
    ScriptRunLimits,
    resolve_interpreter,
    run_script_subprocess,
)


def _script(tmp_path: Path, body: str, name: str = "s.py") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def _pid_is_dead(pid: int, timeout: float = 10.0) -> bool:
    """Poll until ``pid`` no longer exists (it is reparented, so kill(0) is the probe)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        time.sleep(0.05)
    return False


def test_captures_stdout_and_exit_code(tmp_path):
    script = _script(tmp_path, "print('hello from script')")
    result = run_script_subprocess(
        [sys.executable, str(script)], cwd=tmp_path, limits=ScriptRunLimits()
    )
    assert result.exit_code == 0
    assert "hello from script" in result.stdout
    assert result.timed_out is False


def test_nonzero_exit_is_reported_not_raised(tmp_path):
    script = _script(tmp_path, "import sys; sys.stderr.write('boom'); sys.exit(3)")
    result = run_script_subprocess(
        [sys.executable, str(script)], cwd=tmp_path, limits=ScriptRunLimits()
    )
    assert result.exit_code == 3
    assert "boom" in result.stderr


def test_wall_clock_timeout_kills_whole_process_group(tmp_path):
    """A grandchild must die too: proves start_new_session + killpg."""
    marker = tmp_path / "grandchild.pid"
    script = _script(
        tmp_path,
        f"""
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
        open({str(marker)!r}, 'w').write(str(child.pid))
        time.sleep(120)
        """,
    )
    result = run_script_subprocess(
        [sys.executable, str(script)],
        cwd=tmp_path,
        limits=ScriptRunLimits(wall_clock_seconds=2.0),
    )
    assert result.timed_out is True
    grandchild_pid = int(marker.read_text())
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild_pid, 0)
        except OSError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("grandchild survived the process-group kill")


def test_runaway_output_is_capped_and_does_not_buffer_to_eof(tmp_path):
    """OOM safety: an infinite writer returns BOUNDED output and still terminates.

    The child writes far more than the cap; the runner must retain only ~cap
    bytes (never buffering to EOF) and must come back at the wall clock rather
    than hanging on a full pipe.
    """
    script = _script(
        tmp_path,
        """
        import sys
        chunk = 'x' * 65536
        while True:
            sys.stdout.write(chunk)
            sys.stdout.flush()
        """,
    )
    started = time.monotonic()
    result = run_script_subprocess(
        [sys.executable, str(script)],
        cwd=tmp_path,
        limits=ScriptRunLimits(output_cap_bytes=4096, wall_clock_seconds=5.0),
    )
    elapsed = time.monotonic() - started
    assert result.truncated_stdout is True
    assert result.output_capped is True
    assert len(result.stdout) <= 4096, "retained output must not exceed the cap"
    assert result.timed_out is True
    assert elapsed < 20.0, "must return at the wall clock, not hang on a full pipe"


def test_chatty_but_finite_script_still_completes(tmp_path):
    """Past-the-cap output is discarded, not deadlocked: exit code survives."""
    script = _script(
        tmp_path,
        """
        import sys
        for _ in range(50):
            sys.stdout.write('y' * 4096)
        sys.stdout.flush()
        sys.exit(0)
        """,
    )
    result = run_script_subprocess(
        [sys.executable, str(script)],
        cwd=tmp_path,
        limits=ScriptRunLimits(output_cap_bytes=4096, wall_clock_seconds=30.0),
    )
    assert result.exit_code == 0, "a chatty but finite script must not be killed"
    assert result.timed_out is False
    assert result.truncated_stdout is True
    assert len(result.stdout) <= 4096


def test_environment_is_scrubbed(tmp_path):
    os.environ["TLDW_FAKE_API_KEY"] = "super-secret"
    try:
        script = _script(
            tmp_path,
            "import os; print(os.environ.get('TLDW_FAKE_API_KEY', 'ABSENT')); print(os.environ['PATH'])",
        )
        result = run_script_subprocess(
            [sys.executable, str(script)], cwd=tmp_path, limits=ScriptRunLimits()
        )
    finally:
        os.environ.pop("TLDW_FAKE_API_KEY", None)
    assert "super-secret" not in result.stdout
    assert "ABSENT" in result.stdout
    assert "/usr/bin:/bin" in result.stdout


def test_cwd_is_the_supplied_scratch_dir(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    script = _script(tmp_path, "import os; print(os.path.realpath(os.getcwd()))")
    result = run_script_subprocess(
        [sys.executable, str(script)], cwd=scratch, limits=ScriptRunLimits()
    )
    assert str(scratch.resolve()) in result.stdout


def test_file_size_limit_is_enforced(tmp_path):
    """Positive evidence: the script RAN, reached the write, and was refused."""
    script = _script(
        tmp_path,
        """
        import signal
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)  # surface EFBIG instead of dying
        print('STARTED', flush=True)
        try:
            with open('big.bin', 'wb') as handle:
                handle.write(b'x' * (5 * 1024 * 1024))
                handle.flush()
            print('WROTE', flush=True)
        except OSError:
            print('BLOCKED', flush=True)
        """,
    )
    result = run_script_subprocess(
        [sys.executable, str(script)],
        cwd=tmp_path,
        limits=ScriptRunLimits(file_size_bytes=64 * 1024),
    )
    assert "STARTED" in result.stdout, "the script must actually have run"
    assert "BLOCKED" in result.stdout, "the write must have been refused, not merely absent"
    assert "WROTE" not in result.stdout
    assert (tmp_path / "big.bin").stat().st_size <= 64 * 1024


def test_moderate_forking_is_not_blocked(tmp_path):
    """RLIMIT_NPROC must NOT be set (it is per-UID and breaks busy desktops)."""
    script = _script(
        tmp_path,
        """
        import subprocess, sys
        for _ in range(5):
            subprocess.run([sys.executable, '-c', 'pass'], check=True)
        print('FORKED OK')
        """,
    )
    result = run_script_subprocess(
        [sys.executable, str(script)], cwd=tmp_path, limits=ScriptRunLimits()
    )
    assert "FORKED OK" in result.stdout


def test_limits_apply_to_a_non_python_target(tmp_path):
    """The trampoline must bound direct-exec/shell targets too, not just Python.

    The shell reports the RLIMIT_FSIZE it inherited, so a runner that bypassed
    the trampoline for non-Python targets fails here instead of passing on a
    bare exit code.
    """
    script = _script(tmp_path, "echo shell-ran\nulimit -t\nulimit -f\n", name="s.sh")
    file_size_bytes = 64 * 1024
    cpu_seconds = 7
    result = run_script_subprocess(
        ["/bin/sh", str(script)],
        cwd=tmp_path,
        limits=ScriptRunLimits(file_size_bytes=file_size_bytes, cpu_seconds=cpu_seconds),
    )
    assert result.exit_code == 0
    echoed, cpu_limit, file_limit = result.stdout.split()
    assert echoed == "shell-ran"
    # RLIMIT_CPU is reported in seconds by every shell — no unit ambiguity.
    assert cpu_limit == str(cpu_seconds), f"child did not inherit RLIMIT_CPU: {result.stdout!r}"
    # RLIMIT_FSIZE is reported in blocks: 512 bytes per POSIX, 1024 in bash.
    assert file_limit != "unlimited", f"child did not inherit RLIMIT_FSIZE: {result.stdout!r}"
    blocks = int(file_limit)
    assert blocks * 512 == file_size_bytes or blocks * 1024 == file_size_bytes, (
        f"child RLIMIT_FSIZE of {blocks} blocks does not match {file_size_bytes} bytes"
    )


def test_binary_output_does_not_crash_the_runner(tmp_path):
    script = _script(
        tmp_path,
        "import sys; sys.stdout.buffer.write(b'\\xff\\xfe\\x00binary'); sys.stdout.flush()",
    )
    result = run_script_subprocess(
        [sys.executable, str(script)], cwd=tmp_path, limits=ScriptRunLimits()
    )
    assert result.exit_code == 0
    assert isinstance(result.stdout, str)


def test_resolve_interpreter_uses_scrubbed_path_only(tmp_path, monkeypatch):
    """A poisoned ambient PATH must be invisible to resolution.

    Planting real executables on ``os.environ['PATH']`` is what makes this
    non-vacuous: an implementation that consulted the environment would find
    both of them.
    """
    evil_bin = tmp_path / "evil-bin"
    evil_bin.mkdir()
    planted = evil_bin / "tldw-fake-interpreter"
    planted.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    planted.chmod(0o755)
    shadow = evil_bin / "sh"  # shadows a name that DOES exist on the scrubbed PATH
    shadow.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    shadow.chmod(0o755)
    monkeypatch.setenv("PATH", f"{evil_bin}{os.pathsep}{os.environ.get('PATH', '')}")

    # Sanity: the ambient PATH really does resolve these.
    assert shutil.which("tldw-fake-interpreter") == str(planted)
    assert shutil.which("sh") == str(shadow)

    assert resolve_interpreter("tldw-fake-interpreter") is None
    assert resolve_interpreter("sh") == "/bin/sh"
    assert resolve_interpreter("definitely-not-a-real-interpreter-xyz") is None


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root satisfies os.access(X_OK) regardless of the mode bits",
)
def test_resolve_interpreter_rejects_non_executable_absolute_paths(tmp_path):
    """An absolute path must still be an existing, regular, executable file."""
    data_file = tmp_path / "not-an-interpreter"
    data_file.write_text("just data", encoding="utf-8")

    assert resolve_interpreter(str(data_file)) is None, "non-executable file must not resolve"
    assert resolve_interpreter(str(tmp_path)) is None, "a directory must not resolve"
    assert resolve_interpreter(str(tmp_path / "missing")) is None
    assert resolve_interpreter("") is None

    data_file.chmod(0o755)
    assert resolve_interpreter(str(data_file)) == str(data_file)
    assert resolve_interpreter("/bin/sh") == "/bin/sh"


def test_empty_target_argv_is_a_clear_error(tmp_path):
    for bad in ([], [""]):
        with pytest.raises(ValueError):
            run_script_subprocess(bad, cwd=tmp_path, limits=ScriptRunLimits())


def test_returns_promptly_when_a_grandchild_outlives_the_child(tmp_path):
    """The deadline bounds the WHOLE call, and partial output is never lost.

    The direct child exits immediately but leaves a long-sleeping grandchild
    holding the pipes. The runner must not wait on the grandchild, must still
    return the parent's stdout, and must not leave the grandchild running.
    """
    marker = tmp_path / "grandchild.pid"
    script = _script(
        tmp_path,
        f"""
        import subprocess, sys
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        open({str(marker)!r}, 'w').write(str(child.pid))
        print('PARENT DONE', flush=True)
        """,
    )
    started = time.monotonic()
    result = run_script_subprocess(
        [sys.executable, str(script)],
        cwd=tmp_path,
        limits=ScriptRunLimits(wall_clock_seconds=60.0),
    )
    elapsed = time.monotonic() - started

    assert elapsed < 10.0, "must not block on a grandchild that outlives the child"
    assert "PARENT DONE" in result.stdout, "the child's own output must survive teardown"
    assert result.exit_code == 0
    assert result.timed_out is False
    grandchild_pid = int(marker.read_text())
    assert _pid_is_dead(grandchild_pid), "a run must not leave descendants behind"


def test_short_deadline_bounds_the_call_with_a_live_grandchild(tmp_path):
    """A short wall clock returns near the deadline; the whole tree is dead after."""
    marker = tmp_path / "grandchild.pid"
    script = _script(
        tmp_path,
        f"""
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        open({str(marker)!r}, 'w').write(str(child.pid))
        print('SPAWNED', flush=True)
        time.sleep(30)
        """,
    )
    started = time.monotonic()
    result = run_script_subprocess(
        [sys.executable, str(script)],
        cwd=tmp_path,
        limits=ScriptRunLimits(wall_clock_seconds=3.0),
    )
    elapsed = time.monotonic() - started

    assert result.timed_out is True
    assert elapsed >= 2.5, "must actually honour the deadline, not return early"
    assert elapsed < 10.0, "must return at the deadline plus a bounded teardown grace"
    assert "SPAWNED" in result.stdout, "output read before the deadline must survive"
    grandchild_pid = int(marker.read_text())
    assert _pid_is_dead(grandchild_pid), "a run must not leave descendants behind"
