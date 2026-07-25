"""Runner unit tests: real subprocesses, real limits, no mocks."""

import os
import subprocess
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
    script = _script(
        tmp_path,
        """
        try:
            with open('big.bin', 'wb') as handle:
                handle.write(b'x' * (5 * 1024 * 1024))
                handle.flush()
            print('WROTE')
        except Exception:
            print('BLOCKED')
        """,
    )
    result = run_script_subprocess(
        [sys.executable, str(script)],
        cwd=tmp_path,
        limits=ScriptRunLimits(file_size_bytes=64 * 1024),
    )
    assert "WROTE" not in result.stdout


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
    """The trampoline must bound direct-exec/shell targets too, not just Python."""
    script = _script(tmp_path, "echo shell-ran\n", name="s.sh")
    result = run_script_subprocess(
        ["/bin/sh", str(script)], cwd=tmp_path, limits=ScriptRunLimits()
    )
    assert result.exit_code == 0
    assert "shell-ran" in result.stdout


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


def test_resolve_interpreter_uses_scrubbed_path_only():
    assert resolve_interpreter("sh") == "/bin/sh"
    assert resolve_interpreter("definitely-not-a-real-interpreter-xyz") is None
