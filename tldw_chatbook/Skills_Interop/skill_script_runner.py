"""Sandboxed subprocess execution for skill-bundled scripts.

The single place this app spawns a skill's own code. It knows nothing about
skills, trust, or policy — callers resolve and authorize a script, then hand
this module an argv to run under best-effort containment.

Three deliberate divergences from ``Evals/specialized_runners.py`` (whose
limit VALUES this borrows):

1. No ``preexec_fn``. The Agents runtime runs synchronously on a worker
   thread and bridges tool calls through ``asyncio.run``; running arbitrary
   Python between fork and exec in a multi-threaded process can deadlock.
   ``start_new_session=True`` does the session/process-group setup inside
   CPython's own C fork helper, and the resource limits are applied by a
   Python *trampoline* that ``setrlimit``s in a fresh single-threaded
   process and then ``os.execv``s the real target.
2. No ``communicate()``/``capture_output``. Those read to EOF into memory,
   so a script that spews output OOMs the app before any cap applies. A
   bounded reader thread per stream stops at ``output_cap_bytes``.
3. No ``RLIMIT_NPROC``. It is enforced per real-UID across the whole
   session, not per process tree, so an absolute cap makes the child's
   first fork fail on any desktop that already exceeds it.
"""

from __future__ import annotations

import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

SCRUBBED_PATH = "/usr/bin:/bin"

_TRAMPOLINE = """
import os, resource, sys
cpu, addr_space, nofile, fsize = (int(v) for v in sys.argv[1:5])
target = sys.argv[5:]
resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
try:
    resource.setrlimit(resource.RLIMIT_AS, (addr_space, addr_space))
except (ValueError, OSError):
    pass  # Darwin/BSD alias RLIMIT_AS to RSS and refuse to lower it.
os.execv(target[0], target)
"""


@dataclass(frozen=True)
class ScriptRunLimits:
    """Best-effort containment budget for one script run."""

    cpu_seconds: int = 10
    address_space_bytes: int = 512 * 1024 * 1024
    open_files: int = 128
    file_size_bytes: int = 8 * 1024 * 1024
    wall_clock_seconds: float = 60.0
    output_cap_bytes: int = 65536


@dataclass(frozen=True)
class ScriptRunResult:
    """Outcome of one sandboxed script run."""

    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    output_capped: bool
    duration_seconds: float
    truncated_stdout: bool
    truncated_stderr: bool
    sandbox_warnings: tuple[str, ...] = field(default=())


def memory_limit_enforced() -> bool:
    """Return whether RLIMIT_AS can actually cap memory on this platform.

    Returns:
        False on macOS/BSD, where ``setrlimit(RLIMIT_AS, ...)`` raises and the
        memory cap silently does not apply.
    """
    return platform.system() != "Darwin"


def resolve_interpreter(name: str) -> str | None:
    """Resolve an interpreter against the SCRUBBED PATH, never ``os.environ``.

    Args:
        name: Interpreter name (``python3``) or absolute path (``/bin/sh``).

    Returns:
        The absolute path, or None when it does not resolve on the scrubbed
        PATH (the caller surfaces that as an unavailable mechanism rather
        than falling back to the user's environment).
    """
    if os.path.isabs(name):
        return name if os.path.exists(name) else None
    return shutil.which(name, path=SCRUBBED_PATH)


def _scrubbed_env(cwd: Path) -> dict[str, str]:
    env = {
        "PATH": SCRUBBED_PATH,
        "HOME": str(cwd),
        "TMPDIR": str(cwd),
    }
    for passthrough in ("LANG", "LC_ALL"):
        value = os.environ.get(passthrough)
        if value:
            env[passthrough] = value
    return env


def _read_capped(stream, cap: int, sink: dict) -> None:
    """Read a stream to EOF while KEEPING at most ``cap`` bytes.

    Reading deliberately continues past the cap, discarding the excess, so the
    child never blocks writing into a pipe nobody drains. Memory stays bounded
    at ``cap`` (the OOM property) while a chatty but well-behaved script can
    still run to completion; a script that never stops is bounded by the wall
    clock instead.

    Args:
        stream: The child's stdout/stderr pipe, in binary mode.
        cap: Maximum bytes to retain.
        sink: Mutable dict receiving ``{"data": bytes, "capped": bool}``.
    """
    chunks: list[bytes] = []
    kept = 0
    total = 0
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            total += len(chunk)
            if kept < cap:
                room = cap - kept
                chunks.append(chunk[:room])
                kept += min(len(chunk), room)
    except (OSError, ValueError):
        pass
    finally:
        sink["data"] = b"".join(chunks)
        sink["capped"] = total > cap


def _kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            process.kill()
        except OSError:
            pass


def run_script_subprocess(
    target_argv: list[str],
    *,
    cwd: Path,
    limits: ScriptRunLimits,
) -> ScriptRunResult:
    """Run ``target_argv`` under best-effort containment and capped output.

    Args:
        target_argv: Full argv of the real target (interpreter + script +
            args, or an executable + args). Never passed to a shell.
        cwd: Scratch working directory (the caller guarantees this is not the
            skill directory).
        limits: Resource/time/output budget.

    Returns:
        A ScriptRunResult. A non-zero exit or a timeout is a normal result,
        not an exception.

    Raises:
        OSError: The target could not be spawned at all.
    """
    warnings: list[str] = []
    if not memory_limit_enforced():
        warnings.append(
            "memory (RLIMIT_AS) is not enforced on macOS/BSD; this script is "
            "bounded by CPU and wall-clock time but not by peak memory"
        )

    argv = [
        sys.executable,
        "-c",
        _TRAMPOLINE,
        str(limits.cpu_seconds),
        str(limits.address_space_bytes),
        str(limits.open_files),
        str(limits.file_size_bytes),
        *target_argv,
    ]
    started = time.monotonic()
    process = subprocess.Popen(  # noqa: S603 — argv list, shell=False, scrubbed env
        argv,
        cwd=str(cwd),
        env=_scrubbed_env(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        start_new_session=True,
    )
    out_sink: dict = {"data": b"", "capped": False}
    err_sink: dict = {"data": b"", "capped": False}
    readers = [
        threading.Thread(
            target=_read_capped,
            args=(process.stdout, limits.output_cap_bytes, out_sink),
            daemon=True,
        ),
        threading.Thread(
            target=_read_capped,
            args=(process.stderr, limits.output_cap_bytes, err_sink),
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()

    timed_out = False
    deadline = started + limits.wall_clock_seconds
    while True:
        if process.poll() is not None:
            break
        if time.monotonic() >= deadline:
            timed_out = True
            break
        time.sleep(0.02)

    if process.poll() is None:
        _kill_group(process)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("skill script did not reap after SIGKILL")
    for reader in readers:
        reader.join(timeout=2)
    for stream in (process.stdout, process.stderr):
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            pass

    return ScriptRunResult(
        exit_code=process.returncode,
        stdout=out_sink["data"].decode("utf-8", errors="replace"),
        stderr=err_sink["data"].decode("utf-8", errors="replace"),
        timed_out=timed_out,
        output_capped=bool(out_sink["capped"] or err_sink["capped"]),
        duration_seconds=time.monotonic() - started,
        truncated_stdout=bool(out_sink["capped"]),
        truncated_stderr=bool(err_sink["capped"]),
        sandbox_warnings=tuple(warnings),
    )
