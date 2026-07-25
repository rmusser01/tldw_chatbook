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
   bounded reader thread per stream retains at most ``output_cap_bytes``
   while continuing to drain (and discard) the excess.
3. No ``RLIMIT_NPROC``. It is enforced per real-UID across the whole
   session, not per process tree, so an absolute cap makes the child's
   first fork fail on any desktop that already exceeds it.

Teardown is the subtle part. The direct child exiting does NOT mean the run
is over: a descendant it left behind still holds the write end of the pipes,
so the reader threads never see EOF. Every path therefore SIGKILLs the whole
process group before returning — that both releases the pipes and guarantees
a run leaves no descendants behind — and the reader threads own their own
streams so the caller never closes a pipe out from under a blocked reader.
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

_SANDBOX_UNSUPPORTED_MESSAGE = (
    "Skill script execution is POSIX-only (macOS/Linux) and is not "
    "available on this platform."
)

#: How long teardown may spend reaping the SIGKILLed child.
_REAP_TIMEOUT_SECONDS = 2.0
#: How long teardown may wait for a reader thread to drain and finish.
_READER_JOIN_GRACE_SECONDS = 2.0
_POLL_INTERVAL_SECONDS = 0.02
_READ_CHUNK_BYTES = 4096

_TRAMPOLINE = """
import os, resource, sys
cpu, addr_space, nofile, fsize = (int(v) for v in sys.argv[1:5])
target = sys.argv[5:]
if not target or not target[0]:
    sys.stderr.write("skill-script-runner: no target executable supplied\\n")
    raise SystemExit(2)
resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
try:
    resource.setrlimit(resource.RLIMIT_AS, (addr_space, addr_space))
except (ValueError, OSError):
    pass  # Darwin/BSD alias RLIMIT_AS to RSS and refuse to lower it.
try:
    os.execv(target[0], target)
except OSError as exc:
    sys.stderr.write("skill-script-runner: cannot execute %r: %s\\n" % (target[0], exc))
    raise SystemExit(127)
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
    """Outcome of one sandboxed script run.

    ``exit_code`` is None only when the child could not be reaped even after
    SIGKILL; ``sandbox_warnings`` then carries an explicit explanation rather
    than the caller having to guess what a None means.
    """

    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    output_capped: bool
    duration_seconds: float
    truncated_stdout: bool
    truncated_stderr: bool
    sandbox_warnings: tuple[str, ...] = field(default=())


class SandboxUnsupportedError(RuntimeError):
    """Raised when a skill script run is attempted on an unsupported platform.

    The sandbox this module builds depends on POSIX-only primitives
    (``start_new_session=True``, ``os.killpg``/``os.getpgid``, and a
    trampoline that imports the POSIX-only ``resource`` module), none of
    which exist on Windows. Rather than let those calls fail midway through
    a spawn/teardown sequence, :func:`sandbox_supported` is checked first and
    this error is raised BEFORE any subprocess is started.
    """


def sandbox_supported() -> bool:
    """Return whether this platform can run the skill-script sandbox at all.

    The containment this module provides (process-group teardown via
    ``start_new_session``/``os.killpg``, and resource limits applied through
    the ``resource`` module) is POSIX-only. There is no Windows equivalent
    implemented here, so callers must treat script execution as entirely
    unavailable -- never a degraded/unsandboxed fallback -- on Windows.

    Returns:
        False when ``os.name == "nt"`` (Windows); True otherwise.
    """
    return os.name != "nt"


def memory_limit_enforced() -> bool:
    """Return whether RLIMIT_AS can actually cap memory on this platform.

    Returns:
        False on macOS/BSD, where ``setrlimit(RLIMIT_AS, ...)`` raises and the
        memory cap silently does not apply.
    """
    return platform.system() != "Darwin"


def resolve_interpreter(name: str) -> str | None:
    """Resolve an interpreter without ever consulting ``os.environ['PATH']``.

    Exactly two shapes are accepted:

    1. An absolute path (``/bin/sh``), taken at face value (no search) but
       which must still name an existing, regular, executable file.
    2. A bare name with no path separator (``python3``), searched only on
       :data:`SCRUBBED_PATH` so a poisoned ambient PATH cannot substitute an
       attacker's binary.

    A relative name that contains a separator (``sub/evilpy``) is rejected
    outright rather than resolved: ``shutil.which`` special-cases any
    argument with a dirname component by checking THAT path against the
    process's current working directory and silently ignoring ``path=``, so
    the returned string would not actually have been validated against
    :data:`SCRUBBED_PATH` at all — and it would be checked here against the
    app's cwd while the caller later executes it under a different (scratch)
    cwd, so the file inspected would not be the file run.

    Args:
        name: Interpreter name (``python3``) or absolute path (``/bin/sh``).

    Returns:
        The absolute path, or None when it does not resolve on the scrubbed
        PATH / is not an executable regular file / is a relative name
        containing a path separator (the caller surfaces that as an
        unavailable mechanism rather than falling back to the user's
        environment).
    """
    if not name:
        return None
    if os.path.isabs(name):
        if os.path.isfile(name) and os.access(name, os.X_OK):
            return name
        return None
    if os.sep in name or (os.altsep and os.altsep in name):
        return None
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


class _CappedSink:
    """Thread-safe accumulator that RETAINS at most ``cap`` bytes.

    The reader thread publishes into this as it goes, so bytes already read
    survive even if that thread is still blocked in ``read()`` when teardown
    gives up on it. Memory stays bounded at ``cap`` on every path.
    """

    __slots__ = ("_cap", "_lock", "_buf", "_total")

    def __init__(self, cap: int) -> None:
        self._cap = max(0, cap)
        self._lock = threading.Lock()
        self._buf = bytearray()
        self._total = 0

    def add(self, chunk: bytes) -> None:
        """Record ``chunk``, keeping only what still fits under the cap."""
        with self._lock:
            self._total += len(chunk)
            room = self._cap - len(self._buf)
            if room > 0:
                self._buf += chunk[:room]

    def snapshot(self) -> tuple[bytes, bool]:
        """Return ``(retained_bytes, was_capped)`` as of right now."""
        with self._lock:
            return bytes(self._buf), self._total > self._cap


def _read_capped(stream, sink: _CappedSink) -> None:
    """Drain a child stream to EOF, publishing incrementally into ``sink``.

    Reading deliberately continues past the cap, discarding the excess, so the
    child never blocks writing into a pipe nobody drains. The thread OWNS the
    stream: it closes it here and nowhere else, because ``close()`` blocks on
    the buffered-reader lock a blocked ``read()`` still holds.

    Args:
        stream: The child's stdout/stderr pipe, in binary mode.
        sink: Bounded accumulator that publishes as bytes arrive.
    """
    try:
        while True:
            chunk = stream.read(_READ_CHUNK_BYTES)
            if not chunk:
                break
            sink.add(chunk)
    except (OSError, ValueError):
        pass
    except Exception:  # noqa: BLE001 — a reader thread must never escape
        logger.debug("skill script reader thread failed", exc_info=True)
    finally:
        _close_quietly(stream)


def _close_quietly(stream) -> None:
    try:
        if stream is not None:
            stream.close()
    except Exception:  # noqa: BLE001
        pass


def _kill_group(pgid: int) -> None:
    """SIGKILL a whole process group; an already-dead group is not an error.

    Args:
        pgid: Process-group id. ``start_new_session=True`` makes the child a
            session leader, so this is simply the child's pid.
    """
    if pgid <= 0 or pgid == os.getpgrp():  # never signal our own group
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass  # already gone (or not ours) — nothing left to kill


def _reap(process: subprocess.Popen) -> int | None:
    """Wait for a SIGKILLed child, never leaving a zombie behind.

    Args:
        process: The already-signalled child.

    Returns:
        Its exit status, or None if it did not exit within the reap timeout
        (a background thread keeps waiting so it cannot stay a zombie).
    """
    try:
        return process.wait(timeout=_REAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        logger.error(
            "skill script pid {} did not exit after SIGKILL; reaping in background",
            process.pid,
        )
        threading.Thread(
            target=_background_reap,
            args=(process,),
            name="skill-script-reaper",
            daemon=True,
        ).start()
        return None


def _background_reap(process: subprocess.Popen) -> None:
    try:
        process.wait()
    except Exception:  # noqa: BLE001
        pass


def run_script_subprocess(
    target_argv: list[str],
    *,
    cwd: Path,
    limits: ScriptRunLimits,
) -> ScriptRunResult:
    """Run ``target_argv`` under best-effort containment and capped output.

    The call is bounded by ``limits.wall_clock_seconds`` plus a fixed teardown
    grace — the deadline covers the WHOLE call, not just the wait for the
    direct child, so a script that exits while leaving a long-lived
    descendant behind cannot stall the caller. That grace is, worst case,
    ``_REAP_TIMEOUT_SECONDS`` to reap the SIGKILLed child plus
    ``_READER_JOIN_GRACE_SECONDS`` for EACH of the two reader threads
    (stdout, stderr), joined one after another rather than concurrently: with
    the current defaults that is 2.0 + 2.0 + 2.0 = 6.0 seconds, so the true
    worst-case wall time for one call is ``wall_clock_seconds + 6.0``.

    A run leaves no descendants behind: the child's entire process group is
    SIGKILLed before returning on every path, including a clean exit. That is
    also what releases the pipe write ends the descendants inherited, so the
    output already read is returned instead of being lost to a blocked reader.

    Args:
        target_argv: Full argv of the real target (interpreter + script +
            args, or an executable + args). Never passed to a shell.
        cwd: Scratch working directory (the caller guarantees this is not the
            skill directory).
        limits: Resource/time/output budget.

    Returns:
        A ScriptRunResult. A non-zero exit or a timeout is a normal result,
        not an exception. ``exit_code`` is None only when even SIGKILL did not
        settle the child, in which case ``sandbox_warnings`` says so.

    Raises:
        SandboxUnsupportedError: :func:`sandbox_supported` is False for this
            platform (currently: Windows). Checked FIRST, before anything
            else in this function, so an unsupported platform never reaches
            ``Popen`` or the POSIX-only teardown calls below it.
        ValueError: ``target_argv`` is empty or its first element is not a
            non-empty executable path.
        OSError: The target could not be spawned at all.
    """
    if not sandbox_supported():
        raise SandboxUnsupportedError(_SANDBOX_UNSUPPORTED_MESSAGE)
    if not target_argv or not isinstance(target_argv[0], str) or not target_argv[0]:
        raise ValueError("target_argv must start with a non-empty executable path")

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
    out_sink = _CappedSink(limits.output_cap_bytes)
    err_sink = _CappedSink(limits.output_cap_bytes)

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
    # start_new_session makes the child a session/group leader, so its pgid is
    # its pid. Capture it now, before anything can reap the child: POSIX keeps
    # a pid number reserved for as long as it is still in use as a pgid, so
    # killing this pgid while the child is confirmed not yet reaped (the
    # timeout branch below) cannot hit a recycled, unrelated group. See the
    # comment at the kill site for the one branch where that is not fully
    # guaranteed.
    pgid = process.pid
    streams = (process.stdout, process.stderr)

    readers: list[threading.Thread] = []
    try:
        for stream, sink in zip(streams, (out_sink, err_sink)):
            thread = threading.Thread(
                target=_read_capped,
                args=(stream, sink),
                name="skill-script-reader",
                daemon=True,
            )
            thread.start()
            readers.append(thread)
    except BaseException:
        # Anything that fails after Popen must not leak a live child.
        _kill_group(pgid)
        _reap(process)
        for orphan in streams[len(readers) :]:  # no reader owns these
            _close_quietly(orphan)
        for reader in readers:
            reader.join(timeout=_READER_JOIN_GRACE_SECONDS)
        raise

    timed_out = False
    deadline = started + limits.wall_clock_seconds
    while True:
        if process.poll() is not None:
            break
        if time.monotonic() >= deadline:
            timed_out = True
            break
        time.sleep(_POLL_INTERVAL_SECONDS)

    # Unconditional, including the clean-exit path: surviving descendants hold
    # the pipes open (readers would block forever) and would outlive the run.
    #
    # Deliberately kill before doing any further polling of our own: on the
    # timeout branch the loop's last poll() found the child still running, so
    # it is still unreaped right here and this kill cannot race a recycled
    # pgid. On the clean-exit branch the loop's OWN completion-detecting
    # poll() already reaped the child as a side effect of answering "is it
    # done" — there is no portable peek-without-reap primitive available here
    # to avoid that, so a vanishingly small (pid-wraparound-within-
    # microseconds) window remains on that branch only.
    _kill_group(pgid)
    exit_code = _reap(process)
    if exit_code is None:
        warnings.append(
            "the sandboxed process did not exit after SIGKILL; its exit status "
            "is unknown and its output may be incomplete"
        )
    elif timed_out and exit_code != -signal.SIGKILL:
        # The child can finish in the sliver between the last poll and the
        # deadline check above. The SIGKILL just sent lands regardless (it is
        # a no-op if the child is already a zombie), so it can no longer be
        # used to tell "we killed it" from "it had already finished" — the
        # exit status can: a signal-terminated status of exactly SIGKILL means
        # this kill is what ended it; anything else means it had already
        # exited on its own, so this was not actually a timeout.
        timed_out = False

    for reader in readers:
        reader.join(timeout=_READER_JOIN_GRACE_SECONDS)
    if any(reader.is_alive() for reader in readers):
        logger.warning(
            "skill script reader still blocked after process-group kill; "
            "returning the output captured so far"
        )
        warnings.append(
            "output may be incomplete: a reader was still blocked on the "
            "child's pipe when the run was torn down"
        )

    stdout_bytes, stdout_capped = out_sink.snapshot()
    stderr_bytes, stderr_capped = err_sink.snapshot()
    return ScriptRunResult(
        exit_code=exit_code,
        stdout=stdout_bytes.decode("utf-8", errors="replace"),
        stderr=stderr_bytes.decode("utf-8", errors="replace"),
        timed_out=timed_out,
        output_capped=bool(stdout_capped or stderr_capped),
        duration_seconds=time.monotonic() - started,
        truncated_stdout=stdout_capped,
        truncated_stderr=stderr_capped,
        sandbox_warnings=tuple(warnings),
    )
