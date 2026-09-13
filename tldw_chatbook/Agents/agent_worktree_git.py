"""Private bounded subprocess support for fixed confirmed worktree operations."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import BinaryIO

MAX_OUTPUT = 32 * 1024 * 1024
TIMEOUT = 30.0


class OperationError(Exception):
    """A bounded operation refused or could not establish its outcome."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _cleanup_unproven() -> None:
    from .execution_capacity import current_execution_owner

    owner = current_execution_owner()
    if owner is not None:
        owner.mark_cleanup_unproven()


def run_git(root: Path, *args: str, output: BinaryIO | None = None) -> bytes:
    """Run internal argv, capping both pipes and retaining physical completion."""
    env = {
        "PATH": os.defpath,
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_LITERAL_PATHSPECS": "1",
    }
    for key in ("HOME", "XDG_CONFIG_HOME", "SYSTEMROOT", "TMPDIR"):
        value = os.environ.get(key)
        if value and Path(value).is_absolute() and not Path(value).is_relative_to(root):
            env[key] = value
    executable = shutil.which("git", path=os.defpath)
    if executable is None:
        raise OperationError("git_unavailable", "Git is unavailable.")
    data = [bytearray(), bytearray()]
    overflow = threading.Event()
    failed = threading.Event()

    def read_pipe(pipe, index):
        count = 0
        try:
            with pipe:
                while chunk := pipe.read(65536):
                    count += len(chunk)
                    if count > MAX_OUTPUT:
                        overflow.set()
                        continue
                    if index == 0 and output is not None:
                        output.write(chunk)
                    else:
                        data[index].extend(chunk)
        except (OSError, ValueError):
            failed.set()

    readers = []
    started_readers = []
    problem = None
    proc = subprocess.Popen(
        [
            executable,
            "-c",
            f"core.hooksPath={os.devnull}",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "merge.gpgsign=false",
            *args,
        ],
        cwd=root,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=os.name == "posix",
    )
    try:
        for index, pipe in enumerate((proc.stdout, proc.stderr)):
            reader = threading.Thread(target=read_pipe, args=(pipe, index), daemon=True)
            readers.append((reader, pipe))
            try:
                reader.start()
            finally:
                # A start implementation may raise after admitting its thread.
                if reader.ident is not None:
                    started_readers.append(reader)
        deadline = time.monotonic() + TIMEOUT
        while proc.poll() is None or any(reader.is_alive() for reader in started_readers):
            if overflow.is_set() or failed.is_set() or time.monotonic() > deadline:
                problem = "output_limit" if overflow.is_set() else "git_interrupted"
                break
            time.sleep(0.01)
    finally:
        descendants = False
        if os.name == "posix":
            try:
                os.killpg(proc.pid, 0)
                descendants = True
            except ProcessLookupError:
                pass
            except OSError:
                _cleanup_unproven()
                problem = "cleanup_unproven"
        if problem or proc.poll() is None or descendants:
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
                    _cleanup_unproven()
            except ProcessLookupError:
                pass
            except (OSError, subprocess.TimeoutExpired):
                _cleanup_unproven()
                problem = "cleanup_unproven"
            try:
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                _cleanup_unproven()
                problem = "cleanup_unproven"
            if os.name == "posix":
                cleanup_deadline = time.monotonic() + 1
                while True:
                    try:
                        os.killpg(proc.pid, 0)
                    except ProcessLookupError:
                        break
                    except OSError:
                        _cleanup_unproven()
                        problem = "cleanup_unproven"
                        break
                    if time.monotonic() >= cleanup_deadline:
                        _cleanup_unproven()
                        problem = "cleanup_unproven"
                        break
                    time.sleep(0.01)
        for reader in started_readers:
            reader.join(timeout=1)
        for pipe in (proc.stdout, proc.stderr):
            # Closing a BufferedReader held by a live reader can block forever.
            if any(reader.is_alive() and owned is pipe for reader, owned in readers):
                continue
            try:
                pipe.close()
            except (OSError, ValueError):
                _cleanup_unproven()
                problem = "cleanup_unproven"
        if any(reader.is_alive() for reader in started_readers):
            _cleanup_unproven()
            problem = "cleanup_unproven"
    if problem:
        raise OperationError(
            problem, "Git exceeded its output/time limit; work is retained."
        )
    if overflow.is_set() or failed.is_set():
        raise OperationError(
            "output_limit", "Git output could not be captured within the limit."
        )
    if proc.returncode:
        raise OperationError(
            "git_failed", bytes(data[1]).decode("utf-8", "replace")[:400]
        )
    return bytes(data[0])
