#!/usr/bin/env python3
"""Start a real POSIX terminal, report children, then crash without cleanup."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import sys
import time

import psutil

from tldw_chatbook.Terminal.contracts import (
    AdmissionGate,
    CleanupAttempt,
    TerminalLaunchRequest,
)
from tldw_chatbook.Terminal.posix_backend import PosixTerminalBackend


def _drain_until(
    backend: PosixTerminalBackend,
    needle: bytes,
    timeout: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout
    output = bytearray()
    while needle not in output:
        if time.monotonic() >= deadline:
            raise RuntimeError("probe startup failed")
        chunk = backend.read()
        if chunk is None:
            time.sleep(0.01)
            continue
        if chunk == b"":
            raise RuntimeError("probe startup failed")
        output.extend(chunk)


def _wait_for_file(
    backend: PosixTerminalBackend,
    path: Path,
    timeout: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("probe startup failed")
        backend.read()
        time.sleep(0.01)


def _matches(pid: int, birth_time: float) -> bool:
    try:
        return psutil.Process(pid).create_time() == birth_time
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False


def _terminate_exact(pid: int, birth_time: float) -> None:
    if not _matches(pid, birth_time):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 0.5
    while _matches(pid, birth_time) and time.monotonic() < deadline:
        time.sleep(0.01)
    if not _matches(pid, birth_time):
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--ordinary", required=True, type=Path)
    parser.add_argument("--detached", type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    arguments = parser.parse_args()

    backend = PosixTerminalBackend()
    known: dict[int, float] = {}
    armed = False
    try:
        backend.start(
            TerminalLaunchRequest(
                name="crash-probe",
                shell="bash",
                start_directory=str(arguments.report.parent),
                columns=80,
                rows=24,
            ),
            AdmissionGate(admitted=True, token="crash-probe"),
        )
        shell_identity = backend.identity_for_tests
        known[shell_identity.pid] = shell_identity.birth_time
        _drain_until(backend, b"$")
        detached_pid: int | None = None
        if arguments.detached is not None:
            detached = shlex.join(
                [
                    sys.executable,
                    str(arguments.fixture),
                    "spawn-detached",
                    str(arguments.detached),
                ]
            )
            backend.write((detached + "\n").encode())
            _wait_for_file(backend, arguments.detached)
            detached_pid = int(arguments.detached.read_text(encoding="ascii"))
            known[detached_pid] = psutil.Process(detached_pid).create_time()
        ordinary = shlex.join(
            [
                sys.executable,
                str(arguments.fixture),
                "sighup",
                str(arguments.ordinary),
            ]
        )
        backend.write((ordinary + "\n").encode())
        _wait_for_file(backend, arguments.ordinary)
        ordinary_pid = int(arguments.ordinary.read_text(encoding="ascii"))
        known[ordinary_pid] = psutil.Process(ordinary_pid).create_time()
        arguments.report.write_text(
            json.dumps(
                {
                    "detached_birth": (
                        None if detached_pid is None else known[detached_pid]
                    ),
                    "detached_pid": detached_pid,
                    "ordinary_birth": known[ordinary_pid],
                    "ordinary_pid": ordinary_pid,
                    "shell_birth": shell_identity.birth_time,
                    "shell_pid": shell_identity.pid,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        armed = True
        os._exit(73)
    finally:
        if not armed:
            try:
                backend.request_priority_close()
                backend.cleanup(CleanupAttempt(time.monotonic()))
            except Exception:
                pass
            for pid, birth_time in known.items():
                _terminate_exact(pid, birth_time)


if __name__ == "__main__":
    raise SystemExit(main())
