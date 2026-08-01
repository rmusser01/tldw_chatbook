#!/usr/bin/env python3
"""Launch a stubborn child tree for native process-containment tests."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def _emit(event: str) -> None:
    payload = {"event": event, "pid": os.getpid(), "ppid": os.getppid()}
    if os.name != "nt":
        payload.update(pgid=os.getpgrp(), sid=os.getsid(0))
    print(json.dumps(payload, sort_keys=True), flush=True)


def _ignore_graceful_termination() -> None:
    if os.name == "nt":
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)
        return
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def _mark_ready(path: str | None, payload: dict[str, int] | None = None) -> None:
    if path is not None:
        target = Path(path)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload or {"pid": os.getpid()}),
            encoding="utf-8",
        )
        os.replace(temporary, target)


def _wait_forever() -> None:
    threading.Event().wait()


def _run_grandchild(
    ignore_termination: bool,
    *,
    ready_file: str | None,
    close_stdio: bool,
) -> None:
    if ignore_termination:
        _ignore_graceful_termination()
    _emit("grandchild_spawned")
    _mark_ready(ready_file)
    if close_stdio:
        sys.stdout.flush()
        sys.stderr.flush()
        os.close(sys.stdout.fileno())
        os.close(sys.stderr.fileno())
    _wait_forever()


def _run_parent(
    ignore_termination: bool,
    *,
    ready_file: str | None,
    parent_exits_after_ready: bool,
    close_grandchild_stdio: bool,
) -> None:
    if ignore_termination:
        _ignore_graceful_termination()
    grandchild_ready = (
        f"{ready_file}.grandchild" if ready_file is not None else None
    )
    grandchild_argv = [
        sys.executable,
        os.path.abspath(__file__),
        "--grandchild",
    ]
    if ignore_termination:
        grandchild_argv.append("--ignore-termination")
    if grandchild_ready is not None:
        grandchild_argv.extend(("--ready-file", grandchild_ready))
    if close_grandchild_stdio:
        grandchild_argv.append("--close-grandchild-stdio")
    grandchild = subprocess.Popen(  # noqa: S603 - fixed interpreter and argv
        grandchild_argv,
        stdin=subprocess.DEVNULL,
    )
    if grandchild_ready is not None:
        deadline = time.monotonic() + 10.0
        while not Path(grandchild_ready).is_file():
            if grandchild.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("grandchild did not publish readiness")
            time.sleep(0.005)
    _emit("parent_spawned")
    print(
        json.dumps(
            {
                "event": "grandchild_pid",
                "pid": grandchild.pid,
                "ppid": os.getpid(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    readiness = {
        "parent_pid": os.getpid(),
        "grandchild_pid": grandchild.pid,
    }
    if os.name != "nt":
        readiness["pgid"] = os.getpgrp()
    _mark_ready(ready_file, readiness)
    if parent_exits_after_ready:
        return
    _wait_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grandchild", action="store_true")
    parser.add_argument("--ignore-termination", action="store_true")
    parser.add_argument("--ready-file")
    parser.add_argument("--parent-exits-after-ready", action="store_true")
    parser.add_argument("--close-grandchild-stdio", action="store_true")
    arguments = parser.parse_args()
    if arguments.grandchild:
        _run_grandchild(
            arguments.ignore_termination,
            ready_file=arguments.ready_file,
            close_stdio=arguments.close_grandchild_stdio,
        )
        return
    _run_parent(
        arguments.ignore_termination,
        ready_file=arguments.ready_file,
        parent_exits_after_ready=arguments.parent_exits_after_ready,
        close_grandchild_stdio=arguments.close_grandchild_stdio,
    )


if __name__ == "__main__":
    main()
