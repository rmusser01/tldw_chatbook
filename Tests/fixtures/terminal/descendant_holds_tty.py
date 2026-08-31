#!/usr/bin/env python3
"""Fork a same-session descendant that deliberately retains the PTY slave."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys
import time


def main() -> int:
    if len(sys.argv) not in (3, 4, 5):
        return 2
    pid_file = Path(sys.argv[1])
    after_shell_file = Path(sys.argv[2])
    tail_output = None if len(sys.argv) == 3 else sys.argv[3].encode("utf-8")
    release_tail = None if len(sys.argv) < 5 else Path(sys.argv[4])
    child_pid = os.fork()
    if child_pid:
        return 0

    held_slave_fd: int | None = None
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, lambda _signum, _frame: raise_exit())
        session_leader = os.getsid(0)
        slave_path = os.ttyname(0)
        tty_before = tuple(os.isatty(descriptor) for descriptor in (0, 1, 2))
        pid_file.write_text(str(os.getpid()), encoding="ascii")
        while True:
            try:
                os.kill(session_leader, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        held_slave_fd = os.open(slave_path, os.O_RDWR | os.O_NOCTTY)
        after_shell_file.write_text(
            json.dumps(
                {
                    "held_slave_open": _descriptor_open(held_slave_fd),
                    "held_slave_tty": os.isatty(held_slave_fd),
                    "pid": os.getpid(),
                    "stderr_open": _descriptor_open(2),
                    "stderr_tty_after": os.isatty(2),
                    "stderr_tty_before": tty_before[2],
                    "stdin_open": _descriptor_open(0),
                    "stdin_tty_after": os.isatty(0),
                    "stdin_tty_before": tty_before[0],
                    "stdout_open": _descriptor_open(1),
                    "stdout_tty_after": os.isatty(1),
                    "stdout_tty_before": tty_before[1],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        if tail_output is not None:
            while release_tail is not None and not release_tail.exists():
                time.sleep(0.01)
            os.write(held_slave_fd, tail_output + b"\n")
        while True:
            signal.pause()
    finally:
        if held_slave_fd is not None:
            os.close(held_slave_fd)


def _descriptor_open(descriptor: int) -> bool:
    try:
        os.fstat(descriptor)
    except OSError:
        return False
    return True


def raise_exit() -> None:
    raise SystemExit(0)


if __name__ == "__main__":
    raise SystemExit(main())
