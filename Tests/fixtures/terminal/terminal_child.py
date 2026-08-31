#!/usr/bin/env python3
"""Real child-process fixture for POSIX terminal backend tests."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import termios
import time

import psutil


def _window_size() -> tuple[int, int]:
    packed = fcntl.ioctl(0, termios.TIOCGWINSZ, b"\0" * 8)
    rows, columns, _, _ = struct.unpack("HHHH", packed)
    return columns, rows


def _write_json(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()


def _probe(check_fd: int | None) -> int:
    descriptor_closed = True
    if check_fd is not None:
        try:
            os.fstat(check_fd)
        except OSError:
            pass
        else:
            descriptor_closed = False
    columns, rows = _window_size()
    _write_json(
        {
            "columns": columns,
            "cwd": os.getcwd(),
            "descriptor_closed": descriptor_closed,
            "pgid": os.getpgrp(),
            "pid": os.getpid(),
            "rows": rows,
            "sid": os.getsid(0),
            "stderr_tty": os.isatty(2),
            "stdin_tty": os.isatty(0),
            "stdout_tty": os.isatty(1),
            "value": os.environ.get("TERMINAL_CHILD_VALUE", ""),
        }
    )
    return 0


def _sentinel(path: Path, check_fd: int | None) -> int:
    descriptor_closed = True
    if check_fd is not None:
        try:
            os.fstat(check_fd)
        except OSError:
            pass
        else:
            descriptor_closed = False
    path.write_text(
        json.dumps(
            {
                "descriptor_closed": descriptor_closed,
                "pid": os.getpid(),
                "sid": os.getsid(0),
                "stdin_tty": os.isatty(0),
            }
        ),
        encoding="utf-8",
    )
    _write_json({"sentinel": True})
    return 0


def _unicode_echo() -> int:
    sys.stdout.write("UNICODE_READY\n")
    sys.stdout.flush()
    value = sys.stdin.readline().rstrip("\r\n")
    sys.stdout.write(f"UNICODE:{value}\n")
    sys.stdout.flush()
    return 0


def _winch() -> int:
    def report(_signum: int, _frame: object) -> None:
        columns, rows = _window_size()
        os.write(1, f"WINCH:{columns}x{rows}\n".encode("ascii"))

    signal.signal(signal.SIGWINCH, report)
    sys.stdout.write("WINCH_READY\n")
    sys.stdout.flush()
    sys.stdin.buffer.read(1)
    return 0


def _alternate_screen() -> int:
    os.write(1, b"\x1b[?1049hALT_SCREEN\x1b[?1049l\n")
    return 0


def _sleep() -> int:
    while True:
        signal.pause()


def _parser_flood(byte_count: int, ready_file: Path) -> int:
    ready_file.write_text("ready", encoding="ascii")
    chunk = b"x" * (64 * 1024)
    remaining = byte_count
    while remaining:
        written = os.write(1, chunk[:remaining])
        remaining -= written
    os.write(1, b"!")
    while True:
        signal.pause()


def _pgid_transition(before: Path, proceed: Path, after: Path) -> int:
    def exit_on_signal(_signum: int, _frame: object) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGHUP, exit_on_signal)
    signal.signal(signal.SIGTERM, exit_on_signal)
    birth_time = psutil.Process(os.getpid()).create_time()

    def snapshot() -> dict[str, object]:
        return {
            "birth_time": birth_time,
            "pgid": os.getpgrp(),
            "pid": os.getpid(),
            "sid": os.getsid(0),
        }

    before.write_text(json.dumps(snapshot(), sort_keys=True), encoding="utf-8")
    while not proceed.exists():
        time.sleep(0.01)
    os.setpgid(0, 0)
    after.write_text(json.dumps(snapshot(), sort_keys=True), encoding="utf-8")
    while True:
        signal.pause()


def _sighup_wait(ready_file: Path) -> int:
    def exit_on_hangup(_signum: int, _frame: object) -> None:
        raise SystemExit(91)

    signal.signal(signal.SIGHUP, exit_on_hangup)
    ready_file.write_text(str(os.getpid()), encoding="ascii")
    while True:
        signal.pause()


def _spawn_detached(pid_file: Path) -> int:
    process = subprocess.Popen(
        [sys.executable, __file__, "sleep"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    birth_time = psutil.Process(process.pid).create_time()
    try:
        pid_file.write_text(str(process.pid), encoding="ascii")
    except BaseException:
        try:
            current_birth = psutil.Process(process.pid).create_time()
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            current_birth = None
        if current_birth == birth_time:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                if psutil.Process(process.pid).create_time() == birth_time:
                    process.kill()
                process.wait(timeout=0.5)
        raise
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    probe = subparsers.add_parser("probe")
    probe.add_argument("--check-fd", type=int)

    sentinel = subparsers.add_parser("sentinel")
    sentinel.add_argument("path", type=Path)
    sentinel.add_argument("--check-fd", type=int)

    subparsers.add_parser("unicode")
    subparsers.add_parser("winch")
    subparsers.add_parser("alternate")
    subparsers.add_parser("sleep")

    flood = subparsers.add_parser("parser-flood")
    flood.add_argument("byte_count", type=int)
    flood.add_argument("ready_file", type=Path)

    transition = subparsers.add_parser("pgid-transition")
    transition.add_argument("before", type=Path)
    transition.add_argument("proceed", type=Path)
    transition.add_argument("after", type=Path)

    sighup = subparsers.add_parser("sighup")
    sighup.add_argument("ready_file", type=Path)

    detached = subparsers.add_parser("spawn-detached")
    detached.add_argument("pid_file", type=Path)

    arguments = parser.parse_args()
    if arguments.mode == "probe":
        return _probe(arguments.check_fd)
    if arguments.mode == "sentinel":
        return _sentinel(arguments.path, arguments.check_fd)
    if arguments.mode == "unicode":
        return _unicode_echo()
    if arguments.mode == "winch":
        return _winch()
    if arguments.mode == "alternate":
        return _alternate_screen()
    if arguments.mode == "sleep":
        return _sleep()
    if arguments.mode == "parser-flood":
        return _parser_flood(arguments.byte_count, arguments.ready_file)
    if arguments.mode == "pgid-transition":
        return _pgid_transition(
            arguments.before,
            arguments.proceed,
            arguments.after,
        )
    if arguments.mode == "sighup":
        return _sighup_wait(arguments.ready_file)
    if arguments.mode == "spawn-detached":
        return _spawn_detached(arguments.pid_file)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
