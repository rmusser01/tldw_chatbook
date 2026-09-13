#!/usr/bin/env python3
"""Create real foreground and background process groups in one PTY session."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


def _matches(pid: int, birth_time: float) -> bool:
    try:
        return psutil.Process(pid).create_time() == birth_time
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False


def _terminate_exact(process: subprocess.Popen[bytes], birth_time: float) -> None:
    if process.poll() is not None or not _matches(process.pid, birth_time):
        return
    process.terminate()
    try:
        process.wait(timeout=0.5)
        return
    except subprocess.TimeoutExpired:
        pass
    if _matches(process.pid, birth_time):
        process.kill()
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        pass


def _exit_from_signal(_signum: int, _frame: object) -> None:
    raise SystemExit(0)


def main() -> int:
    if len(sys.argv) != 6:
        return 2
    report_path = Path(sys.argv[1])
    child_fixture = Path(sys.argv[2])
    transition_before = Path(sys.argv[3])
    transition_go = Path(sys.argv[4])
    transition_after = Path(sys.argv[5])
    leader: subprocess.Popen[bytes] | None = None
    member: subprocess.Popen[bytes] | None = None
    transition_member: subprocess.Popen[bytes] | None = None
    leader_birth: float | None = None
    member_birth: float | None = None
    transition_birth: float | None = None
    try:
        leader = subprocess.Popen(
            [sys.executable, str(child_fixture), "sleep"],
            process_group=0,
            close_fds=True,
        )
        leader_birth = psutil.Process(leader.pid).create_time()
        member = subprocess.Popen(
            [sys.executable, str(child_fixture), "sleep"],
            process_group=leader.pid,
            close_fds=True,
        )
        member_birth = psutil.Process(member.pid).create_time()
        transition_member = subprocess.Popen(
            [
                sys.executable,
                str(child_fixture),
                "pgid-transition",
                str(transition_before),
                str(transition_go),
                str(transition_after),
            ],
            process_group=leader.pid,
            close_fds=True,
        )
        transition_birth = psutil.Process(transition_member.pid).create_time()
        deadline = time.monotonic() + 3.0
        while not transition_before.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not transition_before.exists():
            return 3
        report_path.write_text(
            json.dumps(
                {
                    "background_leader": leader.pid,
                    "background_member": member.pid,
                    "background_pgid": os.getpgid(leader.pid),
                    "foreground_pgid": os.getpgrp(),
                    "pid": os.getpid(),
                    "sid": os.getsid(0),
                    "transition_birth": transition_birth,
                    "transition_member": transition_member.pid,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        signal.signal(signal.SIGHUP, _exit_from_signal)
        signal.signal(signal.SIGTERM, _exit_from_signal)
        sys.stdout.write("JOB_TREE_READY\n")
        sys.stdout.flush()
        signal.pause()
        return 0
    finally:
        if transition_member is not None and transition_birth is not None:
            _terminate_exact(transition_member, transition_birth)
        if member is not None and member_birth is not None:
            _terminate_exact(member, member_birth)
        if leader is not None and leader_birth is not None:
            _terminate_exact(leader, leader_birth)


if __name__ == "__main__":
    raise SystemExit(main())
