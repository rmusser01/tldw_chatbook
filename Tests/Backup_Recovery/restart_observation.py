"""Bounded observation of fixture child processes (restart receipts, CPU-idle hangs)."""

import contextlib
import json
import math
import subprocess
import time
from pathlib import Path

import psutil


def wait_for_restart(receipt: Path, deadline: float) -> int | None:
    """Wait for the process identified by the private fixture receipt."""
    while not receipt.exists():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("restart_receipt_timeout")
        time.sleep(min(0.05, remaining))
    identity = json.loads(receipt.read_text())
    pid, created = identity["pid"], identity["create_time"]
    if type(pid) is not int or pid <= 0:
        raise ValueError("restart_process_identity")
    if type(created) not in (int, float) or not math.isfinite(created) or created <= 0:
        raise ValueError("restart_process_identity")
    try:
        process = psutil.Process(pid)
        if process.create_time() != created:
            raise ValueError("restart_process_identity")
        return process.wait(timeout=max(0, deadline - time.monotonic()))
    except psutil.NoSuchProcess:
        # The caller must still require its existing successful product result.
        return None
    except psutil.TimeoutExpired as error:
        try:
            current = psutil.Process(pid)
            if current.create_time() != created:
                raise ValueError("restart_process_identity")
            current.kill()  # psutil also checks process identity before signalling.
            current.wait(timeout=5)
        except psutil.NoSuchProcess:
            pass
        raise TimeoutError("restart_process_timeout") from error


def _wait_while_busy(process, *, idle, timeout):
    """Wait until ``process`` exits; kill it and its descendants once its process
    tree has used no CPU for ``idle`` seconds (a hang), or at ``timeout``
    (TASK-40306).

    A fixed budget fails a loaded run that is only slow; a blocked child stops
    using CPU, however loaded the host is. A hang that keeps polling (a lock
    wait that retries every few milliseconds) keeps using CPU, so only
    ``timeout`` stops it (TASK-41104).
    """

    def cpu(member):
        # A zombie or a just-exited descendant cannot be read; count the rest.
        try:
            return sum(member.cpu_times()[:2])
        except psutil.Error:
            return 0.0

    watched = psutil.Process(process.pid)
    started = still = time.monotonic()
    used = None
    try:
        while True:
            try:
                return process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            try:
                now_used = sum(
                    cpu(member)
                    for member in (watched, *watched.children(recursive=True))
                )
            except psutil.Error:
                now_used = used  # The child itself exited mid-read.
            now = time.monotonic()
            if now_used != used:
                used, still = now_used, now
            if now - still > idle or now - started > timeout:
                raise subprocess.TimeoutExpired(process.args, round(now - started))
    finally:
        if process.poll() is None:
            try:
                descendants = watched.children(recursive=True)
            except psutil.Error:
                descendants = []
            for member in descendants:
                with contextlib.suppress(psutil.Error):
                    member.kill()
            process.kill()
            process.wait()
