"""Bounded observation of the fresh Windows recovery test process."""

import json
import math
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
