"""Fresh-process completion must not be inferred from its parent's exit."""

import json
import subprocess
import sys
import time
from contextlib import contextmanager

import psutil
import pytest

from Tests.Backup_Recovery.restart_observation import wait_for_restart


@contextmanager
def child_receipt(tmp_path, *, delay=0.25, identity_offset=0):
    child = subprocess.Popen([sys.executable, "-c", f"import time;time.sleep({delay})"])
    try:
        receipt = tmp_path / "restart-process.json"
        receipt.write_text(
            json.dumps(
                {
                    "pid": child.pid,
                    "create_time": psutil.Process(child.pid).create_time()
                    + identity_offset,
                }
            )
        )
        yield child, receipt
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_waits_for_actual_delayed_child_completion(tmp_path):
    with child_receipt(tmp_path) as (child, receipt):
        wait_for_restart(receipt, time.monotonic() + 5)
        assert child.poll() is not None


def test_deadline_kills_only_identified_child(tmp_path):
    with child_receipt(tmp_path, delay=30) as (child, receipt):
        with pytest.raises(TimeoutError):
            wait_for_restart(receipt, time.monotonic() + 0.05)
        assert child.poll() is not None


def test_identity_mismatch_refuses_without_killing_child(tmp_path):
    with child_receipt(tmp_path, delay=30, identity_offset=1) as (child, receipt):
        with pytest.raises(ValueError, match="restart_process_identity"):
            wait_for_restart(receipt, time.monotonic() + 5)
        assert child.poll() is None


def test_missing_receipt_respects_existing_deadline(tmp_path):
    with pytest.raises(TimeoutError, match="restart_receipt_timeout"):
        wait_for_restart(tmp_path / "absent.json", time.monotonic())
