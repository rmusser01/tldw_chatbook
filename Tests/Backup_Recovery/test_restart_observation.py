"""Fresh-process completion must not be inferred from its parent's exit."""

import json
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext

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


@pytest.mark.parametrize("outcome", ["success", "failure", "timeout", "parent_timeout"])
def test_handoff_observes_fresh_child_after_parent_exit(tmp_path, outcome):
    """The handoff harness must await, reject or retire the actual descendant."""
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    child_script = {
        "success": "import time;time.sleep(.25);print('retired and reopened',flush=True)",
        "failure": "import time;time.sleep(.25);raise SystemExit(7)",
        "timeout": "import time;time.sleep(30)",
        "parent_timeout": "import time;time.sleep(30)",
    }[outcome]
    script = """
import json,os,subprocess,sys,time
from pathlib import Path
import psutil
child=subprocess.Popen([sys.executable,'-c',CHILD_SCRIPT])
receipt=Path.home()/'restart-process.json'
pending=receipt.with_suffix('.tmp')
pending.write_text(json.dumps({'pid':child.pid,'create_time':psutil.Process(child.pid).create_time()}))
pending.replace(receipt)
if sys.argv[2]=='parent_timeout':time.sleep(30)
os._exit(0)
"""
    expectation = {
        "success": nullcontext(),
        "failure": pytest.raises(AssertionError),
        "timeout": pytest.raises(TimeoutError, match="restart_process_timeout"),
        "parent_timeout": pytest.raises(subprocess.TimeoutExpired),
    }[outcome]
    try:
        with expectation:
            _run(
                tmp_path,
                "restart",
                outcome,
                script="CHILD_SCRIPT=" + repr(child_script) + "\n" + script,
                timeout=2,
                expect_restart=True,
            )
        identity = json.loads((tmp_path / "home" / "restart-process.json").read_text())
        try:
            child = psutil.Process(identity["pid"])
            assert child.create_time() == identity["create_time"]
            assert child.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass
    finally:
        receipt = tmp_path / "home" / "restart-process.json"
        if receipt.exists():
            identity = json.loads(receipt.read_text())
            try:
                child = psutil.Process(identity["pid"])
                if child.create_time() == identity["create_time"]:
                    child.kill()
                    child.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass


@pytest.mark.parametrize("mode", ["nonzero", "identity_mismatch", "cleanup_failure"])
def test_parent_failure_preserves_primary_and_safe_child_cleanup(
    tmp_path, monkeypatch, mode
):
    from Tests.Backup_Recovery import restart_observation
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    script = """
import json,os,subprocess,sys
from pathlib import Path
import psutil
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
actual={'pid':child.pid,'create_time':psutil.Process(child.pid).create_time()}
(Path.home()/'actual-review-child.json').write_text(json.dumps(actual))
record=dict(actual)
if MODE=='identity_mismatch':record['create_time']+=1
receipt=Path.home()/'restart-process.json'
pending=receipt.with_suffix('.tmp')
pending.write_text(json.dumps(record))
pending.replace(receipt)
print('synthetic-parent-nonzero',flush=True)
os._exit(7)
"""
    primary = []
    original = restart_observation.wait_for_restart

    def observe_cleanup(*args, **kwargs):
        import sys

        primary.append(sys.exception())
        if mode == "cleanup_failure":
            raise OSError("synthetic-sensitive-cleanup-value")
        return original(*args, **kwargs)

    monkeypatch.setattr(restart_observation, "wait_for_restart", observe_cleanup)
    try:
        with pytest.raises(AssertionError, match="synthetic-parent-nonzero") as caught:
            _run(
                tmp_path,
                "restart",
                mode,
                script="MODE=" + repr(mode) + "\n" + script,
                timeout=2,
                expect_restart=True,
            )
        assert primary == [caught.value]
        identity = json.loads((tmp_path / "home/actual-review-child.json").read_text())
        try:
            child = psutil.Process(identity["pid"])
            assert child.create_time() == identity["create_time"]
            if mode == "nonzero":
                assert child.status() == psutil.STATUS_ZOMBIE
            else:
                assert child.is_running() and child.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            assert mode == "nonzero"
        assert "synthetic-sensitive-cleanup-value" not in str(
            getattr(caught.value, "__notes__", [])
        )
    finally:
        receipt = tmp_path / "home/actual-review-child.json"
        if receipt.exists():
            identity = json.loads(receipt.read_text())
            try:
                child = psutil.Process(identity["pid"])
                if child.create_time() == identity["create_time"]:
                    child.kill()
                    child.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass
