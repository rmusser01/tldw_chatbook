"""The real restore child uses bounded diagnostics without a C watchdog."""

import faulthandler
import json
import threading
import time
from pathlib import Path

import pytest

from Tests import network_guard
from Tests.Backup_Recovery import thread_diagnostics
from Tests.Backup_Recovery.test_eval_retained_definitions import _RETAIN_RESTORE


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_restore_diagnostics_preserve_failure_and_retire_observer(
    tmp_path, monkeypatch, cleanup_fails
):
    failure = RuntimeError("synthetic restore failure")
    path = tmp_path / "restore-stacks.json"
    original_observer = thread_diagnostics.observe_threads
    enabled_before = faulthandler.is_enabled()
    stopped = []

    def reject_c_watchdog(*args, **kwargs):
        raise AssertionError("restore must not start the racy C watchdog")

    def observe(selected, *, interval):
        assert selected == path and interval == 10
        stop = original_observer(selected, interval=0.01)

        def finish():
            stop()
            stopped.append(True)
            if cleanup_fails:
                raise OSError("synthetic diagnostic cleanup failure")

        return finish

    def enter_restore():
        deadline = time.monotonic() + 5
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert path.exists(), "restore must retain bounded stack diagnostics"
        raise failure

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(faulthandler, "dump_traceback_later", reject_c_watchdog)
    monkeypatch.setattr(thread_diagnostics, "observe_threads", observe)
    # Stop the actual child script at its first product entry, before file/network work.
    monkeypatch.setattr(network_guard, "install", enter_restore)
    with pytest.raises(RuntimeError) as caught:
        exec(compile(_RETAIN_RESTORE, "retained-eval-restore", "exec"), {})  # noqa: S102  # nosec B102
    assert caught.value is failure and stopped == [True]
    assert faulthandler.is_enabled() is enabled_before
    assert not any(t.name == "test-thread-diagnostics" for t in threading.enumerate())
    data = json.loads(path.read_text())
    assert 1 <= len(data) <= 4
    assert all(len(snapshot) <= 32 for snapshot in data)
    assert all(len(row["frames"]) <= 64 for snapshot in data for row in snapshot)
    assert "synthetic restore failure" not in path.read_text()
    assert (tmp_path / "restore-stacks-fatal.log").exists()
    assert bool(getattr(failure, "__notes__", ())) is cleanup_fails
