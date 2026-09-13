"""Private Python-owned diagnostic frame regression checks."""

import faulthandler
import json
import sys
import threading
import time

from Tests.Backup_Recovery.thread_diagnostics import _frames, observe_threads


def test_frame_metadata_is_bounded_and_excludes_locals():
    secret = "synthetic-local-must-not-appear"
    frames = _frames(sys._getframe(), limit=2)
    assert len(frames) <= 2
    assert all(set(frame) == {"file", "function", "line"} for frame in frames)
    assert secret not in json.dumps(frames)
    assert all(
        "/" not in frame["file"] and "\\" not in frame["file"] for frame in frames
    )


def test_periodic_observer_stops_and_retains_bounded_snapshots(tmp_path):
    path = tmp_path / "stacks.log"
    enabled_before = faulthandler.is_enabled()
    stop = observe_threads(path, interval=0.01)
    assert faulthandler.is_enabled()
    try:
        deadline = time.monotonic() + 5
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert path.exists()
    finally:
        stop()
    assert faulthandler.is_enabled() is enabled_before
    records = json.loads(path.read_text())
    assert 1 <= len(records) <= 4
    assert all(len(snapshot) <= 32 for snapshot in records)
    assert all(len(row["frames"]) <= 64 for snapshot in records for row in snapshot)
    assert not any(
        thread.name == "test-thread-diagnostics" for thread in threading.enumerate()
    )
    assert (tmp_path / "stacks-fatal.log").exists()


def test_recovery_error_observer_preserves_mapping_without_error_message(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import tldw_chatbook.Backup_Recovery as package
    from Tests.Backup_Recovery.thread_diagnostics import observe_recovery_failures

    received = []

    def original(error, *, kind):
        received.append((error, kind))
        return "permission_denied"

    class Service:
        issue_code = staticmethod(original)

    module = SimpleNamespace(issue_code=original, RecoveryService=Service)
    original_static = Service.__dict__["issue_code"]
    monkeypatch.setattr(package, "recovery_service", module, raising=False)
    path = tmp_path / "errors.log"
    stop = observe_recovery_failures(path)
    try:
        try:
            raise PermissionError(13, "synthetic-secret-must-not-appear")
        except PermissionError as error:
            error.winerror = 5
            assert module.issue_code(error, kind="restore") == "permission_denied"
            assert Service().issue_code(error, kind="open") == "permission_denied"
            assert received == [(error, "restore"), (error, "open")]
    finally:
        stop()
    record = json.loads(path.read_text())[0]
    assert record["errno"] == 13 and record["winerror"] == 5
    assert record["error_class"] == "PermissionError"
    assert "synthetic-secret-must-not-appear" not in path.read_text()
    assert module.issue_code is original
    assert Service.__dict__["issue_code"] is original_static


def test_inventory_observer_records_caught_failure_without_values(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import tldw_chatbook.Backup_Recovery as package
    from Tests.Backup_Recovery.thread_diagnostics import observe_inventory_failures

    def discover():
        try:
            raise OSError(22, "synthetic-private-value")
        except OSError:
            return SimpleNamespace(
                items=(
                    SimpleNamespace(
                        owner="config",
                        status="unavailable",
                        path=None,
                    ),
                )
            )

    class Service:
        def preview_backup(self, *args, **kwargs):
            return discover()

    monkeypatch.setattr(
        package, "inventory", SimpleNamespace(discover=discover), raising=False
    )
    monkeypatch.setattr(
        package,
        "recovery_service",
        SimpleNamespace(RecoveryService=Service),
        raising=False,
    )
    original = Service.preview_backup
    previous_trace = sys.gettrace()
    path = tmp_path / "inventory.log"
    stop = observe_inventory_failures(path)
    try:
        result = Service().preview_backup()
        assert result.items[0].status == "unavailable"
        assert sys.gettrace() is previous_trace
    finally:
        stop()
    assert Service.preview_backup is original
    record = json.loads(path.read_text())[-1]
    assert record["errors"][0]["error_class"] == "OSError"
    assert record["errors"][0]["errno"] == 22
    assert record["blocking"] == [
        {"owner": "config", "status": "unavailable", "has_path": False}
    ]
    assert "synthetic-private-value" not in path.read_text()


def test_startup_observer_records_only_refusals_and_restores_trace(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import tldw_chatbook.Backup_Recovery as package
    from Tests.Backup_Recovery.thread_diagnostics import observe_startup_refusals

    def permission(refuse, root):
        try:
            raise PermissionError(13, "synthetic-private-selector")
        except PermissionError:
            return (False, "recovery_scope_uncertain") if refuse else (True, "startup_allowed")

    module = SimpleNamespace(startup_permission=permission)
    monkeypatch.setattr(package, "bootstrap", module, raising=False)
    path = tmp_path / "startup.log"
    previous = sys.gettrace()
    stop = observe_startup_refusals(path)
    try:
        assert module.startup_permission(False, None) == (True, "startup_allowed")
        assert not path.exists()
        for _ in range(20):
            assert module.startup_permission(True, None) == (False, "recovery_scope_uncertain")
            assert sys.gettrace() is previous
    finally:
        stop()
    records = json.loads(path.read_text())
    assert len(records) == 8
    assert all(row["reason"] == "recovery_scope_uncertain" for row in records)
    assert records[-1]["errors"][0]["errno"] == 13
    assert records[-1]["return_line"] > 0
    assert "synthetic-private-selector" not in path.read_text()
    assert module.startup_permission is permission


def test_startup_observer_preserves_unexpected_exception(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pytest

    import tldw_chatbook.Backup_Recovery as package
    from Tests.Backup_Recovery.thread_diagnostics import observe_startup_refusals

    error = RuntimeError("synthetic-private-error")

    def permission(*args):
        raise error

    module = SimpleNamespace(startup_permission=permission)
    monkeypatch.setattr(package, "bootstrap", module, raising=False)
    previous = sys.gettrace()
    stop = observe_startup_refusals(tmp_path / "startup.log")
    try:
        with pytest.raises(RuntimeError) as caught:
            module.startup_permission(None, None)
        assert caught.value is error
        assert sys.gettrace() is previous
    finally:
        stop()


def test_startup_observer_bounds_direct_refusal_and_thread_local_tracing(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    import tldw_chatbook.Backup_Recovery as package
    from Tests.Backup_Recovery.thread_diagnostics import observe_startup_refusals

    def permission(*args):
        return False, "synthetic-private-reason"

    module = SimpleNamespace(startup_permission=permission)
    monkeypatch.setattr(package, "bootstrap", module, raising=False)
    path = tmp_path / "startup.log"
    stop = observe_startup_refusals(path)

    def worker():
        previous = sys.gettrace()
        result = module.startup_permission(None, None)
        assert sys.gettrace() is previous
        return result

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(lambda _: worker(), range(4))) == [permission()] * 4
    finally:
        stop()
    records = json.loads(path.read_text())
    assert len(records) == 4
    assert all(row["errors"] == [] and row["return_line"] > 0 for row in records)
    assert all(row["reason"] == "unrecognized_startup_refusal" for row in records)
    assert "synthetic-private-reason" not in path.read_text()
