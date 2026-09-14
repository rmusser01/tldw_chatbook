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


def test_capture_observer_reports_metadata_delta_and_original_review(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    import pytest

    from Tests.Backup_Recovery.thread_diagnostics import observe_capture_review
    from tldw_chatbook.Backup_Recovery import capture, capture_service, inventory
    from tldw_chatbook.Backup_Recovery.capture import CaptureReviewRequired
    from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem

    item = StorageItem(
        "db.agent_runs",
        "private-logical-secret",
        tmp_path / "private-path-secret",
        "included",
        (),
    )
    before = Inventory((item,), True, "before", ())
    after = Inventory((replace(item, status="unavailable"),), False, "after", ())
    snapshots = iter((before, after))

    def discover(*args, **kwargs):
        return next(snapshots)

    error = CaptureReviewRequired(("scope_changed",))

    def run(*args, **kwargs):
        capture.discover(())
        raise error

    for module in (capture_service, capture, inventory):
        monkeypatch.setattr(module, "discover", discover)
    monkeypatch.setattr(capture_service, "capture", run)
    path = tmp_path / "capture.log"
    stop = observe_capture_review(path)
    try:
        assert capture_service.discover(()) is before
        with pytest.raises(CaptureReviewRequired) as caught:
            capture_service.capture()
        assert caught.value is error
    finally:
        stop()
    assert all(
        module.discover is discover for module in (capture_service, capture, inventory)
    )
    assert capture_service.capture is run
    records = json.loads(path.read_text())
    changed = next(
        row for row in records if row["event"] == "inventory" and row["scope_changed"]
    )
    assert changed["delta"][0]["owner"] == "db.agent_runs"
    assert changed["delta"][0]["before_status"] == "included"
    assert changed["delta"][0]["after_status"] == "unavailable"
    assert changed["delta"][0]["changed_fields"] == ["status"]
    assert len(changed["delta"][0]["logical_id_sha256"]) == 64
    assert records[-1]["event"] == "capture_review"
    assert records[-1]["error"]["frames"][-1]["function"] == "run"
    assert "private-logical-secret" not in path.read_text()
    assert "private-path-secret" not in path.read_text()


def test_capture_observer_bounds_deltas_and_retained_records(tmp_path, monkeypatch):
    from Tests.Backup_Recovery.thread_diagnostics import observe_capture_review
    from tldw_chatbook.Backup_Recovery import capture, capture_service, inventory
    from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem

    count = 0

    def discover(*args, **kwargs):
        nonlocal count
        count += 1
        return Inventory(
            tuple(
                StorageItem(
                    "config", str(i), None, "included" if count % 2 else "unused", ()
                )
                for i in range(300)
            ),
            True,
            str(count),
            (),
        )

    for module in (capture_service, capture, inventory):
        monkeypatch.setattr(module, "discover", discover)
    path = tmp_path / "capture.log"
    stop = observe_capture_review(path)
    try:
        for _ in range(20):
            assert len(inventory.discover(()).items) == 300
    finally:
        stop()
    records = json.loads(path.read_text())
    assert len(records) == 8
    assert len(records[-1]["delta"]) == 64
    assert records[-1]["delta_truncated"] is True


def test_runtime_observer_preserves_false_drain_and_reports_candidate_hook(tmp_path):
    import asyncio

    import pytest

    from Tests.Backup_Recovery.thread_diagnostics import observe_runtime_settlement
    from tldw_chatbook.Backup_Recovery import runtime_maintenance as runtime
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    calls = []

    class Owner:
        def close(self):
            calls.append("close")

        async def drain(self, deadline):
            calls.append(deadline)
            return False

        def resume(self):
            calls.append("resume")

    owner = Owner()
    hook = runtime._Hook(owner, Owner.close, Owner.drain, Owner.resume)
    original = runtime._settle_stage
    path = tmp_path / "runtime.log"
    stop = observe_runtime_settlement(path)
    closed = []
    try:
        with pytest.raises(RecoveryRequired, match="runtime_work_not_settled"):
            asyncio.run(runtime._settle_stage([hook], closed, 42))
    finally:
        stop()
    assert calls == ["close", 42] and closed == [hook]
    assert runtime._settle_stage is original
    record = json.loads(path.read_text())[-1]
    assert record["event"] == "settle_stage_failure"
    assert record["issue"] == "runtime_work_not_settled"
    assert record["candidate_hooks"][0].endswith("Owner.drain")


def test_runtime_observer_preserves_error_identity_and_hides_text(
    tmp_path, monkeypatch
):
    import asyncio

    import pytest

    from Tests.Backup_Recovery.thread_diagnostics import observe_runtime_settlement
    from tldw_chatbook.Backup_Recovery import runtime_maintenance as runtime

    error = RuntimeError("private-exception-secret")

    async def settle(self, deadline):
        raise error

    monkeypatch.setattr(runtime.RuntimeMaintenance, "settle_producers", settle)
    path = tmp_path / "runtime.log"
    stop = observe_runtime_settlement(path)
    try:
        with pytest.raises(RuntimeError) as caught:
            asyncio.run(runtime.RuntimeMaintenance.settle_producers(object(), 1))
        assert caught.value is error
    finally:
        stop()
    assert runtime.RuntimeMaintenance.settle_producers is settle
    record = json.loads(path.read_text())[-1]
    assert record["event"] == "settle_producers_failure"
    assert record["issue"] == "unrecognized_runtime_issue"
    assert "private-exception-secret" not in path.read_text()


def test_runtime_observer_snapshots_only_native_policy_and_thread_metadata(
    tmp_path, monkeypatch
):
    import asyncio
    from types import SimpleNamespace

    from Tests.Backup_Recovery.thread_diagnostics import observe_runtime_settlement
    from tldw_chatbook.Backup_Recovery import runtime_maintenance as runtime
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.DB.private_sqlite import SQLITE_OWNER_REGISTRY

    lease = storage.StorageLease(None)
    lease._attach_sqlite(
        SQLITE_OWNER_REGISTRY["db.base"], tmp_path / "private-database-secret"
    )
    result = object()

    async def resume(self):
        return result

    monkeypatch.setattr(runtime.RuntimeMaintenance, "resume", resume)
    path = tmp_path / "runtime.log"
    stop = observe_runtime_settlement(path)
    try:
        instance = SimpleNamespace(
            pause=None,
            app=SimpleNamespace(_backup_maintenance_error="runtime_work_not_settled"),
        )
        assert asyncio.run(runtime.RuntimeMaintenance.resume(instance)) is result
    finally:
        stop()
        lease.close()
    assert runtime.RuntimeMaintenance.resume is resume
    record = json.loads(path.read_text())[-1]
    assert record["event"] == "runtime_resume"
    assert any(
        row["owner"] == "db.base"
        and row["thread"] == threading.get_ident()
        and row["count"] >= 1
        for row in record["storage"]["leases"]
    )
    assert "private-database-secret" not in path.read_text()


def test_runtime_observer_preserves_cancellation_and_success(tmp_path, monkeypatch):
    import asyncio

    import pytest

    from Tests.Backup_Recovery.thread_diagnostics import observe_runtime_settlement
    from tldw_chatbook.Backup_Recovery import runtime_maintenance as runtime

    error = asyncio.CancelledError()

    async def settle(self, deadline):
        if deadline == 0:
            raise error
        return "settled"

    monkeypatch.setattr(runtime.RuntimeMaintenance, "settle_producers", settle)
    stop = observe_runtime_settlement(tmp_path / "runtime.log")
    try:
        assert (
            asyncio.run(runtime.RuntimeMaintenance.settle_producers(object(), 1))
            == "settled"
        )
        with pytest.raises(asyncio.CancelledError) as caught:
            asyncio.run(runtime.RuntimeMaintenance.settle_producers(object(), 0))
        assert caught.value is error
    finally:
        stop()
    assert runtime.RuntimeMaintenance.settle_producers is settle


def test_runtime_observer_never_waits_for_storage_owner(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from Tests.Backup_Recovery.thread_diagnostics import observe_runtime_settlement
    from tldw_chatbook.Backup_Recovery import runtime_maintenance as runtime
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    entered, release = threading.Event(), threading.Event()
    def owner():
        with storage._lock:
            entered.set()
            assert release.wait(5)
    async def resume(self): return 'unchanged'
    monkeypatch.setattr(runtime.RuntimeMaintenance, 'resume', resume)
    path = tmp_path / 'busy.log'
    stop = observe_runtime_settlement(path)
    worker = threading.Thread(target=owner)
    worker.start()
    try:
        assert entered.wait(5)
        instance = SimpleNamespace(pause=None, app=SimpleNamespace(_backup_maintenance_error=None))
        assert asyncio.run(runtime.RuntimeMaintenance.resume(instance)) == 'unchanged'
        assert json.loads(path.read_text())[-1]['storage'] == {'available': False}
    finally:
        release.set()
        worker.join(5)
        stop()
    assert not worker.is_alive()


def test_runtime_observer_write_failure_preserves_refusal_and_restores_all_hooks(tmp_path, monkeypatch):
    import asyncio

    import pytest

    from Tests.Backup_Recovery import thread_diagnostics as diagnostics
    from tldw_chatbook.Backup_Recovery import runtime_maintenance as runtime

    error = RuntimeError('private-native-refusal')
    async def settle(self, deadline): raise error
    def write(*args): raise OSError('private-output-path')
    monkeypatch.setattr(runtime.RuntimeMaintenance, 'settle_producers', settle)
    monkeypatch.setattr(diagnostics, '_write', write)
    originals = (runtime._settle_stage, runtime.RuntimeMaintenance.retire_local_caches, runtime.RuntimeMaintenance.resume)
    stop = diagnostics.observe_runtime_settlement(tmp_path / 'error.log')
    try:
        with pytest.raises(RuntimeError) as caught:
            asyncio.run(runtime.RuntimeMaintenance.settle_producers(object(), 1))
        assert caught.value is error
    finally:
        with pytest.raises(RuntimeError, match='^runtime_diagnostic_write_failed$'):
            stop()
    assert runtime.RuntimeMaintenance.settle_producers is settle
    assert (runtime._settle_stage, runtime.RuntimeMaintenance.retire_local_caches, runtime.RuntimeMaintenance.resume) == originals


def test_finalization_observer_preserves_original_failure_and_known_reason(tmp_path, monkeypatch):
    import pytest

    from Tests.Backup_Recovery.thread_diagnostics import observe_finalization_failures
    from tldw_chatbook.Backup_Recovery import publication

    original = publication.finalize_candidate
    for index, error in enumerate((ValueError('installed_content_changed'), ValueError('private-finalization-secret'))):
        def finalize(*args, _error=error, **kwargs): raise _error
        monkeypatch.setattr(publication, 'finalize_candidate', finalize)
        path = tmp_path / f'finalization-{index}.log'
        stop = observe_finalization_failures(path)
        try:
            with pytest.raises(ValueError) as caught:
                publication.finalize_candidate(None, None, None, session=None)
            assert caught.value is error
        finally:
            stop()
        assert publication.finalize_candidate is finalize
        record = json.loads(path.read_text())[-1]
        assert record['issue'] == ('installed_content_changed' if index == 0 else 'unrecognized_finalization_issue')
        assert record['error']['frames'][-1]['function'] == 'finalize'
        assert 'private-finalization-secret' not in path.read_text()
    monkeypatch.setattr(publication, 'finalize_candidate', original)
