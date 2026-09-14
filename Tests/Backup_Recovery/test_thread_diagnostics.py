"""Private Python-owned diagnostic frame regression checks."""

import faulthandler
import json
import sys
import threading
import time

import pytest

from Tests.Backup_Recovery.thread_diagnostics import _frames, observe_threads


@pytest.mark.parametrize("via_stack", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
def test_observer_cleanup_preserves_active_failure(via_stack, cancelled):
    import asyncio
    from contextlib import ExitStack

    from Tests.Backup_Recovery.thread_diagnostics import stop_observer

    primary = asyncio.CancelledError() if cancelled else RuntimeError("product failure")
    calls = []

    def stop():
        calls.append(True)
        raise OSError("private diagnostic detail")

    with pytest.raises(type(primary)) as caught:
        try:
            raise primary
        finally:
            original_trace = primary.__traceback__
            if via_stack:
                stack = ExitStack()
                stack.callback(stop_observer, stop)
                stack.close()
            else:
                stop_observer(stop)

    assert caught.value is primary and primary.__traceback__ is original_trace
    assert calls == [True]
    assert primary.__notes__ == ["Optional test diagnostic cleanup failed."]


def test_observer_cleanup_reports_diagnostic_only_failure():
    from Tests.Backup_Recovery.thread_diagnostics import stop_observer

    failure = OSError("diagnostic failure")

    def stop():
        raise failure

    with pytest.raises(OSError) as caught:
        stop_observer(stop)
    assert caught.value is failure


def test_observer_cleanup_note_failure_cannot_replace_primary():
    from Tests.Backup_Recovery.thread_diagnostics import stop_observer

    class PrimaryError(RuntimeError):
        def add_note(self, _note):
            raise KeyboardInterrupt()

    primary = PrimaryError()

    def stop():
        raise OSError()

    with pytest.raises(PrimaryError) as caught:
        try:
            raise primary
        finally:
            stop_observer(stop)
    assert caught.value is primary


def test_observer_cleanup_success_runs_once_without_a_note():
    from Tests.Backup_Recovery.thread_diagnostics import stop_observer

    calls = []
    primary = RuntimeError()
    with pytest.raises(RuntimeError) as caught:
        try:
            raise primary
        finally:
            stop_observer(lambda: calls.append(True))
    assert caught.value is primary and calls == [True]
    assert not getattr(primary, "__notes__", ())


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


def test_capture_observer_names_unavailable_dependency_without_private_ids(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery.thread_diagnostics import observe_capture_review
    from tldw_chatbook.Backup_Recovery import capture, capture_service, inventory
    from tldw_chatbook.Backup_Recovery.models import Inventory, StorageItem

    source = StorageItem(
        "persona.assets",
        "private-source-id",
        None,
        "included",
        ("private-target-id",),
    )
    target = StorageItem("persona.core", "private-target-id", None, "unused", ())
    result = Inventory((source, target), False, "digest", ("dependency_unavailable",))
    for module in (capture_service, capture, inventory):
        monkeypatch.setattr(module, "discover", lambda *a, **k: result)
    path = tmp_path / "capture.log"
    stop = observe_capture_review(path)
    try:
        assert capture_service.discover(()) is result
    finally:
        stop()
    record = json.loads(path.read_text())[-1]
    dependency = record["unavailable_dependencies"][0]
    assert dependency["owner"] == "persona.assets"
    assert dependency["target_owner"] == "persona.core"
    assert dependency["target_status"] == "unused"
    assert len(dependency["logical_id_sha256"]) == 64
    assert len(dependency["dependency_sha256"]) == 64
    assert "private-source-id" not in path.read_text()
    assert "private-target-id" not in path.read_text()


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
    async def ready(owner, deadline):
        return True

    preceding = runtime._Hook(object(), lambda owner: None, ready, None)
    original = runtime._settle_stage
    path = tmp_path / "runtime.log"
    stop = observe_runtime_settlement(path)
    closed = []
    try:
        with pytest.raises(RecoveryRequired, match="runtime_work_not_settled"):
            asyncio.run(runtime._settle_stage([preceding, hook], closed, 42))
    finally:
        stop()
    assert calls == ["close", 42] and closed == [preceding, hook]
    assert runtime._settle_stage is original
    record = json.loads(path.read_text())[-1]
    assert record["event"] == "settle_stage_failure"
    assert record["issue"] == "runtime_work_not_settled"
    assert record["candidate_hooks"][1].endswith("Owner.drain")
    assert record["failed_hook"] == record["candidate_hooks"][1]
    assert 0 <= record["stage_elapsed"] < 5


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
        assert json.loads(path.read_text())[-1]['admission']['available'] is True
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


@pytest.mark.parametrize("phase", ["close", "drain"])
def test_native_stage_observer_identifies_throwing_close_and_cancelled_drain(tmp_path, phase):
    import asyncio

    import pytest

    from Tests.Backup_Recovery.thread_diagnostics import observe_runtime_settlement
    from tldw_chatbook.Backup_Recovery import runtime_maintenance as runtime

    error = RuntimeError("private-close-detail") if phase == "close" else asyncio.CancelledError("private-cancel-detail")
    calls = []

    class Owner:
        def close(self):
            calls.append("close")
            if phase == "close":
                raise error

        async def drain(self, deadline):
            calls.append(deadline)
            raise error

    hook = runtime._Hook(Owner(), Owner.close, Owner.drain, None)
    path = tmp_path / (phase + ".log")
    original = runtime._settle_stage
    stop = observe_runtime_settlement(path)
    closed = []
    try:
        with pytest.raises(type(error)) as caught:
            asyncio.run(runtime._settle_stage([None, hook], closed, 42))
        assert caught.value is error
    finally:
        stop()
    record = json.loads(path.read_text())[-1]
    assert record["failed_hook"].endswith("Owner.drain")
    assert 0 <= record["stage_elapsed"] < 5
    assert closed == [hook] and calls == (["close"] if phase == "close" else ["close", 42])
    assert "private-close-detail" not in path.read_text() and "private-cancel-detail" not in path.read_text()
    assert runtime._settle_stage is original


def test_native_stage_observer_bounds_original_hook_membership(tmp_path):
    import asyncio

    import pytest

    from Tests.Backup_Recovery.thread_diagnostics import observe_runtime_settlement
    from tldw_chatbook.Backup_Recovery import runtime_maintenance as runtime

    error = RuntimeError("private-native-drain")
    metadata_phase = False
    comparisons = []

    class Owner:
        def __init__(self, fails=False):
            self.fails = fails

        def __eq__(self, other):
            if metadata_phase:
                comparisons.append(True)
                raise AssertionError("observer must use identity")
            return self is other

        def close(self):
            pass

        async def drain(self, deadline):
            nonlocal metadata_phase
            if self.fails:
                metadata_phase = True
                raise error
            return True

    hooks = [runtime._Hook(Owner(index == 64), Owner.close, Owner.drain, None) for index in range(65)]
    path = tmp_path / "bounded.log"
    stop = observe_runtime_settlement(path)
    try:
        with pytest.raises(RuntimeError) as caught:
            asyncio.run(runtime._settle_stage(hooks, [], 42))
        assert caught.value is error
    finally:
        stop()
    record = json.loads(path.read_text())[-1]
    assert record["failed_hook"] is None
    assert len(record["candidate_hooks"]) == 64
    assert not comparisons


@pytest.mark.parametrize("metadata_kind", ["error", "cancel"])
def test_native_stage_observer_metadata_failure_preserves_original(tmp_path, metadata_kind):
    import asyncio

    import pytest

    from Tests.Backup_Recovery.thread_diagnostics import observe_runtime_settlement
    from tldw_chatbook.Backup_Recovery import runtime_maintenance as runtime

    error = RuntimeError("original-private-native-failure")
    metadata_error = ValueError("private-observer-detail") if metadata_kind == "error" else asyncio.CancelledError("private-observer-cancel")

    class Drain:
        async def __call__(self, owner, deadline):
            raise error

        def __getattribute__(self, name):
            if name == "__qualname__":
                raise metadata_error
            return object.__getattribute__(self, name)

    hook = runtime._Hook(object(), lambda owner: None, Drain(), None)
    original = runtime._settle_stage
    stop = observe_runtime_settlement(tmp_path / "metadata-failure.log")
    try:
        with pytest.raises(RuntimeError) as caught:
            asyncio.run(runtime._settle_stage([hook], [], 42))
        assert caught.value is error
    finally:
        with pytest.raises(RuntimeError, match="^runtime_diagnostic_write_failed$"):
            stop()
    assert runtime._settle_stage is original


@pytest.mark.parametrize("kind", ("changed", "unavailable", "private_error", "cancelled", "invalid_arg"))
def test_capture_snapshot_observer_preserves_error_and_bounds_private_metadata(
    tmp_path, monkeypatch, kind
):
    import asyncio

    from Tests.Backup_Recovery.thread_diagnostics import observe_capture_review
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    comparisons = []

    class InvalidArgument:
        def __eq__(self, other):
            comparisons.append(True)
            return False

    errors = {
        "changed": ValueError("preview_sqlite_changed"),
        "unavailable": ValueError("preview_sqlite_unavailable"),
        "private_error": OSError(13, "synthetic-private-error", "synthetic-private-path"),
        "cancelled": asyncio.CancelledError("synthetic-private-cancellation"),
        "invalid_arg": ValueError(InvalidArgument()),
    }
    error = errors[kind]
    result = object()

    def original(scope, source):
        if source is result:
            return result
        raise error

    monkeypatch.setattr(storage._PreviewScope, "sqlite_target", original)
    path = tmp_path / "snapshot.log"
    stop = observe_capture_review(path)
    try:
        assert storage._PreviewScope.sqlite_target(None, result) is result
        assert not path.exists()
        for _ in range(10):
            with pytest.raises(type(error)) as caught:
                storage._PreviewScope.sqlite_target(None, "synthetic-private-source")
            assert caught.value is error
            trace = error.__traceback__
            while trace.tb_next is not None:
                trace = trace.tb_next
            assert trace.tb_frame.f_code is original.__code__
    finally:
        stop()
    assert storage._PreviewScope.sqlite_target is original
    assert not comparisons
    rows = json.loads(path.read_text())
    assert len(rows) == 8
    assert all(row["event"] == "preview_sqlite_failure" for row in rows)
    assert rows[-1]["reason"] == (
        "preview_sqlite_" + kind if kind in {"changed", "unavailable"} else None
    )
    assert rows[-1]["error"]["error_class"] == type(error).__name__
    assert rows[-1]["error"]["frames"][-1]["function"] == "original"
    assert "synthetic-private" not in path.read_text()


@pytest.mark.parametrize("failure_point", ("metadata", "source_change", "write"))
@pytest.mark.parametrize("failure_type", (OSError, RuntimeError, KeyboardInterrupt))
def test_capture_snapshot_diagnostic_failure_cannot_replace_original_error(
    tmp_path, monkeypatch, failure_point, failure_type
):
    from Tests.Backup_Recovery import thread_diagnostics as diagnostic
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    error = ValueError("preview_sqlite_changed")

    def original(scope, source):
        raise error

    def fail(*args):
        raise failure_type("synthetic-diagnostic-error")

    monkeypatch.setattr(storage._PreviewScope, "sqlite_target", original)
    monkeypatch.setattr(
        diagnostic,
        {"metadata": "_error_metadata", "source_change": "_snapshot_change", "write": "_write"}[failure_point],
        fail,
    )
    stop = diagnostic.observe_capture_review(tmp_path / "snapshot.log")
    try:
        with pytest.raises(ValueError) as caught:
            storage._PreviewScope.sqlite_target(None, None)
        assert caught.value is error
    finally:
        stop()
    assert storage._PreviewScope.sqlite_target is original


def test_capture_snapshot_observer_retains_native_concurrent_note_refusal(tmp_path):
    from Tests.Backup_Recovery.test_core_dependency_discovery import _SCRIPT
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    setup = """
from Tests.Backup_Recovery.thread_diagnostics import observe_capture_review
snapshot_log=home/'snapshot-failure.json.log'
stop_snapshot=observe_capture_review(snapshot_log)
"""
    script = _SCRIPT.replace("validations=[]", setup + "\nvalidations=[]", 1)
    script = script.replace(
        " storage._PreviewScope._source_state=staticmethod(original)",
        " stop_snapshot()\n storage._PreviewScope._source_state=staticmethod(original)",
        1,
    )
    _run(tmp_path, "native", "dependency-review", script=script, timeout=20)
    rows = json.loads((tmp_path / "home" / "snapshot-failure.json.log").read_text())
    assert len(rows) == 1
    assert rows[0]["event"] == "preview_sqlite_failure"
    assert rows[0]["reason"] == "preview_sqlite_changed"
    assert rows[0]["source_change"] is None  # Final recheck has no retained unequal pair.
    assert rows[0]["error"]["error_class"] == "ValueError"
    summary = json.loads((tmp_path / "home" / "dependency-discovery.json.log").read_text())
    assert summary["core_status"] == "unavailable"
    assert summary["validation_issues"][0] == ["core_validation_unavailable"]


@pytest.mark.parametrize("member", ("main", "wal"))
@pytest.mark.parametrize("phase", ("opened_state", "copy_growth", "copied_state"))
def test_snapshot_observer_identifies_native_changed_member_without_values(
    tmp_path, monkeypatch, member, phase
):
    import os
    import sqlite3
    from contextlib import closing

    from Tests.Backup_Recovery.thread_diagnostics import observe_capture_review
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.native_files import create_private_file

    source = tmp_path / "private-database-name.db"
    with create_private_file(source):
        pass
    suffix = "" if member == "main" else "-wal"
    watched = source.with_name(source.name + suffix)
    changed = False
    original_state = storage._PreviewScope._source_state
    original_read = storage.os.read
    with closing(sqlite3.connect(source)) as writer:
        if member == "wal":
            writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE fixture(value)")
        writer.execute("INSERT INTO fixture VALUES ('original')")
        writer.commit()
        identity = storage.os.stat(watched).st_ino

        def change():
            nonlocal changed
            changed = True
            if phase == "copied_state":
                info = watched.stat()
                # A real metadata change after EOF exercises the post-copy check.
                os.utime(watched, ns=(info.st_atime_ns, info.st_mtime_ns + 1000000000))
            else:
                writer.execute("INSERT INTO fixture VALUES (?)", ("private-row" * 2048,))
                writer.commit()

        def state(path):
            result = original_state(path)
            if phase == "opened_state" and not changed:
                change()
            return result

        def read(fd, size):
            data = original_read(fd, size)
            if (
                phase != "opened_state" and not changed
                and storage.os.fstat(fd).st_ino == identity
                and (phase == "copy_growth" and data or phase == "copied_state" and not data)
            ):
                change()
            return data

        monkeypatch.setattr(storage._PreviewScope, "_source_state", staticmethod(state))
        monkeypatch.setattr(storage.os, "read", read)
        path = tmp_path / "snapshot-metadata.log"
        stop = observe_capture_review(path)
        try:
            with storage._preview_reads():
                scope = storage._local.preview_scope
                with pytest.raises(ValueError, match="^preview_sqlite_changed$"):
                    scope.sqlite_target(source)
                assert not list(scope.directory.iterdir())
            assert not scope.directory
        finally:
            stop()
    assert changed
    rows = json.loads(path.read_text())
    detail = rows[-1]["source_change"]
    assert detail["member"] == member
    assert detail["phase"] == phase
    expected_field = "mtime_ns" if phase == "copied_state" else "size"
    assert expected_field in detail["changed_fields"]
    assert set(detail) == {"member", "phase", "changed_fields"}
    assert set(detail["changed_fields"]) <= {"device", "inode", "size", "mtime_ns", "ctime_ns"}
    assert "private-" not in path.read_text()


@pytest.mark.parametrize(
    "outcome", ("success", "entry_error", "body_error", "suppressed", "exit_error", "cancelled")
)
def test_settlement_admission_context_delegates_exact_protocol(tmp_path, monkeypatch, outcome):
    import asyncio

    from Tests.Backup_Recovery.thread_diagnostics import observe_runtime_settlement
    from tldw_chatbook.Backup_Recovery import runtime_maintenance as runtime
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    value, suppressed = object(), object()
    error = asyncio.CancelledError("private-cancel") if outcome == "cancelled" else ValueError("private-context")
    calls = []

    class Context:
        def __enter__(self):
            calls.append("enter")
            if outcome == "entry_error":
                raise error
            return value

        def __exit__(self, *triple):
            calls.append(triple)
            if outcome == "exit_error":
                raise error
            return suppressed if outcome == "suppressed" else False

    def original(*args, **kwargs):
        return Context()

    monkeypatch.setattr(storage._Acquisition, "initializing", original)
    path = tmp_path / "phase.log"
    stop = observe_runtime_settlement(path)
    try:
        context = storage._Acquisition.initializing(None, "private-root", "private-path")
        if outcome == "entry_error":
            with pytest.raises(ValueError) as caught:
                context.__enter__()
            assert caught.value is error and calls == ["enter"]
        else:
            assert context.__enter__() is value
            asyncio.run(runtime._settle_stage([], [], 1))
            active = json.loads(path.read_text())[-1]["admission"]["threads"][0]["active"]
            assert active[-1]["phase"] == "initializing_body"
            triple = (type(error), error, error.__traceback__) if outcome in {"body_error", "suppressed", "cancelled"} else (None, None, None)
            if outcome == "exit_error":
                with pytest.raises(ValueError) as caught:
                    context.__exit__(*triple)
                assert caught.value is error
            else:
                assert context.__exit__(*triple) is (suppressed if outcome == "suppressed" else False)
            assert calls[-1] == triple
        asyncio.run(runtime._settle_stage([], [], 1))
    finally:
        stop()
    assert storage._Acquisition.initializing is original
    rows = json.loads(path.read_text())
    assert rows[-1]["admission"]["threads"][0]["active"] == []
    assert "private-" not in path.read_text()


def test_settlement_admission_nested_calls_have_no_per_call_writes(tmp_path, monkeypatch):
    import asyncio
    from contextlib import contextmanager

    from Tests.Backup_Recovery.thread_diagnostics import observe_runtime_settlement
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery import runtime_maintenance as runtime
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    result = object()
    path = tmp_path / "phase.log"

    def permission(*args):
        asyncio.run(runtime._settle_stage([], [], 1))
        return result

    def scope(*args, **kwargs):
        return bootstrap.startup_permission(None, None)

    @contextmanager
    def initializing(*args):
        yield result

    monkeypatch.setattr(bootstrap, "startup_permission", permission)
    monkeypatch.setattr(storage, "_scope", scope)
    monkeypatch.setattr(storage._Acquisition, "initializing", initializing)
    originals = (storage._scope, bootstrap.startup_permission, storage.admission_authority)
    stop = observe_runtime_settlement(path)
    try:
        with storage._Acquisition.initializing(None, None, None):
            assert not path.exists()
            assert storage._scope(None) is result
            row = json.loads(path.read_text())[-1]["admission"]["threads"][0]
            assert [item["phase"] for item in row["active"]] == ["initializing_body", "scope", "permission"]
        before = path.read_bytes()
        with storage._Acquisition.initializing(None, None, None):
            pass
        assert path.read_bytes() == before
        asyncio.run(runtime._settle_stage([], [], 1))
    finally:
        stop()
    assert (storage._scope, bootstrap.startup_permission, storage.admission_authority) == originals
    row = json.loads(path.read_text())[-1]["admission"]["threads"][0]
    assert row["active"] == []
    assert row["calls"]["scope"]["completed"] == 1
    assert row["calls"]["initializing_enter"]["completed"] == 2
    assert row["calls"]["permission"]["elapsed_seconds"] >= 0


@pytest.mark.parametrize("failure_point", ("clock", "metadata"))
@pytest.mark.parametrize("failed", (False, True))
def test_settlement_admission_optional_metadata_preserves_outcome(
    tmp_path, monkeypatch, failure_point, failed
):
    import asyncio

    from Tests.Backup_Recovery import thread_diagnostics as diagnostic
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery import runtime_maintenance as runtime

    result = object()
    error = ValueError("private-native-error")

    def permission(*args):
        if failed:
            raise error
        return result

    def metadata_failure(*args):
        raise KeyboardInterrupt("private-observer-failure")

    monkeypatch.setattr(bootstrap, "startup_permission", permission)
    path = tmp_path / "phase.log"
    stop = diagnostic.observe_runtime_settlement(path)
    try:
        with monkeypatch.context() as fault:
            fault.setattr(time if failure_point == "clock" else diagnostic,
                          "monotonic" if failure_point == "clock" else "_error_metadata",
                          metadata_failure)
            if failed:
                with pytest.raises(ValueError) as caught:
                    bootstrap.startup_permission(None, None)
                assert caught.value is error
            else:
                assert bootstrap.startup_permission(None, None) is result
        assert not path.exists()
        asyncio.run(runtime._settle_stage([], [], 1))
    finally:
        stop()
    assert bootstrap.startup_permission is permission
    rows = json.loads(path.read_text())[-1]["admission"]["threads"]
    assert all(not row["active"] for row in rows)
    assert "private-" not in path.read_text()


def test_settlement_admission_bounds_threads_nesting_and_errors(tmp_path, monkeypatch):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from Tests.Backup_Recovery.thread_diagnostics import observe_runtime_settlement
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery import runtime_maintenance as runtime

    error = ValueError("private-error")
    arrived = threading.Barrier(35)

    def permission(depth):
        if depth:
            return bootstrap.startup_permission(depth - 1)
        raise error

    def worker(index):
        arrived.wait(timeout=5)
        try:
            bootstrap.startup_permission(10)
        except ValueError:
            return threading.get_ident()

    monkeypatch.setattr(bootstrap, "startup_permission", permission)
    path = tmp_path / "phase.log"
    stop = observe_runtime_settlement(path)
    try:
        with ThreadPoolExecutor(max_workers=35) as pool:
            identifiers = set(pool.map(worker, range(35)))
        assert len(identifiers) == 35
        assert not path.exists()
        asyncio.run(runtime._settle_stage([], [], 1))
    finally:
        stop()
    observed = json.loads(path.read_text())[-1]["admission"]
    assert len(observed["threads"]) == 32 and observed["truncated"]
    assert len(observed["errors"]) == 8
    assert all(not row["active"] for row in observed["threads"])
    assert "private-error" not in path.read_text()


def test_settlement_admission_native_initializing_wait_is_observable(tmp_path):
    from Tests.Backup_Recovery.test_bound_config_companions import _SCRIPT
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    script = _SCRIPT.split("assert config.get_cli_setting")[0] + r'''
import asyncio,json,threading,time
from Tests.Backup_Recovery.thread_diagnostics import observe_runtime_settlement
from tldw_chatbook.Backup_Recovery import storage_admission as storage,runtime_maintenance as runtime
root=storage.bootstrap.default_bootstrap_root()
path=home/'admission-phase.json.log'
entered,release,waiting=threading.Event(),threading.Event(),threading.Event()
errors=[];done=[]
def worker(leader):
 attempt=storage._Acquisition()
 try:
  if not leader:waiting.set()
  with attempt.initializing(root,selected):
   if leader:
    entered.set();assert release.wait(5)
   assert storage.bootstrap.startup_permission(selected,root)==(True,'startup_allowed')
   authority=storage.admission_authority(root)
   assert authority._identity
   assert storage._scope(root,selected,selected)
  done.append(leader)
 except BaseException as error:errors.append(type(error).__name__)
 finally:attempt.close()
stop=observe_runtime_settlement(path)
leader=threading.Thread(target=worker,args=(True,));follower=threading.Thread(target=worker,args=(False,))
try:
 leader.start();assert entered.wait(5)
 follower.start();assert waiting.wait(5)
 deadline=time.monotonic()+2
 while True:
  asyncio.run(runtime._settle_stage([],[],deadline))
  rows=json.loads(path.read_text())[-1]['admission']['threads']
  phases={row['thread']:[item['phase'] for item in row['active']] for row in rows}
  if phases.get(follower.ident)==['initializing_enter']:break
  assert time.monotonic()<deadline
  time.sleep(.01)
 assert phases[leader.ident]==['initializing_body']
 assert not done
finally:
 release.set();leader.join(5)
 if follower.ident is not None:follower.join(5)
 stop()
assert not leader.is_alive() and not follower.is_alive()
assert not errors and sorted(done)==[False,True],errors
print('retired and reopened')
'''
    _run(tmp_path, "native", "phase-observation", script=script, timeout=20)


@pytest.mark.parametrize("snapshot", [False, True])
def test_native_sqlite_error_metadata_retains_extended_code(tmp_path, snapshot):
    import sqlite3
    from contextlib import closing

    from Tests.Backup_Recovery.thread_diagnostics import _error_metadata

    path = tmp_path / "error.sqlite"
    with closing(sqlite3.connect(path, isolation_level=None, timeout=0)) as first, closing(
        sqlite3.connect(path, isolation_level=None, timeout=0)
    ) as other:
        first.execute("PRAGMA journal_mode=WAL")
        first.execute("CREATE TABLE sample(value INTEGER)")
        if snapshot:
            first.execute("BEGIN")
            first.execute("SELECT * FROM sample").fetchall()
            other.execute("INSERT INTO sample VALUES(1)")
        else:
            other.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError) as caught:
            first.execute("INSERT INTO sample VALUES(2)")
        record = _error_metadata(caught.value)
        assert record["sqlite_errorcode"] == (517 if snapshot else 5)
        assert record["sqlite_errorname"] == (
            "SQLITE_BUSY_SNAPSHOT" if snapshot else "SQLITE_BUSY"
        )


@pytest.mark.parametrize("code,name", [(True, "SQLITE_BUSY"), (-1, "SQLITE_BUSY"),
    (65536, "SQLITE_BUSY"), (5, "SQLITE_PRIVATE_SECRET"), (5, object()),
    (5, "SQLITE_BUSY_SNAPSHOT")])
def test_sqlite_error_metadata_rejects_unproved_names(code, name):
    import sqlite3

    from Tests.Backup_Recovery.thread_diagnostics import _error_metadata

    error = sqlite3.OperationalError("synthetic private detail")
    error.sqlite_errorcode, error.sqlite_errorname = code, name
    record = _error_metadata(error)
    assert record["sqlite_errorcode"] == (5 if type(code) is int and code == 5 else None)
    assert record["sqlite_errorname"] is None
    assert "synthetic private detail" not in json.dumps(record)


@pytest.mark.parametrize("branch", ["outer", "nested", "borrowed", "provided-cursor"])
def test_notes_failed_write_reports_retained_entry_without_database_calls(
    tmp_path, monkeypatch, branch
):
    import sqlite3
    from contextlib import nullcontext

    from Tests.Backup_Recovery import thread_diagnostics as diagnostics
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    database = CharactersRAGDB(":memory:", client_id="write-observer")
    connection = database.get_connection()
    connection.execute("CREATE TRIGGER diagnostic_failure BEFORE INSERT ON notes "
                       "BEGIN SELECT missing_diagnostic_function(); END")
    if branch == "borrowed":
        connection.execute("BEGIN")
    enclosing = database.transaction() if branch in ("nested", "provided-cursor") else nullcontext()
    path = tmp_path / "notes-write.log"
    try:
        with enclosing as cursor:
            with (
                pytest.raises(sqlite3.OperationalError) as caught,
                diagnostics.observe_notes_write_failure(path, database),
            ):
                database.add_note("synthetic-private-title", "synthetic-private-body",
                    cursor=cursor if branch == "provided-cursor" else None)
            error = caught.value
            trace = error.__traceback__
            def no_database_access(*args, **kwargs):
                raise AssertionError("diagnostic attempted a database call")
            with monkeypatch.context() as patch:
                patch.setattr(database, "get_connection", no_database_access)
                entry = diagnostics._notes_transaction_entry(error, database)
            assert error.__traceback__ is trace
            expected = None if branch == "provided-cursor" else {
                "outermost": branch == "outer", "borrowed": branch == "borrowed",
                "immediate": False,
            }
            assert entry == expected
            record = json.loads(path.read_text())
            assert record["transaction_entry"] == expected
            assert record["error"]["sqlite_errorcode"] == 1
            assert len(record["threads"]) <= 32
            assert "synthetic-private" not in path.read_text()
    finally:
        if connection.in_transaction:
            connection.rollback()
        database.close_connection()


@pytest.mark.parametrize("failure", ["metadata", "snapshot", "write"])
@pytest.mark.parametrize("cancelled", [False, True])
def test_notes_write_observer_preserves_original_error(tmp_path, monkeypatch, failure, cancelled):
    import asyncio
    import sqlite3

    from Tests.Backup_Recovery import thread_diagnostics as diagnostics

    primary = sqlite3.OperationalError("original SQLite failure")
    secondary = asyncio.CancelledError() if cancelled else OSError("observer failed")
    def fail(*args, **kwargs):
        raise secondary
    function = {"metadata": "_error_metadata", "snapshot": "_snapshot", "write": "_write"}[failure]
    monkeypatch.setattr(diagnostics, function, fail)
    with (
        pytest.raises(sqlite3.OperationalError) as caught,
        diagnostics.observe_notes_write_failure(tmp_path / "failure.log", None),
    ):
        try:
            raise primary
        except sqlite3.OperationalError:
            original_trace = primary.__traceback__
            raise
    assert caught.value is primary
    assert primary.__traceback__ is original_trace


def test_notes_write_observer_success_has_no_diagnostic_side_effect(tmp_path, monkeypatch):
    import sqlite3
    from contextlib import closing

    from Tests.Backup_Recovery import thread_diagnostics as diagnostics

    def forbidden(*args, **kwargs):
        raise AssertionError("successful write requested failure diagnostics")
    monkeypatch.setattr(diagnostics, "_error_metadata", forbidden)
    monkeypatch.setattr(diagnostics, "_snapshot", forbidden)
    monkeypatch.setattr(diagnostics, "_write", forbidden)
    path = tmp_path / "success.log"
    with closing(sqlite3.connect(":memory:")) as database:
        database.execute("CREATE TABLE sample(value INTEGER)")
        with diagnostics.observe_notes_write_failure(path, None):
            database.execute("INSERT INTO sample VALUES(7)")
        assert database.execute("SELECT value FROM sample").fetchall() == [(7,)]
    assert not path.exists()


def test_notes_write_observer_does_not_classify_failed_transaction_entry(tmp_path, monkeypatch):
    import sqlite3

    from Tests.Backup_Recovery import thread_diagnostics as diagnostics
    from tldw_chatbook.DB import ChaChaNotes_DB as notes

    database = notes.CharactersRAGDB(":memory:", client_id="entry-observer")
    try:
        with pytest.raises(sqlite3.OperationalError) as failure:
            database.get_connection().execute("SELECT missing_diagnostic_function()")
        primary = failure.value
        def fail_entry(connection):
            raise primary
        monkeypatch.setattr(notes, "begin_managed_transaction", fail_entry)
        path = tmp_path / "entry.log"
        with (
            pytest.raises(sqlite3.OperationalError) as caught,
            diagnostics.observe_notes_write_failure(path, database),
        ):
            database.add_note("synthetic-private-title", "synthetic-private-body")
        assert caught.value is primary
        assert json.loads(path.read_text())["transaction_entry"] is None
        assert not database.get_connection().in_transaction
    finally:
        database.close_connection()
