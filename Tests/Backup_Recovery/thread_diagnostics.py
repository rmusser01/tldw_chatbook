"""Bounded test diagnostics using Python-owned frames, without a C watchdog."""

from __future__ import annotations

import faulthandler
import json
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path


def _frames(frame, limit: int = 64) -> list[dict]:
    """Copy code metadata while Python retains every traversed frame."""
    result = []
    while frame is not None and len(result) < limit:
        result.append(
            {
                "file": Path(frame.f_code.co_filename).name[:128],
                "function": frame.f_code.co_name[:128],
                "line": frame.f_lineno,
            }
        )
        frame = frame.f_back
    return result


def _write(path: Path, records: list | dict) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(records, output)


def _snapshot() -> list[dict]:
    current = sys._current_frames()
    return [
        {"thread": identifier, "frames": _frames(frame)}
        for identifier, frame in sorted(current.items())[:32]
    ]


def snapshot_threads(path: Path) -> None:
    """Write an immediate bounded snapshot without reading source or locals."""
    _write(Path(path), [_snapshot()])


def observe_threads(path: Path, *, interval: float = 60) -> Callable[[], None]:
    """Sample bounded metadata on a Python thread; retain native fatal reporting.

    The C dump_traceback_later watchdog can race active interpreter frames
    (CPython gh-140815). sys._current_frames returns owned Python frame objects.
    This observer never records source lines, local variables or exception text.
    """
    if interval <= 0:
        raise ValueError("positive_diagnostic_interval_required")
    path = Path(path)
    fatal = path.with_name(path.stem + "-fatal.log")
    descriptor = os.open(fatal, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    fatal_output = os.fdopen(descriptor, "w", encoding="utf-8")
    was_enabled = faulthandler.is_enabled()
    faulthandler.enable(file=fatal_output)
    stopping = threading.Event()
    records, failures = [], []

    def sample():
        try:
            while not stopping.wait(interval):
                records.append(_snapshot())
                del records[:-4]
                _write(path, records)
        except Exception as error:  # noqa: BLE001 - stop() fails the test for observer errors.
            failures.append(type(error).__name__)

    worker = threading.Thread(
        target=sample, name="test-thread-diagnostics", daemon=True
    )
    worker.start()

    def stop():
        stopping.set()
        worker.join(timeout=5)
        if worker.is_alive():
            raise RuntimeError("thread_diagnostic_stop_failed")
        if was_enabled:
            faulthandler.enable()
        else:
            faulthandler.disable()
        fatal_output.close()
        if failures:
            raise RuntimeError("thread_diagnostic_failed:" + failures[0])

    return stop


def _error_metadata(error: BaseException) -> dict:
    record = {"error_class": type(error).__name__[:80], "frames": []}
    for key in ("errno", "winerror"):
        value = getattr(error, key, None)
        record[key] = value if type(value) is int else None
    trace = error.__traceback__
    while trace is not None and len(record["frames"]) < 64:
        record["frames"].append(
            {
                "file": Path(trace.tb_frame.f_code.co_filename).name[:128],
                "function": trace.tb_frame.f_code.co_name[:128],
                "line": trace.tb_lineno,
            }
        )
        trace = trace.tb_next
    return record


def observe_inventory_failures(path: Path) -> Callable[[], None]:
    """Observe exceptions caught in discovery on only its calling worker thread."""
    from tldw_chatbook.Backup_Recovery import inventory, recovery_service

    service = recovery_service.RecoveryService
    original = service.preview_backup
    records, lock = [], threading.Lock()

    def observed(self, *args, **kwargs):
        errors = []

        def trace(frame, event, argument):
            if frame.f_code is not inventory.discover.__code__:
                return None
            if event == "exception":
                errors.append(_error_metadata(argument[1]))
                del errors[:-16]
            return trace

        previous = sys.gettrace()
        sys.settrace(trace)
        try:
            result = original(self, *args, **kwargs)
        finally:
            sys.settrace(previous)
        blocking = [
            {
                "owner": row.owner[:128],
                "status": row.status,
                "has_path": row.path is not None,
            }
            for row in result.items
            if row.status in {"unsupported", "unavailable", "missing_required"}
        ][:128]
        with lock:
            records.append({"errors": errors, "blocking": blocking})
            del records[:-8]
            _write(Path(path), records)
        return result

    service.preview_backup = observed

    def stop():
        service.preview_backup = original

    return stop


def observe_recovery_failures(path: Path) -> Callable[[], None]:
    """Record only bounded original error metadata, then use the real mapper."""
    from tldw_chatbook.Backup_Recovery import recovery_service

    original = recovery_service.issue_code
    service_class = recovery_service.RecoveryService
    original_static = service_class.__dict__["issue_code"]
    records, lock = [], threading.Lock()

    def observed(error, *, kind=""):
        record = _error_metadata(error)
        with lock:
            records.append(record)
            del records[:-16]
            _write(Path(path), records)
        return original(error, kind=kind)

    recovery_service.issue_code = observed
    service_class.issue_code = staticmethod(observed)

    def stop():
        recovery_service.issue_code = original
        service_class.issue_code = original_static

    return stop


def observe_startup_refusals(path: Path) -> Callable[[], None]:
    """Trace caught startup errors only during its calling thread's check."""
    from tldw_chatbook.Backup_Recovery import bootstrap

    original = bootstrap.startup_permission
    records, lock = [], threading.Lock()

    def observed(*args, **kwargs):
        errors, return_line = [], None

        def trace(frame, event, argument):
            nonlocal return_line
            if frame.f_code is not original.__code__:
                return None
            if event == "exception":
                errors.append(_error_metadata(argument[1]))
                del errors[:-16]
            elif event == "return":
                return_line = frame.f_lineno
            return trace

        previous = sys.gettrace()
        sys.settrace(trace)
        try:
            result = original(*args, **kwargs)
        finally:
            sys.settrace(previous)
        if result[0] is False:
            reason = result[1]
            if reason not in {"recovery_scope_uncertain", "recovery_pending"}:
                reason = "unrecognized_startup_refusal"
            record = {
                "reason": reason,
                "return_line": return_line,
                "errors": errors,
                "callers": _frames(sys._getframe().f_back, limit=16),
            }
            with lock:
                records.append(record)
                del records[:-8]
                _write(Path(path), records)
        return result

    bootstrap.startup_permission = observed

    def stop():
        bootstrap.startup_permission = original

    return stop


def observe_capture_review(path: Path) -> Callable[[], None]:
    """Observe capture checkpoints and bounded metadata changes, never source bytes."""
    import hashlib

    from tldw_chatbook.Backup_Recovery import capture, capture_service, inventory

    originals = [
        (module, module.discover) for module in (inventory, capture, capture_service)
    ]
    original_capture = capture_service.capture
    records, failures = [], []
    previous = None

    def record(row):
        records.append(row)
        del records[:-8]
        try:
            _write(Path(path), records)
        except OSError as write_error:
            failures.append(type(write_error).__name__)

    def observed_discover(original, *args, **kwargs):
        nonlocal previous
        result = original(*args, **kwargs)
        current = {}
        for item in result.items[:4096]:
            # Paths, dependency IDs and metadata remain comparison-only. The
            # output identifies logical IDs by hash because IDs may contain paths.
            identifier = hashlib.sha256(item.logical_id.encode()).hexdigest()
            current[identifier] = {
                "owner": item.owner[:128],
                "status": item.status[:64],
                "path": item.path,
                "dependencies": item.dependencies,
                "shared_group": item.shared_group,
                "deletion_validated": item.deletion_validated,
                "metadata": None if item.metadata is None else (
                    item.metadata.version, item.metadata.root_id,
                    item.metadata.relative_path, item.metadata.parent_id,
                    item.metadata.kind, item.metadata.policy,
                ),
            }
        delta = []
        if previous is not None:
            for identifier in sorted(previous[1].keys() | current.keys()):
                old, new = previous[1].get(identifier), current.get(identifier)
                if old != new:
                    delta.append(
                        {
                            "owner": (new or old)["owner"],
                            "logical_id_sha256": identifier,
                            "before_status": None if old is None else old["status"],
                            "after_status": None if new is None else new["status"],
                            "changed_fields": ["presence"]
                            if old is None or new is None
                            else sorted(key for key in old if old[key] != new[key]),
                        }
                    )
        record(
            {
                "event": "inventory",
                "callers": _frames(sys._getframe().f_back, limit=4),
                "scope_changed": previous is not None
                and previous[0] != result.scope_digest,
                "delta": delta[:64],
                "delta_truncated": len(delta) > 64,
                "inventory_truncated": len(result.items) > 4096,
            }
        )
        previous = result.scope_digest, current
        return result

    def wrapper(original):
        def observed(*args, **kwargs):
            return observed_discover(original, *args, **kwargs)

        return observed

    def observed_capture(*args, **kwargs):
        try:
            return original_capture(*args, **kwargs)
        except capture.CaptureReviewRequired as error:
            record({"event": "capture_review", "error": _error_metadata(error)})
            raise

    for module, original in originals:
        module.discover = wrapper(original)
    capture_service.capture = observed_capture

    def stop():
        for module, original in originals:
            module.discover = original
        capture_service.capture = original_capture
        if failures:
            raise RuntimeError("capture_diagnostic_write_failed")

    return stop


def observe_runtime_settlement(path: Path) -> Callable[[], None]:
    """Observe original settlement calls without tracing, waiting or retiring owners."""
    import hashlib
    from collections import Counter
    from itertools import islice

    from tldw_chatbook.Backup_Recovery import participants
    from tldw_chatbook.Backup_Recovery import runtime_maintenance as runtime
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.DB.private_sqlite import SQLITE_OWNER_REGISTRY

    original_stage = runtime._settle_stage
    original_settle = runtime.RuntimeMaintenance.settle_producers
    original_retire = runtime.RuntimeMaintenance.retire_local_caches
    original_resume = runtime.RuntimeMaintenance.resume
    policies = {id(policy): owner for owner, policy in SQLITE_OWNER_REGISTRY.items()}
    records, failures = [], []
    issues = {
        "runtime_work_not_settled",
        "runtime_native_resources_not_settled",
        "needs_user_save_discard",
        "runtime_owner_unqualified",
        "runtime_owner_maintenance_unavailable",
        "runtime_screens_changed",
        "runtime_owners_changed",
        "participant_runtime_coverage_incomplete",
    }

    def lease_owner(lease):
        participant = getattr(lease, "resource_participant", None)
        if participant in participants._installed_repositories:
            return participant.owner_id[:128]
        return policies.get(id(lease.resource_policy), "startup_or_non_sqlite")

    def snapshot():
        # A diagnostic must never wait behind an owner that it is observing.
        if not storage._lock.acquire(blocking=False):
            return {"available": False}
        try:
            counts = Counter(
                (lease_owner(lease), getattr(lease.resource_thread, "ident", None))
                for lease in islice(storage._live_leases, 256)
            )
            names = {
                hashlib.sha256(name.encode()).hexdigest() + ".lease"
                for hold in islice(storage._holds.values(), 64)
                for name in hold.names[:64]
            }
            return {
                "available": True,
                "pending": len(storage._pending_acquisitions),
                "operations": len(storage._operations),
                "raw_operations": len(storage._raw_operations),
                "live_leases": len(storage._live_leases),
                "startups": len(storage._startups),
                "retiring_holds": len(storage._retiring_holds),
                "leases": [
                    {"owner": owner, "thread": thread, "count": count}
                    for (owner, thread), count in counts.items()
                ],
                "lease_counts_truncated": len(storage._live_leases) > 256,
                "native_lease_keys": sorted(names)[:64],
                "native_keys_truncated": len(names) > 64 or len(storage._holds) > 64,
            }
        finally:
            storage._lock.release()

    def issue(error):
        if (
            type(error) is runtime.RecoveryRequired
            and len(error.args) == 1
            and error.args[0] in issues
        ):
            return error.args[0]
        return "unrecognized_runtime_issue"

    def record(event, error=None, **values):
        row = {"event": event, "storage": snapshot(), **values}
        if error is not None:
            row.update(error=_error_metadata(error), issue=issue(error))
        records.append(row)
        del records[:-8]
        try:
            _write(Path(path), records)
        except OSError as write_error:
            failures.append(type(write_error).__name__)

    async def observed_stage(hooks, closed, deadline):
        try:
            return await original_stage(hooks, closed, deadline)
        except BaseException as error:
            # False-returning drains expose only their candidate stage here.
            # Throwing drains also identify the exact callback in the traceback.
            record(
                "settle_stage_failure",
                error,
                candidate_hooks=[
                    hook.drain.__qualname__[:160]
                    for hook in hooks[:64]
                    if hook is not None
                ],
            )
            raise

    async def observed_settle(self, deadline):
        try:
            return await original_settle(self, deadline)
        except BaseException as error:
            record("settle_producers_failure", error)
            raise

    def observed_retire(self):
        try:
            return original_retire(self)
        except BaseException as error:
            record("retire_caches_failure", error)
            raise

    async def observed_resume(self):
        value = self.app._backup_maintenance_error
        record(
            "runtime_resume",
            startup_retired=bool(getattr(self.pause, "_startup_retired", False)),
            issue=value if value in issues else "unrecognized_runtime_issue",
        )
        return await original_resume(self)

    runtime._settle_stage = observed_stage
    runtime.RuntimeMaintenance.settle_producers = observed_settle
    runtime.RuntimeMaintenance.retire_local_caches = observed_retire
    runtime.RuntimeMaintenance.resume = observed_resume

    def stop():
        runtime._settle_stage = original_stage
        runtime.RuntimeMaintenance.settle_producers = original_settle
        runtime.RuntimeMaintenance.retire_local_caches = original_retire
        runtime.RuntimeMaintenance.resume = original_resume
        if failures:
            raise RuntimeError("runtime_diagnostic_write_failed")

    return stop


def observe_finalization_failures(path: Path) -> Callable[[], None]:
    """Retain the original failure before replacement's automatic reversal catches it."""
    from tldw_chatbook.Backup_Recovery import publication

    original = publication.finalize_candidate
    records, failures = [], []
    known = {
        "publication_context_unverified", "publication_incomplete", "safety_source_changed",
        "candidate_receipt_changed", "verified_manifest_changed", "installed_identity_required",
        "installed_sqlite_sidecar_present", "installed_objects_changed", "installed_identity_changed",
        "installed_content_changed", "installed_directory_rollback_required", "installed_metadata_changed",
        "installed_evidence_changed", "directory_metadata_changed", "publication_parent_changed",
        "unsupported_schema", "schema_mismatch", "integrity_check_failed", "foreign_key_check_failed",
    }

    def observed(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except (OSError, ValueError, RuntimeError) as error:
            issue = error.args[0] if type(error) is ValueError and len(error.args) == 1 and type(error.args[0]) is str and error.args[0] in known else "unrecognized_finalization_issue"
            records.append({"error": _error_metadata(error), "issue": issue})
            del records[:-4]
            try:
                _write(Path(path), records)
            except OSError:
                failures.append(True)
            raise

    publication.finalize_candidate = observed

    def stop():
        publication.finalize_candidate = original
        if failures:
            raise RuntimeError("finalization_diagnostic_write_failed")

    return stop


def observe_large_restore(path: Path) -> Callable[[], None]:
    """Aggregate six original call boundaries in the synchronous large restore test.

    Persist at most once per five seconds, plus first calls and fixed phase edges,
    so a killed child retains progress without per-file records or argument values.
    Nested call durations overlap and must not be added together.
    """
    from functools import wraps
    from time import monotonic

    from tldw_chatbook.Backup_Recovery import publication, recovery_files, staging

    targets = (
        ("require_capacity", staging, "require_capacity"),
        ("validation", recovery_files._RawDeclaration, "validate"),
        ("copy", staging, "_copy"),
        ("stage", staging, "stage_restore"),
        ("publication", publication, "publish_candidate"),
        ("finalization", publication, "finalize_candidate"),
    )
    calls = {
        label: {"started": 0, "completed": 0, "active": 0, "failed": 0, "elapsed_seconds": 0.0}
        for label, _, _ in targets
    }
    phases, originals, failures = [], [], []
    began = last_write = monotonic()

    def persist(force=False):
        nonlocal last_write
        now = monotonic()
        if force or now - last_write >= 5:
            try:
                _write(Path(path), {"elapsed_seconds": now - began, "calls": calls, "phases": phases})
            except OSError:
                failures[:] = [True]
            last_write = now

    def wrap(label, original):
        phase = label in {"stage", "publication", "finalization"}

        @wraps(original)
        def observed(*args, **kwargs):
            started = monotonic()
            row = calls[label]
            row["started"] += 1
            row["active"] += 1
            if phase:
                phases.append({"phase": label + "_started", "elapsed_seconds": started - began})
                del phases[:-12]
            persist(force=phase or row["started"] == 1)
            call_started = monotonic()
            try:
                return original(*args, **kwargs)
            except BaseException:
                row["failed"] += 1
                raise
            finally:
                ended = monotonic()
                row["completed"] += 1
                row["active"] -= 1
                row["elapsed_seconds"] += ended - call_started
                if phase:
                    phases.append({"phase": label + "_finished", "elapsed_seconds": ended - began})
                    del phases[:-12]
                persist(force=phase)

        return observed

    for label, owner, name in targets:
        original = getattr(owner, name)
        originals.append((owner, name, original))
        setattr(owner, name, wrap(label, original))
    persist(force=True)

    def stop():
        for owner, name, original in originals:
            setattr(owner, name, original)
        persist(force=True)
        if failures:
            raise RuntimeError("large_restore_diagnostic_write_failed")

    return stop
