"""Bounded test diagnostics using Python-owned frames, without a C watchdog."""

from __future__ import annotations

import faulthandler
import json
import os
import sqlite3
import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path


def stop_observer(stop: Callable[[], None]) -> None:
    """Report cleanup failures without replacing an active product exception."""
    primary = sys.exception()
    try:
        stop()
    except BaseException:
        if primary is None:
            raise
        try:
            primary.add_note("Optional test diagnostic cleanup failed.")
        except BaseException:  # noqa: BLE001, S110 - note metadata cannot mask the primary.  # nosec B110
            pass


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
    if type(error) in (
        sqlite3.Error, sqlite3.DatabaseError, sqlite3.OperationalError,
        sqlite3.IntegrityError, sqlite3.InternalError, sqlite3.ProgrammingError,
        sqlite3.NotSupportedError, sqlite3.DataError, sqlite3.InterfaceError,
    ):
        code = getattr(error, "sqlite_errorcode", None)
        code = code if type(code) is int and 0 < code <= 65535 else None
        name = getattr(error, "sqlite_errorname", None)
        names = (
            "SQLITE_BUSY", "SQLITE_BUSY_SNAPSHOT", "SQLITE_BUSY_RECOVERY",
            "SQLITE_BUSY_TIMEOUT", "SQLITE_LOCKED", "SQLITE_LOCKED_SHAREDCACHE",
            "SQLITE_LOCKED_VTAB",
        )
        record["sqlite_errorcode"] = code
        record["sqlite_errorname"] = (
            name if type(name) is str and name in names and code is not None
            and getattr(sqlite3, name, None) == code else None
        )
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


def _notes_transaction_entry(error: BaseException, database) -> dict | None:
    """Read retained entry flags only after the exact native add body was entered."""
    from tldw_chatbook.DB.ChaChaNotes_DB import (
        CharactersRAGDB,
        TransactionContextManager,
    )

    if type(database) is not CharactersRAGDB:
        return None
    add_code = CharactersRAGDB.add_note.__code__
    body_code = CharactersRAGDB._add_note_with_cursor.__code__
    trace, manager = error.__traceback__, None
    for _ in range(64):
        if trace is None:
            break
        frame = trace.tb_frame
        if frame.f_code is add_code and frame.f_locals.get("self") is database:
            manager = frame.f_locals.get("transaction")
        elif (
            frame.f_code is body_code and frame.f_locals.get("self") is database
            and type(manager) is TransactionContextManager
        ):
            state = vars(manager)
            if state.get("db") is not database:
                return None
            values = tuple(state.get(key) for key in (
                "is_outermost_transaction", "borrows_native_transaction", "immediate",
            ))
            if not all(type(value) is bool for value in values) or all(values[:2]):
                return None
            return dict(zip(("outermost", "borrowed", "immediate"), values))
        trace = trace.tb_next
    return None


@contextmanager
def observe_notes_write_failure(path: Path, database):
    """Keep one bounded failed-write record without altering the original error."""
    try:
        yield
    except sqlite3.Error as error:
        try:
            _write(path, {
                "event": "notes_write_failure", "error": _error_metadata(error),
                "transaction_entry": _notes_transaction_entry(error, database),
                "threads": _snapshot(),
            })
        except BaseException:  # noqa: BLE001, S110 - optional metadata cannot replace the native failure.  # nosec B110
            pass
        raise


def _snapshot_change(error: BaseException, code) -> dict | None:
    """Name proven differences in a failed copy's existing locals, never values."""
    if (
        type(error) is not ValueError or len(error.args) != 1
        or type(error.args[0]) is not str
        or error.args[0] != "preview_sqlite_changed"
    ):
        return None
    trace = error.__traceback__
    for _ in range(64):
        if trace is None:
            return None
        if trace.tb_next is None:
            break
        trace = trace.tb_next
    else:
        return None
    if trace.tb_frame.f_code is not code:
        return None  # A helper failure must not use stale caller locals.
    local = trace.tb_frame.f_locals
    suffix, expected, opened = (
        local.get("suffix"), local.get("expected"), local.get("opened")
    )
    if (
        type(suffix) is not str or suffix not in {"", "-wal"}
        or type(expected) is not tuple or len(expected) != 5
        or any(type(value) is not int for value in expected)
        or type(opened) is not os.stat_result
    ):
        return None
    actual = (
        opened.st_dev, opened.st_ino, opened.st_size,
        opened.st_mtime_ns, opened.st_ctime_ns,
    )
    fields = ("device", "inode", "size", "mtime_ns", "ctime_ns")
    phase = "opened_state"
    changed = [field for field, old, new in zip(fields, expected, actual) if old != new]
    if not changed:
        count, observed = local.get("count"), local.get("observed")
        if type(count) is int and count > expected[2]:
            phase, changed = "copy_growth", ["size"]
        elif (
            type(count) is int
            and type(observed) is tuple and len(observed) == 5
            and all(type(value) is int for value in observed)
        ):
            phase = "copied_state"
            changed = [field for field, old, new in zip(fields, expected, observed) if old != new]
            if count != expected[2] and "size" not in changed:
                changed.append("size")
    if not changed:
        return None  # Reuse/final-source checks have no retained unequal pair.
    return {
        "member": "main" if suffix == "" else "wal",
        "phase": phase, "changed_fields": changed,
    }


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
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    originals = [
        (module, module.discover) for module in (inventory, capture, capture_service)
    ]
    original_capture = capture_service.capture
    original_snapshot = storage._PreviewScope.sqlite_target
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
        observed_items = result.items[:4096]
        by_id = {item.logical_id: item for item in observed_items}
        unavailable = []
        dependency_truncated = len(result.items) > 4096
        for item in observed_items:
            if item.status in {"unused", "intentionally_excluded"}:
                continue
            dependency_truncated |= len(item.dependencies) > 64
            for dependency in item.dependencies[:64]:
                target = by_id.get(dependency)
                if target is None and len(result.items) > 4096:
                    continue  # An unobserved tail is not evidence of absence.
                if (
                    target is None
                    or target.status
                    in inventory.BLOCKING | {"unused", "intentionally_excluded"}
                    or target.status == "intentionally_deleted"
                    and not target.deletion_validated
                ):
                    if len(unavailable) == 64:
                        dependency_truncated = True
                        continue
                    unavailable.append(
                        {
                            "owner": item.owner[:128],
                            "logical_id_sha256": hashlib.sha256(
                                item.logical_id.encode()
                            ).hexdigest(),
                            "dependency_sha256": hashlib.sha256(
                                dependency.encode()
                            ).hexdigest(),
                            "target_owner": None
                            if target is None
                            else target.owner[:128],
                            "target_status": None
                            if target is None
                            else target.status[:64],
                        }
                    )
        current = {}
        for item in observed_items:
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
                "unavailable_dependencies": unavailable,
                "dependencies_truncated": dependency_truncated,
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

    def observed_snapshot(scope, source):
        try:
            return original_snapshot(scope, source)
        except BaseException as error:
            try:
                reason = None
                if (
                    type(error) in (ValueError, OSError)
                    and len(error.args) == 1
                    and type(error.args[0]) is str
                    and error.args[0] in {
                        "preview_sqlite_changed", "preview_sqlite_unavailable",
                        "preview_sqlite_limit", "preview_sqlite_write_failed",
                    }
                ):
                    reason = error.args[0]
                records.append({
                    "event": "preview_sqlite_failure",
                    "error": _error_metadata(error), "reason": reason,
                    "source_change": _snapshot_change(error, original_snapshot.__code__),
                })
                del records[:-8]
                _write(Path(path), records)
            except BaseException:  # noqa: BLE001, S110 - diagnostic failure must not replace the source error.  # nosec B110
                pass
            raise

    for module, original in originals:
        module.discover = wrapper(original)
    capture_service.capture = observed_capture
    storage._PreviewScope.sqlite_target = observed_snapshot

    def stop():
        for module, original in originals:
            module.discover = original
        capture_service.capture = original_capture
        storage._PreviewScope.sqlite_target = original_snapshot
        if failures:
            raise RuntimeError("capture_diagnostic_write_failed")

    return stop


def observe_runtime_settlement(path: Path) -> Callable[[], None]:
    """Observe original settlement calls without tracing, waiting or retiring owners."""
    import hashlib
    import sqlite3
    import time
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
    original_initializing = storage._Acquisition.initializing
    admission_originals = (
        (storage, "_scope", "scope", storage._scope),
        (storage.bootstrap, "startup_permission", "permission", storage.bootstrap.startup_permission),
        (storage, "admission_authority", "authority_open", storage.admission_authority),
    )
    phase_local = threading.local()
    phase_slots, phase_errors = {}, []
    phase_truncated = False
    phase_labels = ("initializing_enter", "initializing_body", "initializing_exit", "scope", "permission", "authority_open")

    def phase_begin(label):
        nonlocal phase_truncated
        try:
            state = getattr(phase_local, "state", None)
            if state is None:
                candidate = {
                    "thread": threading.get_ident(), "stack": [], "truncated": False,
                    "calls": {name: {"started": 0, "completed": 0, "errors": 0, "elapsed_seconds": 0.0, "max_seconds": 0.0} for name in phase_labels},
                }
                # Fixed slot keys bound concurrent registration without a lock.
                for index in range(32):
                    if phase_slots.setdefault(index, candidate) is candidate:
                        state = phase_local.state = candidate
                        break
                if state is None:
                    phase_truncated = True
                    return None
            if len(state["stack"]) == 8:
                state["truncated"] = True
                return None
            frame = (label, time.monotonic())
            state["stack"].append(frame)
            state["calls"][label]["started"] += 1
            return state, frame
        except BaseException:  # noqa: BLE001 - optional metadata cannot prevent admission.
            return None

    def phase_end(token, error=None):
        if token is None:
            return
        try:
            state, frame = token
            if state["stack"] and state["stack"][-1] is frame:
                state["stack"].pop()
            row = state["calls"][frame[0]]
            row["completed"] += 1
            row["errors"] += error is not None
            elapsed = max(0.0, time.monotonic() - frame[1])
            row["elapsed_seconds"] += elapsed
            row["max_seconds"] = max(row["max_seconds"], elapsed)
            if error is not None:
                phase_errors.append({"thread": state["thread"], "phase": frame[0], "error": _error_metadata(error)})
                del phase_errors[:-8]
        except BaseException:  # noqa: BLE001, S110 - retain the original result or error.  # nosec B110
            pass

    def phase_call(label, original, *args, **kwargs):
        token = phase_begin(label)
        try:
            result = original(*args, **kwargs)
        except BaseException as error:
            phase_end(token, error)
            raise
        phase_end(token)
        return result

    class InitializingObservation:
        def __init__(self, context):
            self.context, self.body = context, None

        def __enter__(self):
            result = phase_call("initializing_enter", type(self.context).__enter__, self.context)
            self.body = phase_begin("initializing_body")
            return result

        def __exit__(self, *triple):
            phase_end(self.body)
            return phase_call("initializing_exit", type(self.context).__exit__, self.context, *triple)

    def observed_initializing(*args, **kwargs):
        return InitializingObservation(original_initializing(*args, **kwargs))

    def admission_wrapper(label, original):
        def observed(*args, **kwargs):
            return phase_call(label, original, *args, **kwargs)
        return observed

    def phase_snapshot():
        try:
            now = time.monotonic()
            states = tuple(phase_slots.copy().values())
            return {
                "available": True,
                "truncated": phase_truncated or any(state["truncated"] for state in states),
                "threads": [
                    {"thread": state["thread"],
                     "active": [{"phase": label, "elapsed_seconds": max(0.0, now - started)} for label, started in tuple(state["stack"])],
                     "calls": {name: row.copy() for name, row in state["calls"].items()}}
                    for state in states
                ],
                "errors": phase_errors[-8:],
            }
        except BaseException:  # noqa: BLE001 - a concurrent/failed sample is not authority.
            return {"available": False}

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
            notes_handles = []
            current = threading.current_thread()
            for lease in islice(storage._live_leases, 256):
                if lease_owner(lease) != "db.chachanotes.primary":
                    continue
                participant = getattr(lease, "resource_participant", None)
                registered = participant in participants._installed_repositories
                repository = participant.repository() if registered else None
                row = {
                    "current_thread": lease.resource_thread is current,
                    "registered": registered,
                    "repository_alive": repository is not None,
                    "close_failed": bool(lease.resource_close_failed),
                }
                if repository is not None:
                    row.update(
                        exact_type=participants._repository_types().get(type(repository)) == "db.chachanotes.primary",
                        path_matches=repository.db_path == participant.path,
                        participant_matches=repository._maintenance_participant is participant,
                        memory=bool(repository.is_memory_db),
                        retiring=current in participant.retiring_threads,
                    )
                    for connection, owned_lease in islice(participant.connections.items(), 64):
                        if owned_lease is not lease:
                            continue
                        row["current_cache"] = getattr(repository._local, "conn", None) is connection
                        if lease.resource_thread is current:
                            try:
                                row["in_transaction"] = bool(connection.in_transaction)
                            except sqlite3.Error as error:
                                row["transaction_error_class"] = type(error).__name__[:80]
                notes_handles.append(row)
                if len(notes_handles) == 16:
                    break
            return {
                "available": True,
                "notes_handles": notes_handles,
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
        row = {"event": event, "storage": snapshot(), "admission": phase_snapshot(), **values}
        if error is not None:
            row.update(error=_error_metadata(error), issue=issue(error))
        records.append(row)
        del records[:-8]
        try:
            _write(Path(path), records)
        except OSError as write_error:
            failures.append(type(write_error).__name__)

    async def observed_stage(hooks, closed, deadline):
        candidates, started = (), None
        try:
            candidates = hooks[:64] if type(hooks) in (list, tuple) else ()
            started = time.monotonic()
        except Exception as metadata_error:  # noqa: BLE001 - optional diagnostic setup.
            failures.append(type(metadata_error).__name__[:80])
        try:
            record("settle_stage_begin")
        except BaseException:  # noqa: BLE001, S110 - observation cannot prevent original settlement.  # nosec B110
            pass
        try:
            return await original_stage(hooks, closed, deadline)
        except BaseException as error:
            try:
                failed_hook = None
                trace = error.__traceback__
                for _ in range(64):
                    if trace is None:
                        break
                    if trace.tb_frame.f_code is original_stage.__code__:
                        hook = trace.tb_frame.f_locals.get("hook")
                        if hook is not None and any(hook is candidate for candidate in candidates):
                            # Label the hook, not which close/drain callback failed.
                            failed_hook = hook.drain.__qualname__[:160]
                        break
                    trace = trace.tb_next
                record(
                    "settle_stage_failure",
                    error,
                    candidate_hooks=[
                        hook.drain.__qualname__[:160]
                        for hook in candidates
                        if hook is not None
                    ],
                    failed_hook=failed_hook,
                    stage_elapsed=None if started is None else max(0.0, time.monotonic() - started),
                )
            except BaseException as metadata_error:  # noqa: BLE001 - retain the original stage error.
                failures.append(type(metadata_error).__name__[:80])
            raise

    async def observed_settle(self, deadline):
        try:
            return await original_settle(self, deadline)
        except BaseException as error:
            record("settle_producers_failure", error)
            raise

    def observed_retire(self):
        record("retire_caches_before")
        try:
            result = original_retire(self)
        except BaseException as error:
            record("retire_caches_failure", error)
            raise
        record("retire_caches_after")
        return result

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
    storage._Acquisition.initializing = observed_initializing
    for module, name, label, original in admission_originals:
        setattr(module, name, admission_wrapper(label, original))

    def stop():
        runtime._settle_stage = original_stage
        runtime.RuntimeMaintenance.settle_producers = original_settle
        runtime.RuntimeMaintenance.retire_local_caches = original_retire
        runtime.RuntimeMaintenance.resume = original_resume
        storage._Acquisition.initializing = original_initializing
        for module, name, _label, original in admission_originals:
            setattr(module, name, original)
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
