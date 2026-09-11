"""Process-local lifetime counting over one predeclared cross-process lease.

Lease ownership lives in a dedicated thread, so connections may close on a worker
thread. A process never widens a live scope: config changes require retirement and
re-enrollment. Native-unqualified ordinary use is distinct from recovery admission.
"""

from __future__ import annotations

import atexit
import asyncio
from contextlib import contextmanager
import sqlite3
import os
from pathlib import Path
import stat
import threading
import time

from . import bootstrap
from .admission import Admission, AdmissionCancelled, _local
from .control_records import UNBOUND_NAMESPACE, admission_authority
from .profile_paths import effective_config_path, lexical_path
from .qualification import qualified_for

_lock = threading.RLock()
_holds: dict[tuple[int, str], "_Hold"] = {}
# Last-token closes wait outside _lock. Keep their native lifetime observable
# until positive retirement; future maintenance drain must include this set.
_retiring_holds: set["_Hold"] = set()
_startups: dict[tuple[int, str], "StorageLease"] = {}
_forked_with_owners = False


# This local gate intentionally covers all roots: an acquisition may still be
# resolving its root. Native scopes remain individually bound to their holders.
_pending_acquisitions: set["_Acquisition"] = set()
_live_leases: set["StorageLease"] = set()
_operations: set["_Operation"] = set()
_raw_operations: set[object] = set()
_pause: "_LocalPause | None" = None
_changed = threading.Condition(_lock)
_operation_local = threading.local()


def _task_identity():
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


class _Operation:
    """Live installed producer lifetime with an exact bounded descendant scope."""

    def __init__(self):
        raise TypeError("operation_is_installed_owner_issued")

    def check(self, path=None):
        from .participants import _installed_repositories

        if (
            self not in _operations
            or self.participant not in _installed_repositories
            or self.participant.repository() is None
            or self.participant.read_only
            != getattr(self.participant.repository(), "_read_only", False)
            or self.participant.repository()._maintenance_participant
            is not self.participant
            or self.pid != os.getpid()
            or self.thread is not threading.current_thread()
            or self.task is not _task_identity()
            or self.lease is None
            or self.lease not in _live_leases
        ):
            raise bootstrap.RecoveryRequired("operation_provenance_invalid")
        if path is not None:
            selected = lexical_path(path)
            parent = self.path.parent.stat()
            if (
                selected != self.path
                or selected.resolve() != self.resolved_path
                or (parent.st_dev, parent.st_ino) != self.parent_identity
                or self.participant.repository().db_path != self.path
            ):
                raise bootstrap.RecoveryRequired("operation_path_outside_scope")
        if _pause is not None:
            hold = _holds.get(self.lease._key)
            if (
                hold is None
                or hold is not self.hold
                or hold.key != self.key
                or not hold.ready.is_set()
                or hold.stop.is_set()
                or hold.error is not None
            ):
                raise bootstrap.RecoveryRequired("operation_native_scope_unqualified")


def _check_operation(operation, path=None):
    # Thread-local discovery is not authority. Never dispatch a caller-supplied
    # validation callback or an instance-shadowed method.
    if type(operation) is not _Operation:
        raise bootstrap.RecoveryRequired("operation_provenance_invalid")
    _Operation.check(operation, path)


@contextmanager
def _repository_operation(participant):
    from .participants import _installed_repositories, _check_core_retirement

    previous = getattr(_operation_local, "operation", None)
    with _changed:
        if previous is not None:
            _check_operation(previous, previous.path)
        reuse = previous is not None and previous.participant is participant
        if reuse:
            _check_operation(previous, participant.path)
        else:
            if (
                participant not in _installed_repositories
                or participant.repository() is None
            ):
                raise bootstrap.RecoveryRequired("repository_participant_not_installed")
            _check_core_retirement(participant)
            if participant.closed or _pause is not None:
                raise bootstrap.RecoveryRequired("storage_locally_paused")
            operation = object.__new__(_Operation)
            operation.participant = participant
            operation.pid = os.getpid()
            operation.thread = threading.current_thread()
            operation.task = _task_identity()
            operation.lease = None
            _operations.add(operation)
    if reuse:
        yield previous
        return
    try:
        # A different installed participant gets independent *ordinary* admission
        # while the gate is open. The outer scope stays counted but confers no
        # descendant authority on this new acquisition (ruling55).
        _operation_local.operation = None
        operation.path = participant.path
        operation.resolved_path = participant.path.resolve()
        try:
            parent = participant.path.parent.stat()
        except FileNotFoundError:
            # Preserve the ordinary SQLite private-parent refusal contract.
            from tldw_chatbook.Utils.private_paths import (
                PrivatePathError,
                PrivatePathResult,
                PrivatePathStatus,
            )

            raise PrivatePathError(
                PrivatePathResult(
                    participant.path,
                    PrivatePathStatus.UNSAFE_PARENT,
                    reason="missing_parent",
                )
            ) from None
        operation.parent_identity = (parent.st_dev, parent.st_ino)
        operation.lease = acquire_storage(operation.path)
        with _changed:
            if _pause is not None or participant.closed:
                raise bootstrap.RecoveryRequired("storage_locally_paused")
            _check_core_retirement(participant)
            operation.key = operation.lease._key
            operation.hold = _holds.get(operation.key)
            _operation_local.operation = operation
        yield operation
    finally:
        _operation_local.operation = None
        try:
            # Its descendants retain independent tokens until positive close.
            if operation.lease is not None:
                operation.lease.close()
            with _changed:
                _operations.discard(operation)
                _changed.notify_all()
        finally:
            with _changed:
                if previous is not None:
                    _check_operation(previous, previous.path)
                _operation_local.operation = previous


class _Acquisition:
    def __init__(self):
        self.pid = os.getpid()
        self.thread = threading.current_thread()
        self.task = _task_identity()
        self.initializing_root = None
        self.cancel = threading.Event()
        self.operation = getattr(_operation_local, "operation", None)
        with _lock:
            if self.operation is not None:
                _check_operation(self.operation)
            if _pause is not None and self.operation is None:
                raise bootstrap.RecoveryRequired("storage_locally_paused")
            _pending_acquisitions.add(self)

    def check(self, path=None):
        if self.cancel.is_set():
            raise bootstrap.RecoveryRequired("storage_locally_paused")
        if self.operation is not None:
            if path is None:
                raise bootstrap.RecoveryRequired("operation_path_outside_scope")
            _check_operation(self.operation, path)
        elif _pause is not None:
            raise bootstrap.RecoveryRequired("storage_locally_paused")

    @contextmanager
    def initializing(self, root, path):
        # The marker and native registration are a single same-process first-use
        # interval. Durable incomplete state from any other interval still refuses.
        with _changed:
            while True:
                self.check(path)
                if (
                    self not in _pending_acquisitions
                    or self.pid != os.getpid()
                    or self.thread is not threading.current_thread()
                    or self.task is not _task_identity()
                ):
                    raise bootstrap.RecoveryRequired("acquisition_provenance_invalid")
                leader = next(
                    (
                        other
                        for other in _pending_acquisitions
                        if other.initializing_root == root and other.pid == self.pid
                    ),
                    None,
                )
                if leader is None:
                    self.initializing_root = root
                    break
                if leader.thread is self.thread:
                    raise bootstrap.RecoveryRequired(
                        "recursive_authority_initialization"
                    )
                _changed.wait(0.01)
        try:
            yield
        finally:
            with _changed:
                self.initializing_root = None
                _changed.notify_all()

    def close(self):
        with _changed:
            _pending_acquisitions.discard(self)
            _changed.notify_all()


class _LocalPause:
    """Private local gate authority, NOT a native maintenance/capture capability.

    Startup retirement requires the installed runtime to settle its producers.
    Neither zero counts nor a source census alone grants native maintenance.
    """

    def __init__(self):
        raise TypeError("local_pause_is_coordinator_issued")

    def _check(self):
        if (
            _pause is not self
            or self.pid != os.getpid()
            or self.thread is not threading.current_thread()
            or self.task is not _task_identity()
        ):
            raise bootstrap.RecoveryRequired("local_pause_inactive")

    def drain(self, deadline: float) -> bool:
        """Wait only for observed retirement; never close another thread's owner."""
        with _changed:
            _LocalPause._check(self)
            while True:
                # Startup stays in place; unsupported-native startup also prevents
                # any claim that this local cohort has qualified native retirement.
                startups = set(_startups.values())
                if (
                    not _pending_acquisitions
                    and not _operations
                    and not _raw_operations
                    and not (_live_leases - startups)
                    and not _retiring_holds
                    and all(lease._key is not None for lease in startups)
                ):
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                _changed.wait(min(remaining, 0.05))
                _LocalPause._check(self)

    def require_runtime_coverage(self, runtime=None) -> None:
        _LocalPause._check(self)
        from .runtime_maintenance import RuntimeMaintenance

        if type(runtime) is not RuntimeMaintenance:
            raise bootstrap.RecoveryRequired("participant_runtime_coverage_incomplete")
        RuntimeMaintenance._require_storage_coverage(runtime, self)

    def retire_startup(self, runtime) -> None:
        """Yield process enrollment only after the installed app actually drains."""
        self.require_runtime_coverage(runtime)
        with _changed:
            _LocalPause._check(self)
            key = (os.getpid(), str(bootstrap.default_bootstrap_root()))
            if getattr(self, "_startup_retired", False) or set(_startups) != {key}:
                raise bootstrap.RecoveryRequired("runtime_startup_scope_unqualified")
            lease = _startups[key]
            hold = _holds.get(lease._key)
            if hold is None or hold.count != 1 or hold.key != key:
                raise bootstrap.RecoveryRequired("runtime_native_resources_not_settled")
            self._startup_source = (
                key, effective_config_path(), hold.names, hold.authority._identity
            )
            _, profiles = bootstrap._records(bootstrap.default_bootstrap_root())
            previous = next(
                (r for r in profiles if r["selector"] == str(effective_config_path())),
                None,
            )
            self._startup_roots = (
                tuple(previous["roots"])
                if previous is not None
                and tuple(previous["namespaces"]) == hold.names
                else None
            )
            self._startup_retired = True
            self._startup_thread = None
            self._startup_error = None
            del _startups[key]
        lease.close()

    async def reacquire_startup(self) -> None:
        """Re-enroll under the native gate before reopening any ordinary owner."""
        _LocalPause._check(self)
        if not getattr(self, "_startup_retired", False):
            return
        if self._startup_thread is None:
            self._startup_thread = threading.Thread(
                target=_reacquire_paused_startup,
                args=(self,),
                name="chatbook-startup-readmission",
                daemon=True,
            )
            self._startup_thread.start()
        # The caller owns this attempt through completion. Cancellation of an
        # outer UI waiter must not abandon native re-enrollment or open the gate.
        cancellation = None
        while self._startup_thread.is_alive():
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
        _LocalPause._check(self)
        if self._startup_error is not None:
            raise bootstrap.RecoveryRequired("startup_reacquisition_failed") from None
        with _changed:
            key = self._startup_source[0]
            lease = _startups.get(key)
            if lease is None or lease._key != key or _retiring_holds:
                raise bootstrap.RecoveryRequired("startup_reacquisition_failed")
            self._startup_retired = False
        if cancellation is not None:
            raise cancellation

    def resume(self) -> None:
        global _pause
        with _changed:
            _LocalPause._check(self)
            if getattr(self, "_startup_retired", False):
                raise bootstrap.RecoveryRequired("startup_reacquisition_required")
            _pause = None
            _changed.notify_all()


def _begin_local_pause() -> _LocalPause:
    global _pause
    with _changed:
        if _pause is not None:
            raise bootstrap.RecoveryRequired("local_pause_already_active")
        pause = object.__new__(_LocalPause)
        pause.pid = os.getpid()
        pause.thread = threading.current_thread()
        pause.task = _task_identity()
        _pause = pause
        for attempt in _pending_acquisitions:
            if attempt.operation is None:
                attempt.cancel.set()
        _changed.notify_all()
        return pause


class _StartupReacquisition(_Acquisition):
    """The exact pause-owned readmission thread may acquire startup, nothing else."""

    def __init__(self, pause):
        self.pause = pause
        self.pid = os.getpid()
        self.thread = threading.current_thread()
        self.task = _task_identity()
        self.initializing_root = None
        self.cancel = threading.Event()
        self.operation = None
        with _changed:
            self.check()
            _pending_acquisitions.add(self)

    def check(self, path=None):
        if (
            path is not None
            or _pause is not self.pause
            or self.pause._startup_thread is not threading.current_thread()
            or not self.pause._startup_retired
            or self.pause._startup_source[0]
            != (os.getpid(), str(bootstrap.default_bootstrap_root()))
            or self.pause._startup_source[1] != effective_config_path()
        ):
            raise bootstrap.RecoveryRequired("startup_reacquisition_invalid")


def _reacquire_paused_startup(pause):
    attempt = None
    lease = None
    try:
        attempt = _StartupReacquisition(pause)
        lease = _acquire_storage(None, attempt)
        with _changed:
            attempt.check()
            key, _, names, identity = pause._startup_source
            hold = _holds.get(lease._key)
            if (
                lease._key != key
                or hold is None
                or hold.names != names
                or hold.authority._identity != identity
                or key in _startups
            ):
                raise bootstrap.RecoveryRequired("startup_scope_changed")
            _startups[key] = lease
            lease = None
    except BaseException as error:
        pause._startup_error = error
    finally:
        if lease is not None:
            lease.close()
        if attempt is not None:
            attempt.close()


def _local_pause_requested() -> bool:
    """Probe the exact actual holders, including incomplete native retirement."""
    with _lock:
        holds = tuple(set(_holds.values()) | _retiring_holds)
    # Native filesystem probing must never run under the coordinator lock.
    requested = False
    for hold in holds:
        requested = hold.authority.pause_requested(hold.names) or requested
    return requested


class _Hold:
    def __init__(self, authority: Admission, names: tuple[str, ...], key):
        self.authority = authority
        self.key = key
        self.names = names
        self.count = 0
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run,
            args=(authority,),
            daemon=True,
            name="chatbook-storage-admission",
        )
        self.thread.start()

    def _run(self, authority: Admission) -> None:
        try:
            # A last pending token may cancel before native admission succeeds.
            # Waiting happens on this thread, never under the coordinator lock.
            with authority._admit(self.names, False, None, self.stop):
                self.ready.set()
                self.stop.wait()
        except BaseException as error:
            self.error = error
            self.ready.set()


def _execution_selection_for(path):
    root = bootstrap.default_bootstrap_root()
    selector = effective_config_path()
    source = lexical_path(path) if path is not None else None
    return (
        os.getpid(),
        root,
        selector,
        selector.resolve(),
        source,
        source.resolve() if source is not None else None,
    )


class StorageLease:
    """Idempotent retirement token; successful close releases actual participation."""

    def __init__(self, key: tuple[int, str] | None):
        self._key = key
        self._execution_selection = None
        self.resource_policy = None
        self.resource_path = None
        self.resource_thread = None
        self.resource_close_failed = False
        with _changed:
            _live_leases.add(self)

    def _attach_sqlite(self, policy, path):
        # Policy comes from private_sqlite's validated installed registry.
        self.resource_policy = policy
        self.resource_path = lexical_path(path)
        self.resource_thread = threading.current_thread()

    def execution_scope(self) -> tuple[Path, tuple[str, ...]]:
        """Return only the namespace group protected by this live native hold."""
        with _lock:
            key = self._key
            hold = _holds.get(key)
            if (
                self not in _live_leases
                or key is None
                or key[0] != os.getpid()
                or hold is None
                or not hold.ready.is_set()
                or hold.error is not None
                or hold.stop.is_set()
            ):
                raise bootstrap.RecoveryRequired("execution_scope_not_admitted")
            names = hold.authority._observed_groups.get(hold.names)
            if not names:
                raise bootstrap.RecoveryRequired("execution_scope_not_admitted")
            return Path(key[1]), names

    def execution_context(self, path: Path | None):
        """Verify the exact acquisition selection before borrowing its authority.

        None namespaces describe a real unqualified ordinary acquisition, never
        native admission. Only independent ordinary-history checks may use it.
        """
        observed = _execution_selection_for(path)
        with _lock:
            if (
                self not in _live_leases
                or self._execution_selection is None
                or observed != self._execution_selection
            ):
                raise bootstrap.RecoveryRequired("execution_selection_changed")
            if self._key is None:
                return observed[1], None
            return self.execution_scope()

    def close(self) -> None:
        retired = None
        with _lock:
            key, self._key = self._key, None
            _live_leases.discard(self)
            _changed.notify_all()
            if key is None or key[0] != os.getpid():
                return
            hold = _holds[key]
            hold.count -= 1
            if hold.count == 0:
                hold.stop.set()
                del _holds[key]
                _retiring_holds.add(hold)
                retired = hold
        if retired is not None:
            retired.thread.join()
            with _lock:
                if retired.error is None or isinstance(
                    retired.error, AdmissionCancelled
                ):
                    _retiring_holds.discard(retired)
                _changed.notify_all()

    def __enter__(self) -> "StorageLease":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def _contains_owned_path(root: Path, selected: Path) -> bool:
    """Authorize exact objects or descendants of a declared directory, never parents."""
    resolved_root = root.resolve(strict=True)
    with bootstrap.pinned_directory(resolved_root.parent) as parent:
        info = os.stat(resolved_root.name, dir_fd=parent, follow_symlinks=False)
    resolved_selected = selected.resolve()
    if stat.S_ISDIR(info.st_mode):
        return (
            resolved_selected == resolved_root
            or resolved_root in resolved_selected.parents
        )
    if not stat.S_ISREG(info.st_mode):
        return False
    try:
        selected_info = resolved_selected.stat()
    except FileNotFoundError:
        return False
    return (info.st_dev, info.st_ino) == (selected_info.st_dev, selected_info.st_ino)


def _scope(
    root: Path, selector: Path, path: Path | None, *, startup_attempt=None, authority=None
) -> tuple[str, ...]:
    pending, profiles = bootstrap._records(root)
    registry = bootstrap._registry(root)
    binding = bootstrap._binding(selector, profiles, registry) if profiles else None
    if type(startup_attempt) is _StartupReacquisition:
        _StartupReacquisition.check(startup_attempt, path)
        pause = startup_attempt.pause
        if (
            startup_attempt not in _pending_acquisitions
            or authority is None
            or authority._identity != pause._startup_source[3]
        ):
            raise bootstrap.RecoveryRequired("startup_scope_changed")
        if pause._startup_roots is not None:
            previous = next((r for r in profiles if r["selector"] == str(selector)), None)
            if (
                previous is None
                or tuple(previous["namespaces"]) != pause._startup_source[2]
                or tuple(previous["roots"]) != pause._startup_roots
            ):
                raise bootstrap.RecoveryRequired("startup_scope_changed")
            if binding is None and not pending:
                # Continue only the retired owner's unchanged mapping. This does
                # not update persisted enrollment or admit any ordinary caller.
                snapshot = dict(previous, fingerprint=bootstrap._fingerprint(selector))
                binding = bootstrap._binding(selector, [snapshot], registry)
    if binding is None and not pending:
        live = _holds.get((os.getpid(), str(root)))
        previous = next((r for r in profiles if r["selector"] == str(selector)), None)
        if (
            live is not None
            and previous is not None
            and live.names == tuple(previous["namespaces"])
        ):
            # Existing owners continue only inside their original verified mapping.
            # New process enrollment still requires the saved config fingerprint.
            snapshot = dict(previous, fingerprint=bootstrap._fingerprint(selector))
            binding = bootstrap._binding(selector, [snapshot], registry)
    if binding is None:
        if pending:
            raise bootstrap.RecoveryRequired("recovery_scope_uncertain")
        return (UNBOUND_NAMESPACE,)
    if path is not None and not any(
        _contains_owned_path(Path(p), path)
        for p in binding["roots"] + [binding["selector"]]
    ):
        raise bootstrap.RecoveryRequired("storage_scope_not_enrolled")
    return tuple(binding["namespaces"])


def acquire_storage(path: Path | None = None) -> StorageLease:
    """Acquire ordinary admission, reporting only bounded refusal codes."""
    attempt = _Acquisition()
    try:
        return _acquire_storage(path, attempt)
    except bootstrap.RecoveryRequired:
        raise
    except (OSError, ValueError, RuntimeError):
        raise bootstrap.RecoveryRequired("storage_admission_unavailable") from None
    finally:
        attempt.close()


def _acquire_storage(path: Path | None, attempt: _Acquisition) -> StorageLease:
    """Check fixed evidence and hold declared scope before any owned open/write.

    An ordinary seam used inside maintenance refuses promptly; later capture owners
    need a separately validated capability, never a boolean bypass here.
    """
    if _forked_with_owners:
        raise bootstrap.RecoveryRequired("forked_owner_restart_required")
    root = bootstrap.default_bootstrap_root()
    selector = effective_config_path()
    execution_selection = _execution_selection_for(path)
    if execution_selection[1:3] != (root, selector):
        raise bootstrap.RecoveryRequired("execution_selection_changed")
    with _lock:
        attempt.check(path)
        if (
            attempt.operation is not None
            and attempt.operation.key is not None
            and attempt.operation.key != (os.getpid(), str(root))
        ):
            raise bootstrap.RecoveryRequired("operation_native_scope_changed")
    with attempt.initializing(root, path):
        allowed, reason = bootstrap.startup_permission(selector, root)
        if not allowed:
            raise bootstrap.RecoveryRequired(reason)
        if getattr(_local, "admitted", False):
            raise bootstrap.RecoveryRequired("maintenance_requires_owner_capability")
        existing = root.parent
        while not existing.exists():
            existing = existing.parent
        allowed, reason = qualified_for("admission", existing)
        if not allowed:
            # Preserve a positively disjoint startup decision, while still limiting
            # each owner path to its verified scope. Native unavailability is not a
            # conflict with an unrelated operation and never qualifies maintenance.
            _scope(root, selector, lexical_path(path) if path is not None else None)
            allowed, reason = bootstrap.startup_permission(selector, root)
            if not allowed:
                raise bootstrap.RecoveryRequired(reason)
            with _lock:
                attempt.check(path)
                token = StorageLease(None)
                token._execution_selection = execution_selection
                return token
        # Opening existing authority can wait on the registry. Retiring unrelated
        # owners must remain possible while that or a native gate is contended.
        authority = admission_authority(root)
    with _lock:
        attempt.check(path)
        names = _scope(
            root, selector, lexical_path(path) if path is not None else None,
            startup_attempt=attempt, authority=authority,
        )
        key = (os.getpid(), str(root))
        hold = _holds.get(key)
        if hold is not None and names != hold.names:
            raise bootstrap.RecoveryRequired("close_owners_before_scope_change")
        if hold is None:
            hold = _Hold(authority, names, key)
            _holds[key] = hold
        hold.count += 1
        token = StorageLease(key)
        token._execution_selection = execution_selection
    # Count pending acquisitions before dropping the lock: a drain must see
    # them, and another acquiring thread must share this same native hold.
    try:
        while not hold.ready.wait(0.01):
            with _lock:
                attempt.check(path)
        if hold.error is not None:
            raise bootstrap.RecoveryRequired("storage_admission_unavailable")
        # Enrollment races an unbound selection. Revalidate after acquiring its
        # lease; never enter on a stale pre-enrollment decision.
        with _lock:
            attempt.check(path)
            allowed, reason = bootstrap.startup_permission(selector, root)
            if not allowed:
                raise bootstrap.RecoveryRequired(reason)
            if (
                _scope(
                    root, selector, lexical_path(path) if path is not None else None,
                    startup_attempt=attempt, authority=authority,
                )
                != names
            ):
                raise bootstrap.RecoveryRequired("storage_scope_changed")
        return token
    except BaseException:
        token.close()
        raise


def admit_startup() -> None:
    """Keep process enrollment from before runtime imports through process exit."""
    attempt = _Acquisition()
    try:
        _admit_startup(attempt)
    finally:
        attempt.close()


def _admit_startup(attempt: _Acquisition) -> None:
    bootstrap.require_startup_permission()
    key = (os.getpid(), str(bootstrap.default_bootstrap_root()))
    with _lock:
        attempt.check()
        if key in _startups:
            return
    try:
        lease = acquire_storage()
    except bootstrap.RecoveryRequired as error:
        raise SystemExit("Recovery required: " + str(error)) from None
    try:
        with _lock:
            attempt.check()
            selected = _startups.setdefault(key, lease)
    except BaseException:
        lease.close()
        raise
    if selected is not lease:
        lease.close()


def _shutdown() -> None:
    for lease in list(_startups.values()):
        lease.close()
    _startups.clear()


def _after_fork() -> None:
    # Forked children cannot inherit a fictitious live lease thread/refcount.
    # Spawn imports establish their own leases before loading runtime modules.
    global _lock, _holds, _retiring_holds, _startups, _forked_with_owners
    global _pending_acquisitions, _live_leases, _operations, _raw_operations, _pause
    global _changed, _operation_local
    _forked_with_owners = bool(
        _holds
        or _retiring_holds
        or _pending_acquisitions
        or _operations
        or _raw_operations
        or _live_leases
    )
    _pending_acquisitions = set()
    _live_leases = set()
    _operations = set()
    _raw_operations = set()
    _pause = None
    _operation_local = threading.local()
    _lock = threading.RLock()
    _changed = threading.Condition(_lock)
    _holds = {}
    _retiring_holds = set()
    _startups = {}


atexit.register(_shutdown)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


class _CaptureLease:
    """A capture connection cannot outlive the native maintenance session."""

    def __init__(self, scope):
        self.scope = scope
        self.connection = None
        self.resource_close_failed = False
        self.resource_closed = False
        scope.resources.append(self)

    def attach(self, connection):
        self.connection = connection
        self.resource_closed = False

    def native_closed(self):
        self.connection = None
        self.resource_closed = True

    def close(self):
        if self.resource_close_failed:
            raise bootstrap.RecoveryRequired("capture_resources_not_retired")
        if self in self.scope.resources:
            self.scope.resources.remove(self)

    def retire(self):
        if self.resource_close_failed:
            raise bootstrap.RecoveryRequired("capture_resources_not_retired")
        if self.connection is not None:
            # Retire the native handle before releasing its capture authority.
            closed = False
            try:
                sqlite3.Connection.close(self.connection)
                closed = True
            finally:
                if not closed:
                    self.resource_close_failed = True
            self.native_closed()
        self.close()


class _CaptureScope:
    def __init__(self, session, sources, staging, limits, byte_budget):
        self.session = session
        self.sources = sources
        self.staging = staging
        self.staging_identity = staging.stat()
        self.resources = []
        self.active = True
        self.limits = limits
        self.byte_budget = byte_budget
        self.copied_bytes = 0
        self.sqlite_snapshots = {}
        self.sqlite_targets = {}
        self.sqlite_directory = None
        self.sqlite_directory_identity = None

    def check(self):
        self.session._check()
        if not self.active or getattr(_local, "capture_scope", None) is not self:
            raise bootstrap.RecoveryRequired("capture_scope_inactive")
        current = self.staging.stat()
        if (current.st_dev, current.st_ino) != (
            self.staging_identity.st_dev,
            self.staging_identity.st_ino,
        ):
            raise bootstrap.RecoveryRequired("capture_staging_changed")

    def retire(self):
        try:
            if self.active and self.sqlite_snapshots:
                self.verify_sqlite_sources()
        finally:
            self.active = False
            for resource in tuple(self.resources):
                resource.retire()
            if self.sqlite_directory is not None:
                import shutil

                info = self.sqlite_directory.lstat()
                if (info.st_dev, info.st_ino) != self.sqlite_directory_identity:
                    raise ValueError("capture_sqlite_staging_changed")
                shutil.rmtree(self.sqlite_directory)
                self.sqlite_directory = None

    @staticmethod
    def _sqlite_identity(info):
        return (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_nlink,
            info.st_uid,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    def _sqlite_state(self, source):
        self.check()
        resources = _CaptureFileDescriptors(self)
        try:
            with resources.pinned_directory(source.parent) as parent:
                parent_info = os.fstat(parent)
                result = []
                for suffix in ("", "-wal", "-shm", "-journal"):
                    try:
                        info = os.stat(
                            source.name + suffix, dir_fd=parent, follow_symlinks=False
                        )
                    except FileNotFoundError:
                        result.append(None)
                        continue
                    if (
                        not stat.S_ISREG(info.st_mode)
                        or info.st_nlink != 1
                        or info.st_uid != os.geteuid()
                    ):
                        raise ValueError("capture_sqlite_source_changed")
                    result.append(self._sqlite_identity(info))
                if result[0] is None or (source, *result[0][:2]) not in self.sources:
                    raise ValueError("capture_sqlite_source_changed")
                if result[3] is not None and result[3][5]:
                    raise ValueError("capture_sqlite_hot_journal")
                return (self._sqlite_identity(parent_info), tuple(result))
        finally:
            resources.retire()

    def verify_sqlite_sources(self):
        for source, (before, target) in self.sqlite_snapshots.items():
            if self._sqlite_state(source) != before:
                raise ValueError("capture_sqlite_source_changed")
            if target is not None:
                self.verify_sqlite_target(target)

    def verify_sqlite_target(self, target):
        """Check materialized authority, excluding SQLite's private SHM work."""
        self.check()
        expected = self.sqlite_targets.get(target)
        if expected is None:
            return
        root = self.sqlite_directory.lstat()
        if (root.st_dev, root.st_ino) != self.sqlite_directory_identity:
            raise ValueError("capture_sqlite_staging_changed")
        resources = _CaptureFileDescriptors(self)
        try:
            with resources.pinned_directory(target.parent) as parent:
                info = os.fstat(parent)
                if (info.st_dev, info.st_ino) != expected[0]:
                    raise ValueError("capture_sqlite_target_changed")
                for suffix, identity in zip(("", "-wal"), expected[1], strict=True):
                    try:
                        info = os.stat(
                            target.name + suffix, dir_fd=parent, follow_symlinks=False
                        )
                    except FileNotFoundError:
                        info = None
                    actual = None if info is None else self._sqlite_identity(info)
                    if actual != identity:
                        raise ValueError("capture_sqlite_target_changed")
        finally:
            resources.retire()

    def sqlite_target(self, owner_id, source, progress_guard=None):
        """Materialize only a native-qualified main/WAL set, never live SHM."""
        from tldw_chatbook.DB.private_sqlite import SQLITE_OWNER_REGISTRY

        from .space import require_capacity

        self.check()
        if not SQLITE_OWNER_REGISTRY[owner_id].recovery_capture_allowed:
            raise bootstrap.RecoveryRequired("capture_owner_not_registered")
        source = lexical_path(source)
        if self.staging in source.parents:
            self.verify_sqlite_target(source)
            return None
        if progress_guard is not None:
            progress_guard()
        before = self._sqlite_state(source)
        if source in self.sqlite_snapshots:
            original, target = self.sqlite_snapshots[source]
            if original != before:
                raise ValueError("capture_sqlite_source_changed")
            if target is None:
                raise ValueError("capture_sqlite_copy_incomplete")
            self.verify_sqlite_target(target)
            return target
        required = sum(info[5] for info in before[1][:2] if info is not None)
        if (
            any(
                info is not None and info[5] > self.limits.member_bytes
                for info in before[1][:2]
            )
            or self.copied_bytes + required > self.byte_budget
        ):
            raise ValueError("capture_sqlite_limit")
        require_capacity({self.staging: required * 2})
        resources = _CaptureFileDescriptors(self)
        try:
            if self.sqlite_directory is None:
                from uuid import uuid4

                name = "sqlite-sources-" + uuid4().hex
                with resources.pinned_directory(self.staging) as parent:
                    info = os.fstat(parent)
                    if (info.st_dev, info.st_ino) != (
                        self.staging_identity.st_dev,
                        self.staging_identity.st_ino,
                    ):
                        raise ValueError("capture_sqlite_staging_changed")
                    os.mkdir(name, mode=0o700, dir_fd=parent)
                    self.sqlite_directory = self.staging / name
                    created = os.stat(name, dir_fd=parent, follow_symlinks=False)
                    self.sqlite_directory_identity = created.st_dev, created.st_ino
            target_root = self.sqlite_directory / str(len(self.sqlite_snapshots))
            with resources.pinned_directory(self.sqlite_directory) as parent:
                info = os.fstat(parent)
                if (info.st_dev, info.st_ino) != self.sqlite_directory_identity:
                    raise ValueError("capture_sqlite_staging_changed")
                os.mkdir(target_root.name, mode=0o700, dir_fd=parent)
                created = os.stat(
                    target_root.name, dir_fd=parent, follow_symlinks=False
                )
                target_identity = created.st_dev, created.st_ino
            target = target_root / "source.sqlite3"
            # Failed attempts remain unqualified until scope retirement.
            self.sqlite_snapshots[source] = before, None
            private_identities = [None, None]
            wal_header = False
            main_header = bytearray()
            with resources.pinned_directory(source.parent) as parent:
                if self._sqlite_identity(os.fstat(parent)) != before[0]:
                    raise ValueError("capture_sqlite_source_changed")
                for suffix, expected in zip(("", "-wal"), before[1][:2], strict=True):
                    if expected is None and not (suffix == "-wal" and wal_header):
                        continue
                    if expected is not None:
                        descriptor = os.open(
                            source.name + suffix,
                            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                            dir_fd=parent,
                        )
                        resources.fds.append(descriptor)
                        if self._sqlite_identity(os.fstat(descriptor)) != expected:
                            raise ValueError("capture_sqlite_source_changed")
                    with resources.pinned_directory(target_root) as output_parent:
                        info = os.fstat(output_parent)
                        if (info.st_dev, info.st_ino) != target_identity:
                            raise ValueError("capture_sqlite_staging_changed")
                        output = os.open(
                            target.name + suffix,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                            0o600,
                            dir_fd=output_parent,
                        )
                        resources.fds.append(output)
                    count = 0
                    while expected is not None:
                        if progress_guard is not None:
                            progress_guard()
                        self.check()
                        chunk = os.read(
                            descriptor, min(1024**2, expected[5] - count + 1)
                        )
                        if not chunk:
                            break
                        if not suffix and len(main_header) < 20:
                            # WAL-header databases need an empty WAL even when
                            # the original was cleanly checkpointed. Create and
                            # bind it now; never adopt a later replacement.
                            main_header.extend(chunk[: 20 - len(main_header)])
                            wal_header = (
                                main_header[:16] == b"SQLite format 3\x00"
                                and main_header[18:20] == b"\x02\x02"
                            )
                        count += len(chunk)
                        if count > expected[5]:
                            raise ValueError("capture_sqlite_source_changed")
                        view = memoryview(chunk)
                        while view:
                            written = os.write(output, view)
                            if not written:
                                raise OSError("capture_sqlite_write_failed")
                            view = view[written:]
                    if expected is not None and (
                        count != expected[5]
                        or self._sqlite_identity(os.fstat(descriptor)) != expected
                    ):
                        raise ValueError("capture_sqlite_source_changed")
                    current_parent = target_root.lstat()
                    current = (target_root / (target.name + suffix)).lstat()
                    held = os.fstat(output)
                    if (
                        current_parent.st_dev,
                        current_parent.st_ino,
                    ) != target_identity or (current.st_dev, current.st_ino) != (
                        held.st_dev,
                        held.st_ino,
                    ):
                        raise ValueError("capture_sqlite_staging_changed")
                    private_identities[bool(suffix)] = self._sqlite_identity(held)
        finally:
            try:
                self.verify_sqlite_sources()
            finally:
                resources.retire()
        self.copied_bytes += required
        self.sqlite_targets[target] = target_identity, tuple(private_identities)
        self.sqlite_snapshots[source] = before, target
        return target


def _capture_sqlite_target(owner_id, source):
    scope = getattr(_local, "capture_scope", None)
    return None if scope is None else scope.sqlite_target(owner_id, source)


def _verify_capture_sqlite_target(target):
    scope = getattr(_local, "capture_scope", None)
    if scope is not None:
        scope.verify_sqlite_target(target)


@contextmanager
def _capture_sqlite_source(owner_id, source, progress_guard):
    scope = getattr(_local, "capture_scope", None)
    if scope is None:
        yield source
        return
    try:
        target = scope.sqlite_target(owner_id, source, progress_guard)
        yield source if target is None else target
    finally:
        scope.verify_sqlite_sources()


class _DiscoveryScope:
    """Read-only installed owner probes within an already native-held scope."""

    def __init__(self, session):
        self.session = session
        self.resources = []
        self.active = True

    def check(self):
        self.session._check()
        if not self.active or getattr(_local, "discovery_scope", None) is not self:
            raise bootstrap.RecoveryRequired("discovery_scope_inactive")

    def retire(self):
        self.active = False
        for resource in tuple(self.resources):
            resource.retire()


class _PreviewScope:
    """Bounded read probes over private SQLite copies, without live admission."""

    def __init__(self, limits, byte_budget):
        self.pid = os.getpid()
        self.thread = threading.get_ident()
        self.resources = []
        self.active = True
        self.limits = limits
        self.byte_budget = byte_budget
        self.copied_bytes = 0
        self.directory = None
        self.snapshots = {}

    def check(self):
        if (
            not self.active
            or self.pid != os.getpid()
            or self.thread != threading.get_ident()
            or getattr(_local, "preview_scope", None) is not self
        ):
            raise bootstrap.RecoveryRequired("preview_scope_inactive")

    @staticmethod
    def _source_state(source):
        result = []
        for suffix in ("", "-wal", "-journal"):
            path = source.with_name(source.name + suffix)
            try:
                info = path.lstat()
            except FileNotFoundError:
                result.append(None)
                continue
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_uid != os.geteuid()
            ):
                raise ValueError("preview_sqlite_unavailable")
            result.append(
                (
                    info.st_dev,
                    info.st_ino,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                )
            )
        if result[0] is None or result[2] is not None and result[2][2]:
            # A hot rollback journal needs recovery; preview never runs that on
            # the source or guesses a committed image from an incomplete copy.
            raise ValueError("preview_sqlite_unavailable")
        return tuple(result)

    def sqlite_target(self, source):
        """Copy a stable main/WAL set once; never open live SQLite to back it up."""
        import shutil
        import tempfile

        from .native_files import create_private_directory, create_private_file
        from .space import require_capacity

        self.check()
        source = lexical_path(source)
        before = self._source_state(source)
        if source in self.snapshots:
            previous, target = self.snapshots[source]
            if previous != before:
                raise ValueError("preview_sqlite_changed")
            return target
        required = sum(state[2] for state in before[:2] if state is not None)
        if (
            any(
                state is not None and state[2] > self.limits.member_bytes
                for state in before[:2]
            )
            or self.copied_bytes + required > self.byte_budget
        ):
            raise ValueError("preview_sqlite_limit")
        temporary_parent = Path(tempfile.gettempdir()).resolve()
        require_capacity({temporary_parent: required * 2})
        if self.directory is None:
            self.directory = Path(
                tempfile.mkdtemp(prefix="chatbook-preview-", dir=temporary_parent)
            )
        target_root = self.directory / str(len(self.snapshots))
        create_private_directory(target_root)
        target = target_root / "source.sqlite3"
        try:
            with bootstrap.pinned_directory(source.parent) as parent:
                for suffix, expected in zip(("", "-wal"), before[:2], strict=True):
                    if expected is None:
                        continue
                    descriptor = os.open(
                        source.name + suffix,
                        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                        dir_fd=parent,
                    )
                    try:
                        opened = os.fstat(descriptor)
                        if (
                            opened.st_dev,
                            opened.st_ino,
                            opened.st_size,
                            opened.st_mtime_ns,
                            opened.st_ctime_ns,
                        ) != expected:
                            raise ValueError("preview_sqlite_changed")
                        with create_private_file(
                            target.with_name(target.name + suffix)
                        ) as output:
                            count = 0
                            while chunk := os.read(
                                descriptor, min(1024**2, expected[2] - count + 1)
                            ):
                                count += len(chunk)
                                if count > expected[2]:
                                    raise ValueError("preview_sqlite_changed")
                                view = memoryview(chunk)
                                while view:
                                    written = os.write(output, view)
                                    if not written:
                                        raise OSError("preview_sqlite_write_failed")
                                    view = view[written:]
                            info = os.fstat(descriptor)
                            observed = (
                                info.st_dev,
                                info.st_ino,
                                info.st_size,
                                info.st_mtime_ns,
                                info.st_ctime_ns,
                            )
                            if count != expected[2] or observed != expected:
                                raise ValueError("preview_sqlite_changed")
                    finally:
                        os.close(descriptor)
            if self._source_state(source) != before:
                raise ValueError("preview_sqlite_changed")
        except BaseException:
            shutil.rmtree(target_root)
            raise
        self.copied_bytes += required
        self.snapshots[source] = before, target
        return target

    def retire(self):
        import shutil

        self.active = False
        for resource in tuple(self.resources):
            resource.retire()
        if self.directory is not None:
            shutil.rmtree(self.directory)
            self.directory = None


def _preview_sqlite_target(source):
    scope = getattr(_local, "preview_scope", None)
    return None if scope is None else scope.sqlite_target(source)


@contextmanager
def _preview_reads(*, limits=None, byte_budget=None):
    """Keep installed read-only discovery from bootstrapping ordinary services."""
    from .limits import ArchiveLimits

    limits = ArchiveLimits() if limits is None else limits
    if type(limits) is not ArchiveLimits:
        raise TypeError("invalid_preview_limits")
    byte_budget = limits.expanded_bytes if byte_budget is None else byte_budget
    if type(byte_budget) is not int or not 0 < byte_budget <= limits.expanded_bytes:
        raise ValueError("invalid_preview_budget")
    if (
        getattr(_local, "preview_scope", None) is not None
        or getattr(_local, "maintenance_session", None) is not None
    ):
        raise bootstrap.RecoveryRequired("nested_preview_scope")
    scope = _PreviewScope(limits, byte_budget)
    _local.preview_scope = scope
    try:
        yield
    finally:
        try:
            scope.retire()
        finally:
            _local.preview_scope = None


class MaintenanceSession:
    """Opaque native-held executor authority; only Admission can mint a session.

    Captured sources are exact regular files. Staging is a precreated private
    directory, disjoint from all enrolled and control roots. Connection handles
    are retired before scope exit, even if a caller keeps a Python reference.
    """

    def __init__(self):
        raise TypeError("maintenance_session_is_native_issued")

    def _check(self):
        if (
            not self._active
            or self._pid != os.getpid()
            or self._thread != threading.get_ident()
            or getattr(_local, "maintenance_session", None) is not self
        ):
            raise bootstrap.RecoveryRequired("maintenance_session_inactive")

    def _discover_capture_inventory(self, config_paths, selections, approved_scope):
        """Qualify first-use sources from installed discovery under native locks.

        Ordinary backup need not rewrite the live client's profile enrollment.
        Only files rediscovered here, within the held namespace roots, gain this
        session-local authority. A caller-supplied Inventory grants nothing.
        """
        from .inventory import discover
        from .models import DiscoverySelections

        self._check()
        if (
            type(config_paths) is not tuple
            or not config_paths
            or type(selections) is not DiscoverySelections
        ):
            raise ValueError("invalid_capture_selection")
        if UNBOUND_NAMESPACE not in self._names:
            raise bootstrap.RecoveryRequired("capture_unbound_admission_required")
        selectors = tuple(lexical_path(path) for path in config_paths)
        if any(
            not any(_contains_owned_path(root, path) for root in self._roots)
            for path in selectors
        ):
            raise bootstrap.RecoveryRequired("capture_source_outside_scope")
        with self._discovery_reads():
            current = discover(selectors, selections=selections)
        if current.scope_digest != approved_scope:
            raise ValueError("scope_changed")
        sources = []
        for item in current.items:
            if item.status != "included" or item.path is None:
                continue
            path = lexical_path(item.path)
            if not any(_contains_owned_path(root, path) for root in self._roots):
                raise bootstrap.RecoveryRequired("capture_source_outside_scope")
            info = path.stat()
            if not stat.S_ISREG(info.st_mode):
                raise bootstrap.RecoveryRequired("capture_file_not_regular")
            sources.append((path.resolve(strict=True), info.st_dev, info.st_ino))
        self._discovered_sources = tuple(sources)
        return current

    @contextmanager
    def _discovery_reads(self):
        """Allow bounded installed read probes, never writes, before staging."""
        self._check()
        if getattr(_local, "discovery_scope", None) is not None:
            raise bootstrap.RecoveryRequired("nested_discovery_scope")
        scope = _DiscoveryScope(self)
        self._scopes.append(scope)
        _local.discovery_scope = scope
        try:
            yield
        finally:
            try:
                scope.retire()
            finally:
                _local.discovery_scope = None

    @contextmanager
    def capture_scope(
        self, sources: tuple[Path, ...], staging: Path, *, limits=None, byte_budget=None
    ):
        from .limits import ArchiveLimits

        limits = ArchiveLimits() if limits is None else limits
        if type(limits) is not ArchiveLimits:
            raise TypeError("invalid_capture_limits")
        byte_budget = limits.expanded_bytes if byte_budget is None else byte_budget
        if type(byte_budget) is not int or not 0 < byte_budget <= limits.expanded_bytes:
            raise ValueError("invalid_capture_budget")
        self._check()
        if getattr(_local, "capture_scope", None) is not None:
            raise bootstrap.RecoveryRequired("nested_capture_scope")
        if type(sources) is not tuple or not sources:
            raise bootstrap.RecoveryRequired("capture_sources_required")
        root = bootstrap.default_bootstrap_root()
        if self._control.resolve(strict=True) != (root / "admission").resolve(
            strict=True
        ):
            raise bootstrap.RecoveryRequired("conflicting_admission_authority")
        with bootstrap.pinned_directory(self._control) as control_fd:
            identity = os.fstat(control_fd)
            if (identity.st_dev, identity.st_ino) != self._control_identity:
                raise bootstrap.RecoveryRequired("capture_authority_changed")
        if UNBOUND_NAMESPACE not in self._names:
            raise bootstrap.RecoveryRequired("capture_unbound_admission_required")
        _, profiles = bootstrap._records(root)
        registry = bootstrap._registry(root)
        bindings = [
            binding
            for profile in profiles
            if (
                binding := bootstrap._binding(
                    Path(profile["selector"]), profiles, registry
                )
            )
            is not None
            and set(binding["namespaces"]) <= set(self._names)
        ]
        selected = []
        for source in sources:
            source = lexical_path(source)
            info = source.stat()
            if not stat.S_ISREG(info.st_mode) or not any(
                _contains_owned_path(root, source) for root in self._roots
            ):
                raise bootstrap.RecoveryRequired("capture_source_outside_scope")
            bound = any(
                _contains_owned_path(Path(owned), source)
                for binding in bindings
                for owned in binding["roots"]
            )
            discovered = (
                source.resolve(strict=True),
                info.st_dev,
                info.st_ino,
            ) in getattr(self, "_discovered_sources", ())
            if not bound and not discovered:
                raise bootstrap.RecoveryRequired("capture_source_binding_unverified")
            selected.append((source.resolve(strict=True), info.st_dev, info.st_ino))
        staging = lexical_path(staging)
        with bootstrap.pinned_directory(staging) as fd:
            info = os.fstat(fd)
            if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise bootstrap.RecoveryRequired("capture_staging_not_private")
        staging = staging.resolve(strict=True)
        for root in self._all_roots + (self._control,):
            root = root.resolve(strict=True)
            if root == staging or root in staging.parents or staging in root.parents:
                raise bootstrap.RecoveryRequired("capture_staging_overlaps_source")
        scope = _CaptureScope(self, tuple(selected), staging, limits, byte_budget)
        self._scopes.append(scope)
        _local.capture_scope = scope
        try:
            yield
        finally:
            try:
                scope.retire()
            finally:
                if getattr(_local, "capture_scope", None) is scope:
                    _local.capture_scope = None

    def _retire(self):
        self._active = False
        for scope in self._scopes:
            scope.retire()
        if getattr(_local, "maintenance_session", None) is self:
            _local.maintenance_session = None


def _mint_maintenance_session(roots, all_roots, control, names, control_identity):
    session = object.__new__(MaintenanceSession)
    session._roots = tuple(roots)
    session._all_roots = tuple(all_roots)
    session._control = control
    session._names = names
    session._control_identity = control_identity
    session._pid = os.getpid()
    session._thread = threading.get_ident()
    session._active = True
    session._scopes = []
    _local.maintenance_session = session
    return session


# A native-close failure must retain the actual locks until process exit. This is
# intentionally a conservative quarantine, never GC-driven retry or lock release.
_failed_capture_holds = []


def _acquire_capture_storage(path: Path, *, owner_id: str, read_only: bool):
    """Private-seam consumer of an installed native-held scope, not a bypass flag.

    Owner IDs are resolved through the installed SQLite registry; strings do not
    grant authority. Session identity, physical paths and direction are checked,
    and the returned lease tracks the actual connection lifetime.
    """
    scope = getattr(_local, "capture_scope", None)
    if scope is None:
        scope = getattr(_local, "discovery_scope", None) or getattr(
            _local, "preview_scope", None
        )
        if scope is None:
            return None
        scope.check()
        from tldw_chatbook.DB.private_sqlite import SQLITE_OWNER_REGISTRY

        if (
            not read_only
            or not SQLITE_OWNER_REGISTRY[owner_id].recovery_capture_allowed
        ):
            raise bootstrap.RecoveryRequired("discovery_read_only_required")
        selected = lexical_path(path)
        if type(scope) is _DiscoveryScope and not any(
            _contains_owned_path(root, selected) for root in scope.session._roots
        ):
            raise bootstrap.RecoveryRequired("capture_source_outside_scope")
        return _CaptureLease(scope)
    scope.check()
    from tldw_chatbook.DB.private_sqlite import SQLITE_OWNER_REGISTRY

    if not SQLITE_OWNER_REGISTRY[owner_id].recovery_capture_allowed:
        raise bootstrap.RecoveryRequired("capture_owner_not_registered")
    selected = lexical_path(path).resolve()
    if read_only:
        try:
            info = selected.stat()
        except FileNotFoundError:
            info = None
        if info is not None and any(
            (info.st_dev, info.st_ino) == (device, inode)
            for _, device, inode in scope.sources
        ):
            return _CaptureLease(scope)
    if scope.staging not in selected.parents:
        raise bootstrap.RecoveryRequired("capture_path_outside_scope")
    if selected.exists():
        info = selected.stat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or any(
                (info.st_dev, info.st_ino) == (device, inode)
                for _, device, inode in scope.sources
            )
        ):
            raise bootstrap.RecoveryRequired("capture_target_alias")
    return _CaptureLease(scope)


class _CaptureFileDescriptors:
    """Operation-private FDs with conservative unresolved-close quarantine."""

    def __init__(self, scope):
        self.scope = scope
        self.fds = []
        self.close_failed = False
        if scope is not None:
            scope.resources.append(self)

    def close_descriptor(self, fd: int) -> None:
        """Retire once, including parent traversal; ambiguity retains exclusion."""
        if self.close_failed:
            raise bootstrap.RecoveryRequired("capture_resources_not_retired")
        try:
            os.close(fd)
        except BaseException:
            # Never retry an FD number after an ambiguous native outcome,
            # including attempts by traversal's exception cleanup.
            self.close_failed = True
            raise bootstrap.RecoveryRequired("capture_resources_not_retired") from None

    def pinned_directory(self, root: Path):
        return bootstrap.pinned_directory(root, _close=self.close_descriptor)

    def retire(self):
        if self.close_failed:
            raise bootstrap.RecoveryRequired("capture_resources_not_retired")
        while self.fds:
            fd = self.fds[-1]
            self.close_descriptor(fd)
            self.fds.pop()
        if self.scope is not None and self in self.scope.resources:
            self.scope.resources.remove(self)


def _check_capture_file_identity(scope, selected, info, *, source_only=False):
    scope.check()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise bootstrap.RecoveryRequired("capture_file_not_regular")
    if type(scope) is _DiscoveryScope:
        if not any(
            _contains_owned_path(root, selected) for root in scope.session._roots
        ):
            raise bootstrap.RecoveryRequired("capture_source_outside_scope")
        return
    if type(scope) is _PreviewScope:
        return
    source_identity = any(
        (info.st_dev, info.st_ino) == (dev, ino) for _, dev, ino in scope.sources
    )
    if scope.staging in selected.parents and not source_only:
        if source_identity:
            raise bootstrap.RecoveryRequired("capture_target_alias")
        return
    if not source_identity:
        raise bootstrap.RecoveryRequired("capture_source_outside_scope")


# Explicit installed owners; identifiers grant no path or maintenance authority.
_RAW_RECOVERY_LIMITS = {
    "rag.definitions": 16 * 1024**2,
    "rag.projections": 256 * 1024**3,
    "recovered.media": 512 * 1024**2,
    "generation.assets": 256 * 1024**3,
    "diagnostics.logs": 256 * 1024**3,
    "tokenizers.custom": 256 * 1024**3,
    "skills": 256 * 1024**3,
    "persona.assets": 256 * 1024**3,
    "persona.visual_identity": 256 * 1024**3,
    "persona.visual_identity_builtin": 256 * 1024**3,
    "tts.voices": 256 * 1024**3,
    "models.artifacts": 256 * 1024**3,
    "config": 16 * 1024**2,
    "config.history": 16 * 1024**2,
    "personas": 256 * 1024**3,
    "chat.dictionary_history": 256 * 1024**3,
    "chat.rag_context": 256 * 1024**3,
    "chat.grammars": 256 * 1024**3,
    "feedback": 256 * 1024**3,
    "audio.history": 256 * 1024**3,
    "chat.prompt_history": 256 * 1024**3,
    "ui.state": 256 * 1024**3,
    "ui.emoji_recents": 256 * 1024**3,
    "ui.themes": 256 * 1024**3,
    "chatbooks.registry": 256 * 1024**3,
    "chatbooks.archives": 256 * 1024**3,
    "chat.dictionaries": 256 * 1024**3,
    "chunking.templates": 256 * 1024**3,
    "notes.templates": 256 * 1024**3,
    "chat.prompts": 256 * 1024**3,
    "generation.styles": 256 * 1024**3,
    "external.files": 256 * 1024**3,
    "eval.definitions": 1024**4,
    "mcp.local": 16 * 1024**2,
    "mcp.targets": 16 * 1024**2,
    "mcp.context": 16 * 1024**2,
    "mcp.permissions": 16 * 1024**2,
    "mcp.history": 256 * 1024**3,
    "runtime.source_state": 16 * 1024**2,
    "tamagotchi.config": 16 * 1024**2,
    "workspaces.change_tracking": 256 * 1024**3,
    "agents.history": 256 * 1024**3,
    "subscriptions.assets": 256 * 1024**3,
}


def _recovery_file_limit(owner_id: str, max_bytes: int) -> None:
    if owner_id not in _RAW_RECOVERY_LIMITS:
        raise bootstrap.RecoveryRequired("capture_owner_not_registered")
    if (
        type(max_bytes) is not int
        or max_bytes <= 0
        or max_bytes > _RAW_RECOVERY_LIMITS[owner_id]
    ):
        raise ValueError("invalid_capture_byte_limit")


def _read_recovery_file(owner_id: str, candidate: Path, *, max_bytes: int) -> bytes:
    """Return bounded definition bytes only after positive reader retirement."""
    if type(max_bytes) is not int or max_bytes <= 0 or max_bytes > 16 * 1024**2:
        raise ValueError("invalid_capture_byte_limit")
    return _consume_recovery_file(
        owner_id, candidate, max_bytes=max_bytes, collect=True
    )


def _staged_credential_path(scope, candidate: Path, *, allow_missing=False):
    """Require a private staged regular file, distinct from captured sources."""
    scope.check()
    selected = lexical_path(candidate)
    if scope.staging not in selected.parents:
        raise bootstrap.RecoveryRequired("capture_path_outside_scope")
    try:
        info = selected.stat(follow_symlinks=False)
    except FileNotFoundError:
        if allow_missing:
            return selected, None
        raise
    _check_capture_file_identity(scope, selected, info)
    if info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise bootstrap.RecoveryRequired("credential_staging_required")
    return selected, info


def _read_staged_credential_file(candidate: Path, *, max_bytes: int) -> bytes | None:
    """Read only a native-held staged copy; None selects ordinary staging I/O."""
    scope = getattr(_local, "capture_scope", None)
    if scope is None:
        return None
    selected, _ = _staged_credential_path(scope, candidate)
    return _consume_recovery_file(
        "config", selected, max_bytes=max_bytes, collect=True, private=True
    )


def _write_staged_credential_file(candidate: Path, data: str) -> bool:
    """Write private staged text under capture authority, preserving sources."""
    scope = getattr(_local, "capture_scope", None)
    if scope is None:
        return False
    if type(data) is not str:
        raise TypeError("credential_text_required")
    encoded = data.encode("utf-8")
    if len(encoded) > 16 * 1024**2:
        raise ValueError("credential_resource_limit")
    selected, original = _staged_credential_path(scope, candidate, allow_missing=True)
    import uuid

    temporary = ".credential-" + uuid.uuid4().hex
    resources = _CaptureFileDescriptors(scope)
    try:
        with resources.pinned_directory(selected.parent) as parent:
            fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent,
            )
            resources.fds.append(fd)
            try:
                view = memoryview(encoded)
                while view:
                    count = os.write(fd, view)
                    if count <= 0:
                        raise OSError("capture_write_unavailable")
                    view = view[count:]
                os.fsync(fd)
                _, current = _staged_credential_path(
                    scope, selected, allow_missing=True
                )
                held_parent, current_parent = os.fstat(parent), selected.parent.stat()
                current_identity = (
                    None if current is None else (current.st_dev, current.st_ino)
                )
                original_identity = (
                    None if original is None else (original.st_dev, original.st_ino)
                )
                if current_identity != original_identity or (
                    held_parent.st_dev,
                    held_parent.st_ino,
                ) != (current_parent.st_dev, current_parent.st_ino):
                    raise bootstrap.RecoveryRequired("capture_target_changed")
                if original is None:
                    os.link(
                        temporary,
                        selected.name,
                        src_dir_fd=parent,
                        dst_dir_fd=parent,
                        follow_symlinks=False,
                    )
                else:
                    os.replace(
                        temporary, selected.name, src_dir_fd=parent, dst_dir_fd=parent
                    )
                os.fsync(parent)
            finally:
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass
    finally:
        resources.retire()
    return True


def _check_recovery_file(
    owner_id: str,
    candidate: Path,
    *,
    max_bytes: int,
    cancel: threading.Event | None = None,
) -> None:
    """Check opaque bytes in bounded chunks without exposing a native handle."""
    _consume_recovery_file(
        owner_id, candidate, max_bytes=max_bytes, collect=False, cancel=cancel
    )


def _digest_recovery_file(
    owner_id: str,
    candidate: Path,
    *,
    max_bytes: int,
    cancel: threading.Event | None = None,
) -> tuple[int, str]:
    """Return actual byte count/SHA256 only after positive native retirement."""
    return _consume_recovery_file(
        owner_id,
        candidate,
        max_bytes=max_bytes,
        collect=False,
        cancel=cancel,
        digest=True,
    )


def _consume_recovery_file(
    owner_id: str,
    candidate: Path,
    *,
    max_bytes: int,
    collect: bool,
    cancel: threading.Event | None = None,
    digest: bool = False,
    private: bool = False,
) -> bytes | tuple[int, str] | None:
    """Return bounded definition bytes after native reader retirement."""
    _recovery_file_limit(owner_id, max_bytes)
    selected = lexical_path(candidate)
    scope = (
        getattr(_local, "capture_scope", None)
        or getattr(_local, "discovery_scope", None)
        or getattr(_local, "preview_scope", None)
    )
    lease = acquire_storage(selected) if scope is None else None
    resources = _CaptureFileDescriptors(scope)
    try:
        if scope is not None:
            scope.check()
        with resources.pinned_directory(selected.parent) as parent:
            fd = os.open(
                selected.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=parent,
            )
            resources.fds.append(fd)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise bootstrap.RecoveryRequired("capture_file_not_regular")
            if private and (info.st_uid != os.geteuid() or info.st_mode & 0o077):
                raise bootstrap.RecoveryRequired("credential_staging_required")
            if scope is not None:
                _check_capture_file_identity(scope, selected, info)
            chunks = []
            import hashlib

            hasher = hashlib.sha256() if digest else None
            total = 0
            while True:
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("cancelled")
                if scope is not None:
                    scope.check()
                chunk = os.read(fd, min(1024**2, max_bytes - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("definition_byte_limit")
                if collect:
                    chunks.append(chunk)
                if hasher is not None:
                    hasher.update(chunk)
            if scope is not None:
                after = os.fstat(fd)
                _check_capture_file_identity(scope, selected, after)
                if private and (after.st_uid != os.geteuid() or after.st_mode & 0o077):
                    raise bootstrap.RecoveryRequired("credential_staging_required")
                current_parent = selected.parent.stat()
                held_parent = os.fstat(parent)
                if (current_parent.st_dev, current_parent.st_ino) != (
                    held_parent.st_dev,
                    held_parent.st_ino,
                ):
                    raise bootstrap.RecoveryRequired("capture_target_changed")
            if hasher is not None:
                return total, hasher.hexdigest()
            return b"".join(chunks) if collect else None
    finally:
        # If native retirement fails, ordinary admission is retained too. Never
        # release a lease around a potentially live descriptor.
        resources.retire()
        if lease is not None:
            lease.close()


def copy_capture_file(
    owner_id: str,
    source: Path,
    destination: Path,
    cancel: threading.Event,
    *,
    max_bytes: int,
) -> None:
    """Copy one enrolled regular file into private staging without exposed handles.

    The fixed native maintenance scope grants authority; the installed owner ID
    alone never does. Every acquired descriptor is retired before return, or its
    unresolved resource retains native exclusion through the existing quarantine.
    """
    _recovery_file_limit(owner_id, max_bytes)
    scope = getattr(_local, "capture_scope", None)
    if scope is None:
        raise bootstrap.RecoveryRequired("capture_requires_maintenance")
    scope.check()
    source = lexical_path(source)
    destination = lexical_path(destination)
    if scope.staging not in destination.parents:
        raise bootstrap.RecoveryRequired("capture_path_outside_scope")
    if cancel.is_set():
        raise InterruptedError("cancelled")
    resources = _CaptureFileDescriptors(scope)
    try:
        # Pinned no-follow parent traversal follows the established private-path
        # boundary. Native data descriptors stay tracked across every exception.
        with resources.pinned_directory(source.parent) as parent:
            fd = os.open(
                source.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
            )
            resources.fds.append(fd)
            info = os.fstat(fd)
            _check_capture_file_identity(scope, source, info, source_only=True)
        with resources.pinned_directory(destination.parent) as parent:
            destination_parent_identity = os.fstat(parent)
            scope.check()
            out = os.open(
                destination.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent,
            )
            resources.fds.append(out)
            if not stat.S_ISREG(os.fstat(out).st_mode):
                raise bootstrap.RecoveryRequired("capture_target_alias")
        total = 0
        while True:
            scope.check()
            if cancel.is_set():
                raise InterruptedError("cancelled")
            chunk = os.read(fd, min(1024**2, max_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("capture_byte_limit")
            view = memoryview(chunk)
            while view:
                count = os.write(out, view)
                if count <= 0:
                    raise OSError("capture_write_unavailable")
                view = view[count:]
        os.fsync(out)
        scope.check()
        current_parent = destination.parent.stat()
        if (current_parent.st_dev, current_parent.st_ino) != (
            destination_parent_identity.st_dev,
            destination_parent_identity.st_ino,
        ):
            raise bootstrap.RecoveryRequired("capture_target_changed")
        current_target = destination.stat(follow_symlinks=False)
        held_target = os.fstat(out)
        if (current_target.st_dev, current_target.st_ino) != (
            held_target.st_dev,
            held_target.st_ino,
        ):
            raise bootstrap.RecoveryRequired("capture_target_changed")
    finally:
        resources.retire()
