"""Process-local lifetime counting over one predeclared cross-process lease.

Lease ownership lives in a dedicated thread, so connections may close on a worker
thread. A process never widens a live scope: config changes require retirement and
re-enrollment. Native-unqualified ordinary use is distinct from recovery admission.
"""

from __future__ import annotations

import asyncio
import atexit
import copy
import sqlite3
import stat
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

from tldw_chatbook.Utils.platform_files import os

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

    def _state(self):
        """Validate issued metadata only; caller holds the coordinator."""
        from .participants import _installed_repositories

        if self not in _operations or self.participant not in _installed_repositories:
            raise bootstrap.RecoveryRequired("operation_provenance_invalid")
        repository = self.participant.repository()
        if (
            repository is None
            or self.participant.read_only != getattr(repository, "_read_only", False)
            or repository._maintenance_participant is not self.participant
            or self.pid != os.getpid()
            or self.thread is not threading.current_thread()
            or self.task is not _task_identity()
            or self.lease is None
            or self.lease not in _live_leases
        ):
            raise bootstrap.RecoveryRequired("operation_provenance_invalid")
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
        return (
            self.participant,
            repository,
            self.lease,
            self.path,
            self.resolved_path,
            self.parent_identity,
            self.key,
            self.hold,
            self.lease._key,
        )

    def check(self, path=None, *, fence=True):
        """Check issued state; with ``fence``, also natively re-prove ``path``.

        ``fence=False`` keeps every in-memory provenance/lexical-path check and
        skips only the native parent-identity and resolution walk. Only a
        single acquisition that already fenced this exact path uses it
        (TASK-34601 amendment to ADR-126).
        """
        with _lock:
            expected = _check_operation_state(self)
        fenced = path is not None and fence
        if fenced:
            selected = lexical_path(path)
            parent = os.stat(expected[3].parent)
            resolved = selected.resolve() if selected == expected[3] else None
        with _lock:
            _check_operation_state(self, expected, path)
            if fenced and (
                selected != expected[3]
                or resolved != expected[4]
                or (parent.st_dev, parent.st_ino) != expected[5]
            ):
                raise bootstrap.RecoveryRequired("operation_path_outside_scope")
        return expected


def _check_operation_state(operation, expected=None, path=None):
    """Fence exact issued metadata without performing filesystem observation."""
    if type(operation) is not _Operation:
        raise bootstrap.RecoveryRequired("operation_provenance_invalid")
    current = _Operation._state(operation)
    if expected is not None:
        if any(current[index] is not expected[index] for index in (0, 1, 2, 7)):
            raise bootstrap.RecoveryRequired("operation_provenance_invalid")
        if any(current[index] != expected[index] for index in (3, 4, 5, 6, 8)):
            raise bootstrap.RecoveryRequired("operation_path_outside_scope")
    if path is not None and (
        lexical_path(path) != current[3] or current[1].db_path != current[3]
    ):
        raise bootstrap.RecoveryRequired("operation_path_outside_scope")
    return current


def _check_operation(operation, path=None, *, fence=True):
    # Thread-local discovery is not authority. Never dispatch a caller-supplied
    # validation callback or an instance-shadowed method.
    if type(operation) is not _Operation:
        raise bootstrap.RecoveryRequired("operation_provenance_invalid")
    return _Operation.check(operation, path, fence=fence)


@contextmanager
def _repository_operation(participant):
    from .participants import _check_core_retirement, _installed_repositories

    previous = getattr(_operation_local, "operation", None)
    proof = _check_operation(previous, previous.path) if previous is not None else None
    reuse = previous is not None and previous.participant is participant
    target = participant.path
    if reuse and target != previous.path:
        _check_operation(previous, target)
    with _changed:
        if previous is not None:
            _check_operation_state(previous, proof, previous.path)
        if reuse:
            _check_operation_state(previous, proof, participant.path)
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
            parent = os.stat(participant.path.parent)
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
            _check_operation_state(operation, path=operation.path)
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
            proof = (
                _check_operation(previous, previous.path)
                if previous is not None
                else None
            )
            with _changed:
                if previous is not None:
                    _check_operation_state(previous, proof, previous.path)
                _operation_local.operation = previous


class _Acquisition:
    def __init__(self):
        self.pid = os.getpid()
        self.thread = threading.current_thread()
        self.task = _task_identity()
        self.initializing_root = None
        self.scope_roots = None
        self.cancel = threading.Event()
        self.operation = getattr(_operation_local, "operation", None)
        # Paths this one attempt already natively fenced (thread-confined).
        self._fenced_paths = set()
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
            # One native parent/resolution fence per path per acquisition: the
            # bracket checks before and after the lease count repeat only the
            # in-memory provenance, lexical-path, pause and cancel checks
            # (TASK-34601 amendment to ADR-126; previously 3-4 drive-root walks).
            selected = lexical_path(path)
            proof = _check_operation(
                self.operation, path, fence=selected not in self._fenced_paths
            )
            self._fenced_paths.add(selected)
            return proof
        elif _pause is not None:
            raise bootstrap.RecoveryRequired("storage_locally_paused")

    @contextmanager
    def initializing(self, root, path):
        # The marker and native registration are a single same-process first-use
        # interval. Durable incomplete state from any other interval still refuses.
        while True:
            operation = self.operation
            proof = self.check(path)
            with _changed:
                if (
                    self not in _pending_acquisitions
                    or self.pid != os.getpid()
                    or self.thread is not threading.current_thread()
                    or self.task is not _task_identity()
                    or self.operation is not operation
                ):
                    raise bootstrap.RecoveryRequired("acquisition_provenance_invalid")
                if self.cancel.is_set():
                    raise bootstrap.RecoveryRequired("storage_locally_paused")
                if type(self) is _StartupReacquisition:
                    if path is not None or operation is not None:
                        raise bootstrap.RecoveryRequired(
                            "startup_reacquisition_invalid"
                        )
                    _StartupReacquisition.check(self)
                elif operation is None:
                    if _pause is not None:
                        raise bootstrap.RecoveryRequired("storage_locally_paused")
                else:
                    _check_operation_state(operation, proof, path)
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
                key,
                effective_config_path(),
                hold.names,
                hold.authority._identity,
            )
            _, profiles = bootstrap._records(bootstrap.default_bootstrap_root())
            previous = next(
                (r for r in profiles if r["selector"] == str(effective_config_path())),
                None,
            )
            self._startup_roots = (
                tuple(previous["roots"])
                if previous is not None and tuple(previous["namespaces"]) == hold.names
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
            loop = asyncio.get_running_loop()
            completion = self._startup_completion = loop.create_future()

            def reacquire():
                try:
                    _reacquire_paused_startup(self)
                except BaseException as error:  # noqa: BLE001 - relay native cleanup failure.
                    self._startup_error = error
                finally:
                    loop.call_soon_threadsafe(completion.set_result, None)

            self._startup_thread = threading.Thread(
                target=reacquire,
                name="chatbook-startup-readmission",
                daemon=True,
            )
            self._startup_thread.start()
        # The caller owns this attempt through completion. Cancellation of an
        # outer UI waiter must not abandon native re-enrollment or open the gate.
        cancellation = None
        while not self._startup_completion.done():
            try:
                await asyncio.shield(self._startup_completion)
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
        # Change-notification handles pin their directories' ancestors against
        # rename; maintenance owns the tree from here, so release them now.
        retired = [w for hold in _holds.values() for w in _retire_watches(hold)]
        _changed.notify_all()
    _close_retired_watches(retired)
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
        key, _, names, identity = pause._startup_source
        authority = Admission.open_existing(Path(key[1]) / "admission")
        if authority._identity != identity:
            raise bootstrap.RecoveryRequired("startup_scope_changed")
        # A publishing isolated restore still has pending evidence while holding
        # the unbound gate. Wait behind its actual native finalization before
        # evaluating ordinary startup permission; surviving pending still refuses.
        with authority.normal(names):
            pass
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
        # PERF-07/08 (ADR-126 amendment 2026-09-29): process-local admission
        # evidence, discarded with this hold. Guarded by _lock.
        self.evidence: dict[str, _Evidence] = {}
        self.path_evidence: OrderedDict[tuple, _Evidence] = OrderedDict()
        self.derived_evidence: OrderedDict[tuple, tuple[_Evidence, object]] = (
            OrderedDict()
        )
        # TASK-34601 (Windows): change notifications over evidence last
        # confirmed by a full observation, keyed by entry identity (LRU).
        # Guarded by _lock.
        self.watches: OrderedDict[tuple, _EvidenceWatch] = OrderedDict()
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
            watches = []
            if hold.count == 0:
                hold.stop.set()
                del _holds[key]
                _retiring_holds.add(hold)
                retired = hold
                watches.extend(_retire_watches(hold))
        _close_retired_watches(watches)
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
        selected_info = os.stat(resolved_selected)
    except FileNotFoundError:
        return False
    return (info.st_dev, info.st_ino) == (selected_info.st_dev, selected_info.st_ino)


def _contains_capture_path(roots: Iterable[Path], selected: Path) -> bool:
    """Try likely roots first, retaining native checks and physical alias fallback."""
    candidates = {selected, *selected.parents}
    ordered = sorted(roots, key=lambda root: root not in candidates)
    return any(_contains_owned_path(root, selected) for root in ordered)


_CONFIG_CAPTURE_LEAVES = {
    "ui.state": "ui_state.toml",
    "ui.emoji_recents": "recent_emojis.json",
    "runtime.source_state": "runtime_policy.json",
}


def _config_capture_bindings(root, selectors):
    """Observe every bound selector that can write requested config siblings."""
    pending, profiles = bootstrap._records(root)
    requested = {str(lexical_path(path)) for path in selectors}
    parents = {
        Path(row["selector"]).parent for row in profiles if row["selector"] in requested
    }
    if not parents:
        return ()
    if pending:
        raise bootstrap.RecoveryRequired("capture_config_scope_changed")
    registry = bootstrap._registry(root)
    result = []
    for row in profiles:
        selector = Path(row["selector"])
        if selector.parent not in parents:
            continue
        if bootstrap._binding(selector, profiles, registry) != row:
            raise bootstrap.RecoveryRequired("capture_config_scope_changed")
        with bootstrap.pinned_directory(selector.parent) as parent:
            info = os.fstat(parent)
            if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise bootstrap.RecoveryRequired("capture_config_parent_unsafe")
            result.append((row, (info.st_dev, info.st_ino, info.st_mode)))
    return tuple(result)


def _config_capture_sources(root, selectors, bindings):
    """Derive only installed fixed leaves; foreign registry aliases still refuse."""
    from .service_storage import default_control_root, work_root

    requested = {str(lexical_path(path)) for path in selectors}
    registry = bootstrap._registry(root)
    controls = (root, default_control_root(), work_root(default_control_root()))
    result = []
    for row, parent_identity in bindings:
        if row["selector"] not in requested:
            continue
        # Related profiles are held for exclusion, never adopted as ownership.
        foreign = [
            entry for name, entry in registry.items() if name not in row["namespaces"]
        ]
        selector = Path(row["selector"])
        if not _contains_capture_path(map(Path, row["roots"]), selector):
            raise bootstrap.RecoveryRequired("capture_config_scope_changed")
        for owner, leaf in _CONFIG_CAPTURE_LEAVES.items():
            path = selector.parent / leaf
            tokens = {"path:" + str(path), "path:" + str(path.resolve())}
            try:
                info = os.stat(path, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1
                    or info.st_uid != os.geteuid()
                    or info.st_mode & 0o077
                ):
                    raise bootstrap.RecoveryRequired("capture_config_source_unsafe")
                tokens.add(bootstrap.inode_token(info))
            if any(bootstrap._overlap(path, control) for control in controls) or any(
                tokens.intersection(bootstrap.identity_view(entry["historical"]))
                or any(bootstrap._overlap(path, Path(p)) for p in entry["roots"])
                or any(
                    bootstrap._overlap(path, Path(token[5:]))
                    for token in entry["historical"]
                    if token.startswith("path:")
                )
                for entry in foreign
            ):
                raise bootstrap.RecoveryRequired("capture_config_scope_changed")
            result.append((path, owner, selector, parent_identity))
    return tuple(result)


def _config_capture_item(item, inventory, sources):
    """Require the installed rediscovered file's exact config dependency."""
    for path, owner, selector, _ in sources:
        if item.path != path or item.owner != owner:
            continue
        if any(
            row.owner == "config"
            and row.path == selector
            and item.dependencies == (row.logical_id,)
            and item.metadata is not None
            and item.metadata.kind == "file"
            and item.metadata.policy == "private"
            for row in inventory.items
        ):
            return True
    return False


def _config_capture_file(sources, selected, owner_id, info):
    """Read-only discovery checks the exact leaf and its observed private parent."""
    for path, owner, _, identity in sources:
        if selected != path or owner_id != owner:
            continue
        with bootstrap.pinned_directory(path.parent) as parent:
            current = os.fstat(parent)
            leaf = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if (
                (current.st_dev, current.st_ino, current.st_mode) != identity
                or not stat.S_ISREG(leaf.st_mode)
                or leaf.st_nlink != 1
                or leaf.st_uid != os.geteuid()
                or leaf.st_mode & 0o077
                or (leaf.st_dev, leaf.st_ino) != (info.st_dev, info.st_ino)
            ):
                raise bootstrap.RecoveryRequired("capture_config_source_changed")
        return True
    return False


# ---------------------------------------------------------------------------
# Reusable admission evidence (PERF-07/08; ADR-126 amendment, 2026-09-29).
#
# An ordinary acquisition may reuse the *allowed* result of an unmodified
# derivation while every stamp below is identical when re-observed on the same
# call. Posture stamps cover every path component the derivation walks; content
# stamps cover the records, registry, marker, selector and qualification file.
# Evidence becomes reusable only after two consecutive full derivations bracket
# identical stamps with content change times at least _EVIDENCE_SETTLE_NS old.
# Any mismatch runs the unmodified derivation, the only source of refusals.
# All evidence state is read and written under _lock; nothing relies on the
# GIL for atomicity (free-threaded builds, PEP 779).
# ---------------------------------------------------------------------------

#: Kill switch. The oracle test compares both settings after each mutation.
_EVIDENCE_REUSE = True
#: A content change must be this old before evidence over it is trusted.
_EVIDENCE_SETTLE_NS = 1_000_000_000
#: Per-path containment evidence kept per hold (LRU).
_EVIDENCE_PATHS_MAX = 256
_NOT_BOOTSTRAP_RECORDS = ("admission", "unbound-owner", "projection-dependencies")
_QUALIFICATION_FILE = Path(__file__).with_name("native_qualification.json")


def _posture(path: Path) -> tuple | None:
    """Identity and permission posture, or None when the path is absent."""
    return _observe_stamps((path,), ())[0][0]


def _posture_stamp(info, security=None) -> tuple:
    stamp = (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        stat.S_IMODE(info.st_mode),
        info.st_uid,
    )
    # Exact fresh owner/DACL bytes fence native security changes without
    # invalidating posture on unrelated directory content writes.
    return (*stamp, security) if security is not None else stamp


def _content(path: Path) -> tuple | None:
    """Identity plus change stamps, or None when the path is absent."""
    try:
        info = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


_METADATA_NATIVE_MODULE = sys.modules.get("tldw_chatbook.Utils.windows_files")
_METADATA_CLOSE_ERROR = (
    vars(_METADATA_NATIVE_MODULE).get("_WINDOWS_METADATA_CLOSE_ERROR_ORIGINAL")
    if _METADATA_NATIVE_MODULE is not None
    else None
)


def _metadata_close_uncertain(error):
    # Provenance is retained at the defining native module's completion.
    return type(error) is _METADATA_CLOSE_ERROR  # noqa: E721 - exact defining exception.


def _observe_stamps(posture_paths, content_paths, links=None) -> tuple:
    """Observe each named path once for one fresh evidence snapshot.

    Posture and content of overlapping paths come from the same native open.
    No filesystem observation survives this call. When ``links`` is a list, the
    link count of every present content path is appended to it.
    """
    paths = tuple(dict.fromkeys((*posture_paths, *content_paths)))
    if os.name == "nt":
        observations = os.stat_many_for_admission(paths)
    else:
        observations = {}
        for path in paths:
            try:
                observations[path] = (os.stat(path, follow_symlinks=False), None)
            except FileNotFoundError:
                observations[path] = None
    posture = tuple(
        None if (observed := observations[path]) is None else _posture_stamp(*observed)
        for path in posture_paths
    )
    content = tuple(
        None
        if (observed := observations[path]) is None
        else (
            observed[0].st_dev,
            observed[0].st_ino,
            observed[0].st_size,
            observed[0].st_mtime_ns,
            observed[0].st_ctime_ns,
        )
        for path in content_paths
    )
    if links is not None:
        links.extend(
            observations[path][0].st_nlink
            for path in content_paths
            if observations[path] is not None
        )
    return posture, content


def _chain(path: Path) -> tuple[Path, ...]:
    """Every component from the filesystem root down to ``path`` itself."""
    return (*reversed(path.parents), path)


class _Evidence:
    """Stamps over one derivation's inputs; immutable once published on a hold."""

    __slots__ = ("names", "posture", "content", "epoch", "confirmed")

    def __init__(self, names, posture_paths, content_paths):
        self.names = names
        posture_paths, content_paths = tuple(posture_paths), tuple(content_paths)
        posture, content = _observe_stamps(posture_paths, content_paths)
        self.posture = tuple(zip(posture_paths, posture, strict=True))
        self.content = tuple(zip(content_paths, content, strict=True))
        self.epoch = bootstrap._admission_epoch
        self.confirmed = False

    def dependencies(self) -> tuple:
        return (
            tuple(p for p, _ in self.posture),
            tuple(p for p, _ in self.content),
        )

    def stamps(self) -> tuple:
        return (
            tuple(s for _, s in self.posture),
            tuple(s for _, s in self.content),
        )

    def observe(self) -> tuple:
        return _observe_stamps(
            tuple(p for p, _ in self.posture),
            tuple(p for p, _ in self.content),
        )

    def settled_before(self, when_ns: int) -> bool:
        return all(
            s is None or s[4] <= when_ns - _EVIDENCE_SETTLE_NS for _, s in self.content
        )


def _observe_evidence(entries, links=None) -> tuple:
    """Observe all derivation dependencies once within one fresh snapshot."""
    posture_paths = tuple(
        dict.fromkeys(p for entry in entries for p, _ in entry.posture)
    )
    content_paths = tuple(
        dict.fromkeys(p for entry in entries for p, _ in entry.content)
    )
    posture, content = _observe_stamps(posture_paths, content_paths, links)
    posture_by_path = dict(zip(posture_paths, posture, strict=True))
    content_by_path = dict(zip(content_paths, content, strict=True))
    return tuple(
        (
            tuple(posture_by_path[p] for p, _ in entry.posture),
            tuple(content_by_path[p] for p, _ in entry.content),
        )
        for entry in entries
    )


# ---------------------------------------------------------------------------
# TASK-34601 amendment to ADR-126: change-notified evidence reuse (Windows).
#
# A warm reuse may skip the per-call full re-observation (and the per-call
# qualified_for walk) only while ALL hold: overlapped change notifications on
# every directory of the evidence tree were armed BEFORE the full observation
# that last confirmed these exact evidence objects, and are still quiet when
# checked after the lease is counted; the drive root, which no parent directory
# can watch, is re-stamped directly; every content file had a single link at
# confirmation; no by-id write was made through this process's facade since
# (such writes notify no directory); and that confirmation is younger than the
# backstop. Anything else runs the original full observation -- still the only
# source of refusals -- re-arming first so a change during it is never lost, and
# a full observation that finds any change un-verifies every watch of the hold.
# No watch is armed or used during a local pause; all are released when a pause
# begins or their hold retires.
# ---------------------------------------------------------------------------

#: Kill switch. POSIX observation is already cheap, so Windows only.
_EVIDENCE_WATCH = os.name == "nt"
#: A full observation re-confirms watched evidence at least this often. This also
#: bounds changes no watched directory reports: writes through a handle opened by
#: file id in another process, data written through a still-open handle, a hard
#: link created in another directory, a volume turning read-only.
_EVIDENCE_WATCH_BACKSTOP_S = 0.5
#: Evidence spanning more directories than this keeps per-call observation.
_EVIDENCE_WATCH_MAX_DIRECTORIES = 128
#: Watched evidence tuples kept per hold (a bound hold has one per admitted path).
_EVIDENCE_WATCH_SLOTS = 8


def _native_generation() -> int:
    if os.name != "nt":
        return 0
    from tldw_chatbook.Utils.windows_files import native_mutation_generation

    return native_mutation_generation()


class _EvidenceWatch:
    """Notifications over one exact tuple of confirmed evidence objects.

    ``watch`` is None when arming failed for these entries; arming is not
    retried until the evidence itself is replaced. ``verified_at``,
    ``generation``, ``users`` and ``retired`` are guarded by ``_lock``.
    """

    __slots__ = ("entries", "watch", "verified_at", "generation", "users", "retired")

    def __init__(self, entries, watch):
        self.entries = entries
        self.watch = watch
        self.verified_at = None
        self.generation = None
        self.users = 0
        self.retired = False

    def covers(self, entries) -> bool:
        return len(entries) == len(self.entries) and all(
            mine is theirs for mine, theirs in zip(self.entries, entries)
        )


def _watch_key(entries) -> tuple:
    # A watch holds strong references to its entries, so their ids stay unique.
    return tuple(id(entry) for entry in entries)


def _watch_directories(entries) -> dict | None:
    """``{directory: notify filter}`` for every directory of the evidence tree.

    Posture stamps hold no timestamps, so a directory whose children are only
    posture paths watches names, attributes and security; a parent of any
    content path also watches size and write/creation times. None when a content
    path lies on a drive whose root is not re-stamped as posture.
    """
    from tldw_chatbook.Utils.windows_files import WATCH_CONTENT, WATCH_POSTURE

    content = {p for entry in entries for p, _ in entry.content}
    anchors = {p for entry in entries for p, _ in entry.posture if p.parent == p}
    if any(Path(path.anchor) not in anchors for path in content):
        return None
    nodes = {
        node
        for entry in entries
        for path in (*(p for p, _ in entry.posture), *content)
        for node in (*path.parents, path)
    }
    directories = {}
    for node in nodes:
        if node.parent != node:
            notify = WATCH_CONTENT if node in content else WATCH_POSTURE
            directories[node.parent] = directories.get(node.parent, 0) | notify
    return directories


def _arm_watch(entries) -> _EvidenceWatch:
    """Arm notifications for ``entries``; failure yields a ``watch=None`` marker."""
    from tldw_chatbook.Utils.windows_files import DirectoryWatch

    directories = _watch_directories(entries)
    if directories is None or len(directories) > _EVIDENCE_WATCH_MAX_DIRECTORIES:
        return _EvidenceWatch(entries, None)
    try:
        return _EvidenceWatch(entries, DirectoryWatch(directories))
    except (OSError, ValueError, RuntimeError) as error:
        if _metadata_close_uncertain(error):
            raise
        return _EvidenceWatch(entries, None)


def _retire(watch):
    """Mark one detached watch retired (caller holds ``_lock``); closable now?"""
    watch.retired = True
    return watch if watch.users == 0 else None


def _retire_watches(hold) -> list:
    """Detach every watch of ``hold`` (caller holds ``_lock``); the closable ones."""
    watches, hold.watches = getattr(hold, "watches", None) or {}, OrderedDict()
    return [watch for watch in map(_retire, watches.values()) if watch is not None]


def _close_retired_watches(watches) -> None:
    """Close detached watches outside the coordinator lock."""
    for watch in watches:
        if watch is not None and watch.watch is not None:
            watch.watch.close()


def _release_watch(watch) -> None:
    with _lock:
        watch.users -= 1
        closable = watch.retired and watch.users == 0
    if closable:
        _close_retired_watches((watch,))


def _invalidate_watches(hold) -> None:
    """A full observation found a change: no watch of this hold stays verified."""
    with _lock:
        for watch in (getattr(hold, "watches", None) or {}).values():
            watch.verified_at = None


def _watch_current(watch, entries, now, generation) -> bool:
    """Check a claimed watch's in-memory validity; caller holds ``_lock``."""
    return (
        _pause is None
        and watch is not None
        and not watch.retired
        and watch.watch is not None
        and watch.verified_at is not None
        and now - watch.verified_at < _EVIDENCE_WATCH_BACKSTOP_S
        and watch.generation == generation
        and watch.covers(entries)
    )


def _quiet_watch(hold, entries):
    """Claim the hold's verified watch for ``entries`` within the backstop, or None."""
    if not _EVIDENCE_WATCH:
        return None
    now, generation, key = time.monotonic(), _native_generation(), _watch_key(entries)
    with _lock:
        watches = getattr(hold, "watches", None) or {}
        watch = watches.get(key)
        if not _watch_current(watch, entries, now, generation):
            return None
        watch.users += 1
        watches.move_to_end(key)
        return watch


def _anchors_unchanged(entries) -> bool:
    """Re-stamp every drive root directly; no parent directory can watch it."""
    expected = {}
    for entry in entries:
        for path, stamp in entry.posture:
            if path.parent == path:
                expected[path] = stamp
    if not expected:
        return True
    anchors = tuple(expected)
    posture, _ = _observe_stamps(anchors, ())
    return posture == tuple(expected[path] for path in anchors)


def _observe_watched(hold, entries) -> bool:
    """The original full observation, (re)arming notifications BEFORE it.

    Returns whether every stamp is unchanged. A watch becomes verified only when
    the stamps are unchanged, every content file has a single link, no by-id
    write happened in this process during the observation, and the watch armed
    before the observation is still quiet after it. Any change un-verifies every
    watch of the hold.
    """
    key = _watch_key(entries)
    candidate = fresh = None
    if _EVIDENCE_WATCH:
        arm = False
        with _lock:
            current = (getattr(hold, "watches", None) or {}).get(key)
            if _pause is None:
                if (
                    current is not None
                    and not current.retired
                    and current.covers(entries)
                ):
                    if current.watch is not None:
                        current.users += 1
                        candidate = current
                else:
                    arm = True
        if candidate is not None and not candidate.watch.quiet():
            _release_watch(candidate)
            candidate, arm = None, True
        if arm:
            fresh = _arm_watch(entries)
    generation = _native_generation()
    links = []
    started = time.monotonic()
    try:
        unchanged = _observe_evidence(entries, links) == tuple(
            entry.stamps() for entry in entries
        )
        armed = candidate if candidate is not None else fresh
        verified = (
            unchanged
            and all(count == 1 for count in links)
            and armed is not None
            and armed.watch is not None
            and armed.watch.quiet()
            and _native_generation() == generation
        )
    except BaseException:
        if fresh is not None and fresh.watch is not None:
            fresh.watch.close()
        if candidate is not None:
            _release_watch(candidate)
        raise
    retired = []
    with _lock:
        watches = getattr(hold, "watches", None)
        if watches is not None and not unchanged:
            for watch in watches.values():
                watch.verified_at = None
        if candidate is not None:
            if (
                verified
                and watches is not None
                and watches.get(key) is candidate
                and not candidate.retired
            ):
                candidate.verified_at, candidate.generation = started, generation
        elif fresh is not None and (
            watches is not None
            and _pause is None
            and _holds.get(hold.key) is hold
            and _hold_serving(hold)
        ):
            previous = watches.pop(key, None)
            if previous is not None:
                retired.append(_retire(previous))
            watches[key] = fresh
            if verified:
                fresh.verified_at, fresh.generation = started, generation
            while len(watches) > _EVIDENCE_WATCH_SLOTS:
                retired.append(_retire(watches.popitem(last=False)[1]))
            fresh = None
    _close_retired_watches(retired)
    if fresh is not None and fresh.watch is not None:
        fresh.watch.close()  # never installed
    if candidate is not None:
        _release_watch(candidate)
    return unchanged


def _no_links(evidence: _Evidence) -> bool:
    """Evidence is never kept over a symlink component (or an absent ancestor)."""
    last = len(evidence.posture) - 1
    return all(
        (s is not None and not stat.S_ISLNK(s[2])) or (s is None and i == last)
        for i, (_, s) in enumerate(evidence.posture)
    )


def _selector_evidence(root, selector, names, roots) -> _Evidence | None:
    """Stamp every input the selector-level derivation read, or None if unsafe."""
    try:
        entries = os.listdir(root)
    except OSError:
        return None
    if any(n.startswith(("pending-", "activation-update-")) for n in entries):
        return None
    from .control_records import _creation_name

    # Full authority derivation inspects each exact ancestor creation intent.
    # Their absence must remain observable when reusing a completed decision;
    # a newly unfinished entry must return to the original strict initializer.
    creation_intents = {
        child.parent / _creation_name(child)
        for child in _chain(root)
        if child.parent != child
    }
    admission = root / "admission"
    content = [
        root,
        admission,
        admission / "registry.json",
        admission / "registry.lock",
        root / "unbound-owner",
        _QUALIFICATION_FILE,
        *sorted(creation_intents),
        *sorted(root / n for n in entries if n not in _NOT_BOOTSTRAP_RECORDS),
    ]
    # The selector is an input even when unbound: a profile whose fingerprint
    # stopped matching binds again once the selector is restored.
    walked = [root, admission, selector.parent]
    if names != (UNBOUND_NAMESPACE,):
        if roots is None or not all(os.path.lexists(r) for r in roots):
            return None  # absence-proved roots are never reused
        walked.extend(roots)
    posture = sorted({p for target in walked for p in _chain(Path(target))})
    if os.name == "nt":
        posture = sorted({*posture, Path(_QUALIFICATION_FILE.anchor)})
    try:
        evidence = _Evidence(names, posture, (*content, selector))
    except (OSError, ValueError) as error:
        if _metadata_close_uncertain(error):
            raise
        return None  # Native unobservable paths never become admission evidence.
    posture_ok = all(
        s is not None and not stat.S_ISLNK(s[2]) for _, s in evidence.posture
    )
    # An absent selector is observed as absent; its appearance is a mismatch.
    content_ok = all(
        s is None if p in creation_intents else p == selector or s is not None
        for p, s in evidence.content
    )
    return evidence if posture_ok and content_ok else None


def _path_evidence(names, selected: Path) -> _Evidence | None:
    """Stamp the admitted path's chain (the containment check's only input)."""
    try:
        evidence = _Evidence(names, _chain(selected), ())
    except (OSError, ValueError) as error:
        if _metadata_close_uncertain(error):
            raise
        return None
    return evidence if _no_links(evidence) else None


def _hold_serving(hold) -> bool:
    return (
        hold is not None
        and hold.ready.is_set()
        and hold.error is None
        and not hold.stop.is_set()
    )


def _mount_read_only(path: Path) -> bool:
    """Qualification refuses a read-only mount; the reuse path must too."""
    if os.name == "nt":
        # Qualification reads current local-NTFS identity and read-only flags.
        # The Windows facade has no POSIX statvfs; absence is not a mount verdict.
        return not qualified_for("admission", path)[0]
    statvfs = getattr(os, "statvfs", None)
    if statvfs is None:
        return True
    try:
        return bool(statvfs(path).f_flag & getattr(os, "ST_RDONLY", 1))
    except OSError:
        return True


def _selected_paths(path, related_paths) -> tuple[Path, ...]:
    return tuple(
        selected
        for selected in (
            (lexical_path(path) if path is not None else None),
            *related_paths,
        )
        if selected is not None
    )


def _ordinary_hold(authority=None, lease=None):
    """Select only an actual serving ordinary owner; never create authority."""
    with _lock:
        if (
            not _EVIDENCE_REUSE
            or os.name == "nt"
            or _pause is not None
            or getattr(_local, "admitted", False)
        ):
            return None
        if lease is not None:
            if lease not in _live_leases:
                return None
            hold = _holds.get(lease._key)
        else:
            hold = next((h for h in _holds.values() if h.authority is authority), None)
        return hold if _hold_serving(hold) and hold.key[0] == os.getpid() else None


def _derived_before(hold, key):
    """Observe existing complete inputs before their next original derivation."""
    epoch = bootstrap._admission_epoch
    with _lock:
        item = hold.derived_evidence.get(key) if hold is not None else None
    if item is None:
        return epoch, None
    entry, _ = item
    try:
        return epoch, (entry, entry.observe(), time.time_ns())
    except OSError:
        return epoch, None


def _derived_reuse(hold, key, before):
    """Borrow a defensive copy only after current full stamps and owner checks."""
    epoch, observed = before
    if hold is None or observed is None or _mount_read_only(Path(hold.key[1]).parent):
        return False, None
    entry, stamps, _ = observed
    with _lock:
        item = hold.derived_evidence.get(key)
        if (
            item is None
            or item[0] is not entry
            or not entry.confirmed
            or entry.names != hold.names
            or stamps != entry.stamps()
            or epoch != entry.epoch
            or epoch != bootstrap._admission_epoch
            or _ordinary_hold(hold.authority) is not hold
        ):
            return False, None
        hold.derived_evidence.move_to_end(key)
        return True, copy.deepcopy(item[1])


def _note_derived(hold, key, entry, value, before):
    """Keep existing evidence bounded and confirm two bracketed positive reads."""
    if hold is None:
        return
    epoch, observed = before
    with _lock:
        if _ordinary_hold(hold.authority) is not hold:
            return
        if entry is None:
            hold.derived_evidence.pop(key, None)
            return
        current = hold.derived_evidence.get(key)
        if (
            current is not None
            and current[0].confirmed
            and current[0].epoch == entry.epoch
            and current[0].names == entry.names
            and current[0].dependencies() == entry.dependencies()
            and current[0].stamps() == entry.stamps()
        ):
            # A concurrent cold reader cannot demote already-confirmed inputs.
            hold.derived_evidence.move_to_end(key)
            return
        previous, stamps, when = observed or (None, None, 0)
        entry.confirmed = (
            previous is not None
            and previous.names == entry.names
            and previous.dependencies() == entry.dependencies()
            and previous.stamps() == stamps == entry.stamps()
            and epoch == entry.epoch == bootstrap._admission_epoch
            and entry.settled_before(when)
        )
        hold.derived_evidence[key] = entry, copy.deepcopy(value)
        hold.derived_evidence.move_to_end(key)
        while len(hold.derived_evidence) > _EVIDENCE_PATHS_MAX:
            hold.derived_evidence.popitem(last=False)


def _metadata_evidence(hold, paths=(), *, registry=None, records=None):
    """Complete ordinary metadata inputs, including foreign and historical roots.

    Unknown/absent/aliased dependencies deliberately keep the original path fresh.
    This collector never proves admission: callers supply a positive derivation.
    """
    if hold is None:
        return None
    root, selector = Path(hold.key[1]), effective_config_path()
    try:
        registry = bootstrap._registry(root) if registry is None else registry
        pending, profiles, associations = (
            bootstrap._control_records(root) if records is None else records
        )
        if (
            pending
            or registry is None
            or any(
                row.get("pending") or row.get("proposed") for row in registry.values()
            )
        ):
            return None
        roots = {Path(p) for row in registry.values() for p in row["roots"]}
        roots.update(
            Path(p) for row in profiles + associations for p in row.get("roots", ())
        )
        historical = {
            Path(token[5:])
            for row in registry.values()
            for token in row.get("historical", ())
            if token.startswith("path:")
        }
        # Resolution is consulted by source-scope admission even with unchanged
        # registry bytes. Complete no-link chains, or no reusable evidence.
        walked = roots | historical | {Path(p) for p in paths}
        content = []
        for row in profiles + associations:
            witness = row.get("activation")
            if witness:
                from .activation import ActivationStore

                store = ActivationStore(Path(witness["store_root"]))
                generation = store._generation(witness["generation"])
                walked.update((store.root, generation))
                content.append(generation / "required.json")
        if not all(os.path.lexists(p) for p in roots | historical):
            return None
        base = _selector_evidence(root, selector, hold.names, tuple(roots))
        if base is None:
            return None
        posture = sorted(
            {p for target in walked for p in _chain(target)}
            | {p for p, _ in base.posture}
        )
        evidence = _Evidence(
            hold.names, posture, tuple(p for p, _ in base.content) + tuple(content)
        )
        # A selected source can have an absent leaf, but no missing ancestor or
        # historical chain may be silently omitted from the complete inputs.
        optional = {Path(p) for p in paths}
        if any(
            s is None and p not in optional or s is not None and stat.S_ISLNK(s[2])
            for p, s in evidence.posture
        ):
            return None
        # Selector evidence has already validated these exact absent ancestor
        # creation intents. Preserve their absence through this second snapshot:
        # appearance here refuses reuse, and later appearance changes the stamp.
        absent_intents = {
            dependency
            for dependency, stamp in base.content
            if stamp is None and dependency != selector
        }
        if any(
            stamp is not None if dependency in absent_intents else stamp is None
            for dependency, stamp in evidence.content
            if dependency != selector
        ):
            return None
        return evidence
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _reuse_evidence(
    root, selector, path, related_paths, check, execution_selection, check_state
):
    """Return a lease from confirmed evidence, or None to run the derivation.

    Like the derivation, the lease is counted before the final revalidation, so a
    drain that starts meanwhile sees it. Every stamp and the admission epoch are
    then observed again; any difference closes the lease and falls back.
    """
    key = (os.getpid(), str(root))
    selected = _selected_paths(path, related_paths)
    with _lock:
        hold = _holds.get(key)
        if not _hold_serving(hold):
            return None
        evidence = hold.evidence.get(str(selector))
        if (
            evidence is None
            or not evidence.confirmed
            or evidence.epoch != bootstrap._admission_epoch
            or evidence.names != hold.names
        ):
            return None
        epoch = evidence.epoch
        bound = evidence.names != (UNBOUND_NAMESPACE,)
        per_path = []
        retained_paths = []
        fresh_children = []
        for item in selected if bound else ():
            entry = hold.path_evidence.get((str(selector), str(item)))
            if (
                entry is None
                or not entry.confirmed
                or entry.epoch != bootstrap._admission_epoch
            ):
                # A new history temporary (or another fresh child) may borrow
                # ONLY a twice-derived directory containment proof. Its own
                # full current chain/leaf is observed again after counting.
                directory_key = ("directory", str(selector), str(item.parent))
                directory = hold.path_evidence.get(directory_key)
                if (
                    directory is None
                    or not directory.confirmed
                    or directory.epoch != bootstrap._admission_epoch
                ):
                    return None
                hold.path_evidence.move_to_end(directory_key)
                per_path.append(directory)
                retained_paths.append((directory_key, directory))
                fresh_children.append(item)
            else:
                path_key = (str(selector), str(item))
                hold.path_evidence.move_to_end(path_key)
                per_path.append(entry)
                retained_paths.append((path_key, entry))
    # Snapshot only under _lock. New child stamps, like final observations,
    # perform filesystem work outside the coordinator lock.
    for item in fresh_children:
        entry = _path_evidence(hold.names, item)
        if entry is None:
            return None
        per_path.append(entry)
    entries = (evidence, *per_path)
    # Fresh child stamps are new objects on every call, so evidence including
    # them is never watched; it keeps the original per-call observation.
    watched = None if fresh_children else _quiet_watch(hold, entries)
    try:
        # A quiet verified watch covers the qualification inputs as well; the
        # backstop bounds a volume turning read-only (TASK-34601 amendment).
        if watched is None and _mount_read_only(root.parent):
            return None
        proof = check()
        with _lock:
            check_state(proof)
        if getattr(_local, "admitted", False):
            raise bootstrap.RecoveryRequired("maintenance_requires_owner_capability")
        proof = check()
        with _lock:
            check_state(proof)
            if (
                _holds.get(key) is not hold
                or not _hold_serving(hold)
                or hold.evidence.get(str(selector)) is not evidence
                or evidence.epoch != bootstrap._admission_epoch
            ):
                return None
            hold.count += 1
            token = StorageLease(key)
            token._execution_selection = execution_selection
        # Native filesystem observation never runs under the coordinator lock.
        # The notification check replaces the final re-observation at the same
        # point: after the lease is counted, before the final owner checks.
        try:
            # Recheck notifications and in-memory validity after the native
            # root read: a by-id write, expiry or invalidation can race that read
            # without signaling the watch. This adds no native observation.
            observed = False
            if (
                watched is not None
                and _anchors_unchanged(entries)
                and watched.watch.quiet()
            ):
                with _lock:
                    observed = _watch_current(
                        watched, entries, time.monotonic(), _native_generation()
                    )
            if not observed:
                if watched is not None:
                    observed = not _mount_read_only(root.parent) and _observe_watched(
                        hold, entries
                    )
                elif fresh_children:
                    observed = _observe_evidence(entries) == tuple(
                        entry.stamps() for entry in entries
                    )
                    if not observed:
                        _invalidate_watches(hold)  # shared selector evidence moved
                else:
                    observed = _observe_watched(hold, entries)
            unchanged = observed and evidence.epoch == bootstrap._admission_epoch
            proof = check()
            with _lock:
                check_state(proof)
                unchanged = (
                    unchanged
                    and token in _live_leases
                    and token._key == key
                    and _holds.get(key) is hold
                    and _hold_serving(hold)
                    and hold.evidence.get(str(selector)) is evidence
                    and evidence.epoch == epoch == bootstrap._admission_epoch
                    and evidence.names == hold.names
                    and all(
                        hold.path_evidence.get(path_key) is entry
                        for path_key, entry in retained_paths
                    )
                )
        except (OSError, ValueError) as error:
            token.close()
            if _metadata_close_uncertain(error):
                raise
            return None  # Only the unchanged full derivation decides refusal codes.
        except BaseException:
            token.close()  # as the derivation does: a counted lease never leaks
            raise
        if not unchanged:
            token.close()
            return None
        return token
    finally:
        if watched is not None:
            _release_watch(watched)


def _observe_candidates(root, selector, path, related_paths):
    """Before a derivation, re-observe the evidence it may confirm."""
    key = (os.getpid(), str(root))
    epoch = bootstrap._admission_epoch
    with _lock:
        hold = _holds.get(key)
        if hold is None:
            return epoch, {}
        wanted = {str(selector): hold.evidence.get(str(selector))}
        for item in _selected_paths(path, related_paths):
            wanted[(str(selector), str(item))] = hold.path_evidence.get(
                (str(selector), str(item))
            )
            directory_key = ("directory", str(selector), str(item.parent))
            wanted[directory_key] = hold.path_evidence.get(directory_key)
    now = time.time_ns()
    observations = {}
    for name, entry in wanted.items():
        if entry is not None:
            try:
                observations[name] = (entry, entry.observe(), now)
            except (OSError, ValueError) as error:
                if _metadata_close_uncertain(error):
                    raise
                continue  # A candidate that cannot be observed cannot confirm.
    return epoch, observations


def _note_evidence(hold, root, selector, path, related_paths, names, roots, before):
    """Publish post-derivation stamps; confirm them if they bracket the derivation."""
    epoch_before, observations = before
    fresh = {str(selector): _selector_evidence(root, selector, names, roots)}
    if names != (UNBOUND_NAMESPACE,):
        items = _selected_paths(path, related_paths)
        for item in items:
            fresh[(str(selector), str(item))] = _path_evidence(names, item)
        if roots is not None:
            for parent in {item.parent for item in items}:
                # This is a separate original containment derivation for the
                # directory itself, never inference from a file-only root.
                if _contains_capture_path((*roots, selector), parent):
                    fresh[("directory", str(selector), str(parent))] = _path_evidence(
                        names, parent
                    )
    with _lock:
        if _holds.get(hold.key) is not hold or hold.names != names:
            return
        for name, entry in fresh.items():
            store = hold.evidence if isinstance(name, str) else hold.path_evidence
            if entry is None:
                store.pop(name, None)
                continue
            current = store.get(name)
            if (
                current is not None
                and current.confirmed
                and current.epoch == entry.epoch
                and current.names == entry.names
                and current.dependencies() == entry.dependencies()
                and current.stamps() == entry.stamps()
            ):
                # Concurrent derivations over unchanged inputs must not replace
                # confirmed evidence with a fresh, unconfirmed copy of itself.
                if store is hold.path_evidence:
                    store.move_to_end(name)
                continue
            # Sound when this derivation was bracketed by identical stamps over
            # the same inputs: observed before it, and observed again after.
            previous, observed, observed_at = observations.get(name, (None, None, 0))
            entry.confirmed = (
                bootstrap._admission_epoch == epoch_before == entry.epoch
                and previous is not None
                and previous.names == entry.names
                and previous.dependencies() == entry.dependencies()
                and previous.stamps() == observed == entry.stamps()
                and entry.settled_before(observed_at)
            )
            store[name] = entry
            if store is hold.path_evidence:
                store.move_to_end(name)
                while len(store) > _EVIDENCE_PATHS_MAX:
                    store.popitem(last=False)


class _ScopeProof:
    """One call's synchronized source dependencies, never a permission verdict."""

    def __init__(self, root, selector, attempt, authority):
        self.root = root
        self.selector = selector
        self.key = (os.getpid(), str(root))
        self.thread = threading.current_thread()
        self.task = _task_identity()
        self.attempt = attempt
        self.authority = authority
        self.continuation = False
        with _lock:
            self.hold = _holds.get(self.key)
            self.names = self.hold.names if self.hold is not None else None
            self.pause = (
                attempt.pause if type(attempt) is _StartupReacquisition else None
            )
            self.startup_source = (
                self.pause._startup_source if self.pause is not None else None
            )
            self.startup_roots = (
                self.pause._startup_roots if self.pause is not None else None
            )
            self.check()

    def check(self):
        """Fence only current actor, lexical selection and captured live metadata."""
        if (
            self.key != (os.getpid(), str(bootstrap.default_bootstrap_root()))
            or self.selector != effective_config_path()
            or self.thread is not threading.current_thread()
            or self.task is not _task_identity()
        ):
            raise bootstrap.RecoveryRequired("execution_selection_changed")
        if self.pause is not None:
            _StartupReacquisition.check(self.attempt)
            if (
                self.attempt not in _pending_acquisitions
                or self.attempt.pause is not self.pause
                or self.pause._startup_source != self.startup_source
                or self.pause._startup_roots != self.startup_roots
                or self.authority is None
                or self.authority._identity != self.startup_source[3]
            ):
                raise bootstrap.RecoveryRequired("startup_scope_changed")
        if self.continuation and (
            _holds.get(self.key) is not self.hold
            or self.hold is None
            or self.hold.names != self.names
            or self.hold.count <= 0
            or self.hold.stop.is_set()
        ):
            raise bootstrap.RecoveryRequired("storage_scope_changed")


def _check_acquisition_state(attempt, operation, proofs, paths, execution_selection):
    """Repeat pure issued-state checks after an original fresh path proof."""
    if (
        attempt not in _pending_acquisitions
        or attempt.pid != os.getpid()
        or attempt.thread is not threading.current_thread()
        or attempt.task is not _task_identity()
        or attempt.operation is not operation
    ):
        raise bootstrap.RecoveryRequired("acquisition_provenance_invalid")
    if attempt.cancel.is_set():
        raise bootstrap.RecoveryRequired("storage_locally_paused")
    if (
        execution_selection[0] != os.getpid()
        or execution_selection[1] != bootstrap.default_bootstrap_root()
        or execution_selection[2] != effective_config_path()
    ):
        raise bootstrap.RecoveryRequired("execution_selection_changed")
    if type(attempt) is _StartupReacquisition:
        _StartupReacquisition.check(attempt)
    elif operation is None:
        if _pause is not None:
            raise bootstrap.RecoveryRequired("storage_locally_paused")
    else:
        for selected, proof in zip(paths, proofs, strict=True):
            if proof is None:
                raise bootstrap.RecoveryRequired("operation_provenance_invalid")
            _check_operation_state(operation, proof, selected)


def _scope(
    root: Path,
    selector: Path,
    path: Path | None,
    *,
    startup_attempt=None,
    authority=None,
    related_paths: tuple[Path, ...] = (),
    proof: _ScopeProof | None = None,
) -> tuple[str, ...]:
    if proof is None:
        proof = _ScopeProof(root, selector, startup_attempt, authority)
    if (
        type(proof) is not _ScopeProof
        or proof.root != root
        or proof.selector != selector
        or proof.attempt is not startup_attempt
        or proof.authority is not authority
    ):
        raise bootstrap.RecoveryRequired("storage_scope_changed")
    pending, profiles = bootstrap._records(root)
    registry = bootstrap._registry(root)
    binding = bootstrap._binding(selector, profiles, registry) if profiles else None
    if type(startup_attempt) is _StartupReacquisition:
        with _lock:
            proof.check()
            _StartupReacquisition.check(startup_attempt, path)
        if proof.startup_roots is not None:
            previous = next(
                (r for r in profiles if r["selector"] == str(selector)), None
            )
            if (
                previous is None
                or tuple(previous["namespaces"]) != proof.startup_source[2]
                or tuple(previous["roots"]) != proof.startup_roots
            ):
                raise bootstrap.RecoveryRequired("startup_scope_changed")
            if binding is None and not pending:
                # Continue only the retired owner's unchanged mapping. This does
                # not update persisted enrollment or admit any ordinary caller.
                snapshot = dict(previous, fingerprint=bootstrap._fingerprint(selector))
                binding = bootstrap._binding(selector, [snapshot], registry)
    if binding is None and not pending:
        live = proof.hold
        previous = next((r for r in profiles if r["selector"] == str(selector)), None)
        if (
            live is not None
            and previous is not None
            and proof.names == tuple(previous["namespaces"])
        ):
            proof.continuation = True
            # Existing owners continue only inside their original verified mapping.
            # New process enrollment still requires the saved config fingerprint.
            snapshot = dict(previous, fingerprint=bootstrap._fingerprint(selector))
            binding = bootstrap._binding(selector, [snapshot], registry)
    if binding is None:
        if pending:
            raise bootstrap.RecoveryRequired("recovery_scope_uncertain")
        with _lock:
            proof.check()
        return (UNBOUND_NAMESPACE,)
    from .bootstrap import effective_roots

    roots = effective_roots(
        binding["roots"],
        (registry[name] for name in binding["namespaces"]),
    )
    if startup_attempt is not None:
        # The DECLARED roots, not the effective ones: effective_roots drops an
        # absence-proved alias, and that proof is no stamp -- the alias can
        # reappear (e.g. as a FIFO) without changing its parent's posture. An
        # absent declared root makes _selector_evidence refuse to record.
        startup_attempt.scope_roots = tuple(
            dict.fromkeys(Path(r) for r in binding["roots"])
        )
    for selected in (path, *related_paths):
        if selected is not None and not _contains_capture_path(
            (*roots, Path(binding["selector"])), selected
        ):
            raise bootstrap.RecoveryRequired("storage_scope_not_enrolled")
    with _lock:
        proof.check()
    return tuple(binding["namespaces"])


def acquire_storage(
    path: Path | None = None, *, related_paths: tuple[Path, ...] = ()
) -> StorageLease:
    """Admit every operation path under one native group, without creating files.

    Companion paths receive the same pre/post-lock scope checks as the primary.
    The returned lease's borrowable execution selection remains the primary path.
    """
    attempt = _Acquisition()
    try:
        return _acquire_storage(path, attempt, related_paths=related_paths)
    except bootstrap.RecoveryRequired:
        raise
    except (OSError, ValueError, RuntimeError) as error:
        if _metadata_close_uncertain(error):
            raise
        raise bootstrap.RecoveryRequired("storage_admission_unavailable") from None
    finally:
        attempt.close()


# Direct callable inputs for the installed raw-member batch only. These are
# source bindings, never permission or retained native admission evidence.
_RAW_MEMBER_ACQUIRE_BINDING = (
    acquire_storage,
    acquire_storage.__code__,
    acquire_storage.__globals__,
    acquire_storage.__defaults__,
    acquire_storage.__kwdefaults__,
    tuple((acquire_storage.__kwdefaults__ or {}).items()),
)


def _acquire_storage(
    path: Path | None, attempt: _Acquisition, *, related_paths: tuple[Path, ...] = ()
) -> StorageLease:
    """Check fixed evidence and hold declared scope before any owned open/write.

    An ordinary seam used inside maintenance refuses promptly; later capture owners
    need a separately validated capability, never a boolean bypass here.
    """
    if _forked_with_owners:
        raise bootstrap.RecoveryRequired("forked_owner_restart_required")
    root = bootstrap.default_bootstrap_root()
    selector = effective_config_path()
    related_paths = tuple(lexical_path(selected) for selected in related_paths)

    paths = (path, *related_paths)
    operation = attempt.operation

    def check():
        return tuple(attempt.check(selected) for selected in paths)

    execution_selection = _execution_selection_for(path)
    if execution_selection[1:3] != (root, selector):
        raise bootstrap.RecoveryRequired("execution_selection_changed")

    def check_state(proofs):
        _check_acquisition_state(attempt, operation, proofs, paths, execution_selection)

    proof = check()
    with _lock:
        check_state(proof)
        if (
            operation is not None
            and operation.key is not None
            and operation.key != (os.getpid(), str(root))
        ):
            raise bootstrap.RecoveryRequired("operation_native_scope_changed")
    before = None
    if _EVIDENCE_REUSE and type(attempt) is _Acquisition:
        reused = _reuse_evidence(
            root, selector, path, related_paths, check, execution_selection, check_state
        )
        if reused is not None:
            return reused
        before = _observe_candidates(root, selector, path, related_paths)
    with attempt.initializing(root, path):
        check()
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
            _scope(
                root,
                selector,
                lexical_path(path) if path is not None else None,
                related_paths=related_paths,
            )
            allowed, reason = bootstrap.startup_permission(selector, root)
            if not allowed:
                raise bootstrap.RecoveryRequired(reason)
            proof = check()
            with _lock:
                check_state(proof)
                token = StorageLease(None)
                token._execution_selection = execution_selection
                return token
        authority = admission_authority(root)
    scope_proof = _ScopeProof(root, selector, attempt, authority)
    names = _scope(
        root,
        selector,
        lexical_path(path) if path is not None else None,
        startup_attempt=attempt,
        authority=authority,
        related_paths=related_paths,
        proof=scope_proof,
    )
    proof = check()
    with _lock:
        check_state(proof)
        scope_proof.check()
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
    # Count pending acquisitions before the final fresh proof, so drain sees them.
    try:
        while not hold.ready.wait(0.01):
            proof = check()
            with _lock:
                check_state(proof)
        if hold.error is not None:
            raise bootstrap.RecoveryRequired("storage_admission_unavailable")
        allowed, reason = bootstrap.startup_permission(selector, root)
        if not allowed:
            raise bootstrap.RecoveryRequired(reason)
        scope_proof = _ScopeProof(root, selector, attempt, authority)
        final_names = _scope(
            root,
            selector,
            lexical_path(path) if path is not None else None,
            startup_attempt=attempt,
            authority=authority,
            related_paths=related_paths,
            proof=scope_proof,
        )
        proof = check()
        with _lock:
            check_state(proof)
            scope_proof.check()
            if (
                final_names != names
                or token not in _live_leases
                or token._key != key
                or _holds.get(key) is not hold
                or hold.names != names
                or not _hold_serving(hold)
            ):
                raise bootstrap.RecoveryRequired("storage_scope_changed")
        if before is not None:
            _note_evidence(
                hold,
                root,
                selector,
                path,
                related_paths,
                names,
                attempt.scope_roots,
                before,
            )
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
        self.staging_identity = os.stat(staging)
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
        current = os.stat(self.staging)
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

                info = os.stat(self.sqlite_directory, follow_symlinks=False)
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
        root = os.stat(self.sqlite_directory, follow_symlinks=False)
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
                    current_parent = os.stat(target_root, follow_symlinks=False)
                    current = os.stat(
                        target_root / (target.name + suffix), follow_symlinks=False
                    )
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
    discovery = getattr(_local, "discovery_scope", None)
    if discovery is not None:
        discovery.check()
        # Reuse selected images so read-only SQLite cannot mutate live SHM.
        # Other held owners remain discovery-only, outside capture authority.
        if scope is None or lexical_path(source) not in scope.sqlite_snapshots:
            return discovery.sqlite_target(owner_id, source)
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

    def __init__(self, session, *, private_sqlite=False, limits=None, byte_budget=None):
        self.session = session
        self.config_sources = getattr(session, "_config_capture_sources", ())
        self.resources = []
        self.active = True
        self.sqlite_copies = (
            _PreviewScope(limits, byte_budget, native_scope=self)
            if private_sqlite
            else None
        )

    def check(self):
        self.session._check()
        if not self.active or getattr(_local, "discovery_scope", None) is not self:
            raise bootstrap.RecoveryRequired("discovery_scope_inactive")

    def sqlite_target(self, owner_id, source):
        """Copy only an installed reader's original native-held source."""
        from tldw_chatbook.DB.private_sqlite import SQLITE_OWNER_REGISTRY

        self.check()
        if self.sqlite_copies is None:
            return None
        if not SQLITE_OWNER_REGISTRY[owner_id].recovery_capture_allowed:
            raise bootstrap.RecoveryRequired("discovery_read_only_required")
        if not _contains_capture_path(self.session._roots, lexical_path(source)):
            raise bootstrap.RecoveryRequired("capture_source_outside_scope")
        return self.sqlite_copies.sqlite_target(source)

    def retire(self):
        try:
            if self.active and self.sqlite_copies is not None:
                self.sqlite_copies.verify_sources()
        finally:
            self.active = False
            for resource in tuple(self.resources):
                resource.retire()
            if self.sqlite_copies is not None:
                self.sqlite_copies.retire()


class _PreviewScope:
    """Bounded read probes over private SQLite copies, without live admission."""

    def __init__(self, limits, byte_budget, *, native_scope=None):
        self.native_scope = native_scope
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
        ):
            raise bootstrap.RecoveryRequired("preview_scope_inactive")
        if self.native_scope is None:
            if getattr(_local, "preview_scope", None) is not self:
                raise bootstrap.RecoveryRequired("preview_scope_inactive")
        else:
            if (
                type(self.native_scope) is not _DiscoveryScope
                or self.native_scope.sqlite_copies is not self
            ):
                raise bootstrap.RecoveryRequired("discovery_scope_inactive")
            self.native_scope.check()

    def verify_sources(self):
        """Native-held discovery cannot reuse observations after a source change."""
        self.check()
        for source, (before, _) in self.snapshots.items():
            if self._source_state(source) != before:
                raise ValueError("preview_sqlite_changed")

    @staticmethod
    def _source_state(source):
        result = []
        for suffix in ("", "-wal", "-journal"):
            path = source.with_name(source.name + suffix)
            try:
                info = os.stat(path, follow_symlinks=False)
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
        if source in self.snapshots:
            previous, target = self.snapshots[source]
            if self.native_scope is not None and self._source_state(source) != previous:
                raise ValueError("preview_sqlite_changed")
            # Owners sharing this source inspect one verified image per preview.
            # Ordinary live commits cannot invalidate that private image; a
            # replaced or unsafe source path still cannot reuse its identity.
            with bootstrap.pinned_directory(source.parent) as parent:
                current = os.stat(source.name, dir_fd=parent, follow_symlinks=False)
                if (
                    not stat.S_ISREG(current.st_mode)
                    or current.st_nlink != 1
                    or current.st_uid != os.geteuid()
                    or (current.st_dev, current.st_ino) != previous[0][:2]
                ):
                    raise ValueError("preview_sqlite_changed")
            return target
        before = self._source_state(source)
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

    def _refresh_roots(self):
        """Use held directory authority for a now-absent redundant alias."""
        from .bootstrap import effective_roots

        self._check()
        roots = effective_roots(self._declared_roots, self._root_entries)
        all_roots = effective_roots(self._declared_all_roots, self._all_root_entries)
        if not roots or not all_roots:
            raise bootstrap.RecoveryRequired("maintenance_roots_required")
        self._roots, self._all_roots = roots, all_roots

    def _discover_capture_inventory(
        self, config_paths, selections, approved_scope, *, config_bindings=None
    ):
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
        root = self._control.parent
        bindings = _config_capture_bindings(root, selectors)
        if (config_bindings is not None and bindings != config_bindings) or any(
            not set(row["namespaces"]) <= set(self._names) for row, _ in bindings
        ):
            raise bootstrap.RecoveryRequired("capture_config_scope_changed")
        self._config_capture_sources = _config_capture_sources(
            root, selectors, bindings
        )
        if any(not _contains_capture_path(self._roots, path) for path in selectors):
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
            if not _contains_capture_path(
                self._roots, path
            ) and not _config_capture_item(item, current, self._config_capture_sources):
                raise bootstrap.RecoveryRequired("capture_source_outside_scope")
            info = os.stat(path)
            if not stat.S_ISREG(info.st_mode):
                raise bootstrap.RecoveryRequired("capture_file_not_regular")
            sources.append((path.resolve(strict=True), info.st_dev, info.st_ino))
        self._discovered_sources = tuple(sources)
        return current

    @contextmanager
    def _discovery_reads(self, *, private_sqlite=False, limits=None, byte_budget=None):
        """Allow installed reads, optionally using bounded private SQLite copies."""
        from .limits import ArchiveLimits

        self._check()
        if getattr(_local, "discovery_scope", None) is not None:
            raise bootstrap.RecoveryRequired("nested_discovery_scope")
        if type(private_sqlite) is not bool:
            raise TypeError("invalid_discovery_copy_options")
        if private_sqlite:
            limits = ArchiveLimits() if limits is None else limits
            if type(limits) is not ArchiveLimits:
                raise TypeError("invalid_preview_limits")
            byte_budget = limits.expanded_bytes if byte_budget is None else byte_budget
            if (
                type(byte_budget) is not int
                or not 0 < byte_budget <= limits.expanded_bytes
            ):
                raise ValueError("invalid_preview_budget")
        elif limits is not None or byte_budget is not None:
            raise ValueError("invalid_discovery_copy_options")
        scope = _DiscoveryScope(
            self, private_sqlite=private_sqlite, limits=limits, byte_budget=byte_budget
        )
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
            info = os.stat(source)
            discovered = (
                source.resolve(strict=True),
                info.st_dev,
                info.st_ino,
            ) in getattr(self, "_discovered_sources", ())
            if not stat.S_ISREG(info.st_mode) or (
                not discovered and not _contains_capture_path(self._roots, source)
            ):
                raise bootstrap.RecoveryRequired("capture_source_outside_scope")
            from .bootstrap import effective_roots

            bound = _contains_capture_path(
                (
                    owned
                    for binding in bindings
                    for owned in effective_roots(
                        binding["roots"],
                        (registry[name] for name in binding["namespaces"]),
                    )
                ),
                source,
            )
            if not bound and not discovered:
                raise bootstrap.RecoveryRequired("capture_source_binding_unverified")
            selected.append((source.resolve(strict=True), info.st_dev, info.st_ino))
        with self._capture_bound_sources(tuple(selected), staging, limits, byte_budget):
            yield

    @contextmanager
    def _replacement_capture_scope(
        self, plan, journal, staging, *, limits, byte_budget
    ):
        """Bind only rechecked prepared local originals, including damaged config."""
        from .limits import ArchiveLimits
        from .replacement import _checked_originals

        self._check()
        if (
            type(limits) is not ArchiveLimits
            or type(byte_budget) is not int
            or not 0 < byte_budget <= limits.expanded_bytes
        ):
            raise ValueError("invalid_capture_budget")
        if getattr(_local, "capture_scope", None) is not None:
            raise bootstrap.RecoveryRequired("nested_capture_scope")
        _, inventory = _checked_originals(plan, journal, self)
        selected = []
        for item in inventory.items:
            if item.path is not None and item.path.is_file():
                info = os.stat(item.path)
                selected.append(
                    (item.path.resolve(strict=True), info.st_dev, info.st_ino)
                )
        if not selected:
            raise bootstrap.RecoveryRequired("capture_sources_required")
        with self._capture_bound_sources(tuple(selected), staging, limits, byte_budget):
            yield

    @contextmanager
    def _capture_bound_sources(self, selected, staging, limits, byte_budget):
        staging = lexical_path(staging)
        with bootstrap.pinned_directory(staging) as fd:
            info = os.fstat(fd)
            if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise bootstrap.RecoveryRequired("capture_staging_not_private")
        staging = staging.resolve(strict=True)
        # Recovery can move these names while reading private credential material
        # under the journal lock. Preserve the admitted canonical source names
        # for this overlap check; source access still requires its own checked scope.
        protected = {
            *(self._recovery_roots or ()),
            *(path for path, _ in self._publication_roots),
        }
        for root in self._all_roots + (self._control,):
            protected.add(root.resolve(strict=self._recovery_roots is None))
        for root in protected:
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


def _mint_maintenance_session(
    roots,
    all_roots,
    control,
    names,
    control_identity,
    *,
    root_entries=(),
    all_root_entries=(),
    recovery_roots=None,
    publication_roots=(),
):
    session = object.__new__(MaintenanceSession)
    session._declared_roots = tuple(roots)
    session._declared_all_roots = tuple(all_roots)
    session._root_entries = tuple(root_entries)
    session._all_root_entries = tuple(all_root_entries)
    session._control = control
    session._names = names
    session._control_identity = control_identity
    session._recovery_roots = recovery_roots
    session._publication_roots = publication_roots
    session._pid = os.getpid()
    session._thread = threading.get_ident()
    session._active = True
    session._scopes = []
    _local.maintenance_session = session
    try:
        session._refresh_roots()
    except BaseException:
        session._active = False
        _local.maintenance_session = None
        raise
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
    discovery = getattr(_local, "discovery_scope", None)
    if scope is None or discovery is not None:
        # Final discovery is read-only within the held native roots, which can
        # include owners deliberately omitted from the narrower payload scope.
        scope = discovery or getattr(_local, "preview_scope", None)
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
        if type(scope) is _DiscoveryScope and not _contains_capture_path(
            scope.session._roots, selected
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
            info = os.stat(selected)
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
        info = os.stat(selected)
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


def _check_capture_file_identity(
    scope, selected, info, *, source_only=False, owner_id=None
):
    scope.check()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise bootstrap.RecoveryRequired("capture_file_not_regular")
    if type(scope) is _DiscoveryScope:
        if not _contains_capture_path(
            scope.session._roots, selected
        ) and not _config_capture_file(scope.config_sources, selected, owner_id, info):
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
    "research.paste_staging": 64 * 1024,
    "collections.archives": 1,
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
        info = os.stat(selected, follow_symlinks=False)
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
                held_parent, current_parent = os.fstat(parent), os.stat(selected.parent)
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
    # Final rediscovery may run inside capture_scope. Its bounded reads use the
    # explicitly held discovery namespace, not the narrower payload source set.
    scope = (
        getattr(_local, "discovery_scope", None)
        or getattr(_local, "capture_scope", None)
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
                _check_capture_file_identity(scope, selected, info, owner_id=owner_id)
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
                _check_capture_file_identity(scope, selected, after, owner_id=owner_id)
                if private and (after.st_uid != os.geteuid() or after.st_mode & 0o077):
                    raise bootstrap.RecoveryRequired("credential_staging_required")
                current_parent = os.stat(selected.parent)
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
        current_parent = os.stat(destination.parent)
        if (current_parent.st_dev, current_parent.st_ino) != (
            destination_parent_identity.st_dev,
            destination_parent_identity.st_ino,
        ):
            raise bootstrap.RecoveryRequired("capture_target_changed")
        current_target = os.stat(destination, follow_symlinks=False)
        held_target = os.fstat(out)
        if (current_target.st_dev, current_target.st_ino) != (
            held_target.st_dev,
            held_target.st_ino,
        ):
            raise bootstrap.RecoveryRequired("capture_target_changed")
    finally:
        resources.retire()
