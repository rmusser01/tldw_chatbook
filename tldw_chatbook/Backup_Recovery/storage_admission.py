"""Process-local lifetime counting over one predeclared cross-process lease.

Lease ownership lives in a dedicated thread, so connections may close on a worker
thread. A process never widens a live scope: config changes require retirement and
re-enrollment. Native-unqualified ordinary use is distinct from recovery admission.
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path
import stat
import threading

from . import bootstrap
from .admission import Admission, _local
from .control_records import UNBOUND_NAMESPACE, admission_authority
from .profile_paths import effective_config_path, lexical_path
from .qualification import qualified_for

_lock = threading.RLock()
_holds: dict[tuple[int, str], "_Hold"] = {}
_startups: dict[tuple[int, str], "StorageLease"] = {}
_forked_with_owners = False


class _Hold:
    def __init__(self, authority: Admission, names: tuple[str, ...]):
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
        self.ready.wait()
        if self.error is not None:
            raise bootstrap.RecoveryRequired("storage_admission_unavailable") from None

    def _run(self, authority: Admission) -> None:
        try:
            with authority.normal(self.names):
                self.ready.set()
                self.stop.wait()
        except BaseException as error:
            self.error = error
            self.ready.set()


class StorageLease:
    """Idempotent retirement token; successful close releases actual participation."""

    def __init__(self, key: tuple[int, str] | None):
        self._key = key

    def close(self) -> None:
        with _lock:
            key, self._key = self._key, None
            if key is None or key[0] != os.getpid():
                return
            hold = _holds[key]
            hold.count -= 1
            if hold.count == 0:
                hold.stop.set()
                hold.thread.join()
                del _holds[key]

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


def _scope(root: Path, selector: Path, path: Path | None) -> tuple[str, ...]:
    pending, profiles = bootstrap._records(root)
    registry = bootstrap._registry(root)
    binding = bootstrap._binding(selector, profiles, registry) if profiles else None
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
    try:
        return _acquire_storage(path)
    except bootstrap.RecoveryRequired:
        raise
    except (OSError, ValueError, RuntimeError):
        raise bootstrap.RecoveryRequired("storage_admission_unavailable") from None


def _acquire_storage(path: Path | None = None) -> StorageLease:
    """Check fixed evidence and hold declared scope before any owned open/write.

    An ordinary seam used inside maintenance refuses promptly; later capture owners
    need a separately validated capability, never a boolean bypass here.
    """
    if _forked_with_owners:
        raise bootstrap.RecoveryRequired("forked_owner_restart_required")
    root = bootstrap.default_bootstrap_root()
    selector = effective_config_path()
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
        return StorageLease(None)
    with _lock:
        authority = admission_authority(root)
        names = _scope(root, selector, lexical_path(path) if path is not None else None)
        key = (os.getpid(), str(root))
        hold = _holds.get(key)
        if hold is not None and names != hold.names:
            raise bootstrap.RecoveryRequired("close_owners_before_scope_change")
        if hold is None:
            hold = _Hold(authority, names)
            _holds[key] = hold
        hold.count += 1
        token = StorageLease(key)
        # Enrollment races an unbound selection. Revalidate after acquiring its
        # lease; never enter on a stale pre-enrollment decision.
        try:
            allowed, reason = bootstrap.startup_permission(selector, root)
            if not allowed:
                raise bootstrap.RecoveryRequired(reason)
            if (
                _scope(root, selector, lexical_path(path) if path is not None else None)
                != names
            ):
                raise bootstrap.RecoveryRequired("storage_scope_changed")
        except BaseException:
            token.close()
            raise
        return token


def admit_startup() -> None:
    """Keep process enrollment from before runtime imports through process exit."""
    bootstrap.require_startup_permission()
    key = (os.getpid(), str(bootstrap.default_bootstrap_root()))
    with _lock:
        if key not in _startups:
            try:
                _startups[key] = acquire_storage()
            except bootstrap.RecoveryRequired as error:
                raise SystemExit("Recovery required: " + str(error)) from None


def _shutdown() -> None:
    for lease in list(_startups.values()):
        lease.close()
    _startups.clear()


def _after_fork() -> None:
    # Forked children cannot inherit a fictitious live lease thread/refcount.
    # Spawn imports establish their own leases before loading runtime modules.
    global _lock, _holds, _startups, _forked_with_owners
    _forked_with_owners = bool(_holds)
    _lock = threading.RLock()
    _holds = {}
    _startups = {}


atexit.register(_shutdown)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)
