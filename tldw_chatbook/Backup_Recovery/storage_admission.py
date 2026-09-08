"""Process-local lifetime counting over one predeclared cross-process lease.

Lease ownership lives in a dedicated thread, so connections may close on a worker
thread. A process never widens a live scope: config changes require retirement and
re-enrollment. Native-unqualified ordinary use is distinct from recovery admission.
"""

from __future__ import annotations

import atexit
from contextlib import contextmanager
import sqlite3
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


class _CaptureLease:
    """A capture connection cannot outlive the native maintenance session."""

    def __init__(self, scope):
        self.scope = scope
        self.connection = None
        scope.resources.append(self)

    def attach(self, connection):
        self.connection = connection

    def close(self):
        if self in self.scope.resources:
            self.scope.resources.remove(self)

    def retire(self):
        if self.connection is not None:
            # Retire the native handle before releasing its capture authority.
            sqlite3.Connection.close(self.connection)
        self.close()


class _CaptureScope:
    def __init__(self, session, sources, staging):
        self.session = session
        self.sources = sources
        self.staging = staging
        self.staging_identity = staging.stat()
        self.resources = []
        self.active = True

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
        self.active = False
        for resource in tuple(self.resources):
            resource.retire()


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

    @contextmanager
    def capture_scope(self, sources: tuple[Path, ...], staging: Path):
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
            if not any(
                _contains_owned_path(Path(owned), source)
                for binding in bindings
                for owned in binding["roots"]
            ):
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
        scope = _CaptureScope(self, tuple(selected), staging)
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
        return None
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


def _consume_recovery_file(
    owner_id: str,
    candidate: Path,
    *,
    max_bytes: int,
    collect: bool,
    cancel: threading.Event | None = None,
) -> bytes | None:
    """Return bounded definition bytes after native reader retirement."""
    _recovery_file_limit(owner_id, max_bytes)
    selected = lexical_path(candidate)
    scope = getattr(_local, "capture_scope", None)
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
            if scope is not None:
                _check_capture_file_identity(scope, selected, info)
            chunks = []
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
            if scope is not None:
                _check_capture_file_identity(scope, selected, os.fstat(fd))
                current_parent = selected.parent.stat()
                held_parent = os.fstat(parent)
                if (current_parent.st_dev, current_parent.st_ino) != (
                    held_parent.st_dev,
                    held_parent.st_ino,
                ):
                    raise bootstrap.RecoveryRequired("capture_target_changed")
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
