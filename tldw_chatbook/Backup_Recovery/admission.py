"""Stable, cooperative cross-process admission outside managed data (ADR-126).

An admitted owner keeps its normal context until all connections/transactions are
retired. Cross-owner work declares its complete namespace set up front. Nested
admission is forbidden: retirement must never acquire another admission hold.
Legacy instance locks and arbitrary external editors do not implement this protocol.
"""

from __future__ import annotations

try:
    import fcntl
except ImportError:  # Unqualified platforms still expose a capability refusal.
    fcntl = None
import hashlib
import math
import os
import stat
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import ContextManager, Iterator

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .native_files import create_private_directory, flush_directory, pinned_directory
from .qualification import _qualified_identity, native_identity, qualified_for


class AdmissionError(RuntimeError):
    """Sanitized refusal; callers retain drafts and drain normally."""


class AdmissionTimeout(AdmissionError):
    pass


class AdmissionCancelled(AdmissionError):
    pass


class _Entry(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    roots: list[str]
    historical: list[str] = Field(default_factory=list)
    pending: str | None = None
    proposed: list[str] = Field(default_factory=list)


class _Registry(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    version: int = 1
    entries: dict[str, _Entry] = Field(default_factory=dict)


_local = threading.local()


class Admission:
    """Participating-owner leases with stable persistent lock objects."""

    def __init__(self, control_root: Path):
        self.control_root = control_root
        allowed, reason = qualified_for("admission", control_root.parent)
        if not allowed:
            raise AdmissionError(reason)
        created = False
        try:
            create_private_directory(control_root)
            created = True
        except FileExistsError:
            pass
        with self._directory() as fd:
            if created:
                self._create_lock(fd, "registry.lock")
                self._write(fd, _Registry())
            # Never reconstruct missing state in an existing control root.
            with self._lock(fd, "registry.lock", fcntl.LOCK_SH):
                self._read(fd)
        self._identity = self._root_identity()

    def _root_identity(self) -> tuple[int, int]:
        with self._directory() as fd:
            info = os.fstat(fd)
            return info.st_dev, info.st_ino

    @contextmanager
    def _directory(self) -> Iterator[int]:
        with pinned_directory(self.control_root) as fd:
            allowed, reason = _qualified_identity("admission", native_identity(fd))
            if not allowed:
                raise AdmissionError(reason)
            info = os.fstat(fd)
            if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise AdmissionError("control_root_not_private")
            expected = getattr(self, "_identity", None)
            if expected is not None and expected != (info.st_dev, info.st_ino):
                raise AdmissionError("control_root_changed")
            yield fd

    @staticmethod
    def _create_lock(parent: int, name: str) -> None:
        fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent,
        )
        try:
            os.fsync(fd)
            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
        finally:
            os.close(fd)
        flush_directory(parent)

    @staticmethod
    def _open(parent: int, name: str, flags: int) -> int:
        fd = os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o077
        ):
            os.close(fd)
            raise AdmissionError("control_file_unsafe")
        return fd

    @contextmanager
    def _lock(
        self,
        parent: int,
        name: str,
        mode: int,
        deadline: float | None = None,
        cancel: threading.Event | None = None,
    ) -> Iterator[int]:
        fd = self._open(parent, name, os.O_RDWR)
        try:
            while True:
                self._check(deadline, cancel)
                try:
                    fcntl.flock(fd, mode | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.01)
            yield fd
        finally:
            os.close(fd)

    @staticmethod
    def _check(deadline: float | None, cancel: threading.Event | None) -> None:
        if cancel is not None and cancel.is_set():
            raise AdmissionCancelled("admission_cancelled")
        if deadline is not None and time.monotonic() >= deadline:
            raise AdmissionTimeout("admission_timeout")

    @staticmethod
    def _deadline(timeout: float) -> float:
        if (
            not isinstance(timeout, (float, int))
            or isinstance(timeout, bool)
            or not math.isfinite(timeout)
            or timeout < 0
        ):
            raise ValueError("invalid_admission_timeout")
        return time.monotonic() + timeout

    def _read(self, parent: int) -> _Registry:
        fd = self._open(parent, "registry.json", os.O_RDONLY)
        try:
            data = os.read(fd, 1048577)
            if len(data) > 1048576:
                raise AdmissionError("registry_too_large")
            result = _Registry.model_validate_json(data)
            if result.version != 1:
                raise AdmissionError("registry_version_unsupported")
            return result
        except (ValidationError, ValueError):
            raise AdmissionError("registry_invalid") from None
        finally:
            os.close(fd)

    def _write(self, parent: int, registry: _Registry) -> None:
        name = f"registry-{uuid.uuid4().hex}.tmp"
        data = registry.model_dump_json().encode()
        if len(data) > 1048576:
            raise AdmissionError("registry_too_large")
        fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent,
        )
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]
            os.fsync(fd)
            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
        finally:
            os.close(fd)
        os.replace(name, "registry.json", src_dir_fd=parent, dst_dir_fd=parent)
        flush_directory(parent)

    @staticmethod
    def _names(namespaces: tuple[str, ...]) -> tuple[str, ...]:
        if (
            type(namespaces) is not tuple
            or not namespaces
            or any(
                type(n) is not str or not n or len(n) > 256 or "\x00" in n
                for n in namespaces
            )
        ):
            raise ValueError("invalid_namespaces")
        return tuple(sorted(set(namespaces)))

    @staticmethod
    def _key(namespace: str, kind: str) -> str:
        return hashlib.sha256(namespace.encode()).hexdigest() + "." + kind

    def _tokens(self, roots: tuple[Path, ...]) -> set[str]:
        if type(roots) is not tuple or not roots:
            raise ValueError("invalid_roots")
        result = set()
        for root in roots:
            if not root.is_absolute() or ".." in root.parts:
                raise AdmissionError("absolute_root_required")
            resolved = root.resolve(strict=True)
            with pinned_directory(resolved.parent) as parent:
                info = os.stat(resolved.name, dir_fd=parent, follow_symlinks=False)
                observed = root.stat()
                if (info.st_dev, info.st_ino) != (
                    observed.st_dev,
                    observed.st_ino,
                ) or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                    raise AdmissionError("root_identity_unverified")
                if (
                    resolved == self.control_root
                    or resolved in self.control_root.parents
                    or self.control_root in resolved.parents
                ):
                    raise AdmissionError("control_root_overlaps_target")
                result.update(
                    (
                        "path:" + str(root),
                        "path:" + str(resolved),
                        f"inode:{info.st_dev}:{info.st_ino}",
                    )
                )
        return result

    def _groups(self, registry: _Registry, names: tuple[str, ...]) -> tuple[str, ...]:
        if any(n not in registry.entries for n in names):
            raise AdmissionError("namespace_unregistered")
        tokens = {
            n: set(entry.historical)
            | self._tokens(tuple(Path(r) for r in entry.roots + entry.proposed))
            for n, entry in registry.entries.items()
        }
        selected = set(names)
        while True:
            shared = set().union(*(tokens[n] for n in selected))

            def overlaps(value: set[str]) -> bool:
                if value & shared:
                    return True
                paths = [Path(t[5:]) for t in value if t.startswith("path:")]
                other_paths = [Path(t[5:]) for t in shared if t.startswith("path:")]
                return any(
                    a in b.parents or b in a.parents for a in paths for b in other_paths
                )

            expanded = selected | {n for n, value in tokens.items() if overlaps(value)}
            if expanded == selected:
                return tuple(sorted(selected))
            selected = expanded

    @contextmanager
    def _nonnested(self) -> Iterator[None]:
        if getattr(_local, "admitted", False):
            raise AdmissionError("nested_admission_forbidden")
        _local.admitted = True
        try:
            yield
        finally:
            _local.admitted = False

    def register(self, namespace: str, roots: tuple[Path, ...]) -> None:
        """Register physically verified roots; busy alias introduction is refused."""
        self._names((namespace,))
        with (
            self._nonnested(),
            self._directory() as parent,
            self._lock(parent, "registry.lock", fcntl.LOCK_EX),
        ):
            registry = self._read(parent)
            tokens = self._tokens(roots)
            if namespace in registry.entries:
                if registry.entries[namespace].roots != [str(r) for r in roots]:
                    raise AdmissionError("remap_required")
                return
            registry.entries[namespace] = _Entry(
                roots=[str(r) for r in roots], historical=sorted(tokens)
            )
            group = self._groups(registry, (namespace,))
            with ExitStack() as stack:
                for name in group:
                    if name == namespace:
                        continue
                    if registry.entries[name].pending:
                        raise AdmissionError("remap_recovery_required")
                    # No waiting while holding registry authority.
                    fd = self._open(parent, self._key(name, "gate"), os.O_RDWR)
                    stack.callback(os.close, fd)
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        lease = self._open(parent, self._key(name, "lease"), os.O_RDWR)
                        stack.callback(os.close, lease)
                        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        raise AdmissionError("namespace_busy") from None
                for kind in ("gate", "lease", "incompatible"):
                    self._create_lock(parent, self._key(namespace, kind))
                self._write(parent, registry)

    @contextmanager
    def _admit(
        self,
        namespaces: tuple[str, ...],
        maintenance: bool,
        deadline: float | None,
        cancel: threading.Event | None,
        incompatible: bool = False,
    ) -> Iterator[None]:
        names = self._names(namespaces)
        with self._nonnested(), self._directory() as parent, ExitStack() as leases:
            # Gate acquisition never waits under registry lock: established owners
            # can retire without consulting either registry or gate.
            while True:
                self._check(deadline, cancel)
                trial = ExitStack()
                try:
                    with self._lock(
                        parent, "registry.lock", fcntl.LOCK_SH, deadline, cancel
                    ):
                        registry = self._read(parent)
                        group = self._groups(registry, names)
                        if any(registry.entries[n].pending for n in group):
                            raise AdmissionError("remap_recovery_required")
                        for name in group:
                            fd = self._open(parent, self._key(name, "gate"), os.O_RDWR)
                            trial.callback(os.close, fd)
                            fcntl.flock(
                                fd,
                                (fcntl.LOCK_EX if maintenance else fcntl.LOCK_SH)
                                | fcntl.LOCK_NB,
                            )
                        if not maintenance:
                            for name in group:
                                kind = "incompatible" if incompatible else "lease"
                                leases.enter_context(
                                    self._lock(
                                        parent,
                                        self._key(name, kind),
                                        fcntl.LOCK_SH,
                                        deadline,
                                        cancel,
                                    )
                                )
                    break
                except BlockingIOError:
                    trial.close()
                    time.sleep(0.01)
                except BaseException:
                    trial.close()
                    raise
            with trial:
                if maintenance:
                    for name in group:
                        fd = self._open(
                            parent, self._key(name, "incompatible"), os.O_RDWR
                        )
                        leases.callback(os.close, fd)
                        try:
                            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            raise AdmissionError("known_incompatible_client") from None
                    for name in group:
                        leases.enter_context(
                            self._lock(
                                parent,
                                self._key(name, "lease"),
                                fcntl.LOCK_EX,
                                deadline,
                                cancel,
                            )
                        )
                    # The final scope is immutable during exclusive capture, but
                    # the registry was not held while waiting for retirement.
                    leases.enter_context(
                        self._lock(
                            parent, "registry.lock", fcntl.LOCK_SH, deadline, cancel
                        )
                    )
                    current = self._read(parent)
                    if self._groups(current, names) != group or any(
                        current.entries[n].pending for n in group
                    ):
                        raise AdmissionError("admission_scope_changed")
                    self._check(deadline, cancel)
                    yield
                else:
                    trial.close()
                    yield

    def normal(self, namespaces: tuple[str, ...]) -> ContextManager[None]:
        """Hold before opening owners, until transactions and connections retire."""
        return self._admit(namespaces, False, None, None)

    def maintenance(
        self,
        namespaces: tuple[str, ...],
        timeout: float,
        *,
        cancel: threading.Event | None = None,
    ) -> ContextManager[None]:
        """Close new admission, drain safely, then hold exclusive owner access."""
        return self._admit(namespaces, True, self._deadline(timeout), cancel)

    def incompatible(self, namespaces: tuple[str, ...]) -> ContextManager[None]:
        """Represent positively observed incompatible activity for its OS lifetime."""
        return self._admit(namespaces, False, None, None, incompatible=True)

    def remap(
        self,
        namespace: str,
        roots: tuple[Path, ...],
        timeout: float,
        *,
        cancel: threading.Event | None = None,
    ) -> None:
        """Reserve old/new aliases durably, drain without registry, publish mapping.

        Interrupted/failed remaps retain pending evidence and refuse admission until
        an explicit recovery owner reconciles them. Historical aliases are retained
        conservatively; no automatic stale-evidence cleanup or PID-based inference.
        """
        names = self._names((namespace,))
        deadline = self._deadline(timeout)
        self._check(deadline, cancel)
        with self._nonnested(), self._directory() as parent:
            with self._lock(parent, "registry.lock", fcntl.LOCK_EX, deadline, cancel):
                registry = self._read(parent)
                self._groups(registry, names)
                entry = registry.entries[namespace]
                if entry.pending:
                    raise AdmissionError("remap_recovery_required")
                tokens = self._tokens(roots)
                entry.proposed = [str(r) for r in roots]
                entry.historical = sorted(
                    set(entry.historical)
                    | self._tokens(tuple(Path(r) for r in entry.roots))
                    | tokens
                )
                group = self._groups(registry, names)
                if any(registry.entries[n].pending for n in group):
                    raise AdmissionError("remap_recovery_required")
                operation = uuid.uuid4().hex
                entry.pending = operation
                self._write(parent, registry)
            with ExitStack() as holds:
                for name in group:
                    holds.enter_context(
                        self._lock(
                            parent,
                            self._key(name, "gate"),
                            fcntl.LOCK_EX,
                            deadline,
                            cancel,
                        )
                    )
                for name in group:
                    incompatible_fd = self._open(
                        parent, self._key(name, "incompatible"), os.O_RDWR
                    )
                    holds.callback(os.close, incompatible_fd)
                    try:
                        fcntl.flock(incompatible_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        raise AdmissionError("known_incompatible_client") from None
                    holds.enter_context(
                        self._lock(
                            parent,
                            self._key(name, "lease"),
                            fcntl.LOCK_EX,
                            deadline,
                            cancel,
                        )
                    )
                with self._lock(
                    parent, "registry.lock", fcntl.LOCK_EX, deadline, cancel
                ):
                    registry = self._read(parent)
                    entry = registry.entries[namespace]
                    if (
                        entry.pending != operation
                        or self._groups(registry, names) != group
                        or self._tokens(roots) != tokens
                    ):
                        raise AdmissionError("remap_scope_changed")
                    self._check(deadline, cancel)
                    entry.roots, entry.proposed, entry.pending = (
                        entry.proposed,
                        [],
                        None,
                    )
                    self._write(parent, registry)
