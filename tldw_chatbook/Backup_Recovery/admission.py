"""Stable, cooperative cross-process admission outside managed data (ADR-126).

An admitted owner keeps its normal context until all connections/transactions are
retired. Cross-owner work declares its complete namespace set up front. Nested
admission is forbidden: retirement must never acquire another admission hold.
Legacy instance locks and arbitrary external editors do not implement this protocol.
"""

from __future__ import annotations

import hashlib
import math
import stat
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import ContextManager, Iterator

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tldw_chatbook.Utils.platform_files import fcntl, os

from .native_files import create_private_directory, flush_directory, pinned_directory
from .native_platform import flush_file
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
    version: int
    entries: dict[str, _Entry] = Field(default_factory=dict)


class _WriteIntent(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    version: int
    write_id: str
    before: _Registry | None
    after: _Registry


_REGISTRY_LIMIT = 1048576
_INTENT_LIMIT = 2 * _REGISTRY_LIMIT + 65536
_INTENT_NAME = "registry.pending.json"
_local = threading.local()


class Admission:
    """Participating-owner leases with stable persistent lock objects."""

    def __init__(self, control_root: Path):
        self.control_root = control_root
        self._observed_gates: dict[str, tuple[int, int]] = {}
        self._observed_groups: dict[tuple[str, ...], tuple[str, ...]] = {}
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
                self._write(fd, _Registry(version=1), initial=True)
            # Never reconstruct missing state in an existing control root.
            with self._lock(fd, "registry.lock", fcntl.LOCK_SH):
                self._read(fd)
        self._identity = self._root_identity()

    @classmethod
    def open_existing(cls, control_root: Path) -> "Admission":
        """Open established authority without recreating lost control evidence."""
        authority = object.__new__(cls)
        authority.control_root = control_root
        authority._observed_gates = {}
        authority._observed_groups = {}
        with authority._directory() as parent:
            with authority._lock(parent, "registry.lock", fcntl.LOCK_SH):
                authority._read(parent)
                info = os.fstat(parent)
                authority._identity = (info.st_dev, info.st_ino)
        return authority

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
            flush_file(fd)
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

    def _read_intent(self, parent: int) -> _WriteIntent | None:
        try:
            fd = self._open(parent, _INTENT_NAME, os.O_RDONLY)
        except FileNotFoundError:
            return None
        try:
            raw = os.read(fd, _INTENT_LIMIT + 1)
            if len(raw) > _INTENT_LIMIT:
                raise ValueError("intent_too_large")
            intent = _WriteIntent.model_validate_json(raw)
            states = (
                (intent.after,)
                if intent.before is None
                else (intent.before, intent.after)
            )
            if (
                intent.version != 1
                or len(intent.write_id) != 32
                or any(c not in "0123456789abcdef" for c in intent.write_id)
            ):
                raise ValueError("intent_invalid")
            if any(
                state.version != 1
                or len(state.model_dump_json().encode()) > _REGISTRY_LIMIT
                for state in states
            ):
                raise ValueError("intent_state_invalid")
            return intent
        except (ValidationError, ValueError):
            raise AdmissionError("registry_publication_recovery_required") from None
        finally:
            os.close(fd)

    def _read(self, parent: int) -> _Registry:
        if self._read_intent(parent) is not None:
            raise AdmissionError("registry_publication_recovery_required")
        fd = self._open(parent, "registry.json", os.O_RDONLY)
        try:
            data = os.read(fd, _REGISTRY_LIMIT + 1)
            if len(data) > _REGISTRY_LIMIT:
                raise AdmissionError("registry_too_large")
            result = _Registry.model_validate_json(data)
            if result.version != 1:
                raise AdmissionError("registry_version_unsupported")
            return result
        except (ValidationError, ValueError):
            raise AdmissionError("registry_invalid") from None
        finally:
            os.close(fd)

    @staticmethod
    def _write_new_record(parent: int, name: str, data: bytes) -> None:
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
            flush_file(fd)
        finally:
            os.close(fd)

    def _write(
        self, parent: int, registry: _Registry, *, initial: bool = False
    ) -> None:
        """Journal local authority before replacement; ambiguous writes stay fenced."""
        try:
            before = self._read(parent)
        except FileNotFoundError:
            if not initial:
                raise
            before = None  # Only initial construction may publish the first registry.
        if initial and before is not None:
            raise AdmissionError("registry_initialization_conflict")
        write_id = uuid.uuid4().hex
        intent = _WriteIntent(
            version=1, write_id=write_id, before=before, after=registry
        )
        data = registry.model_dump_json().encode()
        intent_data = intent.model_dump_json().encode()
        if (
            registry.version != 1
            or len(data) > _REGISTRY_LIMIT
            or len(intent_data) > _INTENT_LIMIT
        ):
            raise AdmissionError("registry_too_large")
        # Exclusive creation refuses any existing intent, even if it looks stale.
        # A failed/partial intent write is deliberately retained and fails closed.
        self._write_new_record(parent, _INTENT_NAME, intent_data)
        flush_directory(parent)
        name = f"registry-{write_id}.tmp"
        self._write_new_record(parent, name, data)
        os.replace(name, "registry.json", src_dir_fd=parent, dst_dir_fd=parent)
        flush_directory(parent)
        # Only this successfully flushed generation permits intent retirement.
        # Cleanup failure either leaves evidence or exposes this durable mapping.
        if self._read_intent(parent) != intent:
            raise AdmissionError("registry_publication_recovery_required")
        os.unlink(_INTENT_NAME, dir_fd=parent)
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
                observed = os.stat(root)
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

    def _groups(
        self, registry: _Registry, names: tuple[str, ...], recovery_journal=None
    ) -> tuple[str, ...]:
        if any(n not in registry.entries for n in names):
            raise AdmissionError("namespace_unregistered")
        tokens, missing = {}, {}
        for name, entry in registry.entries.items():
            roots = tuple(Path(r) for r in entry.roots + entry.proposed)
            tokens[name] = set(entry.historical)
            try:
                tokens[name].update(self._tokens(roots))
            except FileNotFoundError:
                if recovery_journal is None:
                    raise
                for root in roots:
                    try:
                        tokens[name].update(self._tokens((root,)))
                    except FileNotFoundError:
                        missing.setdefault(root, set()).add(name)
        if recovery_journal is not None:
            from .replacement import _recovery_admission_aliases

            aliases = _recovery_admission_aliases(
                recovery_journal, self.control_root, names, tuple(missing)
            )
            for root, (resolved, held) in aliases.items():
                recovered = self._tokens(held) | {
                    "path:" + str(root),
                    "path:" + str(resolved),
                }
                for name in missing[root]:
                    if (
                        root != resolved
                        and "path:" + str(resolved)
                        not in registry.entries[name].historical
                    ):
                        raise AdmissionError("recovery_root_alias_changed")
                    tokens[name].update(recovered)
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

    def _publication_root(self, root):
        """Observe the kind and canonical name of one natively held root."""
        tokens = self._tokens((root,))
        resolved = root.resolve(strict=True)
        info = os.stat(root)
        if (
            "path:" + str(resolved) not in tokens
            or f"inode:{info.st_dev}:{info.st_ino}" not in tokens
        ):
            raise AdmissionError("root_identity_unverified")
        return resolved, stat.S_ISDIR(info.st_mode)

    def _publication_roots(self, roots, names, recovery_journal):
        """Retain selected publication scope across this session's native moves."""
        observed, missing = set(), []
        for root in roots:
            try:
                observed.add(self._publication_root(root))
            except FileNotFoundError:
                if recovery_journal is None:
                    raise
                missing.append(root)
        if missing:
            from .replacement import _recovery_admission_aliases

            aliases = _recovery_admission_aliases(
                recovery_journal, self.control_root, names, tuple(missing)
            )
            for resolved, held in aliases.values():
                kinds = [self._publication_root(path)[1] for path in held]
                # If a replacement changes kind, neither version can widen an
                # exact-file namespace to descendants during this operation.
                observed.add((resolved, all(kinds)))
        return tuple(sorted(observed))

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
        recovery_journal=None,
    ) -> Iterator[None]:
        names = self._names(namespaces)
        with self._nonnested(), self._directory() as parent, ExitStack() as leases:
            # Gate acquisition never waits under registry lock: established owners
            # can retire without consulting either registry or gate.
            while True:
                self._check(deadline, cancel)
                trial = ExitStack()
                blocked_request = None
                try:
                    with self._lock(
                        parent, "registry.lock", fcntl.LOCK_SH, deadline, cancel
                    ):
                        registry = self._read(parent)
                        if not maintenance and not any(
                            entry.pending for entry in registry.entries.values()
                        ):
                            # Every expanded group includes the requested names.
                            # A closed requested gate already prevents entry; do
                            # not compete with its maintenance owner by scanning
                            # all target roots before waiting. This probe grants
                            # no admission: release it before fresh group checks.
                            # Pending remaps still need immediate alias-aware
                            # refusal through the existing full group check.
                            if any(name not in registry.entries for name in names):
                                raise AdmissionError("namespace_unregistered")
                            with ExitStack() as requested:
                                for name in names:
                                    fd = self._open(
                                        parent, self._key(name, "gate"), os.O_RDWR
                                    )
                                    requested.callback(os.close, fd)
                                    self._observe_gate(parent, name, fd)
                                    try:
                                        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                                    except BlockingIOError:
                                        blocked_request = name
                                        raise
                                    self._observe_gate(parent, name, fd)
                        group = self._groups(registry, names, recovery_journal)
                        if any(registry.entries[n].pending for n in group):
                            raise AdmissionError("remap_recovery_required")
                        for name in group:
                            fd = self._open(parent, self._key(name, "gate"), os.O_RDWR)
                            trial.callback(os.close, fd)
                            self._observe_gate(parent, name, fd)
                            fcntl.flock(
                                fd,
                                (fcntl.LOCK_EX if maintenance else fcntl.LOCK_SH)
                                | fcntl.LOCK_NB,
                            )
                            self._observe_gate(parent, name, fd)
                        if not maintenance:
                            self._observed_groups[names] = group
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
                    if blocked_request is None:
                        time.sleep(0.01)
                    else:
                        # Wait on one held native descriptor without registry
                        # authority or repeated filesystem/ACL reconstruction.
                        # This grants no admission: release the temporary gate
                        # before re-reading the registry and all target roots.
                        with self._lock(
                            parent,
                            self._key(blocked_request, "gate"),
                            fcntl.LOCK_SH,
                            deadline,
                            cancel,
                        ) as fd:
                            self._observe_gate(parent, blocked_request, fd)
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
                    if self._groups(current, names, recovery_journal) != group or any(
                        current.entries[n].pending for n in group
                    ):
                        raise AdmissionError("admission_scope_changed")
                    self._check(deadline, cancel)
                    from .storage_admission import (
                        _failed_capture_holds,
                        _mint_maintenance_session,
                    )

                    recovery_roots = None
                    if recovery_journal is not None:
                        from .replacement import _recovery_staging_roots

                        recovery_roots = _recovery_staging_roots(
                            recovery_journal,
                            self.control_root,
                            names,
                            tuple(
                                Path(r)
                                for entry in current.entries.values()
                                for r in entry.roots
                            ),
                        )

                    session = _mint_maintenance_session(
                        (Path(r) for n in group for r in current.entries[n].roots),
                        (
                            Path(r)
                            for entry in current.entries.values()
                            for r in entry.roots
                        ),
                        self.control_root,
                        group,
                        self._identity,
                        recovery_roots=recovery_roots,
                        publication_roots=self._publication_roots(
                            tuple(
                                Path(r) for n in group for r in current.entries[n].roots
                            ),
                            names,
                            recovery_journal,
                        ),
                    )
                    try:
                        yield session
                    finally:
                        try:
                            session._retire()
                        except BaseException:
                            # Transfer native ownership before context unwinding;
                            # an unresolved connection can still mutate storage.
                            _failed_capture_holds.append(
                                (trial.pop_all(), leases.pop_all(), session)
                            )
                            raise AdmissionError(
                                "capture_resources_not_retired"
                            ) from None
                else:
                    trial.close()
                    yield

    def normal(self, namespaces: tuple[str, ...]) -> ContextManager[None]:
        """Hold before opening owners, until transactions and connections retire."""
        return self._admit(namespaces, False, None, None)

    def pause_requested(self, namespaces: tuple[str, ...]) -> bool:
        """Probe native contention without waiting or granting capture authority.

        Registry contention defers this observation: registering source scopes
        is not a request to retire a live app. Only a validated contended gate
        requests pause. Missing, unsafe, replaced or pending evidence refuses.
        The holder must retain its normal
        lease until its actual producers and resources retire; this observation
        never acknowledges drain and creates no durable request/ack records.
        """
        names = self._names(namespaces)
        with self._directory() as parent, ExitStack() as descriptors:
            registry_fd = self._open(parent, "registry.lock", os.O_RDWR)
            descriptors.callback(os.close, registry_fd)
            try:
                fcntl.flock(registry_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            registry = self._read(parent)
            group = self._groups(registry, names)
            if any(registry.entries[name].pending for name in group):
                raise AdmissionError("remap_recovery_required")
            previous = self._observed_groups.setdefault(names, group)
            if previous != group:
                raise AdmissionError("admission_scope_changed")
            contended = False
            for name in group:
                fd = self._open(parent, self._key(name, "gate"), os.O_RDWR)
                descriptors.callback(os.close, fd)
                self._observe_gate(parent, name, fd)
                try:
                    fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    contended = True
                # Continue checking every gate, even when another is contended.
                self._observe_gate(parent, name, fd)
            return contended

    def _observe_gate(self, parent: int, name: str, fd: int) -> None:
        """Bind pause observations to the same pinned gates used for admission."""
        held = os.fstat(fd)
        current = os.stat(self._key(name, "gate"), dir_fd=parent, follow_symlinks=False)
        identity = held.st_dev, held.st_ino
        previous = self._observed_gates.setdefault(name, identity)
        if identity != previous or identity != (current.st_dev, current.st_ino):
            raise AdmissionError("admission_gate_changed")

    def maintenance(
        self,
        namespaces: tuple[str, ...],
        timeout: float,
        *,
        cancel: threading.Event | None = None,
    ) -> ContextManager[None]:
        """Close new admission, drain safely, then hold exclusive owner access."""
        return self._admit(namespaces, True, self._deadline(timeout), cancel)

    def _replacement_recovery(self, journal, timeout, *, cancel=None):
        """Hold the same stable locks across journal-proven rename gaps only."""
        from .replacement import _recovery_admission_record

        names, _, _ = _recovery_admission_record(journal, self.control_root)
        return self._admit(
            names, True, self._deadline(timeout), cancel, recovery_journal=journal
        )

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
