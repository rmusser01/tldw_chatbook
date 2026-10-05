"""Typed state and app-owned readiness for workspace Change Review."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
import inspect
from queue import Empty, Full, Queue
from threading import RLock, Thread
import sys
import time
from typing import Callable, Protocol

from tldw_chatbook.Workspaces.models import RuntimeBindingStatus


class _WorkspaceCompositeEntryError(RuntimeError):
    """A qualified optional connection failed before the first original read."""


class ChangeReviewState(str, Enum):
    """Availability or consent state for Change Review."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ChangeReviewCapability:
    """Global Change Review capability state."""

    state: ChangeReviewState


MISSING_CHANGE_REVIEW_REVISION = "missing"


@dataclass(frozen=True, slots=True)
class ChangeReviewConsent:
    """One durable per-workspace consent observation."""

    state: ChangeReviewState
    revision: str = ""


class ChangeReviewStateConflict(RuntimeError):
    """Raised when a consent compare-and-set observation is stale."""


class RootReadinessState(str, Enum):
    """Initialization state for one consent-bound workspace root."""

    PREPARING = "preparing"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SkippedReviewRoot:
    """Alias-only explanation for a root omitted from one turn."""

    alias: str
    reason: str


@dataclass(frozen=True, slots=True)
class ChangeReviewAdmission:
    """Immutable Change Review inputs captured at turn admission."""

    ready_roots: tuple[str, ...] = ()
    skipped_roots: tuple[SkippedReviewRoot, ...] = ()
    ready_aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RootReadiness:
    """Public alias-only readiness projection for Settings."""

    alias: str
    state: RootReadinessState
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ChangeReviewStatus:
    """One revision-consistent consent and readiness observation."""

    capability: ChangeReviewCapability
    consent: ChangeReviewConsent
    roots: tuple[RootReadiness, ...] = ()


class _FolderBinding(Protocol):
    binding_id: str
    locator: str
    status: object


class _Registry(Protocol):
    def read_change_review_consent(self, workspace_id: str) -> ChangeReviewConsent: ...

    def compare_and_set_change_review_consent(
        self,
        workspace_id: str,
        *,
        expected: ChangeReviewConsent,
        enabled: bool,
    ) -> ChangeReviewConsent: ...

    def list_folder_bindings(self, workspace_id: str) -> tuple[_FolderBinding, ...]: ...


@dataclass(frozen=True, slots=True)
class _InitializationWork:
    workspace_id: str
    root: str
    alias: str
    revision: str
    generation: int


@dataclass(frozen=True, slots=True)
class _ReadinessEntry:
    alias: str
    revision: str
    state: RootReadinessState
    reason: str = ""


def _default_capability_reader() -> ChangeReviewCapability:
    from tldw_chatbook.Workspaces.change_bounds import read_change_review_capability

    return read_change_review_capability()


def _default_root_initializer(root: str) -> None:
    from tldw_chatbook.Workspaces.change_turn_tracker import initialize_shadow_root

    initialize_shadow_root(root)


class ChangeReviewConsentService:
    """Own consent linearization and bounded root initialization.

    Admission, toggles, and initializer completion all linearize on one lock.
    Filesystem initialization runs on a fixed daemon pool so a slow root never
    creates an unbounded thread population or blocks chat admission.
    """

    _PREPARING_REASON = "Preparing change history; this turn continues without it."
    _FAILED_REASON = (
        "Change history preparation failed; this turn continues without it."
    )
    _QUEUE_FULL_REASON = "Change history preparation is busy; retry from Settings."

    def __init__(
        self,
        registry: _Registry,
        *,
        initialize_root: Callable[[str], None] | None = None,
        capability_reader: Callable[[], ChangeReviewCapability] | None = None,
        worker_count: int = 2,
        queue_capacity: int = 32,
    ) -> None:
        if worker_count < 1:
            raise ValueError("worker_count must be positive")
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be positive")
        self._registry = registry
        self._initialize_root = initialize_root or _default_root_initializer
        self._capability_reader = capability_reader or _default_capability_reader
        self._lock = RLock()
        self._queue: Queue[_InitializationWork] = Queue(maxsize=queue_capacity)
        self._readiness: dict[tuple[str, str], _ReadinessEntry] = {}
        self._generation = 0
        self._disposed = False
        self._workers = tuple(
            Thread(
                target=self._worker_loop,
                name=f"change-review-init-{index}",
                daemon=True,
            )
            for index in range(worker_count)
        )
        self._workers_started = False

    def admit_turn(self, workspace_id: str) -> ChangeReviewAdmission:
        """Capture ready roots without waiting for filesystem initialization."""
        with self._lock:
            if self._disposed:
                return ChangeReviewAdmission()
            capability = self._capability_reader()
            try:
                registry = self._registry
                registry_module = sys.modules.get(
                    "tldw_chatbook.Workspaces.registry_service"
                )
                scope = nullcontext(None)
                if registry_module is not None and _workspace_consent_current(
                    self, registry_module
                ):
                    scope = registry_module._workspace_composite_context(
                        registry,
                        lambda: _workspace_consent_current(self, registry_module)
                        and self._registry is registry,
                        enabled=(
                            type(workspace_id) is str and bool(workspace_id.strip())  # noqa: E721 -- stock input avoids custom validation callbacks
                        ),
                    )
                with scope as check:
                    consent = self._registry.read_change_review_consent(workspace_id)
                    if check is not None:
                        check()
                    if (
                        capability.state is not ChangeReviewState.ENABLED
                        or consent.state is not ChangeReviewState.ENABLED
                    ):
                        return ChangeReviewAdmission()
                    return self._admit_enabled_locked(workspace_id, consent)
            except _WorkspaceCompositeEntryError:
                return ChangeReviewAdmission()

    def status(self, workspace_id: str) -> ChangeReviewStatus:
        """Return a revision-consistent alias-only Settings projection.

        Args:
            workspace_id: Workspace whose consent and root readiness to inspect.

        Returns:
            The capability, consent, and alias-only readiness captured under one
            service lock.
        """
        with self._lock:
            capability = self._capability_reader()
            try:
                registry = self._registry
                registry_module = sys.modules.get(
                    "tldw_chatbook.Workspaces.registry_service"
                )
                scope = nullcontext(None)
                if registry_module is not None and _workspace_consent_current(
                    self, registry_module
                ):
                    scope = registry_module._workspace_composite_context(
                        registry,
                        lambda: _workspace_consent_current(self, registry_module)
                        and self._registry is registry,
                        enabled=(
                            type(workspace_id) is str and bool(workspace_id.strip())  # noqa: E721 -- stock input avoids custom validation callbacks
                        ),
                    )
                with scope as check:
                    consent = self._registry.read_change_review_consent(workspace_id)
                    if check is not None:
                        check()
                    roots: list[RootReadiness] = []
                    if (
                        not self._disposed
                        and capability.state is ChangeReviewState.ENABLED
                        and consent.state is ChangeReviewState.ENABLED
                    ):
                        for binding in self._current_bindings(workspace_id):
                            # Folder bindings are canonicalized at registry admission and
                            # READY bindings are rechecked for symlink/resolve drift.
                            root = binding.locator
                            entry = self._readiness.get((workspace_id, root))
                            if entry is not None and entry.revision == consent.revision:
                                roots.append(
                                    RootReadiness(
                                        alias=binding.binding_id,
                                        state=entry.state,
                                        reason=entry.reason,
                                    )
                                )
                    return ChangeReviewStatus(capability, consent, tuple(roots))
            except _WorkspaceCompositeEntryError:
                return ChangeReviewStatus(
                    capability, ChangeReviewConsent(ChangeReviewState.UNAVAILABLE), ()
                )

    def toggle(
        self,
        workspace_id: str,
        *,
        expected: ChangeReviewConsent,
        enabled: bool,
    ) -> ChangeReviewConsent:
        """CAS-toggle consent while preserving admission linearization."""
        with self._lock:
            if self._disposed:
                raise RuntimeError("Change Review service is shut down")
            capability = self._capability_reader()
            if capability.state is not ChangeReviewState.ENABLED:
                raise RuntimeError("Change Review capability is unavailable")
            committed = self._registry.compare_and_set_change_review_consent(
                workspace_id,
                expected=expected,
                enabled=enabled,
            )
            self._discard_workspace_readiness_locked(workspace_id)
            if enabled:
                self._admit_enabled_locked(workspace_id, committed)
            return committed

    def binding_added(self, workspace_id: str, binding: _FolderBinding) -> None:
        """Best-effort readiness scheduling after a durable binding add."""
        del binding
        self.admit_turn(workspace_id)

    def retry_failed_roots(self, workspace_id: str) -> int:
        """Retry currently failed roots once, returning the scheduled count."""
        with self._lock:
            if self._disposed:
                return 0
            capability = self._capability_reader()
            consent = self._registry.read_change_review_consent(workspace_id)
            if (
                capability.state is not ChangeReviewState.ENABLED
                or consent.state is not ChangeReviewState.ENABLED
            ):
                return 0
            scheduled = 0
            for binding in self._current_bindings(workspace_id):
                root = binding.locator
                key = (workspace_id, root)
                entry = self._readiness.get(key)
                if (
                    entry is None
                    or entry.revision != consent.revision
                    or entry.state is not RootReadinessState.FAILED
                ):
                    continue
                if self._schedule_locked(
                    workspace_id,
                    root,
                    binding.binding_id,
                    consent.revision,
                ):
                    scheduled += 1
            return scheduled

    def shutdown(self, *, timeout: float = 1.0) -> None:
        """Cancel queued work and join workers until one shared deadline."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._lock:
            if not self._disposed:
                self._disposed = True
                self._generation += 1
                self._readiness.clear()
            while True:
                try:
                    self._queue.get_nowait()
                except Empty:
                    break
                else:
                    self._queue.task_done()
        workers = self._workers if self._workers_started else ()
        for worker in workers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            worker.join(remaining)

    def _admit_enabled_locked(
        self,
        workspace_id: str,
        consent: ChangeReviewConsent,
    ) -> ChangeReviewAdmission:
        ready: list[str] = []
        ready_aliases: list[str] = []
        skipped: list[SkippedReviewRoot] = []
        for binding in self._current_bindings(workspace_id):
            root = binding.locator
            key = (workspace_id, root)
            entry = self._readiness.get(key)
            if entry is None or entry.revision != consent.revision:
                self._schedule_locked(
                    workspace_id,
                    root,
                    binding.binding_id,
                    consent.revision,
                )
                entry = self._readiness[key]
            if entry.state is RootReadinessState.READY:
                ready.append(root)
                ready_aliases.append(binding.binding_id)
            else:
                skipped.append(
                    SkippedReviewRoot(alias=binding.binding_id, reason=entry.reason)
                )
        return ChangeReviewAdmission(
            ready_roots=tuple(ready),
            skipped_roots=tuple(skipped),
            ready_aliases=tuple(ready_aliases),
        )

    def _current_bindings(self, workspace_id: str) -> tuple[_FolderBinding, ...]:
        return tuple(
            binding
            for binding in self._registry.list_folder_bindings(workspace_id)
            if binding.status is RuntimeBindingStatus.READY
        )

    def _schedule_locked(
        self,
        workspace_id: str,
        root: str,
        alias: str,
        revision: str,
    ) -> bool:
        if not self._workers_started:
            for worker in self._workers:
                worker.start()
            self._workers_started = True
        work = _InitializationWork(
            workspace_id=workspace_id,
            root=root,
            alias=alias,
            revision=revision,
            generation=self._generation,
        )
        try:
            self._queue.put_nowait(work)
        except Full:
            self._readiness[(workspace_id, root)] = _ReadinessEntry(
                alias=alias,
                revision=revision,
                state=RootReadinessState.FAILED,
                reason=self._QUEUE_FULL_REASON,
            )
            return False
        self._readiness[(workspace_id, root)] = _ReadinessEntry(
            alias=alias,
            revision=revision,
            state=RootReadinessState.PREPARING,
            reason=self._PREPARING_REASON,
        )
        return True

    def _worker_loop(self) -> None:
        while True:
            with self._lock:
                if self._disposed:
                    return
            try:
                work = self._queue.get(timeout=0.05)
            except Empty:
                continue
            error: Exception | None = None
            try:
                self._initialize_root(work.root)
            except Exception as exc:  # noqa: BLE001 -- failure is readiness state
                error = exc
            finally:
                self._queue.task_done()
            self._complete(work, error)

    def _complete(
        self,
        work: _InitializationWork,
        error: Exception | None,
    ) -> None:
        with self._lock:
            if self._disposed or work.generation != self._generation:
                return
            consent = self._registry.read_change_review_consent(work.workspace_id)
            if (
                consent.state is not ChangeReviewState.ENABLED
                or consent.revision != work.revision
            ):
                key = (work.workspace_id, work.root)
                current = self._readiness.get(key)
                if current is not None and current.revision == work.revision:
                    self._readiness.pop(key, None)
                return
            current = self._readiness.get((work.workspace_id, work.root))
            if (
                current is None
                or current.revision != work.revision
                or current.state is not RootReadinessState.PREPARING
            ):
                return
            if error is None:
                replacement = _ReadinessEntry(
                    alias=work.alias,
                    revision=work.revision,
                    state=RootReadinessState.READY,
                )
            else:
                replacement = _ReadinessEntry(
                    alias=work.alias,
                    revision=work.revision,
                    state=RootReadinessState.FAILED,
                    reason=self._FAILED_REASON,
                )
            self._readiness[(work.workspace_id, work.root)] = replacement

    def _discard_workspace_readiness_locked(self, workspace_id: str) -> None:
        for key in tuple(self._readiness):
            if key[0] == workspace_id:
                self._readiness.pop(key, None)


_WORKSPACE_COMPOSITE_CONSENT_OWNER = ChangeReviewConsentService
_WORKSPACE_COMPOSITE_CONSENT_LOOKUP = inspect.getattr_static(
    ChangeReviewConsentService, "__getattribute__"
)
_WORKSPACE_COMPOSITE_CONSENT_REGISTRY_DESCRIPTOR = inspect.getattr_static(
    ChangeReviewConsentService, "_registry", None
)
_WORKSPACE_COMPOSITE_CONSENT_SOURCE = (
    globals(),
    __file__,
    __spec__,
    getattr(__spec__, "origin", None),
    tuple(
        (
            name,
            (
                function,
                function.__code__,
                function.__globals__,
                function.__defaults__,
                function.__kwdefaults__,
                tuple((function.__kwdefaults__ or {}).items()),
                function.__closure__,
                tuple(
                    (cell, cell.cell_contents) for cell in function.__closure__ or ()
                ),
            ),
        )
        for name in (
            "admit_turn",
            "status",
            "_admit_enabled_locked",
            "_current_bindings",
            "_schedule_locked",
        )
        for function in (vars(ChangeReviewConsentService)[name],)
    ),
    (
        _default_capability_reader,
        _default_capability_reader.__code__,
        _default_capability_reader.__globals__,
        _default_capability_reader.__defaults__,
        _default_capability_reader.__kwdefaults__,
        tuple((_default_capability_reader.__kwdefaults__ or {}).items()),
        _default_capability_reader.__closure__,
        tuple(
            (cell, cell.cell_contents)
            for cell in _default_capability_reader.__closure__ or ()
        ),
    ),
)


def _workspace_consent_current(service, registry_module):
    defining, path, spec, origin, methods, capability = (
        _WORKSPACE_COMPOSITE_CONSENT_SOURCE
    )
    owner = _WORKSPACE_COMPOSITE_CONSENT_OWNER
    return (
        ChangeReviewConsentService is owner
        and type(service) is owner
        and inspect.getattr_static(owner, "__getattribute__")
        is _WORKSPACE_COMPOSITE_CONSENT_LOOKUP
        and inspect.getattr_static(owner, "_registry", None)
        is _WORKSPACE_COMPOSITE_CONSENT_REGISTRY_DESCRIPTOR
        and globals() is defining
        and __file__ == path
        and __spec__ is spec
        and getattr(__spec__, "origin", None) == origin
        and service._capability_reader is capability[0]
        and _default_capability_reader is capability[0]
        and registry_module._workspace_function_current(capability)
        and all(
            vars(owner).get(name) is record[0]
            and name not in vars(service)
            and registry_module._workspace_function_current(record)
            for name, record in methods
        )
    )
