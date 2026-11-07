"""Typed state and app-owned readiness for workspace Change Review."""

from __future__ import annotations

import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum
from queue import Empty, Full, Queue
from threading import RLock, Thread
from typing import Protocol

from tldw_chatbook.Agents.execution_capacity import CapacityRefused, WorkOrigin
from tldw_chatbook.DB.base_db import operation_owned_connection
from tldw_chatbook.Workspaces.models import RuntimeBindingStatus


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
    binding: object
    origin: WorkOrigin
    token: object = field(default_factory=object)
    source: object = None
    ticket: object = None


@dataclass(frozen=True, slots=True)
class _ReadinessEntry:
    alias: str
    revision: str
    state: RootReadinessState
    reason: str = ""
    work_token: object = None


def _default_capability_reader() -> ChangeReviewCapability:
    from tldw_chatbook.Workspaces.change_bounds import read_change_review_capability

    return read_change_review_capability()


def _default_root_initializer(root: str, *, shadow_service=None) -> None:
    from tldw_chatbook.Workspaces.change_turn_tracker import initialize_shadow_root

    initialize_shadow_root(root, service=shadow_service)


class ChangeReviewConsentService:
    """Own consent linearization and bounded root initialization.

    Memory-only fences linearize admission, toggles and readiness publication.
    Registry reads, durable CAS and callbacks run outside the bookkeeping lock.
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
        self._custom_capability_reader = capability_reader is not None
        self._lock = RLock()
        self._queue: Queue[_InitializationWork] = Queue(maxsize=queue_capacity)
        self._readiness: dict[tuple[str, str], _ReadinessEntry] = {}
        self._generation = 0
        self._admission_revision = 0
        self._epochs: dict[str, int] = {}
        self._toggling: set[str] = set()
        self._native = None
        self._native_expected = False
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

    @property
    def admission_revision(self) -> int:
        """A number that moves whenever ``admit_turn``'s answer may change.

        It moves when consent is toggled, when a root finishes (or fails)
        preparation, when readiness is discarded or a ready root is
        rescheduled, and at shutdown -- never on a read that changes nothing,
        including the first read that only schedules preparation. Reading it
        takes no lock and touches no database, so the UI loop can use it
        where ``admit_turn`` (a registry read) is too slow: the Console
        Environment panel compares it to drop a landing whose root changed
        after its worker read the root (TASK-33620.15.1.6). Folder-binding
        changes are the registry's ``mutation_generation``; a consent change
        written by another process, or a folder removed on disk, moves
        neither.
        """
        return self._admission_revision


    def _observe(self, workspace_id):
        with self._lock:
            if self._disposed or workspace_id in self._toggling:
                return None
            stamp = (self._generation, self._epochs.get(workspace_id, 0))
        with operation_owned_connection(getattr(self._registry, "db", None)):
            capability = self._capability_reader()
            consent = self._registry.read_change_review_consent(workspace_id)
            bindings = (
                tuple(
                    deepcopy(binding)
                    for binding in self._current_bindings(workspace_id)
                )
                if capability.state is ChangeReviewState.ENABLED
                and consent.state is ChangeReviewState.ENABLED
                else ()
            )
        return stamp, capability, consent, bindings

    def _observation_current(self, workspace_id, stamp):
        return (
            not self._disposed
            and workspace_id not in self._toggling
            and stamp == (self._generation, self._epochs.get(workspace_id, 0))
        )

    def admit_turn(self, workspace_id: str) -> ChangeReviewAdmission:
        """Return current ready roots, scheduling missing roots without waiting."""
        return self._admit(workspace_id, origin=WorkOrigin.AUTOMATIC)

    def capture_turn(self, workspace_id: str) -> ChangeReviewAdmission:
        """Read the same consent-bound roots without scheduling initialization."""
        return self._admit(
            workspace_id, origin=WorkOrigin.AUTOMATIC, schedule_missing=False
        )

    def _admit(self, workspace_id, *, origin, retry=False, schedule_missing=True):
        observed = self._observe(workspace_id)
        if observed is None:
            return 0 if retry else ChangeReviewAdmission()
        stamp, capability, consent, bindings = observed
        pending = []
        with self._lock:
            if (
                not self._observation_current(workspace_id, stamp)
                or capability.state is not ChangeReviewState.ENABLED
                or consent.state is not ChangeReviewState.ENABLED
            ):
                return 0 if retry else ChangeReviewAdmission()
            for binding in bindings:
                key = (workspace_id, binding.locator)
                entry = self._readiness.get(key)
                missing = entry is None or entry.revision != consent.revision
                if schedule_missing and (
                    (retry and not missing and entry.state is RootReadinessState.FAILED)
                    or (not retry and missing)
                ):
                    work = _InitializationWork(
                        workspace_id,
                        binding.locator,
                        binding.binding_id,
                        consent.revision,
                        self._generation,
                        binding,
                        origin,
                    )
                    if entry is not None and entry.state is RootReadinessState.READY:
                        self._admission_revision += 1
                    self._readiness[key] = _ReadinessEntry(
                        work.alias,
                        work.revision,
                        RootReadinessState.PREPARING,
                        self._PREPARING_REASON,
                        work.token,
                    )
                    pending.append(work)
        scheduled = sum(self._schedule(work) for work in pending)
        if retry:
            return scheduled
        with self._lock:
            if not self._observation_current(workspace_id, stamp):
                return ChangeReviewAdmission()
            ready, aliases, skipped = [], [], []
            for binding in bindings:
                entry = self._readiness.get((workspace_id, binding.locator))
                if entry is None or entry.revision != consent.revision:
                    continue
                if entry.state is RootReadinessState.READY:
                    ready.append(binding.locator)
                    aliases.append(binding.binding_id)
                else:
                    skipped.append(SkippedReviewRoot(binding.binding_id, entry.reason))
            return ChangeReviewAdmission(tuple(ready), tuple(skipped), tuple(aliases))

    def status(self, workspace_id: str) -> ChangeReviewStatus:
        """Read alias-only readiness without holding the lock across source reads."""
        observed = self._observe(workspace_id)
        if observed is None:
            return ChangeReviewStatus(
                ChangeReviewCapability(ChangeReviewState.UNAVAILABLE),
                ChangeReviewConsent(ChangeReviewState.UNAVAILABLE),
            )
        stamp, capability, consent, bindings = observed
        with self._lock:
            roots = []
            if self._observation_current(workspace_id, stamp):
                for binding in bindings:
                    entry = self._readiness.get((workspace_id, binding.locator))
                    if entry is not None and entry.revision == consent.revision:
                        roots.append(
                            RootReadiness(binding.binding_id, entry.state, entry.reason)
                        )
            return ChangeReviewStatus(capability, consent, tuple(roots))

    def toggle(
        self, workspace_id: str, *, expected: ChangeReviewConsent, enabled: bool
    ) -> ChangeReviewConsent:
        """Fence stale reads while the original registry performs its durable CAS."""
        with self._lock:
            if self._disposed:
                raise RuntimeError("Change Review service is shut down")
            if workspace_id in self._toggling:
                raise ChangeReviewStateConflict("Change Review consent is changing")
            self._toggling.add(workspace_id)
            self._epochs[workspace_id] = self._epochs.get(workspace_id, 0) + 1
        try:
            with operation_owned_connection(getattr(self._registry, "db", None)):
                capability = self._capability_reader()
                if capability.state is not ChangeReviewState.ENABLED:
                    raise RuntimeError("Change Review capability is unavailable")
                committed = self._registry.compare_and_set_change_review_consent(
                    workspace_id,
                    expected=expected,
                    enabled=enabled,
                )
            with self._lock:
                self._discard_workspace_readiness_locked(workspace_id)
                self._admission_revision += 1
        finally:
            with self._lock:
                self._toggling.discard(workspace_id)
                self._epochs[workspace_id] += 1
        if enabled:
            self._admit(workspace_id, origin=WorkOrigin.MANUAL)
        return committed

    def binding_added(self, workspace_id: str, binding: _FolderBinding) -> None:
        """Schedule automatic preparation after the original durable binding add."""
        self.admit_turn(workspace_id)

    def retry_failed_roots(self, workspace_id: str) -> int:
        """Retry unsuccessful roots once using the explicit Settings cause."""
        return self._admit(workspace_id, origin=WorkOrigin.MANUAL, retry=True)

    def begin_close(self) -> None:
        """Seal ingress and conclusively discard queued, never-started work."""
        discarded = []
        with self._lock:
            if not self._disposed:
                self._disposed = True
                self._generation += 1
                self._admission_revision += 1
                self._readiness.clear()
            while True:
                try:
                    discarded.append(self._queue.get_nowait())
                except Empty:
                    break
                else:
                    self._queue.task_done()
        for work in discarded:
            if work.ticket is not None:
                with work.ticket.activate():
                    work.ticket.returned(worker=True, handoff=True)

    def shutdown(self, *, timeout: float = 1.0) -> None:
        """Seal ingress and bounded-join; returning does not prove physical drain."""
        self.begin_close()
        deadline = time.monotonic() + max(0.0, timeout)
        for worker in self._workers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if worker.ident is not None:
                worker.join(remaining)

    def wait_closed(self) -> None:
        """Wait for actual callback, readiness handoff and worker retirement."""
        self.begin_close()
        for worker in self._workers:
            if worker.ident is not None:
                worker.join()

    def _current_bindings(self, workspace_id: str) -> tuple[_FolderBinding, ...]:
        return tuple(
            binding
            for binding in self._registry.list_folder_bindings(workspace_id)
            if binding.status is RuntimeBindingStatus.READY
        )

    def _pending(self, work):
        current = self._readiness.get((work.workspace_id, work.root))
        return (
            not self._disposed
            and work.generation == self._generation
            and current is not None
            and current.work_token is work.token
            and current.state is RootReadinessState.PREPARING
        )

    def _schedule(self, work):
        ticket = None
        queued = False
        try:
            if self._native_expected:
                if self._native is None:
                    raise ValueError("native initialization source unavailable")
                work = replace(work, source=self._native.capture(work))
                if work.source.ordinary_tree is not None:
                    ticket = work.source.admit()
                    work = replace(work, ticket=ticket)
            with self._lock:
                if not self._pending(work):
                    return False
                if not self._workers_started:
                    for worker in self._workers:
                        worker.start()
                    self._workers_started = True
                self._queue.put_nowait(work)
                queued = True
            return True
        except Exception as error:  # noqa: BLE001 -- scheduling failure becomes failed readiness
            self._fail(work, busy=isinstance(error, (Full, CapacityRefused)))
            return False
        finally:
            if ticket is not None and not queued:
                with ticket.activate():
                    ticket.returned(worker=True, handoff=True)

    def _work_current(self, work):
        with self._lock:
            if not self._pending(work):
                return False
        observed = self._observe(work.workspace_id)
        if observed is None:
            return False
        stamp, capability, consent, bindings = observed
        with self._lock:
            return (
                self._pending(work)
                and self._observation_current(work.workspace_id, stamp)
                and capability.state is ChangeReviewState.ENABLED
                and consent.state is ChangeReviewState.ENABLED
                and consent.revision == work.revision
                and work.binding in bindings
            )

    def _worker_loop(self) -> None:
        while True:
            with self._lock:
                if self._disposed:
                    return
            try:
                work = self._queue.get(timeout=0.05)
            except Empty:
                continue
            ticket = None
            try:
                if work.source is not None:
                    ticket = work.ticket or work.source.admit()
                    with ticket.activate():
                        try:
                            self._run_work(work)
                        finally:
                            ticket.returned(worker=True, handoff=True)
                elif self._native_expected:
                    raise ValueError("native initialization source unavailable")
                else:
                    with operation_owned_connection(
                        getattr(self._registry, "db", None)
                    ):
                        self._run_work(work)
            except Exception as error:  # noqa: BLE001 -- one failed job must not kill the bounded worker
                self._fail(work, busy=isinstance(error, CapacityRefused))
            finally:
                self._queue.task_done()

    def _run_work(self, work):
        error = None
        try:
            if not self._work_current(work):
                raise ValueError("initialization consent changed")
            if work.source is None:
                self._initialize_root(work.root)
            else:
                work.source.initialize()
        except Exception as caught:  # noqa: BLE001 -- preserve initializer failure for its handoff
            error = caught
        self._complete(work, error)

    def _complete(self, work, error):
        if not self._work_current(work):
            self._fail(work)
            return
        # Handoff and consent reads may block; validate at the actual consumer.
        if work.source is not None:
            try:
                work.source.check()
            except Exception as caught:  # noqa: BLE001 -- changed authority cannot publish READY
                error = caught
        with self._lock:
            if not self._pending(work):
                return
            if work.workspace_id in self._toggling:
                error = ValueError("initialization consent changed")
            # Acquiring this lock can wait after external source validation.
            if work.source is not None and not work.source.issuer.identity_current():
                error = ValueError("initialization source changed")
            self._admission_revision += 1
            self._readiness[(work.workspace_id, work.root)] = _ReadinessEntry(
                work.alias,
                work.revision,
                RootReadinessState.READY
                if error is None
                else RootReadinessState.FAILED,
                "" if error is None else self._FAILED_REASON,
            )

    def _fail(self, work, *, busy=False):
        with self._lock:
            if self._pending(work):
                self._admission_revision += 1
                self._readiness[(work.workspace_id, work.root)] = _ReadinessEntry(
                    work.alias,
                    work.revision,
                    RootReadinessState.FAILED,
                    self._QUEUE_FULL_REASON if busy else self._FAILED_REASON,
                )

    def _discard_workspace_readiness_locked(self, workspace_id: str) -> None:
        for key in tuple(self._readiness):
            if key[0] == workspace_id:
                self._readiness.pop(key, None)
                self._admission_revision += 1


_NATIVE_ORDINARY_METADATA_CLASS = ChangeReviewConsentService
_NATIVE_ORDINARY_METADATA_METHODS = tuple(
    (
        name,
        getattr(ChangeReviewConsentService, name),
        getattr(ChangeReviewConsentService, name).__code__,
    )
    for name in (
        "status",
        "capture_turn",
        "_admit",
        "_observe",
        "_observation_current",
        "_current_bindings",
    )
)
_NATIVE_ORDINARY_CAPABILITY_READER = (
    _default_capability_reader,
    _default_capability_reader.__code__,
)


def _ordinary_metadata_sources(service):
    """Return only static original-source checks; never invoke a supplied reader."""
    from inspect import getattr_static

    from . import change_bounds, registry_service

    if type(service) is not _NATIVE_ORDINARY_METADATA_CLASS:
        return (), False
    registry = getattr_static(service, "_registry", None)
    if type(registry) is not registry_service.LocalWorkspaceRegistryService:
        return (), False
    reader, code = _NATIVE_ORDINARY_CAPABILITY_READER
    records = [(service, "_capability_reader", reader, code)]
    records.extend((service, *entry) for entry in _NATIVE_ORDINARY_METADATA_METHODS)
    records.extend(
        (registry, *entry) for entry in registry_service._NATIVE_ORDINARY_REVIEW_READS
    )
    records.extend(
        (change_bounds, *entry)
        for entry in change_bounds._NATIVE_ORDINARY_CAPABILITY_READS
    )
    return tuple(records), all(
        getattr_static(obj, name, None) is original
        and original.__code__ is implementation
        for obj, name, original, implementation in records
    )
