"""Session-owned review files and atomic append ledger (ADR-281).

Host APIs own paths and bindings. Wire references never grant authority. This is
an in-memory session ledger, not a crash-recovery protocol. Lock order is tab,
context, then scratch generation guard. Scratch cleanup callbacks run only after
leases drain and outside scratch/receipt locks.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import threading
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Callable, ContextManager, Iterator, Literal

from pydantic import Field

from tldw_chatbook.Agents.code_review_models import (
    TAB_MAX_CONTEXTS,
    CheckActionRecord,
    CheckEvidence,
    CoverageRecord,
    Digest,
    DispositionRecord,
    FindingRecord,
    NonemptyText,
    ReviewLimits,
    ReviewRefused,
    ReviewRunBinding,
    ReviewWireModel,
    SourceEvidence,
    SourceLocation,
    WireTuple,
    canonical_record_bytes,
    validate_record_chunk,
)
from tldw_chatbook.Agents.run_context import current_run_id
from tldw_chatbook.Chat.console_scratch_space import (
    ConsoleScratchSnapshot,
    ConsoleScratchSpaceManager,
    ConsoleScratchSpaceUnavailable,
)


class CapturedSource(ReviewWireModel):
    """Host-only immutable file descriptor; never serialize its physical path."""

    path: Path = Field(repr=False)
    digest: Digest
    line_count: int = Field(ge=0)


class SubmissionAck(ReviewWireModel):
    accepted_ids: WireTuple[NonemptyText]
    submission_revision: int = Field(ge=0)
    cursor: NonemptyText


class ReleaseReceipt(ReviewWireModel):
    review_id: NonemptyText
    revision: int = Field(ge=0)
    tombstoned: bool
    cleaned: bool


@dataclass(frozen=True)
class ReviewUsage:
    """Content-free retained and reserved tab occupancy."""

    retained_bytes: int
    reserved_bytes: int
    retained_files: int
    reserved_files: int
    retained_contexts: int


@dataclass
class _File:
    path: Path
    category: str
    participant_id: str | None
    size: int = 0
    identity: tuple[int, int] | None = None


@dataclass
class _Stream:
    revision: int = 0
    submissions: dict[str, tuple[str, SubmissionAck]] = field(default_factory=dict)
    local_keys: set[str] = field(default_factory=set)
    run_id: str = ""
    raw_ref: str = field(default_factory=lambda: "raw-" + secrets.token_hex(16))
    raw_chunks: list[Path] = field(default_factory=list)
    capture_terminal: bool = False
    capture_receipt: str | None = None
    capture_problem: str = ""
    withheld: bool = False
    host_sequence: int = 0
    terminal_status: str | None = None
    terminal_receipt: str = ""
    coverage: tuple[Path, int] | None = None
    completion_requested: bool = False
    rejected: dict[str, str] = field(default_factory=dict)
    routes: list[dict] = field(default_factory=list)
    verification: object | None = field(default=None, repr=False)


@dataclass
class ReviewContext:
    """Host-owned mutable context; later owners add their stage-specific state.

    Source capture populates ``sources`` only after bounded materialization.
    ``original_index`` maps host original IDs to immutable chunk record offsets.
    Completion requests and native/capture receipts have separate sequences.
    """

    review_id: str
    session_id: str
    conversation_id: str | None
    target_digest: str
    source_epoch: int
    scratch: ConsoleScratchSnapshot = field(repr=False)
    bundle_path: Path = field(repr=False)
    manifest_path: Path = field(repr=False)
    bundle_identity: tuple[int, int] = field(repr=False)
    revision: int = 0
    state: str = "live"
    publication_sequence: int = 0
    source_state: object | None = field(default=None, repr=False)
    preparation: object | None = field(default=None, repr=False)
    launch_claims: dict[str, object] = field(default_factory=dict, repr=False)
    continuation_status: str | None = None
    presented_attempts: set = field(default_factory=set)
    primary_authorities: dict = field(default_factory=dict, repr=False)
    stage_slots: dict = field(default_factory=dict, repr=False)
    attempt_slots: dict = field(default_factory=dict, repr=False)
    scheduled_attempts: dict = field(default_factory=dict, repr=False)
    source_finalized: bool = False
    sources: dict[str, CapturedSource] = field(default_factory=dict, repr=False)
    original_index: dict[str, tuple[Path, int]] = field(
        default_factory=dict, repr=False
    )
    report_cursors: dict[str, Path] = field(default_factory=dict, repr=False)
    report_pages: dict[str, Path] = field(default_factory=dict, repr=False)
    finding_sequences: dict[str, int] = field(default_factory=dict, repr=False)
    checks: dict = field(default_factory=dict, repr=False)
    check_actors: dict[str, str] = field(default_factory=dict, repr=False)
    check_streams: dict = field(default_factory=dict, repr=False)
    check_proposal_ids: set[str] = field(default_factory=set, repr=False)
    check_receipt_ids: set[str] = field(default_factory=set, repr=False)
    verification_unit_ids: set[str] = field(default_factory=set, repr=False)
    active_attempts: dict[str, ReviewRunBinding] = field(
        default_factory=dict, repr=False
    )
    streams: dict[tuple[str, str], _Stream] = field(default_factory=dict, repr=False)
    _files: dict[Path, _File] = field(default_factory=dict, repr=False)
    _reservations: list[ReviewReservation] = field(default_factory=list, repr=False)
    _leases: int = 0
    _head_chunk: str | None = None
    _release_revision: int | None = None
    _cleaned: bool = False
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


@dataclass
class _Tab:
    limits: ReviewLimits
    contexts: dict[str, ReviewContext] = field(default_factory=dict)
    release_receipts: OrderedDict[str, ReleaseReceipt] = field(
        default_factory=OrderedDict
    )
    issued_contexts: int = 0
    runs: dict[str, ReviewRunBinding] = field(default_factory=dict)
    retired_attempts: set[tuple[str, str, str]] = field(default_factory=set)
    lock: threading.RLock = field(default_factory=threading.RLock)
    closed: threading.Event = field(default_factory=threading.Event)


_REGISTRY_LOCK = threading.Lock()
_TABS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_ROLE_KINDS = {
    "reviewer": {"findings", "coverage", "reviewer_complete"},
    "verifier": {"dispositions", "check_proposals", "verifier_complete"},
    "primary": {"check_actions"},
}


def _discard_payload(context: ReviewContext) -> None:
    context.state = "tombstoned"
    context.sources.clear()
    context.source_state = None
    context.launch_claims.clear()
    context.presented_attempts.clear()
    context.primary_authorities.clear()
    context.stage_slots.clear()
    context.attempt_slots.clear()
    context.scheduled_attempts.clear()
    context.preparation = None
    context.original_index.clear()
    context.finding_sequences.clear()
    context.report_cursors.clear()
    context.report_pages.clear()
    for check in context.checks.values():
        check.changed.set()
    context.checks.clear()
    context.check_actors.clear()
    context.check_streams.clear()
    context.check_proposal_ids.clear()
    context.check_receipt_ids.clear()
    context.verification_unit_ids.clear()
    context.active_attempts.clear()
    context.streams.clear()
    context._head_chunk = None


def _prepare_generation_cleanup(tab: _Tab, snapshot: ConsoleScratchSnapshot) -> None:
    # Runs only on the native cleanup worker after all generation leases drain.
    with tab.lock:
        if ReviewStore._matches(snapshot.root, snapshot.identity):
            for context in tab.contexts.values():
                ReviewStore._cleanup(context)


def _retire_generation(
    manager_ref: weakref.ReferenceType, token: str, tab: _Tab, outcome: str
) -> None:
    # Native leases have settled before the scratch owner delivers this callback.
    with tab.lock:
        tab.closed.set()
        for context in tab.contexts.values():
            _discard_payload(context)
            if outcome == "deleted" and not context._files:
                context._cleaned = True
        tab.contexts = {
            key: context
            for key, context in tab.contexts.items()
            if not context._cleaned
        }
        tab.runs.clear()
        tab.retired_attempts.clear()
        tab.release_receipts.clear()
    with _REGISTRY_LOCK:
        manager = manager_ref()
        generations = _TABS.get(manager) if manager is not None else None
        if generations is not None and generations.get(token) is tab:
            generations.pop(token)
            if not generations:
                _TABS.pop(manager, None)


def _counts(*values: int) -> None:
    if any(type(value) is not int or value < 0 for value in values):
        raise ReviewRefused("invalid_storage_counts")


class ReviewReservation:
    """Bounded host writer holding scratch until its actual native work settles.

    Keep the context manager open in the worker, including after a caller stops
    waiting. Exiting discards uncommitted files, charges failed deletions and
    releases only unused allowance. Paths returned by ``write`` are host-only.
    """

    def __init__(
        self,
        store: ReviewStore,
        context: ReviewContext,
        source: int,
        report: int,
        files: int,
        participant_id: str | None,
        lease: ContextManager[Path],
    ) -> None:
        self._store = store
        self._context = context
        self._source = source
        self._report = report
        self._file_limit = files
        self._participant = participant_id
        self._lease = lease
        self._files: list[_File] = []
        self._committed = False
        self._closed = False

    def __enter__(self) -> ReviewReservation:
        if self._closed:
            raise ReviewRefused("reservation_closed")
        return self

    def _actual(self) -> tuple[int, int, int]:
        files = [
            item for item in self._files if self._context._files.get(item.path) is item
        ]
        return (
            sum(item.size for item in files if item.category == "source"),
            sum(item.size for item in files if item.category == "report"),
            len(files),
        )

    def _remaining(self) -> tuple[int, int, int]:
        source, report, files = self._actual()
        return (
            max(0, self._source - source),
            max(0, self._report - report),
            max(0, self._file_limit - files),
        )

    def write(
        self, data: bytes, *, category: Literal["source", "report"] = "source"
    ) -> Path:
        """Materialize bytes privately, checking remaining quota before growth."""
        if not isinstance(data, bytes) or category not in ("source", "report"):
            raise ReviewRefused("invalid_storage_write")
        with self._store._tab.lock:
            self._store._live(self._context)
            if self._closed or self._committed:
                raise ReviewRefused("reservation_closed")
            source, report, files = self._remaining()
            if files < 1 or len(data) > (source if category == "source" else report):
                raise ReviewRefused("reservation_exceeded")
            path = self._context.bundle_path / secrets.token_hex(16)
            item = _File(path, category, self._participant, len(data))
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            self._files.append(item)
            self._context._files[path] = item
            try:
                remaining = memoryview(data)
                while remaining:
                    written = os.write(fd, remaining)
                    if written <= 0:
                        raise OSError("short review write")
                    remaining = remaining[written:]
            finally:
                try:
                    metadata = os.fstat(fd)
                    item.size = metadata.st_size
                    item.identity = (metadata.st_dev, metadata.st_ino)
                finally:
                    os.close(fd)
            return path

    def commit(
        self, actual_source_bytes: int, actual_report_bytes: int, actual_files: int
    ) -> None:
        """Retain the writer's files after verifying the declared actual totals."""
        _counts(actual_source_bytes, actual_report_bytes, actual_files)
        with self._store._tab.lock:
            self._store._live(self._context)
            if self._closed or self._committed:
                raise ReviewRefused("reservation_closed")
            for item in self._files:
                metadata = item.path.lstat()
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or (metadata.st_dev, metadata.st_ino) != item.identity
                ):
                    raise ReviewRefused("invalid_storage_file")
                item.size = metadata.st_size
            source, report, files = self._actual()
            if (
                source > self._source
                or report > self._report
                or files > self._file_limit
            ):
                raise ReviewRefused("reservation_exceeded")
            if (source, report, files) != (
                actual_source_bytes,
                actual_report_bytes,
                actual_files,
            ):
                raise ReviewRefused("reservation_actual_mismatch")
            self._committed = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        with self._store._tab.lock:
            if self._closed:
                return
            if not self._committed:
                try:
                    with self._store._manager.live_guard(self._store._scratch):
                        for item in self._files:
                            self._store._delete(self._context, item)
                except ConsoleScratchSpaceUnavailable:
                    pass
            self._context._reservations.remove(self)
            self._context._leases -= 1
            self._closed = True
        self._lease.__exit__(exc_type, exc, traceback)


class ReviewStore:
    """One quota ledger per live scratch generation, shared across host handles."""

    def __init__(
        self,
        manager: ConsoleScratchSpaceManager,
        session_id: str,
        *,
        limits: ReviewLimits = ReviewLimits(),
    ) -> None:
        self._manager = manager
        self._session_id = session_id
        self._scratch = manager.snapshot(session_id)
        self._cleanup_receipt = manager.observe_cleanup(self._scratch)
        with _REGISTRY_LOCK:
            generations = _TABS.setdefault(manager, {})
            created = self._scratch.token not in generations
            self._tab = generations.setdefault(self._scratch.token, _Tab(limits))
            if self._tab.limits != limits:
                raise ReviewRefused("storage_limits_conflict")
        if created:
            manager_ref, token, tab = (
                weakref.ref(manager),
                self._scratch.token,
                self._tab,
            )
            snapshot = self._scratch
            self._cleanup_receipt.add_cleanup_callback(
                lambda: _prepare_generation_cleanup(tab, snapshot)
            )
            self._cleanup_receipt.add_done_callback(
                lambda outcome: _retire_generation(manager_ref, token, tab, outcome)
            )

    def _live(self, context: ReviewContext | None = None) -> None:
        if self._tab.closed.is_set() or not self._manager.is_live(self._scratch):
            self._tab.closed.set()
            for item in self._tab.contexts.values():
                item.state = "tombstoned"
                for check in item.checks.values():
                    check.changed.set()
                    if check.result_reservation is not None:
                        check.result_reservation.__exit__(None, None, None)
                        check.result_reservation = None
            raise ReviewRefused("review_expired")
        if context is not None and context.state != "live":
            raise ReviewRefused("review_expired")

    def _context(self, review_id: str) -> ReviewContext:
        context = self._tab.contexts.get(review_id)
        if context is None:
            raise ReviewRefused(
                "review_expired"
                if review_id in self._tab.release_receipts
                else "unknown_review"
            )
        return context

    def create(
        self,
        *,
        target_digest: str,
        source_epoch: int = 0,
        conversation_id: str | None = None,
        admit_id: Callable[[str], object] | None = None,
    ) -> ReviewContext:
        """Mint a new context identity; retired IDs are never caller-selectable."""
        _counts(source_epoch)
        if (
            not isinstance(target_digest, str)
            or len(target_digest) != 64
            or any(c not in "0123456789abcdef" for c in target_digest)
        ):
            raise ReviewRefused("invalid_target_digest")
        with self._tab.lock:
            self._live()
            if self.usage().retained_contexts >= self._tab.limits.tab_max_contexts:
                raise ReviewRefused("context_quota_exceeded")
            review_id = (
                f"review-{self._tab.issued_contexts + 1:x}-{secrets.token_hex(16)}"
            )
            if admit_id is not None:
                admit_id(review_id)
            self._tab.issued_contexts += 1
            try:
                with self._manager.lease(self._scratch) as root:
                    path = root / secrets.token_hex(16)
                    path.mkdir(mode=0o700)
                    metadata = path.lstat()
                    context = ReviewContext(
                        review_id,
                        self._session_id,
                        conversation_id,
                        target_digest,
                        source_epoch,
                        self._scratch,
                        path,
                        path / secrets.token_hex(16),
                        (metadata.st_dev, metadata.st_ino),
                    )
                    self._tab.contexts[review_id] = context
                    return context
            except ConsoleScratchSpaceUnavailable as exc:
                raise ReviewRefused("review_expired") from exc

    @contextmanager
    def lease(self, review_id: str) -> Iterator[ReviewContext]:
        """Hold a context and its scratch generation across actual host work."""
        with self._tab.lock:
            self._live()
            context = self._context(review_id)
            self._live(context)
            scratch_lease = self._manager.lease(self._scratch)
            try:
                scratch_lease.__enter__()
            except ConsoleScratchSpaceUnavailable as exc:
                raise ReviewRefused("review_expired") from exc
            context._leases += 1
        try:
            yield context
        finally:
            with self._tab.lock:
                context._leases -= 1
            scratch_lease.__exit__(None, None, None)

    def finalize_capture(
        self,
        review_id: str,
        *,
        target_digest: str,
        source_state: object,
        sources: dict[str, CapturedSource],
        reservation: ReviewReservation,
        actual_source_bytes: int,
        actual_report_bytes: int,
        actual_files: int,
    ) -> None:
        """Seal the initial host capture once, before publishing any run identity."""
        with self._tab.lock:
            context = self._context(review_id)
            self._live(context)
            if (
                context.source_finalized
                or context.active_attempts
                or context.streams
                or context.publication_sequence
                or context.sources
            ):
                raise ReviewRefused("source_already_published")
            if reservation._context is not context:
                raise ReviewRefused("review_scope_mismatch")
            if len(target_digest) != 64 or any(
                c not in "0123456789abcdef" for c in target_digest
            ):
                raise ReviewRefused("invalid_target_digest")
            try:
                with self._manager.live_guard(self._scratch):
                    reservation.commit(
                        actual_source_bytes, actual_report_bytes, actual_files
                    )
                    context.target_digest = target_digest
                    context.sources.update(sources)
                    context.source_state = source_state
                    context.source_finalized = True
            except ConsoleScratchSpaceUnavailable as exc:
                raise ReviewRefused("review_expired") from exc

    def bind_attempt(self, run_id: str, binding: ReviewRunBinding) -> None:
        """Associate a host-validated run; replacing a slot revokes its old attempt."""
        binding = ReviewRunBinding.model_validate(binding)
        if not isinstance(run_id, str) or not run_id:
            raise ReviewRefused("invalid_run_id")
        with self._tab.lock:
            context = self._context(binding.review_id)
            self._live(context)
            existing = self._tab.runs.get(run_id)
            if existing is not None:
                if existing != binding:
                    raise ReviewRefused("run_binding_conflict")
                self._bound(context, run_id)
                return
            if binding.target_digest != context.target_digest:
                raise ReviewRefused("target_digest_mismatch")
            if binding.source_epoch != context.source_epoch:
                raise ReviewRefused("source_epoch_mismatch")
            identity = (binding.review_id, binding.slot_id, binding.attempt_id)
            if identity in self._tab.retired_attempts:
                raise ReviewRefused("stale_attempt")
            previous = context.active_attempts.get(binding.slot_id)
            if previous is not None and previous != binding:
                if previous.attempt_id == binding.attempt_id:
                    raise ReviewRefused("attempt_binding_conflict")
                self._tab.retired_attempts.add(
                    (previous.review_id, previous.slot_id, previous.attempt_id)
                )
                context.revision += 1
            context.active_attempts[binding.slot_id] = binding
            context.streams.setdefault(
                (binding.slot_id, binding.attempt_id), _Stream(run_id=run_id)
            )
            self._tab.runs[run_id] = binding

    def bind_check_actor(
        self, primary_run_id: str, review_id: str, verifier_attempt_id: str
    ) -> None:
        """Bind a real primary to one live verifier; never impersonate a participant."""
        with self._tab.lock, self.lease(review_id) as context:
            actor = self._bound(context, primary_run_id)
            if actor.role != "primary":
                raise ReviewRefused("role_denied")
            verifier = next(
                (
                    b
                    for b in context.active_attempts.values()
                    if b.role == "verifier" and b.attempt_id == verifier_attempt_id
                ),
                None,
            )
            if (
                verifier is None
                or context.streams[
                    (verifier.slot_id, verifier.attempt_id)
                ].terminal_status
                is not None
            ):
                raise ReviewRefused("verification_attempt_closed")
            previous = context.check_actors.get(primary_run_id)
            if previous is not None and previous != verifier_attempt_id:
                raise ReviewRefused("check_actor_conflict")
            context.check_actors[primary_run_id] = verifier_attempt_id
            context.check_streams.setdefault(
                (primary_run_id, verifier_attempt_id), _Stream(run_id=primary_run_id)
            )

    def submission_stream(self, context, binding, kind):
        if kind != "check_actions":
            return context.streams[(binding.slot_id, binding.attempt_id)]
        run_id = current_run_id()
        attempt = context.check_actors.get(run_id)
        if binding.role != "primary" or attempt is None:
            raise ReviewRefused("check_actor_unbound")
        if any(
            stream.terminal_status is not None
            for (slot, current_attempt), stream in context.streams.items()
            if current_attempt == attempt
        ):
            raise ReviewRefused("verification_attempt_closed")
        return context.check_streams[(run_id, attempt)]

    def retire_unstarted_attempt(self, run_id: str, binding: ReviewRunBinding) -> None:
        """Retire an exact empty association after proven native non-start."""
        with self._tab.lock:
            context = self._context(binding.review_id)
            if (
                self._tab.runs.get(run_id) != binding
                or context.active_attempts.get(binding.slot_id) != binding
            ):
                raise ReviewRefused("run_binding_conflict")
            stream = context.streams[(binding.slot_id, binding.attempt_id)]
            if (
                stream.revision
                or stream.submissions
                or stream.host_sequence
                or stream.raw_chunks
            ):
                raise ReviewRefused("review_attempt_has_publications")
            self._tab.retired_attempts.add(
                (binding.review_id, binding.slot_id, binding.attempt_id)
            )
            del self._tab.runs[run_id]
            del context.active_attempts[binding.slot_id]
            del context.streams[(binding.slot_id, binding.attempt_id)]

    def _bound(
        self, context: ReviewContext, run_id: str | None = None
    ) -> ReviewRunBinding:
        binding = self._tab.runs.get(current_run_id() if run_id is None else run_id)
        if binding is None:
            raise ReviewRefused("run_not_bound")
        if binding.review_id != context.review_id:
            raise ReviewRefused("review_scope_mismatch")
        if binding.target_digest != context.target_digest:
            raise ReviewRefused("target_digest_mismatch")
        if binding.source_epoch != context.source_epoch:
            raise ReviewRefused("source_epoch_mismatch")
        if context.active_attempts.get(binding.slot_id) != binding:
            raise ReviewRefused("stale_attempt")
        return binding

    def _totals(
        self, context: ReviewContext, participant: str | None = None
    ) -> tuple[int, int, int]:
        files = tuple(context._files.values())
        if participant is not None:
            files = tuple(item for item in files if item.participant_id == participant)
        source = sum(item.size for item in files if item.category == "source")
        report = sum(item.size for item in files if item.category == "report")
        count = len(files)
        for reservation in context._reservations:
            if participant is None or reservation._participant == participant:
                rs, rr, rf = reservation._remaining()
                source, report, count = source + rs, report + rr, count + rf
        return source, report, count

    def reserve(
        self,
        review_id: str,
        *,
        source_bytes: int,
        report_bytes: int,
        files: int,
        participant_id: str | None = None,
    ) -> ReviewReservation:
        """Atomically reserve peak physical occupancy before starting native writes."""
        _counts(source_bytes, report_bytes, files)
        with self._tab.lock:
            context = self._context(review_id)
            self._live(context)
            usage = self.usage()
            source, report, count = self._totals(context)
            limits = self._tab.limits
            if (
                source + source_bytes > limits.source_max_bytes
                or report + report_bytes > limits.reports_max_bytes
                or source + report + source_bytes + report_bytes
                > limits.review_max_bytes
                or count + files > limits.review_max_files
                or usage.retained_bytes
                + usage.reserved_bytes
                + source_bytes
                + report_bytes
                > limits.tab_max_bytes
                or (
                    participant_id is not None
                    and self._totals(context, participant_id)[1] + report_bytes
                    > limits.participant_max_bytes
                )
            ):
                raise ReviewRefused("storage_quota_exceeded")
            scratch_lease = self._manager.lease(self._scratch)
            try:
                scratch_lease.__enter__()
            except ConsoleScratchSpaceUnavailable as exc:
                raise ReviewRefused("review_expired") from exc
            reservation = ReviewReservation(
                self,
                context,
                source_bytes,
                report_bytes,
                files,
                participant_id,
                scratch_lease,
            )
            context._reservations.append(reservation)
            context._leases += 1
            return reservation

    @staticmethod
    def _validate_references(context: ReviewContext, records: tuple) -> None:
        def source(location: SourceLocation) -> None:
            captured = context.sources.get(location.source_ref)
            if captured is None or location.end_line > captured.line_count:
                raise ReviewRefused("invalid_source_reference")
            if (
                isinstance(location, SourceEvidence)
                and location.digest != captured.digest
            ):
                raise ReviewRefused("source_digest_mismatch")

        for record in records:
            if (
                isinstance(record, CheckActionRecord)
                and record.proposal_id not in context.check_proposal_ids
            ):
                raise ReviewRefused("invalid_check_proposal")
            if isinstance(record, FindingRecord):
                for location in record.locations:
                    source(location)
            if isinstance(record, CoverageRecord):
                refs = (
                    *record.inspected_refs,
                    *(
                        gap.source_ref
                        for gap in record.gaps
                        if gap.source_ref is not None
                    ),
                )
                if any(ref not in context.sources for ref in refs):
                    raise ReviewRefused("invalid_source_reference")
            if isinstance(record, DispositionRecord):
                if record.verification_unit_id not in context.verification_unit_ids or (
                    record.duplicate_unit_id is not None
                    and record.duplicate_unit_id not in context.verification_unit_ids
                ):
                    raise ReviewRefused("invalid_verification_unit")
            for evidence in getattr(record, "evidence", ()):
                if isinstance(evidence, SourceEvidence):
                    source(evidence)
                elif (
                    isinstance(evidence, CheckEvidence)
                    and evidence.receipt_id not in context.check_receipt_ids
                ):
                    raise ReviewRefused("invalid_check_receipt")
            if any(
                unit not in context.verification_unit_ids
                for unit in getattr(record, "verification_unit_ids", ())
            ):
                raise ReviewRefused("invalid_verification_unit")

    def verification_ledger(self, context, binding):
        """Build and charge the private map once per explicitly bound attempt."""
        from .code_review_verification import (
            VerificationLedger,
            build_verification_units,
        )

        if binding.role != "verifier":
            raise ReviewRefused("role_denied")
        stream = context.streams[(binding.slot_id, binding.attempt_id)]
        if stream.verification is None:
            originals = tuple(
                {"original_id": key, "finding": self.read_record(context, location)}
                for key, location in context.original_index.items()
            )
            units = build_verification_units(originals, context.source_state.manifest)
            self.write_verification_state(
                context,
                {
                    "attempt_id": binding.attempt_id,
                    "original_to_unit": units.original_to_unit,
                    "public_units": units.public_units,
                },
            )
            stream.verification = VerificationLedger(units)
            context.verification_unit_ids.update(
                row["verification_unit_id"] for row in units.public_units
            )
        return stream.verification

    def write_verification_state(self, context, value):
        """Charge an immutable host checkpoint before publishing in-memory state."""
        payload = canonical_record_bytes(value)
        with self.reserve(
            context.review_id, source_bytes=0, report_bytes=len(payload), files=1
        ) as reservation:
            path = reservation.write(payload, category="report")
            reservation.commit(0, len(payload), 1)
        return path

    def finalize_verification(self, context, binding):
        from .code_review_verification import unresolved

        ledger = self.verification_ledger(context, binding)
        missing = {
            row["verification_unit_id"]: unresolved(
                row["verification_unit_id"],
                next(
                    (
                        check.transport_reason or check.receipt.evidence
                        for check in reversed(tuple(context.checks.values()))
                        if check.binding == binding
                        and row["verification_unit_id"]
                        in check.proposal.verification_unit_ids
                        and (
                            check.transport_reason
                            or (check.receipt and check.receipt.outcome != "executed")
                        )
                    ),
                    "verification_incomplete",
                ),
            )
            for row in ledger.units.public_units
            if row["verification_unit_id"] not in ledger.dispositions
        }
        if missing:
            self.write_verification_state(
                context, {"attempt_id": binding.attempt_id, "dispositions": missing}
            )
            ledger.dispositions.update(missing)

    def _publish(
        self,
        review_id: str,
        *,
        expected_revision: int,
        submission_id: str,
        records: list[dict],
        kind: str = "findings",
        admit_ack: Callable[[SubmissionAck], object] | None = None,
    ) -> SubmissionAck:
        """Append one immutable chunk, then atomically move its manifest pointer."""
        _counts(expected_revision)
        if not isinstance(submission_id, str) or not submission_id.strip():
            raise ReviewRefused("invalid_submission_id")
        with self._tab.lock, self.lease(review_id) as context, context._lock:
            binding = self._bound(context)
            if not isinstance(kind, str) or kind not in _ROLE_KINDS[binding.role]:
                raise ReviewRefused("role_kind_mismatch")
            stream = self.submission_stream(context, binding, kind)
            if stream.terminal_status is not None:
                raise ReviewRefused("review_run_settled")
            parsed = validate_record_chunk(kind, records)
            try:
                digest = hashlib.sha256(
                    canonical_record_bytes({"kind": kind, "records": records})
                ).hexdigest()
            except (ValueError, TypeError, UnicodeError) as exc:
                raise ReviewRefused("invalid_record") from exc
            stream = self.submission_stream(context, binding, kind)
            previous = stream.submissions.get(submission_id)
            if previous is not None:
                if previous[0] != digest:
                    raise ReviewRefused("submission_conflict")
                if admit_ack is not None:
                    admit_ack(previous[1])
                return previous[1]
            if expected_revision != stream.revision:
                refusal = ReviewRefused("submission_revision_conflict")
                refusal.submission_revision = stream.revision
                raise refusal
            if (
                kind == "dispositions"
                and not parsed
                and stream.rejected.get(
                    hashlib.sha256(submission_id.encode()).hexdigest()
                )
                != "disposition_conflict"
            ):
                raise ReviewRefused("invalid_record_count")
            if kind in {"reviewer_complete", "verifier_complete"} and (
                stream.rejected.keys()
                - {hashlib.sha256(submission_id.encode()).hexdigest()}
            ):
                raise ReviewRefused("unresolved_submission_rejections")
            self._validate_references(context, parsed)
            ledger = (
                self.verification_ledger(context, binding)
                if kind in {"dispositions", "verifier_complete"}
                else None
            )
            if kind == "verifier_complete":
                units = set(ledger.units.original_to_unit.values())
                if (
                    units != ledger.delivered
                    or units != ledger.dispositions.keys()
                    or ledger.request_gaps
                ):
                    raise ReviewRefused("verification_incomplete")
            if kind == "dispositions":
                unit_ids = [record.verification_unit_id for record in parsed]
                if len(set(unit_ids)) != len(unit_ids):
                    raise ReviewRefused("duplicate_verification_unit")
                for record in parsed:
                    if record.verification_unit_id not in ledger.delivered:
                        raise ReviewRefused("verification_unit_not_delivered")
                    if record.verification_unit_id in ledger.dispositions:
                        raise ReviewRefused("disposition_conflict")
                    if record.duplicate_unit_id == record.verification_unit_id:
                        raise ReviewRefused("invalid_duplicate_reference")
                    if record.verdict != "unresolved" and not record.evidence:
                        raise ReviewRefused("disposition_evidence_required")
                    for evidence in record.evidence:
                        if isinstance(evidence, CheckEvidence):
                            owner = next(
                                (
                                    check
                                    for check in context.checks.values()
                                    if check.receipt
                                    and check.receipt.receipt_id == evidence.receipt_id
                                ),
                                None,
                            )
                            if (
                                owner is None
                                or owner.binding != binding
                                or owner.receipt.outcome != "executed"
                            ):
                                raise ReviewRefused("invalid_executed_check_evidence")
                            if not owner.result_read:
                                raise ReviewRefused("check_result_not_read")
            check = None
            decline_receipt = None
            if kind == "check_proposals":
                from .code_review_checks import PendingCheck

                if any(
                    item.binding.attempt_id == binding.attempt_id
                    and item.receipt is None
                    for item in context.checks.values()
                ):
                    raise ReviewRefused("check_already_pending")
                ledger = self.verification_ledger(context, binding)
                if not set(parsed[0].verification_unit_ids) <= ledger.delivered:
                    raise ReviewRefused("verification_unit_not_delivered")
            if kind == "check_actions":
                check = context.checks.get(parsed[0].proposal_id)
                if (
                    check is None
                    or check.primary_run_id != current_run_id()
                    or check.binding.attempt_id
                    != context.check_actors[current_run_id()]
                ):
                    raise ReviewRefused("check_actor_mismatch")
                verifier_stream = context.streams[
                    (check.binding.slot_id, check.binding.attempt_id)
                ]
                if verifier_stream.terminal_status is not None:
                    raise ReviewRefused("verification_attempt_closed")
                if (
                    parsed[0].action == "decline"
                    and check.native_execution_receipt_id is not None
                ):
                    raise ReviewRefused("check_execution_already_accepted")
                if parsed[0].action == "decline" and check.receipt is None:
                    # The host receipt and accepted action share one immutable
                    # chunk/manifest commit. Failed publication changes neither.
                    decline_receipt = check.result(
                        "declined", parsed[0].reason or "primary_declined"
                    )
            local_keys = {
                item.local_key for item in parsed if isinstance(item, FindingRecord)
            }
            if local_keys & stream.local_keys:
                raise ReviewRefused("duplicate_local_key")
            accepted = (
                tuple(
                    f"{binding.slot_id}-F{context.finding_sequences.get(binding.slot_id, 0) + index + 1}"
                    for index in range(len(parsed))
                )
                if kind == "findings"
                else tuple(secrets.token_hex(16) for _ in parsed)
            )
            ack = SubmissionAck(
                accepted_ids=accepted,
                submission_revision=stream.revision + 1,
                cursor=secrets.token_hex(16),
            )
            if admit_ack is not None:
                admit_ack(ack)
            if kind == "check_proposals":
                check = PendingCheck(
                    accepted[0],
                    parsed[0],
                    binding,
                    current_run_id(),
                    context.preparation.primary_actor.run_id,
                )
                primary_ids = [
                    run
                    for run, attempt in context.check_actors.items()
                    if attempt == binding.attempt_id
                ]
                if primary_ids:
                    check.primary_run_id = primary_ids[-1]
                if binding.check_feedback_mode == "unavailable_inline":
                    check.receipt = check.result(
                        "unavailable", "check_feedback_unavailable_inline"
                    )
                    check.state = "resolved"
                    check.changed.set()
            sequence = context.publication_sequence + 1
            chunk = canonical_record_bytes(
                {
                    "previous": context._head_chunk,
                    "sequence": sequence,
                    "slot_id": binding.slot_id,
                    "attempt_id": context.check_actors[current_run_id()]
                    if kind == "check_actions"
                    else binding.attempt_id,
                    "check_actor": current_run_id()
                    if kind == "check_actions"
                    else None,
                    "kind": kind,
                    "submission_id": submission_id,
                    "digest": digest,
                    "ack": ack.model_dump(mode="json"),
                    "records": records,
                    "host_check_result": check.receipt.model_dump(mode="json")
                    if kind == "check_proposals" and check.receipt
                    else decline_receipt.model_dump(mode="json")
                    if decline_receipt is not None
                    else None,
                }
            )
            # Names are fixed width, so peak pointer space is known before writing.
            pointer_size = len(
                canonical_record_bytes({"chunk": "0" * 32, "sequence": sequence})
            )
            if kind == "check_proposals" and check.receipt is None:
                check.result_reservation = self.reserve(
                    review_id,
                    source_bytes=0,
                    report_bytes=8192,
                    files=1,
                    participant_id=binding.slot_id,
                )
            try:
                with self.reserve(
                    review_id,
                    source_bytes=0,
                    report_bytes=len(chunk) + pointer_size,
                    files=2,
                    participant_id=binding.slot_id,
                ) as reservation:
                    chunk_path = reservation.write(chunk, category="report")
                    pointer = canonical_record_bytes(
                        {"chunk": chunk_path.name, "sequence": sequence}
                    )
                    staging = reservation.write(pointer, category="report")
                    with self._manager.live_guard(self._scratch):
                        self._live(context)
                        self._bound(context)
                        os.replace(staging, context.manifest_path)
                        item = context._files.pop(staging)
                        item.path = context.manifest_path
                        context._files[context.manifest_path] = item
                        reservation._committed = True
                        stream.submissions[submission_id] = (digest, ack)
                        stream.revision = ack.submission_revision
                        if kind == "dispositions":
                            ledger.dispositions.update(
                                {
                                    record.verification_unit_id: record.model_dump(
                                        mode="json"
                                    )
                                    for record in parsed
                                }
                            )
                            # Empty withdrawals must retain outstanding recovery rounds in history.
                            if (
                                parsed
                                and ledger.delivered <= ledger.dispositions.keys()
                            ):
                                ledger.acknowledged_cursors.append(ack.cursor)
                        stream.local_keys.update(local_keys)
                        stream.rejected.pop(
                            hashlib.sha256(submission_id.encode()).hexdigest(), None
                        )
                        if kind in {"coverage", "reviewer_complete"}:
                            stream.coverage = (chunk_path, 0)
                            stream.completion_requested = kind == "reviewer_complete"
                        if kind == "verifier_complete":
                            stream.completion_requested = True
                        context._head_chunk = chunk_path.name
                        context.publication_sequence = sequence
                        if kind == "check_proposals":
                            context.checks[check.proposal_id] = check
                            context.check_proposal_ids.add(check.proposal_id)
                            if check.receipt:
                                context.check_receipt_ids.add(check.receipt.receipt_id)
                        if decline_receipt is not None:
                            check.receipt, check.state = decline_receipt, "resolved"
                            context.check_receipt_ids.add(decline_receipt.receipt_id)
                            check.changed.set()
                        if kind == "findings":
                            context.finding_sequences[binding.slot_id] = (
                                context.finding_sequences.get(binding.slot_id, 0)
                                + len(parsed)
                            )
                            context.original_index.update(
                                {
                                    key: (chunk_path, index)
                                    for index, key in enumerate(accepted)
                                }
                            )
                        return ack
            except ConsoleScratchSpaceUnavailable as exc:
                raise ReviewRefused("review_expired") from exc
            except OSError as exc:
                raise ReviewRefused("publication_failed") from exc
            finally:
                if (
                    kind == "check_proposals"
                    and check.proposal_id not in context.checks
                    and check.result_reservation
                ):
                    check.result_reservation.__exit__(None, None, None)

    def publish(
        self,
        review_id: str,
        *,
        expected_revision: int,
        submission_id: str,
        records: list[dict],
        kind: str = "findings",
        admit_ack: Callable[[SubmissionAck], object] | None = None,
    ) -> SubmissionAck:
        """Publish a participant append, retaining unresolved refusal evidence."""
        try:
            return self._publish(
                review_id,
                expected_revision=expected_revision,
                submission_id=submission_id,
                records=records,
                kind=kind,
                admit_ack=admit_ack,
            )
        except ReviewRefused as exc:
            self.note_rejection(review_id, submission_id, exc.reason)
            raise

    def note_rejection(self, review_id: str, submission_id: str, reason: str) -> None:
        """Record a live participant's format gap without spending its revision."""
        with self._tab.lock, self.lease(review_id) as context:
            binding = self._bound(context)
            stream = context.streams[(binding.slot_id, binding.attempt_id)]
            if binding.role == "primary" or stream.terminal_status is not None:
                return
            # A finite run can repair the same rejected submission identity.
            key = hashlib.sha256(str(submission_id).encode()).hexdigest()
            # Failed withdrawal retries must not erase the recoverable conflict.
            if stream.rejected.get(key) in {reason, "disposition_conflict"}:
                return
            stream.rejected[key] = reason
            self._host_receipt(context, binding, stream)

    def _host_receipt(self, context, binding, stream) -> str | None:
        """Charge host metadata separately from the participant append revision."""
        data = canonical_record_bytes(
            {
                "slot_id": binding.slot_id,
                "attempt_id": binding.attempt_id,
                "run_id": stream.run_id,
                "host_sequence": stream.host_sequence + 1,
                "raw_chunks": [path.name for path in stream.raw_chunks],
                "raw_bytes": sum(
                    context._files[path].size for path in stream.raw_chunks
                ),
                "capture_terminal": stream.capture_terminal,
                "capture_receipt": stream.capture_receipt,
                "capture_problem": stream.capture_problem,
                "withheld": stream.withheld,
                "terminal_status": stream.terminal_status,
                "terminal_receipt": stream.terminal_receipt,
                "rejected": stream.rejected,
                "routes": stream.routes,
            }
        )
        try:
            with self.reserve(
                context.review_id,
                source_bytes=0,
                report_bytes=len(data),
                files=1,
                participant_id=binding.slot_id,
            ) as reservation:
                receipt = reservation.write(data, category="report")
                reservation.commit(0, len(data), 1)
            stream.host_sequence += 1
            return receipt.name
        except (ReviewRefused, OSError):
            stream.capture_problem = "capture_receipt_unavailable"

    def capture_review_output(
        self, run_id: str, text: str, *, terminal: bool, withheld: bool = False
    ) -> None:
        """Append one policy-filtered assistant response; terminal uses empty text."""
        with self._tab.lock:
            binding = self._tab.runs.get(run_id)
            if binding is None or binding.role == "primary":
                raise ReviewRefused("run_not_bound")
            with self.lease(binding.review_id) as context:
                self._bound(context, run_id)
                stream = context.streams[(binding.slot_id, binding.attempt_id)]
                if stream.capture_terminal or (
                    stream.terminal_status is not None and text
                ):
                    raise ReviewRefused("review_capture_closed")
                stream.withheld |= withheld
                if text and not stream.capture_problem:
                    data = text.encode("utf-8")
                    try:
                        with self.reserve(
                            context.review_id,
                            source_bytes=0,
                            report_bytes=len(data),
                            files=1,
                            participant_id=binding.slot_id,
                        ) as reservation:
                            path = reservation.write(data, category="report")
                            reservation.commit(0, len(data), 1)
                        stream.raw_chunks.append(path)
                    except ReviewRefused as exc:
                        stream.capture_problem = exc.reason
                    except OSError:
                        stream.capture_problem = "capture_failed"
                stream.capture_terminal = terminal
                receipt = self._host_receipt(context, binding, stream)
                if terminal:
                    stream.capture_receipt = receipt

    def observe_review_route(self, run_id: str, provider: str, model: str) -> None:
        """Record actual dispatch choices, including attempted fallback changes."""
        with self._tab.lock:
            binding = self._tab.runs[run_id]
            with self.lease(binding.review_id) as context:
                self._bound(context, run_id)
                stream = context.streams[(binding.slot_id, binding.attempt_id)]
                if stream.terminal_status is not None:
                    raise ReviewRefused("review_run_settled")
                route = {"provider": provider, "model": model}
                if not stream.routes or stream.routes[-1] != route:
                    stream.routes.append(route)
                    self._host_receipt(context, binding, stream)

    def settle_review_run(
        self, run_id: str, *, terminal_status: str, receipt_id: str
    ) -> None:
        """Associate actual native settlement; a missing receipt never seals."""
        if terminal_status not in {"done", "error", "cancelled", "stuck", "superseded"}:
            raise ReviewRefused("invalid_terminal_status")
        with self._tab.lock:
            binding = self._tab.runs.get(run_id)
            if binding is None or binding.role == "primary":
                raise ReviewRefused("run_not_bound")
            with self.lease(binding.review_id) as context:
                self._bound(context, run_id)
                stream = context.streams[(binding.slot_id, binding.attempt_id)]
                if stream.terminal_status is not None:
                    if (stream.terminal_status, stream.terminal_receipt) != (
                        terminal_status,
                        receipt_id,
                    ):
                        raise ReviewRefused("terminal_receipt_conflict")
                    return
                if binding.role == "verifier":
                    try:
                        self.finalize_verification(context, binding)
                    except ReviewRefused as exc:
                        stream.capture_problem = exc.reason
                    except OSError:
                        stream.capture_problem = "verification_checkpoint_unavailable"
                stream.terminal_status = terminal_status
                stream.terminal_receipt = receipt_id
                self._host_receipt(context, binding, stream)

    def read_record(self, context: ReviewContext, location: tuple[Path, int]) -> dict:
        """Read an intact original through its storage owner, without model caps."""
        path, offset = location
        return json.loads(self.read_file(context, path))["records"][offset]

    def read_file(
        self,
        context: ReviewContext,
        path: Path,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> bytes:
        self._live(context)
        item = context._files.get(path)
        if item is None or not self._matches(path, item.identity):
            raise ReviewRefused("report_capture_missing")
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as reader:
                actual = os.fstat(reader.fileno())
                if (
                    actual.st_dev,
                    actual.st_ino,
                ) != item.identity or actual.st_size != item.size:
                    raise ReviewRefused("report_capture_missing")
                reader.seek(offset)
                return reader.read(item.size if limit is None else limit)
        except OSError as exc:
            raise ReviewRefused("report_capture_missing") from exc

    @staticmethod
    def _matches(path: Path, identity: tuple[int, int] | None) -> bool:
        try:
            metadata = path.lstat()
        except OSError:
            return False
        return (
            not stat.S_ISLNK(metadata.st_mode)
            and (metadata.st_dev, metadata.st_ino) == identity
        )

    @classmethod
    def _delete(cls, context: ReviewContext, item: _File) -> None:
        if not cls._matches(
            context.bundle_path, context.bundle_identity
        ) or not cls._matches(item.path, item.identity):
            return
        try:
            item.path.unlink()
        except OSError:
            return
        context._files.pop(item.path, None)

    @classmethod
    def _cleanup(cls, context: ReviewContext) -> None:
        if not cls._matches(context.bundle_path, context.bundle_identity):
            return
        for item in tuple(context._files.values()):
            cls._delete(context, item)
        if context._files:
            return
        try:
            context.bundle_path.rmdir()
        except OSError:
            return
        context._cleaned = True

    def release(
        self,
        review_id: str,
        *,
        expected_revision: int,
        admit_ack: Callable[[ReleaseReceipt], object] | None = None,
    ) -> ReleaseReceipt:
        """Tombstone and clean; replay receipts are retained for the latest four releases."""
        _counts(expected_revision)
        with self._tab.lock:
            cached = self._tab.release_receipts.get(review_id)
            if cached is not None:
                if expected_revision not in (cached.revision - 1, cached.revision):
                    raise ReviewRefused("context_revision_conflict")
                if admit_ack is not None:
                    admit_ack(cached)
                return cached
            context = self._context(review_id)
            if context._release_revision is None:
                self._live(context)
                if expected_revision != context.revision:
                    raise ReviewRefused("context_revision_conflict")
                if context._leases:
                    raise ReviewRefused("review_busy")
                if admit_ack is not None:
                    for cleaned in (False, True):
                        admit_ack(
                            ReleaseReceipt(
                                review_id=review_id,
                                revision=context.revision + 1,
                                tombstoned=True,
                                cleaned=cleaned,
                            )
                        )
                context._release_revision = context.revision
                context.revision += 1
                context.state = "tombstoned"
            elif expected_revision not in (context._release_revision, context.revision):
                raise ReviewRefused("context_revision_conflict")
            try:
                with (
                    self._manager.lease(self._scratch),
                    self._manager.live_guard(self._scratch),
                ):
                    self._cleanup(context)
            except ConsoleScratchSpaceUnavailable:
                pass
            _discard_payload(context)
            self._tab.runs = {
                run: binding
                for run, binding in self._tab.runs.items()
                if binding.review_id != review_id
            }
            self._tab.retired_attempts = {
                identity
                for identity in self._tab.retired_attempts
                if identity[0] != review_id
            }
            receipt = ReleaseReceipt(
                review_id=review_id,
                revision=context.revision,
                tombstoned=True,
                cleaned=context._cleaned,
            )
            if context._cleaned:
                self._tab.contexts.pop(review_id)
                self._tab.release_receipts[review_id] = receipt
                while len(self._tab.release_receipts) > TAB_MAX_CONTEXTS:
                    self._tab.release_receipts.popitem(last=False)
            return receipt

    def close(self) -> None:
        """Fence this generation without waiting for the tab's native writers.

        The shared close event and scratch revocation are authoritative; context
        state mirrors and quota reconciliation refresh on the next locked access.
        Existing leases drain through the scratch owner and its cleanup receipt.
        """
        self._tab.closed.set()
        try:
            with self._manager.live_guard(self._scratch):
                self._manager.close(self._session_id)
        except ConsoleScratchSpaceUnavailable:
            pass
        # Never wait for a writer or an in-flight native admission. Its next
        # locked access retires result leases; process leases remain native-owned.
        if self._tab.lock.acquire(blocking=False):
            try:
                self._live()
            except ReviewRefused:
                pass
            finally:
                self._tab.lock.release()

    def usage(self) -> ReviewUsage:
        """Read content-free accounting; refund scratch deletion only when confirmed."""
        with self._tab.lock:
            if not self._manager.is_live(self._scratch):
                self._tab.closed.set()
                for context in self._tab.contexts.values():
                    context.state = "tombstoned"
            retained = reserved = files = reserved_files = contexts = 0
            for context in self._tab.contexts.values():
                # Absence alone may mean displacement, even under an unchanged root.
                if self._matches(
                    self._scratch.root, self._scratch.identity
                ) and self._matches(context.bundle_path, context.bundle_identity):
                    for item in context._files.values():
                        try:
                            metadata = item.path.lstat()
                            if (
                                stat.S_ISREG(metadata.st_mode)
                                and (metadata.st_dev, metadata.st_ino) == item.identity
                            ):
                                item.size = metadata.st_size
                        except OSError:
                            pass
                retained += sum(item.size for item in context._files.values())
                files += len(context._files)
                contexts += int(not context._cleaned)
                for reservation in context._reservations:
                    rs, rr, rf = reservation._remaining()
                    reserved += rs + rr
                    reserved_files += rf
            return ReviewUsage(retained, reserved, files, reserved_files, contexts)
