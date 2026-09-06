"""Best-effort, idempotent settlement for dispatched provider-call traces."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import hashlib
import json
import sqlite3
import threading
from typing import Protocol

from tldw_chatbook.Chat.console_semantic_revision import (
    SemanticRevisionCoordinator,
    project_semantic_revision_provider_message,
)
from tldw_chatbook.Chat.console_trace_models import (
    SemanticRevisionRef,
    TraceCallState,
    TraceContentRef,
)
from tldw_chatbook.Chat.console_trace_redaction import (
    CREDENTIAL_SANITIZER_UNAVAILABLE,
    CredentialSanitizer,
    CredentialSanitizationResult,
)
from tldw_chatbook.Chat.console_trace_repository import (
    ConsoleTraceRepository,
    TraceCallRecord,
    TraceEventType,
    TraceIdentityConflict,
    TraceResponseLinkRecord,
)


TRACE_RESPONSE_MEDIA_TYPE = "application/json"
TRACE_RESPONSE_NORMALIZATION_VERSION = "canonical-json-v1"
DEFAULT_MAX_PENDING_SETTLEMENTS = 64
DEFAULT_OPEN_CALL_RECOVERY_GRACE_SECONDS = 300
MAX_TRACE_RESPONSE_BYTES = 1_048_576
MAX_TRACE_USAGE_BYTES = 65_536
_TERMINAL_SETTLEMENT_STATES = frozenset(
    {
        TraceCallState.COMPLETE,
        TraceCallState.STOPPED,
        TraceCallState.ERROR,
        TraceCallState.INTERRUPTED,
    }
)


class _SettlementConflict(ValueError):
    """A durable call already has a different immutable settlement."""


class _TraceSanitizer(Protocol):
    def sanitize(self, value: object) -> CredentialSanitizationResult:
        """Return a credential-filtered JSON-like value."""


@dataclass(frozen=True, slots=True)
class TraceResponseOmission:
    """Content-free response marker produced before settlement buffering."""

    reason_code: str

    def __post_init__(self) -> None:
        if not self.reason_code:
            raise ValueError("reason_code")


@dataclass(frozen=True, slots=True)
class TraceSettlementRequest:
    """One immutable provider-call seal attempt.

    Response and usage values are hidden from representations and sanitized before
    they can enter the retry queue.
    """

    call_id: str
    outcome: TraceCallState
    response_envelope: object | None = field(repr=False)
    usage: Mapping[str, object] | None = field(default=None, repr=False)
    response_started_at: str = ""
    settled_at: str = ""
    canonical_message_id: str | None = None
    prior_integrity_state: str = "pending"
    prior_omission_reason_code: str | None = None

    def __post_init__(self) -> None:
        if not self.call_id:
            raise ValueError("call_id")
        if self.outcome not in _TERMINAL_SETTLEMENT_STATES:
            raise ValueError("outcome")
        if not self.response_started_at or not self.settled_at:
            raise ValueError("settlement timestamps")
        if self.prior_integrity_state not in {"pending", "complete", "incomplete"}:
            raise ValueError("prior_integrity_state")


@dataclass(frozen=True, slots=True)
class _PreparedSettlement:
    call_id: str
    outcome: TraceCallState
    response_bytes: bytes | None = field(repr=False)
    response_omission: str | None
    usage: Mapping[str, object] | None = field(repr=False)
    usage_omission: str | None
    integrity_omission: str | None
    response_started_at: str
    settled_at: str
    canonical_message_id: str | None
    prior_integrity_state: str
    prior_omission_reason_code: str | None


@dataclass(frozen=True, slots=True)
class ConsoleTraceSettlementHandoff:
    """Opaque sanitized call settlement awaiting canonical persistence."""

    _coordinator: ConsoleTraceSettlementCoordinator = field(
        repr=False,
        compare=False,
    )
    _database: object = field(repr=False, compare=False)
    _prepared: _PreparedSettlement = field(repr=False)

    @property
    def call_id(self) -> str:
        """Return the explicit immutable call identity owned by this handoff."""

        return self._prepared.call_id

    def settle(self, canonical_message_id: str | None) -> bool:
        """Submit after canonical save, or trace-own when no save exists."""

        return self._coordinator._submit_prepared(
            self._database,
            replace(
                self._prepared,
                canonical_message_id=canonical_message_id,
            ),
        )


class ConsoleTraceSettlementCoordinator:
    """Seal provider results independently and recover durable open calls."""

    def __init__(
        self,
        repository: ConsoleTraceRepository | None = None,
        *,
        sanitizer: _TraceSanitizer | None = None,
        max_pending: int = DEFAULT_MAX_PENDING_SETTLEMENTS,
    ) -> None:
        if type(max_pending) is not int or max_pending < 1:
            raise ValueError("max_pending")
        self.repository = repository or ConsoleTraceRepository()
        self._sanitizer = sanitizer or CredentialSanitizer()
        self._max_pending = max_pending
        self._pending: OrderedDict[str, tuple[object, _PreparedSettlement]] = (
            OrderedDict()
        )
        self._inflight: dict[str, str] = {}
        self._queue_lock = threading.Lock()
        self._dropped_count = 0

    @property
    def pending_count(self) -> int:
        """Return the number of distinct queued call seals."""

        with self._queue_lock:
            return len(self._pending)

    @property
    def dropped_count(self) -> int:
        """Return how many oldest queued seals the process bound evicted."""

        with self._queue_lock:
            return self._dropped_count

    def mark_response_started(
        self,
        database: object,
        *,
        call_id: str,
        occurred_at: str,
    ) -> TraceCallRecord:
        """Durably record first provider-response evidence, idempotently."""

        from tldw_chatbook.Chat.console_trace_service import (
            trace_critical_write_checkpoint_policy,
        )

        with trace_critical_write_checkpoint_policy(database):
            with database.transaction(immediate=True) as cursor:  # type: ignore[attr-defined]
                existing = self.repository.get_call(cursor, call_id)
                if existing is None:
                    raise KeyError("call_id")
                if existing.state is TraceCallState.DISPATCH_STARTED:
                    return self.repository.advance_call_state(
                        cursor,
                        call_id=call_id,
                        target=TraceCallState.RESPONSE_STARTED,
                        occurred_at=occurred_at,
                        integrity_state=existing.integrity_state,
                        omission_reason_code=existing.omission_reason_code,
                    )
                if (
                    existing.state is TraceCallState.RESPONSE_STARTED
                    or existing.state in _TERMINAL_SETTLEMENT_STATES
                ):
                    return existing
                raise ValueError("call_has_no_live_response_boundary")

    def settle(
        self,
        database: object,
        request: TraceSettlementRequest,
    ) -> TraceCallRecord:
        """Seal one call now, raising only to the best-effort caller."""

        return self._settle_prepared(database, self._prepare_safely(request))

    def submit(self, database: object, request: TraceSettlementRequest) -> bool:
        """Try one post-dispatch seal and queue a content-safe retry on failure."""

        return self._submit_prepared(database, self._prepare_safely(request))

    def fingerprint(self, request: TraceSettlementRequest) -> str:
        """Return a content-free identity for one sanitized settlement signal."""

        return _prepared_settlement_fingerprint(self._prepare_safely(request))

    def prepare_handoff(
        self,
        database: object,
        request: TraceSettlementRequest,
    ) -> ConsoleTraceSettlementHandoff:
        """Sanitize one response before it crosses the persistence handoff."""

        return ConsoleTraceSettlementHandoff(
            self,
            database,
            self._prepare_safely(request),
        )

    def _submit_prepared(
        self,
        database: object,
        prepared: _PreparedSettlement,
    ) -> bool:
        fingerprint = _prepared_settlement_fingerprint(prepared)
        claimed: tuple[object, _PreparedSettlement] | None = None
        with self._queue_lock:
            if prepared.call_id in self._inflight:
                return False
            queued = self._pending.get(prepared.call_id)
            if queued is not None:
                if _prepared_settlement_fingerprint(queued[1]) != fingerprint:
                    return False
                claimed = self._pending.pop(prepared.call_id)
                self._inflight[prepared.call_id] = fingerprint
        if claimed is not None:
            return self._settle_claimed(*claimed)

        self.retry_pending()
        with self._queue_lock:
            if prepared.call_id in self._inflight:
                return False
            queued = self._pending.get(prepared.call_id)
            if queued is not None:
                if _prepared_settlement_fingerprint(queued[1]) != fingerprint:
                    return False
                claimed = self._pending.pop(prepared.call_id)
            else:
                claimed = (database, prepared)
            self._inflight[prepared.call_id] = fingerprint
        return self._settle_claimed(*claimed)

    def retry_pending(self) -> int:
        """Retry one snapshot of queued settlements and return successes."""

        with self._queue_lock:
            pending = []
            for call_id, queued in tuple(self._pending.items()):
                if call_id in self._inflight:
                    continue
                self._pending.pop(call_id)
                self._inflight[call_id] = _prepared_settlement_fingerprint(queued[1])
                pending.append(queued)
        succeeded = 0
        for database, prepared in pending:
            if self._settle_claimed(database, prepared):
                succeeded += 1
        return succeeded

    def _settle_claimed(
        self,
        database: object,
        prepared: _PreparedSettlement,
    ) -> bool:
        """Settle one lock-claimed signal, atomically releasing queue ownership."""

        from tldw_chatbook.DB.base_db import operation_owned_connection

        try:
            with operation_owned_connection(database):
                self._settle_prepared(database, prepared)
        except (_SettlementConflict, TraceIdentityConflict):
            with self._queue_lock:
                self._inflight.pop(prepared.call_id, None)
            return False
        except Exception:
            with self._queue_lock:
                self._inflight.pop(prepared.call_id, None)
                self._enqueue_locked(database, prepared)
            return False
        with self._queue_lock:
            self._inflight.pop(prepared.call_id, None)
        return True

    def recover_open_calls(
        self,
        database: object,
        *,
        occurred_at: str,
        recovery_grace_seconds: int = DEFAULT_OPEN_CALL_RECOVERY_GRACE_SECONDS,
    ) -> tuple[TraceCallRecord, ...]:
        """Close stale durable calls left open by process death.

        A startup in another app process must not terminate a recently active
        provider call. Calls therefore become recovery candidates only after a
        bounded inactivity grace period.

        Args:
            database: Trace database that owns the open calls.
            occurred_at: Timestamp recorded on recovered terminal states.
            recovery_grace_seconds: Minimum inactivity before recovery.

        Returns:
            Calls transitioned by this recovery pass.

        Raises:
            ValueError: If ``recovery_grace_seconds`` is invalid.
        """

        if type(recovery_grace_seconds) is not int or recovery_grace_seconds < 0:
            raise ValueError("recovery_grace_seconds")

        recovered: list[TraceCallRecord] = []
        with database.transaction(immediate=True) as cursor:  # type: ignore[attr-defined]
            rows = cursor.execute(
                """SELECT call_id FROM console_trace_calls
                    WHERE state IN ('reserved', 'dispatch_started', 'response_started')
                      AND julianday(
                            COALESCE(
                              response_started_at,
                              dispatch_started_at,
                              created_at
                            )
                          ) <= julianday(?, ?)
                    ORDER BY owner_id, turn_id, run_id, call_sequence, call_id""",
                (occurred_at, f"-{recovery_grace_seconds} seconds"),
            ).fetchall()
            for row in rows:
                existing = self.repository.get_call(cursor, str(row[0]))
                if existing is None:
                    continue
                target = {
                    TraceCallState.RESERVED: TraceCallState.NOT_DISPATCHED,
                    TraceCallState.DISPATCH_STARTED: TraceCallState.DISPATCH_UNKNOWN,
                    TraceCallState.RESPONSE_STARTED: TraceCallState.INTERRUPTED,
                }.get(existing.state)
                if target is None:
                    continue
                recovered_call = self.repository.advance_call_state(
                    cursor,
                    call_id=existing.call_id,
                    target=target,
                    occurred_at=occurred_at,
                    integrity_state=existing.integrity_state,
                    omission_reason_code=existing.omission_reason_code,
                )
                self._append_event(
                    cursor,
                    recovered_call,
                    "call_outcome",
                )
                recovered.append(recovered_call)
        return tuple(recovered)

    def _prepare(self, request: TraceSettlementRequest) -> _PreparedSettlement:
        if type(request) is not TraceSettlementRequest:
            raise TypeError("request")
        response_bytes: bytes | None = None
        response_omission: str | None = None
        if isinstance(request.response_envelope, TraceResponseOmission):
            response_omission = request.response_envelope.reason_code
            response_bytes = _canonical_bytes(
                {
                    "omitted": True,
                    "reason": response_omission,
                }
            )
        elif request.response_envelope is not None:
            try:
                response = self._sanitizer.sanitize(request.response_envelope)
                if response.available:
                    response_bytes = _canonical_bytes(response.value)
                    if len(response_bytes) > MAX_TRACE_RESPONSE_BYTES:
                        byte_length = len(response_bytes)
                        response_bytes = _canonical_bytes(
                            {
                                "omitted": True,
                                "reason": "response_size_limit",
                                "byte_length": byte_length,
                            }
                        )
                        response_omission = "response_size_limit"
                else:
                    response_omission = (
                        response.omission_reason_code
                        or CREDENTIAL_SANITIZER_UNAVAILABLE
                    )
            except Exception:
                response_bytes = None
                response_omission = "response_canonicalization_unavailable"

        usage: Mapping[str, object] | None = None
        usage_omission: str | None = None
        if request.usage is not None:
            try:
                sanitized_usage = self._sanitizer.sanitize(request.usage)
                if sanitized_usage.available and isinstance(
                    sanitized_usage.value, Mapping
                ):
                    usage_bytes = _canonical_bytes(sanitized_usage.value)
                    if len(usage_bytes) <= MAX_TRACE_USAGE_BYTES:
                        usage = json.loads(usage_bytes)
                    else:
                        usage_omission = "usage_size_limit"
                else:
                    usage_omission = "usage_" + (
                        sanitized_usage.omission_reason_code
                        or CREDENTIAL_SANITIZER_UNAVAILABLE
                    )
            except Exception:
                usage = None
                usage_omission = "usage_canonicalization_unavailable"
        return _PreparedSettlement(
            call_id=request.call_id,
            outcome=request.outcome,
            response_bytes=response_bytes,
            response_omission=response_omission,
            usage=usage,
            usage_omission=usage_omission,
            integrity_omission=_integrity_omission(
                request.prior_omission_reason_code,
                response_omission,
                usage_omission,
            ),
            response_started_at=request.response_started_at,
            settled_at=request.settled_at,
            canonical_message_id=request.canonical_message_id,
            prior_integrity_state=request.prior_integrity_state,
            prior_omission_reason_code=request.prior_omission_reason_code,
        )

    def _prepare_safely(
        self,
        request: TraceSettlementRequest,
    ) -> _PreparedSettlement:
        """Fail closed without retaining values when preparation itself breaks."""

        if type(request) is not TraceSettlementRequest:
            raise TypeError("request")
        try:
            return self._prepare(request)
        except Exception:
            response_omission = (
                "response_preparation_unavailable"
                if request.response_envelope is not None
                else None
            )
            usage_omission = (
                "usage_preparation_unavailable" if request.usage is not None else None
            )
            return _PreparedSettlement(
                call_id=request.call_id,
                outcome=request.outcome,
                response_bytes=None,
                response_omission=response_omission,
                usage=None,
                usage_omission=usage_omission,
                integrity_omission=_integrity_omission(
                    request.prior_omission_reason_code,
                    response_omission,
                    usage_omission,
                ),
                response_started_at=request.response_started_at,
                settled_at=request.settled_at,
                canonical_message_id=request.canonical_message_id,
                prior_integrity_state=request.prior_integrity_state,
                prior_omission_reason_code=request.prior_omission_reason_code,
            )

    def _settle_prepared(
        self,
        database: object,
        prepared: _PreparedSettlement,
    ) -> TraceCallRecord:
        with database.transaction(immediate=True) as cursor:  # type: ignore[attr-defined]
            existing = self.repository.get_call(cursor, prepared.call_id)
            if existing is None:
                raise KeyError("call_id")
            if existing.state in _TERMINAL_SETTLEMENT_STATES:
                self._verify_terminal_retry(cursor, existing, prepared)
                return existing
            if existing.state is TraceCallState.DISPATCH_UNKNOWN:
                self._verify_dispatch_unknown_retry(cursor, existing, prepared)
                return existing

            if (
                existing.state is TraceCallState.DISPATCH_STARTED
                and _is_pre_response_stop(prepared)
            ):
                settled = self.repository.advance_call_state(
                    cursor,
                    call_id=existing.call_id,
                    target=TraceCallState.DISPATCH_UNKNOWN,
                    occurred_at=prepared.settled_at,
                    usage=prepared.usage,
                    integrity_state=(
                        "incomplete"
                        if prepared.integrity_omission is not None
                        else existing.integrity_state
                    ),
                    omission_reason_code=(
                        prepared.integrity_omission or existing.omission_reason_code
                    ),
                )
                self._append_event(cursor, settled, "call_outcome")
                if prepared.usage is not None:
                    self._append_event(cursor, settled, "usage")
                return settled

            response_link: TraceResponseLinkRecord | None = None
            if prepared.response_bytes is not None:
                if existing.state is TraceCallState.DISPATCH_STARTED:
                    existing = self.repository.advance_call_state(
                        cursor,
                        call_id=existing.call_id,
                        target=TraceCallState.RESPONSE_STARTED,
                        occurred_at=prepared.response_started_at,
                        integrity_state=existing.integrity_state,
                        omission_reason_code=existing.omission_reason_code,
                    )
                response_link = self._store_response(
                    cursor,
                    database=database,
                    call=existing,
                    prepared=prepared,
                )
            elif prepared.response_omission is not None:
                if existing.state is TraceCallState.DISPATCH_STARTED:
                    existing = self.repository.advance_call_state(
                        cursor,
                        call_id=existing.call_id,
                        target=TraceCallState.RESPONSE_STARTED,
                        occurred_at=prepared.response_started_at,
                        integrity_state="incomplete",
                        omission_reason_code=prepared.response_omission,
                    )

            settled = self.repository.advance_call_state(
                cursor,
                call_id=existing.call_id,
                target=prepared.outcome,
                occurred_at=prepared.settled_at,
                usage=prepared.usage,
                integrity_state=(
                    "incomplete"
                    if prepared.integrity_omission is not None
                    else existing.integrity_state
                ),
                omission_reason_code=(
                    prepared.integrity_omission or existing.omission_reason_code
                ),
            )
            if response_link is not None:
                self._append_event(
                    cursor,
                    settled,
                    "response_selection",
                    response_link=response_link,
                )
            self._append_event(cursor, settled, "call_outcome")
            if prepared.usage is not None:
                self._append_event(cursor, settled, "usage")
            return settled

    def _store_response(
        self,
        cursor: sqlite3.Cursor,
        *,
        database: object,
        call: TraceCallRecord,
        prepared: _PreparedSettlement,
    ) -> TraceResponseLinkRecord:
        response = None
        if prepared.canonical_message_id is not None:
            owner = self.repository.get_owner(cursor, call.owner_id)
            if owner is not None and owner.conversation_id is not None:
                try:
                    revision = SemanticRevisionCoordinator(
                        database,  # type: ignore[arg-type]
                        repository=self.repository,
                    ).ensure_current_revision(
                        cursor,
                        message_id=prepared.canonical_message_id,
                        creation_reason="provider_response",
                    )
                    projected = project_semantic_revision_provider_message(
                        cursor,
                        revision_id=revision.revision_id,
                        expected_conversation_id=owner.conversation_id,
                    )
                    sanitized = self._sanitizer.sanitize(projected)
                    if (
                        sanitized.available
                        and _canonical_bytes(sanitized.value) == prepared.response_bytes
                    ):
                        response = SemanticRevisionRef(revision.revision_id)
                except Exception:
                    response = None
        if response is None:
            artifact = self.repository.store_sanitized_artifact(
                cursor,
                sanitized_bytes=prepared.response_bytes or b"null",
                media_type=TRACE_RESPONSE_MEDIA_TYPE,
                normalization_version=TRACE_RESPONSE_NORMALIZATION_VERSION,
            )
            response = TraceContentRef(artifact.artifact_id, "provider_response")
        return self.repository.store_response_link(
            cursor,
            call_id=call.call_id,
            response=response,
        )

    def _verify_terminal_retry(
        self,
        cursor: sqlite3.Cursor,
        existing: TraceCallRecord,
        prepared: _PreparedSettlement,
    ) -> None:
        if existing.state is not prepared.outcome:
            raise _SettlementConflict("settlement_outcome_conflict")
        expected_integrity = (
            "incomplete"
            if prepared.integrity_omission is not None
            else prepared.prior_integrity_state
        )
        expected_omission = (
            prepared.integrity_omission or prepared.prior_omission_reason_code
        )
        if (
            existing.integrity_state != expected_integrity
            or existing.omission_reason_code != expected_omission
        ):
            raise _SettlementConflict("settlement_integrity_conflict")
        if _canonical_bytes(existing.usage) != _canonical_bytes(prepared.usage):
            raise _SettlementConflict("settlement_usage_conflict")
        link = self.repository.get_response_link(cursor, existing.call_id)
        if prepared.response_bytes is None:
            if link is not None:
                raise _SettlementConflict("settlement_response_conflict")
            return
        if link is None:
            raise _SettlementConflict("settlement_response_conflict")
        if link.link_kind == "artifact" and link.artifact_id is not None:
            artifact = self.repository.get_artifact(cursor, link.artifact_id)
            if (
                artifact is not None
                and artifact.sanitized_bytes == prepared.response_bytes
            ):
                return
        if link.link_kind == "revision" and link.semantic_revision_id is not None:
            owner = self.repository.get_owner(cursor, existing.owner_id)
            if owner is not None and owner.conversation_id is not None:
                try:
                    projected = project_semantic_revision_provider_message(
                        cursor,
                        revision_id=link.semantic_revision_id,
                        expected_conversation_id=owner.conversation_id,
                        policy_id=existing.policy_id,
                    )
                    sanitized = self._sanitizer.sanitize(projected)
                    if (
                        sanitized.available
                        and _canonical_bytes(sanitized.value) == prepared.response_bytes
                    ):
                        return
                except Exception:
                    pass
        raise _SettlementConflict("settlement_response_conflict")

    def _verify_dispatch_unknown_retry(
        self,
        cursor: sqlite3.Cursor,
        existing: TraceCallRecord,
        prepared: _PreparedSettlement,
    ) -> None:
        if not _is_pre_response_stop(prepared):
            raise _SettlementConflict("settlement_outcome_conflict")
        expected_integrity = (
            "incomplete"
            if prepared.integrity_omission is not None
            else prepared.prior_integrity_state
        )
        expected_omission = (
            prepared.integrity_omission or prepared.prior_omission_reason_code
        )
        if (
            existing.integrity_state != expected_integrity
            or existing.omission_reason_code != expected_omission
        ):
            raise _SettlementConflict("settlement_integrity_conflict")
        if _canonical_bytes(existing.usage) != _canonical_bytes(prepared.usage):
            raise _SettlementConflict("settlement_usage_conflict")
        if self.repository.get_response_link(cursor, existing.call_id) is not None:
            raise _SettlementConflict("settlement_response_conflict")

    def _append_event(
        self,
        cursor: sqlite3.Cursor,
        call: TraceCallRecord,
        event_type: TraceEventType,
        *,
        response_link: TraceResponseLinkRecord | None = None,
    ) -> None:
        tail = self.repository.get_event_tail(cursor, call.segment_id)
        self.repository.append_event(
            cursor,
            segment_id=call.segment_id,
            sequence=0 if tail is None else tail.sequence + 1,
            event_type=event_type,
            call_id=call.call_id,
            semantic_revision_id=(
                None if response_link is None else response_link.semantic_revision_id
            ),
            artifact_id=None if response_link is None else response_link.artifact_id,
        )

    def _enqueue(self, database: object, prepared: _PreparedSettlement) -> None:
        with self._queue_lock:
            self._enqueue_locked(database, prepared)

    def _enqueue_locked(
        self,
        database: object,
        prepared: _PreparedSettlement,
    ) -> None:
        if prepared.call_id in self._pending:
            return
        self._pending[prepared.call_id] = (database, prepared)
        self._pending.move_to_end(prepared.call_id)
        if len(self._pending) > self._max_pending:
            self._pending.popitem(last=False)
            self._dropped_count += 1


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def _prepared_settlement_fingerprint(prepared: _PreparedSettlement) -> str:
    response_digest = (
        None
        if prepared.response_bytes is None
        else hashlib.sha256(prepared.response_bytes).hexdigest()
    )
    payload = {
        "call_id": prepared.call_id,
        "outcome": prepared.outcome.value,
        "response_digest": response_digest,
        "response_omission": prepared.response_omission,
        "usage": prepared.usage,
        "usage_omission": prepared.usage_omission,
        "integrity_omission": prepared.integrity_omission,
        "canonical_message_id": prepared.canonical_message_id,
        "prior_integrity_state": prepared.prior_integrity_state,
        "prior_omission_reason_code": prepared.prior_omission_reason_code,
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _is_pre_response_stop(prepared: _PreparedSettlement) -> bool:
    return (
        prepared.outcome is TraceCallState.STOPPED
        and prepared.response_bytes is None
        and prepared.response_omission is None
    )


def _integrity_omission(*omissions: str | None) -> str | None:
    present = tuple(value for value in omissions if value is not None)
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    digest = hashlib.sha256("\0".join(present).encode("utf-8")).hexdigest()[:24]
    return f"multiple_settlement_omissions_{digest}"


def _json_default(value: object) -> object:
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
