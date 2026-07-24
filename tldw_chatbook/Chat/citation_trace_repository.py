"""SQLite persistence for complete, sealed citation provenance aggregates."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import hashlib
import hmac
import json
import sqlite3
from typing import Any
import weakref

from pydantic import BaseModel, ConfigDict

from tldw_chatbook.Chat.citation_provenance_runtime import (
    CitationProvenanceRuntimePolicy,
)
from tldw_chatbook.Chat.citation_source_locators import (
    AuthorityScope,
    CitationReadAuthorization,
)
from tldw_chatbook.Chat.citation_trace_identity import (
    CitationFingerprintCodec,
    CitationFingerprintDomain,
    CitationFingerprintKeyProvider,
    CitationFingerprintKeyUnavailable,
    CitationIdentityNamespace,
    LocalCitationIdentityContext,
    TraceNamespace,
    cache_owner_idempotency_key,
    load_fingerprint_codec,
    local_trace_namespace,
    message_owner_idempotency_key,
)
from tldw_chatbook.Chat.citation_trace_models import (
    AnswerAttemptPayload,
    CitationTrace,
    EvidenceRunPayload,
    EvidenceSnapshotPayload,
    EvidenceStorageMode,
    PolicyCapability,
    SealedCitationWrite,
    TraceOrigin,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


_ROW_FAMILIES = frozenset({"trace", "runs", "snapshots", "attempts", "refs", "owner"})


class CitationPersistenceUnavailable(RuntimeError):
    """Bounded fail-closed canonical write denial."""

    def __init__(self, reason_code: str) -> None:
        if not reason_code or len(reason_code) > 256:
            raise ValueError(
                "citation persistence reason code must be 1..256 characters"
            )
        self.reason_code = reason_code
        super().__init__(reason_code)


class CitationHydrationState(str, Enum):
    """Safe aggregate/hydration result states."""

    AUTHORIZED = "authorized"
    TRACE_NOT_FOUND = "trace_not_found"
    PROFILE_DENIED = "profile_denied"
    GOVERNANCE_SCOPE_DENIED = "governance_scope_denied"
    AUTHORITY_DENIED = "authority_denied"
    SNAPSHOT_CAPABILITY_DENIED = "snapshot_capability_denied"
    SOURCE_IDENTITY_CAPABILITY_DENIED = "source_identity_capability_denied"
    PAYLOAD_UNAVAILABLE = "payload_unavailable"
    REDACTED = "redacted"
    REVOKED = "revoked"


class ActiveCitationTraceState(str, Enum):
    """Bounded active message-owner verification states."""

    ACTIVE = "active"
    BODY_MISMATCH = "body_mismatch"
    UNVERIFIABLE = "unverifiable"
    NOT_FOUND = "not_found"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class CitationTraceSummary(_FrozenModel):
    """Immutable non-governed trace data safe to read without a key."""

    namespace: TraceNamespace
    trace: CitationTrace
    visibility_state: str


class GovernedCitationPayloads(_FrozenModel):
    """Fully authorized governed payload graph."""

    evidence_run_payloads: tuple[EvidenceRunPayload, ...]
    evidence_snapshot_payloads: tuple[EvidenceSnapshotPayload, ...]
    answer_attempt_payloads: tuple[AnswerAttemptPayload, ...]


class CitationHydrationResult(_FrozenModel):
    """All-or-nothing governed hydration result."""

    state: CitationHydrationState
    summary: CitationTraceSummary | None = None
    governed_payloads: GovernedCitationPayloads | None = None


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False, init=False)
class ActiveCitationTraceResult:
    """Bounded lookup result; active summaries are repository-issued capabilities."""

    state: ActiveCitationTraceState
    summary: CitationTraceSummary | None

    def __init__(
        self,
        *,
        state: ActiveCitationTraceState,
        summary: CitationTraceSummary | None = None,
    ) -> None:
        if not isinstance(state, ActiveCitationTraceState):
            raise TypeError("state must be an ActiveCitationTraceState")
        if state is ActiveCitationTraceState.ACTIVE or summary is not None:
            raise ValueError("active summary results are repository-issued only")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "summary", None)

    def __copy__(self) -> "ActiveCitationTraceResult":
        raise TypeError("active citation results cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> "ActiveCitationTraceResult":
        del memo
        raise TypeError("active citation results cannot be copied")


@dataclass(frozen=True, slots=True)
class _ActiveTraceProof:
    """Safe opaque owner identity retained outside the public capability."""

    identity_context: LocalCitationIdentityContext
    message_id: str
    message_revision: int
    trace_id: str
    body_fingerprint: str


_SqlValue = str | int | None
_PreparedRow = tuple[_SqlValue, ...]


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class PreparedCitationWrite:
    """Immutable canonical SQL rows issued to one repository.

    The exact object may be retried after a caller-owned transaction rollback
    while it remains alive. Repository registration is removed automatically
    when the object is garbage-collected.
    """

    profile_id: str
    trace_id: str
    sealed_at: str
    selected_answer_body: str = field(repr=False)
    selected_body_fingerprint: str = field(repr=False)
    trace_row: _PreparedRow = field(repr=False)
    run_rows: tuple[_PreparedRow, ...] = field(repr=False)
    snapshot_rows: tuple[_PreparedRow, ...] = field(repr=False)
    answer_rows: tuple[_PreparedRow, ...] = field(repr=False)
    reference_rows: tuple[_PreparedRow, ...] = field(repr=False)
    identity_context: LocalCitationIdentityContext
    repository_token: object = field(repr=False)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _prepared_write_digest(prepared: PreparedCitationWrite) -> bytes:
    """Digest every immutable prepared value without exposing governed content."""

    canonical = _canonical_json(
        (
            prepared.profile_id,
            prepared.trace_id,
            prepared.sealed_at,
            prepared.selected_answer_body,
            prepared.selected_body_fingerprint,
            prepared.trace_row,
            prepared.run_rows,
            prepared.snapshot_rows,
            prepared.answer_rows,
            prepared.reference_rows,
            prepared.identity_context.model_dump(mode="json"),
        )
    )
    return hashlib.sha256(canonical.encode("utf-8")).digest()


def _active_result_digest(result: ActiveCitationTraceResult) -> bytes:
    """Digest the complete repository-issued active result."""

    if result.state is not ActiveCitationTraceState.ACTIVE or result.summary is None:
        raise ValueError("result is not active")
    canonical = _canonical_json(
        (
            result.state.value,
            result.summary.model_dump(mode="json"),
        )
    )
    return hashlib.sha256(canonical.encode("utf-8")).digest()


def load_local_citation_identity_context(
    db: CharactersRAGDB,
) -> LocalCitationIdentityContext | None:
    """Load the stable singleton without acquiring fingerprint key material."""

    row = (
        db.get_connection()
        .execute(
            """
        SELECT profile_id, local_authority_id, fingerprint_key_id
        FROM rag_identity_context
        WHERE context_name = 'default'
        """
        )
        .fetchone()
    )
    if row is None:
        return None
    return LocalCitationIdentityContext(
        profile_id=row["profile_id"],
        local_authority_id=row["local_authority_id"],
        fingerprint_key_id=row["fingerprint_key_id"],
    )


class CitationTraceRepository:
    """Persist and hydrate canonical traces through caller-owned transactions."""

    def __init__(
        self,
        db: CharactersRAGDB,
        *,
        policy: CitationProvenanceRuntimePolicy,
        identity_context: LocalCitationIdentityContext | None,
        fingerprint_codec: CitationFingerprintCodec | None,
        failure_after_row_family: str | None = None,
    ) -> None:
        if (
            failure_after_row_family is not None
            and failure_after_row_family not in _ROW_FAMILIES
        ):
            raise ValueError("unknown citation row-family failure point")
        self.db = db
        self.policy = policy
        self.identity_context = identity_context
        self._fingerprint_codec = fingerprint_codec
        self._failure_after_row_family = failure_after_row_family
        self._prepared_write_token = object()
        self._issued_prepared_writes: dict[
            int,
            tuple[weakref.ReferenceType[PreparedCitationWrite], bytes],
        ] = {}
        self._issued_active_results: dict[
            int,
            tuple[
                weakref.ReferenceType[ActiveCitationTraceResult],
                bytes,
                _ActiveTraceProof,
            ],
        ] = {}

    @classmethod
    def from_key_provider(
        cls,
        db: CharactersRAGDB,
        *,
        policy: CitationProvenanceRuntimePolicy,
        identity_context: LocalCitationIdentityContext | None,
        key_provider: CitationFingerprintKeyProvider,
    ) -> "CitationTraceRepository":
        """Compose lazily; a disabled switch never loads a key."""

        codec = None
        if policy.canonical_writes_enabled and identity_context is not None:
            try:
                codec = load_fingerprint_codec(
                    key_provider,
                    identity_context.fingerprint_key_id,
                )
            except CitationFingerprintKeyUnavailable:
                codec = None
        return cls(
            db,
            policy=policy,
            identity_context=identity_context,
            fingerprint_codec=codec,
        )

    def fingerprint_bearing_rows_exist(self) -> bool:
        """Return whether silently provisioning a replacement key is forbidden."""

        row = (
            self.db.get_connection()
            .execute(
                """
            SELECT
              EXISTS(SELECT 1 FROM rag_message_trace_owners)
              OR EXISTS(SELECT 1 FROM rag_payload_tombstones)
              OR EXISTS(SELECT 1 FROM rag_legacy_migration_journal)
              OR EXISTS(
                SELECT 1 FROM rag_citation_traces
                WHERE origin = 'imported' OR import_package_fingerprint IS NOT NULL
              )
              OR EXISTS(
                SELECT 1 FROM rag_evidence_snapshots
                WHERE content_hash IS NOT NULL OR comparison_fingerprint IS NOT NULL
              )
            """
            )
            .fetchone()
        )
        return bool(row[0])

    def prepare_write(self, sealed_write: SealedCitationWrite) -> PreparedCitationWrite:
        """Fail closed and revalidate all nested bounds before any transaction."""

        if not self.policy.canonical_writes_enabled:
            raise CitationPersistenceUnavailable("canonical_citation_writes_disabled")
        identity = self.identity_context
        if identity is None:
            raise CitationPersistenceUnavailable(
                "citation_identity_context_unavailable"
            )
        codec = self._fingerprint_codec
        if codec is None:
            raise CitationPersistenceUnavailable("fingerprint_key_unavailable")
        persisted = load_local_citation_identity_context(self.db)
        if persisted is None:
            raise CitationPersistenceUnavailable(
                "citation_identity_context_unavailable"
            )
        if persisted != identity:
            raise CitationPersistenceUnavailable("citation_identity_context_mismatch")
        try:
            validated = SealedCitationWrite.model_validate(
                sealed_write.model_dump(mode="python", round_trip=True),
                strict=True,
            )
        except Exception:
            raise CitationPersistenceUnavailable(
                "invalid_sealed_citation_write"
            ) from None
        if validated.trace.origin is not TraceOrigin.LOCAL:
            raise CitationPersistenceUnavailable("unsupported_trace_origin")
        for payload in validated.evidence_run_payloads:
            if (
                payload.authority_id is not None
                and payload.authority_id != identity.local_authority_id
            ):
                raise CitationPersistenceUnavailable("run_authority_mismatch")

        trace = validated.trace
        selected_attempt = next(
            attempt
            for attempt in trace.answer_attempts
            if attempt.attempt_id == trace.selected_attempt_id
        )
        selected_payload = next(
            (
                payload
                for payload in validated.answer_attempt_payloads
                if payload.payload_id == selected_attempt.answer_payload_ref
                and payload.attempt_id == selected_attempt.attempt_id
            ),
            None,
        )
        if (
            selected_payload is None
            or selected_payload.answer_body is None
            or selected_payload.body_integrity_hmac is None
        ):
            raise CitationPersistenceUnavailable("selected_answer_payload_unavailable")
        selected_body_fingerprint = codec.fingerprint(
            CitationFingerprintDomain.MESSAGE_BODY,
            selected_payload.answer_body,
        )
        if not hmac.compare_digest(
            selected_payload.body_integrity_hmac,
            selected_body_fingerprint,
        ):
            raise CitationPersistenceUnavailable("selected_answer_integrity_mismatch")
        profile_id = identity.profile_id
        sealed_at = trace.sealed_at.isoformat()
        run_payloads = {
            payload.payload_id: payload for payload in validated.evidence_run_payloads
        }
        run_rows = tuple(
            (
                profile_id,
                trace.trace_id,
                run.run_id,
                run.run_ordinal,
                run.stage,
                _canonical_json(run_payloads[run.payload_ref].model_dump(mode="json")),
                run.started_at.isoformat(),
                run.ended_at.isoformat() if run.ended_at else None,
            )
            for run in trace.evidence_runs
        )
        snapshot_rows = tuple(
            (
                profile_id,
                payload.payload_id,
                profile_id,
                identity.local_authority_id,
                trace.policy_version,
                payload.payload_id,
                payload.server_reference or payload.payload_id,
                payload.storage_mode.value,
                (
                    "redacted"
                    if payload.storage_mode is EvidenceStorageMode.REDACTED
                    else "available"
                ),
                payload.snapshot_text,
                payload.title,
                _canonical_json(payload.source_identity),
                _canonical_json(payload.locator),
                _canonical_json(payload.lineage),
                _canonical_json(payload.transformations),
                payload.content_hash,
                payload.comparison_hash,
                sealed_at,
            )
            for payload in validated.evidence_snapshot_payloads
        )
        answer_rows = tuple(
            (
                profile_id,
                payload.payload_id,
                trace.trace_id,
                payload.attempt_id,
                (
                    "available"
                    if payload.answer_body is not None
                    and payload.body_integrity_hmac is not None
                    else "purged"
                ),
                payload.answer_body,
                payload.body_integrity_hmac,
                sealed_at,
                (
                    None
                    if payload.answer_body is not None
                    and payload.body_integrity_hmac is not None
                    else sealed_at
                ),
            )
            for payload in validated.answer_attempt_payloads
        )
        reference_rows = tuple(
            (
                profile_id,
                trace.trace_id,
                prompt_set.prompt_set_id,
                entry.evidence_ordinal,
                entry.run_id,
                entry.snapshot_payload_ref,
                entry.marker_ordinal,
                entry.storage_mode.value,
            )
            for prompt_set in trace.prompt_evidence_sets
            for entry in prompt_set.entries
        )
        prepared = PreparedCitationWrite(
            profile_id=profile_id,
            trace_id=trace.trace_id,
            sealed_at=sealed_at,
            selected_answer_body=selected_payload.answer_body,
            selected_body_fingerprint=selected_body_fingerprint,
            trace_row=(
                profile_id,
                trace.trace_id,
                trace.schema_version,
                trace.request_id,
                trace.generation_id,
                profile_id,
                trace.origin.value,
                trace.lifecycle.value,
                trace.completeness_at_seal.value,
                trace.selected_attempt_id,
                trace.policy_version,
                _canonical_json(trace.model_dump(mode="json")),
                trace.created_at.isoformat(),
                sealed_at,
            ),
            run_rows=run_rows,
            snapshot_rows=snapshot_rows,
            answer_rows=answer_rows,
            reference_rows=reference_rows,
            identity_context=identity,
            repository_token=self._prepared_write_token,
        )
        self._register_prepared_write(prepared)
        return prepared

    def write_prepared(
        self,
        cursor: sqlite3.Cursor,
        prepared: PreparedCitationWrite,
        *,
        message_id: str,
        message_revision: int,
        message_body: str,
    ) -> None:
        """Write all row families through an already-active outer transaction."""

        execution_identity = self._require_active_write_cursor(cursor)
        if execution_identity != prepared.identity_context:
            raise CitationPersistenceUnavailable("citation_identity_context_mismatch")
        if not self._owns_prepared_write(prepared):
            raise CitationPersistenceUnavailable("prepared_citation_write_not_owned")
        codec = self._fingerprint_codec
        if codec is None:  # guarded by prepare_write
            raise CitationPersistenceUnavailable("fingerprint_key_unavailable")
        if message_body != prepared.selected_answer_body:
            raise CitationPersistenceUnavailable("selected_answer_message_mismatch")
        body_fingerprint = codec.fingerprint(
            CitationFingerprintDomain.MESSAGE_BODY,
            message_body,
        )
        if not hmac.compare_digest(
            body_fingerprint,
            prepared.selected_body_fingerprint,
        ):
            raise CitationPersistenceUnavailable("selected_answer_integrity_mismatch")
        message = cursor.execute(
            """
            SELECT id, version, content
            FROM messages
            WHERE id = ? AND deleted = 0
            """,
            (message_id,),
        ).fetchone()
        if (
            message is None
            or message["id"] != message_id
            or message["version"] != message_revision
            or message["content"] != message_body
        ):
            raise CitationPersistenceUnavailable("message_row_identity_conflict")
        self._insert_or_verify(
            cursor,
            """
            INSERT OR IGNORE INTO rag_citation_traces(
                profile_id, trace_id, schema_version, request_id, generation_id,
                origin_scope_id, origin, lifecycle, completeness_at_seal,
                selected_attempt_id, policy_version, aggregate_json,
                visibility_state, created_at, sealed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
            """,
            prepared.trace_row,
            """
            SELECT
                profile_id, trace_id, schema_version, request_id, generation_id,
                origin_scope_id, origin, lifecycle, completeness_at_seal,
                selected_attempt_id, policy_version, aggregate_json,
                visibility_state, created_at, sealed_at
            FROM rag_citation_traces
            WHERE profile_id = ? AND trace_id = ?
            """,
            (prepared.profile_id, prepared.trace_id),
            (*prepared.trace_row[:12], "active", *prepared.trace_row[12:]),
            "trace_identity_conflict",
        )
        self._fail_after("trace")

        for row in prepared.run_rows:
            self._insert_or_verify(
                cursor,
                """
                INSERT OR IGNORE INTO rag_evidence_runs(
                    profile_id, trace_id, run_id, run_ordinal, stage,
                    redaction_state, run_payload_json, started_at, ended_at, purged_at
                ) VALUES (?, ?, ?, ?, ?, 'available', ?, ?, ?, NULL)
                """,
                row,
                """
                SELECT
                    profile_id, trace_id, run_id, run_ordinal, stage,
                    redaction_state, run_payload_json, started_at, ended_at, purged_at
                FROM rag_evidence_runs
                WHERE profile_id = ? AND trace_id = ? AND run_id = ?
                """,
                row[:3],
                (*row[:5], "available", *row[5:], None),
                "run_identity_conflict",
            )
        self._fail_after("runs")

        for row in prepared.snapshot_rows:
            self._insert_or_verify(
                cursor,
                """
                INSERT OR IGNORE INTO rag_evidence_snapshots(
                    profile_id, payload_id, governance_scope_id, authority_id,
                    confidentiality_policy_id, revocation_scope_id,
                    origin_namespace, origin_payload_id, storage_mode,
                    redaction_state, retention_class, snapshot_text, title,
                    source_identity_json, locator_json, lineage_json,
                    transformations_json, content_hash, comparison_fingerprint,
                    created_at, retain_until, purged_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, 'local_payload_v1', ?, ?, ?, 'default',
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL
                )
                """,
                row,
                """
                SELECT
                    profile_id, payload_id, governance_scope_id, authority_id,
                    confidentiality_policy_id, revocation_scope_id,
                    origin_namespace, origin_payload_id, storage_mode,
                    redaction_state, retention_class, snapshot_text, title,
                    source_identity_json, locator_json, lineage_json,
                    transformations_json, content_hash, comparison_fingerprint,
                    created_at, retain_until, purged_at
                FROM rag_evidence_snapshots
                WHERE profile_id = ? AND payload_id = ?
                """,
                row[:2],
                (
                    *row[:6],
                    "local_payload_v1",
                    *row[6:9],
                    "default",
                    *row[9:],
                    None,
                    None,
                ),
                "snapshot_identity_conflict",
            )
        self._fail_after("snapshots")

        for row in prepared.answer_rows:
            self._insert_or_verify(
                cursor,
                """
                INSERT OR IGNORE INTO rag_answer_attempt_payloads(
                    profile_id, payload_id, trace_id, attempt_id,
                    redaction_state, retention_class, answer_body,
                    body_integrity_hmac, created_at, retain_until, purged_at
                ) VALUES (?, ?, ?, ?, ?, 'default', ?, ?, ?, NULL, ?)
                """,
                row,
                """
                SELECT
                    profile_id, payload_id, trace_id, attempt_id,
                    redaction_state, retention_class, answer_body,
                    body_integrity_hmac, created_at, retain_until, purged_at
                FROM rag_answer_attempt_payloads
                WHERE profile_id = ? AND payload_id = ?
                """,
                row[:2],
                (*row[:5], "default", *row[5:8], None, row[8]),
                "answer_identity_conflict",
            )
        self._fail_after("attempts")

        for row in prepared.reference_rows:
            self._insert_or_verify(
                cursor,
                """
                INSERT OR IGNORE INTO rag_trace_evidence_refs(
                    profile_id, trace_id, prompt_set_id, evidence_ordinal,
                    run_id, snapshot_payload_id, marker_ordinal, storage_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
                """
                SELECT
                    profile_id, trace_id, prompt_set_id, evidence_ordinal,
                    run_id, snapshot_payload_id, marker_ordinal, storage_mode
                FROM rag_trace_evidence_refs
                WHERE profile_id = ? AND trace_id = ?
                  AND prompt_set_id = ? AND evidence_ordinal = ?
                """,
                row[:4],
                row,
                "reference_identity_conflict",
            )
        self._fail_after("refs")

        namespace = local_trace_namespace(
            prepared.identity_context,
            trace_id=prepared.trace_id,
        )
        idempotency_key = message_owner_idempotency_key(
            codec,
            namespace,
            message_id=message_id,
            message_revision=message_revision,
        )
        prior_owners = cursor.execute(
            """
            SELECT message_revision, state, body_fingerprint, idempotency_key
            FROM rag_message_trace_owners
            WHERE profile_id = ? AND message_id = ? AND trace_id = ?
            """,
            (prepared.profile_id, message_id, prepared.trace_id),
        ).fetchall()
        if prior_owners and not any(
            row["message_revision"] == message_revision
            and row["state"] == "active"
            and hmac.compare_digest(row["body_fingerprint"], body_fingerprint)
            and hmac.compare_digest(row["idempotency_key"], idempotency_key)
            for row in prior_owners
        ):
            raise CitationPersistenceUnavailable("owner_identity_conflict")
        self._insert_owner(
            cursor,
            profile_id=prepared.profile_id,
            message_id=message_id,
            message_revision=message_revision,
            trace_id=prepared.trace_id,
            body_fingerprint=body_fingerprint,
            idempotency_key=idempotency_key,
            timestamp=prepared.sealed_at,
        )
        self._fail_after("owner")

    @staticmethod
    def _insert_or_verify(
        cursor: sqlite3.Cursor,
        insert_sql: str,
        insert_params: tuple[Any, ...],
        select_sql: str,
        select_params: tuple[Any, ...],
        expected: tuple[Any, ...],
        reason_code: str,
    ) -> None:
        """Insert once or prove an existing immutable row is byte-identical."""

        cursor.execute(insert_sql, insert_params)
        row = cursor.execute(select_sql, select_params).fetchone()
        if row is None or tuple(row) != expected:
            raise CitationPersistenceUnavailable(reason_code)

    def _insert_owner(
        self,
        cursor: sqlite3.Cursor,
        *,
        profile_id: str,
        message_id: str,
        message_revision: int,
        trace_id: str,
        body_fingerprint: str,
        idempotency_key: str,
        timestamp: str,
    ) -> None:
        cursor.execute(
            """
            INSERT OR IGNORE INTO rag_message_trace_owners(
                profile_id, message_id, message_revision, trace_id, state,
                body_fingerprint, idempotency_key, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?)
            """,
            (
                profile_id,
                message_id,
                message_revision,
                trace_id,
                body_fingerprint,
                idempotency_key,
                timestamp,
                timestamp,
            ),
        )
        row = cursor.execute(
            """
            SELECT
                profile_id, message_id, message_revision, trace_id, state,
                body_fingerprint, idempotency_key
            FROM rag_message_trace_owners
            WHERE profile_id = ? AND message_id = ?
              AND message_revision = ? AND trace_id = ?
            """,
            (profile_id, message_id, message_revision, trace_id),
        ).fetchone()
        expected = (
            profile_id,
            message_id,
            message_revision,
            trace_id,
            "active",
            body_fingerprint,
            idempotency_key,
        )
        if row is None or tuple(row) != expected:
            raise CitationPersistenceUnavailable("owner_identity_conflict")

    def link_cache_message_owner(
        self,
        cursor: sqlite3.Cursor,
        namespace: TraceNamespace,
        *,
        message_id: str,
        message_revision: int,
        message_body: str,
    ) -> None:
        """Idempotently link a cache-hit message to the original local trace."""

        identity = self._require_active_write_cursor(cursor)
        codec = self._fingerprint_codec
        if codec is None:
            raise CitationPersistenceUnavailable("fingerprint_key_unavailable")
        expected_namespace = local_trace_namespace(
            identity,
            trace_id=namespace.trace_id or "",
            wire_schema_version=namespace.wire_schema_version,
        )
        if namespace != expected_namespace:
            raise CitationPersistenceUnavailable("cache_namespace_identity_conflict")
        message = cursor.execute(
            """
            SELECT version, content
            FROM messages
            WHERE id = ? AND deleted = 0
            """,
            (message_id,),
        ).fetchone()
        if (
            message is None
            or message["version"] != message_revision
            or message["content"] != message_body
        ):
            raise CitationPersistenceUnavailable("cache_owner_identity_conflict")
        body_fingerprint = codec.fingerprint(
            CitationFingerprintDomain.MESSAGE_BODY,
            message_body,
        )
        trace = cursor.execute(
            """
            SELECT
                trace.visibility_state,
                payload.redaction_state,
                payload.answer_body,
                payload.body_integrity_hmac,
                payload.purged_at
            FROM rag_citation_traces AS trace
            LEFT JOIN rag_answer_attempt_payloads AS payload
              ON payload.profile_id = trace.profile_id
             AND payload.trace_id = trace.trace_id
             AND payload.attempt_id = trace.selected_attempt_id
            WHERE trace.profile_id = ? AND trace.trace_id = ?
              AND trace.origin = 'local' AND trace.origin_scope_id = ?
            """,
            (identity.profile_id, namespace.trace_id, identity.profile_id),
        ).fetchone()
        if trace is None:
            raise CitationPersistenceUnavailable("cache_trace_identity_conflict")
        if trace["visibility_state"] != "active":
            raise CitationPersistenceUnavailable("cache_trace_identity_conflict")
        if (
            trace["redaction_state"] != "available"
            or trace["answer_body"] is None
            or trace["body_integrity_hmac"] is None
            or trace["purged_at"] is not None
        ):
            raise CitationPersistenceUnavailable("cache_selected_answer_unavailable")
        if trace["answer_body"] != message_body:
            raise CitationPersistenceUnavailable("cache_selected_answer_mismatch")
        if not hmac.compare_digest(
            trace["body_integrity_hmac"],
            body_fingerprint,
        ):
            raise CitationPersistenceUnavailable(
                "cache_selected_answer_integrity_mismatch"
            )
        still_active = cursor.execute(
            """
            SELECT 1
            FROM rag_citation_traces
            WHERE profile_id = ? AND trace_id = ?
              AND origin = 'local' AND origin_scope_id = ?
              AND visibility_state = 'active'
            """,
            (identity.profile_id, namespace.trace_id, identity.profile_id),
        ).fetchone()
        if still_active is None:
            raise CitationPersistenceUnavailable("cache_trace_identity_conflict")
        self._insert_owner(
            cursor,
            profile_id=identity.profile_id,
            message_id=message_id,
            message_revision=message_revision,
            trace_id=namespace.trace_id or "",
            body_fingerprint=body_fingerprint,
            idempotency_key=cache_owner_idempotency_key(
                codec,
                namespace,
                message_id=message_id,
                message_revision=message_revision,
            ),
            timestamp=datetime.now(UTC).isoformat(),
        )

    def get_active_trace_for_message(
        self,
        message_id: str,
        revision: int,
        current_body: str,
        codec: CitationFingerprintCodec | None,
    ) -> ActiveCitationTraceResult:
        """Return an active trace only after keyed owner-body verification."""

        identity = self.identity_context
        if (
            identity is None
            or codec is None
            or self._fingerprint_codec is None
            or not self._codec_matches(codec)
        ):
            return ActiveCitationTraceResult(
                state=ActiveCitationTraceState.UNVERIFIABLE
            )
        persisted = load_local_citation_identity_context(self.db)
        if persisted != identity:
            return ActiveCitationTraceResult(
                state=ActiveCitationTraceState.UNVERIFIABLE
            )
        row = (
            self.db.get_connection()
            .execute(
                """
                SELECT
                    owner.trace_id, owner.state, owner.body_fingerprint,
                    message.id AS persisted_message_id,
                    message.version AS persisted_revision,
                    message.content AS persisted_body
                FROM rag_message_trace_owners AS owner
                LEFT JOIN messages AS message
                  ON message.id = owner.message_id AND message.deleted = 0
                WHERE owner.profile_id = ?
                  AND owner.message_id = ?
                  AND owner.message_revision = ?
                ORDER BY (owner.state = 'active') DESC, owner.trace_id
                LIMIT 1
                """,
                (identity.profile_id, message_id, revision),
            )
            .fetchone()
        )
        if row is None or row["state"] == "deleted":
            return ActiveCitationTraceResult(state=ActiveCitationTraceState.NOT_FOUND)
        if (
            row["persisted_message_id"] is None
            or row["persisted_revision"] != revision
            or row["persisted_body"] != current_body
        ):
            return ActiveCitationTraceResult(
                state=ActiveCitationTraceState.UNVERIFIABLE
            )
        if row["state"] == "body_mismatch":
            return ActiveCitationTraceResult(
                state=ActiveCitationTraceState.BODY_MISMATCH
            )
        fingerprint = codec.fingerprint(
            CitationFingerprintDomain.MESSAGE_BODY,
            current_body,
        )
        if not hmac.compare_digest(row["body_fingerprint"], fingerprint):
            with self.db.transaction() as cursor:
                cursor.execute(
                    """
                    UPDATE rag_message_trace_owners
                    SET state = 'body_mismatch', updated_at = ?
                    WHERE profile_id = ? AND message_id = ?
                      AND message_revision = ? AND trace_id = ?
                      AND state = 'active' AND body_fingerprint = ?
                    """,
                    (
                        datetime.now(UTC).isoformat(),
                        identity.profile_id,
                        message_id,
                        revision,
                        row["trace_id"],
                        row["body_fingerprint"],
                    ),
                )
            return ActiveCitationTraceResult(
                state=ActiveCitationTraceState.BODY_MISMATCH
            )
        summary = self.get_trace_summary(
            local_trace_namespace(identity, trace_id=row["trace_id"])
        )
        if summary is None or summary.visibility_state != "active":
            return ActiveCitationTraceResult(state=ActiveCitationTraceState.NOT_FOUND)
        return self._issue_active_trace_result(
            summary,
            proof=_ActiveTraceProof(
                identity_context=identity,
                message_id=message_id,
                message_revision=revision,
                trace_id=row["trace_id"],
                body_fingerprint=fingerprint,
            ),
        )

    def verify_active_trace_result(
        self,
        result: ActiveCitationTraceResult,
    ) -> bool:
        """Verify exact repository issuance and integrity before presentation."""

        if not isinstance(result, ActiveCitationTraceResult):
            return False
        issued = self._issued_active_results.get(id(result))
        if issued is None or issued[0]() is not result:
            return False
        try:
            current_digest = _active_result_digest(result)
        except (AttributeError, TypeError, ValueError):
            return self._invalidate_active_trace_result(result)
        if not hmac.compare_digest(issued[1], current_digest):
            return self._invalidate_active_trace_result(result)
        proof = issued[2]
        codec = self._fingerprint_codec
        if codec is None or self.identity_context != proof.identity_context:
            return self._invalidate_active_trace_result(result)
        row = (
            self.db.get_connection()
            .execute(
                """
                SELECT
                    owner.state,
                    owner.body_fingerprint,
                    message.version AS message_revision,
                    message.content AS message_body,
                    trace.visibility_state,
                    trace.origin,
                    trace.origin_scope_id,
                    identity.local_authority_id,
                    identity.fingerprint_key_id
                FROM rag_message_trace_owners AS owner
                JOIN messages AS message
                  ON message.id = owner.message_id AND message.deleted = 0
                JOIN rag_citation_traces AS trace
                  ON trace.profile_id = owner.profile_id
                 AND trace.trace_id = owner.trace_id
                JOIN rag_identity_context AS identity
                  ON identity.context_name = 'default'
                 AND identity.profile_id = owner.profile_id
                WHERE owner.profile_id = ?
                  AND owner.message_id = ?
                  AND owner.message_revision = ?
                  AND owner.trace_id = ?
                """,
                (
                    proof.identity_context.profile_id,
                    proof.message_id,
                    proof.message_revision,
                    proof.trace_id,
                ),
            )
            .fetchone()
        )
        if (
            row is None
            or row["state"] != "active"
            or row["message_revision"] != proof.message_revision
            or row["visibility_state"] != "active"
            or row["origin"] != "local"
            or row["origin_scope_id"] != proof.identity_context.profile_id
            or row["local_authority_id"] != proof.identity_context.local_authority_id
            or row["fingerprint_key_id"] != proof.identity_context.fingerprint_key_id
        ):
            return self._invalidate_active_trace_result(result)
        current_fingerprint = codec.fingerprint(
            CitationFingerprintDomain.MESSAGE_BODY,
            row["message_body"],
        )
        if not (
            hmac.compare_digest(
                row["body_fingerprint"],
                proof.body_fingerprint,
            )
            and hmac.compare_digest(
                current_fingerprint,
                proof.body_fingerprint,
            )
        ):
            return self._invalidate_active_trace_result(result)
        return True

    def transition_owner_for_message_update(
        self,
        cursor: sqlite3.Cursor,
        *,
        message_id: str,
        previous_revision: int,
        new_revision: int,
        new_body: str,
    ) -> ActiveCitationTraceState:
        """Carry forward or invalidate the prior active owner in the edit tx."""

        identity = self._require_repository_cursor(cursor)
        row = cursor.execute(
            """
            SELECT trace_id, body_fingerprint
            FROM rag_message_trace_owners
            WHERE profile_id = ? AND message_id = ?
              AND message_revision = ? AND state = 'active'
            ORDER BY trace_id
            LIMIT 1
            """,
            (identity.profile_id, message_id, previous_revision),
        ).fetchone()
        if row is None:
            return ActiveCitationTraceState.NOT_FOUND
        message = cursor.execute(
            """
            SELECT version, content
            FROM messages
            WHERE id = ? AND deleted = 0
            """,
            (message_id,),
        ).fetchone()
        if (
            message is None
            or message["version"] != new_revision
            or message["content"] != new_body
        ):
            raise CitationPersistenceUnavailable("message_revision_identity_conflict")
        codec = self._fingerprint_codec
        if codec is None:
            self._mark_owner_body_mismatch(
                cursor,
                profile_id=identity.profile_id,
                message_id=message_id,
                message_revision=previous_revision,
                trace_id=row["trace_id"],
            )
            return ActiveCitationTraceState.UNVERIFIABLE
        new_fingerprint = codec.fingerprint(
            CitationFingerprintDomain.MESSAGE_BODY,
            new_body,
        )
        if not hmac.compare_digest(row["body_fingerprint"], new_fingerprint):
            self._mark_owner_body_mismatch(
                cursor,
                profile_id=identity.profile_id,
                message_id=message_id,
                message_revision=previous_revision,
                trace_id=row["trace_id"],
            )
            return ActiveCitationTraceState.BODY_MISMATCH
        namespace = local_trace_namespace(identity, trace_id=row["trace_id"])
        timestamp = datetime.now(UTC).isoformat()
        self._insert_owner(
            cursor,
            profile_id=identity.profile_id,
            message_id=message_id,
            message_revision=new_revision,
            trace_id=row["trace_id"],
            body_fingerprint=new_fingerprint,
            idempotency_key=message_owner_idempotency_key(
                codec,
                namespace,
                message_id=message_id,
                message_revision=new_revision,
            ),
            timestamp=timestamp,
        )
        return ActiveCitationTraceState.ACTIVE

    @staticmethod
    def _mark_owner_body_mismatch(
        cursor: sqlite3.Cursor,
        *,
        profile_id: str,
        message_id: str,
        message_revision: int,
        trace_id: str,
    ) -> None:
        cursor.execute(
            """
            UPDATE rag_message_trace_owners
            SET state = 'body_mismatch', updated_at = ?
            WHERE profile_id = ? AND message_id = ?
              AND message_revision = ? AND trace_id = ? AND state = 'active'
            """,
            (
                datetime.now(UTC).isoformat(),
                profile_id,
                message_id,
                message_revision,
                trace_id,
            ),
        )

    def _codec_matches(self, codec: CitationFingerprintCodec) -> bool:
        expected = self._fingerprint_codec
        if expected is None:
            return False
        sentinel = b"citation-codec-key-check-v1"
        return hmac.compare_digest(
            expected.fingerprint(CitationFingerprintDomain.EXACT_PAYLOAD, sentinel),
            codec.fingerprint(CitationFingerprintDomain.EXACT_PAYLOAD, sentinel),
        )

    def _require_repository_cursor(
        self,
        cursor: sqlite3.Cursor,
    ) -> LocalCitationIdentityContext:
        if cursor.connection is not self.db.get_connection():
            raise RuntimeError(
                "citation persistence requires the repository database transaction"
            )
        if not cursor.connection.in_transaction:
            raise RuntimeError("citation persistence requires an active transaction")
        persisted = load_local_citation_identity_context(self.db)
        if persisted is None:
            raise CitationPersistenceUnavailable(
                "citation_identity_context_unavailable"
            )
        identity = self.identity_context
        if identity is not None and identity != persisted:
            raise CitationPersistenceUnavailable("citation_identity_context_mismatch")
        return persisted

    def _require_active_write_cursor(
        self,
        cursor: sqlite3.Cursor,
    ) -> LocalCitationIdentityContext:
        if not self.policy.canonical_writes_enabled:
            raise CitationPersistenceUnavailable("canonical_citation_writes_disabled")
        identity = self._require_repository_cursor(cursor)
        if self.identity_context is None:
            raise CitationPersistenceUnavailable(
                "citation_identity_context_unavailable"
            )
        return identity

    def _register_prepared_write(self, prepared: PreparedCitationWrite) -> None:
        """Bind exact object identity and canonical rows until object collection."""

        prepared_id = id(prepared)
        repository_ref = weakref.ref(self)

        def discard(
            prepared_ref: weakref.ReferenceType[PreparedCitationWrite],
        ) -> None:
            repository = repository_ref()
            if repository is None:
                return
            issued = repository._issued_prepared_writes.get(prepared_id)
            if issued is not None and issued[0] is prepared_ref:
                repository._issued_prepared_writes.pop(prepared_id, None)

        prepared_ref = weakref.ref(prepared, discard)
        self._issued_prepared_writes[prepared_id] = (
            prepared_ref,
            _prepared_write_digest(prepared),
        )

    def _issue_active_trace_result(
        self,
        summary: CitationTraceSummary,
        *,
        proof: _ActiveTraceProof,
    ) -> ActiveCitationTraceResult:
        """Issue and register one exact active-result capability."""

        result = object.__new__(ActiveCitationTraceResult)
        object.__setattr__(result, "state", ActiveCitationTraceState.ACTIVE)
        object.__setattr__(result, "summary", summary)
        result_id = id(result)
        repository_ref = weakref.ref(self)

        def discard(
            result_ref: weakref.ReferenceType[ActiveCitationTraceResult],
        ) -> None:
            repository = repository_ref()
            if repository is None:
                return
            issued = repository._issued_active_results.get(result_id)
            if issued is not None and issued[0] is result_ref:
                repository._issued_active_results.pop(result_id, None)

        result_ref = weakref.ref(result, discard)
        self._issued_active_results[result_id] = (
            result_ref,
            _active_result_digest(result),
            proof,
        )
        return result

    def _invalidate_active_trace_result(
        self,
        result: ActiveCitationTraceResult,
    ) -> bool:
        """Drop registration for an exact failed or stale capability."""

        issued = self._issued_active_results.get(id(result))
        if issued is not None and issued[0]() is result:
            self._issued_active_results.pop(id(result), None)
        return False

    def _owns_prepared_write(self, prepared: PreparedCitationWrite) -> bool:
        """Verify repository, exact object identity, and every prepared row."""

        if (
            prepared.repository_token is not self._prepared_write_token
            or prepared.identity_context != self.identity_context
        ):
            return False
        issued = self._issued_prepared_writes.get(id(prepared))
        if issued is None or issued[0]() is not prepared:
            return False
        try:
            current_digest = _prepared_write_digest(prepared)
        except (TypeError, ValueError):
            return False
        return hmac.compare_digest(issued[1], current_digest)

    def get_trace_summary(
        self,
        namespace: TraceNamespace,
    ) -> CitationTraceSummary | None:
        """Read immutable non-governed metadata without identity/key material."""

        where, params = self._trace_selector(namespace)
        row = (
            self.db.get_connection()
            .execute(
                f"""
            SELECT aggregate_json, visibility_state
            FROM rag_citation_traces
            WHERE {where}
            """,
                params,
            )
            .fetchone()
        )
        if row is None:
            return None
        return CitationTraceSummary(
            namespace=namespace,
            trace=CitationTrace.model_validate_json(row["aggregate_json"]),
            visibility_state=row["visibility_state"],
        )

    def hydrate_trace(
        self,
        namespace: TraceNamespace,
        *,
        authorization: CitationReadAuthorization,
    ) -> CitationHydrationResult:
        """Hydrate governed fields only after every authorization precondition."""

        summary = self.get_trace_summary(namespace)
        if summary is None:
            return CitationHydrationResult(state=CitationHydrationState.TRACE_NOT_FOUND)
        if not self._profile_permitted(namespace, authorization):
            return CitationHydrationResult(
                state=CitationHydrationState.PROFILE_DENIED,
                summary=summary,
            )
        if not self._governance_scope_permitted(namespace, authorization):
            return CitationHydrationResult(
                state=CitationHydrationState.GOVERNANCE_SCOPE_DENIED,
                summary=summary,
            )
        if namespace.authority_id not in authorization.allowlisted_authority_ids:
            return CitationHydrationResult(
                state=CitationHydrationState.AUTHORITY_DENIED,
                summary=summary,
            )
        if (
            PolicyCapability.VIEW_SNAPSHOT not in summary.trace.policy_capabilities
            or not authorization.view_snapshot
        ):
            return CitationHydrationResult(
                state=CitationHydrationState.SNAPSHOT_CAPABILITY_DENIED,
                summary=summary,
            )
        if (
            PolicyCapability.VIEW_SOURCE_IDENTITY
            not in summary.trace.policy_capabilities
            or not authorization.view_source_identity
        ):
            return CitationHydrationResult(
                state=CitationHydrationState.SOURCE_IDENTITY_CAPABILITY_DENIED,
                summary=summary,
            )

        profile_id = namespace.profile_id
        trace_id = summary.trace.trace_id
        connection = self.db.get_connection()
        default_run_authority_id = namespace.authority_id
        required_run_authority_id: str | None = None
        if namespace.identity_namespace is CitationIdentityNamespace.LOCAL_TRACE:
            local_identity = load_local_citation_identity_context(self.db)
            if (
                local_identity is None
                or local_identity.profile_id != namespace.profile_id
                or local_identity.local_authority_id != namespace.authority_id
            ):
                return CitationHydrationResult(
                    state=CitationHydrationState.AUTHORITY_DENIED,
                    summary=summary,
                )
            default_run_authority_id = local_identity.local_authority_id
            required_run_authority_id = local_identity.local_authority_id
        snapshot_metadata = connection.execute(
            """
            SELECT DISTINCT
                s.payload_id, s.governance_scope_id, s.authority_id,
                s.origin_namespace, s.origin_payload_id, s.redaction_state
            FROM rag_trace_evidence_refs r
            JOIN rag_evidence_snapshots s
              ON s.profile_id = r.profile_id
             AND s.payload_id = r.snapshot_payload_id
            WHERE r.profile_id = ? AND r.trace_id = ?
            ORDER BY s.payload_id
            """,
            (profile_id, trace_id),
        ).fetchall()
        expected_snapshot_ids = {
            entry.snapshot_payload_ref
            for prompt_set in summary.trace.prompt_evidence_sets
            for entry in prompt_set.entries
        }
        if {row["payload_id"] for row in snapshot_metadata} != expected_snapshot_ids:
            return CitationHydrationResult(
                state=CitationHydrationState.PAYLOAD_UNAVAILABLE,
                summary=summary,
            )
        for row in snapshot_metadata:
            if row["governance_scope_id"] != authorization.governance_scope_id:
                return CitationHydrationResult(
                    state=CitationHydrationState.GOVERNANCE_SCOPE_DENIED,
                    summary=summary,
                )
            if row["authority_id"] not in authorization.allowlisted_authority_ids:
                return CitationHydrationResult(
                    state=CitationHydrationState.AUTHORITY_DENIED,
                    summary=summary,
                )
            if row["redaction_state"] != "available":
                return CitationHydrationResult(
                    state=CitationHydrationState.REDACTED,
                    summary=summary,
                )
            tombstone = connection.execute(
                """
                SELECT 1
                FROM rag_payload_tombstones
                WHERE profile_id = ?
                  AND origin_namespace = ?
                  AND origin_payload_id = ?
                """,
                (
                    profile_id,
                    row["origin_namespace"],
                    row["origin_payload_id"],
                ),
            ).fetchone()
            if tombstone is not None:
                return CitationHydrationResult(
                    state=CitationHydrationState.REVOKED,
                    summary=summary,
                )

        run_metadata = connection.execute(
            """
            SELECT
                run_id,
                redaction_state,
                CASE
                  WHEN json_valid(run_payload_json)
                   AND json_type(run_payload_json) = 'object'
                  THEN 1
                  ELSE 0
                END AS payload_json_valid,
                CASE
                  WHEN json_valid(run_payload_json)
                  THEN json_extract(run_payload_json, '$.authority_id')
                  ELSE NULL
                END AS authority_id,
                CASE
                  WHEN json_valid(run_payload_json)
                  THEN json_type(run_payload_json, '$.authority_id')
                  ELSE NULL
                END AS authority_json_type
            FROM rag_evidence_runs
            WHERE profile_id = ? AND trace_id = ?
            """,
            (profile_id, trace_id),
        ).fetchall()
        expected_run_ids = {run.run_id for run in summary.trace.evidence_runs}
        if {row["run_id"] for row in run_metadata} != expected_run_ids:
            return CitationHydrationResult(
                state=CitationHydrationState.PAYLOAD_UNAVAILABLE,
                summary=summary,
            )
        answer_metadata = connection.execute(
            """
            SELECT attempt_id, redaction_state
            FROM rag_answer_attempt_payloads
            WHERE profile_id = ? AND trace_id = ?
            """,
            (profile_id, trace_id),
        ).fetchall()
        expected_answer_attempt_ids = {
            attempt.attempt_id
            for attempt in summary.trace.answer_attempts
            if attempt.answer_payload_ref is not None
        }
        if {
            row["attempt_id"] for row in answer_metadata
        } != expected_answer_attempt_ids:
            return CitationHydrationResult(
                state=CitationHydrationState.PAYLOAD_UNAVAILABLE,
                summary=summary,
            )
        if any(
            row["redaction_state"] != "available"
            for row in (*run_metadata, *answer_metadata)
        ):
            return CitationHydrationResult(
                state=CitationHydrationState.REDACTED,
                summary=summary,
            )
        if any(
            not row["payload_json_valid"]
            or row["authority_json_type"] not in (None, "null", "text")
            for row in run_metadata
        ):
            return CitationHydrationResult(
                state=CitationHydrationState.PAYLOAD_UNAVAILABLE,
                summary=summary,
            )
        run_authority_ids = tuple(
            (
                row["authority_id"]
                if row["authority_id"] is not None
                else default_run_authority_id
            )
            for row in run_metadata
        )
        if any(
            (
                required_run_authority_id is not None
                and authority_id != required_run_authority_id
            )
            or authority_id not in authorization.allowlisted_authority_ids
            for authority_id in run_authority_ids
        ):
            return CitationHydrationResult(
                state=CitationHydrationState.AUTHORITY_DENIED,
                summary=summary,
            )

        expected_runs = tuple(
            {
                "run_id": run.run_id,
                "run_ordinal": run.run_ordinal,
                "payload_id": run.payload_ref,
            }
            for run in summary.trace.evidence_runs
        )
        expected_snapshots = tuple(
            {
                "payload_id": entry.snapshot_payload_ref,
                "storage_mode": entry.storage_mode.value,
            }
            for prompt_set in summary.trace.prompt_evidence_sets
            for entry in prompt_set.entries
        )
        expected_answers = tuple(
            {
                "payload_id": attempt.answer_payload_ref,
                "attempt_id": attempt.attempt_id,
                "attempt_ordinal": attempt.attempt_ordinal,
            }
            for attempt in summary.trace.answer_attempts
            if attempt.answer_payload_ref is not None
        )
        expected_references = tuple(
            {
                "prompt_set_id": prompt_set.prompt_set_id,
                "evidence_ordinal": entry.evidence_ordinal,
                "run_id": entry.run_id,
                "snapshot_payload_id": entry.snapshot_payload_ref,
                "marker_ordinal": entry.marker_ordinal,
                "storage_mode": entry.storage_mode.value,
            }
            for prompt_set in summary.trace.prompt_evidence_sets
            for entry in prompt_set.entries
        )
        self._before_governed_select()
        governed_rows = connection.execute(
            """
            WITH
            expected_runs AS (
                SELECT
                    json_extract(value, '$.run_id') AS run_id,
                    json_extract(value, '$.run_ordinal') AS run_ordinal,
                    json_extract(value, '$.payload_id') AS payload_id
                FROM json_each(?)
            ),
            expected_snapshots AS (
                SELECT DISTINCT
                    json_extract(value, '$.payload_id') AS payload_id,
                    json_extract(value, '$.storage_mode') AS storage_mode
                FROM json_each(?)
            ),
            expected_answers AS (
                SELECT
                    json_extract(value, '$.payload_id') AS payload_id,
                    json_extract(value, '$.attempt_id') AS attempt_id,
                    json_extract(value, '$.attempt_ordinal') AS attempt_ordinal
                FROM json_each(?)
            ),
            expected_references AS (
                SELECT
                    json_extract(value, '$.prompt_set_id') AS prompt_set_id,
                    json_extract(value, '$.evidence_ordinal') AS evidence_ordinal,
                    json_extract(value, '$.run_id') AS run_id,
                    json_extract(value, '$.snapshot_payload_id')
                        AS snapshot_payload_id,
                    json_extract(value, '$.marker_ordinal') AS marker_ordinal,
                    json_extract(value, '$.storage_mode') AS storage_mode
                FROM json_each(?)
            ),
            allowed_authorities AS (
                SELECT value AS authority_id
                FROM json_each(?)
            ),
            invalid(reason) AS (
                SELECT 'trace'
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM rag_citation_traces t
                    WHERE t.profile_id = ?
                      AND t.trace_id = ?
                      AND t.origin = ?
                      AND t.origin_scope_id = ?
                      AND t.aggregate_json = ?
                      AND t.visibility_state = ?
                )
                UNION ALL
                SELECT 'run_count'
                WHERE (
                    SELECT count(*)
                    FROM rag_evidence_runs r
                    WHERE r.profile_id = ? AND r.trace_id = ?
                ) != (SELECT count(*) FROM expected_runs)
                UNION ALL
                SELECT 'run'
                WHERE EXISTS (
                    SELECT 1
                    FROM expected_runs e
                    LEFT JOIN rag_evidence_runs r
                      ON r.profile_id = ?
                     AND r.trace_id = ?
                     AND r.run_id = e.run_id
                    WHERE r.run_id IS NULL
                       OR r.run_ordinal != e.run_ordinal
                       OR r.redaction_state != 'available'
                       OR r.purged_at IS NOT NULL
                       OR NOT json_valid(r.run_payload_json)
                       OR json_type(r.run_payload_json) != 'object'
                       OR json_extract(r.run_payload_json, '$.payload_id')
                            IS NOT e.payload_id
                       OR json_extract(r.run_payload_json, '$.run_id')
                            IS NOT e.run_id
                       OR json_type(r.run_payload_json, '$.authority_id')
                            NOT IN ('null', 'text')
                       OR COALESCE(
                            json_extract(r.run_payload_json, '$.authority_id'),
                            ?
                       ) NOT IN (
                            SELECT authority_id FROM allowed_authorities
                       )
                       OR (
                            ? IS NOT NULL
                            AND COALESCE(
                                json_extract(
                                    r.run_payload_json,
                                    '$.authority_id'
                                ),
                                ?
                            ) != ?
                       )
                )
                UNION ALL
                SELECT 'reference_count'
                WHERE (
                    SELECT count(*)
                    FROM rag_trace_evidence_refs r
                    WHERE r.profile_id = ? AND r.trace_id = ?
                ) != (SELECT count(*) FROM expected_references)
                UNION ALL
                SELECT 'reference'
                WHERE EXISTS (
                    SELECT 1
                    FROM expected_references e
                    LEFT JOIN rag_trace_evidence_refs r
                      ON r.profile_id = ?
                     AND r.trace_id = ?
                     AND r.prompt_set_id = e.prompt_set_id
                     AND r.evidence_ordinal = e.evidence_ordinal
                     AND r.run_id = e.run_id
                     AND r.snapshot_payload_id = e.snapshot_payload_id
                     AND r.marker_ordinal = e.marker_ordinal
                     AND r.storage_mode = e.storage_mode
                    WHERE r.prompt_set_id IS NULL
                )
                UNION ALL
                SELECT 'snapshot'
                WHERE EXISTS (
                    SELECT 1
                    FROM expected_snapshots e
                    LEFT JOIN rag_evidence_snapshots s
                      ON s.profile_id = ?
                     AND s.payload_id = e.payload_id
                    WHERE s.payload_id IS NULL
                       OR s.storage_mode != e.storage_mode
                       OR s.governance_scope_id != ?
                       OR s.authority_id NOT IN (
                            SELECT authority_id FROM allowed_authorities
                       )
                       OR s.redaction_state != 'available'
                       OR s.purged_at IS NOT NULL
                       OR (
                            s.source_identity_json IS NOT NULL
                            AND (
                                NOT json_valid(s.source_identity_json)
                                OR json_type(s.source_identity_json) != 'object'
                            )
                       )
                       OR (
                            s.locator_json IS NOT NULL
                            AND (
                                NOT json_valid(s.locator_json)
                                OR json_type(s.locator_json) != 'object'
                            )
                       )
                       OR (
                            s.lineage_json IS NOT NULL
                            AND (
                                NOT json_valid(s.lineage_json)
                                OR json_type(s.lineage_json) != 'object'
                            )
                       )
                       OR (
                            s.transformations_json IS NOT NULL
                            AND (
                                NOT json_valid(s.transformations_json)
                                OR json_type(s.transformations_json) != 'array'
                            )
                       )
                       OR EXISTS (
                            SELECT 1
                            FROM rag_payload_tombstones tombstone
                            WHERE tombstone.profile_id = s.profile_id
                              AND tombstone.origin_namespace =
                                    s.origin_namespace
                              AND tombstone.origin_payload_id =
                                    s.origin_payload_id
                       )
                )
                UNION ALL
                SELECT 'answer_count'
                WHERE (
                    SELECT count(*)
                    FROM rag_answer_attempt_payloads a
                    WHERE a.profile_id = ? AND a.trace_id = ?
                ) != (SELECT count(*) FROM expected_answers)
                UNION ALL
                SELECT 'answer'
                WHERE EXISTS (
                    SELECT 1
                    FROM expected_answers e
                    LEFT JOIN rag_answer_attempt_payloads a
                      ON a.profile_id = ?
                     AND a.trace_id = ?
                     AND a.payload_id = e.payload_id
                     AND a.attempt_id = e.attempt_id
                    WHERE a.payload_id IS NULL
                       OR a.redaction_state != 'available'
                       OR a.answer_body IS NULL
                       OR a.body_integrity_hmac IS NULL
                       OR a.purged_at IS NOT NULL
                )
            ),
            guarded AS (
                SELECT 1 AS permitted
                WHERE NOT EXISTS (SELECT 1 FROM invalid)
            )
            SELECT
                0 AS family_order,
                'run' AS family,
                e.run_ordinal AS sort_ordinal,
                e.run_id AS sort_key,
                r.run_payload_json,
                NULL AS payload_id,
                NULL AS attempt_id,
                NULL AS storage_mode,
                NULL AS snapshot_text,
                NULL AS title,
                NULL AS source_identity_json,
                NULL AS locator_json,
                NULL AS lineage_json,
                NULL AS transformations_json,
                NULL AS content_hash,
                NULL AS comparison_fingerprint,
                NULL AS origin_payload_id,
                NULL AS answer_body,
                NULL AS body_integrity_hmac
            FROM guarded
            JOIN expected_runs e
            JOIN rag_evidence_runs r
              ON r.profile_id = ?
             AND r.trace_id = ?
             AND r.run_id = e.run_id
            UNION ALL
            SELECT
                1, 'snapshot', 0, e.payload_id, NULL,
                s.payload_id, NULL, s.storage_mode, s.snapshot_text, s.title,
                s.source_identity_json, s.locator_json, s.lineage_json,
                s.transformations_json, s.content_hash,
                s.comparison_fingerprint, s.origin_payload_id, NULL, NULL
            FROM guarded
            JOIN expected_snapshots e
            JOIN rag_evidence_snapshots s
              ON s.profile_id = ?
             AND s.payload_id = e.payload_id
            UNION ALL
            SELECT
                2, 'answer', e.attempt_ordinal, e.attempt_id, NULL,
                a.payload_id, a.attempt_id, NULL, NULL, NULL, NULL, NULL,
                NULL, NULL, NULL, NULL, NULL, a.answer_body,
                a.body_integrity_hmac
            FROM guarded
            JOIN expected_answers e
            JOIN rag_answer_attempt_payloads a
              ON a.profile_id = ?
             AND a.trace_id = ?
             AND a.payload_id = e.payload_id
             AND a.attempt_id = e.attempt_id
            ORDER BY family_order, sort_ordinal, sort_key
            """,
            (
                _canonical_json(expected_runs),
                _canonical_json(expected_snapshots),
                _canonical_json(expected_answers),
                _canonical_json(expected_references),
                _canonical_json(authorization.allowlisted_authority_ids),
                profile_id,
                trace_id,
                summary.trace.origin.value,
                namespace.origin_scope_id,
                _canonical_json(summary.trace.model_dump(mode="json")),
                summary.visibility_state,
                profile_id,
                trace_id,
                profile_id,
                trace_id,
                default_run_authority_id,
                required_run_authority_id,
                default_run_authority_id,
                required_run_authority_id,
                profile_id,
                trace_id,
                profile_id,
                trace_id,
                profile_id,
                authorization.governance_scope_id,
                profile_id,
                trace_id,
                profile_id,
                trace_id,
                profile_id,
                trace_id,
                profile_id,
                profile_id,
                trace_id,
            ),
        ).fetchall()
        run_rows = tuple(row for row in governed_rows if row["family"] == "run")
        snapshot_rows = tuple(
            row for row in governed_rows if row["family"] == "snapshot"
        )
        answer_rows = tuple(row for row in governed_rows if row["family"] == "answer")
        if (
            len(run_rows) != len(expected_runs)
            or len(snapshot_rows)
            != len({item["payload_id"] for item in expected_snapshots})
            or len(answer_rows) != len(expected_answers)
        ):
            return CitationHydrationResult(
                state=CitationHydrationState.PAYLOAD_UNAVAILABLE,
                summary=summary,
            )
        try:
            governed = GovernedCitationPayloads(
                evidence_run_payloads=tuple(
                    EvidenceRunPayload.model_validate_json(row["run_payload_json"])
                    for row in run_rows
                ),
                evidence_snapshot_payloads=tuple(
                    EvidenceSnapshotPayload(
                        payload_id=row["payload_id"],
                        storage_mode=EvidenceStorageMode(row["storage_mode"]),
                        snapshot_text=row["snapshot_text"],
                        server_reference=(
                            row["origin_payload_id"]
                            if row["storage_mode"]
                            == EvidenceStorageMode.SERVER_REFERENCE.value
                            else None
                        ),
                        title=row["title"],
                        source_identity=json.loads(row["source_identity_json"] or "{}"),
                        locator=json.loads(row["locator_json"] or "{}"),
                        lineage=json.loads(row["lineage_json"] or "{}"),
                        transformations=tuple(
                            json.loads(row["transformations_json"] or "[]")
                        ),
                        content_hash=row["content_hash"],
                        comparison_hash=row["comparison_fingerprint"],
                    )
                    for row in snapshot_rows
                ),
                answer_attempt_payloads=tuple(
                    AnswerAttemptPayload(
                        payload_id=row["payload_id"],
                        attempt_id=row["attempt_id"],
                        answer_body=row["answer_body"],
                        body_integrity_hmac=row["body_integrity_hmac"],
                    )
                    for row in answer_rows
                ),
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return CitationHydrationResult(
                state=CitationHydrationState.PAYLOAD_UNAVAILABLE,
                summary=summary,
            )
        return CitationHydrationResult(
            state=CitationHydrationState.AUTHORIZED,
            summary=summary,
            governed_payloads=governed,
        )

    def _before_governed_select(self) -> None:
        """Test seam immediately before the guarded governed read."""

    def _fail_after(self, row_family: str) -> None:
        if self._failure_after_row_family == row_family:
            raise RuntimeError(f"forced_failure_after_{row_family}")

    @staticmethod
    def _profile_permitted(
        namespace: TraceNamespace,
        authorization: CitationReadAuthorization,
    ) -> bool:
        if authorization.authority_scope is AuthorityScope.LOCAL_PROFILE:
            return authorization.profile_id == namespace.profile_id
        return namespace.authenticated_tenant_id is not None

    @staticmethod
    def _governance_scope_permitted(
        namespace: TraceNamespace,
        authorization: CitationReadAuthorization,
    ) -> bool:
        expected = (
            namespace.profile_id
            if namespace.identity_namespace is CitationIdentityNamespace.LOCAL_TRACE
            else namespace.origin_scope_id
        )
        return authorization.governance_scope_id == expected

    @staticmethod
    def _trace_selector(namespace: TraceNamespace) -> tuple[str, tuple[Any, ...]]:
        if namespace.identity_namespace is CitationIdentityNamespace.LOCAL_TRACE:
            return (
                "profile_id = ? AND trace_id = ? AND origin = 'local'",
                (namespace.profile_id, namespace.trace_id),
            )
        if namespace.identity_namespace is CitationIdentityNamespace.SERVER_TRACE:
            return (
                """
                profile_id = ? AND origin = 'server'
                AND connection_authority_id = ? AND origin_scope_id = ?
                AND server_trace_id = ? AND wire_schema_version = ?
                """,
                (
                    namespace.profile_id,
                    namespace.authority_id,
                    namespace.origin_scope_id,
                    namespace.server_trace_id,
                    namespace.wire_schema_version,
                ),
            )
        return (
            """
            profile_id = ? AND origin = 'imported'
            AND import_package_fingerprint = ? AND external_trace_id = ?
            """,
            (
                namespace.profile_id,
                namespace.import_package_fingerprint,
                namespace.external_trace_id,
            ),
        )


__all__ = [
    "ActiveCitationTraceResult",
    "ActiveCitationTraceState",
    "CitationHydrationResult",
    "CitationHydrationState",
    "CitationPersistenceUnavailable",
    "CitationTraceRepository",
    "CitationTraceSummary",
    "GovernedCitationPayloads",
    "PreparedCitationWrite",
    "load_local_citation_identity_context",
]
