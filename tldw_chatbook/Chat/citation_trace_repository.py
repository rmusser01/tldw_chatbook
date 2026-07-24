"""SQLite persistence for complete, sealed citation provenance aggregates."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from typing import Any

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
    load_fingerprint_codec,
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
    PAYLOAD_UNAVAILABLE = "payload_unavailable"
    REDACTED = "redacted"
    REVOKED = "revoked"


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


_SqlValue = str | int | None
_PreparedRow = tuple[_SqlValue, ...]


@dataclass(frozen=True)
class PreparedCitationWrite:
    """Immutable canonical SQL rows safe to hand to an active transaction."""

    profile_id: str
    trace_id: str
    sealed_at: str
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
        if self._fingerprint_codec is None:
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
        except Exception as exc:
            raise CitationPersistenceUnavailable(
                "invalid_sealed_citation_write"
            ) from exc
        if validated.trace.origin is not TraceOrigin.LOCAL:
            raise CitationPersistenceUnavailable("unsupported_trace_origin")
        for payload in validated.evidence_run_payloads:
            if (
                payload.authority_id is not None
                and payload.authority_id != identity.local_authority_id
            ):
                raise CitationPersistenceUnavailable("run_authority_mismatch")

        trace = validated.trace
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
        return PreparedCitationWrite(
            profile_id=profile_id,
            trace_id=trace.trace_id,
            sealed_at=sealed_at,
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

        if (
            prepared.repository_token is not self._prepared_write_token
            or prepared.identity_context != self.identity_context
        ):
            raise CitationPersistenceUnavailable("prepared_citation_write_not_owned")
        if cursor.connection is not self.db.get_connection():
            raise RuntimeError(
                "citation persistence requires the repository database transaction"
            )
        if not cursor.connection.in_transaction:
            raise RuntimeError("citation persistence requires an active transaction")
        codec = self._fingerprint_codec
        if codec is None:  # guarded by prepare_write
            raise CitationPersistenceUnavailable("fingerprint_key_unavailable")
        cursor.execute(
            """
            INSERT INTO rag_citation_traces(
                profile_id, trace_id, schema_version, request_id, generation_id,
                origin_scope_id, origin, lifecycle, completeness_at_seal,
                selected_attempt_id, policy_version, aggregate_json,
                visibility_state, created_at, sealed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
            """,
            prepared.trace_row,
        )
        self._fail_after("trace")

        for row in prepared.run_rows:
            cursor.execute(
                """
                INSERT INTO rag_evidence_runs(
                    profile_id, trace_id, run_id, run_ordinal, stage,
                    redaction_state, run_payload_json, started_at, ended_at, purged_at
                ) VALUES (?, ?, ?, ?, ?, 'available', ?, ?, ?, NULL)
                """,
                row,
            )
        self._fail_after("runs")

        for row in prepared.snapshot_rows:
            cursor.execute(
                """
                INSERT INTO rag_evidence_snapshots(
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
            )
        self._fail_after("snapshots")

        for row in prepared.answer_rows:
            cursor.execute(
                """
                INSERT INTO rag_answer_attempt_payloads(
                    profile_id, payload_id, trace_id, attempt_id,
                    redaction_state, retention_class, answer_body,
                    body_integrity_hmac, created_at, retain_until, purged_at
                ) VALUES (?, ?, ?, ?, ?, 'default', ?, ?, ?, NULL, ?)
                """,
                row,
            )
        self._fail_after("attempts")

        for row in prepared.reference_rows:
            cursor.execute(
                """
                INSERT INTO rag_trace_evidence_refs(
                    profile_id, trace_id, prompt_set_id, evidence_ordinal,
                    run_id, snapshot_payload_id, marker_ordinal, storage_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
        self._fail_after("refs")

        body_fingerprint = codec.fingerprint(
            CitationFingerprintDomain.MESSAGE_BODY,
            message_body,
        )
        idempotency_key = codec.fingerprint(
            CitationFingerprintDomain.OWNER_OPERATION,
            prepared.profile_id,
            message_id,
            str(message_revision),
            prepared.trace_id,
        )
        cursor.execute(
            """
            INSERT INTO rag_message_trace_owners(
                profile_id, message_id, message_revision, trace_id, state,
                body_fingerprint, idempotency_key, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?)
            """,
            (
                prepared.profile_id,
                message_id,
                message_revision,
                prepared.trace_id,
                body_fingerprint,
                idempotency_key,
                prepared.sealed_at,
                prepared.sealed_at,
            ),
        )
        self._fail_after("owner")

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

        run_rows = connection.execute(
            """
            SELECT run_payload_json
            FROM rag_evidence_runs
            WHERE profile_id = ? AND trace_id = ?
            ORDER BY run_ordinal
            """,
            (profile_id, trace_id),
        ).fetchall()
        snapshot_rows = connection.execute(
            """
            SELECT DISTINCT
                s.payload_id, s.storage_mode, s.snapshot_text, s.title,
                s.source_identity_json, s.locator_json, s.lineage_json,
                s.transformations_json, s.content_hash,
                s.comparison_fingerprint, s.origin_payload_id
            FROM rag_trace_evidence_refs r
            JOIN rag_evidence_snapshots s
              ON s.profile_id = r.profile_id
             AND s.payload_id = r.snapshot_payload_id
            WHERE r.profile_id = ? AND r.trace_id = ?
            ORDER BY s.payload_id
            """,
            (profile_id, trace_id),
        ).fetchall()
        answer_rows = connection.execute(
            """
            SELECT payload_id, attempt_id, answer_body, body_integrity_hmac
            FROM rag_answer_attempt_payloads
            WHERE profile_id = ? AND trace_id = ?
            ORDER BY attempt_id
            """,
            (profile_id, trace_id),
        ).fetchall()
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
        return CitationHydrationResult(
            state=CitationHydrationState.AUTHORIZED,
            summary=summary,
            governed_payloads=governed,
        )

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
    "CitationHydrationResult",
    "CitationHydrationState",
    "CitationPersistenceUnavailable",
    "CitationTraceRepository",
    "CitationTraceSummary",
    "GovernedCitationPayloads",
    "PreparedCitationWrite",
    "load_local_citation_identity_context",
]
