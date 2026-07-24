"""SQLite persistence for complete, sealed citation provenance aggregates."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import hashlib
import hmac
import json
import sqlite3
import threading
from typing import TYPE_CHECKING, Any, Callable
import weakref

from pydantic import BaseModel, ConfigDict

from tldw_chatbook.Chat.citation_provenance_runtime import (
    CitationProvenanceRuntimePolicy,
)
from tldw_chatbook.Chat.citation_source_locators import (
    AuthorityScope,
    CanonicalSourceKind,
    CitationReadAuthorization,
    CitationSourceAvailability,
    CitationSourceObservation,
    CitationSourcePermission,
    CitationContentState,
    CitationLocationState,
    SOURCE_INVENTORY_BY_SCOPE_V1,
    SourceCapability,
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
    CitationCompleteness,
    CitationTrace,
    EvidenceRunPayload,
    EvidenceSnapshotPayload,
    EvidenceStorageMode,
    PolicyCapability,
    SealedCitationWrite,
    TraceOrigin,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

if TYPE_CHECKING:
    from tldw_chatbook.Chat.citation_artifact_ownership import (
        ArtifactOwnerBinding,
        ArtifactOwnerOperation,
        ArtifactOwnerOperationKind,
    )
    from tldw_chatbook.Chat.citation_payload_lifecycle import SnapshotDedupeScope


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


class CitationAvailabilityWarning(str, Enum):
    """Safe active-presentation warning without governed payload content."""

    EVIDENCE_REVOKED = "evidence_revoked"


class CitationObservationWriteOutcome(str, Enum):
    """Bounded compare-and-replace result for one observation key."""

    INSERTED = "inserted"
    REPLACED = "replaced"
    STALE = "stale"
    IDEMPOTENT = "idempotent"


@dataclass(frozen=True, slots=True)
class _SourceObservationReferencePolicy:
    """Current non-governed policy result for one exact trace reference."""

    allowed_capabilities: frozenset[SourceCapability]
    revoked: bool
    unavailable: bool


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
    availability_warning: CitationAvailabilityWarning | None

    def __init__(
        self,
        *,
        state: ActiveCitationTraceState,
        summary: CitationTraceSummary | None = None,
        availability_warning: CitationAvailabilityWarning | None = None,
    ) -> None:
        if not isinstance(state, ActiveCitationTraceState):
            raise TypeError("state must be an ActiveCitationTraceState")
        if (
            state is ActiveCitationTraceState.ACTIVE
            or summary is not None
            or availability_warning is not None
        ):
            raise ValueError("active summary results are repository-issued only")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "summary", None)
        object.__setattr__(self, "availability_warning", None)

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


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False, init=False)
class CitationArtifactOwnerRequest:
    """Opaque repository-issued request to retain one active local trace."""

    namespace: TraceNamespace
    message_id: str
    message_revision: int

    def __init__(
        self,
        *,
        namespace: TraceNamespace,
        message_id: str,
        message_revision: int,
    ) -> None:
        del namespace, message_id, message_revision
        raise ValueError("artifact owner requests are repository-issued only")

    def __copy__(self) -> "CitationArtifactOwnerRequest":
        raise TypeError("artifact owner requests cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> "CitationArtifactOwnerRequest":
        del memo
        raise TypeError("artifact owner requests cannot be copied")


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


def _bounded_observation_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CitationPersistenceUnavailable(f"source_observation_{field_name}_invalid")
    if len(value.encode("utf-8")) > 256:
        raise CitationPersistenceUnavailable(f"source_observation_{field_name}_invalid")
    return value


def _observation_capabilities_json(
    observation: CitationSourceObservation,
) -> str:
    return _canonical_json(
        {
            "capabilities": [
                capability.value for capability in observation.capabilities
            ],
            "request_generation": observation.request_generation,
        }
    )


def _observation_from_row(row: sqlite3.Row) -> CitationSourceObservation:
    try:
        payload = json.loads(row["capabilities_json"])
        if not isinstance(payload, dict) or set(payload) != {
            "capabilities",
            "request_generation",
        }:
            raise ValueError("invalid observation capability payload")
        raw_capabilities = payload["capabilities"]
        if not isinstance(raw_capabilities, list):
            raise ValueError("invalid observation capabilities")
        return CitationSourceObservation(
            resolver_kind=CanonicalSourceKind(row["resolver_kind"]),
            resolver_version=row["resolver_version"],
            availability=CitationSourceAvailability(row["availability"]),
            permission=CitationSourcePermission(row["permission_state"]),
            content_state=CitationContentState(row["content_state"]),
            location_state=CitationLocationState(row["location_state"]),
            capabilities=tuple(
                SourceCapability(capability) for capability in raw_capabilities
            ),
            observed_at=datetime.fromisoformat(row["observed_at"]),
            request_generation=payload["request_generation"],
            request_nonce=row["request_nonce"],
            error_code=row["error_code"],
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise CitationPersistenceUnavailable("source_observation_invalid") from None


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
            (
                None
                if result.availability_warning is None
                else result.availability_warning.value
            ),
        )
    )
    return hashlib.sha256(canonical.encode("utf-8")).digest()


def _artifact_owner_request_digest(
    request: CitationArtifactOwnerRequest,
) -> bytes:
    """Digest the complete repository-issued artifact capability."""

    canonical = _canonical_json(
        (
            request.namespace.model_dump(mode="json"),
            request.message_id,
            request.message_revision,
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
        self._issued_artifact_owner_requests: dict[
            int,
            tuple[
                weakref.ReferenceType[CitationArtifactOwnerRequest],
                bytes,
                _ActiveTraceProof,
            ],
        ] = {}
        self._artifact_collection_barrier_provider: Callable[[], Any] | None = None
        self._artifact_collection_guard_factory: (
            Callable[[], AbstractContextManager[Any]] | None
        ) = None
        self._artifact_collection_store_identity: object | None = None
        self._artifact_collection_registration_lock = threading.Lock()

    def register_artifact_collection_barrier(
        self,
        *,
        store_identity: object,
        provider: Callable[[], Any],
        guard: Callable[[], AbstractContextManager[Any]],
    ) -> None:
        """Install the authoritative cross-store barrier and mutation guard."""

        if store_identity is None or not callable(provider) or not callable(guard):
            raise TypeError("artifact collection barrier contract is invalid")
        with self._artifact_collection_registration_lock:
            current_identity = self._artifact_collection_store_identity
            if current_identity is not None and current_identity != store_identity:
                raise ValueError("artifact_collection_barrier_already_registered")
            self._artifact_collection_store_identity = store_identity
            self._artifact_collection_barrier_provider = provider
            self._artifact_collection_guard_factory = guard

    def artifact_collection_guard(self) -> AbstractContextManager[Any]:
        """Return the registered cross-store guard or a no-op context."""

        factory = self._artifact_collection_guard_factory
        return nullcontext() if factory is None else factory()

    def artifact_collection_barriers(self) -> Any | None:
        """Read authoritative cross-store barriers while the guard is held."""

        provider = self._artifact_collection_barrier_provider
        return None if provider is None else provider()

    @property
    def artifact_binding_verification_available(self) -> bool:
        """Return whether signed artifact bindings can currently be verified."""

        return self._fingerprint_codec is not None and self.identity_context is not None

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
        for snapshot_row in prepared.snapshot_rows:
            self.assert_payload_origin_writable(
                cursor,
                profile_id=prepared.profile_id,
                origin_namespace="local_payload_v1",
                origin_payload_id=str(snapshot_row[6]),
                seam="write",
            )
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
        revoked = cursor.execute(
            """
            SELECT 1
            FROM rag_trace_evidence_refs AS reference
            JOIN rag_evidence_snapshots AS snapshot
              ON snapshot.profile_id = reference.profile_id
             AND snapshot.payload_id = reference.snapshot_payload_id
            JOIN rag_payload_tombstones AS tombstone
              ON tombstone.profile_id = snapshot.profile_id
             AND tombstone.origin_namespace = snapshot.origin_namespace
             AND tombstone.origin_payload_id = snapshot.origin_payload_id
            WHERE reference.profile_id = ? AND reference.trace_id = ?
            LIMIT 1
            """,
            (identity.profile_id, namespace.trace_id),
        ).fetchone()
        if revoked is not None:
            raise CitationPersistenceUnavailable("payload_origin_revoked")
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
        presentation_allowed, availability_warning = self._active_presentation_warning(
            summary,
            profile_id=identity.profile_id,
            trace_id=row["trace_id"],
        )
        if not presentation_allowed:
            return ActiveCitationTraceResult(state=ActiveCitationTraceState.NOT_FOUND)
        return self._issue_active_trace_result(
            summary,
            availability_warning=availability_warning,
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
        presentation_allowed, availability_warning = self._active_presentation_warning(
            result.summary,
            profile_id=proof.identity_context.profile_id,
            trace_id=proof.trace_id,
        )
        if (
            not presentation_allowed
            or availability_warning is not result.availability_warning
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

    def get_artifact_owner_request(
        self,
        *,
        message_id: str,
        message_revision: int,
        current_body: str,
    ) -> CitationArtifactOwnerRequest | None:
        """Issue an opaque artifact request for an exact active message owner."""

        if not self.policy.canonical_writes_enabled:
            return None
        result = self.get_active_trace_for_message(
            message_id,
            message_revision,
            current_body,
            self._fingerprint_codec,
        )
        if (
            result.state is not ActiveCitationTraceState.ACTIVE
            or result.summary is None
            or not self.verify_active_trace_result(result)
        ):
            return None
        active_issued = self._issued_active_results.get(id(result))
        if active_issued is None:
            return None
        proof = active_issued[2]
        request = object.__new__(CitationArtifactOwnerRequest)
        object.__setattr__(request, "namespace", result.summary.namespace)
        object.__setattr__(request, "message_id", proof.message_id)
        object.__setattr__(request, "message_revision", proof.message_revision)
        request_id = id(request)
        repository_ref = weakref.ref(self)

        def discard(
            request_ref: weakref.ReferenceType[CitationArtifactOwnerRequest],
        ) -> None:
            repository = repository_ref()
            if repository is None:
                return
            issued = repository._issued_artifact_owner_requests.get(request_id)
            if issued is not None and issued[0] is request_ref:
                repository._issued_artifact_owner_requests.pop(request_id, None)

        request_ref = weakref.ref(request, discard)
        self._issued_artifact_owner_requests[request_id] = (
            request_ref,
            _artifact_owner_request_digest(request),
            proof,
        )
        return request

    def prepare_artifact_owner_operation(
        self,
        request: CitationArtifactOwnerRequest,
        *,
        artifact_store_id: str,
        artifact_id: str,
        artifact_revision: int,
        artifact_body: str,
        operation_kind: ArtifactOwnerOperationKind,
    ) -> ArtifactOwnerOperation:
        """Derive a stable link operation from an exact issued owner request."""

        from tldw_chatbook.Chat.citation_artifact_ownership import (
            ArtifactOwnerBinding,
            ArtifactOwnerOperation,
            ArtifactOwnerOperationKind,
        )

        if operation_kind is not ArtifactOwnerOperationKind.LINK:
            raise CitationPersistenceUnavailable("artifact_operation_kind_invalid")
        proof = self._verify_artifact_owner_request(request)
        if proof is None:
            raise CitationPersistenceUnavailable("artifact_owner_request_invalid")
        codec = self._fingerprint_codec
        if codec is None:
            raise CitationPersistenceUnavailable("fingerprint_key_unavailable")
        binding = self._artifact_owner_binding(
            ArtifactOwnerBinding,
            codec=codec,
            profile_id=proof.identity_context.profile_id,
            artifact_store_id=artifact_store_id,
            artifact_id=artifact_id,
            artifact_revision=artifact_revision,
            trace_id=proof.trace_id,
            artifact_body_fingerprint=codec.fingerprint(
                CitationFingerprintDomain.OWNER_OPERATION,
                "artifact-body",
                artifact_body,
            ),
        )
        return ArtifactOwnerOperation(
            operation_id=self._artifact_owner_operation_id(
                codec,
                binding_id=binding.binding_id,
                operation_kind=operation_kind.value,
            ),
            operation_kind=operation_kind,
            binding=binding,
            created_at=datetime.now(UTC),
        )

    def prepare_artifact_unlink_operation(
        self,
        binding: ArtifactOwnerBinding,
    ) -> ArtifactOwnerOperation:
        """Derive a stable unlink only for an integrity-checked local binding."""

        from tldw_chatbook.Chat.citation_artifact_ownership import (
            ArtifactOwnerBinding,
            ArtifactOwnerOperation,
            ArtifactOwnerOperationKind,
        )

        validated = ArtifactOwnerBinding.model_validate(
            binding.model_dump(mode="python"),
            strict=True,
        )
        codec = self._fingerprint_codec
        identity = self.identity_context
        if not self.policy.canonical_writes_enabled:
            raise CitationPersistenceUnavailable("canonical_citation_writes_disabled")
        if codec is None or identity is None:
            raise CitationPersistenceUnavailable("fingerprint_key_unavailable")
        if (
            validated.profile_id != identity.profile_id
            or not self._artifact_binding_matches(codec, validated)
        ):
            raise CitationPersistenceUnavailable("artifact_owner_binding_invalid")
        return ArtifactOwnerOperation(
            operation_id=self._artifact_owner_operation_id(
                codec,
                binding_id=validated.binding_id,
                operation_kind=ArtifactOwnerOperationKind.UNLINK.value,
            ),
            operation_kind=ArtifactOwnerOperationKind.UNLINK,
            binding=validated,
            created_at=datetime.now(UTC),
        )

    def verify_artifact_owner_binding(
        self,
        binding: ArtifactOwnerBinding,
        *,
        artifact_body: str,
    ) -> None:
        """Verify binding authenticity and its exact artifact representation."""

        from tldw_chatbook.Chat.citation_artifact_ownership import (
            ArtifactOwnerBinding,
        )

        validated = ArtifactOwnerBinding.model_validate(
            binding.model_dump(mode="python"),
            strict=True,
        )
        codec = self._fingerprint_codec
        identity = self.identity_context
        if codec is None or identity is None:
            raise CitationPersistenceUnavailable("fingerprint_key_unavailable")
        if (
            validated.profile_id != identity.profile_id
            or not self._artifact_binding_matches(codec, validated)
        ):
            raise CitationPersistenceUnavailable("artifact_owner_binding_invalid")
        expected_body = codec.fingerprint(
            CitationFingerprintDomain.OWNER_OPERATION,
            "artifact-body",
            artifact_body,
        )
        if not hmac.compare_digest(
            expected_body,
            validated.artifact_body_fingerprint,
        ):
            raise CitationPersistenceUnavailable("artifact_body_integrity_invalid")

    def apply_artifact_owner_operation(
        self,
        operation: ArtifactOwnerOperation,
    ) -> None:
        """Idempotently apply one durable artifact-side operation receipt."""

        from tldw_chatbook.Chat.citation_artifact_ownership import (
            ArtifactOwnerOperationKind,
        )

        validated = self._validate_artifact_owner_operation(operation)
        with self.db.transaction() as cursor:
            identity = self._require_active_write_cursor(cursor)
            if identity.profile_id != validated.binding.profile_id:
                raise CitationPersistenceUnavailable("artifact_owner_profile_mismatch")
            self._acquire_artifact_owner_lock(cursor, identity.profile_id)
            binding = validated.binding
            now = datetime.now(UTC).isoformat()
            if validated.operation_kind is ArtifactOwnerOperationKind.LINK:
                trace = cursor.execute(
                    """
                    SELECT 1
                    FROM rag_citation_traces
                    WHERE profile_id = ? AND trace_id = ?
                      AND origin = 'local' AND origin_scope_id = ?
                      AND visibility_state = 'active'
                    """,
                    (binding.profile_id, binding.trace_id, binding.profile_id),
                ).fetchone()
                if trace is None:
                    raise CitationPersistenceUnavailable("artifact_trace_unavailable")
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO rag_artifact_owner_leases(
                        profile_id, artifact_store_id, artifact_id,
                        artifact_revision, trace_id, lease_id, state,
                        created_at, updated_at, retain_until
                    ) VALUES (?, ?, ?, ?, ?, ?, 'link_pending', ?, ?, NULL)
                    """,
                    (
                        binding.profile_id,
                        binding.artifact_store_id,
                        binding.artifact_id,
                        binding.artifact_revision,
                        binding.trace_id,
                        binding.lease_id,
                        validated.created_at.isoformat(),
                        now,
                    ),
                )
            lease = cursor.execute(
                """
                SELECT lease_id, state
                FROM rag_artifact_owner_leases
                WHERE profile_id = ? AND artifact_store_id = ?
                  AND artifact_id = ? AND artifact_revision = ? AND trace_id = ?
                """,
                (
                    binding.profile_id,
                    binding.artifact_store_id,
                    binding.artifact_id,
                    binding.artifact_revision,
                    binding.trace_id,
                ),
            ).fetchone()
            if lease is None or not hmac.compare_digest(
                lease["lease_id"], binding.lease_id
            ):
                raise CitationPersistenceUnavailable("artifact_lease_identity_conflict")
            if (
                validated.operation_kind is ArtifactOwnerOperationKind.UNLINK
                and lease["state"] == "link_pending"
            ):
                raise CitationPersistenceUnavailable("artifact_link_not_live")
            cursor.execute(
                """
                INSERT OR IGNORE INTO rag_artifact_owner_operations(
                    profile_id, operation_id, artifact_store_id, artifact_id,
                    artifact_revision, trace_id, operation_kind, state,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    binding.profile_id,
                    validated.operation_id,
                    binding.artifact_store_id,
                    binding.artifact_id,
                    binding.artifact_revision,
                    binding.trace_id,
                    validated.operation_kind.value,
                    validated.created_at.isoformat(),
                    now,
                ),
            )
            receipt = cursor.execute(
                """
                SELECT
                    artifact_store_id, artifact_id, artifact_revision,
                    trace_id, operation_kind, state
                FROM rag_artifact_owner_operations
                WHERE profile_id = ? AND operation_id = ?
                """,
                (binding.profile_id, validated.operation_id),
            ).fetchone()
            expected = (
                binding.artifact_store_id,
                binding.artifact_id,
                binding.artifact_revision,
                binding.trace_id,
                validated.operation_kind.value,
            )
            if receipt is None or tuple(receipt)[:5] != expected:
                raise CitationPersistenceUnavailable(
                    "artifact_operation_identity_conflict"
                )
            cursor.execute(
                """
                UPDATE rag_artifact_owner_operations
                SET state = 'applied', updated_at = ?
                WHERE profile_id = ? AND operation_id = ? AND state = 'pending'
                """,
                (now, binding.profile_id, validated.operation_id),
            )
            if validated.operation_kind is ArtifactOwnerOperationKind.LINK:
                cursor.execute(
                    """
                    UPDATE rag_artifact_owner_leases
                    SET state = 'live', updated_at = ?
                    WHERE profile_id = ? AND artifact_store_id = ?
                      AND artifact_id = ? AND artifact_revision = ?
                      AND trace_id = ? AND state = 'link_pending'
                    """,
                    (
                        now,
                        binding.profile_id,
                        binding.artifact_store_id,
                        binding.artifact_id,
                        binding.artifact_revision,
                        binding.trace_id,
                    ),
                )
            elif lease["state"] != "released":
                cursor.execute(
                    """
                    UPDATE rag_artifact_owner_leases
                    SET state = 'unlink_pending', updated_at = ?
                    WHERE profile_id = ? AND artifact_store_id = ?
                      AND artifact_id = ? AND artifact_revision = ?
                      AND trace_id = ? AND state = 'live'
                    """,
                    (
                        now,
                        binding.profile_id,
                        binding.artifact_store_id,
                        binding.artifact_id,
                        binding.artifact_revision,
                        binding.trace_id,
                    ),
                )

    def validate_shared_database_artifact_owner_operation(
        self,
        cursor: sqlite3.Cursor,
        operation: ArtifactOwnerOperation,
    ) -> ArtifactOwnerOperation:
        """Validate one signed owner mutation under the caller's SQLite tx."""

        validated = self._validate_artifact_owner_operation(operation)
        identity = self._require_active_write_cursor(cursor)
        binding = validated.binding
        if identity.profile_id != binding.profile_id:
            raise CitationPersistenceUnavailable("artifact_owner_profile_mismatch")
        trace = cursor.execute(
            """
            SELECT 1
            FROM rag_citation_traces
            WHERE profile_id = ? AND trace_id = ?
              AND origin = 'local' AND origin_scope_id = ?
              AND visibility_state = 'active'
            """,
            (binding.profile_id, binding.trace_id, binding.profile_id),
        ).fetchone()
        if trace is None:
            raise CitationPersistenceUnavailable("artifact_trace_unavailable")
        return validated

    def _validate_artifact_owner_operation(
        self,
        operation: ArtifactOwnerOperation,
    ) -> ArtifactOwnerOperation:
        from tldw_chatbook.Chat.citation_artifact_ownership import (
            ArtifactOwnerOperation,
        )

        validated = ArtifactOwnerOperation.model_validate(
            operation.model_dump(mode="python"),
            strict=True,
        )
        codec = self._fingerprint_codec
        if codec is None or not self._artifact_binding_matches(
            codec, validated.binding
        ):
            raise CitationPersistenceUnavailable("artifact_owner_binding_invalid")
        expected_operation_id = self._artifact_owner_operation_id(
            codec,
            binding_id=validated.binding.binding_id,
            operation_kind=validated.operation_kind.value,
        )
        if not hmac.compare_digest(validated.operation_id, expected_operation_id):
            raise CitationPersistenceUnavailable("artifact_operation_identity_invalid")
        return validated

    def acknowledge_artifact_owner_operation(
        self,
        operation: ArtifactOwnerOperation,
    ) -> None:
        """Finalize an artifact-acknowledged receipt and release unlinks."""

        from tldw_chatbook.Chat.citation_artifact_ownership import (
            ArtifactOwnerOperation,
            ArtifactOwnerOperationKind,
            ArtifactOwnerOutboxState,
        )

        validated = ArtifactOwnerOperation.model_validate(
            operation.model_dump(mode="python"),
            strict=True,
        )
        if validated.state is not ArtifactOwnerOutboxState.ACKNOWLEDGED:
            raise CitationPersistenceUnavailable(
                "artifact_registry_acknowledgement_required"
            )
        self.apply_artifact_owner_operation(validated)
        binding = validated.binding
        with self.db.transaction() as cursor:
            identity = self._require_active_write_cursor(cursor)
            self._acquire_artifact_owner_lock(cursor, identity.profile_id)
            receipt = cursor.execute(
                """
                SELECT operation_kind, state
                FROM rag_artifact_owner_operations
                WHERE profile_id = ? AND operation_id = ?
                """,
                (binding.profile_id, validated.operation_id),
            ).fetchone()
            if (
                receipt is None
                or receipt["operation_kind"] != validated.operation_kind.value
            ):
                raise CitationPersistenceUnavailable(
                    "artifact_operation_identity_conflict"
                )
            now = datetime.now(UTC).isoformat()
            if validated.operation_kind is ArtifactOwnerOperationKind.UNLINK:
                cursor.execute(
                    """
                    UPDATE rag_artifact_owner_leases
                    SET state = 'released', updated_at = ?
                    WHERE profile_id = ? AND artifact_store_id = ?
                      AND artifact_id = ? AND artifact_revision = ?
                      AND trace_id = ? AND state = 'unlink_pending'
                    """,
                    (
                        now,
                        binding.profile_id,
                        binding.artifact_store_id,
                        binding.artifact_id,
                        binding.artifact_revision,
                        binding.trace_id,
                    ),
                )
                state = cursor.execute(
                    """
                    SELECT state FROM rag_artifact_owner_leases
                    WHERE profile_id = ? AND artifact_store_id = ?
                      AND artifact_id = ? AND artifact_revision = ?
                      AND trace_id = ?
                    """,
                    (
                        binding.profile_id,
                        binding.artifact_store_id,
                        binding.artifact_id,
                        binding.artifact_revision,
                        binding.trace_id,
                    ),
                ).fetchone()
                if state is None or state["state"] != "released":
                    raise CitationPersistenceUnavailable(
                        "artifact_unlink_release_failed"
                    )
            cursor.execute(
                """
                UPDATE rag_artifact_owner_operations
                SET state = 'acknowledged', updated_at = ?
                WHERE profile_id = ? AND operation_id = ?
                  AND state IN ('pending','applied')
                """,
                (now, binding.profile_id, validated.operation_id),
            )

    def _verify_artifact_owner_request(
        self,
        request: CitationArtifactOwnerRequest,
    ) -> _ActiveTraceProof | None:
        if not isinstance(request, CitationArtifactOwnerRequest):
            return None
        issued = self._issued_artifact_owner_requests.get(id(request))
        if issued is None or issued[0]() is not request:
            return None
        try:
            digest = _artifact_owner_request_digest(request)
        except (AttributeError, TypeError, ValueError):
            return None
        if not hmac.compare_digest(issued[1], digest):
            return None
        proof = issued[2]
        codec = self._fingerprint_codec
        if codec is None or self.identity_context != proof.identity_context:
            return None
        message = (
            self.db.get_connection()
            .execute(
                """
            SELECT message.version, message.content, owner.state,
                   owner.body_fingerprint, trace.visibility_state
            FROM messages AS message
            JOIN rag_message_trace_owners AS owner
              ON owner.message_id = message.id
             AND owner.message_revision = message.version
            JOIN rag_citation_traces AS trace
              ON trace.profile_id = owner.profile_id
             AND trace.trace_id = owner.trace_id
            WHERE message.id = ? AND message.deleted = 0
              AND owner.profile_id = ? AND owner.trace_id = ?
            """,
                (
                    proof.message_id,
                    proof.identity_context.profile_id,
                    proof.trace_id,
                ),
            )
            .fetchone()
        )
        if (
            message is None
            or message["version"] != proof.message_revision
            or message["state"] != "active"
            or message["visibility_state"] != "active"
        ):
            return None
        current = codec.fingerprint(
            CitationFingerprintDomain.MESSAGE_BODY,
            message["content"],
        )
        if not (
            hmac.compare_digest(message["body_fingerprint"], proof.body_fingerprint)
            and hmac.compare_digest(current, proof.body_fingerprint)
        ):
            return None
        return proof

    @staticmethod
    def _artifact_owner_binding(
        binding_type: type[ArtifactOwnerBinding],
        *,
        codec: CitationFingerprintCodec,
        profile_id: str,
        artifact_store_id: str,
        artifact_id: str,
        artifact_revision: int,
        trace_id: str,
        artifact_body_fingerprint: str,
    ) -> ArtifactOwnerBinding:
        parts = (
            profile_id,
            artifact_store_id,
            artifact_id,
            str(artifact_revision),
            trace_id,
            artifact_body_fingerprint,
        )
        return binding_type(
            profile_id=profile_id,
            artifact_store_id=artifact_store_id,
            artifact_id=artifact_id,
            artifact_revision=artifact_revision,
            trace_id=trace_id,
            lease_id=codec.fingerprint(
                CitationFingerprintDomain.OWNER_OPERATION,
                "artifact-lease",
                *parts,
            ),
            binding_id=codec.fingerprint(
                CitationFingerprintDomain.OWNER_OPERATION,
                "artifact-binding",
                *parts,
            ),
            artifact_body_fingerprint=artifact_body_fingerprint,
        )

    @staticmethod
    def _artifact_owner_operation_id(
        codec: CitationFingerprintCodec,
        *,
        binding_id: str,
        operation_kind: str,
    ) -> str:
        return codec.fingerprint(
            CitationFingerprintDomain.OWNER_OPERATION,
            "artifact-operation",
            binding_id,
            operation_kind,
        )

    @classmethod
    def _artifact_binding_matches(
        cls,
        codec: CitationFingerprintCodec,
        binding: ArtifactOwnerBinding,
    ) -> bool:
        try:
            expected = cls._artifact_owner_binding(
                type(binding),
                codec=codec,
                profile_id=binding.profile_id,
                artifact_store_id=binding.artifact_store_id,
                artifact_id=binding.artifact_id,
                artifact_revision=binding.artifact_revision,
                trace_id=binding.trace_id,
                artifact_body_fingerprint=binding.artifact_body_fingerprint,
            )
        except (TypeError, ValueError):
            return False
        return hmac.compare_digest(expected.binding_id, binding.binding_id) and (
            hmac.compare_digest(expected.lease_id, binding.lease_id)
        )

    @staticmethod
    def _acquire_artifact_owner_lock(
        cursor: sqlite3.Cursor,
        profile_id: str,
    ) -> None:
        cursor.execute(
            """
            UPDATE rag_identity_context
            SET profile_id = profile_id
            WHERE context_name = 'default' AND profile_id = ?
            """,
            (profile_id,),
        )
        if cursor.rowcount != 1:
            raise CitationPersistenceUnavailable("artifact_owner_lock_failed")

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

    def find_reusable_snapshot_payload_id(
        self,
        scope: "SnapshotDedupeScope",
    ) -> str | None:
        """Return one available exact-scope match without exposing its content."""

        from tldw_chatbook.Chat.citation_payload_lifecycle import SnapshotDedupeScope

        validated = SnapshotDedupeScope.model_validate(
            scope.model_dump(mode="python"),
            strict=True,
        )
        row = (
            self.db.get_connection()
            .execute(
                """
                SELECT snapshot.payload_id
                FROM rag_evidence_snapshots AS snapshot
                WHERE snapshot.governance_scope_id = ?
                  AND snapshot.authority_id = ?
                  AND snapshot.confidentiality_policy_id = ?
                  AND snapshot.revocation_scope_id = ?
                  AND snapshot.content_hash = ?
                  AND snapshot.redaction_state = 'available'
                  AND snapshot.purged_at IS NULL
                  AND NOT EXISTS (
                      SELECT 1
                      FROM rag_payload_tombstones AS tombstone
                      WHERE tombstone.profile_id = snapshot.profile_id
                        AND tombstone.origin_namespace =
                              snapshot.origin_namespace
                        AND tombstone.origin_payload_id =
                              snapshot.origin_payload_id
                  )
                ORDER BY snapshot.profile_id, snapshot.payload_id
                LIMIT 1
                """,
                (
                    validated.governance_scope_id,
                    validated.authority_id,
                    validated.confidentiality_policy_id,
                    validated.revocation_scope_id,
                    validated.exact_content_identity,
                ),
            )
            .fetchone()
        )
        return None if row is None else str(row["payload_id"])

    def assert_payload_origin_writable(
        self,
        cursor: sqlite3.Cursor,
        *,
        profile_id: str,
        origin_namespace: str,
        origin_payload_id: str,
        seam: str,
    ) -> None:
        """Fail closed before any write/import/Sync replay can restore content."""

        identity = self._require_active_write_cursor(cursor)
        if (
            profile_id != identity.profile_id
            or not origin_namespace
            or not origin_payload_id
            or not seam
            or any(
                len(value.encode("utf-8")) > 256
                for value in (
                    profile_id,
                    origin_namespace,
                    origin_payload_id,
                    seam,
                )
            )
        ):
            raise CitationPersistenceUnavailable("payload_origin_identity_invalid")
        tombstone = cursor.execute(
            """
            SELECT 1
            FROM rag_payload_tombstones
            WHERE profile_id = ?
              AND origin_namespace = ?
              AND origin_payload_id = ?
            """,
            (profile_id, origin_namespace, origin_payload_id),
        ).fetchone()
        if tombstone is not None:
            raise CitationPersistenceUnavailable("payload_origin_revoked")

    def upsert_source_observation(
        self,
        cursor: sqlite3.Cursor,
        namespace: TraceNamespace,
        *,
        prompt_set_id: str,
        evidence_ordinal: int,
        snapshot_payload_id: str,
        observation: CitationSourceObservation,
        authorization: CitationReadAuthorization,
    ) -> CitationObservationWriteOutcome:
        """Keep only the deterministic latest safe observation for one ref."""

        identity = self._acquire_source_observation_write_lock(cursor)
        (
            validated_namespace,
            validated_observation,
            validated_authorization,
        ) = self._validate_source_observation_request(
            namespace,
            observation=observation,
            authorization=authorization,
            identity=identity,
            require_refresh=True,
        )
        prompt_id, ordinal, payload_id = self._validate_observation_ref_key(
            prompt_set_id,
            evidence_ordinal,
            snapshot_payload_id,
        )
        reference_policy = self._validate_source_observation_reference(
            cursor,
            validated_namespace,
            prompt_set_id=prompt_id,
            evidence_ordinal=ordinal,
            snapshot_payload_id=payload_id,
            resolver_kind=validated_observation.resolver_kind,
            resolver_version=validated_observation.resolver_version,
            authorization=validated_authorization,
            capabilities=validated_observation.capabilities,
        )
        if reference_policy.revoked and not self._is_safe_revoked_observation(
            validated_observation
        ):
            raise CitationPersistenceUnavailable("source_observation_revoked")
        if reference_policy.unavailable and not reference_policy.revoked:
            raise CitationPersistenceUnavailable(
                "source_observation_payload_unavailable"
            )

        key = (
            identity.profile_id,
            validated_namespace.trace_id,
            prompt_id,
            ordinal,
            payload_id,
            validated_observation.resolver_kind.value,
            validated_observation.resolver_version,
        )
        values = (
            *key,
            validated_observation.availability.value,
            validated_observation.permission.value,
            validated_observation.content_state.value,
            validated_observation.location_state.value,
            _observation_capabilities_json(validated_observation),
            validated_observation.request_nonce,
            validated_observation.observed_at.isoformat(),
            validated_observation.error_code,
        )
        cursor.execute(
            """
            INSERT OR IGNORE INTO rag_source_observations(
                profile_id, trace_id, prompt_set_id, evidence_ordinal,
                snapshot_payload_id, resolver_kind, resolver_version,
                availability, permission_state, content_state, location_state,
                capabilities_json, request_nonce, observed_at, error_code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        if cursor.rowcount == 1:
            return CitationObservationWriteOutcome.INSERTED

        current_row = cursor.execute(
            """
            SELECT
                resolver_kind, resolver_version, availability,
                permission_state, content_state, location_state,
                capabilities_json, request_nonce, observed_at, error_code
            FROM rag_source_observations
            WHERE profile_id = ? AND trace_id = ? AND prompt_set_id = ?
              AND evidence_ordinal = ? AND snapshot_payload_id = ?
              AND resolver_kind = ? AND resolver_version = ?
            """,
            key,
        ).fetchone()
        if current_row is None:
            raise CitationPersistenceUnavailable(
                "source_observation_reference_mismatch"
            )
        current = _observation_from_row(current_row)
        incoming_order = (
            validated_observation.observed_at,
            validated_observation.request_generation,
            validated_observation.request_nonce,
        )
        current_order = (
            current.observed_at,
            current.request_generation,
            current.request_nonce,
        )
        if incoming_order < current_order:
            return CitationObservationWriteOutcome.STALE
        if incoming_order == current_order:
            if validated_observation == current:
                return CitationObservationWriteOutcome.IDEMPOTENT
            raise CitationPersistenceUnavailable("source_observation_nonce_conflict")

        cursor.execute(
            """
            UPDATE rag_source_observations
            SET availability = ?, permission_state = ?, content_state = ?,
                location_state = ?, capabilities_json = ?, request_nonce = ?,
                observed_at = ?, error_code = ?
            WHERE profile_id = ? AND trace_id = ? AND prompt_set_id = ?
              AND evidence_ordinal = ? AND snapshot_payload_id = ?
              AND resolver_kind = ? AND resolver_version = ?
            """,
            (
                validated_observation.availability.value,
                validated_observation.permission.value,
                validated_observation.content_state.value,
                validated_observation.location_state.value,
                _observation_capabilities_json(validated_observation),
                validated_observation.request_nonce,
                validated_observation.observed_at.isoformat(),
                validated_observation.error_code,
                *key,
            ),
        )
        if cursor.rowcount != 1:
            raise CitationPersistenceUnavailable("source_observation_write_failed")
        return CitationObservationWriteOutcome.REPLACED

    def record_source_observation_revocations(
        self,
        cursor: sqlite3.Cursor,
        *,
        profile_id: str,
        origin_namespace: str,
        origin_payload_id: str,
        revoked_at: datetime,
    ) -> None:
        """Replace previously bound observations with one explicit revoke event."""

        identity = self._require_active_write_cursor(cursor)
        if profile_id != identity.profile_id:
            raise CitationPersistenceUnavailable(
                "source_observation_revocation_identity_mismatch"
            )
        namespace = _bounded_observation_identifier(
            origin_namespace,
            "origin_namespace",
        )
        payload_id = _bounded_observation_identifier(
            origin_payload_id,
            "origin_payload_id",
        )
        try:
            if revoked_at.utcoffset() is None:
                raise ValueError("naive revocation timestamp")
            timestamp = revoked_at.astimezone(UTC)
        except (AttributeError, OverflowError, TypeError, ValueError):
            raise CitationPersistenceUnavailable(
                "source_observation_revocation_time_invalid"
            ) from None
        revocation_identity = _canonical_json(
            (profile_id, namespace, payload_id, timestamp.isoformat())
        )
        request_nonce = (
            "revocation-"
            + hashlib.sha256(revocation_identity.encode("utf-8")).hexdigest()
        )
        cursor.execute(
            """
            UPDATE rag_source_observations
            SET availability = 'unknown',
                permission_state = 'revoked',
                content_state = 'unknown',
                location_state = 'unknown',
                capabilities_json = ?,
                request_nonce = ?,
                observed_at = ?,
                error_code = 'source_revoked'
            WHERE profile_id = ?
              AND snapshot_payload_id IN (
                  SELECT payload_id
                  FROM rag_evidence_snapshots
                  WHERE profile_id = ?
                    AND origin_namespace = ?
                    AND origin_payload_id = ?
              )
            """,
            (
                _canonical_json(
                    {
                        "capabilities": [],
                        "request_generation": 0,
                    }
                ),
                request_nonce,
                timestamp.isoformat(),
                profile_id,
                profile_id,
                namespace,
                payload_id,
            ),
        )

    def _acquire_source_observation_write_lock(
        self,
        cursor: sqlite3.Cursor,
    ) -> LocalCitationIdentityContext:
        """Serialize deferred SQLite transactions before any observation reads."""

        if cursor.connection is not self.db.get_connection():
            raise RuntimeError(
                "citation persistence requires the repository database transaction"
            )
        if not cursor.connection.in_transaction:
            raise RuntimeError("citation persistence requires an active transaction")
        if not self.policy.canonical_writes_enabled:
            raise CitationPersistenceUnavailable("canonical_citation_writes_disabled")
        return self._acquire_source_observation_policy_lock(cursor)

    def _acquire_source_observation_policy_lock(
        self,
        cursor: sqlite3.Cursor,
    ) -> LocalCitationIdentityContext:
        """Serialize observation policy checks with revocation transactions."""

        if cursor.connection is not self.db.get_connection():
            raise RuntimeError(
                "citation persistence requires the repository database transaction"
            )
        if not cursor.connection.in_transaction:
            raise RuntimeError("citation persistence requires an active transaction")
        identity = self.identity_context
        if identity is None:
            raise CitationPersistenceUnavailable(
                "citation_identity_context_unavailable"
            )
        cursor.execute(
            """
            UPDATE rag_identity_context
            SET profile_id = profile_id
            WHERE context_name = 'default' AND profile_id = ?
            """,
            (identity.profile_id,),
        )
        if cursor.rowcount != 1:
            raise CitationPersistenceUnavailable(
                "source_observation_identity_lock_failed"
            )
        return self._require_repository_cursor(cursor)

    def get_source_observation(
        self,
        namespace: TraceNamespace,
        *,
        prompt_set_id: str,
        evidence_ordinal: int,
        snapshot_payload_id: str,
        resolver_kind: CanonicalSourceKind,
        resolver_version: str,
        authorization: CitationReadAuthorization,
    ) -> CitationSourceObservation | None:
        """Read a safe keyed status, or ``None`` when currently unavailable."""

        with self.db.transaction() as cursor:
            identity = self._acquire_source_observation_policy_lock(cursor)
            validated_namespace, _, validated_authorization = (
                self._validate_source_observation_request(
                    namespace,
                    observation=None,
                    authorization=authorization,
                    identity=identity,
                    require_refresh=False,
                )
            )
            if not isinstance(resolver_kind, CanonicalSourceKind):
                raise CitationPersistenceUnavailable(
                    "source_observation_resolver_unsupported"
                )
            version = _bounded_observation_identifier(
                resolver_version,
                "resolver_version",
            )
            prompt_id, ordinal, payload_id = self._validate_observation_ref_key(
                prompt_set_id,
                evidence_ordinal,
                snapshot_payload_id,
            )
            reference_policy = self._validate_source_observation_reference(
                cursor,
                validated_namespace,
                prompt_set_id=prompt_id,
                evidence_ordinal=ordinal,
                snapshot_payload_id=payload_id,
                resolver_kind=resolver_kind,
                resolver_version=version,
                authorization=validated_authorization,
                capabilities=(),
            )
            row = cursor.execute(
                """
                SELECT
                    resolver_kind, resolver_version, availability,
                    permission_state, content_state, location_state,
                    capabilities_json, request_nonce, observed_at, error_code
                FROM rag_source_observations
                WHERE profile_id = ? AND trace_id = ? AND prompt_set_id = ?
                  AND evidence_ordinal = ? AND snapshot_payload_id = ?
                  AND resolver_kind = ? AND resolver_version = ?
                """,
                (
                    identity.profile_id,
                    validated_namespace.trace_id,
                    prompt_id,
                    ordinal,
                    payload_id,
                    resolver_kind.value,
                    version,
                ),
            ).fetchone()
            if row is None or (
                reference_policy.unavailable and not reference_policy.revoked
            ):
                return None
            observation = _observation_from_row(row)
            if reference_policy.revoked:
                return (
                    observation
                    if self._is_safe_revoked_observation(observation)
                    else None
                )
            allowed_capabilities = reference_policy.allowed_capabilities
            return observation.model_copy(
                update={
                    "capabilities": tuple(
                        capability
                        for capability in observation.capabilities
                        if capability in allowed_capabilities
                    )
                }
            )

    @staticmethod
    def _validate_observation_ref_key(
        prompt_set_id: str,
        evidence_ordinal: int,
        snapshot_payload_id: str,
    ) -> tuple[str, int, str]:
        prompt_id = _bounded_observation_identifier(
            prompt_set_id,
            "prompt_set_id",
        )
        payload_id = _bounded_observation_identifier(
            snapshot_payload_id,
            "snapshot_payload_id",
        )
        if (
            isinstance(evidence_ordinal, bool)
            or not isinstance(evidence_ordinal, int)
            or evidence_ordinal < 1
        ):
            raise CitationPersistenceUnavailable(
                "source_observation_evidence_ordinal_invalid"
            )
        return prompt_id, evidence_ordinal, payload_id

    @staticmethod
    def _validate_source_observation_request(
        namespace: TraceNamespace,
        *,
        observation: CitationSourceObservation | None,
        authorization: CitationReadAuthorization,
        identity: LocalCitationIdentityContext,
        require_refresh: bool,
    ) -> tuple[
        TraceNamespace,
        CitationSourceObservation | None,
        CitationReadAuthorization,
    ]:
        try:
            validated_namespace = TraceNamespace.model_validate(
                namespace.model_dump(mode="python"),
                strict=True,
            )
            validated_authorization = CitationReadAuthorization.model_validate(
                authorization.model_dump(mode="python"),
                strict=True,
            )
            validated_observation = (
                None
                if observation is None
                else CitationSourceObservation.model_validate(
                    observation.model_dump(mode="python"),
                    strict=True,
                )
            )
        except (AttributeError, TypeError, ValueError):
            raise CitationPersistenceUnavailable(
                "source_observation_request_invalid"
            ) from None

        expected_namespace = local_trace_namespace(
            identity,
            trace_id=validated_namespace.trace_id or "",
            wire_schema_version=validated_namespace.wire_schema_version,
        )
        if validated_namespace != expected_namespace:
            raise CitationPersistenceUnavailable(
                "source_observation_namespace_mismatch"
            )
        if (
            validated_authorization.authority_scope is not AuthorityScope.LOCAL_PROFILE
            or validated_authorization.profile_id != identity.profile_id
            or validated_authorization.governance_scope_id != identity.profile_id
            or identity.local_authority_id
            not in validated_authorization.allowlisted_authority_ids
            or (require_refresh and not validated_authorization.refresh_observation)
        ):
            raise CitationPersistenceUnavailable(
                "source_observation_authorization_denied"
            )
        return (
            validated_namespace,
            validated_observation,
            validated_authorization,
        )

    @staticmethod
    def _is_safe_revoked_observation(
        observation: CitationSourceObservation,
    ) -> bool:
        return (
            not observation.capabilities
            and observation.availability is CitationSourceAvailability.UNKNOWN
            and observation.permission is CitationSourcePermission.REVOKED
            and observation.content_state is CitationContentState.UNKNOWN
            and observation.location_state is CitationLocationState.UNKNOWN
        )

    @staticmethod
    def _validate_source_observation_reference(
        connection: sqlite3.Connection | sqlite3.Cursor,
        namespace: TraceNamespace,
        *,
        prompt_set_id: str,
        evidence_ordinal: int,
        snapshot_payload_id: str,
        resolver_kind: CanonicalSourceKind,
        resolver_version: str,
        authorization: CitationReadAuthorization,
        capabilities: tuple[SourceCapability, ...],
    ) -> _SourceObservationReferencePolicy:
        inventory = SOURCE_INVENTORY_BY_SCOPE_V1.get(
            (resolver_kind, AuthorityScope.LOCAL_PROFILE)
        )
        if inventory is None or resolver_version != str(inventory.locator_version):
            raise CitationPersistenceUnavailable(
                "source_observation_resolver_unsupported"
            )
        row = connection.execute(
            """
            SELECT
                trace.aggregate_json, trace.visibility_state,
                snapshot.governance_scope_id, snapshot.authority_id,
                snapshot.redaction_state, snapshot.locator_json,
                snapshot.purged_at,
                EXISTS(
                    SELECT 1
                    FROM rag_payload_tombstones AS tombstone
                    WHERE tombstone.profile_id = snapshot.profile_id
                      AND tombstone.origin_namespace =
                            snapshot.origin_namespace
                      AND tombstone.origin_payload_id =
                            snapshot.origin_payload_id
                ) AS origin_revoked
            FROM rag_trace_evidence_refs AS reference
            JOIN rag_citation_traces AS trace
              ON trace.profile_id = reference.profile_id
             AND trace.trace_id = reference.trace_id
            JOIN rag_evidence_snapshots AS snapshot
              ON snapshot.profile_id = reference.profile_id
             AND snapshot.payload_id = reference.snapshot_payload_id
            WHERE reference.profile_id = ? AND reference.trace_id = ?
              AND reference.prompt_set_id = ?
              AND reference.evidence_ordinal = ?
              AND reference.snapshot_payload_id = ?
              AND trace.origin = 'local'
              AND trace.origin_scope_id = ?
            """,
            (
                namespace.profile_id,
                namespace.trace_id,
                prompt_set_id,
                evidence_ordinal,
                snapshot_payload_id,
                namespace.profile_id,
            ),
        ).fetchone()
        if row is None:
            raise CitationPersistenceUnavailable(
                "source_observation_reference_mismatch"
            )
        if (
            row["visibility_state"] != "active"
            or row["governance_scope_id"] != namespace.profile_id
            or row["authority_id"] != namespace.authority_id
            or authorization.governance_scope_id != row["governance_scope_id"]
            or row["authority_id"] not in authorization.allowlisted_authority_ids
        ):
            raise CitationPersistenceUnavailable(
                "source_observation_authorization_denied"
            )
        try:
            trace = CitationTrace.model_validate_json(row["aggregate_json"])
        except (TypeError, ValueError):
            raise CitationPersistenceUnavailable(
                "source_observation_reference_mismatch"
            ) from None
        prompt_set = next(
            (
                item
                for item in trace.prompt_evidence_sets
                if item.prompt_set_id == prompt_set_id
            ),
            None,
        )
        entry = (
            None
            if prompt_set is None
            else next(
                (
                    item
                    for item in prompt_set.entries
                    if item.evidence_ordinal == evidence_ordinal
                ),
                None,
            )
        )
        if (
            trace.trace_id != namespace.trace_id
            or entry is None
            or entry.snapshot_payload_ref != snapshot_payload_id
        ):
            raise CitationPersistenceUnavailable(
                "source_observation_reference_mismatch"
            )
        if PolicyCapability.RESOLVE_CURRENT_SOURCE not in trace.policy_capabilities:
            raise CitationPersistenceUnavailable(
                "source_observation_trace_policy_denied"
            )

        revoked = bool(row["origin_revoked"]) or (
            row["redaction_state"] == "purged" and row["purged_at"] is not None
        )
        unavailable = (
            revoked
            or row["redaction_state"] != "available"
            or row["locator_json"] is None
        )
        if revoked:
            binding = connection.execute(
                """
                SELECT 1
                FROM rag_source_observations
                WHERE profile_id = ? AND trace_id = ? AND prompt_set_id = ?
                  AND evidence_ordinal = ? AND snapshot_payload_id = ?
                  AND resolver_kind = ? AND resolver_version = ?
                """,
                (
                    namespace.profile_id,
                    namespace.trace_id,
                    prompt_set_id,
                    evidence_ordinal,
                    snapshot_payload_id,
                    resolver_kind.value,
                    resolver_version,
                ),
            ).fetchone()
            if binding is None:
                raise CitationPersistenceUnavailable(
                    "source_observation_resolver_binding_unavailable"
                )
        elif not unavailable:
            try:
                locator = json.loads(row["locator_json"])
                locator_kind = CanonicalSourceKind(locator["source_kind"])
                locator_version = locator.get(
                    "resolver_payload_version",
                    locator.get("locator_version", 1),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                raise CitationPersistenceUnavailable(
                    "source_observation_resolver_mismatch"
                ) from None
            if (
                locator_kind is not resolver_kind
                or str(locator_version) != resolver_version
            ):
                raise CitationPersistenceUnavailable(
                    "source_observation_resolver_mismatch"
                )

        trace_policy = {
            SourceCapability.VIEW_SNAPSHOT: PolicyCapability.VIEW_SNAPSHOT,
            SourceCapability.VIEW_SOURCE_IDENTITY: (
                PolicyCapability.VIEW_SOURCE_IDENTITY
            ),
            SourceCapability.RESOLVE_CURRENT: (PolicyCapability.RESOLVE_CURRENT_SOURCE),
            SourceCapability.OPEN_NATIVE: PolicyCapability.OPEN_NATIVE,
            SourceCapability.OPEN_EXTERNAL: PolicyCapability.OPEN_EXTERNAL,
            SourceCapability.COMPARE: PolicyCapability.COMPARE_CURRENT_SOURCE,
            SourceCapability.REFRESH_OBSERVATION: (
                PolicyCapability.RESOLVE_CURRENT_SOURCE
            ),
            SourceCapability.EXPORT: PolicyCapability.EXPORT_SNAPSHOT,
        }
        allowed_capabilities = frozenset(
            capability
            for capability in SourceCapability
            if not unavailable
            and inventory.default_policy.permits(capability)
            and authorization.permits(capability)
            and trace_policy[capability] in trace.policy_capabilities
        )
        if not revoked and any(
            capability not in allowed_capabilities for capability in capabilities
        ):
            raise CitationPersistenceUnavailable("source_observation_capability_denied")
        return _SourceObservationReferencePolicy(
            allowed_capabilities=allowed_capabilities,
            revoked=revoked,
            unavailable=unavailable,
        )

    def invalidate_trace_capabilities(
        self,
        profile_id: str,
        trace_id: str,
    ) -> None:
        """Invalidate every issued active capability for one trace identity."""

        stale_ids = [
            result_id
            for result_id, issued in self._issued_active_results.items()
            if issued[2].identity_context.profile_id == profile_id
            and issued[2].trace_id == trace_id
        ]
        for result_id in stale_ids:
            self._issued_active_results.pop(result_id, None)

    def _trace_payloads_available(
        self,
        profile_id: str,
        trace_id: str,
    ) -> bool:
        """Recheck current governed access before issuing or verifying trust."""

        row = (
            self.db.get_connection()
            .execute(
                """
                SELECT 1
                FROM rag_citation_traces AS trace
                JOIN rag_answer_attempt_payloads AS selected
                  ON selected.profile_id = trace.profile_id
                 AND selected.trace_id = trace.trace_id
                 AND selected.attempt_id = trace.selected_attempt_id
                WHERE trace.profile_id = ? AND trace.trace_id = ?
                  AND trace.completeness_at_seal = 'complete'
                  AND selected.redaction_state = 'available'
                  AND selected.answer_body IS NOT NULL
                  AND selected.body_integrity_hmac IS NOT NULL
                  AND selected.purged_at IS NULL
                  AND EXISTS (
                      SELECT 1
                      FROM rag_evidence_runs AS run
                      WHERE run.profile_id = trace.profile_id
                        AND run.trace_id = trace.trace_id
                  )
                  AND EXISTS (
                      SELECT 1
                      FROM rag_trace_evidence_refs AS reference
                      WHERE reference.profile_id = trace.profile_id
                        AND reference.trace_id = trace.trace_id
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM rag_evidence_runs AS run
                      WHERE run.profile_id = trace.profile_id
                        AND run.trace_id = trace.trace_id
                        AND (
                            run.redaction_state != 'available'
                            OR run.run_payload_json IS NULL
                            OR run.purged_at IS NOT NULL
                            OR CASE
                                WHEN NOT json_valid(run.run_payload_json)
                                THEN 1
                                WHEN json_type(run.run_payload_json) != 'object'
                                THEN 1
                                WHEN json_extract(
                                    run.run_payload_json,
                                    '$.run_id'
                                ) IS NOT run.run_id
                                THEN 1
                                ELSE 0
                               END
                        )
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM rag_answer_attempt_payloads AS attempt
                      WHERE attempt.profile_id = trace.profile_id
                        AND attempt.trace_id = trace.trace_id
                        AND (
                            attempt.redaction_state != 'available'
                            OR attempt.answer_body IS NULL
                            OR attempt.body_integrity_hmac IS NULL
                            OR attempt.purged_at IS NOT NULL
                        )
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM rag_trace_evidence_refs AS reference
                      JOIN rag_evidence_snapshots AS snapshot
                        ON snapshot.profile_id = reference.profile_id
                       AND snapshot.payload_id = reference.snapshot_payload_id
                      WHERE reference.profile_id = trace.profile_id
                        AND reference.trace_id = trace.trace_id
                        AND (
                            snapshot.redaction_state != 'available'
                            OR snapshot.purged_at IS NOT NULL
                            OR (
                                snapshot.storage_mode = 'embedded'
                                AND snapshot.snapshot_text IS NULL
                            )
                            OR EXISTS (
                                SELECT 1
                                FROM rag_payload_tombstones AS tombstone
                                WHERE tombstone.profile_id = snapshot.profile_id
                                  AND tombstone.origin_namespace =
                                        snapshot.origin_namespace
                                  AND tombstone.origin_payload_id =
                                        snapshot.origin_payload_id
                            )
                        )
                  )
                """,
                (profile_id, trace_id),
            )
            .fetchone()
        )
        return row is not None

    def _active_presentation_warning(
        self,
        summary: CitationTraceSummary,
        *,
        profile_id: str,
        trace_id: str,
    ) -> tuple[bool, CitationAvailabilityWarning | None]:
        """Return whether a sealed summary is safe and its non-content warning."""

        if summary.trace.completeness_at_seal is not CitationCompleteness.COMPLETE:
            return False, None
        if not self._trace_row_identities_match(
            summary,
            profile_id=profile_id,
            trace_id=trace_id,
        ):
            return False, None
        if self._trace_payloads_available(profile_id, trace_id):
            return True, None
        if self._trace_evidence_revoked(profile_id, trace_id):
            return True, CitationAvailabilityWarning.EVIDENCE_REVOKED
        return False, None

    def _trace_row_identities_match(
        self,
        summary: CitationTraceSummary,
        *,
        profile_id: str,
        trace_id: str,
    ) -> bool:
        """Match every persisted governed identity to the sealed aggregate."""

        connection = self.db.get_connection()
        run_rows = connection.execute(
            """
            SELECT run_id, run_ordinal, stage
            FROM rag_evidence_runs
            WHERE profile_id = ? AND trace_id = ?
            """,
            (profile_id, trace_id),
        ).fetchall()
        expected_runs = {
            (run.run_id, run.run_ordinal, run.stage)
            for run in summary.trace.evidence_runs
        }
        if {tuple(row) for row in run_rows} != expected_runs:
            return False
        answer_rows = connection.execute(
            """
            SELECT payload_id, attempt_id
            FROM rag_answer_attempt_payloads
            WHERE profile_id = ? AND trace_id = ?
            """,
            (profile_id, trace_id),
        ).fetchall()
        expected_answers = {
            (attempt.answer_payload_ref, attempt.attempt_id)
            for attempt in summary.trace.answer_attempts
            if attempt.answer_payload_ref is not None
        }
        if {tuple(row) for row in answer_rows} != expected_answers:
            return False
        reference_rows = connection.execute(
            """
            SELECT
                prompt_set_id, evidence_ordinal, run_id,
                snapshot_payload_id, marker_ordinal, storage_mode
            FROM rag_trace_evidence_refs
            WHERE profile_id = ? AND trace_id = ?
            """,
            (profile_id, trace_id),
        ).fetchall()
        expected_references = {
            (
                prompt_set.prompt_set_id,
                entry.evidence_ordinal,
                entry.run_id,
                entry.snapshot_payload_ref,
                entry.marker_ordinal,
                entry.storage_mode.value,
            )
            for prompt_set in summary.trace.prompt_evidence_sets
            for entry in prompt_set.entries
        }
        return {tuple(row) for row in reference_rows} == expected_references

    def _trace_evidence_revoked(
        self,
        profile_id: str,
        trace_id: str,
    ) -> bool:
        """Recheck durable non-content revocation metadata for a complete seal."""

        row = (
            self.db.get_connection()
            .execute(
                """
                SELECT 1
                FROM rag_citation_traces AS trace
                WHERE trace.profile_id = ? AND trace.trace_id = ?
                  AND trace.completeness_at_seal = 'complete'
                  AND EXISTS (
                      SELECT 1
                      FROM rag_trace_evidence_refs AS reference
                      JOIN rag_evidence_snapshots AS snapshot
                        ON snapshot.profile_id = reference.profile_id
                       AND snapshot.payload_id = reference.snapshot_payload_id
                      JOIN rag_payload_tombstones AS tombstone
                        ON tombstone.profile_id = snapshot.profile_id
                       AND tombstone.origin_namespace =
                           snapshot.origin_namespace
                       AND tombstone.origin_payload_id =
                           snapshot.origin_payload_id
                       AND tombstone.revocation_scope_id =
                           snapshot.revocation_scope_id
                      WHERE reference.profile_id = trace.profile_id
                        AND reference.trace_id = trace.trace_id
                        AND snapshot.redaction_state = 'purged'
                        AND snapshot.purged_at IS NOT NULL
                  )
                  AND EXISTS (
                      SELECT 1
                      FROM rag_evidence_runs AS run
                      WHERE run.profile_id = trace.profile_id
                        AND run.trace_id = trace.trace_id
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM rag_evidence_runs AS run
                      WHERE run.profile_id = trace.profile_id
                        AND run.trace_id = trace.trace_id
                        AND (
                            run.redaction_state != 'purged'
                            OR run.run_payload_json IS NOT NULL
                            OR run.purged_at IS NULL
                        )
                  )
                  AND EXISTS (
                      SELECT 1
                      FROM rag_answer_attempt_payloads AS selected
                      WHERE selected.profile_id = trace.profile_id
                        AND selected.trace_id = trace.trace_id
                        AND selected.attempt_id = trace.selected_attempt_id
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM rag_answer_attempt_payloads AS attempt
                      WHERE attempt.profile_id = trace.profile_id
                        AND attempt.trace_id = trace.trace_id
                        AND (
                            attempt.redaction_state != 'purged'
                            OR attempt.answer_body IS NOT NULL
                            OR attempt.body_integrity_hmac IS NOT NULL
                            OR attempt.purged_at IS NULL
                        )
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM rag_trace_evidence_refs AS reference
                      JOIN rag_evidence_snapshots AS snapshot
                        ON snapshot.profile_id = reference.profile_id
                       AND snapshot.payload_id = reference.snapshot_payload_id
                      WHERE reference.profile_id = trace.profile_id
                        AND reference.trace_id = trace.trace_id
                        AND (
                            snapshot.redaction_state = 'redacted'
                            OR (
                                snapshot.redaction_state = 'purged'
                                AND NOT EXISTS (
                                    SELECT 1
                                    FROM rag_payload_tombstones AS tombstone
                                    WHERE tombstone.profile_id =
                                          snapshot.profile_id
                                      AND tombstone.origin_namespace =
                                          snapshot.origin_namespace
                                      AND tombstone.origin_payload_id =
                                          snapshot.origin_payload_id
                                      AND tombstone.revocation_scope_id =
                                          snapshot.revocation_scope_id
                                )
                            )
                            OR (
                                snapshot.redaction_state = 'available'
                                AND EXISTS (
                                    SELECT 1
                                    FROM rag_payload_tombstones AS tombstone
                                    WHERE tombstone.profile_id =
                                          snapshot.profile_id
                                      AND tombstone.origin_namespace =
                                          snapshot.origin_namespace
                                      AND tombstone.origin_payload_id =
                                          snapshot.origin_payload_id
                                      AND tombstone.revocation_scope_id =
                                          snapshot.revocation_scope_id
                                )
                            )
                        )
                  )
                """,
                (profile_id, trace_id),
            )
            .fetchone()
        )
        return row is not None

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
        availability_warning: CitationAvailabilityWarning | None,
        proof: _ActiveTraceProof,
    ) -> ActiveCitationTraceResult:
        """Issue and register one exact active-result capability."""

        result = object.__new__(ActiveCitationTraceResult)
        object.__setattr__(result, "state", ActiveCitationTraceState.ACTIVE)
        object.__setattr__(result, "summary", summary)
        object.__setattr__(result, "availability_warning", availability_warning)
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
            if row["redaction_state"] != "available":
                return CitationHydrationResult(
                    state=CitationHydrationState.REDACTED,
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
    "CitationArtifactOwnerRequest",
    "CitationAvailabilityWarning",
    "CitationHydrationResult",
    "CitationHydrationState",
    "CitationObservationWriteOutcome",
    "CitationPersistenceUnavailable",
    "CitationTraceRepository",
    "CitationTraceSummary",
    "GovernedCitationPayloads",
    "PreparedCitationWrite",
    "load_local_citation_identity_context",
]
