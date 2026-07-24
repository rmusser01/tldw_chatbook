"""Governed citation payload revocation, retention, and collection."""

from __future__ import annotations

import base64
import binascii
from datetime import UTC, datetime, timedelta
import json
import sqlite3
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    model_validator,
)

from tldw_chatbook.Chat.citation_trace_identity import (
    BoundedIdentifier,
    CitationIdentityNamespace,
    TraceNamespace,
)
from tldw_chatbook.Chat.citation_trace_models import (
    EVIDENCE_ENTRIES_PER_PROMPT_MAX,
    PROMPT_EVIDENCE_SETS_MAX,
)
from tldw_chatbook.Chat.citation_trace_repository import (
    CitationPersistenceUnavailable,
    CitationTraceRepository,
    load_local_citation_identity_context,
)


_COLLECTION_BARRIERS_MAX = 1_000
_COLLECTION_BATCH_MAX = 1_000
_COLLECTION_SCAN_PAGE_SIZE = 32
_COLLECTION_CURSOR_BYTES_MAX = 4_096
_RETENTION_SECONDS_MAX = 10 * 365 * 24 * 60 * 60


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


UtcDateTime = Annotated[datetime, AfterValidator(_aware_utc)]
CitationCollectionContinuationCursor = Annotated[
    str,
    Field(min_length=1, max_length=_COLLECTION_CURSOR_BYTES_MAX),
]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
        strict=True,
    )


class SnapshotDedupeScope(_FrozenModel):
    """Complete non-null governance key for one secret-scoped exact payload."""

    schema_version: Literal[1] = 1
    governance_scope_id: BoundedIdentifier
    authority_id: BoundedIdentifier
    confidentiality_policy_id: BoundedIdentifier
    revocation_scope_id: BoundedIdentifier
    exact_content_identity: BoundedIdentifier


class PayloadRetentionPolicy(_FrozenModel):
    """Bounded local retention policy used by lifecycle transactions."""

    schema_version: Literal[1] = 1
    policy_version: BoundedIdentifier
    soft_deleted_owner_retention_seconds: int = Field(
        ge=0,
        le=_RETENTION_SECONDS_MAX,
    )
    max_collection_batch_size: int = Field(ge=1, le=_COLLECTION_BATCH_MAX)


class PayloadTombstone(_FrozenModel):
    """Permitted durable non-content record for one revoked origin payload."""

    schema_version: Literal[1] = 1
    profile_id: BoundedIdentifier
    origin_namespace: BoundedIdentifier
    origin_payload_id: BoundedIdentifier
    revocation_scope_id: BoundedIdentifier
    reason_code: BoundedIdentifier
    policy_version: BoundedIdentifier
    revoked_at: UtcDateTime
    retain_until: UtcDateTime

    @model_validator(mode="after")
    def _validate_retention(self) -> "PayloadTombstone":
        if self.retain_until < self.revoked_at:
            raise ValueError("retain_until must not precede revoked_at")
        return self


class CitationCollectionBarriers(_FrozenModel):
    """Caller-supplied bounded Sync retention and tombstone barriers."""

    schema_version: Literal[1] = 1
    trace_ids: tuple[BoundedIdentifier, ...] = Field(
        default=(),
        max_length=_COLLECTION_BARRIERS_MAX,
    )
    payload_origins: tuple[
        tuple[BoundedIdentifier, BoundedIdentifier],
        ...,
    ] = Field(default=(), max_length=_COLLECTION_BARRIERS_MAX)

    @model_validator(mode="after")
    def _validate_unique(self) -> "CitationCollectionBarriers":
        if len(set(self.trace_ids)) != len(self.trace_ids):
            raise ValueError("trace barriers must be unique")
        if len(set(self.payload_origins)) != len(self.payload_origins):
            raise ValueError("payload-origin barriers must be unique")
        return self


class CitationCollectionResult(_FrozenModel):
    """Bounded counts from one collection transaction."""

    schema_version: Literal[1] = 1
    traces_examined: int = Field(ge=0, le=_COLLECTION_BATCH_MAX)
    traces_collected: int = Field(ge=0, le=_COLLECTION_BATCH_MAX)
    snapshots_collected: int = Field(
        ge=0,
        le=(
            _COLLECTION_BATCH_MAX
            * PROMPT_EVIDENCE_SETS_MAX
            * EVIDENCE_ENTRIES_PER_PROMPT_MAX
        ),
    )
    tombstones_collected: int = Field(ge=0, le=_COLLECTION_BATCH_MAX)
    continuation_cursor: CitationCollectionContinuationCursor | None = None


class _CollectionCursorState(_FrozenModel):
    schema_version: Literal[1] = 1
    profile_id: BoundedIdentifier
    trace_after: tuple[BoundedIdentifier, BoundedIdentifier] | None = None
    trace_until: tuple[BoundedIdentifier, BoundedIdentifier] | None = None
    tombstone_after: (
        tuple[BoundedIdentifier, BoundedIdentifier, BoundedIdentifier] | None
    ) = None
    tombstone_until: (
        tuple[BoundedIdentifier, BoundedIdentifier, BoundedIdentifier] | None
    ) = None


class CitationPayloadLifecycle:
    """Apply lifecycle policy through one citation repository and database."""

    def __init__(
        self,
        repository: CitationTraceRepository,
        *,
        retention_policy: PayloadRetentionPolicy,
    ) -> None:
        if not isinstance(repository, CitationTraceRepository):
            raise TypeError("repository must be a CitationTraceRepository")
        self.repository = repository
        self.retention_policy = PayloadRetentionPolicy.model_validate(
            retention_policy.model_dump(mode="python"),
            strict=True,
        )

    def revoke(
        self,
        namespace: TraceNamespace,
        *,
        snapshot_payload_id: str,
        tombstone: PayloadTombstone,
        cursor: sqlite3.Cursor | None = None,
    ) -> PayloadTombstone:
        """Atomically tombstone an exact origin and purge its governed graph."""

        validated_tombstone = PayloadTombstone.model_validate(
            tombstone.model_dump(mode="python"),
            strict=True,
        )
        self._validate_revoke_preconditions(
            namespace,
            snapshot_payload_id=snapshot_payload_id,
            tombstone=validated_tombstone,
        )
        if cursor is not None:
            stored_tombstone, affected_trace_ids = self._revoke_in_transaction(
                cursor,
                namespace=namespace,
                snapshot_payload_id=snapshot_payload_id,
                tombstone=validated_tombstone,
            )
        else:
            with self.repository.db.transaction() as transaction_cursor:
                stored_tombstone, affected_trace_ids = self._revoke_in_transaction(
                    transaction_cursor,
                    namespace=namespace,
                    snapshot_payload_id=snapshot_payload_id,
                    tombstone=validated_tombstone,
                )
        for trace_id in affected_trace_ids:
            self.repository.invalidate_trace_capabilities(
                validated_tombstone.profile_id,
                trace_id,
            )
        return stored_tombstone

    def _validate_revoke_preconditions(
        self,
        namespace: TraceNamespace,
        *,
        snapshot_payload_id: str,
        tombstone: PayloadTombstone,
    ) -> None:
        if not self.repository.policy.canonical_writes_enabled:
            raise CitationPersistenceUnavailable("canonical_citation_writes_disabled")
        identity = self.repository.identity_context
        persisted = load_local_citation_identity_context(self.repository.db)
        if identity is None or persisted is None:
            raise CitationPersistenceUnavailable(
                "citation_identity_context_unavailable"
            )
        if identity != persisted:
            raise CitationPersistenceUnavailable("citation_identity_context_mismatch")
        if (
            namespace.identity_namespace is not CitationIdentityNamespace.LOCAL_TRACE
            or namespace.profile_id != identity.profile_id
            or namespace.origin_scope_id != identity.profile_id
            or namespace.authority_id != identity.local_authority_id
            or not namespace.trace_id
        ):
            raise CitationPersistenceUnavailable("revoke_trace_identity_mismatch")
        if tombstone.profile_id != identity.profile_id:
            raise CitationPersistenceUnavailable("revoke_profile_identity_mismatch")
        if tombstone.policy_version != self.retention_policy.policy_version:
            raise CitationPersistenceUnavailable("revoke_policy_mismatch")
        TypeAdapter(BoundedIdentifier).validate_python(
            snapshot_payload_id,
            strict=True,
        )

    def _revoke_in_transaction(
        self,
        cursor: sqlite3.Cursor,
        *,
        namespace: TraceNamespace,
        snapshot_payload_id: str,
        tombstone: PayloadTombstone,
    ) -> tuple[PayloadTombstone, tuple[str, ...]]:
        identity = self.repository._require_active_write_cursor(cursor)
        if identity.profile_id != tombstone.profile_id:
            raise CitationPersistenceUnavailable("revoke_profile_identity_mismatch")
        snapshot = cursor.execute(
            """
            SELECT
                snapshot.origin_namespace,
                snapshot.origin_payload_id,
                snapshot.governance_scope_id,
                snapshot.authority_id,
                snapshot.confidentiality_policy_id,
                snapshot.revocation_scope_id
            FROM rag_trace_evidence_refs AS reference
            JOIN rag_evidence_snapshots AS snapshot
              ON snapshot.profile_id = reference.profile_id
             AND snapshot.payload_id = reference.snapshot_payload_id
            WHERE reference.profile_id = ?
              AND reference.trace_id = ?
              AND reference.snapshot_payload_id = ?
            LIMIT 1
            """,
            (
                tombstone.profile_id,
                namespace.trace_id,
                snapshot_payload_id,
            ),
        ).fetchone()
        if snapshot is None:
            raise CitationPersistenceUnavailable("revoke_trace_payload_mismatch")
        if (
            snapshot["origin_namespace"] != tombstone.origin_namespace
            or snapshot["origin_payload_id"] != tombstone.origin_payload_id
        ):
            raise CitationPersistenceUnavailable("revoke_origin_identity_mismatch")
        if snapshot["revocation_scope_id"] != tombstone.revocation_scope_id:
            raise CitationPersistenceUnavailable("revoke_scope_identity_mismatch")
        origin_snapshots = cursor.execute(
            """
            SELECT
                payload_id, governance_scope_id, authority_id,
                confidentiality_policy_id, revocation_scope_id
            FROM rag_evidence_snapshots
            WHERE profile_id = ?
              AND origin_namespace = ?
              AND origin_payload_id = ?
            ORDER BY payload_id
            """,
            (
                tombstone.profile_id,
                tombstone.origin_namespace,
                tombstone.origin_payload_id,
            ),
        ).fetchall()
        origin_policy = (
            snapshot["governance_scope_id"],
            snapshot["authority_id"],
            snapshot["confidentiality_policy_id"],
            snapshot["revocation_scope_id"],
        )
        if not origin_snapshots or any(
            (
                row["governance_scope_id"],
                row["authority_id"],
                row["confidentiality_policy_id"],
                row["revocation_scope_id"],
            )
            != origin_policy
            for row in origin_snapshots
        ):
            raise CitationPersistenceUnavailable("revoke_origin_policy_collision")
        affected_trace_ids = tuple(
            row["trace_id"]
            for row in cursor.execute(
                """
                SELECT DISTINCT reference.trace_id
                FROM rag_trace_evidence_refs AS reference
                JOIN rag_evidence_snapshots AS snapshot
                  ON snapshot.profile_id = reference.profile_id
                 AND snapshot.payload_id = reference.snapshot_payload_id
                WHERE snapshot.profile_id = ?
                  AND snapshot.origin_namespace = ?
                  AND snapshot.origin_payload_id = ?
                ORDER BY reference.trace_id
                """,
                (
                    tombstone.profile_id,
                    tombstone.origin_namespace,
                    tombstone.origin_payload_id,
                ),
            ).fetchall()
        )
        if namespace.trace_id not in affected_trace_ids:
            raise CitationPersistenceUnavailable("revoke_trace_payload_mismatch")

        prior = cursor.execute(
            """
            SELECT revocation_scope_id
            FROM rag_payload_tombstones
            WHERE profile_id = ?
              AND origin_namespace = ?
              AND origin_payload_id = ?
            """,
            (
                tombstone.profile_id,
                tombstone.origin_namespace,
                tombstone.origin_payload_id,
            ),
        ).fetchone()
        if (
            prior is not None
            and prior["revocation_scope_id"] != tombstone.revocation_scope_id
        ):
            raise CitationPersistenceUnavailable("revoke_scope_identity_mismatch")
        cursor.execute(
            """
            INSERT INTO rag_payload_tombstones(
                profile_id, origin_namespace, origin_payload_id,
                revocation_scope_id, reason_code, policy_version,
                revoked_at, retain_until
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_id, origin_namespace, origin_payload_id)
            DO UPDATE SET
                reason_code = excluded.reason_code,
                policy_version = excluded.policy_version,
                revoked_at = min(rag_payload_tombstones.revoked_at, excluded.revoked_at),
                retain_until = max(
                    rag_payload_tombstones.retain_until,
                    excluded.retain_until
                )
            """,
            (
                tombstone.profile_id,
                tombstone.origin_namespace,
                tombstone.origin_payload_id,
                tombstone.revocation_scope_id,
                tombstone.reason_code,
                tombstone.policy_version,
                tombstone.revoked_at.isoformat(),
                tombstone.retain_until.isoformat(),
            ),
        )
        stored_row = cursor.execute(
            """
            SELECT
                profile_id, origin_namespace, origin_payload_id,
                revocation_scope_id, reason_code, policy_version,
                revoked_at, retain_until
            FROM rag_payload_tombstones
            WHERE profile_id = ?
              AND origin_namespace = ?
              AND origin_payload_id = ?
            """,
            (
                tombstone.profile_id,
                tombstone.origin_namespace,
                tombstone.origin_payload_id,
            ),
        ).fetchone()
        if stored_row is None:
            raise CitationPersistenceUnavailable("revoke_tombstone_write_failed")
        stored_tombstone = PayloadTombstone(
            profile_id=stored_row["profile_id"],
            origin_namespace=stored_row["origin_namespace"],
            origin_payload_id=stored_row["origin_payload_id"],
            revocation_scope_id=stored_row["revocation_scope_id"],
            reason_code=stored_row["reason_code"],
            policy_version=stored_row["policy_version"],
            revoked_at=_parse_timestamp(stored_row["revoked_at"]),
            retain_until=_parse_timestamp(stored_row["retain_until"]),
        )
        purge_time = stored_tombstone.revoked_at.isoformat()
        cursor.execute(
            """
            UPDATE rag_evidence_runs
            SET redaction_state = 'purged',
                run_payload_json = NULL,
                purged_at = ?
            WHERE profile_id = ?
              AND trace_id IN (
                  SELECT reference.trace_id
                  FROM rag_trace_evidence_refs AS reference
                  JOIN rag_evidence_snapshots AS snapshot
                    ON snapshot.profile_id = reference.profile_id
                   AND snapshot.payload_id = reference.snapshot_payload_id
                  WHERE snapshot.profile_id = ?
                    AND snapshot.origin_namespace = ?
                    AND snapshot.origin_payload_id = ?
              )
            """,
            (
                purge_time,
                tombstone.profile_id,
                tombstone.profile_id,
                tombstone.origin_namespace,
                tombstone.origin_payload_id,
            ),
        )
        cursor.execute(
            """
            UPDATE rag_evidence_snapshots
            SET redaction_state = 'purged',
                snapshot_text = NULL,
                title = NULL,
                source_identity_json = NULL,
                locator_json = NULL,
                lineage_json = NULL,
                transformations_json = NULL,
                content_hash = NULL,
                comparison_fingerprint = NULL,
                purged_at = ?
            WHERE profile_id = ?
              AND origin_namespace = ?
              AND origin_payload_id = ?
              AND governance_scope_id = ?
              AND authority_id = ?
              AND confidentiality_policy_id = ?
              AND revocation_scope_id = ?
            """,
            (
                purge_time,
                tombstone.profile_id,
                tombstone.origin_namespace,
                tombstone.origin_payload_id,
                *origin_policy,
            ),
        )
        if cursor.rowcount != len(origin_snapshots):
            raise CitationPersistenceUnavailable("revoke_payload_identity_changed")
        self._after_snapshot_purge()
        cursor.execute(
            """
            UPDATE rag_answer_attempt_payloads
            SET redaction_state = 'purged',
                answer_body = NULL,
                body_integrity_hmac = NULL,
                purged_at = ?
            WHERE profile_id = ?
              AND trace_id IN (
                  SELECT reference.trace_id
                  FROM rag_trace_evidence_refs AS reference
                  JOIN rag_evidence_snapshots AS snapshot
                    ON snapshot.profile_id = reference.profile_id
                   AND snapshot.payload_id = reference.snapshot_payload_id
                  WHERE snapshot.profile_id = ?
                    AND snapshot.origin_namespace = ?
                    AND snapshot.origin_payload_id = ?
              )
            """,
            (
                purge_time,
                tombstone.profile_id,
                tombstone.profile_id,
                tombstone.origin_namespace,
                tombstone.origin_payload_id,
            ),
        )
        return stored_tombstone, affected_trace_ids

    def collect(
        self,
        *,
        now: datetime,
        barriers: CitationCollectionBarriers | None = None,
        limit: int | None = None,
        continuation_cursor: CitationCollectionContinuationCursor | None = None,
    ) -> CitationCollectionResult:
        """Collect unowned graphs and expired tombstones in one bounded transaction."""

        current_time = TypeAdapter(UtcDateTime).validate_python(now, strict=True)
        validated_cursor = (
            None
            if continuation_cursor is None
            else TypeAdapter(CitationCollectionContinuationCursor).validate_python(
                continuation_cursor,
                strict=True,
            )
        )
        active_barriers = CitationCollectionBarriers.model_validate(
            (barriers or CitationCollectionBarriers()).model_dump(mode="python"),
            strict=True,
        )
        batch_limit = (
            self.retention_policy.max_collection_batch_size if limit is None else limit
        )
        if (
            isinstance(batch_limit, bool)
            or not isinstance(batch_limit, int)
            or not 1 <= batch_limit <= self.retention_policy.max_collection_batch_size
        ):
            raise ValueError(
                "limit must be within the retention policy collection bound"
            )
        if not self.repository.policy.canonical_writes_enabled:
            raise CitationPersistenceUnavailable("canonical_citation_writes_disabled")

        traces_examined = 0
        traces_collected = 0
        snapshots_collected = 0
        tombstones_collected = 0
        trace_after: tuple[str, str] | None = None
        trace_until: tuple[str, str] | None = None
        tombstone_after: tuple[str, str, str] | None = None
        tombstone_until: tuple[str, str, str] | None = None
        with self.repository.db.transaction() as cursor:
            identity = self.repository._require_active_write_cursor(cursor)
            cursor_state = _decode_collection_cursor(
                validated_cursor,
                profile_id=identity.profile_id,
            )
            trace_after = cursor_state.trace_after
            trace_until = cursor_state.trace_until
            tombstone_after = cursor_state.tombstone_after
            tombstone_until = cursor_state.tombstone_until
            cursor.execute(
                """
                UPDATE rag_citation_traces
                SET visibility_state = visibility_state
                WHERE profile_id = ?
                """,
                (identity.profile_id,),
            )
            scan_limit = max(batch_limit, _COLLECTION_SCAN_PAGE_SIZE)
            candidates, trace_after, trace_until = _trace_scan_page(
                cursor,
                profile_id=identity.profile_id,
                after=trace_after,
                until=trace_until,
                limit=scan_limit,
            )
            for row in candidates:
                if traces_collected >= batch_limit:
                    break
                trace_id = row["trace_id"]
                trace_after = (row["sealed_at"], trace_id)
                traces_examined += 1
                if self._trace_has_barrier(
                    cursor,
                    profile_id=identity.profile_id,
                    trace_id=trace_id,
                    now=current_time,
                    barriers=active_barriers,
                ):
                    continue
                self._before_collect_delete(cursor, identity.profile_id, trace_id)
                if self._trace_has_barrier(
                    cursor,
                    profile_id=identity.profile_id,
                    trace_id=trace_id,
                    now=current_time,
                    barriers=active_barriers,
                ):
                    continue
                snapshot_rows = cursor.execute(
                    """
                    SELECT DISTINCT
                        snapshot.payload_id,
                        snapshot.origin_namespace,
                        snapshot.origin_payload_id,
                        snapshot.retain_until
                    FROM rag_trace_evidence_refs AS reference
                    JOIN rag_evidence_snapshots AS snapshot
                      ON snapshot.profile_id = reference.profile_id
                     AND snapshot.payload_id = reference.snapshot_payload_id
                    WHERE reference.profile_id = ? AND reference.trace_id = ?
                    """,
                    (identity.profile_id, trace_id),
                ).fetchall()
                cursor.execute(
                    """
                    DELETE FROM rag_artifact_owner_operations
                    WHERE profile_id = ? AND trace_id = ?
                      AND state = 'acknowledged'
                    """,
                    (identity.profile_id, trace_id),
                )
                cursor.execute(
                    """
                    DELETE FROM rag_artifact_owner_leases
                    WHERE profile_id = ? AND trace_id = ? AND state = 'released'
                    """,
                    (identity.profile_id, trace_id),
                )
                cursor.execute(
                    """
                    DELETE FROM rag_message_trace_owners
                    WHERE profile_id = ? AND trace_id = ?
                    """,
                    (identity.profile_id, trace_id),
                )
                cursor.execute(
                    """
                    DELETE FROM rag_source_observations
                    WHERE profile_id = ? AND trace_id = ?
                    """,
                    (identity.profile_id, trace_id),
                )
                cursor.execute(
                    """
                    DELETE FROM rag_trace_evidence_refs
                    WHERE profile_id = ? AND trace_id = ?
                    """,
                    (identity.profile_id, trace_id),
                )
                self._after_collect_refs()
                cursor.execute(
                    """
                    DELETE FROM rag_evidence_runs
                    WHERE profile_id = ? AND trace_id = ?
                    """,
                    (identity.profile_id, trace_id),
                )
                cursor.execute(
                    """
                    DELETE FROM rag_answer_attempt_payloads
                    WHERE profile_id = ? AND trace_id = ?
                    """,
                    (identity.profile_id, trace_id),
                )
                cursor.execute(
                    """
                    DELETE FROM rag_citation_traces
                    WHERE profile_id = ? AND trace_id = ?
                      AND NOT EXISTS (
                          SELECT 1 FROM rag_message_trace_owners
                          WHERE profile_id = ? AND trace_id = ?
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM rag_artifact_owner_leases
                          WHERE profile_id = ? AND trace_id = ?
                      )
                    """,
                    (
                        identity.profile_id,
                        trace_id,
                        identity.profile_id,
                        trace_id,
                        identity.profile_id,
                        trace_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise CitationPersistenceUnavailable(
                        "citation_collection_barrier_changed"
                    )
                traces_collected += 1
                self.repository.invalidate_trace_capabilities(
                    identity.profile_id,
                    trace_id,
                )
                for snapshot in snapshot_rows:
                    origin = (
                        snapshot["origin_namespace"],
                        snapshot["origin_payload_id"],
                    )
                    if origin in active_barriers.payload_origins:
                        continue
                    if _retained(snapshot["retain_until"], current_time):
                        continue
                    cursor.execute(
                        """
                        DELETE FROM rag_evidence_snapshots
                        WHERE profile_id = ? AND payload_id = ?
                          AND NOT EXISTS (
                              SELECT 1
                              FROM rag_trace_evidence_refs
                              WHERE profile_id = ? AND snapshot_payload_id = ?
                          )
                        """,
                        (
                            identity.profile_id,
                            snapshot["payload_id"],
                            identity.profile_id,
                            snapshot["payload_id"],
                        ),
                    )
                    snapshots_collected += max(cursor.rowcount, 0)

            (
                expired_tombstones,
                tombstone_after,
                tombstone_until,
            ) = _tombstone_scan_page(
                cursor,
                profile_id=identity.profile_id,
                now=current_time.isoformat(),
                after=tombstone_after,
                until=tombstone_until,
                limit=scan_limit,
            )
            for tombstone in expired_tombstones:
                if tombstones_collected >= batch_limit:
                    break
                tombstone_after = (
                    tombstone["retain_until"],
                    tombstone["origin_namespace"],
                    tombstone["origin_payload_id"],
                )
                origin = (
                    tombstone["origin_namespace"],
                    tombstone["origin_payload_id"],
                )
                if origin in active_barriers.payload_origins:
                    continue
                cursor.execute(
                    """
                    DELETE FROM rag_payload_tombstones
                    WHERE profile_id = ?
                      AND origin_namespace = ?
                      AND origin_payload_id = ?
                      AND retain_until <= ?
                    """,
                    (
                        identity.profile_id,
                        *origin,
                        current_time.isoformat(),
                    ),
                )
                tombstones_collected += max(cursor.rowcount, 0)

        return CitationCollectionResult(
            traces_examined=traces_examined,
            traces_collected=traces_collected,
            snapshots_collected=snapshots_collected,
            tombstones_collected=tombstones_collected,
            continuation_cursor=_encode_collection_cursor(
                _CollectionCursorState(
                    profile_id=identity.profile_id,
                    trace_after=trace_after,
                    trace_until=trace_until,
                    tombstone_after=tombstone_after,
                    tombstone_until=tombstone_until,
                )
            ),
        )

    def _trace_has_barrier(
        self,
        cursor: sqlite3.Cursor,
        *,
        profile_id: str,
        trace_id: str,
        now: datetime,
        barriers: CitationCollectionBarriers,
    ) -> bool:
        if trace_id in barriers.trace_ids:
            return True
        owners = cursor.execute(
            """
            SELECT
                owner.state,
                owner.updated_at AS owner_updated_at,
                message.deleted AS message_deleted,
                message.last_modified AS message_updated_at,
                conversation.deleted AS conversation_deleted,
                conversation.last_modified AS conversation_updated_at
            FROM rag_message_trace_owners AS owner
            LEFT JOIN messages AS message ON message.id = owner.message_id
            LEFT JOIN conversations AS conversation
              ON conversation.id = message.conversation_id
            WHERE owner.profile_id = ? AND owner.trace_id = ?
            """,
            (profile_id, trace_id),
        ).fetchall()
        retention = timedelta(
            seconds=self.retention_policy.soft_deleted_owner_retention_seconds
        )
        for owner in owners:
            live_message = (
                owner["state"] != "deleted"
                and owner["message_deleted"] == 0
                and owner["conversation_deleted"] == 0
            )
            if live_message:
                return True
            timestamps = tuple(
                _parse_timestamp(owner[field])
                for field in (
                    "owner_updated_at",
                    "message_updated_at",
                    "conversation_updated_at",
                )
                if owner[field] is not None
            )
            latest = max(timestamps, default=datetime.max.replace(tzinfo=UTC))
            if latest == datetime.max.replace(tzinfo=UTC) or latest + retention > now:
                return True
        if cursor.execute(
            """
            SELECT 1
            FROM rag_artifact_owner_leases
            WHERE profile_id = ? AND trace_id = ? AND state != 'released'
            LIMIT 1
            """,
            (profile_id, trace_id),
        ).fetchone():
            return True
        if cursor.execute(
            """
            SELECT 1
            FROM rag_artifact_owner_operations
            WHERE profile_id = ? AND trace_id = ? AND state != 'acknowledged'
            LIMIT 1
            """,
            (profile_id, trace_id),
        ).fetchone():
            return True
        retained = cursor.execute(
            """
            SELECT retain_until
            FROM rag_evidence_snapshots
            WHERE profile_id = ? AND payload_id IN (
                SELECT snapshot_payload_id
                FROM rag_trace_evidence_refs
                WHERE profile_id = ? AND trace_id = ?
            )
            UNION ALL
            SELECT retain_until
            FROM rag_answer_attempt_payloads
            WHERE profile_id = ? AND trace_id = ?
            """,
            (profile_id, profile_id, trace_id, profile_id, trace_id),
        ).fetchall()
        if any(_retained(row["retain_until"], now) for row in retained):
            return True
        sync_origins = set(barriers.payload_origins)
        if sync_origins:
            origins = cursor.execute(
                """
                SELECT DISTINCT
                    snapshot.origin_namespace,
                    snapshot.origin_payload_id
                FROM rag_trace_evidence_refs AS reference
                JOIN rag_evidence_snapshots AS snapshot
                  ON snapshot.profile_id = reference.profile_id
                 AND snapshot.payload_id = reference.snapshot_payload_id
                WHERE reference.profile_id = ? AND reference.trace_id = ?
                """,
                (profile_id, trace_id),
            ).fetchall()
            if any(
                (row["origin_namespace"], row["origin_payload_id"]) in sync_origins
                for row in origins
            ):
                return True
        return False

    def _after_snapshot_purge(self) -> None:
        """Test seam after snapshot clearing and before attempt clearing."""

    def _before_collect_delete(
        self,
        cursor: sqlite3.Cursor,
        profile_id: str,
        trace_id: str,
    ) -> None:
        """Test seam before the final barrier recheck."""

    def _after_collect_refs(self) -> None:
        """Test seam after reference deletion for rollback verification."""


def _trace_scan_page(
    cursor: sqlite3.Cursor,
    *,
    profile_id: str,
    after: tuple[str, str] | None,
    until: tuple[str, str] | None,
    limit: int,
) -> tuple[list[sqlite3.Row], tuple[str, str] | None, tuple[str, str] | None]:
    if until is None or (after is not None and after >= until):
        after = None
        until = _trace_high_water(cursor, profile_id=profile_id)
    if until is None:
        return [], None, None
    rows = _query_trace_scan_page(
        cursor,
        profile_id=profile_id,
        after=after,
        until=until,
        limit=limit,
    )
    if not rows and after is not None:
        after = None
        until = _trace_high_water(cursor, profile_id=profile_id)
        if until is not None:
            rows = _query_trace_scan_page(
                cursor,
                profile_id=profile_id,
                after=None,
                until=until,
                limit=limit,
            )
    return rows, after, until


def _trace_high_water(
    cursor: sqlite3.Cursor,
    *,
    profile_id: str,
) -> tuple[str, str] | None:
    row = cursor.execute(
        """
        SELECT sealed_at, trace_id
        FROM rag_citation_traces
        WHERE profile_id = ?
        ORDER BY sealed_at DESC, trace_id DESC
        LIMIT 1
        """,
        (profile_id,),
    ).fetchone()
    return None if row is None else (row["sealed_at"], row["trace_id"])


def _query_trace_scan_page(
    cursor: sqlite3.Cursor,
    *,
    profile_id: str,
    after: tuple[str, str] | None,
    until: tuple[str, str],
    limit: int,
) -> list[sqlite3.Row]:
    return cursor.execute(
        """
        SELECT sealed_at, trace_id
        FROM rag_citation_traces
        WHERE profile_id = ?
          AND (
              ? IS NULL
              OR sealed_at > ?
              OR (sealed_at = ? AND trace_id > ?)
          )
          AND (
              sealed_at < ?
              OR (sealed_at = ? AND trace_id <= ?)
          )
        ORDER BY sealed_at, trace_id
        LIMIT ?
        """,
        (
            profile_id,
            None if after is None else after[0],
            None if after is None else after[0],
            None if after is None else after[0],
            None if after is None else after[1],
            until[0],
            until[0],
            until[1],
            limit,
        ),
    ).fetchall()


def _tombstone_scan_page(
    cursor: sqlite3.Cursor,
    *,
    profile_id: str,
    now: str,
    after: tuple[str, str, str] | None,
    until: tuple[str, str, str] | None,
    limit: int,
) -> tuple[
    list[sqlite3.Row],
    tuple[str, str, str] | None,
    tuple[str, str, str] | None,
]:
    if until is None or (after is not None and after >= until):
        after = None
        until = _tombstone_high_water(cursor, profile_id=profile_id, now=now)
    if until is None:
        return [], None, None
    rows = _query_tombstone_scan_page(
        cursor,
        profile_id=profile_id,
        now=now,
        after=after,
        until=until,
        limit=limit,
    )
    if not rows and after is not None:
        after = None
        until = _tombstone_high_water(cursor, profile_id=profile_id, now=now)
        if until is not None:
            rows = _query_tombstone_scan_page(
                cursor,
                profile_id=profile_id,
                now=now,
                after=None,
                until=until,
                limit=limit,
            )
    return rows, after, until


def _tombstone_high_water(
    cursor: sqlite3.Cursor,
    *,
    profile_id: str,
    now: str,
) -> tuple[str, str, str] | None:
    row = cursor.execute(
        """
        SELECT retain_until, origin_namespace, origin_payload_id
        FROM rag_payload_tombstones
        WHERE profile_id = ? AND retain_until <= ?
        ORDER BY retain_until DESC, origin_namespace DESC, origin_payload_id DESC
        LIMIT 1
        """,
        (profile_id, now),
    ).fetchone()
    if row is None:
        return None
    return (
        row["retain_until"],
        row["origin_namespace"],
        row["origin_payload_id"],
    )


def _query_tombstone_scan_page(
    cursor: sqlite3.Cursor,
    *,
    profile_id: str,
    now: str,
    after: tuple[str, str, str] | None,
    until: tuple[str, str, str],
    limit: int,
) -> list[sqlite3.Row]:
    return cursor.execute(
        """
        SELECT retain_until, origin_namespace, origin_payload_id
        FROM rag_payload_tombstones
        WHERE profile_id = ? AND retain_until <= ?
          AND (
              ? IS NULL
              OR retain_until > ?
              OR (retain_until = ? AND origin_namespace > ?)
              OR (
                  retain_until = ?
                  AND origin_namespace = ?
                  AND origin_payload_id > ?
              )
          )
          AND (
              retain_until < ?
              OR (retain_until = ? AND origin_namespace < ?)
              OR (
                  retain_until = ?
                  AND origin_namespace = ?
                  AND origin_payload_id <= ?
              )
          )
        ORDER BY retain_until, origin_namespace, origin_payload_id
        LIMIT ?
        """,
        (
            profile_id,
            now,
            None if after is None else after[0],
            None if after is None else after[0],
            None if after is None else after[0],
            None if after is None else after[1],
            None if after is None else after[0],
            None if after is None else after[1],
            None if after is None else after[2],
            until[0],
            until[0],
            until[1],
            until[0],
            until[1],
            until[2],
            limit,
        ),
    ).fetchall()


def _decode_collection_cursor(
    value: str | None,
    *,
    profile_id: str,
) -> _CollectionCursorState:
    if value is None:
        return _CollectionCursorState(profile_id=profile_id)
    try:
        padding = "=" * (-len(value) % 4)
        raw = base64.b64decode(
            value + padding,
            altchars=b"-_",
            validate=True,
        )
        state = _CollectionCursorState.model_validate_json(raw, strict=True)
    except (binascii.Error, UnicodeError, ValueError) as error:
        raise ValueError("invalid collection continuation cursor") from error
    if state.profile_id != profile_id:
        raise CitationPersistenceUnavailable("collection_cursor_profile_mismatch")
    return state


def _encode_collection_cursor(
    state: _CollectionCursorState,
) -> CitationCollectionContinuationCursor:
    raw = json.dumps(
        state.model_dump(mode="json"),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return TypeAdapter(CitationCollectionContinuationCursor).validate_python(
        encoded,
        strict=True,
    )


def _parse_timestamp(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (AttributeError, TypeError, ValueError):
            return datetime.max.replace(tzinfo=UTC)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        # ChaChaNotes' older CURRENT_TIMESTAMP columns are UTC but timezone-naive.
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _retained(value: str | None, now: datetime) -> bool:
    return value is not None and _parse_timestamp(value) > now


__all__ = [
    "CitationCollectionBarriers",
    "CitationCollectionContinuationCursor",
    "CitationCollectionResult",
    "CitationPayloadLifecycle",
    "PayloadRetentionPolicy",
    "PayloadTombstone",
    "SnapshotDedupeScope",
]
