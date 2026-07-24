"""Crash-safe cross-store ownership for saved citation artifacts."""

from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import UTC, datetime
from enum import Enum
import re
import sqlite3
from typing import TYPE_CHECKING, Annotated, Any, Protocol, runtime_checkable

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from tldw_chatbook.Chat.citation_trace_identity import BoundedIdentifier
from tldw_chatbook.Chat.citation_trace_repository import (
    CitationArtifactOwnerRequest,
    CitationPersistenceUnavailable,
    CitationTraceRepository,
)

if TYPE_CHECKING:
    from tldw_chatbook.Chat.citation_payload_lifecycle import (
        CitationCollectionBarriers,
    )


ARTIFACT_PROVENANCE_OUTBOX_MAX_ENTRIES = 2_048
ARTIFACT_RECONCILIATION_BATCH_MAX = 100
_ERROR_CODE_MAX_CHARS = 128
_SAFE_ERROR_CODE = re.compile(r"[a-z][a-z0-9_]{0,127}\Z")
_SQL_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")


def _aware_utc(value: datetime) -> datetime:
    if value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


UtcDateTime = Annotated[datetime, AfterValidator(_aware_utc)]
ArtifactRevision = Annotated[int, Field(strict=True, ge=0)]
ErrorCode = Annotated[
    str,
    Field(
        min_length=1,
        max_length=_ERROR_CODE_MAX_CHARS,
        pattern=r"^[a-z][a-z0-9_]{0,127}$",
    ),
]


class ArtifactBackendMode(str, Enum):
    """Durability boundary used by an artifact store."""

    CROSS_STORE = "cross_store"
    SHARED_DATABASE = "shared_database"


class ArtifactOwnerOperationKind(str, Enum):
    """Idempotent transition requested by the artifact registry."""

    LINK = "link"
    UNLINK = "unlink"


class ArtifactOwnerOutboxState(str, Enum):
    """Artifact-side durable handshake state."""

    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class ArtifactOwnerBinding(_FrozenModel):
    """Opaque stable owner tuple retained by the artifact registry."""

    schema_version: Annotated[int, Field(strict=True, ge=1, le=1)] = 1
    profile_id: BoundedIdentifier
    artifact_store_id: BoundedIdentifier
    artifact_id: BoundedIdentifier
    artifact_revision: ArtifactRevision
    trace_id: BoundedIdentifier
    lease_id: BoundedIdentifier
    binding_id: BoundedIdentifier


class ArtifactOwnerOperation(_FrozenModel):
    """One bounded registry outbox entry."""

    schema_version: Annotated[int, Field(strict=True, ge=1, le=1)] = 1
    operation_id: BoundedIdentifier
    operation_kind: ArtifactOwnerOperationKind
    binding: ArtifactOwnerBinding
    state: ArtifactOwnerOutboxState = ArtifactOwnerOutboxState.PENDING
    created_at: UtcDateTime
    acknowledged_at: UtcDateTime | None = None
    error_code: ErrorCode | None = None

    @model_validator(mode="after")
    def _validate_state(self) -> "ArtifactOwnerOperation":
        if self.state is ArtifactOwnerOutboxState.PENDING:
            if self.acknowledged_at is not None:
                raise ValueError("pending operation cannot have acknowledged_at")
        elif self.acknowledged_at is None:
            raise ValueError("acknowledged operation requires acknowledged_at")
        return self


class ArtifactReconciliationResult(_FrozenModel):
    """Bounded summary suitable for startup diagnostics."""

    examined: Annotated[int, Field(strict=True, ge=0)]
    completed: Annotated[int, Field(strict=True, ge=0)]
    failed: Annotated[int, Field(strict=True, ge=0)]
    disabled: bool = False
    operation_ids: tuple[BoundedIdentifier, ...] = ()
    reason_codes: tuple[ErrorCode, ...] = ()


@runtime_checkable
class CrossStoreArtifactOwnershipStore(Protocol):
    """Minimal durable outbox seam used by the coordinator."""

    artifact_backend_mode: ArtifactBackendMode
    artifact_store_id: str

    def list_provenance_outbox(self, *, limit: int) -> list[ArtifactOwnerOperation]:
        """Read a bounded ordered batch."""

    def mark_provenance_operation_acknowledged(self, operation_id: str) -> None:
        """Durably mark the artifact-side mutation acknowledged."""

    def prune_provenance_operation(self, operation_id: str) -> None:
        """Prune an operation only after trace-side finalization."""

    def record_provenance_operation_failure(
        self,
        operation_id: str,
        reason_code: str,
    ) -> None:
        """Persist a bounded sanitized retry reason."""

    def provenance_collection_guard(self) -> AbstractContextManager[Any]:
        """Serialize registry mutation with barrier-protected collection."""


@runtime_checkable
class SharedDatabaseArtifactOwnershipStore(Protocol):
    """Artifact owner store mutated through the repository SQLite cursor."""

    artifact_backend_mode: ArtifactBackendMode
    artifact_store_id: str
    artifact_database: Any
    artifact_table: str
    artifact_owner_table: str

    def apply_shared_database_owner_mutation(
        self,
        cursor: sqlite3.Cursor,
        operation: ArtifactOwnerOperation,
    ) -> None:
        """Mutate artifact row and its real-FK owner through ``cursor``."""


class CitationArtifactOwnershipCoordinator:
    """Coordinate the artifact registry and trace repository handshakes."""

    def __init__(
        self,
        *,
        artifact_store: Any,
        trace_repository: CitationTraceRepository,
    ) -> None:
        mode = getattr(artifact_store, "artifact_backend_mode", None)
        if mode is ArtifactBackendMode.SHARED_DATABASE:
            _validate_shared_database_contract(artifact_store, trace_repository)
        elif mode is ArtifactBackendMode.CROSS_STORE:
            required = (
                "list_provenance_outbox",
                "mark_provenance_operation_acknowledged",
                "prune_provenance_operation",
                "record_provenance_operation_failure",
                "provenance_collection_guard",
            )
            if any(
                not callable(getattr(artifact_store, name, None)) for name in required
            ):
                raise ValueError("cross_store_outbox_contract_required")
        else:
            raise ValueError("artifact_backend_mode_unsupported")
        self.artifact_store = artifact_store
        self.trace_repository = trace_repository
        if mode is ArtifactBackendMode.CROSS_STORE:
            trace_repository.register_artifact_collection_barrier(
                provider=self.collection_barriers,
                guard=artifact_store.provenance_collection_guard,
            )
        self.backend_mode = mode

    @property
    def writes_enabled(self) -> bool:
        """Return the authoritative canonical-write recovery switch."""

        return self.trace_repository.policy.canonical_writes_enabled

    def prepare_link_operation(
        self,
        request: CitationArtifactOwnerRequest,
        *,
        artifact_id: str,
        artifact_revision: int,
    ) -> ArtifactOwnerOperation:
        """Derive one stable link only from a repository-issued owner request."""

        if not self.writes_enabled:
            raise CitationPersistenceUnavailable("canonical_citation_writes_disabled")
        return self.trace_repository.prepare_artifact_owner_operation(
            request,
            artifact_store_id=self.artifact_store.artifact_store_id,
            artifact_id=artifact_id,
            artifact_revision=artifact_revision,
            operation_kind=ArtifactOwnerOperationKind.LINK,
        )

    def owner_request_for_message(
        self,
        *,
        message_id: str,
        message_revision: int,
        current_body: str,
    ) -> CitationArtifactOwnerRequest | None:
        """Issue an opaque owner request only while canonical writes are enabled."""

        if not self.writes_enabled:
            return None
        return self.trace_repository.get_artifact_owner_request(
            message_id=message_id,
            message_revision=message_revision,
            current_body=current_body,
        )

    def prepare_unlink_operation(
        self,
        binding: ArtifactOwnerBinding,
    ) -> ArtifactOwnerOperation:
        """Derive the separately idempotent unlink for one verified binding."""

        if not self.writes_enabled:
            raise CitationPersistenceUnavailable("canonical_citation_writes_disabled")
        return self.trace_repository.prepare_artifact_unlink_operation(binding)

    def apply_shared_database_owner_operation(
        self,
        operation: ArtifactOwnerOperation,
    ) -> None:
        """Apply the artifact and real-FK owner mutation in one SQLite tx."""

        if self.backend_mode is not ArtifactBackendMode.SHARED_DATABASE:
            raise ValueError("shared_database_owner_contract_required")
        with self.trace_repository.db.transaction() as cursor:
            validated = (
                self.trace_repository.validate_shared_database_artifact_owner_operation(
                    cursor,
                    operation,
                )
            )
            self.artifact_store.apply_shared_database_owner_mutation(
                cursor,
                validated,
            )

    def reconcile_pending(
        self,
        *,
        limit: int = 25,
    ) -> ArtifactReconciliationResult:
        """Reconcile a bounded ordered batch without exposing payload details."""

        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= ARTIFACT_RECONCILIATION_BATCH_MAX
        ):
            raise ValueError(f"limit must be 1..{ARTIFACT_RECONCILIATION_BATCH_MAX}")
        if not self.writes_enabled:
            return ArtifactReconciliationResult(
                examined=0,
                completed=0,
                failed=0,
                disabled=True,
            )
        try:
            operations = self.artifact_store.list_provenance_outbox(limit=limit)
            if not isinstance(operations, list) or len(operations) > limit:
                raise ValueError("artifact outbox batch contract violated")
        except Exception:
            return ArtifactReconciliationResult(
                examined=0,
                completed=0,
                failed=1,
                reason_codes=("artifact_registry_unavailable",),
            )

        completed = 0
        failed = 0
        completed_ids: list[str] = []
        failed_ids: list[str] = []
        reasons: list[str] = []
        for operation in operations:
            try:
                validated = ArtifactOwnerOperation.model_validate(
                    operation.model_dump(mode="python"),
                    strict=True,
                )
                self.trace_repository.apply_artifact_owner_operation(validated)
                if validated.state is ArtifactOwnerOutboxState.PENDING:
                    self.artifact_store.mark_provenance_operation_acknowledged(
                        validated.operation_id
                    )
                    validated = validated.model_copy(
                        update={
                            "state": ArtifactOwnerOutboxState.ACKNOWLEDGED,
                            "acknowledged_at": datetime.now(UTC),
                        }
                    )
                self.trace_repository.acknowledge_artifact_owner_operation(validated)
                self.artifact_store.prune_provenance_operation(validated.operation_id)
            except Exception as exc:
                reason = _reconciliation_reason(exc)
                failed += 1
                operation_id = getattr(operation, "operation_id", None)
                if (
                    isinstance(operation_id, str)
                    and operation_id
                    and len(operation_id.encode("utf-8")) <= 256
                ):
                    failed_ids.append(operation_id)
                reasons.append(reason)
                try:
                    self.artifact_store.record_provenance_operation_failure(
                        operation.operation_id,
                        reason,
                    )
                except Exception:
                    pass
                continue
            completed += 1
            completed_ids.append(validated.operation_id)

        return ArtifactReconciliationResult(
            examined=len(operations),
            completed=completed,
            failed=failed,
            operation_ids=tuple(failed_ids if failed else completed_ids),
            reason_codes=tuple(reasons),
        )

    def collection_barriers(self) -> CitationCollectionBarriers:
        """Return pending cross-store trace IDs or fail closed on registry damage."""

        from tldw_chatbook.Chat.citation_payload_lifecycle import (
            CitationCollectionBarriers,
        )

        if not self.writes_enabled:
            return CitationCollectionBarriers()
        try:
            operations = self.artifact_store.list_provenance_outbox(
                limit=ARTIFACT_PROVENANCE_OUTBOX_MAX_ENTRIES
            )
            if (
                not isinstance(operations, list)
                or len(operations) > ARTIFACT_PROVENANCE_OUTBOX_MAX_ENTRIES
            ):
                raise ValueError("artifact outbox batch contract violated")
            trace_ids = {
                ArtifactOwnerOperation.model_validate(
                    operation.model_dump(mode="python"),
                    strict=True,
                ).binding.trace_id
                for operation in operations
            }
        except Exception:
            raise CitationPersistenceUnavailable(
                "artifact_registry_unavailable"
            ) from None
        return CitationCollectionBarriers(trace_ids=tuple(sorted(trace_ids)))


def _reconciliation_reason(exc: Exception) -> str:
    if (
        isinstance(exc, CitationPersistenceUnavailable)
        and _SAFE_ERROR_CODE.fullmatch(exc.reason_code) is not None
    ):
        return exc.reason_code
    if isinstance(exc, (TypeError, ValueError)):
        return "artifact_operation_invalid"
    return "artifact_reconciliation_failed"


def _validate_shared_database_contract(
    artifact_store: Any,
    trace_repository: CitationTraceRepository,
) -> None:
    if getattr(
        artifact_store, "artifact_database", None
    ) is not trace_repository.db or not callable(
        getattr(artifact_store, "apply_shared_database_owner_mutation", None)
    ):
        raise ValueError("shared_database_owner_contract_required")
    artifact_table = getattr(artifact_store, "artifact_table", None)
    owner_table = getattr(artifact_store, "artifact_owner_table", None)
    if not all(
        isinstance(table, str) and _SQL_IDENTIFIER.fullmatch(table)
        for table in (artifact_table, owner_table)
    ):
        raise ValueError("shared_database_owner_contract_required")
    connection = trace_repository.db.get_connection()
    if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise ValueError("shared_database_owner_contract_required")
    if (
        not connection.execute(f'PRAGMA table_info("{artifact_table}")').fetchall()
        or not connection.execute(f'PRAGMA table_info("{owner_table}")').fetchall()
    ):
        raise ValueError("shared_database_owner_contract_required")

    groups: dict[tuple[int, str], set[tuple[str, str]]] = {}
    for row in connection.execute(
        f'PRAGMA foreign_key_list("{owner_table}")'
    ).fetchall():
        groups.setdefault((row["id"], row["table"]), set()).add(
            (row["from"], row["to"])
        )
    trace_fk = {("profile_id", "profile_id"), ("trace_id", "trace_id")}
    artifact_fk = {
        ("artifact_id", "artifact_id"),
        ("artifact_revision", "artifact_revision"),
    }
    if not any(
        table == "rag_citation_traces" and trace_fk <= columns
        for (_identifier, table), columns in groups.items()
    ) or not any(
        table == artifact_table and artifact_fk <= columns
        for (_identifier, table), columns in groups.items()
    ):
        raise ValueError("shared_database_owner_contract_required")


__all__ = [
    "ARTIFACT_PROVENANCE_OUTBOX_MAX_ENTRIES",
    "ARTIFACT_RECONCILIATION_BATCH_MAX",
    "ArtifactBackendMode",
    "ArtifactOwnerBinding",
    "ArtifactOwnerOperation",
    "ArtifactOwnerOperationKind",
    "ArtifactOwnerOutboxState",
    "ArtifactReconciliationResult",
    "CitationArtifactOwnershipCoordinator",
    "SharedDatabaseArtifactOwnershipStore",
]
