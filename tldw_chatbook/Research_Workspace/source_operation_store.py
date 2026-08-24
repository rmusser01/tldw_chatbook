"""Small optimistic SQLite store for Research source-operation receipts."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import sqlite3

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

from .source_operations import (
    ResearchSourceOperation,
    SourceOperationStage,
    SourceOperationStatus,
    SourceOperationValidationError,
    validate_source_operation_id,
)


MAX_INCOMPLETE_PAGE = 100
MAX_INCOMPLETE_OFFSET = 10_000


class SourceOperationConflictError(RuntimeError):
    """Raised for duplicate identity or optimistic-revision conflicts."""


_COLUMNS = (
    "operation_id",
    "idempotency_key",
    "data_source",
    "server_profile_id",
    "principal_id",
    "workspace_id",
    "ingest_job_id",
    "canonical_item_type",
    "canonical_item_id",
    "workspace_source_id",
    "desired_selected",
    "catalog_status",
    "association_status",
    "readiness_status",
    "error_stage",
    "error_code",
    "error_message",
    "revision",
    "created_at",
    "updated_at",
)
_SELECT_COLUMNS = ", ".join(_COLUMNS)
_STATUS_FIELDS = {
    SourceOperationStage.CATALOG: "catalog_status",
    SourceOperationStage.ASSOCIATION: "association_status",
    SourceOperationStage.READINESS: "readiness_status",
}


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _operation_from_row(row: tuple[object, ...]) -> ResearchSourceOperation:
    values = dict(zip(_COLUMNS, row))
    return ResearchSourceOperation(
        operation_id=values["operation_id"],
        idempotency_key=values["idempotency_key"],
        data_source=values["data_source"],
        server_profile_id=values["server_profile_id"],
        principal_id=values["principal_id"],
        workspace_id=values["workspace_id"],
        ingest_job_id=values["ingest_job_id"],
        canonical_item_type=values["canonical_item_type"],
        canonical_item_id=values["canonical_item_id"],
        workspace_source_id=values["workspace_source_id"],
        desired_selected=bool(values["desired_selected"]),
        catalog_status=values["catalog_status"],
        association_status=values["association_status"],
        readiness_status=values["readiness_status"],
        error_stage=values["error_stage"],
        error_code=values["error_code"],
        error_message=values["error_message"],
        revision=values["revision"],
        created_at=values["created_at"],
        updated_at=values["updated_at"],
    )


class ResearchSourceOperationStore:
    """Persist source intents and monotonic stage transitions in WorkspaceDB."""

    def __init__(self, db: WorkspaceDB) -> None:
        if not isinstance(db, WorkspaceDB):
            raise TypeError("db must be WorkspaceDB")
        self._db = db

    def create(self, operation: ResearchSourceOperation) -> ResearchSourceOperation:
        """Insert one validated intent before catalog ingestion begins."""

        if not isinstance(operation, ResearchSourceOperation):
            raise TypeError("operation must be ResearchSourceOperation")
        if (
            operation.revision != 1
            or operation.ingest_job_id
            or operation.canonical_item_id
            or operation.workspace_source_id
            or operation.catalog_status is not SourceOperationStatus.PENDING
            or operation.association_status is not SourceOperationStatus.PENDING
            or operation.readiness_status is not SourceOperationStatus.PENDING
            or operation.error_stage is not None
            or operation.error_code
            or operation.error_message
        ):
            raise SourceOperationValidationError(
                "create requires a pristine revision-1 durable intent"
            )
        values = (
            operation.operation_id,
            operation.idempotency_key,
            operation.data_source.value,
            operation.server_profile_id,
            operation.principal_id,
            operation.workspace_id,
            operation.ingest_job_id,
            operation.canonical_item_type.value,
            operation.canonical_item_id,
            operation.workspace_source_id,
            int(operation.desired_selected),
            operation.catalog_status.value,
            operation.association_status.value,
            operation.readiness_status.value,
            operation.error_stage.value if operation.error_stage is not None else None,
            operation.error_code,
            operation.error_message,
            operation.revision,
            operation.created_at,
            operation.updated_at,
        )
        try:
            with self._db.transaction() as connection:
                connection.execute(
                    f"""
                    INSERT INTO research_source_operations ({_SELECT_COLUMNS})
                    VALUES ({", ".join("?" for _ in _COLUMNS)})
                    """,
                    values,
                )
        except sqlite3.IntegrityError as exc:
            message = str(exc).lower()
            field = "idempotency key" if "idempotency" in message else "operation id"
            raise SourceOperationConflictError(
                f"duplicate source-operation {field}"
            ) from None
        return operation

    def get(self, operation_id: str) -> ResearchSourceOperation | None:
        """Return one receipt by bounded opaque ID."""

        normalized_id = validate_source_operation_id(operation_id)
        with self._db.connection() as connection:
            row = connection.execute(
                f"""
                SELECT {_SELECT_COLUMNS}
                FROM research_source_operations
                WHERE operation_id = ?
                """,
                (normalized_id,),
            ).fetchone()
        return _operation_from_row(row) if row is not None else None

    def get_by_idempotency_key(
        self, idempotency_key: str
    ) -> ResearchSourceOperation | None:
        """Return the existing qualified intent for idempotent convergence."""

        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise SourceOperationValidationError(
                "idempotency_key must be nonblank text"
            )
        normalized = idempotency_key.strip()
        if len(normalized) > 512 or len(normalized.encode("utf-8")) > 512:
            raise SourceOperationValidationError("idempotency_key is too long")
        with self._db.connection() as connection:
            row = connection.execute(
                f"""
                SELECT {_SELECT_COLUMNS}
                FROM research_source_operations
                WHERE idempotency_key = ?
                """,
                (normalized,),
            ).fetchone()
        return _operation_from_row(row) if row is not None else None

    def list_incomplete(
        self, *, limit: int = 50, offset: int = 0
    ) -> tuple[ResearchSourceOperation, ...]:
        """Return a bounded stable page of receipts not fully succeeded."""

        if type(limit) is not int or not 1 <= limit <= MAX_INCOMPLETE_PAGE:
            raise SourceOperationValidationError(
                f"limit must be between 1 and {MAX_INCOMPLETE_PAGE}"
            )
        if type(offset) is not int or not 0 <= offset <= MAX_INCOMPLETE_OFFSET:
            raise SourceOperationValidationError(
                f"offset must be between 0 and {MAX_INCOMPLETE_OFFSET}"
            )
        with self._db.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT {_SELECT_COLUMNS}
                FROM research_source_operations
                WHERE NOT (
                    catalog_status = ?
                    AND association_status = ?
                    AND readiness_status = ?
                )
                ORDER BY
                    CASE WHEN (
                        catalog_status = ?
                        OR association_status = ?
                        OR readiness_status = ?
                    ) THEN 1 ELSE 0 END ASC,
                    created_at ASC,
                    operation_id ASC
                LIMIT ? OFFSET ?
                """,
                (
                    SourceOperationStatus.SUCCEEDED.value,
                    SourceOperationStatus.SUCCEEDED.value,
                    SourceOperationStatus.SUCCEEDED.value,
                    SourceOperationStatus.FAILED.value,
                    SourceOperationStatus.FAILED.value,
                    SourceOperationStatus.FAILED.value,
                    limit,
                    offset,
                ),
            ).fetchall()
        return tuple(_operation_from_row(row) for row in rows)

    def list_association_actionable(
        self, *, limit: int = 50
    ) -> tuple[ResearchSourceOperation, ...]:
        """Return a bounded stable page actionable by catalog/association recovery."""

        if type(limit) is not int or not 1 <= limit <= MAX_INCOMPLETE_PAGE:
            raise SourceOperationValidationError(
                f"limit must be between 1 and {MAX_INCOMPLETE_PAGE}"
            )
        with self._db.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT {_SELECT_COLUMNS}
                FROM research_source_operations
                WHERE (
                        catalog_status IN (?, ?)
                        AND ingest_job_id <> ''
                    )
                    OR (
                        catalog_status = ?
                        AND association_status IN (?, ?)
                    )
                ORDER BY created_at ASC, operation_id ASC
                LIMIT ?
                """,
                (
                    SourceOperationStatus.PENDING.value,
                    SourceOperationStatus.IN_PROGRESS.value,
                    SourceOperationStatus.SUCCEEDED.value,
                    SourceOperationStatus.PENDING.value,
                    SourceOperationStatus.IN_PROGRESS.value,
                    limit,
                ),
            ).fetchall()
        return tuple(_operation_from_row(row) for row in rows)

    def list_readiness_actionable(
        self, *, limit: int = 50
    ) -> tuple[ResearchSourceOperation, ...]:
        """Return a bounded page whose association is ready for status refresh."""

        if type(limit) is not int or not 1 <= limit <= MAX_INCOMPLETE_PAGE:
            raise SourceOperationValidationError(
                f"limit must be between 1 and {MAX_INCOMPLETE_PAGE}"
            )
        with self._db.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT {_SELECT_COLUMNS}
                FROM research_source_operations
                WHERE association_status = ?
                    AND readiness_status IN (?, ?)
                    AND workspace_source_id <> ''
                ORDER BY created_at ASC, operation_id ASC
                LIMIT ?
                """,
                (
                    SourceOperationStatus.SUCCEEDED.value,
                    SourceOperationStatus.PENDING.value,
                    SourceOperationStatus.IN_PROGRESS.value,
                    limit,
                ),
            ).fetchall()
        return tuple(_operation_from_row(row) for row in rows)

    def advance_stage(
        self,
        operation_id: str,
        *,
        stage: SourceOperationStage,
        status: SourceOperationStatus,
        expected_revision: int,
        ingest_job_id: str | None = None,
        canonical_item_id: str | None = None,
        workspace_source_id: str | None = None,
        error_code: str = "",
        error_message: str = "",
        timestamp: str | None = None,
    ) -> ResearchSourceOperation:
        """Advance one stage without regressions, skips, or stale writes.

        ``pending`` may move to ``in_progress``, ``succeeded``, or ``failed``;
        ``in_progress`` may move only to ``succeeded`` or ``failed``. A
        succeeded or failed stage is terminal. Only :meth:`retry_failed_stage`
        can clear a failed stage back to ``pending``.
        """

        normalized_stage = self._stage(stage)
        normalized_status = self._status(status)
        if normalized_status is SourceOperationStatus.PENDING:
            raise SourceOperationValidationError(
                "advance_stage cannot set pending; use retry_failed_stage"
            )
        self._expected_revision(expected_revision)
        with self._db.transaction() as connection:
            current = self._get_in_transaction(connection, operation_id)
            self._assert_revision(current, expected_revision)
            field_name = _STATUS_FIELDS[normalized_stage]
            current_status = getattr(current, field_name)
            if current_status in {
                SourceOperationStatus.SUCCEEDED,
                SourceOperationStatus.FAILED,
            }:
                raise SourceOperationValidationError(
                    f"{normalized_stage.value} is terminal; failed stages require retry_failed_stage"
                )
            if (
                current_status is SourceOperationStatus.IN_PROGRESS
                and normalized_status is SourceOperationStatus.IN_PROGRESS
            ):
                raise SourceOperationValidationError(
                    f"{normalized_stage.value} status must advance forward"
                )
            self._assert_stage_prerequisites(current, normalized_stage)
            if normalized_status is SourceOperationStatus.FAILED:
                next_error_stage: SourceOperationStage | None = normalized_stage
                next_error_code = error_code
                next_error_message = error_message
            else:
                if error_code or error_message:
                    raise SourceOperationValidationError(
                        "diagnostics are allowed only for a failed stage"
                    )
                next_error_stage = None
                next_error_code = ""
                next_error_message = ""

            changes: dict[str, object] = {
                field_name: normalized_status,
                "error_stage": next_error_stage,
                "error_code": next_error_code,
                "error_message": next_error_message,
                "revision": current.revision + 1,
                "updated_at": timestamp or _utc_now(),
            }
            if ingest_job_id is not None:
                if normalized_stage is not SourceOperationStage.CATALOG:
                    raise SourceOperationValidationError(
                        "ingest_job_id may change only during the catalog stage"
                    )
                changes["ingest_job_id"] = ingest_job_id
            if canonical_item_id is not None:
                if normalized_stage is not SourceOperationStage.CATALOG:
                    raise SourceOperationValidationError(
                        "canonical_item_id may change only during the catalog stage"
                    )
                changes["canonical_item_id"] = canonical_item_id
            if workspace_source_id is not None:
                if normalized_stage is not SourceOperationStage.ASSOCIATION:
                    raise SourceOperationValidationError(
                        "workspace_source_id may change only during the association stage"
                    )
                changes["workspace_source_id"] = workspace_source_id
            updated = replace(current, **changes)
            self._write_transition(connection, updated, expected_revision)
        return updated

    def retry_failed_stage(
        self,
        operation_id: str,
        *,
        stage: SourceOperationStage,
        expected_revision: int,
        timestamp: str | None = None,
    ) -> ResearchSourceOperation:
        """Clear exactly the named failed stage to pending for explicit retry."""

        normalized_stage = self._stage(stage)
        self._expected_revision(expected_revision)
        with self._db.transaction() as connection:
            current = self._get_in_transaction(connection, operation_id)
            self._assert_revision(current, expected_revision)
            if (
                current.error_stage is not normalized_stage
                or getattr(current, _STATUS_FIELDS[normalized_stage])
                is not SourceOperationStatus.FAILED
            ):
                raise SourceOperationConflictError(
                    f"{normalized_stage.value} is not the operation's failed stage"
                )
            updated = replace(
                current,
                **{
                    _STATUS_FIELDS[normalized_stage]: SourceOperationStatus.PENDING,
                    "error_stage": None,
                    "error_code": "",
                    "error_message": "",
                    "revision": current.revision + 1,
                    "updated_at": timestamp or _utc_now(),
                },
            )
            self._write_transition(connection, updated, expected_revision)
        return updated

    @staticmethod
    def _stage(value: object) -> SourceOperationStage:
        try:
            return SourceOperationStage(value)
        except (TypeError, ValueError):
            raise SourceOperationValidationError("stage is invalid") from None

    @staticmethod
    def _status(value: object) -> SourceOperationStatus:
        try:
            return SourceOperationStatus(value)
        except (TypeError, ValueError):
            raise SourceOperationValidationError("status is invalid") from None

    @staticmethod
    def _expected_revision(value: object) -> None:
        if type(value) is not int or value < 1:
            raise SourceOperationValidationError(
                "expected_revision must be a positive integer"
            )

    @staticmethod
    def _assert_revision(
        operation: ResearchSourceOperation, expected_revision: int
    ) -> None:
        if operation.revision != expected_revision:
            raise SourceOperationConflictError(
                f"source-operation revision changed: expected {expected_revision}, "
                f"found {operation.revision}"
            )

    @staticmethod
    def _assert_stage_prerequisites(
        operation: ResearchSourceOperation, stage: SourceOperationStage
    ) -> None:
        if (
            stage is SourceOperationStage.ASSOCIATION
            and operation.catalog_status is not SourceOperationStatus.SUCCEEDED
        ):
            raise SourceOperationValidationError(
                "catalog must succeed before association can advance"
            )
        if (
            stage is SourceOperationStage.READINESS
            and operation.association_status is not SourceOperationStatus.SUCCEEDED
        ):
            raise SourceOperationValidationError(
                "association must succeed before readiness can advance"
            )

    def _get_in_transaction(
        self, connection: sqlite3.Connection, operation_id: str
    ) -> ResearchSourceOperation:
        normalized_id = validate_source_operation_id(operation_id)
        row = connection.execute(
            f"SELECT {_SELECT_COLUMNS} FROM research_source_operations WHERE operation_id = ?",
            (normalized_id,),
        ).fetchone()
        if row is None:
            raise SourceOperationValidationError("source operation does not exist")
        return _operation_from_row(row)

    @staticmethod
    def _write_transition(
        connection: sqlite3.Connection,
        operation: ResearchSourceOperation,
        expected_revision: int,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE research_source_operations
            SET ingest_job_id = ?, canonical_item_id = ?, workspace_source_id = ?,
                catalog_status = ?, association_status = ?, readiness_status = ?,
                error_stage = ?, error_code = ?, error_message = ?,
                revision = ?, updated_at = ?
            WHERE operation_id = ? AND revision = ?
            """,
            (
                operation.ingest_job_id,
                operation.canonical_item_id,
                operation.workspace_source_id,
                operation.catalog_status.value,
                operation.association_status.value,
                operation.readiness_status.value,
                operation.error_stage.value
                if operation.error_stage is not None
                else None,
                operation.error_code,
                operation.error_message,
                operation.revision,
                operation.updated_at,
                operation.operation_id,
                expected_revision,
            ),
        )
        if cursor.rowcount != 1:
            raise SourceOperationConflictError(
                "source-operation revision changed during update"
            )
