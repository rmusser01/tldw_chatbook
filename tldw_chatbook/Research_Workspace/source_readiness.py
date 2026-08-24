"""Pure readiness normalization plus durable readiness receipt recovery."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
import re
from typing import Any

from .contracts import (
    QualifiedWorkspaceRef,
    RetrievalMode,
    SourceReadiness,
    SourceReadinessState,
    WorkspaceDataSource,
)
from .source_operations import (
    ResearchSourceOperation,
    SourceOperationStage,
    SourceOperationStatus,
)


_SERVER_STATE_MAP = {
    "queued": SourceReadinessState.ATTACHED,
    "ingesting": SourceReadinessState.PARSING,
    "extracting": SourceReadinessState.PARSING,
    "chunking": SourceReadinessState.INDEXING,
    "indexing": SourceReadinessState.INDEXING,
    "retrying": SourceReadinessState.INDEXING,
    "queryable": SourceReadinessState.VECTOR_READY,
    "partially_queryable": SourceReadinessState.FTS_READY,
    "failed": SourceReadinessState.FAILED,
    "missing_media": SourceReadinessState.UNAVAILABLE,
    "blocked_by_permissions": SourceReadinessState.UNAVAILABLE,
}
_SECRET_DIAGNOSTIC = re.compile(
    r"(?i)(?:bearer\s+\S+|sk-[A-Za-z0-9_-]+|"
    r"(?:api[_-]?key|password|secret|token)\s*[:=])"
)


def _safe_detail(value: object) -> str:
    text = str(value or "").strip()
    if _SECRET_DIAGNOSTIC.search(text):
        return "Readiness diagnostic withheld."
    return text[:512]


def _strict_bool(row: Mapping[str, Any], field_name: str) -> bool:
    value = row.get(field_name, False)
    if type(value) is not bool:
        raise ValueError(f"readiness {field_name} must be bool")
    return value


def normalize_server_readiness(
    *, ref: QualifiedWorkspaceRef, status: Mapping[str, Any]
) -> SourceReadiness:
    """Map one validated server source status to the closed domain vocabulary."""

    lifecycle = str(status.get("state") or "").strip()
    state = _SERVER_STATE_MAP.get(lifecycle)
    if state is None:
        raise ValueError("server source lifecycle is unknown")
    readiness = status.get("readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("server source readiness must be an object")
    stale = _strict_bool(status, "stale")
    retry_eligible = _strict_bool(status, "retry_eligible")
    if stale:
        state = SourceReadinessState.STALE
    source_id = str(status.get("id") or "").strip()
    media_id = status.get("media_id")
    if type(media_id) is not int or media_id < 1:
        raise ValueError("server source media_id must be a positive integer")
    next_action = (
        "Re-add source"
        if lifecycle in {"failed", "missing_media", "blocked_by_permissions"}
        else "Refresh status"
    )
    return SourceReadiness(
        ref=ref,
        source_id=source_id,
        catalog_item_id=str(media_id),
        state=state,
        metadata_ready=_strict_bool(readiness, "metadata_ready"),
        text_ready=_strict_bool(readiness, "text_extracted"),
        fts_ready=_strict_bool(readiness, "fts_ready"),
        vector_ready=_strict_bool(readiness, "vector_ready"),
        citation_ready=_strict_bool(readiness, "citation_ready"),
        summary_ready=_strict_bool(readiness, "summary_ready"),
        tool_ready=_strict_bool(readiness, "tool_accessible"),
        stale=stale,
        retry_eligible=retry_eligible,
        next_action=next_action,
        detail=_safe_detail(status.get("status_reason")),
    )


def normalize_local_readiness(
    *,
    ref: QualifiedWorkspaceRef,
    source_id: str,
    catalog_item_id: str,
    detail: Mapping[str, Any],
) -> SourceReadiness:
    """Derive honest Local FTS/vector readiness from canonical Media detail."""

    status = str(detail.get("chunking_status") or detail.get("status") or "").lower()
    vector_status = detail.get("vector_processing")
    text_ready = bool(
        detail.get("has_transcript")
        or detail.get("content")
        or detail.get("transcription")
    )
    fts_ready = bool(detail.get("has_chunks")) and text_ready
    vector_ready = vector_status is True or str(vector_status).lower() in {
        "complete",
        "completed",
        "ready",
        "indexed",
    }
    stale = detail.get("stale") is True
    failed = status in {"failed", "error"}
    missing = status in {"missing", "missing_media"} or detail.get("deleted") is True
    if stale:
        state = SourceReadinessState.STALE
    elif missing:
        state = SourceReadinessState.UNAVAILABLE
    elif vector_ready and fts_ready:
        state = SourceReadinessState.VECTOR_READY
    elif fts_ready:
        state = SourceReadinessState.FTS_READY
    elif failed:
        state = SourceReadinessState.FAILED
    elif status in {"chunking", "indexing", "pending", "processing"}:
        state = SourceReadinessState.INDEXING
    elif text_ready:
        state = SourceReadinessState.INDEXING
    elif status in {"extracting", "parsing", "ingesting"}:
        state = SourceReadinessState.PARSING
    else:
        state = SourceReadinessState.ATTACHED
    return SourceReadiness(
        ref=ref,
        source_id=source_id,
        catalog_item_id=catalog_item_id,
        state=state,
        metadata_ready=not missing,
        text_ready=text_ready,
        fts_ready=fts_ready,
        vector_ready=vector_ready,
        stale=stale,
        retry_eligible=failed or missing,
        next_action="Re-add source" if failed or missing else "Refresh status",
        detail=_safe_detail(status),
    )


def effective_source_ids(
    desired_ids: Sequence[str],
    readiness: Sequence[SourceReadiness],
    *,
    mode: RetrievalMode | str,
) -> tuple[str, ...]:
    """Return selected owner IDs currently usable for the requested mode."""

    try:
        requested_mode = RetrievalMode(mode)
    except (TypeError, ValueError):
        raise ValueError("mode must be fts, semantic, or hybrid") from None
    desired = frozenset(str(source_id) for source_id in desired_ids)
    usable: list[str] = []
    for row in readiness:
        if row.desired_id not in desired:
            continue
        if requested_mode is RetrievalMode.FTS:
            ready = row.fts_ready
        elif requested_mode is RetrievalMode.SEMANTIC:
            ready = row.vector_ready
        else:
            ready = row.fts_ready and row.vector_ready
        if ready:
            usable.append(row.desired_id)
    return tuple(usable)


class ResearchSourceReadinessCoordinator:
    """Refresh canonical readiness and advance only the readiness receipt."""

    def __init__(self, *, operation_store: Any, adapters: Mapping[Any, Any]) -> None:
        self._operation_store = operation_store
        self._adapters = {
            WorkspaceDataSource(data_source): adapter
            for data_source, adapter in adapters.items()
        }

    async def resume(self, operation_id: str) -> ResearchSourceOperation:
        operation = await self._require_operation(operation_id)
        if (
            operation.association_status is not SourceOperationStatus.SUCCEEDED
            or operation.readiness_status
            in {SourceOperationStatus.SUCCEEDED, SourceOperationStatus.FAILED}
        ):
            return operation
        adapter = self._adapters.get(operation.data_source)
        if adapter is None:
            return await self._fail(
                operation,
                error_code="readiness_service_unavailable",
                error_message="Source readiness service is unavailable.",
            )
        ref = self._operation_ref(operation)
        try:
            rows = await adapter.get_readiness(
                ref, source_ids=(operation.workspace_source_id,)
            )
            if len(rows) != 1:
                raise ValueError("readiness source count mismatch")
            readiness = rows[0]
            if (
                readiness.ref != ref
                or readiness.source_id != operation.workspace_source_id
                or readiness.catalog_item_id != operation.canonical_item_id
            ):
                raise ValueError("readiness source identity mismatch")
        except Exception:
            return await self._fail(
                operation,
                error_code="readiness_refresh_failed",
                error_message="Source readiness could not be refreshed.",
            )
        if readiness.state in {
            SourceReadinessState.ATTACHED,
            SourceReadinessState.PARSING,
            SourceReadinessState.INDEXING,
            SourceReadinessState.STALE,
        }:
            return operation
        if readiness.state in {
            SourceReadinessState.FAILED,
            SourceReadinessState.UNAVAILABLE,
        }:
            code = (
                "source_readiness_failed"
                if readiness.state is SourceReadinessState.FAILED
                else "source_unavailable"
            )
            return await self._fail(
                operation,
                error_code=code,
                error_message="Source is not ready for grounded retrieval.",
            )
        if not readiness.fts_ready:
            return operation
        operation = await self._mark_in_progress(operation)
        return await asyncio.to_thread(
            self._operation_store.advance_stage,
            operation.operation_id,
            stage=SourceOperationStage.READINESS,
            status=SourceOperationStatus.SUCCEEDED,
            expected_revision=operation.revision,
        )

    async def retry(self, operation_id: str) -> ResearchSourceOperation:
        """Clear only a failed readiness receipt, then refresh its association."""

        operation = await self._require_operation(operation_id)
        operation = await asyncio.to_thread(
            self._operation_store.retry_failed_stage,
            operation.operation_id,
            stage=SourceOperationStage.READINESS,
            expected_revision=operation.revision,
        )
        return await self.resume(operation.operation_id)

    async def resume_incomplete(self, *, limit: int = 50) -> None:
        operations = await asyncio.to_thread(
            self._operation_store.list_readiness_actionable,
            limit=limit,
        )
        await asyncio.gather(
            *(self.resume(operation.operation_id) for operation in operations),
            return_exceptions=True,
        )

    async def _require_operation(self, operation_id: str) -> ResearchSourceOperation:
        operation = await asyncio.to_thread(self._operation_store.get, operation_id)
        if operation is None:
            raise ValueError("Research source operation does not exist.")
        return operation

    async def _mark_in_progress(
        self, operation: ResearchSourceOperation
    ) -> ResearchSourceOperation:
        if operation.readiness_status is SourceOperationStatus.IN_PROGRESS:
            return operation
        return await asyncio.to_thread(
            self._operation_store.advance_stage,
            operation.operation_id,
            stage=SourceOperationStage.READINESS,
            status=SourceOperationStatus.IN_PROGRESS,
            expected_revision=operation.revision,
        )

    async def _fail(
        self,
        operation: ResearchSourceOperation,
        *,
        error_code: str,
        error_message: str,
    ) -> ResearchSourceOperation:
        operation = await self._mark_in_progress(operation)
        return await asyncio.to_thread(
            self._operation_store.advance_stage,
            operation.operation_id,
            stage=SourceOperationStage.READINESS,
            status=SourceOperationStatus.FAILED,
            expected_revision=operation.revision,
            error_code=error_code,
            error_message=error_message,
        )

    @staticmethod
    def _operation_ref(operation: ResearchSourceOperation) -> QualifiedWorkspaceRef:
        return QualifiedWorkspaceRef(
            data_source=operation.data_source,
            workspace_id=operation.workspace_id,
            server_profile_id=operation.server_profile_id,
            principal_id=operation.principal_id,
        )
