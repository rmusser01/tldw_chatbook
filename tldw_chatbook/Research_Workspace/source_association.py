"""Durable catalog-to-workspace association coordination."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

from tldw_chatbook.Library.library_ingest_jobs import (
    IngestJobState,
    LibraryIngestJobRegistry,
)
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
from tldw_chatbook.runtime_policy.server_event_scope import (
    event_principal_id_from_active_context,
)

from .contracts import WorkspaceDataSource
from .source_operation_store import ResearchSourceOperationStore
from .source_operations import (
    ResearchSourceOperation,
    SourceOperationStage,
    SourceOperationStatus,
)


class ResearchSourceAssociationCoordinator:
    """Resume source association strictly from durable operation intent."""

    def __init__(
        self,
        *,
        operation_store: ResearchSourceOperationStore,
        ingest_jobs: LibraryIngestJobRegistry,
        local_registry: LocalWorkspaceRegistryService | None = None,
        server_service: Any | None = None,
        server_context_provider: Any | None = None,
        catalog_requeuer: Any | None = None,
        catalog_dispatcher: Any | None = None,
    ) -> None:
        self._operation_store = operation_store
        self._ingest_jobs = ingest_jobs
        self._local_registry = local_registry
        self._server_service = server_service
        self._server_context_provider = server_context_provider
        self._catalog_requeuer = catalog_requeuer
        self._catalog_dispatcher = catalog_dispatcher

    async def resume(self, operation_id: str) -> ResearchSourceOperation:
        """Resume catalog and association stages for one durable operation."""

        operation = await self._require_operation(operation_id)
        operation = await self._resume_catalog(operation)
        if operation.catalog_status is not SourceOperationStatus.SUCCEEDED:
            return operation
        if operation.association_status is SourceOperationStatus.SUCCEEDED:
            return operation
        if operation.association_status is SourceOperationStatus.FAILED:
            return operation

        operation = await self._mark_association_in_progress(operation)
        if operation.data_source is WorkspaceDataSource.LOCAL:
            return await self._associate_local(operation)
        return await self._associate_server(operation)

    async def retry(
        self,
        operation_id: str,
        *,
        stage: SourceOperationStage,
    ) -> ResearchSourceOperation:
        """Explicitly retry only the named failed catalog or association stage."""

        if stage not in {
            SourceOperationStage.CATALOG,
            SourceOperationStage.ASSOCIATION,
        }:
            raise ValueError("Only catalog or association stages can be retried here.")
        operation = await self._require_operation(operation_id)
        operation = await asyncio.to_thread(
            self._operation_store.retry_failed_stage,
            operation.operation_id,
            stage=stage,
            expected_revision=operation.revision,
        )
        if stage is SourceOperationStage.ASSOCIATION:
            return await self.resume(operation.operation_id)
        if self._catalog_requeuer is None or self._catalog_dispatcher is None:
            return await self._catalog_failed(
                operation,
                error_code="catalog_retry_unavailable",
                error_message="Catalog retry is unavailable.",
            )
        try:
            retry_job = self._catalog_requeuer(operation.ingest_job_id)
        except Exception:
            retry_job = None
        expected_origin = (
            "local" if operation.data_source is WorkspaceDataSource.LOCAL else "server"
        )
        if (
            retry_job is None
            or retry_job.research_source_operation_id != operation.operation_id
            or retry_job.origin != expected_origin
        ):
            return await self._catalog_failed(
                operation,
                error_code="catalog_retry_failed",
                error_message="Catalog retry could not be started.",
            )
        operation = await self._advance_stage(
            operation.operation_id,
            stage=SourceOperationStage.CATALOG,
            status=SourceOperationStatus.IN_PROGRESS,
            expected_revision=operation.revision,
            ingest_job_id=retry_job.job_id,
        )
        try:
            self._catalog_dispatcher(retry_job.job_id)
        except Exception:
            return await self._catalog_failed(
                operation,
                error_code="catalog_retry_failed",
                error_message="Catalog retry could not be started.",
            )
        return await self.resume(operation.operation_id)

    async def _require_operation(self, operation_id: str) -> ResearchSourceOperation:
        operation = await asyncio.to_thread(self._operation_store.get, operation_id)
        if operation is None:
            raise ValueError("Research source operation does not exist.")
        return operation

    async def _resume_catalog(
        self, operation: ResearchSourceOperation
    ) -> ResearchSourceOperation:
        if operation.catalog_status in {
            SourceOperationStatus.SUCCEEDED,
            SourceOperationStatus.FAILED,
        }:
            return operation
        if not operation.ingest_job_id:
            return operation
        job = self._ingest_jobs.get_job(operation.ingest_job_id)
        expected_origin = (
            "local" if operation.data_source is WorkspaceDataSource.LOCAL else "server"
        )
        if (
            job is None
            or job.research_source_operation_id != operation.operation_id
            or job.origin != expected_origin
        ):
            return await self._catalog_failed(
                operation,
                error_code="ingest_job_mismatch",
                error_message="Linked catalog ingest job does not match this operation.",
            )
        if job.state in {
            IngestJobState.FAILED,
            IngestJobState.CANCELLED,
            IngestJobState.SKIPPED,
        }:
            return await self._catalog_failed(
                operation,
                error_code="catalog_ingest_failed",
                error_message="Catalog ingest did not complete successfully.",
            )
        if job.state is not IngestJobState.DONE:
            return operation
        if operation.data_source is WorkspaceDataSource.LOCAL:
            if (
                isinstance(job.media_id, bool)
                or not isinstance(job.media_id, int)
                or job.media_id < 1
            ):
                code = (
                    "missing_canonical_item"
                    if job.media_id is None
                    else "invalid_canonical_item"
                )
                return await self._catalog_failed(
                    operation,
                    error_code=code,
                    error_message="Catalog completion did not return a valid canonical item id.",
                )
            canonical_item_id = str(job.media_id)
        else:
            remote_media_id = str(job.remote_media_id or "")
            if not remote_media_id:
                return await self._catalog_failed(
                    operation,
                    error_code="missing_canonical_item",
                    error_message="Catalog completion did not return a canonical item id.",
                )
            if not remote_media_id.isdigit() or int(remote_media_id) < 1:
                return await self._catalog_failed(
                    operation,
                    error_code="invalid_canonical_item",
                    error_message="Catalog completion did not return a valid canonical item id.",
                )
            canonical_item_id = str(int(remote_media_id))
        return await self._advance_stage(
            operation.operation_id,
            stage=SourceOperationStage.CATALOG,
            status=SourceOperationStatus.SUCCEEDED,
            expected_revision=operation.revision,
            canonical_item_id=canonical_item_id,
        )

    async def _catalog_failed(
        self,
        operation: ResearchSourceOperation,
        *,
        error_code: str,
        error_message: str,
    ) -> ResearchSourceOperation:
        return await self._advance_stage(
            operation.operation_id,
            stage=SourceOperationStage.CATALOG,
            status=SourceOperationStatus.FAILED,
            expected_revision=operation.revision,
            error_code=error_code,
            error_message=error_message,
        )

    async def _mark_association_in_progress(
        self, operation: ResearchSourceOperation
    ) -> ResearchSourceOperation:
        if operation.association_status is SourceOperationStatus.IN_PROGRESS:
            return operation
        return await self._advance_stage(
            operation.operation_id,
            stage=SourceOperationStage.ASSOCIATION,
            status=SourceOperationStatus.IN_PROGRESS,
            expected_revision=operation.revision,
        )

    async def _associate_local(
        self, operation: ResearchSourceOperation
    ) -> ResearchSourceOperation:
        if self._local_registry is None:
            return await self._association_failed(operation)
        try:
            membership = await asyncio.to_thread(
                self._local_registry.link_membership,
                operation.workspace_id,
                item_type="media",
                item_id=operation.canonical_item_id,
                role="source",
            )
        except Exception:
            return await self._association_failed(operation)
        return await self._advance_stage(
            operation.operation_id,
            stage=SourceOperationStage.ASSOCIATION,
            status=SourceOperationStatus.SUCCEEDED,
            expected_revision=operation.revision,
            workspace_source_id=membership.membership_id,
        )

    async def _associate_server(
        self, operation: ResearchSourceOperation
    ) -> ResearchSourceOperation:
        if self._server_service is None or self._server_context_provider is None:
            return await self._association_failed(operation)
        try:
            context = self._server_context_provider.get_active_context()
            profile_id = str(getattr(context, "active_server_id", "") or "").strip()
            principal_id = event_principal_id_from_active_context(context) or ""
            if (
                profile_id != operation.server_profile_id
                or principal_id != operation.principal_id
            ):
                return await self._association_failed(
                    operation,
                    error_code="server_context_changed",
                )
            media_id = int(operation.canonical_item_id)
            if media_id < 1 or str(media_id) != operation.canonical_item_id:
                return await self._association_failed(
                    operation,
                    error_code="invalid_canonical_item",
                )
            job = self._ingest_jobs.get_job(operation.ingest_job_id)
            if job is None:
                return await self._association_failed(operation)
            source_id = _server_source_id(
                operation.idempotency_key,
                operation.canonical_item_id,
            )
            row = await self._server_service.save_workspace_source(
                workspace_id=operation.workspace_id,
                source_id=source_id,
                media_id=media_id,
                title=job.title or Path(job.source_path).stem or "Source",
                source_type=job.detected_type or "media",
                version=None,
            )
            if (
                str(row.get("id") or "") != source_id
                or str(row.get("workspace_id") or "") != operation.workspace_id
                or int(row.get("media_id")) != media_id
            ):
                return await self._association_failed(
                    operation,
                    error_code="server_association_mismatch",
                )
        except Exception:
            return await self._association_failed(operation)
        return await self._advance_stage(
            operation.operation_id,
            stage=SourceOperationStage.ASSOCIATION,
            status=SourceOperationStatus.SUCCEEDED,
            expected_revision=operation.revision,
            workspace_source_id=source_id,
        )

    async def _association_failed(
        self,
        operation: ResearchSourceOperation,
        *,
        error_code: str = "association_failed",
    ) -> ResearchSourceOperation:
        return await self._advance_stage(
            operation.operation_id,
            stage=SourceOperationStage.ASSOCIATION,
            status=SourceOperationStatus.FAILED,
            expected_revision=operation.revision,
            error_code=error_code,
            error_message="Catalog item saved, but workspace association failed.",
        )

    async def _advance_stage(
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
    ) -> ResearchSourceOperation:
        """Run a thread-safe SQLite transition away from the Textual loop."""

        return await asyncio.to_thread(
            self._operation_store.advance_stage,
            operation_id,
            stage=stage,
            status=status,
            expected_revision=expected_revision,
            ingest_job_id=ingest_job_id,
            canonical_item_id=canonical_item_id,
            workspace_source_id=workspace_source_id,
            error_code=error_code,
            error_message=error_message,
        )


class _OperationFence:
    """One keyed lock plus holders and waiters needed for ABA-safe cleanup."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.users = 0


class ResearchSourceAssociationScheduler:
    """Fence same-operation resumes while permitting unrelated operations."""

    def __init__(self, *, coordinator: Any, operation_store: Any) -> None:
        self._coordinator = coordinator
        self._operation_store = operation_store
        self._operation_fences: dict[str, _OperationFence] = {}

    @property
    def active_fence_count(self) -> int:
        """Return the number of operation keys with holders or waiters."""

        return len(self._operation_fences)

    async def resume(self, operation_id: str) -> ResearchSourceOperation | None:
        """Resume one operation behind its operation-specific lock."""

        return await self._run_fenced(
            operation_id,
            self._coordinator.resume,
        )

    async def retry(
        self,
        operation_id: str,
        *,
        stage: SourceOperationStage,
    ) -> ResearchSourceOperation | None:
        """Retry one operation behind the same operation-specific lock."""

        return await self._run_fenced(
            operation_id,
            self._coordinator.retry,
            stage=stage,
        )

    async def _run_fenced(
        self,
        operation_id: str,
        action: Any,
        **kwargs: Any,
    ) -> ResearchSourceOperation | None:
        """Run one coordinator action with ABA-safe keyed-lock cleanup."""

        fence = self._operation_fences.get(operation_id)
        if fence is None:
            fence = _OperationFence()
            self._operation_fences[operation_id] = fence
        fence.users += 1
        try:
            async with fence.lock:
                return await action(operation_id, **kwargs)
        finally:
            fence.users -= 1
            if fence.users == 0 and self._operation_fences.get(operation_id) is fence:
                del self._operation_fences[operation_id]

    async def resume_incomplete(self, *, limit: int = 50) -> None:
        """Resume a bounded startup page of catalog/association work."""

        operations = await asyncio.to_thread(
            self._operation_store.list_association_actionable,
            limit=limit,
        )
        actionable = (
            operation
            for operation in operations
            if SourceOperationStatus.FAILED
            not in {operation.catalog_status, operation.association_status}
        )
        await asyncio.gather(
            *(self.resume(operation.operation_id) for operation in actionable),
            return_exceptions=True,
        )


def _server_source_id(idempotency_key: str, media_id: str) -> str:
    """Return a collision-resistant deterministic server workspace-source id."""

    def frame(value: str) -> bytes:
        encoded = value.encode("utf-8")
        return len(encoded).to_bytes(4, "big") + encoded

    digest = hashlib.sha256(
        frame("research-workspace-source-v1") + frame(idempotency_key) + frame(media_id)
    ).hexdigest()
    return f"research-source-{digest}"
