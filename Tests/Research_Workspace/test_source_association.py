"""Research catalog-to-workspace association contracts."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem
from tldw_chatbook.Library.library_ingest_jobs import (
    IngestJobState,
    LibraryIngestJobRegistry,
    _job_from_row,
)
from tldw_chatbook.Research_Workspace.contracts import WorkspaceDataSource
from tldw_chatbook.Research_Workspace.source_association import (
    ResearchSourceAssociationCoordinator,
    ResearchSourceAssociationScheduler,
)
from tldw_chatbook.Research_Workspace.source_operation_store import (
    ResearchSourceOperationStore,
)
from tldw_chatbook.Research_Workspace.source_operations import (
    CanonicalItemType,
    ResearchSourceOperation,
    SourceOperationStage,
    SourceOperationStatus,
)
from tldw_chatbook.runtime_policy.server_event_scope import (
    event_principal_id_from_active_context,
)
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _operation(
    *,
    operation_id: str,
    data_source: WorkspaceDataSource,
    workspace_id: str,
    server_profile_id: str = "",
    principal_id: str = "",
    desired_selected: bool = True,
) -> ResearchSourceOperation:
    timestamp = _timestamp()
    return ResearchSourceOperation(
        operation_id=operation_id,
        idempotency_key=f"idempotency:{operation_id}",
        data_source=data_source,
        server_profile_id=server_profile_id,
        principal_id=principal_id,
        workspace_id=workspace_id,
        canonical_item_type=(
            CanonicalItemType.LOCAL_LIBRARY
            if data_source is WorkspaceDataSource.LOCAL
            else CanonicalItemType.SERVER_MEDIA
        ),
        desired_selected=desired_selected,
        created_at=timestamp,
        updated_at=timestamp,
    )


def _operation_store(tmp_path: Path) -> ResearchSourceOperationStore:
    return ResearchSourceOperationStore(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="association-test")
    )


@pytest.mark.asyncio
async def test_local_done_links_canonical_media_to_captured_workspace(
    tmp_path: Path,
) -> None:
    store = _operation_store(tmp_path)
    local_registry = LocalWorkspaceRegistryService(store._db)
    local_registry.create_workspace(workspace_id="ws-captured", name="Captured")
    local_registry.create_workspace(workspace_id="ws-visible", name="Visible Later")
    local_registry.set_active_workspace("ws-visible")
    jobs = LibraryIngestJobRegistry()
    operation = store.create(
        _operation(
            operation_id="research-op-local",
            data_source=WorkspaceDataSource.LOCAL,
            workspace_id="ws-captured",
        )
    )
    job = jobs.submit(
        source_path="source.txt",
        title="Source title",
        detected_type="document",
        research_source_operation_id=operation.operation_id,
    )
    jobs.mark_parsing(job.job_id)
    jobs.mark_writing(job.job_id)
    jobs.mark_done(job.job_id, media_id=41)
    store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.IN_PROGRESS,
        expected_revision=operation.revision,
        ingest_job_id=job.job_id,
    )

    coordinator = ResearchSourceAssociationCoordinator(
        operation_store=store,
        ingest_jobs=jobs,
        local_registry=local_registry,
    )
    receipt = await coordinator.resume(operation.operation_id)

    assert receipt.catalog_status is SourceOperationStatus.SUCCEEDED
    assert receipt.canonical_item_id == "41"
    assert receipt.association_status is SourceOperationStatus.SUCCEEDED
    assert receipt.readiness_status is SourceOperationStatus.PENDING
    assert receipt.workspace_source_id
    memberships = local_registry.get_item_memberships("media", "41")
    assert [(item.workspace_id, item.role) for item in memberships] == [
        ("ws-captured", "source")
    ]
    assert local_registry.get_workspace_scope("ws-captured") == RagScope(
        items=(ScopeItem("media", "41"),),
        updated_at=local_registry.get_workspace_scope("ws-captured").updated_at,
        empty_is_scoped=True,
    )


@pytest.mark.asyncio
async def test_local_unselected_association_persists_explicit_empty_desired_scope(
    tmp_path: Path,
) -> None:
    store = _operation_store(tmp_path)
    local_registry = LocalWorkspaceRegistryService(store._db)
    local_registry.create_workspace(workspace_id="ws-a", name="Workspace A")
    operation = store.create(
        _operation(
            operation_id="research-op-unselected",
            data_source=WorkspaceDataSource.LOCAL,
            workspace_id="ws-a",
            desired_selected=False,
        )
    )
    operation = store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.SUCCEEDED,
        expected_revision=operation.revision,
        canonical_item_id="42",
    )

    receipt = await ResearchSourceAssociationCoordinator(
        operation_store=store,
        ingest_jobs=LibraryIngestJobRegistry(),
        local_registry=local_registry,
    ).resume(operation.operation_id)

    assert receipt.association_status is SourceOperationStatus.SUCCEEDED
    assert local_registry.get_workspace_scope("ws-a") == RagScope(
        items=(),
        updated_at=local_registry.get_workspace_scope("ws-a").updated_at,
        empty_is_scoped=True,
    )


@pytest.mark.asyncio
async def test_local_duplicate_reuses_existing_membership(tmp_path: Path) -> None:
    store = _operation_store(tmp_path)
    local_registry = LocalWorkspaceRegistryService(store._db)
    local_registry.create_workspace(workspace_id="ws-a", name="Workspace A")
    existing = local_registry.link_membership(
        "ws-a", item_type="media", item_id="41", role="source"
    )
    jobs = LibraryIngestJobRegistry()
    operation = store.create(
        _operation(
            operation_id="research-op-duplicate",
            data_source=WorkspaceDataSource.LOCAL,
            workspace_id="ws-a",
        )
    )
    job = jobs.submit(
        source_path="duplicate.txt",
        research_source_operation_id=operation.operation_id,
    )
    jobs.mark_done(job.job_id, media_id=41, progress={"outcome": "matched"})
    store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.IN_PROGRESS,
        expected_revision=operation.revision,
        ingest_job_id=job.job_id,
    )

    receipt = await ResearchSourceAssociationCoordinator(
        operation_store=store,
        ingest_jobs=jobs,
        local_registry=local_registry,
    ).resume(operation.operation_id)

    assert receipt.workspace_source_id == existing.membership_id
    assert len(local_registry.get_item_memberships("media", "41")) == 1


@pytest.mark.asyncio
async def test_association_failure_preserves_catalog_and_retry_does_not_reingest(
    tmp_path: Path,
) -> None:
    store = _operation_store(tmp_path)
    local_registry = LocalWorkspaceRegistryService(store._db)
    local_registry.create_workspace(workspace_id="ws-a", name="Workspace A")
    jobs = LibraryIngestJobRegistry()
    operation = store.create(
        _operation(
            operation_id="research-op-association-retry",
            data_source=WorkspaceDataSource.LOCAL,
            workspace_id="ws-a",
        )
    )
    job = jobs.submit(
        source_path="source.txt",
        research_source_operation_id=operation.operation_id,
    )
    jobs.mark_done(job.job_id, media_id=51)
    store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.IN_PROGRESS,
        expected_revision=operation.revision,
        ingest_job_id=job.job_id,
    )

    class FailOnceRegistry:
        def __init__(self) -> None:
            self.calls = 0

        def link_membership(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("secret=/private/user/source.txt")
            return local_registry.link_membership(*args, **kwargs)

    flaky_registry = FailOnceRegistry()
    catalog_retry_calls: list[str] = []
    coordinator = ResearchSourceAssociationCoordinator(
        operation_store=store,
        ingest_jobs=jobs,
        local_registry=flaky_registry,
        catalog_requeuer=lambda job_id: catalog_retry_calls.append(job_id),
        catalog_dispatcher=lambda job_id: catalog_retry_calls.append(job_id),
    )

    failed = await coordinator.resume(operation.operation_id)
    retried = await coordinator.retry(
        operation.operation_id, stage=SourceOperationStage.ASSOCIATION
    )

    assert failed.catalog_status is SourceOperationStatus.SUCCEEDED
    assert failed.canonical_item_id == "51"
    assert failed.association_status is SourceOperationStatus.FAILED
    assert (
        failed.error_message == "Catalog item saved, but workspace association failed."
    )
    assert retried.association_status is SourceOperationStatus.SUCCEEDED
    assert retried.canonical_item_id == "51"
    assert catalog_retry_calls == []
    assert flaky_registry.calls == 2


@pytest.mark.asyncio
async def test_server_done_uses_remote_media_and_durable_server_target(
    tmp_path: Path,
) -> None:
    store = _operation_store(tmp_path)
    jobs = LibraryIngestJobRegistry()
    context = SimpleNamespace(
        active_server_id="server-profile-a",
        auth_token="fixture-token-a",
        credential_source="fixture",
    )
    principal_id = event_principal_id_from_active_context(context) or ""
    operation = store.create(
        _operation(
            operation_id="research-op-server",
            data_source=WorkspaceDataSource.SERVER,
            workspace_id="remote-workspace-7",
            server_profile_id="server-profile-a",
            principal_id=principal_id,
        )
    )
    job = jobs.submit(
        source_path="paper.pdf",
        title="Paper",
        detected_type="pdf",
        origin="server",
        research_source_operation_id=operation.operation_id,
    )
    jobs.mark_remote_done(job.job_id, remote_media_id="884")
    store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.IN_PROGRESS,
        expected_revision=operation.revision,
        ingest_job_id=job.job_id,
    )

    class ServerService:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def save_workspace_source(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "id": kwargs["source_id"],
                "workspace_id": kwargs["workspace_id"],
                "media_id": kwargs["media_id"],
                "title": kwargs["title"],
                "source_type": kwargs["source_type"],
                "version": 1,
            }

    class NoLocalRegistry:
        def link_membership(self, *args, **kwargs):
            raise AssertionError("Server association must not call Local membership.")

    server_service = ServerService()
    coordinator = ResearchSourceAssociationCoordinator(
        operation_store=store,
        ingest_jobs=jobs,
        local_registry=NoLocalRegistry(),
        server_service=server_service,
        server_context_provider=SimpleNamespace(
            get_active_context=lambda: context,
        ),
    )

    receipt = await coordinator.resume(operation.operation_id)
    replayed = await coordinator.resume(operation.operation_id)

    assert receipt.catalog_status is SourceOperationStatus.SUCCEEDED
    assert receipt.canonical_item_id == "884"
    assert receipt.association_status is SourceOperationStatus.SUCCEEDED
    assert receipt.workspace_source_id.startswith("research-source-")
    assert replayed == receipt
    assert server_service.calls == [
        {
            "workspace_id": "remote-workspace-7",
            "source_id": receipt.workspace_source_id,
            "media_id": 884,
            "title": "Paper",
            "source_type": "pdf",
            "selected": True,
            "version": None,
        }
    ]


@pytest.mark.asyncio
async def test_server_context_switch_fails_closed_before_dynamic_service_call(
    tmp_path: Path,
) -> None:
    store = _operation_store(tmp_path)
    jobs = LibraryIngestJobRegistry()
    submitted_context = SimpleNamespace(
        active_server_id="server-profile-a",
        auth_token="fixture-token-a",
        credential_source="fixture",
    )
    operation = store.create(
        _operation(
            operation_id="research-op-context-switch",
            data_source=WorkspaceDataSource.SERVER,
            workspace_id="remote-workspace-7",
            server_profile_id="server-profile-a",
            principal_id=(
                event_principal_id_from_active_context(submitted_context) or ""
            ),
        )
    )
    job = jobs.submit(
        source_path="paper.pdf",
        origin="server",
        research_source_operation_id=operation.operation_id,
    )
    jobs.mark_remote_done(job.job_id, remote_media_id="884")
    store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.IN_PROGRESS,
        expected_revision=operation.revision,
        ingest_job_id=job.job_id,
    )
    service_calls: list[dict[str, object]] = []

    class ServerService:
        async def save_workspace_source(self, **kwargs):
            service_calls.append(kwargs)
            raise AssertionError("Changed context must be rejected before this call.")

    changed_context = SimpleNamespace(
        active_server_id="server-profile-b",
        auth_token="fixture-token-b",
        credential_source="fixture",
    )
    receipt = await ResearchSourceAssociationCoordinator(
        operation_store=store,
        ingest_jobs=jobs,
        server_service=ServerService(),
        server_context_provider=SimpleNamespace(
            get_active_context=lambda: changed_context,
        ),
    ).resume(operation.operation_id)

    assert receipt.catalog_status is SourceOperationStatus.SUCCEEDED
    assert receipt.association_status is SourceOperationStatus.FAILED
    assert receipt.error_code == "server_context_changed"
    assert service_calls == []


@pytest.mark.asyncio
async def test_server_done_without_remote_media_id_fails_catalog(
    tmp_path: Path,
) -> None:
    store = _operation_store(tmp_path)
    jobs = LibraryIngestJobRegistry()
    context = SimpleNamespace(
        active_server_id="server-profile-a",
        auth_token="fixture-token-a",
        credential_source="fixture",
    )
    operation = store.create(
        _operation(
            operation_id="research-op-server-missing-id",
            data_source=WorkspaceDataSource.SERVER,
            workspace_id="remote-workspace-7",
            server_profile_id="server-profile-a",
            principal_id=event_principal_id_from_active_context(context) or "",
        )
    )
    job = jobs.submit(
        source_path="https://example.test/article",
        origin="server",
        research_source_operation_id=operation.operation_id,
    )
    jobs.mark_remote_done(job.job_id)
    store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.IN_PROGRESS,
        expected_revision=operation.revision,
        ingest_job_id=job.job_id,
    )

    receipt = await ResearchSourceAssociationCoordinator(
        operation_store=store,
        ingest_jobs=jobs,
        server_service=SimpleNamespace(
            save_workspace_source=lambda **kwargs: pytest.fail(
                "Missing canonical id must not associate."
            )
        ),
        server_context_provider=SimpleNamespace(
            get_active_context=lambda: context,
        ),
    ).resume(operation.operation_id)

    assert receipt.catalog_status is SourceOperationStatus.FAILED
    assert receipt.error_code == "missing_canonical_item"
    assert receipt.association_status is SourceOperationStatus.PENDING


@pytest.mark.asyncio
async def test_catalog_rejects_job_linked_to_another_operation(tmp_path: Path) -> None:
    store = _operation_store(tmp_path)
    local_registry = LocalWorkspaceRegistryService(store._db)
    local_registry.create_workspace(workspace_id="ws-a", name="Workspace A")
    jobs = LibraryIngestJobRegistry()
    operation = store.create(
        _operation(
            operation_id="research-op-intended",
            data_source=WorkspaceDataSource.LOCAL,
            workspace_id="ws-a",
        )
    )
    job = jobs.submit(
        source_path="other.txt",
        research_source_operation_id="research-op-other",
    )
    jobs.mark_done(job.job_id, media_id=61)
    store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.IN_PROGRESS,
        expected_revision=operation.revision,
        ingest_job_id=job.job_id,
    )

    receipt = await ResearchSourceAssociationCoordinator(
        operation_store=store,
        ingest_jobs=jobs,
        local_registry=local_registry,
    ).resume(operation.operation_id)

    assert receipt.catalog_status is SourceOperationStatus.FAILED
    assert receipt.error_code == "ingest_job_mismatch"
    assert local_registry.get_item_memberships("media", "61") == ()


@pytest.mark.asyncio
async def test_catalog_rejects_job_from_other_authority(tmp_path: Path) -> None:
    store = _operation_store(tmp_path)
    local_registry = LocalWorkspaceRegistryService(store._db)
    local_registry.create_workspace(workspace_id="ws-a", name="Workspace A")
    jobs = LibraryIngestJobRegistry()
    operation = store.create(
        _operation(
            operation_id="research-op-local-origin",
            data_source=WorkspaceDataSource.LOCAL,
            workspace_id="ws-a",
        )
    )
    job = jobs.submit(
        source_path="remote.pdf",
        origin="server",
        research_source_operation_id=operation.operation_id,
    )
    jobs.mark_remote_done(job.job_id, remote_media_id="91")
    store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.IN_PROGRESS,
        expected_revision=operation.revision,
        ingest_job_id=job.job_id,
    )

    receipt = await ResearchSourceAssociationCoordinator(
        operation_store=store,
        ingest_jobs=jobs,
        local_registry=local_registry,
    ).resume(operation.operation_id)

    assert receipt.catalog_status is SourceOperationStatus.FAILED
    assert receipt.error_code == "ingest_job_mismatch"
    assert local_registry.get_item_memberships("media", "91") == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("media_id", [0, -1])
async def test_local_catalog_requires_positive_media_id(
    tmp_path: Path, media_id: int
) -> None:
    store = _operation_store(tmp_path)
    local_registry = LocalWorkspaceRegistryService(store._db)
    local_registry.create_workspace(workspace_id="ws-a", name="Workspace A")
    jobs = LibraryIngestJobRegistry()
    operation = store.create(
        _operation(
            operation_id=f"research-op-local-invalid-{media_id}",
            data_source=WorkspaceDataSource.LOCAL,
            workspace_id="ws-a",
        )
    )
    job = jobs.submit(
        source_path="source.txt",
        research_source_operation_id=operation.operation_id,
    )
    jobs.mark_done(job.job_id, media_id=media_id)
    store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.IN_PROGRESS,
        expected_revision=operation.revision,
        ingest_job_id=job.job_id,
    )

    receipt = await ResearchSourceAssociationCoordinator(
        operation_store=store,
        ingest_jobs=jobs,
        local_registry=local_registry,
    ).resume(operation.operation_id)

    assert receipt.catalog_status is SourceOperationStatus.FAILED
    assert receipt.error_code == "invalid_canonical_item"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_state", [IngestJobState.FAILED, IngestJobState.CANCELLED]
)
async def test_terminal_unsuccessful_job_advances_catalog_failure(
    tmp_path: Path, terminal_state: IngestJobState
) -> None:
    store = _operation_store(tmp_path)
    jobs = LibraryIngestJobRegistry()
    operation = store.create(
        _operation(
            operation_id=f"research-op-{terminal_state.value}",
            data_source=WorkspaceDataSource.LOCAL,
            workspace_id="ws-a",
        )
    )
    job = jobs.submit(
        source_path="/private/user/secret.txt",
        research_source_operation_id=operation.operation_id,
    )
    if terminal_state is IngestJobState.FAILED:
        jobs.mark_failed(
            job.job_id,
            error="Bearer secret-token at /private/user/secret.txt",
        )
    else:
        jobs.mark_cancelled(job.job_id, reason="Cancelled /private/user/secret.txt")
    store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.IN_PROGRESS,
        expected_revision=operation.revision,
        ingest_job_id=job.job_id,
    )

    receipt = await ResearchSourceAssociationCoordinator(
        operation_store=store,
        ingest_jobs=jobs,
    ).resume(operation.operation_id)

    assert receipt.catalog_status is SourceOperationStatus.FAILED
    assert receipt.error_code == "catalog_ingest_failed"
    assert receipt.error_message == "Catalog ingest did not complete successfully."


@pytest.mark.asyncio
async def test_catalog_retry_requeues_linked_job_and_records_new_lineage(
    tmp_path: Path,
) -> None:
    store = _operation_store(tmp_path)
    local_registry = LocalWorkspaceRegistryService(store._db)
    local_registry.create_workspace(workspace_id="ws-a", name="Workspace A")
    jobs = LibraryIngestJobRegistry()
    operation = store.create(
        _operation(
            operation_id="research-op-catalog-retry",
            data_source=WorkspaceDataSource.LOCAL,
            workspace_id="ws-a",
        )
    )
    failed_job = jobs.submit(
        source_path="source.txt",
        research_source_operation_id=operation.operation_id,
    )
    jobs.mark_failed(failed_job.job_id, error="Temporary catalog failure")
    in_progress = store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.IN_PROGRESS,
        expected_revision=operation.revision,
        ingest_job_id=failed_job.job_id,
    )
    failed = store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.FAILED,
        expected_revision=in_progress.revision,
        error_code="catalog_ingest_failed",
        error_message="Catalog ingest did not complete successfully.",
    )
    retry_calls: list[str] = []

    def retry_catalog(job_id: str):
        retry_calls.append(job_id)
        return jobs.requeue(job_id)

    coordinator = ResearchSourceAssociationCoordinator(
        operation_store=store,
        ingest_jobs=jobs,
        local_registry=local_registry,
        catalog_requeuer=retry_catalog,
        catalog_dispatcher=lambda job_id: None,
    )

    retrying = await coordinator.retry(
        failed.operation_id, stage=SourceOperationStage.CATALOG
    )
    assert retrying.catalog_status is SourceOperationStatus.IN_PROGRESS
    assert retrying.ingest_job_id != failed_job.job_id
    assert retry_calls == [failed_job.job_id]
    retry_job = jobs.get_job(retrying.ingest_job_id)
    assert retry_job is not None
    assert retry_job.research_source_operation_id == failed.operation_id

    jobs.mark_done(retry_job.job_id, media_id=71)
    receipt = await coordinator.resume(failed.operation_id)
    assert receipt.catalog_status is SourceOperationStatus.SUCCEEDED
    assert receipt.canonical_item_id == "71"
    assert receipt.association_status is SourceOperationStatus.SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["local", "server"])
async def test_catalog_retry_records_replacement_before_immediate_dispatch_failure(
    tmp_path: Path,
    origin: str,
) -> None:
    workspace_db = WorkspaceDB(
        tmp_path / f"workspaces-{origin}.sqlite",
        client_id=f"retry-{origin}",
    )
    store = ResearchSourceOperationStore(workspace_db)
    ingest_db = LibraryIngestJobsDB(tmp_path / f"ingest-{origin}.sqlite")
    jobs = LibraryIngestJobRegistry()
    jobs.attach_store(ingest_db)
    authority = (
        WorkspaceDataSource.LOCAL if origin == "local" else WorkspaceDataSource.SERVER
    )
    operation = store.create(
        _operation(
            operation_id=f"research-op-immediate-{origin}",
            data_source=authority,
            workspace_id="ws-a",
            server_profile_id="server-a" if origin == "server" else "",
            principal_id="principal-a" if origin == "server" else "",
        )
    )
    failed_job = jobs.submit(
        source_path=(
            "https://example.test/source.pdf" if origin == "server" else "source.txt"
        ),
        origin=origin,
        research_source_operation_id=operation.operation_id,
    )
    jobs.mark_failed(failed_job.job_id, error="initial failure")
    operation = store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.IN_PROGRESS,
        expected_revision=operation.revision,
        ingest_job_id=failed_job.job_id,
    )
    operation = store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.FAILED,
        expected_revision=operation.revision,
        error_code="catalog_ingest_failed",
        error_message="Catalog ingest did not complete successfully.",
    )
    dispatch_observations: list[tuple[str, str]] = []

    def dispatch(job_id: str) -> None:
        durable = store.get(operation.operation_id)
        assert durable is not None
        persisted_job = next(
            row for row in ingest_db.all_jobs() if row["job_id"] == job_id
        )
        assert persisted_job["state"] == "queued"
        dispatch_observations.append((durable.ingest_job_id, job_id))
        jobs.mark_failed(job_id, error="immediate dispatch failure")

    coordinator = ResearchSourceAssociationCoordinator(
        operation_store=store,
        ingest_jobs=jobs,
        catalog_requeuer=jobs.requeue,
        catalog_dispatcher=dispatch,
    )
    scheduler = ResearchSourceAssociationScheduler(
        coordinator=coordinator,
        operation_store=store,
    )

    receipt = await scheduler.retry(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
    )

    assert receipt.catalog_status is SourceOperationStatus.FAILED
    assert receipt.error_code == "catalog_ingest_failed"
    assert receipt.ingest_job_id != failed_job.job_id
    assert dispatch_observations == [(receipt.ingest_job_id, receipt.ingest_job_id)]
    assert store.get(operation.operation_id) == receipt

    rows = ingest_db.all_jobs()
    restored_jobs = LibraryIngestJobRegistry()
    restored_jobs.restore(
        [_job_from_row(row) for row in rows],
        next_id=max(row["seq"] for row in rows) + 1,
    )
    restored_jobs.attach_store(ingest_db)
    replay = await ResearchSourceAssociationScheduler(
        coordinator=ResearchSourceAssociationCoordinator(
            operation_store=store,
            ingest_jobs=restored_jobs,
        ),
        operation_store=store,
    ).resume(operation.operation_id)

    assert replay == receipt
    ingest_db.close()
    workspace_db.close()


@pytest.mark.asyncio
async def test_retry_rejects_readiness_without_mutating_receipt(tmp_path: Path) -> None:
    store = _operation_store(tmp_path)
    operation = store.create(
        _operation(
            operation_id="research-op-readiness-retry",
            data_source=WorkspaceDataSource.LOCAL,
            workspace_id="ws-a",
        )
    )
    catalog = store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.SUCCEEDED,
        expected_revision=operation.revision,
        canonical_item_id="41",
    )
    association = store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.ASSOCIATION,
        status=SourceOperationStatus.SUCCEEDED,
        expected_revision=catalog.revision,
        workspace_source_id="membership-41",
    )
    failed = store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.READINESS,
        status=SourceOperationStatus.FAILED,
        expected_revision=association.revision,
        error_code="readiness_failed",
        error_message="Readiness failed.",
    )
    coordinator = ResearchSourceAssociationCoordinator(
        operation_store=store,
        ingest_jobs=LibraryIngestJobRegistry(),
    )

    with pytest.raises(ValueError, match="catalog or association"):
        await coordinator.retry(
            failed.operation_id,
            stage=SourceOperationStage.READINESS,
        )

    assert store.get(failed.operation_id) == failed


@pytest.mark.asyncio
async def test_coordinator_db_read_does_not_block_event_loop() -> None:
    operation = _operation(
        operation_id="research-op-responsive",
        data_source=WorkspaceDataSource.LOCAL,
        workspace_id="ws-a",
    )

    class SlowStore:
        def get(self, operation_id: str) -> ResearchSourceOperation:
            assert operation_id == operation.operation_id
            time.sleep(0.1)
            return operation

    coordinator = ResearchSourceAssociationCoordinator(
        operation_store=SlowStore(),
        ingest_jobs=LibraryIngestJobRegistry(),
    )
    order: list[str] = []

    async def resume() -> None:
        await coordinator.resume(operation.operation_id)
        order.append("resume")

    async def pulse() -> None:
        await asyncio.sleep(0.01)
        order.append("pulse")

    await asyncio.gather(resume(), pulse())

    assert order == ["pulse", "resume"]


@pytest.mark.asyncio
async def test_scheduler_serializes_same_operation_but_allows_unrelated_work() -> None:
    entered_a = asyncio.Event()
    entered_b = asyncio.Event()
    release_a = asyncio.Event()
    calls: list[str] = []
    active_a = 0
    max_active_a = 0

    class Coordinator:
        async def resume(self, operation_id: str) -> None:
            nonlocal active_a, max_active_a
            calls.append(operation_id)
            if operation_id == "op-a":
                active_a += 1
                max_active_a = max(max_active_a, active_a)
                entered_a.set()
                await release_a.wait()
                active_a -= 1
            else:
                entered_b.set()

    scheduler = ResearchSourceAssociationScheduler(
        coordinator=Coordinator(),
        operation_store=SimpleNamespace(
            list_association_actionable=lambda **kwargs: ()
        ),
    )
    first_a = asyncio.create_task(scheduler.resume("op-a"))
    await entered_a.wait()
    second_a = asyncio.create_task(scheduler.resume("op-a"))
    unrelated = asyncio.create_task(scheduler.resume("op-b"))
    await asyncio.wait_for(entered_b.wait(), timeout=0.2)

    assert calls == ["op-a", "op-b"]
    release_a.set()
    await asyncio.gather(first_a, second_a, unrelated)
    assert calls == ["op-a", "op-b", "op-a"]
    assert max_active_a == 1
    assert scheduler.active_fence_count == 0


@pytest.mark.asyncio
async def test_scheduler_serializes_retry_and_resume_per_operation() -> None:
    retry_entered = asyncio.Event()
    unrelated_entered = asyncio.Event()
    release_retry = asyncio.Event()
    calls: list[tuple[str, str]] = []

    class Coordinator:
        async def retry(self, operation_id: str, *, stage: SourceOperationStage):
            calls.append(("retry", operation_id))
            if operation_id == "op-a":
                retry_entered.set()
                await release_retry.wait()

        async def resume(self, operation_id: str):
            calls.append(("resume", operation_id))
            if operation_id == "op-b":
                unrelated_entered.set()

    scheduler = ResearchSourceAssociationScheduler(
        coordinator=Coordinator(),
        operation_store=SimpleNamespace(
            list_association_actionable=lambda **kwargs: ()
        ),
    )
    retry_a = asyncio.create_task(
        scheduler.retry("op-a", stage=SourceOperationStage.CATALOG)
    )
    await retry_entered.wait()
    resume_a = asyncio.create_task(scheduler.resume("op-a"))
    resume_b = asyncio.create_task(scheduler.resume("op-b"))
    await asyncio.wait_for(unrelated_entered.wait(), timeout=0.2)

    assert calls == [("retry", "op-a"), ("resume", "op-b")]
    release_retry.set()
    await asyncio.gather(retry_a, resume_a, resume_b)
    assert calls == [
        ("retry", "op-a"),
        ("resume", "op-b"),
        ("resume", "op-a"),
    ]
    assert scheduler.active_fence_count == 0


@pytest.mark.asyncio
async def test_startup_resume_is_bounded_and_skips_failed_receipts() -> None:
    pending = _operation(
        operation_id="op-pending",
        data_source=WorkspaceDataSource.LOCAL,
        workspace_id="ws-a",
    )
    failed = replace(
        _operation(
            operation_id="op-failed",
            data_source=WorkspaceDataSource.LOCAL,
            workspace_id="ws-a",
        ),
        catalog_status=SourceOperationStatus.FAILED,
        error_stage=SourceOperationStage.CATALOG,
        error_code="catalog_failed",
        error_message="Catalog failed.",
    )
    list_calls: list[dict[str, int]] = []
    resumed: list[str] = []

    class Store:
        def list_association_actionable(self, **kwargs):
            list_calls.append(kwargs)
            return (pending, failed)

    class Coordinator:
        async def resume(self, operation_id: str) -> None:
            resumed.append(operation_id)

    scheduler = ResearchSourceAssociationScheduler(
        coordinator=Coordinator(),
        operation_store=Store(),
    )

    await scheduler.resume_incomplete(limit=25)

    assert list_calls == [{"limit": 25}]
    assert resumed == [pending.operation_id]


@pytest.mark.asyncio
async def test_startup_resume_isolates_one_operation_failure() -> None:
    first = _operation(
        operation_id="op-corrupt",
        data_source=WorkspaceDataSource.LOCAL,
        workspace_id="ws-a",
    )
    second = _operation(
        operation_id="op-actionable",
        data_source=WorkspaceDataSource.LOCAL,
        workspace_id="ws-a",
    )
    resumed: list[str] = []

    class Coordinator:
        async def resume(self, operation_id: str) -> None:
            resumed.append(operation_id)
            if operation_id == first.operation_id:
                raise RuntimeError("one bad receipt")

    scheduler = ResearchSourceAssociationScheduler(
        coordinator=Coordinator(),
        operation_store=SimpleNamespace(
            list_association_actionable=lambda **kwargs: (first, second)
        ),
    )

    await scheduler.resume_incomplete()

    assert set(resumed) == {first.operation_id, second.operation_id}
    assert scheduler.active_fence_count == 0
