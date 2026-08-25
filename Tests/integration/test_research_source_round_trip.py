"""Cross-owner round trips for Research Workspace source intake.

These tests intentionally cross the canonical catalog, durable ingest receipt,
workspace association, readiness, restart, and unlink boundaries.  Unit tests
own the individual adapters; this file proves those owners agree on identity.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from textual.app import App

from tldw_chatbook.app import LibraryIngestQueueMixin
from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Library.library_ingest_jobs import (
    IngestJobState,
    LibraryIngestJobRegistry,
    plan_restore,
)
from tldw_chatbook.Research_Workspace.contracts import (
    QualifiedWorkspaceRef,
    SourceReadiness,
    SourceReadinessState,
    WorkspaceDataSource,
)
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
from tldw_chatbook.Research_Workspace.source_readiness import (
    ResearchSourceReadinessCoordinator,
)
from tldw_chatbook.runtime_policy.server_event_scope import (
    event_principal_id_from_active_context,
)
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _operation(
    operation_id: str,
    *,
    data_source: WorkspaceDataSource,
    workspace_id: str,
    server_profile_id: str = "",
    principal_id: str = "",
) -> ResearchSourceOperation:
    timestamp = _now()
    return ResearchSourceOperation(
        operation_id=operation_id,
        idempotency_key=f"round-trip:{operation_id}",
        data_source=data_source,
        server_profile_id=server_profile_id,
        principal_id=principal_id,
        workspace_id=workspace_id,
        canonical_item_type=(
            CanonicalItemType.LOCAL_LIBRARY
            if data_source is WorkspaceDataSource.LOCAL
            else CanonicalItemType.SERVER_MEDIA
        ),
        desired_selected=True,
        created_at=timestamp,
        updated_at=timestamp,
    )


def _link_job(
    store: ResearchSourceOperationStore,
    operation: ResearchSourceOperation,
    job_id: str,
) -> None:
    store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.IN_PROGRESS,
        expected_revision=operation.revision,
        ingest_job_id=job_id,
    )


def _restore_jobs(path: Path) -> tuple[LibraryIngestJobRegistry, LibraryIngestJobsDB]:
    persisted = LibraryIngestJobsDB(path)
    plan = plan_restore(
        persisted.all_jobs(),
        max_persisted=500,
        now_iso=_now(),
    )
    for job in plan.upsert:
        persisted.upsert_job(job)
    for job_id in plan.delete_ids:
        persisted.delete_job(job_id)
    registry = LibraryIngestJobRegistry()
    registry.restore(plan.jobs, plan.next_id)
    registry.attach_store(persisted)
    return registry, persisted


class _ExactReadyAdapter:
    def __init__(self, *, catalog_item_id: str) -> None:
        self.catalog_item_id = catalog_item_id
        self.calls: list[tuple[QualifiedWorkspaceRef, tuple[str, ...]]] = []

    async def get_readiness(
        self,
        ref: QualifiedWorkspaceRef,
        *,
        source_ids: tuple[str, ...],
    ) -> tuple[SourceReadiness, ...]:
        self.calls.append((ref, source_ids))
        return (
            SourceReadiness(
                ref=ref,
                source_id=source_ids[0],
                catalog_item_id=self.catalog_item_id,
                state=SourceReadinessState.FTS_READY,
                metadata_ready=True,
                text_ready=True,
                fts_ready=True,
                vector_ready=False,
            ),
        )


@pytest.mark.asyncio
async def test_local_catalog_membership_restart_idempotency_and_unlink_round_trip(
    tmp_path: Path,
) -> None:
    """A Local item remains globally owned while its captured link comes and goes."""

    media_db = MediaDatabase(tmp_path / "media.sqlite", client_id="round-trip")
    content = "Durable Research source body."
    media_id, _, _ = media_db.add_media_with_keywords(
        title="Research paper",
        content=content,
        media_type="document",
        keywords=["workspace:captured"],
    )
    assert media_id is not None
    duplicate_id, _, duplicate_message = media_db.add_media_with_keywords(
        url="file:///a-different-source-name.txt",
        title="Research paper copy",
        content=content,
        media_type="document",
        keywords=["workspace:visible"],
    )
    duplicate_owner = media_db.get_media_by_hash(
        hashlib.sha256(content.encode()).hexdigest()
    )
    # The canonical owner may return the matched ID while applying its allowed
    # one-way URL canonicalization, but it must not create a second row.
    assert duplicate_id == media_id
    assert "canonicalized" in duplicate_message.lower()
    assert duplicate_owner is not None and duplicate_owner["id"] == media_id
    library_page = media_db.search_library_media_page(
        query="Research paper", limit=25, offset=0
    )
    assert library_page["total"] == 1
    assert library_page["items"][0]["id"] == media_id

    workspace_path = tmp_path / "workspaces.sqlite"
    first_db = WorkspaceDB(workspace_path, client_id="round-trip-first")
    first_registry = LocalWorkspaceRegistryService(first_db)
    first_registry.create_workspace(workspace_id="captured", name="Captured")
    first_registry.create_workspace(workspace_id="visible", name="Visible")
    first_registry.set_active_workspace("captured")
    first_store = ResearchSourceOperationStore(first_db)
    operation = first_store.create(
        _operation(
            "local-round-trip",
            data_source=WorkspaceDataSource.LOCAL,
            workspace_id="captured",
        )
    )

    jobs_path = tmp_path / "ingest-jobs.sqlite"
    jobs_db = LibraryIngestJobsDB(jobs_path)
    jobs = LibraryIngestJobRegistry()
    jobs.attach_store(jobs_db)
    job = jobs.submit(
        source_path="source.txt",
        title="Research paper",
        detected_type="document",
        research_source_operation_id=operation.operation_id,
    )
    _link_job(first_store, operation, job.job_id)
    jobs.mark_done(job.job_id, media_id=media_id)
    first_registry.set_active_workspace("visible")
    jobs_db.close()

    restored_jobs, restored_jobs_db = _restore_jobs(jobs_path)
    restarted_db = WorkspaceDB(workspace_path, client_id="round-trip-restart")
    restarted_registry = LocalWorkspaceRegistryService(restarted_db)
    restarted_store = ResearchSourceOperationStore(restarted_db)
    ready_adapter = _ExactReadyAdapter(catalog_item_id=str(media_id))
    readiness = ResearchSourceReadinessCoordinator(
        operation_store=restarted_store,
        adapters={WorkspaceDataSource.LOCAL: ready_adapter},
    )
    scheduler = ResearchSourceAssociationScheduler(
        coordinator=ResearchSourceAssociationCoordinator(
            operation_store=restarted_store,
            ingest_jobs=restored_jobs,
            local_registry=restarted_registry,
        ),
        operation_store=restarted_store,
        readiness_coordinator=readiness,
    )

    await scheduler.resume_startup(association_limit=1, readiness_limit=1)

    settled = restarted_store.get(operation.operation_id)
    assert settled is not None
    assert settled.catalog_status is SourceOperationStatus.SUCCEEDED
    assert settled.association_status is SourceOperationStatus.SUCCEEDED
    assert settled.readiness_status is SourceOperationStatus.SUCCEEDED
    assert restarted_registry.get_active_workspace().workspace_id == "visible"
    memberships = restarted_registry.get_item_memberships("media", str(media_id))
    assert [(row.workspace_id, row.role) for row in memberships] == [
        ("captured", "source")
    ]
    assert ready_adapter.calls == [
        (
            QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "captured"),
            (settled.workspace_source_id,),
        )
    ]

    duplicate_operation = restarted_store.create(
        _operation(
            "local-round-trip-duplicate",
            data_source=WorkspaceDataSource.LOCAL,
            workspace_id="captured",
        )
    )
    duplicate_operation = restarted_store.advance_stage(
        duplicate_operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.SUCCEEDED,
        expected_revision=duplicate_operation.revision,
        canonical_item_id=str(media_id),
    )
    duplicate_receipt = await scheduler.resume(duplicate_operation.operation_id)
    assert duplicate_receipt is not None
    assert duplicate_receipt.workspace_source_id == settled.workspace_source_id
    assert len(restarted_registry.get_item_memberships("media", str(media_id))) == 1

    restarted_registry.set_workspace_scope(
        "captured",
        RagScope(
            items=(ScopeItem("media", str(media_id)),),
            updated_at=_now(),
            empty_is_scoped=True,
        ),
    )
    assert restarted_registry.unlink_membership(
        "captured", item_type="media", item_id=str(media_id), role="source"
    )
    assert restarted_registry.get_item_memberships("media", str(media_id)) == ()
    assert restarted_registry.get_workspace_scope("captured").items == ()
    assert media_db.get_media_by_id(media_id)["id"] == media_id
    assert (
        media_db.search_library_media_page(query="Research paper", limit=25, offset=0)[
            "total"
        ]
        == 1
    )
    restored_jobs_db.close()


@pytest.mark.asyncio
async def test_association_failure_keeps_local_catalog_item(tmp_path: Path) -> None:
    media_db = MediaDatabase(tmp_path / "media.sqlite", client_id="partial-failure")
    media_id, _, _ = media_db.add_media_with_keywords(
        title="Surviving paper",
        content="The catalog commit is independent.",
        media_type="document",
    )
    assert media_id is not None
    store = ResearchSourceOperationStore(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="partial-failure")
    )
    operation = store.create(
        _operation(
            "local-association-failure",
            data_source=WorkspaceDataSource.LOCAL,
            workspace_id="missing-workspace",
        )
    )
    jobs = LibraryIngestJobRegistry()
    job = jobs.submit(
        source_path="source.txt",
        research_source_operation_id=operation.operation_id,
    )
    _link_job(store, operation, job.job_id)
    jobs.mark_done(job.job_id, media_id=media_id)

    failed = await ResearchSourceAssociationCoordinator(
        operation_store=store,
        ingest_jobs=jobs,
        local_registry=LocalWorkspaceRegistryService(store._db),
    ).resume(operation.operation_id)

    assert failed.catalog_status is SourceOperationStatus.SUCCEEDED
    assert failed.association_status is SourceOperationStatus.FAILED
    assert failed.canonical_item_id == str(media_id)
    assert media_db.get_media_by_id(media_id)["id"] == media_id
    assert (
        media_db.search_library_media_page(query="Surviving paper", limit=25, offset=0)[
            "total"
        ]
        == 1
    )


def test_workspace_keyword_is_projection_not_membership(tmp_path: Path) -> None:
    media_db = MediaDatabase(tmp_path / "media.sqlite", client_id="keyword-projection")
    media_id, _, _ = media_db.add_media_with_keywords(
        title="Tagged only",
        content="A keyword cannot create ownership.",
        media_type="document",
        keywords=["workspace:research"],
    )
    assert media_id is not None
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="keyword-projection")
    )
    registry.create_workspace(workspace_id="research", name="Research")

    memberships, total = registry.list_workspace_source_memberships("research")

    assert memberships == ()
    assert total == 0
    assert media_db.get_media_by_id(media_id)["id"] == media_id


class _ServerRoundTripApp(LibraryIngestQueueMixin, App):
    """Bounded real ingest host: production mixin, registry, and workers."""

    REMOTE_INGEST_POLL_SECONDS = 0.01

    def __init__(self, local_media_spy: object) -> None:
        super().__init__()
        self._init_library_ingest_runtime_state()
        self.media_db = local_media_spy
        self.runtime_policy = SimpleNamespace(
            state=SimpleNamespace(active_source="server")
        )

    def _resolve_ingest_backend(self) -> str:
        return "server"


class _ServerCatalog:
    """Fake external owner whose catalog is created only by real dispatch/poll."""

    def __init__(self, *, mismatched_result: bool = False) -> None:
        self._mismatched_result = mismatched_result
        self._next_remote_job_id = 40
        self._next_media_id = 880
        self._pending: dict[str, dict[str, object]] = {}
        self.media: dict[int, dict[str, object]] = {}
        self.sources: dict[str, dict[str, object]] = {}
        self.submit_calls: list[dict[str, object]] = []
        self.batch_calls: list[str] = []

    async def submit_ingest_jobs(self, **kwargs: object) -> dict[str, object]:
        self.submit_calls.append(dict(kwargs))
        self._next_remote_job_id += 1
        self._next_media_id += 1
        remote_job_id = str(self._next_remote_job_id)
        batch_id = f"batch-{remote_job_id}"
        self._pending[remote_job_id] = {
            "batch_id": batch_id,
            "media_id": self._next_media_id,
            "title": str(kwargs.get("title") or "Server paper"),
        }
        return {
            "batch_id": batch_id,
            "jobs": [{"id": int(remote_job_id), "status": "queued"}],
            "errors": [],
        }

    async def list_media_ingest_jobs(
        self,
        batch_id: str,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, object]:
        del offset, limit
        self.batch_calls.append(batch_id)
        remote_job_id, pending = next(
            (
                (job_id, row)
                for job_id, row in self._pending.items()
                if row["batch_id"] == batch_id
            ),
            ("", None),
        )
        if pending is None:
            return {"batch_id": batch_id, "jobs": []}
        media_id = int(pending["media_id"])
        self.media.setdefault(
            media_id,
            {
                "id": media_id,
                "title": pending["title"],
                "remote_job_id": remote_job_id,
            },
        )
        result_media_id = media_id + 100 if self._mismatched_result else media_id
        return {
            "batch_id": batch_id,
            "jobs": [
                {
                    "id": int(remote_job_id),
                    "status": "completed",
                    "result": {"media_id": result_media_id},
                }
            ],
            "has_more": False,
        }

    async def save_workspace_source(self, **kwargs: object) -> dict[str, object]:
        media_id = int(kwargs["media_id"])
        if media_id not in self.media:
            raise ValueError("workspace association referenced non-catalog media")
        row = {
            "id": kwargs["source_id"],
            "workspace_id": kwargs["workspace_id"],
            "media_id": media_id,
            "title": kwargs.get("title", "Server paper"),
            "source_type": kwargs.get("source_type", "document"),
            "selected": kwargs.get("selected", True),
            "version": 1,
        }
        self.sources[str(row["id"])] = row
        return row

    def my_media(self) -> tuple[dict[str, object], ...]:
        return tuple(self.media[media_id] for media_id in sorted(self.media))


def _wire_server_round_trip(
    tmp_path: Path,
    *,
    active_context: SimpleNamespace,
    captured_context: SimpleNamespace | None = None,
    mismatched_result: bool = False,
) -> tuple[
    _ServerRoundTripApp,
    ResearchSourceOperationStore,
    ResearchSourceOperation,
    _ServerCatalog,
    MagicMock,
    MagicMock,
    LibraryIngestJobsDB,
]:
    local_media = MediaDatabase(tmp_path / "local-media.sqlite", client_id="no-blend")
    local_media_spy = MagicMock(spec=MediaDatabase, wraps=local_media)
    local_registry_spy = MagicMock(spec=LocalWorkspaceRegistryService)
    app = _ServerRoundTripApp(local_media_spy)
    jobs_db = LibraryIngestJobsDB(tmp_path / "server-ingest-jobs.sqlite")
    app.library_ingest_jobs.attach_store(jobs_db)
    app.server_context_provider = SimpleNamespace(
        get_active_context=lambda: active_context
    )
    server = _ServerCatalog(mismatched_result=mismatched_result)
    app.server_media_reading_service = server
    store = ResearchSourceOperationStore(
        WorkspaceDB(tmp_path / "receipts.sqlite", client_id="server-round-trip")
    )
    operation_context = captured_context or active_context
    operation = store.create(
        _operation(
            "server-round-trip",
            data_source=WorkspaceDataSource.SERVER,
            workspace_id="server-workspace-7",
            server_profile_id="profile-a",
            principal_id=(
                event_principal_id_from_active_context(operation_context) or ""
            ),
        )
    )
    coordinator = ResearchSourceAssociationCoordinator(
        operation_store=store,
        ingest_jobs=app.library_ingest_jobs,
        local_registry=local_registry_spy,
        server_service=server,
        server_context_provider=app.server_context_provider,
    )
    app.research_source_operation_store = store
    app.research_source_association_scheduler = ResearchSourceAssociationScheduler(
        coordinator=coordinator,
        operation_store=store,
    )
    return (
        app,
        store,
        operation,
        server,
        local_media_spy,
        local_registry_spy,
        jobs_db,
    )


async def _wait_for_operation_status(
    app: _ServerRoundTripApp,
    pilot,
    store: ResearchSourceOperationStore,
    operation_id: str,
    *,
    association_status: SourceOperationStatus,
) -> ResearchSourceOperation:
    for _ in range(300):
        operation = store.get(operation_id)
        if operation is not None and operation.association_status is association_status:
            return operation
        await pilot.pause(0.02)
    raise AssertionError(f"operation never reached {association_status.value}")


@pytest.mark.asyncio
async def test_server_catalog_and_workspace_source_remain_remote_only(
    tmp_path: Path,
) -> None:
    context = SimpleNamespace(
        active_server_id="profile-a",
        auth_token="fixture-token",
        credential_source="fixture",
    )
    (
        app,
        store,
        operation,
        server,
        local_media_spy,
        local_registry_spy,
        jobs_db,
    ) = _wire_server_round_trip(tmp_path, active_context=context)
    source = tmp_path / "server-paper.pdf"
    source.write_bytes(b"%PDF-1.4\nfixture")
    assert server.my_media() == ()

    async with app.run_test() as pilot:
        job = app.prepare_research_source_ingest_job(
            source_path=str(source),
            title="Server paper",
            research_source_operation_id=operation.operation_id,
            required_origin="server",
        )
        operation = store.advance_stage(
            operation.operation_id,
            stage=SourceOperationStage.CATALOG,
            status=SourceOperationStatus.IN_PROGRESS,
            expected_revision=operation.revision,
            ingest_job_id=job.job_id,
        )
        app._dispatch_research_source_catalog_job(job.job_id)
        settled = await _wait_for_operation_status(
            app,
            pilot,
            store,
            operation.operation_id,
            association_status=SourceOperationStatus.SUCCEEDED,
        )

    catalog = server.my_media()
    assert len(catalog) == 1
    canonical_media = catalog[0]
    assert canonical_media["title"] == "Server paper"
    assert settled.catalog_status is SourceOperationStatus.SUCCEEDED
    assert settled.canonical_item_id == str(canonical_media["id"])
    assert settled.workspace_source_id in server.sources
    assert server.sources[settled.workspace_source_id] == {
        "id": settled.workspace_source_id,
        "workspace_id": "server-workspace-7",
        "media_id": canonical_media["id"],
        "title": "Server paper",
        "source_type": "pdf",
        "selected": True,
        "version": 1,
    }
    terminal = app.library_ingest_jobs.get_job(job.job_id)
    assert terminal is not None and terminal.state is IngestJobState.DONE
    assert canonical_media["remote_job_id"] == terminal.remote_job_id
    assert terminal.media_id is None
    assert terminal.remote_media_id == str(canonical_media["id"])
    assert server.submit_calls and server.batch_calls
    assert local_media_spy.mock_calls == []
    assert local_registry_spy.mock_calls == []
    jobs_db.close()


@pytest.mark.asyncio
async def test_server_remote_result_missing_from_my_media_fails_closed(
    tmp_path: Path,
) -> None:
    context = SimpleNamespace(
        active_server_id="profile-a",
        auth_token="fixture-token",
        credential_source="fixture",
    )
    (
        app,
        store,
        operation,
        server,
        local_media_spy,
        local_registry_spy,
        jobs_db,
    ) = _wire_server_round_trip(
        tmp_path,
        active_context=context,
        mismatched_result=True,
    )
    source = tmp_path / "mismatched-server-paper.pdf"
    source.write_bytes(b"%PDF-1.4\nfixture")

    async with app.run_test() as pilot:
        job = app.prepare_research_source_ingest_job(
            source_path=str(source),
            title="Server paper",
            research_source_operation_id=operation.operation_id,
            required_origin="server",
        )
        operation = store.advance_stage(
            operation.operation_id,
            stage=SourceOperationStage.CATALOG,
            status=SourceOperationStatus.IN_PROGRESS,
            expected_revision=operation.revision,
            ingest_job_id=job.job_id,
        )
        app._dispatch_research_source_catalog_job(job.job_id)
        failed = await _wait_for_operation_status(
            app,
            pilot,
            store,
            operation.operation_id,
            association_status=SourceOperationStatus.FAILED,
        )

    assert failed.catalog_status is SourceOperationStatus.SUCCEEDED
    assert failed.error_code == "association_failed"
    assert server.my_media()
    assert failed.canonical_item_id != str(server.my_media()[0]["id"])
    assert server.sources == {}
    assert local_media_spy.mock_calls == []
    assert local_registry_spy.mock_calls == []
    terminal = app.library_ingest_jobs.get_job(job.job_id)
    assert terminal is not None and terminal.media_id is None
    jobs_db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["profile", "principal"])
async def test_server_identity_mismatch_fails_closed_without_local_call(
    tmp_path: Path,
    mismatch: str,
) -> None:
    submitted = SimpleNamespace(
        active_server_id="profile-a",
        auth_token="submitted-token",
        credential_source="fixture",
    )
    active = SimpleNamespace(
        active_server_id="profile-b" if mismatch == "profile" else "profile-a",
        auth_token="active-token" if mismatch == "principal" else "submitted-token",
        credential_source="fixture",
    )
    (
        app,
        _store,
        operation,
        server,
        local_media_spy,
        local_registry_spy,
        jobs_db,
    ) = _wire_server_round_trip(
        tmp_path,
        active_context=active,
        captured_context=submitted,
    )
    source = tmp_path / "identity-mismatch.pdf"
    source.write_bytes(b"%PDF-1.4\nfixture")

    with pytest.raises(ValueError, match="captured Server workspace authority"):
        app.prepare_research_source_ingest_job(
            source_path=str(source),
            research_source_operation_id=operation.operation_id,
            required_origin="server",
        )

    assert app.library_ingest_jobs.jobs() == ()
    assert server.submit_calls == []
    assert server.my_media() == ()
    assert server.sources == {}
    assert local_media_spy.mock_calls == []
    assert local_registry_spy.mock_calls == []
    jobs_db.close()
