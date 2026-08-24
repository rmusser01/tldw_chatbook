from __future__ import annotations

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Research_Workspace.contracts import (
    QualifiedWorkspaceRef,
    RetrievalMode,
    SourceReadiness,
    SourceReadinessState,
    WorkspaceDataSource,
)
from tldw_chatbook.Research_Workspace.source_readiness import (
    ResearchSourceReadinessCoordinator,
    effective_source_ids,
    normalize_local_readiness,
    normalize_server_readiness,
)
from tldw_chatbook.Research_Workspace.source_association import (
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


REF = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-1")


def _readiness(
    source_id: str,
    *,
    state: SourceReadinessState,
    fts: bool,
    vector: bool,
) -> SourceReadiness:
    return SourceReadiness(
        ref=REF,
        source_id=source_id,
        catalog_item_id=source_id,
        state=state,
        metadata_ready=True,
        text_ready=fts,
        fts_ready=fts,
        vector_ready=vector,
    )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (RetrievalMode.FTS, ("1", "2")),
        (RetrievalMode.SEMANTIC, ("2",)),
        (RetrievalMode.HYBRID, ("2",)),
    ],
)
def test_effective_ids_require_the_requested_retrieval_paths(mode, expected) -> None:
    rows = (
        _readiness("1", state=SourceReadinessState.FTS_READY, fts=True, vector=False),
        _readiness("2", state=SourceReadinessState.VECTOR_READY, fts=True, vector=True),
        _readiness("3", state=SourceReadinessState.INDEXING, fts=False, vector=False),
    )

    assert effective_source_ids(("3", "2", "1"), rows, mode=mode) == expected


def test_explicit_empty_desired_scope_remains_empty() -> None:
    row = _readiness(
        "1", state=SourceReadinessState.VECTOR_READY, fts=True, vector=True
    )

    assert effective_source_ids((), (row,), mode=RetrievalMode.HYBRID) == ()


def test_missing_embeddings_is_honest_fts_only_not_hybrid() -> None:
    row = normalize_local_readiness(
        ref=REF,
        source_id="membership-1",
        catalog_item_id="12",
        detail={
            "has_transcript": True,
            "has_chunks": True,
            "chunking_status": "completed",
            "vector_processing": False,
        },
    )

    assert row.state is SourceReadinessState.FTS_READY
    assert row.fts_ready is True
    assert row.vector_ready is False
    assert row.next_action == "Refresh status"
    assert effective_source_ids(("12",), (row,), mode=RetrievalMode.HYBRID) == ()


@pytest.mark.parametrize(
    ("lifecycle", "expected"),
    [
        ("queued", SourceReadinessState.ATTACHED),
        ("ingesting", SourceReadinessState.PARSING),
        ("extracting", SourceReadinessState.PARSING),
        ("chunking", SourceReadinessState.INDEXING),
        ("indexing", SourceReadinessState.INDEXING),
        ("retrying", SourceReadinessState.INDEXING),
        ("queryable", SourceReadinessState.VECTOR_READY),
        ("partially_queryable", SourceReadinessState.FTS_READY),
        ("failed", SourceReadinessState.FAILED),
        ("missing_media", SourceReadinessState.UNAVAILABLE),
        ("blocked_by_permissions", SourceReadinessState.UNAVAILABLE),
    ],
)
def test_server_lifecycle_maps_to_closed_normalized_vocabulary(
    lifecycle, expected
) -> None:
    ref = QualifiedWorkspaceRef(
        WorkspaceDataSource.SERVER,
        "workspace-1",
        server_profile_id="profile-1",
        principal_id="principal-1",
    )
    row = normalize_server_readiness(
        ref=ref,
        status={
            "id": "source-1",
            "media_id": 12,
            "state": lifecycle,
            "readiness": {
                "metadata_ready": True,
                "text_extracted": lifecycle in {"queryable", "partially_queryable"},
                "fts_ready": lifecycle in {"queryable", "partially_queryable"},
                "vector_ready": lifecycle == "queryable",
                "citation_ready": False,
                "summary_ready": False,
                "tool_accessible": False,
            },
            "retry_eligible": lifecycle == "failed",
            "stale": False,
            "next_action": "retry_vector_indexing",
        },
    )

    assert row.state is expected
    assert row.next_action in {"Refresh status", "Re-add source"}
    assert "retry" not in row.next_action.lower()


def test_stale_server_projection_overrides_live_state_without_losing_flags() -> None:
    ref = QualifiedWorkspaceRef(
        WorkspaceDataSource.SERVER,
        "workspace-1",
        server_profile_id="profile-1",
        principal_id="principal-1",
    )
    row = normalize_server_readiness(
        ref=ref,
        status={
            "id": "source-1",
            "media_id": 12,
            "state": "queryable",
            "readiness": {
                "metadata_ready": True,
                "text_extracted": True,
                "fts_ready": True,
                "vector_ready": True,
                "citation_ready": True,
                "summary_ready": True,
                "tool_accessible": True,
            },
            "retry_eligible": False,
            "stale": True,
        },
    )

    assert row.state is SourceReadinessState.STALE
    assert row.fts_ready is True
    assert row.vector_ready is True


def test_unknown_server_lifecycle_fails_closed() -> None:
    ref = QualifiedWorkspaceRef(
        WorkspaceDataSource.SERVER,
        "workspace-1",
        server_profile_id="profile-1",
        principal_id="principal-1",
    )

    with pytest.raises(ValueError, match="lifecycle"):
        normalize_server_readiness(
            ref=ref,
            status={
                "id": "source-1",
                "media_id": 12,
                "state": "magically_ready",
                "readiness": {},
            },
        )


def test_readiness_diagnostics_do_not_expose_secret_material() -> None:
    ref = QualifiedWorkspaceRef(
        WorkspaceDataSource.SERVER,
        "workspace-1",
        server_profile_id="profile-1",
        principal_id="principal-1",
    )
    row = normalize_server_readiness(
        ref=ref,
        status={
            "id": "source-1",
            "media_id": 12,
            "state": "failed",
            "status_reason": "api_key=secret-value",
            "readiness": {},
            "retry_eligible": True,
            "stale": False,
        },
    )

    assert row.detail == "Readiness diagnostic withheld."
    assert "secret-value" not in row.detail


NOW = "2026-08-24T12:00:00Z"


def _associated_operation(store, *, index=1):
    operation = store.create(
        ResearchSourceOperation(
            operation_id=f"operation-{index}",
            idempotency_key=f"local:workspace-1:{index}",
            data_source=WorkspaceDataSource.LOCAL,
            workspace_id="workspace-1",
            canonical_item_type=CanonicalItemType.LOCAL_LIBRARY,
            desired_selected=True,
            created_at=NOW,
            updated_at=NOW,
        )
    )
    operation = store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.SUCCEEDED,
        expected_revision=operation.revision,
        canonical_item_id=str(100 + index),
    )
    return store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.ASSOCIATION,
        status=SourceOperationStatus.SUCCEEDED,
        expected_revision=operation.revision,
        workspace_source_id=f"membership-{index}",
    )


class ReadinessAdapter:
    def __init__(self, rows):
        self.rows = list(rows)
        self.calls = []

    async def get_readiness(self, ref, *, source_ids=()):
        self.calls.append((ref, source_ids))
        row = self.rows.pop(0)
        return (row,)


def _operation_readiness(state, *, fts=False, vector=False):
    return SourceReadiness(
        ref=REF,
        source_id="membership-1",
        catalog_item_id="101",
        state=state,
        metadata_ready=True,
        text_ready=fts,
        fts_ready=fts,
        vector_ready=vector,
    )


@pytest.mark.asyncio
async def test_readiness_receipt_advances_optimistically_and_fts_only_succeeds(
    tmp_path,
) -> None:
    store = ResearchSourceOperationStore(WorkspaceDB(tmp_path / "workspace.sqlite"))
    operation = _associated_operation(store)
    adapter = ReadinessAdapter(
        [_operation_readiness(SourceReadinessState.FTS_READY, fts=True)]
    )
    coordinator = ResearchSourceReadinessCoordinator(
        operation_store=store,
        adapters={WorkspaceDataSource.LOCAL: adapter},
    )

    settled = await coordinator.resume(operation.operation_id)

    assert settled.readiness_status is SourceOperationStatus.SUCCEEDED
    assert settled.revision == operation.revision + 2
    assert adapter.calls == [(REF, ("membership-1",))]


@pytest.mark.asyncio
async def test_pending_indexing_stays_pending_for_restart_resume(tmp_path) -> None:
    store = ResearchSourceOperationStore(WorkspaceDB(tmp_path / "workspace.sqlite"))
    operation = _associated_operation(store)
    adapter = ReadinessAdapter([_operation_readiness(SourceReadinessState.INDEXING)])
    coordinator = ResearchSourceReadinessCoordinator(
        operation_store=store,
        adapters={WorkspaceDataSource.LOCAL: adapter},
    )

    pending = await coordinator.resume(operation.operation_id)

    assert pending == operation
    assert (
        store.get(operation.operation_id).readiness_status
        is SourceOperationStatus.PENDING
    )


@pytest.mark.asyncio
async def test_readiness_retry_only_rechecks_existing_association(tmp_path) -> None:
    store = ResearchSourceOperationStore(WorkspaceDB(tmp_path / "workspace.sqlite"))
    operation = _associated_operation(store)
    failed = SourceReadiness(
        ref=REF,
        source_id="membership-1",
        catalog_item_id="101",
        state=SourceReadinessState.FAILED,
        next_action="Re-add source",
    )
    ready = _operation_readiness(
        SourceReadinessState.VECTOR_READY, fts=True, vector=True
    )
    adapter = ReadinessAdapter([failed, ready])
    coordinator = ResearchSourceReadinessCoordinator(
        operation_store=store,
        adapters={WorkspaceDataSource.LOCAL: adapter},
    )
    first = await coordinator.resume(operation.operation_id)

    retried = await coordinator.retry(operation.operation_id)

    assert first.readiness_status is SourceOperationStatus.FAILED
    assert first.error_code == "source_readiness_failed"
    assert retried.catalog_status is SourceOperationStatus.SUCCEEDED
    assert retried.association_status is SourceOperationStatus.SUCCEEDED
    assert retried.readiness_status is SourceOperationStatus.SUCCEEDED
    assert [call[1] for call in adapter.calls] == [
        ("membership-1",),
        ("membership-1",),
    ]


def test_readiness_actionable_query_filters_before_limit_and_rejects_bool(
    tmp_path,
) -> None:
    store = ResearchSourceOperationStore(WorkspaceDB(tmp_path / "workspace.sqlite"))
    for index in range(1, 56):
        operation = _associated_operation(store, index=index)
        store.advance_stage(
            operation.operation_id,
            stage=SourceOperationStage.READINESS,
            status=SourceOperationStatus.SUCCEEDED,
            expected_revision=operation.revision,
        )
    pending_56 = _associated_operation(store, index=56)
    pending_57 = _associated_operation(store, index=57)

    page = store.list_readiness_actionable(limit=2)

    assert [operation.operation_id for operation in page] == [
        pending_56.operation_id,
        pending_57.operation_id,
    ]
    with pytest.raises(ValueError, match="limit"):
        store.list_readiness_actionable(limit=True)


@pytest.mark.asyncio
async def test_association_scheduler_chains_readiness_and_routes_readiness_retry(
    tmp_path,
) -> None:
    store = ResearchSourceOperationStore(WorkspaceDB(tmp_path / "workspace.sqlite"))
    operation = _associated_operation(store)

    class AssociationCoordinator:
        def __init__(self):
            self.calls = []

        async def resume(self, operation_id):
            self.calls.append(("resume", operation_id))
            return store.get(operation_id)

        async def retry(self, operation_id, *, stage):
            self.calls.append(("retry", operation_id, stage))
            return store.get(operation_id)

    class ReadinessCoordinator:
        def __init__(self):
            self.calls = []

        async def resume(self, operation_id):
            self.calls.append(("resume", operation_id))
            return store.get(operation_id)

        async def retry(self, operation_id):
            self.calls.append(("retry", operation_id))
            return store.get(operation_id)

        async def resume_incomplete(self, *, limit):
            self.calls.append(("startup", limit))

    association = AssociationCoordinator()
    readiness = ReadinessCoordinator()
    scheduler = ResearchSourceAssociationScheduler(
        coordinator=association,
        operation_store=store,
        readiness_coordinator=readiness,
    )

    await scheduler.resume(operation.operation_id)
    await scheduler.retry(operation.operation_id, stage=SourceOperationStage.READINESS)
    await scheduler.resume_readiness_incomplete(limit=17)

    assert association.calls == [("resume", operation.operation_id)]
    assert readiness.calls == [
        ("resume", operation.operation_id),
        ("retry", operation.operation_id),
        ("startup", 17),
    ]


@pytest.mark.asyncio
async def test_startup_orders_bounded_association_before_bounded_readiness() -> None:
    trace = []

    class StartupStore:
        def list_association_actionable(self, *, limit):
            trace.append(("association", limit))
            return ()

    class AssociationCoordinator:
        async def resume(self, operation_id):
            raise AssertionError("no association rows were returned")

    class ReadinessCoordinator:
        async def resume_incomplete(self, *, limit):
            trace.append(("readiness", limit))

    scheduler = ResearchSourceAssociationScheduler(
        coordinator=AssociationCoordinator(),
        operation_store=StartupStore(),
        readiness_coordinator=ReadinessCoordinator(),
    )

    await scheduler.resume_startup(association_limit=13, readiness_limit=17)

    assert trace == [("association", 13), ("readiness", 17)]
