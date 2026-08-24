"""Private, operation-bound paste staging lifecycle."""

from __future__ import annotations

import json
import stat

from tldw_chatbook.Library.library_ingest_jobs import LibraryIngestJobRegistry
from tldw_chatbook.Research_Workspace.contracts import WorkspaceDataSource
from tldw_chatbook.Research_Workspace.paste_staging import ResearchPasteStagingStore
from tldw_chatbook.Research_Workspace.source_operations import (
    CanonicalItemType,
    ResearchSourceOperation,
    SourceOperationStage,
    SourceOperationStatus,
)


def operation(
    operation_id: str,
    *,
    catalog_status: SourceOperationStatus = SourceOperationStatus.PENDING,
) -> ResearchSourceOperation:
    kwargs = {}
    if catalog_status is SourceOperationStatus.SUCCEEDED:
        kwargs["canonical_item_id"] = "7"
    elif catalog_status is SourceOperationStatus.FAILED:
        kwargs.update(
            error_stage=SourceOperationStage.CATALOG,
            error_code="catalog_ingest_failed",
            error_message="Catalog ingest did not complete successfully.",
        )
    return ResearchSourceOperation(
        operation_id=operation_id,
        idempotency_key=f"key-{operation_id}",
        data_source=WorkspaceDataSource.LOCAL,
        workspace_id="workspace",
        canonical_item_type=CanonicalItemType.LOCAL_LIBRARY,
        desired_selected=True,
        created_at="2026-08-24T10:00:00Z",
        updated_at="2026-08-24T10:00:00Z",
        catalog_status=catalog_status,
        **kwargs,
    )


def test_staging_is_private_and_sweep_retains_only_retryable_operations(
    tmp_path,
) -> None:
    store = ResearchPasteStagingStore(tmp_path / "paste-staging")
    pending = operation("operation-pending")
    failed = operation("operation-failed", catalog_status=SourceOperationStatus.FAILED)
    succeeded = operation(
        "operation-succeeded", catalog_status=SourceOperationStatus.SUCCEEDED
    )
    paths = {
        item.operation_id: store.stage(
            item.operation_id,
            title="Private title",
            body=f"PRIVATE BODY {item.operation_id}",
        )
        for item in (pending, failed, succeeded)
    }
    operation_store = type(
        "OperationStore",
        (),
        {
            "get": lambda _self, operation_id: {
                pending.operation_id: pending,
                failed.operation_id: failed,
                succeeded.operation_id: succeeded,
            }.get(operation_id)
        },
    )()

    swept = store.sweep(operation_store, limit=100)

    assert swept.deleted == 1
    assert swept.retained == 2
    assert paths[pending.operation_id].exists()
    assert paths[failed.operation_id].exists()
    assert not paths[succeeded.operation_id].exists()
    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(paths[pending.operation_id].stat().st_mode) == 0o600
    index_text = store.index_path.read_text(encoding="utf-8")
    assert "PRIVATE BODY" not in index_text
    assert str(tmp_path) not in index_text
    assert set(json.loads(index_text)["operations"].values()) == {
        pending.operation_id,
        failed.operation_id,
    }


def test_cancel_cleanup_deletes_only_the_bound_artifact(tmp_path) -> None:
    store = ResearchPasteStagingStore(tmp_path / "paste-staging")
    bound = store.stage("operation-bound", title="Paste", body="Private body")
    user_upload = tmp_path / "user-upload.txt"
    user_upload.write_text("user file", encoding="utf-8")

    assert store.delete("operation-bound")

    assert not bound.exists()
    assert user_upload.read_text(encoding="utf-8") == "user file"


def test_startup_sweep_retains_missing_operation_with_durable_held_job(
    tmp_path,
) -> None:
    """A concurrent sweep cannot outrun held-job reconciliation."""

    store = ResearchPasteStagingStore(tmp_path / "paste-staging")
    artifact = store.stage("operation-held", title="Paste", body="Private held body")
    registry = LibraryIngestJobRegistry()
    registry.submit(
        source_path=str(artifact),
        origin="local",
        research_source_operation_id="operation-held",
        dispatch_held=True,
    )
    missing_operation_store = type(
        "MissingOperationStore",
        (),
        {"get": lambda _self, _operation_id: None},
    )()

    swept = store.sweep(missing_operation_store, job_registry=registry, limit=100)

    assert swept.deleted == 0
    assert swept.retained == 1
    assert artifact.exists()
