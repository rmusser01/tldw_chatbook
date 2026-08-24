"""Regression coverage for Library/Home retries of Research-owned ingest jobs."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Library.library_ingest_jobs import (
    IngestJobState,
    LibraryIngestJob,
    LibraryIngestJobRegistry,
    _job_from_row,
)
from tldw_chatbook.Home.active_work_adapter import HomeControlResultStatus
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
from tldw_chatbook.app import TldwCli


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _research_retry_app(
    tmp_path: Path,
    *,
    origin: str = "local",
) -> tuple[
    TldwCli,
    ResearchSourceOperationStore,
    LibraryIngestJobsDB,
    LibraryIngestJob,
    list[Any],
    list[tuple[str, str, bool]],
]:
    """Build the real durable retry owners while stubbing only dispatch work."""

    app = object.__new__(TldwCli)
    app.library_ingest_jobs = LibraryIngestJobRegistry()
    ingest_path = tmp_path / f"ingest-{origin}.sqlite"
    ingest_db = LibraryIngestJobsDB(ingest_path)
    app.library_ingest_jobs.attach_store(ingest_db)
    workspace_db = WorkspaceDB(
        tmp_path / f"workspace-{origin}.sqlite",
        client_id=f"retry-routing-{origin}",
    )
    operation_store = ResearchSourceOperationStore(workspace_db)
    app.research_source_operation_store = operation_store
    app.media_db = object()
    app.notify = Mock()
    scheduled: list[Any] = []
    app.run_worker = lambda awaitable, **_kwargs: scheduled.append(awaitable)

    operation_id = f"source-op-retry-{origin}"
    timestamp = _timestamp()
    operation = operation_store.create(
        ResearchSourceOperation(
            operation_id=operation_id,
            idempotency_key=f"retry-routing:{origin}",
            data_source=(
                WorkspaceDataSource.LOCAL
                if origin == "local"
                else WorkspaceDataSource.SERVER
            ),
            server_profile_id="profile-a" if origin == "server" else "",
            principal_id="principal-a" if origin == "server" else "",
            workspace_id="workspace-a",
            canonical_item_type=(
                CanonicalItemType.LOCAL_LIBRARY
                if origin == "local"
                else CanonicalItemType.SERVER_MEDIA
            ),
            desired_selected=True,
            created_at=timestamp,
            updated_at=timestamp,
        )
    )
    failed = app.library_ingest_jobs.submit(
        source_path=str(tmp_path / "evidence.txt"),
        origin=origin,
        detected_type="document",
        ingest_options={"generic": {"chunk": True}},
        research_source_operation_id=operation.operation_id,
        require_persisted=True,
    )
    app.library_ingest_jobs.mark_failed(
        failed.job_id,
        error="Temporary catalog failure.",
        require_persisted=True,
    )
    operation = operation_store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.IN_PROGRESS,
        expected_revision=operation.revision,
        ingest_job_id=failed.job_id,
    )
    operation_store.advance_stage(
        operation.operation_id,
        stage=SourceOperationStage.CATALOG,
        status=SourceOperationStatus.FAILED,
        expected_revision=operation.revision,
        error_code="catalog_ingest_failed",
        error_message="Catalog ingest did not complete successfully.",
    )

    dispatch_trace: list[tuple[str, str, bool]] = []

    def dispatch(job_id: str) -> None:
        operation_at_dispatch = operation_store.get(operation_id)
        job = app.library_ingest_jobs.get_job(job_id)
        assert operation_at_dispatch is not None
        assert job is not None
        dispatch_trace.append(
            (operation_at_dispatch.ingest_job_id, job.job_id, job.dispatch_held)
        )
        released = app.library_ingest_jobs.release_dispatch_hold(
            job_id,
            require_persisted=True,
        )
        assert released is not None

    coordinator = ResearchSourceAssociationCoordinator(
        operation_store=operation_store,
        ingest_jobs=app.library_ingest_jobs,
        catalog_requeuer=app._requeue_research_source_catalog_job,
        catalog_dispatcher=dispatch,
    )
    app.research_source_association_scheduler = ResearchSourceAssociationScheduler(
        coordinator=coordinator,
        operation_store=operation_store,
    )
    app._top_up_ingest_parse_pool = Mock()
    return app, operation_store, ingest_db, failed, scheduled, dispatch_trace


async def _run_scheduled(scheduled: list[Any]) -> list[Any]:
    awaitables = list(scheduled)
    scheduled.clear()
    return list(await asyncio.gather(*awaitables))


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["local", "server"])
async def test_direct_research_retry_uses_held_scheduler_lineage_and_returns_exact_job(
    tmp_path: Path,
    origin: str,
) -> None:
    """Removing the Research branch reintroduces an unheld orphan replacement."""

    app, store, ingest_db, failed, scheduled, dispatch_trace = _research_retry_app(
        tmp_path,
        origin=origin,
    )
    requeue_calls: list[bool] = []
    real_requeue = app.library_ingest_jobs.requeue

    def record_requeue(job_id: str, **kwargs: Any) -> LibraryIngestJob | None:
        requeue_calls.append(bool(kwargs.get("dispatch_held", False)))
        return real_requeue(job_id, **kwargs)

    app.library_ingest_jobs.requeue = record_requeue

    immediate = app.retry_library_ingest_job(failed.job_id)

    assert immediate is None
    assert len(scheduled) == 1
    [replacement] = await _run_scheduled(scheduled)
    assert replacement is not None
    receipt = store.get(failed.research_source_operation_id)
    assert receipt is not None
    assert replacement.job_id == receipt.ingest_job_id
    assert replacement.retry_of_job_id == failed.job_id
    assert (
        replacement.research_source_operation_id == failed.research_source_operation_id
    )
    assert replacement.dispatch_held is False
    assert requeue_calls == [True]
    assert dispatch_trace == [(replacement.job_id, replacement.job_id, True)]
    app._top_up_ingest_parse_pool.assert_not_called()

    ingest_path = ingest_db.db_path
    ingest_db.close()
    restarted = LibraryIngestJobsDB(ingest_path)
    try:
        row = next(
            item
            for item in restarted.all_jobs()
            if item["job_id"] == replacement.job_id
        )
        assert row["dispatch_held"] == 0
        assert (
            row["research_source_operation_id"] == failed.research_source_operation_id
        )
    finally:
        restarted.close()


@pytest.mark.asyncio
async def test_provider_recovery_for_research_routes_original_snapshot_through_scheduler(
    tmp_path: Path,
) -> None:
    """A provider-specific Library action cannot bypass durable Research retry."""

    app, store, _db, failed, scheduled, dispatch_trace = _research_retry_app(tmp_path)

    immediate = app.retry_library_ingest_job_with_provider(
        failed.job_id,
        "faster-whisper",
    )

    assert immediate is None
    [replacement] = await _run_scheduled(scheduled)
    assert replacement.ingest_options == failed.ingest_options
    assert store.get(failed.research_source_operation_id).ingest_job_id == (
        replacement.job_id
    )
    assert dispatch_trace == [(replacement.job_id, replacement.job_id, True)]
    app._top_up_ingest_parse_pool.assert_not_called()


@pytest.mark.asyncio
async def test_home_research_retry_reports_request_and_uses_scheduler(
    tmp_path: Path,
) -> None:
    """Home must not mistake an async Research retry request for a stale job."""

    app, store, _db, failed, scheduled, _trace = _research_retry_app(tmp_path)

    result = app.retry_active_home_item(target_id=f"local:ingest:{failed.job_id}")

    assert result.status is HomeControlResultStatus.HANDLED
    assert "Research source retry requested" in result.message
    [replacement] = await _run_scheduled(scheduled)
    assert store.get(failed.research_source_operation_id).ingest_job_id == (
        replacement.job_id
    )
    app._top_up_ingest_parse_pool.assert_not_called()


@pytest.mark.asyncio
async def test_research_retry_without_scheduler_fails_closed_without_generic_requeue(
    tmp_path: Path,
) -> None:
    """Unavailable Research ownership must mutate neither queue nor parse pool."""

    app, store, _db, failed, scheduled, _trace = _research_retry_app(tmp_path)
    app.research_source_association_scheduler = None
    before = store.get(failed.research_source_operation_id)

    result = app.retry_library_ingest_job(failed.job_id)

    assert result is None
    assert scheduled == []
    assert app.library_ingest_jobs.get_job(failed.job_id).state is IngestJobState.FAILED
    assert store.get(failed.research_source_operation_id) == before
    app._top_up_ingest_parse_pool.assert_not_called()
    assert "Research Workspace" in app.notify.call_args.args[0]


@pytest.mark.asyncio
async def test_research_retry_worker_error_is_sanitized_and_does_not_mutate(
    tmp_path: Path,
) -> None:
    """Scheduler failures expose fixed recovery, never internals or fallback work."""

    app, store, _db, failed, scheduled, dispatch_trace = _research_retry_app(tmp_path)

    class FailingScheduler:
        async def retry(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("Bearer secret-token at /private/sensitive/source.txt")

    app.research_source_association_scheduler = FailingScheduler()
    before = store.get(failed.research_source_operation_id)
    app.notify.reset_mock()

    assert app.retry_library_ingest_job(failed.job_id) is None
    [result] = await _run_scheduled(scheduled)

    assert result is None
    assert store.get(failed.research_source_operation_id) == before
    assert app.library_ingest_jobs.get_job(failed.job_id).state is IngestJobState.FAILED
    assert dispatch_trace == []
    app._top_up_ingest_parse_pool.assert_not_called()
    assert app.notify.call_args.args[0] == (
        "Research source retry is unavailable. Open Research Workspace "
        "and retry from its receipt."
    )
    assert "secret-token" not in app.notify.call_args.args[0]
    assert "/private/" not in app.notify.call_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["local", "server"])
async def test_release_then_raise_settles_replacement_before_terminal_receipt(
    tmp_path: Path,
    origin: str,
) -> None:
    """A dispatcher failure cannot leave its released retry queueable."""

    app, store, ingest_db, failed, scheduled, _trace = _research_retry_app(
        tmp_path,
        origin=origin,
    )
    dispatch_calls: list[str] = []

    def release_then_raise(job_id: str) -> None:
        released = app.library_ingest_jobs.release_dispatch_hold(
            job_id,
            require_persisted=True,
        )
        assert released is not None
        assert released.origin == origin
        assert released.dispatch_held is False
        dispatch_calls.append(job_id)
        raise RuntimeError("Bearer secret-token at /private/sensitive/source.txt")

    coordinator = ResearchSourceAssociationCoordinator(
        operation_store=store,
        ingest_jobs=app.library_ingest_jobs,
        catalog_requeuer=app._requeue_research_source_catalog_job,
        catalog_dispatcher=release_then_raise,
    )
    app.research_source_association_scheduler = ResearchSourceAssociationScheduler(
        coordinator=coordinator,
        operation_store=store,
    )
    app.notify.reset_mock()

    assert app.retry_library_ingest_job(failed.job_id) is None
    [result] = await _run_scheduled(scheduled)

    receipt = store.get(failed.research_source_operation_id)
    assert result is None
    assert receipt is not None
    assert receipt.catalog_status is SourceOperationStatus.FAILED
    assert receipt.error_code == "catalog_retry_failed"
    assert receipt.ingest_job_id != failed.job_id
    replacement = app.library_ingest_jobs.get_job(receipt.ingest_job_id)
    assert replacement is not None
    assert replacement.state is IngestJobState.FAILED
    assert replacement.origin == origin
    assert replacement.dispatch_held is False
    assert app.library_ingest_jobs.next_queued() is None
    assert dispatch_calls == [replacement.job_id]
    app._top_up_ingest_parse_pool.assert_not_called()
    assert app.notify.call_args.args[0] == (
        "Research source retry is unavailable. Open Research Workspace "
        "and retry from its receipt."
    )
    assert "secret-token" not in app.notify.call_args.args[0]
    assert "/private/" not in app.notify.call_args.args[0]

    ingest_path = ingest_db.db_path
    workspace_path = store._db.db_path
    ingest_db.close()
    store._db.close()
    reopened_db = LibraryIngestJobsDB(ingest_path)
    reopened_workspace_db = WorkspaceDB(
        workspace_path,
        client_id=f"retry-routing-reopened-{origin}",
    )
    try:
        rows = reopened_db.all_jobs()
        restored = LibraryIngestJobRegistry()
        restored.restore(
            [_job_from_row(row) for row in rows],
            next_id=max(row["seq"] for row in rows) + 1,
        )
        restored.attach_store(reopened_db)
        restored_replacement = restored.get_job(replacement.job_id)
        assert restored_replacement is not None
        assert restored_replacement.state is IngestJobState.FAILED
        assert restored.next_queued() is None
        restored_receipt = ResearchSourceOperationStore(reopened_workspace_db).get(
            receipt.operation_id
        )
        assert restored_receipt is not None
        assert restored_receipt.catalog_status is SourceOperationStatus.FAILED
        assert restored_receipt.ingest_job_id == restored_replacement.job_id
    finally:
        reopened_db.close()
        reopened_workspace_db.close()


def test_home_research_retry_without_scheduler_names_research_recovery(
    tmp_path: Path,
) -> None:
    """Home cannot describe a missing Research owner as a stale Library job."""

    app, store, _db, failed, scheduled, _trace = _research_retry_app(tmp_path)
    app.research_source_association_scheduler = None
    before = store.get(failed.research_source_operation_id)

    result = app.retry_active_home_item(target_id=f"local:ingest:{failed.job_id}")

    assert result.status is HomeControlResultStatus.UNAVAILABLE
    assert result.message == (
        "Research source retry is unavailable. Open Research Workspace "
        "and retry from its receipt."
    )
    assert result.recovery_route == "research_workspace"
    assert result.target_route == "research_workspace"
    assert scheduled == []
    assert store.get(failed.research_source_operation_id) == before
    app._top_up_ingest_parse_pool.assert_not_called()


@pytest.mark.asyncio
async def test_research_retry_rejects_clicked_job_lineage_mismatch_before_scheduler(
    tmp_path: Path,
) -> None:
    """A forged/stale job link cannot retarget the durable operation's job."""

    app, store, _db, failed, scheduled, dispatch_trace = _research_retry_app(tmp_path)
    rogue = app.library_ingest_jobs.submit(
        source_path=str(tmp_path / "rogue.txt"),
        origin="local",
        research_source_operation_id=failed.research_source_operation_id,
        require_persisted=True,
    )
    app.library_ingest_jobs.mark_failed(
        rogue.job_id,
        error="Rogue failure.",
        require_persisted=True,
    )
    before = store.get(failed.research_source_operation_id)

    assert app.retry_library_ingest_job(rogue.job_id) is None
    [result] = await _run_scheduled(scheduled)

    assert result is None
    assert store.get(failed.research_source_operation_id) == before
    assert app.library_ingest_jobs.get_job(rogue.job_id).state is IngestJobState.FAILED
    assert dispatch_trace == []
    app._top_up_ingest_parse_pool.assert_not_called()


@pytest.mark.asyncio
async def test_research_retry_rejects_authority_mismatch_without_blending(
    tmp_path: Path,
) -> None:
    """A Server-labelled job cannot retry a Local operation or invoke Local work."""

    app, store, _db, failed, scheduled, dispatch_trace = _research_retry_app(tmp_path)
    registry = app.library_ingest_jobs
    index = registry._find_index(failed.job_id)
    registry._jobs[index] = replace(registry._jobs[index], origin="server")
    before = store.get(failed.research_source_operation_id)

    assert app.retry_library_ingest_job(failed.job_id) is None
    [result] = await _run_scheduled(scheduled)

    assert result is None
    assert store.get(failed.research_source_operation_id) == before
    assert dispatch_trace == []
    app._top_up_ingest_parse_pool.assert_not_called()


@pytest.mark.asyncio
async def test_concurrent_research_retry_clicks_create_one_replacement(
    tmp_path: Path,
) -> None:
    """The scheduler fence permits only one durable replacement lineage."""

    app, store, _db, failed, scheduled, dispatch_trace = _research_retry_app(tmp_path)

    assert app.retry_library_ingest_job(failed.job_id) is None
    assert app.retry_library_ingest_job(failed.job_id) is None
    results = await _run_scheduled(scheduled)

    replacements = [result for result in results if result is not None]
    assert len(replacements) == 2
    assert len({replacement.job_id for replacement in replacements}) == 1
    assert store.get(failed.research_source_operation_id).ingest_job_id == (
        replacements[0].job_id
    )
    assert len(dispatch_trace) == 1
    assert len(app.library_ingest_jobs.jobs()) == 1


def test_ordinary_library_retry_retains_legacy_return_and_top_up(
    tmp_path: Path,
) -> None:
    """Removing the ordinary branch would break the established sync contract."""

    app = object.__new__(TldwCli)
    app.library_ingest_jobs = LibraryIngestJobRegistry()
    app.media_db = object()
    app._top_up_ingest_parse_pool = Mock()
    job = app.library_ingest_jobs.submit(source_path=str(tmp_path / "ordinary.txt"))
    failed = app.library_ingest_jobs.mark_failed(job.job_id, error="Temporary failure")

    replacement = app.retry_library_ingest_job(failed.job_id)

    assert replacement is not None
    assert replacement.retry_of_job_id == failed.job_id
    assert replacement.research_source_operation_id is None
    assert replacement.dispatch_held is False
    app._top_up_ingest_parse_pool.assert_called_once_with()
