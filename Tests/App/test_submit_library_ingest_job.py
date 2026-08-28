"""Tests for the app-level Library ingest job submission seam."""

from __future__ import annotations

import asyncio
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger
from textual.app import App

from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Library.ingest_capabilities import get_capabilities
from tldw_chatbook.Library.library_ingest_jobs import (
    ActiveIngestConsentScope,
    ActiveIngestJobRef,
    ActiveIngestSubmissionRefused,
    DEFAULT_CHUNK_SIZE,
    IngestJobState,
    LibraryIngestJob,
    LibraryIngestJobRegistry,
    build_active_ingest_consent_scope,
    plan_restore,
)
from tldw_chatbook.Library.library_ingest_state import LibraryIngestFormState
from tldw_chatbook.Model_Artifacts.service import ArtifactRef
from tldw_chatbook.Research_Workspace.contracts import WorkspaceDataSource
from tldw_chatbook.Research_Workspace.source_operations import (
    CanonicalItemType,
    ResearchSourceOperation,
    SourceOperationStatus,
)
from tldw_chatbook.Research_Workspace.source_operation_store import (
    ResearchSourceOperationStore,
)
from tldw_chatbook.Research_Workspace.paste_staging import ResearchPasteStagingStore
from tldw_chatbook.runtime_policy.server_event_scope import (
    event_principal_id_from_active_context,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.app import TldwCli
import tldw_chatbook.app as app_module


def _minimal_app(media_db: Any = None) -> TldwCli:
    """Return a TldwCli instance without running its heavy __init__."""
    app = object.__new__(TldwCli)
    app.library_ingest_jobs = LibraryIngestJobRegistry()
    app.media_db = media_db
    app._top_up_ingest_parse_pool = lambda: None  # type: ignore[method-assign]
    return app


def test_required_persisted_submit_is_durable_before_listener_visibility(
    tmp_path: Path,
) -> None:
    """A prepared row is on disk before any lifecycle listener can observe it."""

    registry = LibraryIngestJobRegistry()
    store = LibraryIngestJobsDB(tmp_path / "prepared.sqlite")
    registry.attach_store(store)
    observations: list[tuple[str, str]] = []
    registry.add_listener(
        lambda: observations.extend(
            (row["job_id"], row["state"]) for row in store.all_jobs()
        )
    )

    job = registry.submit(
        source_path="https://example.invalid/evidence",
        origin="server",
        research_source_operation_id="operation-durable-prepare",
        require_persisted=True,
    )

    assert observations == [(job.job_id, "queued")]
    assert store.all_jobs()[0]["research_source_operation_id"] == (
        "operation-durable-prepare"
    )
    store.close()
    restarted = LibraryIngestJobsDB(tmp_path / "prepared.sqlite")
    try:
        assert restarted.all_jobs()[0]["state"] == "queued"
    finally:
        restarted.close()


def test_required_persisted_submit_without_store_mutates_no_registry() -> None:
    """Research preparation cannot degrade to an in-memory-only queue row."""

    registry = LibraryIngestJobRegistry()

    with pytest.raises(RuntimeError, match="persistence store"):
        registry.submit(
            source_path="https://example.invalid/evidence",
            origin="server",
            research_source_operation_id="operation-no-store",
            require_persisted=True,
        )

    assert registry.jobs() == ()


@pytest.mark.parametrize("settlement", ["cancelled", "failed"])
def test_required_prepared_settlement_is_durable_before_terminal_listener(
    tmp_path: Path,
    settlement: str,
) -> None:
    """Link/dispatch failures cannot leave a restartable queued row on disk."""

    registry = LibraryIngestJobRegistry()
    store = LibraryIngestJobsDB(tmp_path / f"prepared-{settlement}.sqlite")
    registry.attach_store(store)
    job = registry.submit(
        source_path="https://example.invalid/evidence",
        origin="server",
        research_source_operation_id=f"operation-{settlement}",
        require_persisted=True,
    )
    observations: list[str] = []
    registry.add_listener(lambda: observations.append(store.all_jobs()[0]["state"]))

    if settlement == "cancelled":
        registry.mark_cancelled(
            job.job_id,
            reason="Research source operation could not be linked.",
            require_persisted=True,
        )
    else:
        registry.mark_failed(
            job.job_id,
            error="Research catalog dispatch could not be started.",
            require_persisted=True,
        )

    assert observations == [settlement]
    assert store.all_jobs()[0]["state"] == settlement
    store.close()


@pytest.mark.parametrize("origin", ["local", "server"])
def test_research_prepare_persists_exact_authority_without_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    origin: str,
) -> None:
    """Initial Research intake stops at a durable queued row until linked."""

    app = _minimal_app(media_db="present")
    ingest_store = LibraryIngestJobsDB(tmp_path / f"prepare-{origin}.sqlite")
    app.library_ingest_jobs.attach_store(ingest_store)
    operation = ResearchSourceOperation(
        operation_id=f"operation-prepare-{origin}",
        idempotency_key=f"prepare-{origin}",
        data_source=(
            WorkspaceDataSource.LOCAL
            if origin == "local"
            else WorkspaceDataSource.SERVER
        ),
        server_profile_id="profile" if origin == "server" else "",
        principal_id="principal" if origin == "server" else "",
        workspace_id=f"workspace-{origin}",
        canonical_item_type=(
            CanonicalItemType.LOCAL_LIBRARY
            if origin == "local"
            else CanonicalItemType.SERVER_MEDIA
        ),
        desired_selected=True,
        created_at="2026-08-24T10:00:00Z",
        updated_at="2026-08-24T10:00:00Z",
    )
    monkeypatch.setattr(app, "_resolve_ingest_backend", lambda: origin)
    monkeypatch.setattr(
        app,
        "_validate_research_source_operation_authority",
        lambda operation_id, *, expected_origin: operation,
    )
    top_up = MagicMock()
    monkeypatch.setattr(app, "_top_up_ingest_parse_pool", top_up)
    remote_send = MagicMock()
    monkeypatch.setattr(app, "_send_web_clip_job", remote_send)
    source = (
        "https://example.invalid/evidence"
        if origin == "server"
        else str(tmp_path / "evidence.txt")
    )
    if origin == "local":
        Path(source).write_text("Evidence", encoding="utf-8")

    job = app.prepare_research_source_ingest_job(
        source_path=source,
        research_source_operation_id=operation.operation_id,
        required_origin=origin,
    )

    assert job.state is IngestJobState.QUEUED
    assert job.origin == origin
    assert job.research_source_operation_id == operation.operation_id
    assert job.dispatch_held is True
    assert ingest_store.all_jobs()[0]["state"] == "queued"
    assert ingest_store.all_jobs()[0]["dispatch_held"] == 1
    top_up.assert_not_called()
    remote_send.assert_not_called()
    ingest_store.close()


def test_ordinary_submission_is_never_dispatch_held(tmp_path: Path) -> None:
    """The eligibility barrier is opt-in for Research preparation only."""

    app = _minimal_app(media_db="present")
    store = LibraryIngestJobsDB(tmp_path / "ordinary.sqlite")
    app.library_ingest_jobs.attach_store(store)

    job = app.submit_library_ingest_job(source_path=str(tmp_path / "ordinary.txt"))

    assert job.dispatch_held is False
    assert store.all_jobs()[0]["dispatch_held"] == 0
    store.close()


def _pending_source_operation(
    operation_id: str, *, origin: str
) -> ResearchSourceOperation:
    return ResearchSourceOperation(
        operation_id=operation_id,
        idempotency_key=f"idempotency-{operation_id}",
        data_source=(
            WorkspaceDataSource.LOCAL
            if origin == "local"
            else WorkspaceDataSource.SERVER
        ),
        server_profile_id="profile" if origin == "server" else "",
        principal_id="principal" if origin == "server" else "",
        workspace_id=f"workspace-{origin}",
        canonical_item_type=(
            CanonicalItemType.LOCAL_LIBRARY
            if origin == "local"
            else CanonicalItemType.SERVER_MEDIA
        ),
        desired_selected=True,
        created_at="2026-08-24T10:00:00Z",
        updated_at="2026-08-24T10:00:00Z",
    )


@pytest.mark.asyncio
async def test_restart_reconciles_pending_held_job_before_dispatch(
    tmp_path: Path,
) -> None:
    """A crash between prepare and link resumes from both durable owners."""

    jobs_path = tmp_path / "jobs.sqlite"
    ingest_store = LibraryIngestJobsDB(jobs_path)
    original = LibraryIngestJobRegistry()
    original.attach_store(ingest_store)
    held = original.submit(
        source_path=str(tmp_path / "managed.txt"),
        origin="local",
        research_source_operation_id="operation-restart",
        dispatch_held=True,
        require_persisted=True,
    )
    workspace_db = WorkspaceDB(tmp_path / "workspace.sqlite")
    operation_store = ResearchSourceOperationStore(workspace_db)
    operation_store.create(
        _pending_source_operation("operation-restart", origin="local")
    )
    staging = ResearchPasteStagingStore(tmp_path / "staging")
    staged_path = staging.stage(
        "operation-restart", title="Paste", body="private staged body"
    )

    restored = LibraryIngestJobRegistry()
    plan = plan_restore(
        ingest_store.all_jobs(),
        max_persisted=100,
        now_iso="2026-08-24T10:01:00+00:00",
    )
    restored.restore(plan.jobs, plan.next_id)
    restored.attach_store(ingest_store)
    app = _minimal_app(media_db="present")
    app.library_ingest_jobs = restored
    app._library_ingest_jobs_store = ingest_store
    app.research_source_operation_store = operation_store
    app.research_paste_staging_store = staging
    dispatched: list[str] = []
    app._dispatch_research_source_catalog_job = dispatched.append  # type: ignore[method-assign]

    await app._reconcile_research_source_held_jobs(limit=1)

    operation = operation_store.get("operation-restart")
    assert operation is not None
    assert operation.catalog_status is SourceOperationStatus.IN_PROGRESS
    assert operation.ingest_job_id == held.job_id
    assert restored.get_job(held.job_id).dispatch_held is False
    assert ingest_store.all_jobs()[0]["dispatch_held"] == 0
    assert dispatched == [held.job_id]
    assert staged_path.exists()
    ingest_store.close()
    workspace_db.close()


@pytest.mark.asyncio
async def test_startup_reconcile_isolates_transient_row_and_releases_next(
    tmp_path: Path,
) -> None:
    """One unreadable operation cannot strand a later held job in the page."""

    ingest_store = LibraryIngestJobsDB(tmp_path / "jobs.sqlite")
    registry = LibraryIngestJobRegistry()
    registry.attach_store(ingest_store)
    first = registry.submit(
        source_path="/managed/first.txt",
        origin="local",
        research_source_operation_id="operation-transient",
        dispatch_held=True,
        require_persisted=True,
    )
    second = registry.submit(
        source_path="/managed/second.txt",
        origin="local",
        research_source_operation_id="operation-releasable",
        dispatch_held=True,
        require_persisted=True,
    )
    workspace_db = WorkspaceDB(tmp_path / "workspace.sqlite")
    real_store = ResearchSourceOperationStore(workspace_db)
    real_store.create(_pending_source_operation("operation-releasable", origin="local"))

    class OneTransientStore:
        def get(self, operation_id):
            if operation_id == "operation-transient":
                raise OSError("private store unavailable")
            return real_store.get(operation_id)

        def advance_stage(self, *args, **kwargs):
            return real_store.advance_stage(*args, **kwargs)

    app = _minimal_app(media_db="present")
    app.library_ingest_jobs = registry
    app._library_ingest_jobs_store = ingest_store
    app.research_source_operation_store = OneTransientStore()
    app.research_paste_staging_store = None
    dispatched: list[str] = []
    app._dispatch_research_source_catalog_job = dispatched.append  # type: ignore[method-assign]

    await app._reconcile_research_source_held_jobs(limit=2)

    assert registry.get_job(first.job_id).dispatch_held is True
    assert registry.get_job(second.job_id).dispatch_held is False
    assert dispatched == [second.job_id]
    ingest_store.close()
    workspace_db.close()


@pytest.mark.asyncio
async def test_startup_reconcile_terminal_listener_schedules_exactly_once(
    tmp_path: Path,
) -> None:
    """Release visibility plus immediate settlement cannot duplicate resume work."""

    ingest_store = LibraryIngestJobsDB(tmp_path / "jobs.sqlite")
    registry = LibraryIngestJobRegistry()
    registry.attach_store(ingest_store)
    held = registry.submit(
        source_path="/managed/source.txt",
        origin="local",
        research_source_operation_id="operation-listener-once",
        dispatch_held=True,
        require_persisted=True,
    )
    workspace_db = WorkspaceDB(tmp_path / "workspace.sqlite")
    operation_store = ResearchSourceOperationStore(workspace_db)
    operation_store.create(
        _pending_source_operation("operation-listener-once", origin="local")
    )
    app = _minimal_app(media_db="present")
    app.library_ingest_jobs = registry
    app._library_ingest_jobs_store = ingest_store
    app.research_source_operation_store = operation_store
    app.research_paste_staging_store = None
    app.research_source_association_scheduler = object()
    app._research_source_terminal_jobs_scheduled = set()
    app._research_source_restore_in_progress = False
    groups: list[str] = []

    def record_worker(awaitable, *, group: str) -> None:
        groups.append(group)
        awaitable.close()

    app.run_worker = record_worker  # type: ignore[method-assign]
    registry.add_listener(app._schedule_settled_research_source_operations)
    app._dispatch_research_source_catalog_job = (  # type: ignore[method-assign]
        lambda job_id: registry.mark_failed(job_id, error="Immediate owner failure")
    )

    await app._reconcile_research_source_held_jobs(limit=1)

    assert groups == ["research_source_association"]
    assert app._research_source_terminal_jobs_scheduled == {held.job_id}
    ingest_store.close()
    workspace_db.close()


@pytest.mark.asyncio
async def test_incompatible_held_job_cleans_staging_only_after_durable_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed cancellation keeps both the eligibility hold and paste payload."""

    ingest_store = LibraryIngestJobsDB(tmp_path / "jobs.sqlite")
    registry = LibraryIngestJobRegistry()
    registry.attach_store(ingest_store)
    held = registry.submit(
        source_path="/managed/paste.txt",
        origin="server",
        research_source_operation_id="operation-missing",
        dispatch_held=True,
        require_persisted=True,
    )
    staging = ResearchPasteStagingStore(tmp_path / "staging")
    artifact = staging.stage("operation-missing", title="Paste", body="private")
    app = _minimal_app(media_db="present")
    app.library_ingest_jobs = registry
    app._library_ingest_jobs_store = ingest_store
    app.research_source_operation_store = SimpleNamespace(
        get=lambda _operation_id: None
    )
    app.research_paste_staging_store = staging
    monkeypatch.setattr(
        app,
        "_cancel_research_source_prepared_job",
        MagicMock(side_effect=OSError("cancel store unavailable")),
    )

    await app._reconcile_research_source_held_jobs(limit=1)

    assert registry.get_job(held.job_id).state is IngestJobState.QUEUED
    assert registry.get_job(held.job_id).dispatch_held is True
    assert artifact.exists()

    monkeypatch.setattr(
        app,
        "_cancel_research_source_prepared_job",
        TldwCli._cancel_research_source_prepared_job.__get__(app, TldwCli),
    )
    await app._reconcile_research_source_held_jobs(limit=1)

    assert registry.get_job(held.job_id).state is IngestJobState.CANCELLED
    assert not artifact.exists()
    ingest_store.close()


def test_research_local_classification_warning_omits_managed_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unexpected classifier failure logs lineage, never a private staging path."""

    app = _minimal_app(media_db="present")
    managed_path = str(tmp_path / "research" / "paste" / "PRIVATE-name.txt")
    monkeypatch.setattr(
        app_module,
        "classify_ingest_source",
        MagicMock(side_effect=RuntimeError(f"classifier broke for {managed_path!r}")),
    )
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        app._prepare_library_ingest_job_admitted(
            source_path=managed_path,
            ingest_options={},
            title="Paste",
            author="",
            keywords=(),
            perform_analysis=False,
            chunk_enabled=False,
            chunk_size=DEFAULT_CHUNK_SIZE,
            batch_id=None,
            backend="local",
            research_source_operation_id="operation-private-log",
            require_persisted=False,
        )
    finally:
        logger.remove(sink)

    rendered = "".join(messages)
    assert managed_path not in rendered
    assert "PRIVATE-name.txt" not in rendered
    assert "operation-private-log" in rendered


def test_dispatch_failure_settlement_does_not_renotify_an_already_terminal_job(
    tmp_path: Path,
) -> None:
    """A dispatcher that settled before raising must not schedule twice."""

    app = _minimal_app(media_db="present")
    store = LibraryIngestJobsDB(tmp_path / "terminal-dispatch.sqlite")
    app.library_ingest_jobs.attach_store(store)
    job = app.library_ingest_jobs.submit(
        source_path="https://example.invalid/evidence",
        origin="server",
        research_source_operation_id="operation-terminal-dispatch",
        require_persisted=True,
    )
    app.library_ingest_jobs.mark_failed(job.job_id, error="Owner already settled")
    notifications: list[IngestJobState] = []
    app.library_ingest_jobs.add_listener(
        lambda: notifications.append(app.library_ingest_jobs.get_job(job.job_id).state)
    )

    settled = app._fail_research_source_prepared_job(job.job_id)

    assert settled.state is IngestJobState.FAILED
    assert settled.error == "Owner already settled"
    assert notifications == []
    store.close()


def _minimal_stt_app() -> TldwCli:
    app = object.__new__(TldwCli)
    app.library_ingest_jobs = LibraryIngestJobRegistry()
    app._ingest_shutdown = False
    app._local_stt_executor_lock = threading.RLock()
    app._local_stt_executor = None
    app._local_stt_dispatch_coordinator = None
    app._parakeet_source_service = None
    app._parakeet_source_registry_listener = None
    app._ingest_local_stt_jobs = {}
    app._ingest_parse_pool = None
    app._ingest_parse_pool_stop_event = None
    app._marshal_local_stt_call = lambda callback: callback()  # type: ignore[method-assign]
    app._top_up_ingest_parse_pool = lambda: None  # type: ignore[method-assign]
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("job_state", "catalog_status", "expected_delete"),
    [
        ("done", SourceOperationStatus.SUCCEEDED, True),
        ("cancelled", SourceOperationStatus.FAILED, True),
        ("failed", SourceOperationStatus.FAILED, False),
    ],
)
async def test_terminal_research_job_cleans_paste_only_after_success_or_cancel(
    job_state: str,
    catalog_status: SourceOperationStatus,
    expected_delete: bool,
) -> None:
    app = object.__new__(TldwCli)
    scheduler = SimpleNamespace(
        resume=AsyncMock(return_value=SimpleNamespace(catalog_status=catalog_status))
    )
    staging = SimpleNamespace(delete=MagicMock(return_value=True))
    app.research_source_association_scheduler = scheduler
    app.research_paste_staging_store = staging
    app._research_source_terminal_jobs_scheduled = {"job-1"}
    app.library_ingest_jobs = SimpleNamespace(
        get_job=lambda _job_id: SimpleNamespace(state=SimpleNamespace(value=job_state))
    )

    await app._resume_settled_research_source_operation("job-1", "operation-1")

    scheduler.resume.assert_awaited_once_with("operation-1")
    assert staging.delete.called is expected_delete
    if expected_delete:
        staging.delete.assert_called_once_with("operation-1")


def test_local_stt_accessors_share_one_executor_and_coordinator_without_deadlock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _minimal_stt_app()
    created: list[object] = []

    def create_executor() -> object:
        executor = object()
        created.append(executor)
        return executor

    monkeypatch.setattr(app, "_create_local_stt_executor", create_executor)
    barrier = threading.Barrier(8)

    def resolve(index: int) -> tuple[object, object]:
        barrier.wait(timeout=2.0)
        if index % 2:
            coordinator = app._ensure_local_stt_dispatch_coordinator()
            executor = app._ensure_local_stt_executor()
        else:
            executor = app._ensure_local_stt_executor()
            coordinator = app._ensure_local_stt_dispatch_coordinator()
        return executor, coordinator

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(resolve, index) for index in range(8)]
        resolved = [future.result(timeout=2.0) for future in futures]

    executors = {id(executor) for executor, _coordinator in resolved}
    coordinators = {id(coordinator) for _executor, coordinator in resolved}
    assert len(created) == 1
    assert len(executors) == 1
    assert len(coordinators) == 1
    assert resolved[0][1]._executor is resolved[0][0]


def test_recycle_idle_local_stt_reference_uses_existing_executor_without_creating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _minimal_stt_app()
    create_executor = MagicMock(side_effect=AssertionError("must stay lazy"))
    monkeypatch.setattr(app, "_create_local_stt_executor", create_executor)
    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")

    assert app._recycle_idle_local_stt_reference(reference) is False
    create_executor.assert_not_called()

    executor = MagicMock()
    executor.recycle_idle_managed_reference.return_value = True
    app._local_stt_executor = executor

    assert app._recycle_idle_local_stt_reference(reference) is True
    executor.recycle_idle_managed_reference.assert_called_once_with(
        ("parakeet-v2", "immutable-revision", "int8")
    )
    create_executor.assert_not_called()


def test_console_dictation_factory_injects_app_owned_coordinator() -> None:
    app = _minimal_stt_app()
    coordinator = object()
    source_service = object()
    source_threads: list[int] = []
    app._ensure_local_stt_dispatch_coordinator = lambda: coordinator  # type: ignore[method-assign]

    def _source_service() -> object:
        source_threads.append(threading.get_ident())
        return source_service

    app._ensure_parakeet_source_service = _source_service  # type: ignore[method-assign]

    class _FakeTranscriptionService:
        def __init__(
            self,
            *,
            local_stt_dispatcher: object,
            parakeet_source_service: object,
        ) -> None:
            self.local_stt_dispatcher = local_stt_dispatcher
            self.parakeet_source_service = parakeet_source_service

    class _FakeLazyService:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    with (
        patch(
            "tldw_chatbook.Audio.dictation_service_lazy.LazyLiveDictationService",
            _FakeLazyService,
        ),
        patch(
            "tldw_chatbook.Local_Ingestion.transcription_service.TranscriptionService",
            _FakeTranscriptionService,
        ),
    ):
        service = app._create_console_dictation_service(
            transcription_provider="parakeet-onnx",
            language="en",
        )

    assert service.kwargs["transcription_provider"] == "parakeet-onnx"
    assert service.kwargs["language"] == "en"
    assert source_threads == [threading.get_ident()]
    transcription = service.kwargs["transcription_service_factory"]()
    assert transcription.local_stt_dispatcher is coordinator
    assert transcription.parakeet_source_service is source_service
    assert source_threads == [threading.get_ident()]


@pytest.mark.asyncio
async def test_console_dictation_factory_marshals_source_setup_to_app_thread() -> None:
    events: list[tuple[str, int]] = []

    class _Registry(LibraryIngestJobRegistry):
        def add_listener(self, callback) -> None:
            events.append(("listener", threading.get_ident()))
            super().add_listener(callback)

        def jobs(self) -> tuple[LibraryIngestJob, ...]:
            events.append(("initial-read", threading.get_ident()))
            return super().jobs()

    class _SourceService:
        def release_scopes_except(self, _active: set[str]) -> None:
            events.append(("initial-sync", threading.get_ident()))

    class _Host(app_module.LibraryIngestQueueMixin, App[None]):
        def __init__(self) -> None:
            super().__init__()
            self.library_ingest_jobs = _Registry()
            self._ingest_shutdown = False
            self._local_stt_executor_lock = threading.RLock()
            self._parakeet_source_service = None
            self._parakeet_source_registry_listener = None

        def _create_parakeet_source_service(self) -> _SourceService:
            events.append(("create", threading.get_ident()))
            return _SourceService()

    class _FakeLazyService:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    host = _Host()
    with patch(
        "tldw_chatbook.Audio.dictation_service_lazy.LazyLiveDictationService",
        _FakeLazyService,
    ):
        async with host.run_test():
            service = await asyncio.to_thread(
                host._create_console_dictation_service,
                transcription_provider="parakeet-onnx",
                language="en",
            )

    assert service.kwargs["language"] == "en"
    assert events == [
        ("create", host._thread_id),
        ("listener", host._thread_id),
        ("initial-read", host._thread_id),
        ("initial-sync", host._thread_id),
    ]


def test_parakeet_source_accessor_constructs_one_service_and_one_listener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _minimal_stt_app()
    created: list[object] = []

    def create_service() -> object:
        service = SimpleNamespace(release_scopes_except=lambda _active: None)
        created.append(service)
        return service

    monkeypatch.setattr(
        app, "_create_parakeet_source_service", create_service, raising=False
    )
    barrier = threading.Barrier(8)

    def resolve() -> object:
        barrier.wait(timeout=2.0)
        return app._ensure_parakeet_source_service()

    with ThreadPoolExecutor(max_workers=8) as pool:
        resolved = [
            future.result(timeout=2.0)
            for future in [pool.submit(resolve) for _ in range(8)]
        ]

    assert len(created) == 1
    assert {id(service) for service in resolved} == {id(created[0])}
    assert app.library_ingest_jobs._listeners == [  # noqa: SLF001
        app._parakeet_source_registry_listener
    ]


def _parakeet_job(
    *,
    job_id: str = "ingest-job-1",
    batch_id: str | None = None,
    scope_id: str | None = None,
) -> LibraryIngestJob:
    audio_options: dict[str, object] = {
        "transcription_provider": "parakeet-onnx",
        "transcription_model_dir": "/user/parakeet",
    }
    if scope_id is not None:
        audio_options["transcription_external_scope_id"] = scope_id
    return LibraryIngestJob(
        job_id=job_id,
        source_path="/tmp/audio.wav",
        detected_type="audio",
        ingest_options={"audio_video": audio_options},
        batch_id=batch_id,
    )


def test_build_local_stt_dispatch_uses_authoritative_source_service_and_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey

    app = _minimal_stt_app()
    local_source = object()
    dispatch = SimpleNamespace(
        identity=object(),
        local_source=local_source,
        managed_store_root=Path("/managed"),
        managed_artifact_ref=None,
        managed_dependency_refs=(("silero-vad", "rev", "onnx"),),
        option_updates={"transcription_model_dir": "/user/parakeet"},
    )
    calls: list[dict[str, object]] = []

    class _SourceService:
        def resolve(self, key: object, **kwargs: object) -> object:
            calls.append({"key": key, **kwargs})
            return dispatch

    app._ensure_parakeet_source_service = lambda: _SourceService()  # type: ignore[method-assign]
    job = _parakeet_job(scope_id="preflight-scope")
    options = {
        "transcription_provider": "parakeet-onnx",
        "transcription_model": "nemo-parakeet-tdt-0.6b-v3",
        "transcription_precision": "f32",
        "transcription_model_dir": "/user/parakeet",
    }
    monkeypatch.setattr(
        "tldw_chatbook.STT.parakeet_dispatch.resolve_parakeet_dispatch",
        lambda **_kwargs: pytest.fail("legacy resolver must not be consulted"),
    )

    resolved = app._build_local_stt_dispatch(job, options)

    assert calls == [
        {
            "key": ParakeetSourceKey.V3_F32,
            "override": "/user/parakeet",
            "scope_id": "preflight-scope",
        }
    ]
    assert resolved["local_source"] is local_source
    assert resolved["managed_dependency_refs"] == (("silero-vad", "rev", "onnx"),)


def test_folder_siblings_reuse_one_real_verified_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.STT.parakeet_external as parakeet_external
    from tldw_chatbook.Model_Artifacts.service import (
        ArtifactDescriptor,
        ArtifactFile,
        ArtifactFormat,
        ArtifactRef,
        ArtifactRole,
        ProvenanceClass,
    )
    from tldw_chatbook.STT.parakeet_external import ExternalParakeetVerifier
    from tldw_chatbook.STT.parakeet_sources import ParakeetSourceService

    app = _minimal_stt_app()
    payload = b"verified-model"
    root = tmp_path / "external-parakeet"
    root.mkdir()
    model_path = root / "model.onnx"
    model_path.write_bytes(payload)
    descriptor = ArtifactDescriptor(
        reference=ArtifactRef("parakeet-v2", "test-revision", "int8"),
        model_id="nemo-parakeet-tdt-0.6b-v2",
        role=ArtifactRole.ROOT,
        format=ArtifactFormat.ONNX,
        consumer="stt",
        model_family="parakeet",
        upstream_repository="example/parakeet-v2",
        upstream_revision="test-revision",
        source_url="https://example.invalid/model.onnx",
        precision="int8",
        expected_installed_bytes=len(payload),
        license_id="cc-by-4.0",
        license_url="https://example.invalid/license",
        usage_notice="test",
        runtime_name="onnx-asr",
        runtime_version_constraint="==0.12.0",
        supported_os=("linux", "darwin", "windows"),
        supported_architectures=("x86-64", "arm64"),
        provenance=(ProvenanceClass.CHATBOOK_CURATED,),
        files=(
            ArtifactFile(
                "model.onnx",
                len(payload),
                hashlib.sha256(payload).hexdigest(),
            ),
        ),
    )
    source_service = ParakeetSourceService(
        verifier=ExternalParakeetVerifier(),
        read_setting=lambda _section, _key, default: default,
        write_settings=lambda _values: True,
        descriptor_for=lambda _model, _precision: descriptor,
        active_managed=lambda _model, _precision: None,
        dispatch_resolver=lambda **_kwargs: pytest.fail(
            "external source must not use managed fallback"
        ),
        vad_ready=lambda: True,
        managed_service=SimpleNamespace(
            artifacts_path=tmp_path / "managed" / "artifacts"
        ),
    )
    app._ensure_parakeet_source_service = lambda: source_service  # type: ignore[method-assign]
    options = {
        "transcription_provider": "parakeet-onnx",
        "transcription_model": "nemo-parakeet-tdt-0.6b-v2",
        "transcription_precision": "int8",
        "transcription_model_dir": str(root),
    }
    real_open = parakeet_external.os.open
    open_count = 0

    def counted_open(path: Path, flags: int) -> int:
        nonlocal open_count
        if Path(path) == model_path:
            open_count += 1
        return real_open(path, flags)

    monkeypatch.setattr(parakeet_external.os, "open", counted_open)
    try:
        first = app._build_local_stt_dispatch(
            _parakeet_job(job_id="one", batch_id="folder-batch"),
            dict(options),
        )
        second = app._build_local_stt_dispatch(
            _parakeet_job(job_id="two", batch_id="folder-batch"),
            dict(options),
        )
    finally:
        source_service.close()

    assert open_count == 1
    assert first["local_source"] is second["local_source"]


def test_library_submission_forwards_exact_dependency_refs_to_coordinator() -> None:
    app = _minimal_stt_app()
    dependency_refs = (("silero-vad", "rev", "onnx"),)
    dispatch = {
        "attempt_id": "job-1-attempt-1",
        "identity": object(),
        "local_source": object(),
        "managed_store_root": Path("/managed"),
        "managed_artifact_ref": None,
        "managed_dependency_refs": dependency_refs,
    }
    calls: list[dict[str, object]] = []

    class _Coordinator:
        def submit_library(self, **kwargs: object) -> int:
            calls.append(kwargs)
            return 3

    app._build_local_stt_dispatch = lambda _job, _options: dispatch  # type: ignore[method-assign]
    app._ensure_local_stt_dispatch_coordinator = lambda: _Coordinator()  # type: ignore[method-assign]
    app._marshal_local_stt_call = lambda callback, *args: callback(*args)  # type: ignore[method-assign]
    app._on_ingest_local_stt_submitted = lambda *_args: None  # type: ignore[method-assign]
    job = _parakeet_job(job_id="job-1")
    app._ingest_local_stt_jobs[job.job_id] = (0, "job-1-attempt-1")

    app._dispatch_local_stt_job(job, {}, "job-1-attempt-1")

    assert calls[0]["managed_dependency_refs"] == dependency_refs


def test_parakeet_submission_registers_source_listener_before_dispatch_thread() -> None:
    app = _minimal_stt_app()
    caller_thread = threading.get_ident()
    source_threads: list[int] = []
    dispatched = threading.Event()
    app._ensure_parakeet_source_service = lambda: source_threads.append(  # type: ignore[method-assign]
        threading.get_ident()
    )
    app._dispatch_local_stt_job = lambda *_args: dispatched.set()  # type: ignore[method-assign]
    job = _parakeet_job()

    app._submit_local_stt_job(
        job,
        {"transcription_provider": "parakeet-onnx"},
    )

    assert dispatched.wait(2.0)
    assert source_threads == [caller_thread]


def test_preferred_source_failure_is_path_private_and_does_not_fall_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.STT.parakeet_sources import (
        ParakeetSourceError,
        ParakeetSourceErrorCode,
    )

    app = _minimal_stt_app()
    selected_path = "/private/user/parakeet"

    class _SourceService:
        def resolve(self, _key: object, **_kwargs: object) -> object:
            raise ParakeetSourceError(ParakeetSourceErrorCode.INVALID_SELECTION)

    app._ensure_parakeet_source_service = lambda: _SourceService()  # type: ignore[method-assign]
    job = app.library_ingest_jobs.submit(
        source_path="/tmp/audio.wav",
        detected_type="audio",
        ingest_options={
            "audio_video": {
                "transcription_provider": "parakeet-onnx",
                "transcription_model_dir": selected_path,
            }
        },
    )
    options = {
        "transcription_provider": "parakeet-onnx",
        "transcription_model": "nemo-parakeet-tdt-0.6b-v2",
        "transcription_precision": "int8",
        "transcription_model_dir": selected_path,
    }
    monkeypatch.setattr(
        "tldw_chatbook.STT.parakeet_dispatch.resolve_parakeet_dispatch",
        lambda **_kwargs: pytest.fail("fallback resolver must not be consulted"),
    )
    logged: list[str] = []
    monkeypatch.setattr(
        "tldw_chatbook.app.logger.error", lambda message: logged.append(str(message))
    )
    attempt_id = f"{job.job_id}-attempt-1"
    app._ingest_local_stt_jobs[job.job_id] = (0, attempt_id)
    app._marshal_local_stt_call = lambda callback, *args: callback(*args)  # type: ignore[method-assign]

    app._dispatch_local_stt_job(job, options, attempt_id)

    failed = app.library_ingest_jobs.get_job(job.job_id)
    assert failed is not None
    assert failed.state is IngestJobState.FAILED
    assert failed.error_detail is not None
    assert failed.error_detail["category"] == "stt_failure"
    assert failed.error_detail["code"] == "artifact_incompatible"
    assert failed.error_detail["actions"] == ["retry_faster_whisper"]
    assert selected_path not in str(failed.error_detail)
    assert selected_path not in "".join(logged)


def test_headless_missing_vad_is_model_not_installed_without_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.STT.parakeet_sources import (
        ParakeetSourceError,
        ParakeetSourceErrorCode,
    )

    app = _minimal_stt_app()
    selected_path = "/private/user/parakeet"

    class _SourceService:
        def resolve(self, _key: object, **_kwargs: object) -> object:
            raise ParakeetSourceError(ParakeetSourceErrorCode.VAD_UNAVAILABLE)

    app._ensure_parakeet_source_service = lambda: _SourceService()  # type: ignore[method-assign]
    job = app.library_ingest_jobs.submit(
        source_path="/tmp/audio.wav",
        detected_type="audio",
        ingest_options={
            "audio_video": {
                "transcription_provider": "parakeet-onnx",
                "transcription_model_dir": selected_path,
            }
        },
    )
    options = {
        "transcription_provider": "parakeet-onnx",
        "transcription_model": "nemo-parakeet-tdt-0.6b-v2",
        "transcription_precision": "int8",
        "transcription_model_dir": selected_path,
    }
    monkeypatch.setattr(
        "tldw_chatbook.Local_Ingestion.parakeet_v2_artifact.run_parakeet_vad_preflight",
        lambda **_kwargs: pytest.fail("headless submission must not preflight"),
    )
    monkeypatch.setattr(
        "tldw_chatbook.Local_Ingestion.parakeet_v2_artifact.run_parakeet_vad_provision",
        lambda *_args, **_kwargs: pytest.fail("headless submission must not download"),
    )
    attempt_id = f"{job.job_id}-attempt-1"
    app._ingest_local_stt_jobs[job.job_id] = (0, attempt_id)
    app._marshal_local_stt_call = lambda callback, *args: callback(*args)  # type: ignore[method-assign]

    app._dispatch_local_stt_job(job, options, attempt_id)

    failed = app.library_ingest_jobs.get_job(job.job_id)
    assert failed is not None
    assert failed.state is IngestJobState.FAILED
    assert failed.error_detail is not None
    assert failed.error_detail["code"] == "model_not_installed"
    assert failed.error_detail["actions"] == ["retry_faster_whisper"]
    assert selected_path not in str(failed.error_detail)


def test_folder_and_retry_context_keep_one_external_scope(tmp_path: Path) -> None:
    app = _minimal_app(media_db=object())
    folder = tmp_path / "batch"
    folder.mkdir()
    for name in ("one.wav", "two.wav"):
        (folder / name).write_bytes(b"audio")
    scope_id = "library-external-batch-scope"
    model_dir = str(tmp_path / "user-owned-parakeet")
    ingest_options = {
        "audio_video": {
            "transcription_provider": "parakeet-onnx",
            "transcription_model_dir": model_dir,
            "transcription_external_scope_id": scope_id,
            "language": "en",
        }
    }

    app.submit_library_ingest_job(
        source_path=str(folder),
        ingest_options=ingest_options,
    )

    siblings = app.library_ingest_jobs.jobs()
    assert len(siblings) == 2
    assert len({job.batch_id for job in siblings}) == 1
    for job in siblings:
        options = app._ingest_job_options(job)
        assert options["transcription_model_dir"] == model_dir
        assert options["transcription_context"]["external_scope_id"] == scope_id

    failed = app.library_ingest_jobs.mark_failed(siblings[0].job_id, error="failed")
    assert failed is not None
    retry = app.retry_library_ingest_job(failed.job_id)
    assert retry is not None
    retry_options = app._ingest_job_options(retry)
    assert retry_options["transcription_model_dir"] == model_dir
    assert retry_options["transcription_context"]["external_scope_id"] == scope_id


class _ScopeTrackingSourceService:
    def __init__(self, events: list[str] | None = None) -> None:
        self.observed: set[str] = set()
        self.released: list[str] = []
        self.active_snapshots: list[set[str]] = []
        self.events = events

    def release_scopes_except(self, active_scope_ids: set[str]) -> None:
        active = set(active_scope_ids)
        self.active_snapshots.append(active)
        self.released.extend(sorted(self.observed - active))
        self.observed = active

    def close(self) -> None:
        if self.events is not None:
            self.events.append("source")


def test_registry_releases_batch_scope_only_after_last_sibling_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _minimal_stt_app()
    service = _ScopeTrackingSourceService()
    monkeypatch.setattr(
        app, "_create_parakeet_source_service", lambda: service, raising=False
    )
    app._ensure_parakeet_source_service()
    options = {
        "audio_video": {
            "transcription_external_scope_id": "folder-preflight",
        }
    }
    first = app.library_ingest_jobs.submit(
        source_path="/tmp/one.wav",
        detected_type="audio",
        ingest_options=options,
        batch_id="folder-batch",
    )
    second = app.library_ingest_jobs.submit(
        source_path="/tmp/two.wav",
        detected_type="audio",
        ingest_options=options,
        batch_id="folder-batch",
    )

    app.library_ingest_jobs.mark_failed(first.job_id, error="failed")

    assert "folder-preflight" not in service.released
    assert service.active_snapshots[-1] == {"folder-preflight"}

    app.library_ingest_jobs.mark_cancelled(second.job_id, reason="cancelled")

    assert service.released == ["folder-preflight"]
    assert service.active_snapshots[-1] == set()


def test_folder_submission_keeps_scope_while_terminal_siblings_are_constructed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _minimal_stt_app()
    app.media_db = None
    app._resolve_ingest_backend = lambda: "local"  # type: ignore[method-assign]
    release_at_job_counts: list[int] = []

    class _ConstructionTrackingService(_ScopeTrackingSourceService):
        def release_scopes_except(self, active_scope_ids: set[str]) -> None:
            before = list(self.released)
            super().release_scopes_except(active_scope_ids)
            if self.released != before:
                release_at_job_counts.append(len(app.library_ingest_jobs.jobs()))

    service = _ConstructionTrackingService()
    monkeypatch.setattr(
        app, "_create_parakeet_source_service", lambda: service, raising=False
    )
    app._ensure_parakeet_source_service()
    scope_id = "folder-construction-scope"
    service.observed.add(scope_id)
    folder = tmp_path / "batch"
    folder.mkdir()
    for name in ("one.wav", "two.wav"):
        (folder / name).write_bytes(b"audio")

    app.submit_library_ingest_job(
        source_path=str(folder),
        ingest_options={
            "audio_video": {
                "transcription_provider": "parakeet-onnx",
                "transcription_model_dir": str(tmp_path / "external"),
                "transcription_external_scope_id": scope_id,
            }
        },
    )

    jobs = app.library_ingest_jobs.jobs()
    assert len(jobs) == 2
    assert all(job.state is IngestJobState.FAILED for job in jobs)
    assert release_at_job_counts == [2]
    assert service.released == [scope_id]


def test_registry_scope_falls_back_to_batch_then_job_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _minimal_stt_app()
    service = _ScopeTrackingSourceService()
    monkeypatch.setattr(
        app, "_create_parakeet_source_service", lambda: service, raising=False
    )
    app._ensure_parakeet_source_service()

    batched = app.library_ingest_jobs.submit(
        source_path="/tmp/one.wav",
        detected_type="audio",
        batch_id="headless-batch",
    )
    assert service.active_snapshots[-1] == {"headless-batch"}
    app.library_ingest_jobs.mark_cancelled(batched.job_id)
    assert service.released == ["headless-batch"]

    single = app.library_ingest_jobs.submit(
        source_path="/tmp/two.wav",
        detected_type="audio",
    )
    assert service.active_snapshots[-1] == {single.job_id}
    app.library_ingest_jobs.mark_cancelled(single.job_id)
    assert service.released[-1] == single.job_id


def test_unrelated_registry_mutation_does_not_release_unobserved_pre_enqueue_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _minimal_stt_app()
    service = _ScopeTrackingSourceService()
    monkeypatch.setattr(
        app, "_create_parakeet_source_service", lambda: service, raising=False
    )
    app._ensure_parakeet_source_service()

    unrelated = app.library_ingest_jobs.submit(source_path="/tmp/note.txt")

    assert unrelated.job_id in service.observed
    assert "pending-preflight" not in service.released

    app.library_ingest_jobs.mark_cancelled(unrelated.job_id)

    assert unrelated.job_id in service.released
    assert "pending-preflight" not in service.released


def test_shutdown_detaches_and_closes_every_resource_off_thread_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _minimal_stt_app()
    events: list[str] = []
    source_started = threading.Event()
    release_source = threading.Event()

    class _Source(_ScopeTrackingSourceService):
        def close(self) -> None:
            events.append("source")
            source_started.set()
            assert release_source.wait(timeout=2.0)

    service = _Source(events)
    monkeypatch.setattr(
        app, "_create_parakeet_source_service", lambda: service, raising=False
    )
    app._ensure_parakeet_source_service()

    class _Coordinator:
        def close(self) -> None:
            events.append("coordinator")

    class _Executor:
        def close(self) -> None:
            events.append("executor")

    class _Pool:
        def terminate(self) -> None:
            events.append("pool.terminate")

        def join(self) -> None:
            events.append("pool.join")

    app._local_stt_dispatch_coordinator = _Coordinator()
    app._local_stt_executor = _Executor()
    app._ingest_parse_pool = _Pool()

    with ThreadPoolExecutor(max_workers=1) as callers:
        pending = callers.submit(app._shutdown_ingest_parse_pool)
        assert source_started.wait(timeout=2.0)
        try:
            teardown = pending.result(timeout=0.2)
            assert app._parakeet_source_service is None
            assert app._parakeet_source_registry_listener is None
            assert app.library_ingest_jobs._listeners == []  # noqa: SLF001
            assert events == ["source"]
        finally:
            release_source.set()
    assert teardown is not None
    teardown.join(timeout=2.0)

    assert not teardown.is_alive()
    assert events == [
        "source",
        "coordinator",
        "executor",
        "pool.terminate",
        "pool.join",
    ]


@pytest.mark.parametrize("resource", ["source", "coordinator"])
def test_shutdown_starts_teardown_for_source_or_coordinator_only(
    resource: str,
) -> None:
    app = _minimal_stt_app()
    closed: list[str] = []
    owner = SimpleNamespace(close=lambda: closed.append(resource))
    if resource == "source":
        app._parakeet_source_service = owner
    else:
        app._local_stt_dispatch_coordinator = owner

    teardown = app._shutdown_ingest_parse_pool()

    assert teardown is not None
    teardown.join(timeout=2.0)
    assert not teardown.is_alive()
    assert closed == [resource]
    assert app._shutdown_ingest_parse_pool() is None
    assert closed == [resource]


def _make_job(
    *,
    source_path: str = "/tmp/test.txt",
    ingest_options: dict[str, Any] | None = None,
    perform_analysis: bool = False,
    chunk_enabled: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    origin: str = "local",
) -> LibraryIngestJob:
    """Build a minimal LibraryIngestJob for _ingest_job_options tests."""
    return LibraryIngestJob(
        job_id="ingest-job-test",
        source_path=source_path,
        perform_analysis=perform_analysis,
        chunk_enabled=chunk_enabled,
        chunk_size=chunk_size,
        ingest_options=ingest_options or {},
        origin=origin,
    )


def _direct_failed_attempt() -> dict[str, object]:
    return {
        "attempt_id": "attempt-1",
        "batch_id": None,
        "job_id": "ingest-job-1",
        "provider_id": "transcribe-cpp",
        "model_id": "local-gguf:whisper",
        "artifact_root": None,
        "artifact_dependencies": [],
        "precision": "native",
        "requested_device": "auto",
        "effective_device": None,
        "requested_language": "en",
        "effective_language": "en",
        "detected_language": None,
        "task": "transcribe",
        "error_code": "inference_failed",
    }


class TestIngestJobOptions:
    """Coverage for TldwCli._ingest_job_options."""

    def test_empty_ingest_options_uses_deprecated_job_fields(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.txt",
            perform_analysis=True,
            chunk_enabled=True,
            chunk_size=1234,
        )
        options = app._ingest_job_options(job)

        assert options["title"] is None
        assert options["author"] is None
        assert options["keywords"] is None
        assert options["perform_analysis"] is True
        # task-3301: overlap default is the generic schema default (100, the
        # value the UI shows), not the old hardcoded 50; ``max_size`` mirrors
        # ``size`` because ``improved_chunking_process`` reads that spelling;
        # no ``method`` is forced -- each consumer applies its own default.
        assert options["chunk_options"] == {
            "size": 1234,
            "max_size": 1234,
            "overlap": 100,
        }

    def test_generic_ingest_options_override_deprecated_fields(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.txt",
            perform_analysis=False,
            chunk_enabled=False,
            chunk_size=DEFAULT_CHUNK_SIZE,
            ingest_options={
                "generic": {
                    "analyze": True,
                    "chunk": True,
                    "chunk_size": 2048,
                    "chunk_overlap": 100,
                }
            },
        )
        options = app._ingest_job_options(job)

        assert options["perform_analysis"] is True
        assert options["chunk_options"] == {
            "size": 2048,
            "max_size": 2048,
            "overlap": 100,
        }

    def test_pdf_group_options(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.pdf",
            ingest_options={
                "generic": {"analyze": True},
                "pdf": {
                    "pdf_engine": "docling",
                    "extract_images": True,
                    "enable_ocr": True,
                },
            },
        )
        options = app._ingest_job_options(job)

        assert options["perform_analysis"] is True
        assert options["pdf_engine"] == "docling"
        assert options["extract_images"] is True
        assert options["ocr"] is True
        assert options["page_range"] is None

    def test_pdf_group_falls_back_to_canonical_names(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.pdf",
            ingest_options={
                "pdf": {
                    "engine": "pymupdf",
                    "pages": "1-10",
                    "ocr": True,
                },
            },
        )
        options = app._ingest_job_options(job)

        assert options["pdf_engine"] == "pymupdf"
        assert options["page_range"] == "1-10"
        assert options["ocr"] is True

    def test_audio_video_group_options(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {
                    "transcription_provider": "parakeet-onnx",
                    "transcription_model_dir": "/models/parakeet-v2-int8",
                    "transcription_model": "base",
                    "language": "en",
                    "timestamps": False,
                    "diarization": True,
                },
            },
        )
        options = app._ingest_job_options(job)

        assert options["transcription_provider"] == "parakeet-onnx"
        assert options["transcription_model_dir"] == "/models/parakeet-v2-int8"
        assert options["transcription_model"] == "nemo-parakeet-tdt-0.6b-v2"
        assert options["language"] == "en"
        assert options["transcription_precision"] == "int8"
        assert options["transcription_local_files_only"] is True
        assert options["transcription_batch_route_resolved"] is True
        assert options["timestamps"] is False
        assert options["diarization"] is True

    def test_supported_non_english_parakeet_route_uses_v3(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp4",
            ingest_options={
                "audio_video": {
                    "transcription_provider": "parakeet-onnx",
                    "transcription_model_dir": "/models/parakeet-v3-int8",
                    "language": " DE ",
                },
            },
        )

        options = app._ingest_job_options(job)

        assert options["transcription_provider"] == "parakeet-onnx"
        assert options["transcription_model"] == "nemo-parakeet-tdt-0.6b-v3"
        assert options["transcription_model_dir"] == "/models/parakeet-v3-int8"
        assert options["language"] == "de"
        assert options["transcription_precision"] == "int8"
        assert options["transcription_local_files_only"] is True
        assert options["transcription_batch_route_resolved"] is True

    def test_explicit_parakeet_f32_is_preserved_in_worker_options(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp4",
            ingest_options={
                "audio_video": {
                    "transcription_provider": "parakeet-onnx",
                    "transcription_precision": "F32",
                    "language": "de",
                },
            },
        )

        options = app._ingest_job_options(job)

        assert options["transcription_model"] == "nemo-parakeet-tdt-0.6b-v3"
        assert options["transcription_precision"] == "f32"

    def test_parakeet_onnx_defaults_language_to_english(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {
                    "transcription_provider": "parakeet-onnx",
                },
            },
        )

        options = app._ingest_job_options(job)

        assert options["language"] == "en"
        assert options["transcription_model_dir"] is None

    @pytest.mark.parametrize(
        "provider",
        [None, "default"],
        ids=["absent-provider", "explicit-default"],
    )
    def test_semantic_default_stays_on_faster_whisper_and_drops_stale_model(
        self, provider: str | None
    ) -> None:
        app = _minimal_app()
        audio_options = {
            "transcription_model_dir": "/models/parakeet-v2-int8",
            "transcription_model": "small",
            "language": " FR ",
        }
        if provider is not None:
            audio_options["transcription_provider"] = provider
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={"audio_video": audio_options},
        )

        options = app._ingest_job_options(job)

        assert options["transcription_provider"] == "faster-whisper"
        assert options["transcription_model"] is None
        assert options["transcription_model_dir"] is None
        assert options["language"] == "fr"
        assert options["transcription_precision"] == "int8"
        assert options["transcription_local_files_only"] is True

    def test_document_group_options(self) -> None:
        """(task-3303 AC1) The document branch feeds ``process_document``."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/report.docx",
            ingest_options={
                "generic": {"chunk": True, "chunk_size": 800, "chunk_overlap": 80},
                "document": {
                    "processing_method": "docling",
                    "ocr": True,
                    "ocr_language": "de",
                },
            },
        )
        options = app._ingest_job_options(job)

        assert options["processing_method"] == "docling"
        assert options["enable_ocr"] is True
        assert options["ocr_language"] == "de"
        # The generic base group still applies to document files: analyze/
        # chunk/size travel exactly as they did when documents rode the
        # generic panel (task-3301's layering).
        assert options["chunk_options"] == {
            "size": 800,
            "max_size": 800,
            "overlap": 80,
        }

    def test_document_group_defaults_without_snapshot(self) -> None:
        app = _minimal_app()
        job = _make_job(source_path="/tmp/report.odt")

        options = app._ingest_job_options(job)

        assert options["processing_method"] == "auto"
        assert options["enable_ocr"] is False
        assert options["ocr_language"] == "en"

    def test_pdf_ocr_language_and_backend_travel(self) -> None:
        """(task-3303 AC2) OCR language/backend reach the pdf options."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.pdf",
            ingest_options={
                "pdf": {
                    "pdf_engine": "docext",
                    "ocr": True,
                    "ocr_language": "fr",
                    "ocr_backend": "tesseract",
                },
            },
        )
        options = app._ingest_job_options(job)

        assert options["pdf_engine"] == "docext"
        assert options["ocr"] is True
        assert options["ocr_language"] == "fr"
        assert options["ocr_backend"] == "tesseract"

    def test_pdf_ocr_detail_defaults(self) -> None:
        app = _minimal_app()
        job = _make_job(source_path="/tmp/test.pdf")

        options = app._ingest_job_options(job)

        assert options["ocr_language"] == "en"
        assert options["ocr_backend"] == "auto"

    def test_ebook_chapters_choice_maps_to_ebook_chapters_method(self) -> None:
        """(task-3303 AC3) The human "chapters" choice becomes the real method."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/book.epub",
            ingest_options={
                "generic": {"chunk": True, "chunk_size": 1000},
                "ebook": {"chunk_method": "chapters"},
            },
        )
        options = app._ingest_job_options(job)

        assert options["chunk_options"]["method"] == "ebook_chapters"

    def test_ebook_sentences_choice_travels_verbatim(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/book.epub",
            ingest_options={
                "generic": {"chunk": True},
                "ebook": {"chunk_method": "sentences"},
            },
        )
        options = app._ingest_job_options(job)

        assert options["chunk_options"]["method"] == "sentences"

    def test_ebook_legacy_snapshot_without_chunk_method_keeps_sentences(
        self,
    ) -> None:
        """(task-3303 xhigh review round 2, F11) A persisted job whose
        snapshot predates the ebook ``chunk_method`` field must keep the
        pre-branch chunking scheme on retry/requeue: the old builder forced
        ``method='sentences'`` for every group, so falling through to
        ``process_epub``'s chapters default silently switched a legacy
        job's scheme. Post-branch snapshots always carry the field (the
        submit-time snapshot seeds the schema default), so absence IS the
        legacy marker."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/book.epub",
            ingest_options={"generic": {"chunk": True}},
        )
        options = app._ingest_job_options(job)

        assert options["chunk_options"] is not None
        assert options["chunk_options"]["method"] == "sentences"

    def test_fresh_untouched_form_snapshot_maps_to_ebook_chapters(self) -> None:
        """(F11) A FRESH untouched submission still chunks e-books by
        chapter: the submit-time snapshot carries the schema default
        ("chapters"), which the builder maps to the chunker's real
        ``ebook_chapters`` -- distinguishing it from a legacy snapshot
        where the field is absent."""
        screen = object.__new__(LibraryScreen)
        screen._library_ingest_form = LibraryIngestFormState()
        snapshot = screen._build_ingest_options_snapshot()

        assert snapshot["ebook"]["chunk_method"] == "chapters"

        app = _minimal_app()
        job = _make_job(source_path="/tmp/book.epub", ingest_options=snapshot)
        options = app._ingest_job_options(job)

        assert options["chunk_options"]["method"] == "ebook_chapters"

    def test_ebook_method_ignored_when_chunking_off(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/book.epub",
            ingest_options={
                "generic": {"chunk": False},
                "ebook": {"chunk_method": "chapters"},
            },
        )
        options = app._ingest_job_options(job)

        assert options["chunk_options"] is None

    def test_translate_to_english_maps_to_target_language(self) -> None:
        """(task-3303 AC4) The translate toggle becomes target_language=en."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {
                    "transcription_provider": "faster-whisper",
                    "translate_to_english": True,
                },
            },
        )
        options = app._ingest_job_options(job)

        assert options["translation_target_language"] == "en"
        assert options["transcription_provider"] == "faster-whisper"

    def test_translate_under_default_provider_routes_to_faster_whisper(
        self,
    ) -> None:
        """Only faster-whisper translates; the semantic default must route
        there rather than to Parakeet when translation is requested."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {"translate_to_english": True},
            },
        )
        options = app._ingest_job_options(job)

        assert options["transcription_provider"] == "faster-whisper"
        assert options["translation_target_language"] == "en"

    def test_translate_off_sets_no_target_language(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {"translate_to_english": False},
            },
        )
        options = app._ingest_job_options(job)

        assert options["translation_target_language"] is None

    def test_explicit_target_language_wins_over_translate_checkbox(self) -> None:
        """An explicit target (retry overrides, older snapshots) stays
        authoritative; the checkbox only fills the gap."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {
                    "transcription_provider": "faster-whisper",
                    "translation_target_language": "en",
                    "translate_to_english": False,
                },
            },
        )
        options = app._ingest_job_options(job)

        assert options["translation_target_language"] == "en"

    def test_stale_translate_under_transcribe_cpp_is_ignored_not_fatal(
        self,
    ) -> None:
        """(task-3303 xhigh review round 2, F9) The translate checkbox is
        gated in the FORM to the default/faster-whisper providers
        (``enabled_when_values``), but a value checked under one provider
        and left stale after switching to transcribe-cpp used to be
        forwarded anyway -- ``resolve_batch_stt_route`` then raised
        ``BatchSTTRoutingError`` and every audio/video job in the batch
        FAILED at dispatch. The builder must consult the same schema gate
        the form does."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {
                    "transcription_provider": "transcribe-cpp",
                    "translate_to_english": True,
                },
            },
        )

        options = app._ingest_job_options(job)  # must not raise

        assert options["transcription_provider"] == "transcribe-cpp"
        assert options["translation_target_language"] is None

    def test_stale_translate_under_parakeet_is_ignored_not_fatal(self) -> None:
        """(F9) Same stale-checkbox hazard under parakeet-onnx, which
        rejects translation outright in ``_parakeet_route``."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {
                    "transcription_provider": "parakeet-onnx",
                    "language": "en",
                    "translate_to_english": True,
                },
            },
        )

        options = app._ingest_job_options(job)  # must not raise

        assert options["transcription_provider"] == "parakeet-onnx"
        assert options["translation_target_language"] is None

    def test_vad_filter_travels(self) -> None:
        """(task-3303 AC4) The VAD toggle reaches the transcription options."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={"audio_video": {"vad_filter": True}},
        )
        options = app._ingest_job_options(job)

        assert options["vad_filter"] is True

    def test_vad_filter_defaults_off(self) -> None:
        app = _minimal_app()
        job = _make_job(source_path="/tmp/test.mp3")

        options = app._ingest_job_options(job)

        assert options["vad_filter"] is False

    def test_trim_fields_travel_stripped(self) -> None:
        """(task-3306) Start/Stop trim bounds reach the worker options."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {"start_time": " 0:30 ", "end_time": "10:00"}
            },
        )
        options = app._ingest_job_options(job)

        assert options["start_time"] == "0:30"
        assert options["end_time"] == "10:00"

    def test_trim_fields_default_unbounded(self) -> None:
        """Untouched (or blank) trim inputs mean no trim at all."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={"audio_video": {"start_time": "", "end_time": "  "}},
        )
        options = app._ingest_job_options(job)

        assert options["start_time"] is None
        assert options["end_time"] is None

    def test_cookies_file_maps_to_use_cookies_and_cookies(self, tmp_path: Any) -> None:
        """(task-3306) A cookies FILE PATH travels; its presence IS the
        use_cookies flag -- there is no separate toggle to go stale.

        (xhigh review round) The path must EXIST: this test used to assert
        against an invented ``/home/u/cookies.txt``, which is precisely the
        input the option boundary now has to reject.
        """
        cookies = tmp_path / "cookies.txt"
        cookies.write_text("# Netscape HTTP Cookie File\n")
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp4",
            ingest_options={"audio_video": {"cookies_file": f"  {cookies}  "}},
        )
        options = app._ingest_job_options(job)

        assert options["use_cookies"] is True
        assert options["cookies"] == str(cookies)
        assert "cookies_problem" not in options

    def test_nonexistent_cookies_file_is_rejected_not_forwarded(
        self, tmp_path: Any
    ) -> None:
        """(xhigh review round) An unvalidated path used to be forwarded
        verbatim; ``download_video`` then tried to parse it as cookie JSON
        and logged only "Invalid cookie format", so a typo'd path looked
        like a working gated import that mysteriously failed. The option
        boundary rejects it and records a reason the job can show."""
        app = _minimal_app()
        missing = tmp_path / "nope" / "cookies.txt"
        job = _make_job(
            source_path="/tmp/test.mp4",
            ingest_options={"audio_video": {"cookies_file": str(missing)}},
        )
        options = app._ingest_job_options(job)

        assert options["use_cookies"] is False
        assert options["cookies"] is None
        assert "cookies.txt" in options["cookies_problem"]

    def test_directory_as_cookies_file_is_rejected(self, tmp_path: Any) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp4",
            ingest_options={"audio_video": {"cookies_file": str(tmp_path)}},
        )
        options = app._ingest_job_options(job)

        assert options["use_cookies"] is False
        assert options["cookies_problem"]

    def test_unsafe_cookies_path_is_rejected(self) -> None:
        """Repo security rule: file paths go through ``path_validation``.
        A shell-metacharacter path never becomes a yt-dlp argument."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp4",
            ingest_options={
                "audio_video": {"cookies_file": "/tmp/$(whoami)/cookies.txt"}
            },
        )
        options = app._ingest_job_options(job)

        assert options["use_cookies"] is False
        assert options["cookies"] is None
        assert options["cookies_problem"]

    def test_blank_cookies_file_means_no_cookies(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp4",
            ingest_options={"audio_video": {"cookies_file": "   "}},
        )
        options = app._ingest_job_options(job)

        assert options["use_cookies"] is False
        assert options["cookies"] is None

    def test_summarize_recursively_travels(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={"audio_video": {"summarize_recursively": True}},
        )
        options = app._ingest_job_options(job)

        assert options["summarize_recursively"] is True

    def test_summarize_recursively_defaults_off(self) -> None:
        app = _minimal_app()
        job = _make_job(source_path="/tmp/test.mp3")

        options = app._ingest_job_options(job)

        assert options["summarize_recursively"] is False

    def test_faster_whisper_preserves_normalized_translation_target(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {
                    "transcription_provider": "faster-whisper",
                    "transcription_model": "small",
                    "language": " JA ",
                    "target_language": " EN ",
                },
            },
        )

        options = app._ingest_job_options(job)

        assert options["transcription_model"] == "small"
        assert options["language"] == "ja"
        assert options["translation_target_language"] == "en"

    def test_transcribe_cpp_reads_dedicated_path_into_private_worker_context(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        secret_path = "/private/models/speech.gguf"
        monkeypatch.setattr(
            app_module,
            "get_cli_setting",
            lambda key, *args: (
                secret_path
                if key == "transcription.transcribe_cpp.model_path"
                else args[0]
                if args
                else None
            ),
        )
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {
                    "transcription_provider": "transcribe-cpp",
                    "language": "en",
                    "timestamps": True,
                }
            },
        )

        options = app._ingest_job_options(job)

        assert options["transcription_provider"] == "transcribe-cpp"
        assert options["transcription_model"] is None
        assert options["transcription_precision"] == "native"
        assert options["language"] == "en"
        assert options["transcription_context"] == {
            "model_path": secret_path,
            "attempt_id": "ingest-job-test-attempt-1",
            "batch_id": None,
            "job_id": "ingest-job-test",
            "retry_of_attempt_id": None,
            "retry_of_job_id": None,
            "retry_source_failure_provenance": None,
        }
        assert "transcription_model_path" not in options
        assert secret_path not in str(job.ingest_options)

    def test_untouched_exact_faster_whisper_model_uses_visible_base_default(
        self,
    ) -> None:
        screen = object.__new__(LibraryScreen)
        screen._library_ingest_form = LibraryIngestFormState()
        screen._library_ingest_form.type_options["audio_video"] = {
            "transcription_provider": "faster-whisper",
        }
        snapshot = screen._build_ingest_options_snapshot()

        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options=snapshot,
        )

        options = app._ingest_job_options(job)

        assert options["transcription_provider"] == "faster-whisper"
        assert options["transcription_model"] == "base"

    def test_explicit_empty_translation_target_does_not_fall_back_to_alias(
        self,
    ) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "audio_video": {
                    "transcription_provider": "faster-whisper",
                    "language": "ja",
                    "translation_target_language": "",
                    "target_language": "fr",
                },
            },
        )

        options = app._ingest_job_options(job)

        assert options["translation_target_language"] is None

    def test_untouched_audio_form_snapshot_resolves_closed_gate_default(self) -> None:
        provider_field = next(
            field
            for field in get_capabilities("audio_video").fields
            if field.name == "transcription_provider"
        )
        screen = object.__new__(LibraryScreen)
        screen._library_ingest_form = LibraryIngestFormState()
        snapshot = screen._build_ingest_options_snapshot()
        submitted_audio_options = snapshot.get("audio_video", {})

        assert provider_field.default == "default"
        assert submitted_audio_options.get("transcription_provider") not in {
            "parakeet-onnx",
            "faster-whisper",
        }

        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options=snapshot,
        )

        options = app._ingest_job_options(job)

        assert options["transcription_provider"] == "faster-whisper"
        assert options["transcription_model_dir"] is None
        assert options["language"] == "en"
        assert options["transcription_precision"] == "int8"
        assert options["transcription_local_files_only"] is True

    def test_ebook_group_options(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.epub",
            ingest_options={
                "ebook": {
                    "html_converter": "html2text",
                    "extract_toc": False,
                },
            },
        )
        options = app._ingest_job_options(job)

        assert options["extraction_method"] == "html2text"
        assert options["include_toc"] is False
        assert options["split_chapters"] is True

    def test_ebook_group_options_canonical_extraction_method(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.epub",
            ingest_options={
                "ebook": {
                    "extraction_method": "markdown",
                    "include_toc": True,
                },
            },
        )
        options = app._ingest_job_options(job)

        assert options["extraction_method"] == "markdown"
        assert options["include_toc"] is True

    def test_pdf_chunk_options_carry_explicit_words_method(self) -> None:
        """(task-3301/3303 xhigh review round 2, F12) The generic chunk-size
        hint promises WORDS ('words · 100–5000'), but ``process_pdf``
        setdefaults ``method='sentences'`` when none travels -- a ~10-30x
        unit lie (500 SENTENCES is roughly one chunk per document). The
        builder now makes the hint true by always sending the words
        method for the pdf group."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.pdf",
            ingest_options={"generic": {"chunk": True, "chunk_size": 500}},
        )
        options = app._ingest_job_options(job)

        assert options["chunk_options"]["method"] == "words"

    def test_audio_video_chunk_options_carry_explicit_words_method(self) -> None:
        """(F12) Same unit contract for the audio/video branch, whose
        processor defaults to sentences as well."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.mp3",
            ingest_options={"generic": {"chunk": True, "chunk_size": 500}},
        )
        options = app._ingest_job_options(job)

        assert options["chunk_options"]["method"] == "words"

    def test_pdf_chunk_size_governs_word_budget_through_processor_tail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """(F12) GOVERNANCE, not kwargs-arrival: the builder's pdf
        chunk_options drive ``process_pdf``'s REAL chunking tail (its own
        sentences-setdefault seam plus the real chunking service), and a
        size of 120 must mean ~120 WORDS per chunk. Only the
        pymupdf-backed extraction/metadata seams are stubbed (pymupdf is
        absent in this venv); the chunker is never stubbed
        (kwargs-arrival-vs-governance lesson). With the method injection
        dropped, the processor's sentences default makes this word-soup
        content ONE chunk and the test goes RED."""
        from tldw_chatbook.Local_Ingestion import PDF_Processing_Lib

        content = " ".join(f"word{i}" for i in range(600))
        monkeypatch.setattr(
            PDF_Processing_Lib,
            "pymupdf4llm_parse_pdf",
            lambda *args, **kwargs: content,
        )

        class _FakePyMuPDF:
            class FileDataError(Exception):
                pass

            class EmptyFileError(Exception):
                pass

            @staticmethod
            def open(*args, **kwargs):
                raise RuntimeError("pymupdf absent in this test venv")

        monkeypatch.setattr(PDF_Processing_Lib, "pymupdf", _FakePyMuPDF)

        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/budget.pdf",
            ingest_options={
                "generic": {"chunk": True, "chunk_size": 120, "chunk_overlap": 0}
            },
        )
        options = app._ingest_job_options(job)

        result = PDF_Processing_Lib.process_pdf(
            file_input=b"%PDF-1.4 fake bytes",
            filename="budget.pdf",
            perform_chunking=True,
            chunk_options=options["chunk_options"],
        )

        chunks = result["chunks"]
        assert chunks and len(chunks) >= 4, (
            f"size 120 produced {len(chunks or [])} chunk(s) from 600 words "
            "-- the size unit is not words"
        )
        for chunk in chunks:
            assert len(chunk["text"].split()) <= 120

    def test_type_specific_overrides_generic(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.pdf",
            ingest_options={
                "generic": {"analyze": False, "chunk_size": 100},
                "pdf": {"analyze": True, "chunk": True, "chunk_size": 999},
            },
        )
        options = app._ingest_job_options(job)

        assert options["perform_analysis"] is True
        assert options["chunk_options"]["size"] == 999

    def test_disabled_chunking_returns_none_chunk_options(self) -> None:
        app = _minimal_app()
        job = _make_job(source_path="/tmp/test.txt", chunk_enabled=False)
        options = app._ingest_job_options(job)

        assert options["chunk_options"] is None


class TestIngestJobOptionsWiring:
    """task-3301: the dead controls resolve to real option values."""

    def test_shared_generic_values_are_explicitly_projected_for_local_parser(
        self,
    ) -> None:
        """The parser receives shared form state without relying on its group."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/report.pdf",
            ingest_options={
                "generic": {
                    "analyze": True,
                    "overwrite_existing": True,
                    "custom_prompt": "Extract decisions.",
                    "system_prompt": "Be concise.",
                    "generate_embeddings": False,
                    "keep_original_file": True,
                }
            },
        )

        options = app._ingest_job_options(job)

        assert options["overwrite_existing"] is True
        assert options["custom_prompt"] == "Extract decisions."
        assert options["system_prompt"] == "Be concise."
        assert options["generate_embeddings"] is False
        assert "keep_original_file" not in options

    def test_local_parser_omits_analysis_prompts_when_analysis_is_off(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/report.pdf",
            ingest_options={
                "generic": {
                    "analyze": False,
                    "custom_prompt": "Extract decisions.",
                    "system_prompt": "Be concise.",
                }
            },
        )

        options = app._ingest_job_options(job)

        assert options["perform_analysis"] is False
        assert "custom_prompt" not in options
        assert "system_prompt" not in options

    def test_untouched_overlap_default_is_schema_default(self) -> None:
        """Local fallback overlap == the generic schema default (100), the
        value the UI displays -- it used to be a hardcoded 50."""
        from tldw_chatbook.Library.ingest_capabilities import get_capabilities

        schema_overlap = next(
            f.default
            for f in get_capabilities("generic").fields
            if f.name == "chunk_overlap"
        )
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"chunk": True, "chunk_size": 1000}},
        )

        options = app._ingest_job_options(job)

        assert options["chunk_options"]["overlap"] == schema_overlap == 100

    def test_untouched_form_local_and_server_paths_agree_on_overlap(self) -> None:
        from tldw_chatbook.Library.server_ingest_request import (
            build_server_ingest_kwargs,
        )

        screen = object.__new__(LibraryScreen)
        screen._library_ingest_form = LibraryIngestFormState()
        snapshot = screen._build_ingest_options_snapshot()

        app = _minimal_app()
        job = _make_job(source_path="/tmp/test.txt", ingest_options=snapshot)
        local_options = app._ingest_job_options(job)
        server_kwargs = build_server_ingest_kwargs("/tmp/test.txt", options=snapshot)

        assert local_options["chunk_options"] is not None
        assert (
            local_options["chunk_options"]["overlap"] == server_kwargs["chunk_overlap"]
        )
        assert local_options["chunk_options"]["size"] == server_kwargs["chunk_size"]

    def test_fresh_snapshot_seeds_shared_generic_schema_defaults(self) -> None:
        screen = object.__new__(LibraryScreen)
        screen._library_ingest_form = LibraryIngestFormState()

        snapshot = screen._build_ingest_options_snapshot()
        defaults = {
            field.name: field.default
            for field in get_capabilities("generic").fields
            if field.name
            in {
                "overwrite_existing",
                "custom_prompt",
                "system_prompt",
                "generate_embeddings",
                "keep_original_file",
            }
        }

        assert {name: snapshot["generic"][name] for name in defaults} == defaults

    def test_server_request_strips_external_parakeet_path_and_scope(self) -> None:
        from tldw_chatbook.Library.server_ingest_request import (
            build_server_ingest_kwargs,
        )

        private_path = "/private/user-owned/parakeet-sentinel"
        private_scope = "library-external-private-scope"
        options = {
            "audio_video": {
                "transcription_provider": "parakeet-onnx",
                "transcription_model_dir": private_path,
                "transcription_external_scope_id": private_scope,
                "language": "en",
            }
        }

        kwargs = build_server_ingest_kwargs("/tmp/speech.wav", options=options)

        assert "transcription_provider" not in kwargs
        assert kwargs["transcription_language"] == "en"
        assert "transcription_model_dir" not in kwargs
        assert "transcription_external_scope_id" not in kwargs
        assert private_path not in str(kwargs)
        assert private_scope not in str(kwargs)
        assert options["audio_video"]["transcription_model_dir"] == private_path
        assert (
            options["audio_video"]["transcription_external_scope_id"] == private_scope
        )

    def test_display_string_sizes_are_coerced_to_int(self) -> None:
        """The panel Inputs hand back display text; processors get ints."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={
                "generic": {
                    "chunk": True,
                    "chunk_size": "1000",
                    "chunk_overlap": "150",
                }
            },
        )

        options = app._ingest_job_options(job)

        assert options["chunk_options"]["size"] == 1000
        assert options["chunk_options"]["overlap"] == 150
        assert isinstance(options["chunk_options"]["size"], int)
        assert isinstance(options["chunk_options"]["overlap"], int)

    def test_chunk_options_carry_max_size_for_chunking_service(self) -> None:
        """``improved_chunking_process`` reads ``max_size``; the legacy
        audio/video option map reads ``size``. Both spellings must travel."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"chunk": True, "chunk_size": 777}},
        )

        options = app._ingest_job_options(job)

        assert options["chunk_options"]["size"] == 777
        assert options["chunk_options"]["max_size"] == 777

    def test_encoding_selection_reaches_options(self) -> None:
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"encoding": "latin-1"}},
        )

        options = app._ingest_job_options(job)

        assert options["encoding"] == "latin-1"

    def test_analysis_provider_resolved_from_config(self) -> None:
        app = _minimal_app()
        app.app_config = {
            "analysis_defaults": {"provider": "OpenAI"},
            "api_settings": {"openai": {"api_key": "sk-test-configured"}},
        }
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"analyze": True}},
        )

        options = app._ingest_job_options(job)

        assert options["perform_analysis"] is True
        # (task-3301 xhigh review round) The NORMALIZED dispatch name
        # travels -- it is what `chat_api_call` (and the summarizer's
        # alias map) accept; the display spelling only ever fed logs.
        assert options["api_name"] == "openai"
        assert options["api_key"] == "sk-test-configured"
        assert "analysis_skipped_reason" not in options

    def test_analysis_provider_resolved_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-env")
        app = _minimal_app()
        app.app_config = {"analysis_defaults": {"provider": "OpenAI"}}
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"analyze": True}},
        )

        options = app._ingest_job_options(job)

        assert options["api_name"] == "openai"
        assert options["api_key"] == "sk-test-env"

    def test_unready_analysis_records_skip_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        app = _minimal_app()
        app.app_config = {"analysis_defaults": {"provider": "OpenAI"}}
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"analyze": True}},
        )

        options = app._ingest_job_options(job)

        assert options.get("api_key") is None
        assert options["analysis_skipped_reason"]
        assert "OpenAI" in options["analysis_skipped_reason"]

    def test_no_provider_configured_records_skip_reason(self) -> None:
        app = _minimal_app()
        app.app_config = {}
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"analyze": True}},
        )

        options = app._ingest_job_options(job)

        assert options["analysis_skipped_reason"]
        assert "provider" in options["analysis_skipped_reason"]

    def test_analyze_off_skips_provider_resolution_entirely(self) -> None:
        app = _minimal_app()
        app.app_config = {
            "analysis_defaults": {"provider": "OpenAI"},
            "api_settings": {"openai": {"api_key": "sk-test-configured"}},
        }
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"analyze": False}},
        )

        options = app._ingest_job_options(job)

        assert options["perform_analysis"] is False
        assert "api_name" not in options
        assert "api_key" not in options
        assert "analysis_skipped_reason" not in options

    def test_analysis_call_settings_travel(self) -> None:
        """(task-3301 xhigh review round, F10) The full [analysis_defaults]
        call shape travels to the worker, not just the provider name."""
        app = _minimal_app()
        app.app_config = {
            "analysis_defaults": {
                "provider": "OpenAI",
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "top_p": 0.9,
                "min_p": 0.01,
                "max_tokens": 512,
                "system_prompt": "Analyze thoroughly.",
            },
            "api_settings": {"openai": {"api_key": "sk-test-configured"}},
        }
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"analyze": True}},
        )

        options = app._ingest_job_options(job)

        assert options["analysis_call"] == {
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "top_p": 0.9,
            "min_p": 0.01,
            "max_tokens": 512,
        }
        assert options["system_prompt"] == "Analyze thoroughly."

    def test_keyless_provider_sets_explicit_opt_in(self) -> None:
        """(task-3301 xhigh review round, F8) Keyless-ready providers get
        the explicit opt-in flag; keyed providers never do."""
        app = _minimal_app()
        app.app_config = {"analysis_defaults": {"provider": "Ollama"}}
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"analyze": True}},
        )

        options = app._ingest_job_options(job)

        assert options["api_name"] == "ollama"
        assert options.get("api_key") is None
        assert options["analysis_keyless_ok"] is True

    def test_keyed_provider_does_not_set_keyless_opt_in(self) -> None:
        app = _minimal_app()
        app.app_config = {
            "analysis_defaults": {"provider": "OpenAI"},
            "api_settings": {"openai": {"api_key": "sk-test-configured"}},
        }
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"analyze": True}},
        )

        options = app._ingest_job_options(job)

        assert "analysis_keyless_ok" not in options

    def test_undispatchable_provider_records_skip_reason(self) -> None:
        """(task-3301 xhigh review round, F5) A readiness-ready provider
        with no chat dispatch handler must skip with a reason, not error
        at analysis time."""
        app = _minimal_app()
        app.app_config = {"analysis_defaults": {"provider": "custom"}}
        job = _make_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"analyze": True}},
        )

        options = app._ingest_job_options(job)

        assert "api_name" not in options
        assert (
            "not supported for ingest analysis" in (options["analysis_skipped_reason"])
        )


class TestIngestDoneProgress:
    """task-3301: the done row records analysis skipped-with-reason."""

    def test_plain_import_message(self) -> None:
        progress = app_module._library_ingest_done_progress(
            "/tmp/notes.txt", was_duplicate=False, payload={}
        )
        assert progress == {"message": "Imported notes.txt"}

    def test_analysis_skip_reason_appended(self) -> None:
        progress = app_module._library_ingest_done_progress(
            "/tmp/notes.txt",
            was_duplicate=False,
            payload={
                "analysis_skipped_reason": "OpenAI is not ready (Missing API key)"
            },
        )
        assert (
            progress["message"]
            == "Imported notes.txt — analysis skipped: OpenAI is not ready "
            "(Missing API key)"
        )
        assert progress["analysis_skipped"] == "OpenAI is not ready (Missing API key)"

    def test_analysis_failed_reason_appended(self) -> None:
        """(task-3301 xhigh review round, F4) A failed analysis annotates
        the done row the same way a skipped one does -- never silence."""
        progress = app_module._library_ingest_done_progress(
            "/tmp/notes.txt",
            was_duplicate=False,
            payload={"analysis_failed_reason": "Invalid API Name 'custom'"},
        )
        assert (
            progress["message"]
            == "Imported notes.txt — analysis failed: Invalid API Name 'custom'"
        )
        assert progress["analysis_failed"] == "Invalid API Name 'custom'"

    def test_cookies_problem_is_visible_on_the_done_row(self) -> None:
        """(task-3306 xhigh review round) A rejected cookies path must be
        SEEN. The import still succeeds (public URLs need no cookies), so
        the only honest signal is the same done-row annotation the analysis
        skip/failure reasons use."""
        progress = app_module._library_ingest_done_progress(
            "/tmp/clip.mp4",
            was_duplicate=False,
            payload={"cookies_problem": "Cookies file not found: /tmp/c.txt"},
        )
        assert (
            progress["message"]
            == "Imported clip.mp4 — cookies ignored: Cookies file not found: "
            "/tmp/c.txt"
        )
        assert progress["cookies_problem"] == ("Cookies file not found: /tmp/c.txt")

    def test_duplicate_message_keeps_matched_prefix(self) -> None:
        from tldw_chatbook.Library.library_ingest_jobs import (
            INGEST_DUPLICATE_PROGRESS_PREFIX,
        )

        progress = app_module._library_ingest_done_progress(
            "/tmp/notes.txt",
            was_duplicate=True,
            payload={"analysis_skipped_reason": "whatever"},
        )
        assert progress["message"].startswith(INGEST_DUPLICATE_PROGRESS_PREFIX)


def test_submit_refuses_active_local_duplicate_before_second_append(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _minimal_app(media_db="present")
    source = tmp_path / "a.txt"
    source.write_text("body")
    first = app.submit_library_ingest_job(source_path=str(source))
    before_ids = [job.job_id for job in app.library_ingest_jobs.jobs()]
    allocate_job_id = MagicMock(
        wraps=app.library_ingest_jobs._allocate_job_id  # noqa: SLF001
    )
    monkeypatch.setattr(
        app.library_ingest_jobs,
        "_allocate_job_id",
        allocate_job_id,
    )

    with pytest.raises(ActiveIngestSubmissionRefused) as caught:
        app.submit_library_ingest_job(source_path=str(source))

    assert [job.job_id for job in app.library_ingest_jobs.jobs()] == before_ids
    allocate_job_id.assert_not_called()
    assert caught.value.matches == (ActiveIngestJobRef(first.job_id, first.state),)


def test_research_ingest_required_origin_fails_before_queue_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _minimal_app(media_db="present")
    monkeypatch.setattr(app, "_resolve_ingest_backend", lambda: "local")
    admitted = MagicMock()
    monkeypatch.setattr(app, "_submit_library_ingest_job_admitted", admitted)

    with pytest.raises(ValueError, match="selected Server authority"):
        app.submit_library_ingest_job(
            source_path="https://example.invalid/paper",
            research_source_operation_id="operation-server-1",
            required_origin="server",
        )

    assert app.library_ingest_jobs.jobs() == ()
    admitted.assert_not_called()


def test_research_ingest_rejects_changed_server_identity_before_queue_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _minimal_app(media_db="present")
    captured_context = SimpleNamespace(
        active_server_id="server-a",
        auth_token="captured-token",
        credential_source="fixture",
    )
    changed_context = SimpleNamespace(
        active_server_id="server-b",
        auth_token="changed-token",
        credential_source="fixture",
    )
    operation = ResearchSourceOperation(
        operation_id="operation-server-qualified",
        idempotency_key="intake-server-qualified",
        data_source=WorkspaceDataSource.SERVER,
        server_profile_id="server-a",
        principal_id=event_principal_id_from_active_context(captured_context) or "",
        workspace_id="workspace-server",
        canonical_item_type=CanonicalItemType.SERVER_MEDIA,
        desired_selected=True,
        created_at="2026-08-24T10:00:00Z",
        updated_at="2026-08-24T10:00:00Z",
    )
    app.research_source_operation_store = SimpleNamespace(
        get=lambda operation_id: (
            operation if operation_id == operation.operation_id else None
        )
    )
    app.server_context_provider = SimpleNamespace(
        get_active_context=lambda: changed_context
    )
    monkeypatch.setattr(app, "_resolve_ingest_backend", lambda: "server")
    admitted = MagicMock()
    monkeypatch.setattr(app, "_submit_library_ingest_job_admitted", admitted)

    with pytest.raises(ValueError, match="captured Server workspace authority"):
        app.submit_library_ingest_job(
            source_path="https://example.invalid/paper",
            research_source_operation_id=operation.operation_id,
            required_origin="server",
        )

    assert app.library_ingest_jobs.jobs() == ()
    admitted.assert_not_called()


@pytest.mark.asyncio
async def test_remote_research_dispatch_rechecks_qualified_identity_before_service_call() -> (
    None
):
    app = _minimal_app(media_db="present")
    captured_context = SimpleNamespace(
        active_server_id="server-a",
        auth_token="captured-token",
        credential_source="fixture",
    )
    changed_context = SimpleNamespace(
        active_server_id="server-b",
        auth_token="changed-token",
        credential_source="fixture",
    )
    operation = ResearchSourceOperation(
        operation_id="operation-delayed-server",
        idempotency_key="intake-delayed-server",
        data_source=WorkspaceDataSource.SERVER,
        server_profile_id="server-a",
        principal_id=event_principal_id_from_active_context(captured_context) or "",
        workspace_id="workspace-server",
        canonical_item_type=CanonicalItemType.SERVER_MEDIA,
        desired_selected=True,
        created_at="2026-08-24T10:00:00Z",
        updated_at="2026-08-24T10:00:00Z",
    )
    app.research_source_operation_store = SimpleNamespace(
        get=lambda _operation_id: operation
    )
    app.server_context_provider = SimpleNamespace(
        get_active_context=lambda: changed_context
    )
    submit = AsyncMock(return_value={"jobs": [], "errors": []})
    app.server_media_reading_service = SimpleNamespace(submit_ingest_jobs=submit)
    job = app.library_ingest_jobs.submit(
        source_path="paper.pdf",
        origin="server",
        research_source_operation_id=operation.operation_id,
    )

    await TldwCli._send_server_ingest_job.__wrapped__(app, job.job_id, {"files": []})

    submit.assert_not_called()
    failed = app.library_ingest_jobs.get_job(job.job_id)
    assert failed is not None
    assert failed.state is IngestJobState.FAILED
    assert "captured Server workspace authority" in failed.error


def test_terminal_local_job_does_not_block_reingestion(tmp_path: Path) -> None:
    app = _minimal_app(media_db="present")
    source = tmp_path / "a.txt"
    source.write_text("body")
    first = app.submit_library_ingest_job(source_path=str(source))
    app.library_ingest_jobs.mark_parsing(first.job_id)
    app.library_ingest_jobs.mark_writing(first.job_id)
    app.library_ingest_jobs.mark_done(first.job_id, media_id=1)

    second = app.submit_library_ingest_job(source_path=str(source))

    assert second.job_id != first.job_id


def test_local_active_job_does_not_block_server_submission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _minimal_app(media_db="present")
    source = tmp_path / "a.txt"
    source.write_text("body")
    app.submit_library_ingest_job(source_path=str(source))
    monkeypatch.setattr(app, "_resolve_ingest_backend", lambda: "server")
    remote = MagicMock(return_value=_make_job(origin="server"))
    monkeypatch.setattr(app, "_submit_server_ingest_job", remote)

    app.submit_library_ingest_job(source_path=str(source))

    remote.assert_called_once()


def test_submit_refuses_active_server_duplicate_before_remote_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _minimal_app(media_db="present")
    source = tmp_path / "a.txt"
    source.write_text("body")
    monkeypatch.setattr(app, "_resolve_ingest_backend", lambda: "server")
    active = app.library_ingest_jobs.submit(source_path=str(source), origin="server")
    remote = MagicMock()
    monkeypatch.setattr(app, "_submit_server_ingest_job", remote)

    with pytest.raises(ActiveIngestSubmissionRefused) as caught:
        app.submit_library_ingest_job(source_path=str(source))

    assert caught.value.matches == (
        ActiveIngestJobRef(active.job_id, IngestJobState.QUEUED),
    )
    remote.assert_not_called()


def test_folder_refusal_occurs_before_any_admitted_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _minimal_app(media_db="present")
    folder = tmp_path / "batch"
    folder.mkdir()
    first = folder / "first.txt"
    matching = folder / "matching.txt"
    first.write_text("first")
    matching.write_text("matching")
    app.submit_library_ingest_job(source_path=str(matching))
    admitted = MagicMock()
    monkeypatch.setattr(app, "_submit_library_ingest_job_admitted", admitted)
    uuid4 = MagicMock()
    monkeypatch.setattr(app_module.uuid, "uuid4", uuid4)
    app._parakeet_submitting_scope_ids = {"existing-scope"}
    sync_scopes = MagicMock()
    monkeypatch.setattr(app, "_sync_parakeet_source_scopes", sync_scopes)

    with pytest.raises(ActiveIngestSubmissionRefused):
        app.submit_library_ingest_job(
            source_path=str(folder),
            ingest_options={
                "audio_video": {
                    "transcription_external_scope_id": "refused-scope",
                }
            },
        )

    admitted.assert_not_called()
    uuid4.assert_not_called()
    assert app._parakeet_submitting_scope_ids == {"existing-scope"}
    sync_scopes.assert_not_called()


def test_confirmed_folder_routes_every_member_once_without_reentry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _minimal_app(media_db="present")
    folder = tmp_path / "batch"
    folder.mkdir()
    paths = [folder / "a.txt", folder / "b.txt"]
    for path in paths:
        path.write_text(path.stem)
    active = app.library_ingest_jobs.submit(source_path=str(paths[1]))
    consent_scope = build_active_ingest_consent_scope(
        [str(path) for path in paths],
        origin="local",
        active_job_ids=(active.job_id,),
        active_source_count=1,
    )
    resolve_backend = MagicMock(wraps=app._resolve_ingest_backend)
    expand_source = MagicMock(wraps=app._expand_library_ingest_source)
    monkeypatch.setattr(app, "_resolve_ingest_backend", resolve_backend)
    monkeypatch.setattr(app, "_expand_library_ingest_source", expand_source)
    original = app._submit_library_ingest_job_admitted
    admitted_calls = []

    def record(**kwargs: Any) -> LibraryIngestJob:
        admitted_calls.append((kwargs["source_path"], kwargs["batch_id"]))
        return original(**kwargs)

    monkeypatch.setattr(app, "_submit_library_ingest_job_admitted", record)

    app.submit_library_ingest_job(
        source_path=str(folder), active_duplicate_consent=consent_scope
    )

    resolve_backend.assert_called_once_with()
    expand_source.assert_called_once_with(str(folder))
    assert sorted(source for source, _batch_id in admitted_calls) == [
        str(path) for path in paths
    ]
    batch_ids = {batch_id for _source, batch_id in admitted_calls}
    assert len(batch_ids) == 1
    assert None not in batch_ids

    admitted_count = len(admitted_calls)
    with pytest.raises(ActiveIngestSubmissionRefused):
        app.submit_library_ingest_job(source_path=str(folder))

    assert resolve_backend.call_count == 2
    assert expand_source.call_count == 2
    assert len(admitted_calls) == admitted_count


@pytest.mark.parametrize("mutation", ["added", "removed", "changed"])
def test_folder_candidate_mutation_refuses_stale_consent_before_any_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    app = _minimal_app(media_db="present")
    folder = tmp_path / "batch"
    folder.mkdir()
    paths = [folder / "a.txt", folder / "b.txt"]
    for path in paths:
        path.write_text(path.stem)
    active = app.library_ingest_jobs.submit(source_path=str(paths[1]))
    consent_scope = build_active_ingest_consent_scope(
        [str(path) for path in paths],
        origin="local",
        active_job_ids=(active.job_id,),
        active_source_count=1,
    )
    if mutation == "added":
        (folder / "c.txt").write_text("c")
    elif mutation == "removed":
        paths[0].unlink()
    else:
        paths[0].rename(folder / "renamed.txt")
    admitted = MagicMock()
    monkeypatch.setattr(app, "_submit_library_ingest_job_admitted", admitted)

    with pytest.raises(ActiveIngestSubmissionRefused) as caught:
        app.submit_library_ingest_job(
            source_path=str(folder),
            active_duplicate_consent=consent_scope,
        )

    assert caught.value.candidate_changed is True
    assert isinstance(caught.value.consent_scope, ActiveIngestConsentScope)
    assert caught.value.consent_scope != consent_scope
    admitted.assert_not_called()


def test_new_active_match_absent_from_consent_refuses_before_any_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _minimal_app(media_db="present")
    folder = tmp_path / "batch"
    folder.mkdir()
    paths = [folder / "a.txt", folder / "b.txt"]
    for path in paths:
        path.write_text(path.stem)
    first = app.library_ingest_jobs.submit(source_path=str(paths[0]))
    consent_scope = build_active_ingest_consent_scope(
        [str(path) for path in paths],
        origin="local",
        active_job_ids=(first.job_id,),
        active_source_count=1,
    )
    second = app.library_ingest_jobs.submit(source_path=str(paths[1]))
    admitted = MagicMock()
    monkeypatch.setattr(app, "_submit_library_ingest_job_admitted", admitted)

    with pytest.raises(ActiveIngestSubmissionRefused) as caught:
        app.submit_library_ingest_job(
            source_path=str(folder),
            active_duplicate_consent=consent_scope,
        )

    assert caught.value.candidate_changed is False
    assert caught.value.consent_scope.active_job_ids == (
        first.job_id,
        second.job_id,
    )
    admitted.assert_not_called()


def test_consent_scope_allows_matching_job_to_finish_before_second_press(
    tmp_path: Path,
) -> None:
    app = _minimal_app(media_db="present")
    source = tmp_path / "a.txt"
    source.write_text("body")
    active = app.library_ingest_jobs.submit(source_path=str(source))
    consent_scope = build_active_ingest_consent_scope(
        [str(source)],
        origin="local",
        active_job_ids=(active.job_id,),
        active_source_count=1,
    )
    app.library_ingest_jobs.mark_parsing(active.job_id)
    app.library_ingest_jobs.mark_writing(active.job_id)
    app.library_ingest_jobs.mark_done(active.job_id, media_id=1)

    submitted = app.submit_library_ingest_job(
        source_path=str(source),
        active_duplicate_consent=consent_scope,
    )

    assert submitted.job_id != active.job_id


def test_direct_refusal_is_privacy_safe_and_starts_no_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app = _minimal_app(media_db="present")
    source = tmp_path / "private-name.txt"
    source.write_text("secret")
    active = app.submit_library_ingest_job(
        source_path=str(source),
        title="Private title",
        author="Private author",
        keywords=("private-keyword",),
        ingest_options={"generic": {"custom_prompt": "private-prompt"}},
    )
    app.library_ingest_jobs.update_progress(
        active.job_id,
        progress={"message": "Private progress"},
    )
    top_up = MagicMock()
    monkeypatch.setattr(app, "_top_up_ingest_parse_pool", top_up)

    with pytest.raises(ActiveIngestSubmissionRefused) as caught:
        app.submit_library_ingest_job(
            source_path=str(source),
            title="Private title",
            author="Private author",
            keywords=("private-keyword",),
            ingest_options={"generic": {"custom_prompt": "private-prompt"}},
        )

    rendered = (
        f"{caught.value!s} {caught.value!r} "
        f"{caught.value.args!r} {caught.value.matches!r} "
        f"{caught.value.match_count!r}"
    )
    for secret in (
        str(source),
        "Private title",
        "Private author",
        "private-keyword",
        "private-prompt",
        "Private progress",
    ):
        assert secret not in rendered
    top_up.assert_not_called()


class TestSubmitLibraryIngestJob:
    """Coverage for TldwCli.submit_library_ingest_job."""

    def test_submit_passes_ingest_options_to_registry(self) -> None:
        app = _minimal_app(media_db="present")
        ingest_options = {
            "generic": {"analyze": True},
            "pdf": {"pdf_engine": "docling"},
        }
        job = app.submit_library_ingest_job(
            source_path="/tmp/test.pdf",
            ingest_options=ingest_options,
        )

        assert job.ingest_options == ingest_options
        stored = next(
            (j for j in app.library_ingest_jobs.jobs() if j.job_id == job.job_id),
            None,
        )
        assert stored is not None
        assert stored.ingest_options == ingest_options

    def test_submit_defaults_ingest_options_to_empty_dict(self) -> None:
        app = _minimal_app(media_db="present")
        job = app.submit_library_ingest_job(source_path="/tmp/test.txt")

        assert job.ingest_options == {}

    def test_submit_without_media_db_marks_job_failed(self) -> None:
        app = _minimal_app(media_db=None)
        job = app.submit_library_ingest_job(
            source_path="/tmp/test.txt",
            ingest_options={"generic": {"analyze": True}},
        )

        assert job.state.name == "FAILED"
        assert job.error == "Media database is unavailable."
        # ingest_options should still be preserved on the failed job.
        assert job.ingest_options == {"generic": {"analyze": True}}

    def test_explicit_faster_whisper_retry_overrides_only_provider_and_links_job(
        self,
    ) -> None:
        app = _minimal_app(media_db="present")
        original = app.submit_library_ingest_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "generic": {"chunk": True},
                "audio_video": {
                    "transcription_provider": "transcribe-cpp",
                    "language": "en",
                    "timestamps": True,
                },
            },
        )
        failed = app.library_ingest_jobs.mark_failed(
            original.job_id,
            error="Speech-to-text inference failed.",
            stt_failure_provenance=_direct_failed_attempt(),
        )

        retry = app.retry_library_ingest_job_with_provider(
            failed.job_id,
            "faster-whisper",
        )

        assert retry.retry_of_job_id == failed.job_id
        assert retry.retry_source_failure_provenance == _direct_failed_attempt()
        assert retry.ingest_options == {
            "generic": {"chunk": True},
            "audio_video": {
                "transcription_provider": "faster-whisper",
                "language": "en",
                "timestamps": True,
            },
        }
        assert original.ingest_options["audio_video"]["transcription_provider"] == (
            "transcribe-cpp"
        )

    def test_retry_preserves_shared_generic_snapshot_values(self) -> None:
        app = _minimal_app(media_db="present")
        generic = {
            "analyze": True,
            "overwrite_existing": True,
            "custom_prompt": "Extract decisions.",
            "system_prompt": "Be concise.",
            "generate_embeddings": False,
            "keep_original_file": True,
        }
        original = app.submit_library_ingest_job(
            source_path="/tmp/test.mp3",
            ingest_options={
                "generic": generic,
                "audio_video": {"transcription_provider": "transcribe-cpp"},
            },
        )
        failed = app.library_ingest_jobs.mark_failed(
            original.job_id,
            error="Speech-to-text inference failed.",
            stt_failure_provenance=_direct_failed_attempt(),
        )

        retry = app.retry_library_ingest_job_with_provider(
            failed.job_id, "faster-whisper"
        )

        assert retry.ingest_options["generic"] == generic


@pytest.mark.parametrize(
    "source_path,expected_group",
    [
        ("/tmp/test.pdf", "pdf"),
        ("/tmp/test.mp3", "audio_video"),
        ("/tmp/test.epub", "ebook"),
        ("/tmp/test.txt", "generic"),
    ],
)
def test_ingest_job_options_detects_type_group(
    source_path: str, expected_group: str
) -> None:
    app = _minimal_app()
    job = _make_job(source_path=source_path)
    options = app._ingest_job_options(job)

    if expected_group == "pdf":
        assert "pdf_engine" in options
    elif expected_group == "audio_video":
        assert "transcription_model" in options
    elif expected_group == "ebook":
        assert "extraction_method" in options
    else:
        assert "pdf_engine" not in options
        assert "transcription_model" not in options
        assert "extraction_method" not in options


@pytest.mark.parametrize(
    ("invalid_audio_options", "error_fragment"),
    [
        (
            {
                "transcription_provider": "parakeet-onnx",
                "language": "auto",
            },
            "Retry with faster-whisper",
        ),
        (
            {
                "transcription_provider": "parakeet-onnx",
                "language": 7,
            },
            "language",
        ),
        (
            {
                "transcription_provider": 0,
                "language": "en",
            },
            "provider",
        ),
        (
            {
                "transcription_provider": False,
                "language": "en",
            },
            "provider",
        ),
        (
            {
                "transcription_provider": "",
                "language": "en",
            },
            "Unsupported batch STT provider",
        ),
        (
            {
                "transcription_provider": "faster-whisper",
                "language": "en",
                "translation_target_language": 0,
            },
            "target_language",
        ),
        (
            {
                "transcription_provider": "faster-whisper",
                "language": "en",
                "target_language": False,
            },
            "target_language",
        ),
    ],
)
def test_invalid_audio_request_allows_next_job_to_dispatch(
    invalid_audio_options: dict[str, Any],
    error_fragment: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = object.__new__(TldwCli)
    app.library_ingest_jobs = LibraryIngestJobRegistry()
    app._ingest_shutdown = False
    app._ingest_parse_worker_count = lambda: 1  # type: ignore[method-assign]
    app._ingest_heavy_lane_max_workers = lambda: 1  # type: ignore[method-assign]
    app._ingest_parse_pool_generation = 1
    app._ingest_parse_jobs_by_generation = {1: set()}
    app._ingest_parse_pool_mode = None
    app._ingest_parse_pool_retiring = False
    app._ingest_parse_pool_retirement_error = None
    app._ingest_local_stt_jobs = {}
    warning_messages: list[str] = []
    monkeypatch.setattr(
        "tldw_chatbook.app.logger.warning",
        lambda message: warning_messages.append(str(message)),
    )

    invalid = app.library_ingest_jobs.submit(
        source_path="/tmp/invalid.mp3",
        detected_type="audio",
        ingest_options={"audio_video": invalid_audio_options},
    )
    valid = app.library_ingest_jobs.submit(
        source_path="/tmp/valid.mp3",
        detected_type="audio",
        ingest_options={
            "audio_video": {
                "transcription_provider": "faster-whisper",
                "transcription_model": "small",
                "language": "en",
            }
        },
    )

    class _Pool:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, ...]] = []

        def apply_async(self, function, args, callback, error_callback) -> None:
            self.calls.append((function, args, callback, error_callback))

    pool = _Pool()
    pool_creation_calls = 0

    def ensure_pool(_mode: str) -> _Pool:
        nonlocal pool_creation_calls
        pool_creation_calls += 1
        return pool

    app._ensure_ingest_parse_pool = ensure_pool  # type: ignore[method-assign]

    app._top_up_ingest_parse_pool()

    jobs_by_id = {job.job_id: job for job in app.library_ingest_jobs.jobs()}
    invalid_job = jobs_by_id[invalid.job_id]
    valid_job = jobs_by_id[valid.job_id]
    assert invalid_job.state is IngestJobState.FAILED
    assert invalid_job.permanent is False
    assert invalid_job.error is not None
    assert error_fragment in invalid_job.error
    assert "\n" not in invalid_job.error
    assert len(invalid_job.error) <= 200
    assert valid_job.state is IngestJobState.PARSING
    assert pool_creation_calls == 1
    assert len(pool.calls) == 1
    _, (source_path, options, progress_context), _, _ = pool.calls[0]
    assert source_path == valid.source_path
    assert progress_context == (app._ingest_parse_pool_generation, valid.job_id)
    assert options["transcription_provider"] == "faster-whisper"
    routing_warnings = [
        message for message in warning_messages if "batch STT routing failed" in message
    ]
    assert len(routing_warnings) == 1
    assert invalid.job_id in routing_warnings[0]
    assert "detected_type=audio" in routing_warnings[0]
    assert error_fragment in routing_warnings[0]


# --- task-3307: image group branch in _ingest_job_options --------------------


class TestImageIngestJobOptions:
    def test_image_group_options_travel(self) -> None:
        """The image branch feeds ``process_image``'s OCR knobs."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/scan.png",
            ingest_options={
                "generic": {"chunk": True, "chunk_size": 800, "chunk_overlap": 80},
                "image": {
                    "ocr": True,
                    "ocr_language": "de",
                    "ocr_backend": "tesseract",
                },
            },
        )

        options = app._ingest_job_options(job)

        assert options["ocr"] is True
        assert options["ocr_language"] == "de"
        assert options["ocr_backend"] == "tesseract"
        # The generic base still applies (task-3301 layering), and the
        # chunk method is the explicit words unit the size hint promises
        # (task-3303 F12) -- process_image chunks the OCR text itself via
        # improved_chunking_process.
        assert options["chunk_options"] == {
            "size": 800,
            "max_size": 800,
            "overlap": 80,
            "method": "words",
        }

    def test_image_group_defaults_without_snapshot(self) -> None:
        """An untouched form mirrors the schema/processor defaults: OCR on
        (the extracted text IS the imported content), auto backend, en."""
        app = _minimal_app()
        job = _make_job(source_path="/tmp/scan.jpg")

        options = app._ingest_job_options(job)

        assert options["ocr"] is True
        assert options["ocr_language"] == "en"
        assert options["ocr_backend"] == "auto"

    def test_image_ocr_off_travels(self) -> None:
        """OCR off must reach the processor as False -- the parse then
        extracts nothing and the persist seam fails the job honestly."""
        app = _minimal_app()
        job = _make_job(
            source_path="/tmp/scan.png",
            ingest_options={"image": {"ocr": False}},
        )

        options = app._ingest_job_options(job)

        assert options["ocr"] is False


class TestWriteStageFailureCategory:
    """(xhigh review round, F4) The write stage's DEFAULT category.

    task-14821 removed the optimistic "a retry can succeed if transient"
    branch from every cause that isn't genuinely retryable -- but the
    writer still stamped ``write_error`` as the default for EVERY
    exception ``persist_parsed_media`` re-wraps, and ``write_error`` is
    precisely the one category that still earns that advisory. The defect
    the task was filed to remove stayed reachable through the default.
    """

    def test_an_exception_that_declares_its_category_keeps_it(self) -> None:
        from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
            NoContentExtractedError,
        )

        exc = NoContentExtractedError("No text could be extracted from x.pdf.")
        assert app_module._library_ingest_write_failure_category(exc) == "no_content"

    def test_a_real_database_failure_is_a_write_error(self) -> None:
        """The one cause a bare retry can genuinely clear keeps its name."""
        from tldw_chatbook.DB.Client_Media_DB_v2 import DatabaseError

        exc = DatabaseError("database is locked")
        assert app_module._library_ingest_write_failure_category(exc) == "write_error"

    def test_an_unknown_write_stage_failure_is_silent_not_optimistic(
        self,
    ) -> None:
        """(task-14821 AC#2) An unknown cause is silent. Stamping it
        ``write_error`` told the user "the file itself parsed fine" about a
        failure nobody had classified."""
        from tldw_chatbook.Library.library_ingest_state import (
            ingest_retry_advice,
        )

        category = app_module._library_ingest_write_failure_category(
            RuntimeError("something nobody classified")
        )
        assert category == ""
        assert (
            ingest_retry_advice(
                category=category, message="something nobody classified"
            )
            == ""
        )


class TestServerSubmitRefusesZeroByteSources:
    """(task-14910) A 0-byte file never leaves this machine.

    The forecast counts every 0-byte staged file as a certain failure on
    both backends. On the server path that used to be an unearned claim:
    ``_submit_server_ingest_job`` built kwargs for the empty file and sent
    it, so only the server -- which this process cannot inspect -- decided
    its fate. The client now refuses it with the reason it already knows,
    which is what makes the forecast true rather than lucky.
    """

    @staticmethod
    def _server_app() -> TldwCli:
        app = _minimal_app(media_db="present")
        app._sent = []  # type: ignore[attr-defined]
        app._send_server_ingest_job = (  # type: ignore[method-assign]
            lambda job_id, kwargs: app._sent.append((job_id, kwargs))
        )
        return app

    def test_a_zero_byte_file_fails_locally_and_is_never_sent(self, tmp_path) -> None:
        from tldw_chatbook.Library.server_ingest_request import (
            server_ingest_refusal,
        )

        empty = tmp_path / "empty.txt"
        empty.write_text("")
        app = self._server_app()

        job = app._submit_server_ingest_job(
            source_path=str(empty),
            ingest_options={},
            title="",
            author="",
            keywords=(),
            perform_analysis=False,
        )

        assert job.state is IngestJobState.FAILED
        assert job.error == server_ingest_refusal(str(empty))
        assert "empty.txt" in job.error
        assert app._sent == [], (
            "a 0-byte file reached the wire, so the forecast's "
            "'will fail' was a claim about someone else's machine"
        )

    def test_the_refusal_is_permanent_because_a_retry_cannot_change_it(
        self, tmp_path
    ) -> None:
        """A 0-byte file fails identically on every attempt -- the same
        reason the LOCAL path raises a ``PermanentIngestError`` for one, so
        the queue row withholds Retry on both backends."""
        empty = tmp_path / "empty.md"
        empty.write_text("")
        app = self._server_app()

        job = app._submit_server_ingest_job(
            source_path=str(empty),
            ingest_options={},
            title="",
            author="",
            keywords=(),
            perform_analysis=False,
        )

        assert job.state is IngestJobState.FAILED
        assert job.permanent is True

    def test_a_file_with_content_still_reaches_the_wire(self, tmp_path) -> None:
        """Guard: the refusal must not become a general server-submit
        block."""
        real = tmp_path / "notes.txt"
        real.write_text("Tides are driven by the moon.")
        app = self._server_app()

        job = app._submit_server_ingest_job(
            source_path=str(real),
            ingest_options={},
            title="",
            author="",
            keywords=(),
            perform_analysis=False,
        )

        assert job.state is IngestJobState.QUEUED
        assert [kwargs["file_paths"] for _job_id, kwargs in app._sent] == [[str(real)]]
