"""App-level Library ingest coordinator + writer contracts (F3 Task 4).

Drives the real ``LibraryIngestQueueMixin`` (mixed into ``TldwCli`` in
``app.py``) through a minimal Textual ``App`` test-harness -- mirroring the
``LibraryHarness`` pattern in ``Tests/UI/test_library_shell.py`` -- against a
real file-backed ``MediaDatabase`` and real small ``.txt`` files. The full
``TldwCli`` is never booted; only the mixin + a real registry + a real
``media_db`` attribute are exercised.

F3 splits the old single-worker queue-runner into a parse-pool coordinator
(UI thread) and a narrowed single-writer worker. Real parsing/persistence
still runs through the production seam (``run_parse_job``/
``persist_parsed_media``), but the *pool* itself is faked (see
``_FakeIngestParsePool`` below) so these pilots stay fast and deterministic
without spawning real OS processes -- a real ``multiprocessing.get_context
("spawn").Pool`` is already covered end-to-end by Task 2's
``Tests/Local_Ingestion/test_ingest_parse_worker.py::
test_run_parse_job_through_real_spawn_pool`` (marked ``integration``); this
file's job is proving the *coordinator's* wiring (top-up, completion
handling, claim-or-release, broken-pool recovery, shutdown), not re-proving
that a spawned process can run ``run_parse_job``.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import multiprocessing
import os
import queue
import subprocess
import sys
import textwrap
import threading
import time
from types import MappingProxyType, SimpleNamespace
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import patch

import pytest
from textual.app import App

import tldw_chatbook.app as _app_module
import tldw_chatbook.STT.parakeet_dispatch as _parakeet_dispatch_module
import tldw_chatbook.STT.parakeet_external as _parakeet_external_module
from tldw_chatbook.app import LibraryIngestQueueMixin
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Library.library_ingest_jobs import (
    IngestJobState,
    LibraryIngestJob,
    LibraryIngestJobRegistry,
)
from tldw_chatbook.Local_Ingestion.ingest_parse_worker import (
    initialize_ingest_parse_worker,
    run_parse_job,
)
from tldw_chatbook.Local_Ingestion.ingest_parse_progress import (
    INGEST_PARSE_PROGRESS_QUEUE_MAXSIZE,
    ParseProgressEvent,
)
from tldw_chatbook.Model_Artifacts.service import (
    ArtifactDescriptor,
    ArtifactFile,
    ArtifactFormat,
    ArtifactRef,
    ArtifactRole,
    ProvenanceClass,
)
from tldw_chatbook.STT.contracts import (
    ExecutionDevice,
    FileAudioSource,
    TranscriptionFailureCode,
)
from tldw_chatbook.STT.executor import (
    ExecutorEvent,
    ExecutorFailure,
    ExecutorResult,
    ExecutorUnavailableError,
    ModelIdentity,
    WorkerPhase,
)
from tldw_chatbook.STT.parakeet_dispatch import ParakeetDispatch

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Bounded polling: never wait unboundedly for a background worker.
_POLL_ATTEMPTS = 300
_POLL_INTERVAL = 0.02
# Gated-fake pattern (plan's Global Constraints): bound every fake-pool join
# at 30.0s so a stuck background thread can never hang a test run.
_FAKE_POOL_JOIN_TIMEOUT = 30.0


class _FakeIngestParsePool:
    """Test-only, in-process stand-in for a real spawn ``Pool`` (F3 seam).

    Monkeypatched in via ``_IngestRunnerHarness._create_ingest_parse_pool``.
    Two modes:

    - ``auto_run=True`` (default): every ``apply_async`` call spawns a
      plain background ``threading.Thread`` that runs ``func`` (typically
      the real ``run_parse_job``, against a real small file -- fully
      in-process, no real subprocess) and invokes ``callback``/
      ``error_callback`` from that same thread. A background thread (never
      the caller's thread) is required: the coordinator's callbacks call
      ``App.call_from_thread``, which raises ``RuntimeError`` if invoked
      from the app's own (UI) thread.
    - ``auto_run=False`` ("manual" mode): ``apply_async`` only *records*
      the call (in ``self.calls``) and does nothing further -- the test
      drives completion explicitly via ``trigger_success``/
      ``trigger_error`` (also always on a background thread, for the same
      reason). Used for backpressure/broken-pool pilots that need to hold
      a job in ``PARSING`` under direct test control.
    """

    def __init__(self, *, auto_run: bool = True) -> None:
        self.auto_run = auto_run
        self.calls: list[dict[str, Any]] = []
        self.terminated = False
        # Thread ident `terminate()` was invoked on -- the quit-deadlock
        # pilots assert teardown runs OFF the app's event-loop thread.
        self.terminate_thread_ident: Optional[int] = None
        self.join_thread_ident: Optional[int] = None
        self._threads: list[threading.Thread] = []

    def apply_async(
        self,
        func: Callable[..., Any],
        args: tuple = (),
        kwds: Optional[dict] = None,
        callback: Optional[Callable[[Any], None]] = None,
        error_callback: Optional[Callable[[BaseException], None]] = None,
    ) -> None:
        record = {
            "func": func,
            "args": args,
            "kwds": kwds or {},
            "callback": callback,
            "error_callback": error_callback,
        }
        self.calls.append(record)
        if self.auto_run:
            self._spawn(self._run_one, record)

    def _run_one(self, record: dict[str, Any]) -> None:
        try:
            result = record["func"](*record["args"], **record["kwds"])
        except Exception as exc:  # noqa: BLE001 - mirrors a real Pool's error_callback path
            if record["error_callback"] is not None:
                record["error_callback"](exc)
            return
        if record["callback"] is not None:
            record["callback"](result)

    def trigger_success(self, index: int, result: Any) -> None:
        """Manually complete the ``index``-th ``apply_async`` call (manual mode)."""
        callback = self.calls[index]["callback"]
        if callback is not None:
            self._spawn(callback, result)

    def trigger_error(self, index: int, exc: BaseException) -> None:
        """Manually fail the ``index``-th ``apply_async`` call (manual mode)."""
        error_callback = self.calls[index]["error_callback"]
        if error_callback is not None:
            self._spawn(error_callback, exc)

    def _spawn(self, target: Callable[..., Any], *args: Any) -> None:
        thread = threading.Thread(target=target, args=args, daemon=True)
        self._threads.append(thread)
        thread.start()

    def terminate(self) -> None:
        self.terminated = True
        self.terminate_thread_ident = threading.get_ident()

    def join(self) -> None:
        self.join_thread_ident = threading.get_ident()
        for thread in self._threads:
            thread.join(timeout=_FAKE_POOL_JOIN_TIMEOUT)

    def close(self) -> None:
        pass


class _RecordingIngestJobStore:
    """Minimal persistence sink that exposes registry write-through effects."""

    def __init__(self) -> None:
        self.upserts: list[str] = []
        self.deletes: list[str] = []
        self.retries: list[tuple[str, str]] = []

    def upsert_job(self, job: LibraryIngestJob) -> None:
        self.upserts.append(job.job_id)

    def delete_job(self, job_id: str) -> None:
        self.deletes.append(job_id)

    def upsert_retry(
        self,
        superseded_job: LibraryIngestJob,
        retry_job: LibraryIngestJob,
    ) -> None:
        self.retries.append((superseded_job.job_id, retry_job.job_id))


class _FakeLocalSTTExecutor:
    """Manual executor stand-in that records dispatch and emits off-thread."""

    def __init__(self, *, submit_error: BaseException | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.submit_error = submit_error
        self.generation = 1
        self.submit_thread_ident: int | None = None
        self.close_thread_ident: int | None = None
        self.cancel_calls: list[str] = []
        self.force_stop_calls: list[str] = []
        self.retiring = False
        self._retirement_complete = threading.Event()
        self._retirement_complete.set()

    def submit(self, **kwargs: Any) -> int:
        self.submit_thread_ident = threading.get_ident()
        if self.submit_error is not None:
            raise self.submit_error
        self.calls.append(kwargs)
        return self.generation

    def trigger_event(self, index: int, event: ExecutorEvent) -> None:
        self._spawn(self.calls[index]["on_event"], event)

    def trigger_result(self, index: int, result: ExecutorResult) -> None:
        self._spawn(self.calls[index]["on_result"], result)

    def trigger_failure(self, index: int, failure: ExecutorFailure) -> None:
        self._spawn(self.calls[index]["on_failure"], failure)

    def cancel(self, attempt_id: str) -> bool:
        self.cancel_calls.append(attempt_id)
        return True

    def force_stop(self, attempt_id: str) -> bool:
        self.force_stop_calls.append(attempt_id)
        self.retiring = True
        self._retirement_complete.clear()
        return True

    def wait_for_retirement(self, timeout: float | None = None) -> bool:
        return self._retirement_complete.wait(timeout)

    def complete_retirement(self) -> None:
        self.retiring = False
        self._retirement_complete.set()

    @staticmethod
    def _spawn(callback: Callable[[Any], None], value: Any) -> None:
        threading.Thread(target=callback, args=(value,), daemon=True).start()

    def close(self) -> None:
        self.close_thread_ident = threading.get_ident()


class _IngestRunnerHarness(LibraryIngestQueueMixin, App):
    """Minimal headless App hosting the ingest registry + coordinator + writer.

    Defaults to an auto-run ``_FakeIngestParsePool`` (real ``run_parse_job``/
    ``persist_parsed_media``, fake pool) so pilots never spawn real OS
    processes. Tests needing manual control over completion timing (backpressure,
    broken-pool) pass their own ``pool_factory``; tests needing a specific
    worker cap pass ``worker_count``.
    """

    def __init__(
        self,
        media_db: Optional[MediaDatabase],
        *,
        pool_factory: Optional[Callable[[], Any]] = None,
        worker_count: Optional[int] = None,
        heavy_lane: Optional[int] = None,
        local_stt_executor: _FakeLocalSTTExecutor | None = None,
        local_stt_dispatch_factory: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        # task-3315: runtime state DERIVED from the app's own initializer
        # (LibraryIngestQueueMixin._init_library_ingest_runtime_state) instead
        # of hand-listed, so a new `self._ingest_*` read in app.py can never
        # silently drift this harness. Host inputs (media_db, the fake
        # local-STT executor) are then applied on top.
        self._init_library_ingest_runtime_state()
        self.media_db = media_db
        self._local_stt_executor = local_stt_executor
        self._pool_factory = pool_factory or (lambda: _FakeIngestParsePool())
        self._pool_create_count = 0
        self._worker_count_override = worker_count
        self._heavy_lane_override = heavy_lane
        self._local_stt_dispatch_factory = local_stt_dispatch_factory

    def _create_ingest_parse_pool(self):
        self._pool_create_count += 1
        pool_or_resources = self._pool_factory()
        if isinstance(pool_or_resources, _app_module._IngestParsePoolResources):
            return pool_or_resources
        return _app_module._IngestParsePoolResources(
            pool_or_resources,
            queue.Queue(maxsize=INGEST_PARSE_PROGRESS_QUEUE_MAXSIZE),
        )

    def _ingest_parse_worker_count(self) -> int:
        if self._worker_count_override is not None:
            return self._worker_count_override
        return super()._ingest_parse_worker_count()

    def _ingest_heavy_lane_max_workers(self) -> int:
        if self._heavy_lane_override is not None:
            return self._heavy_lane_override
        return super()._ingest_heavy_lane_max_workers()

    def _build_local_stt_dispatch(self, job, options):
        if self._local_stt_dispatch_factory is not None:
            return self._local_stt_dispatch_factory(job, options)
        return super()._build_local_stt_dispatch(job, options)


def _fake_local_stt_dispatch(job, options) -> dict[str, Any]:
    provider = options["transcription_provider"]
    return {
        "attempt_id": f"{job.job_id}-attempt-{job.retry_count + 1}",
        "identity": ModelIdentity(
            provider_id=provider,
            model_id=options.get("transcription_model") or "local-gguf:whisper",
            root_revision=None,
            closure_fingerprint=None,
            precision=options.get("transcription_precision") or "native",
            device=ExecutionDevice.CPU,
        ),
        "local_source": None,
        "managed_store_root": None,
        "managed_artifact_ref": None,
        "managed_dependency_refs": (),
    }


def _fake_parakeet_dispatch() -> ParakeetDispatch:
    return ParakeetDispatch(
        identity=ModelIdentity(
            provider_id="parakeet-onnx",
            model_id="nemo-parakeet-tdt-0.6b-v2",
            root_revision=None,
            closure_fingerprint=None,
            precision="int8",
            device=ExecutionDevice.CPU,
        ),
        local_source=None,
        managed_store_root=None,
        managed_artifact_ref=None,
        option_updates=MappingProxyType({}),
    )


def _allow_test_external_root(app: _IngestRunnerHarness, root: Path) -> None:
    """Give dispatch fixtures exact hashes and ready VAD without global state."""

    service = app._ensure_parakeet_source_service()
    service._vad_ready = lambda: True

    def descriptor(model_id: str, precision: str) -> ArtifactDescriptor:
        files = tuple(
            ArtifactFile(
                path.name,
                path.stat().st_size,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in sorted(root.iterdir())
        )
        version = "v3" if model_id.endswith("-v3") else "v2"
        return ArtifactDescriptor(
            reference=ArtifactRef(f"parakeet-{version}", "fixture", precision),
            model_id=model_id,
            role=ArtifactRole.ROOT,
            format=ArtifactFormat.ONNX,
            consumer="stt",
            model_family="parakeet",
            upstream_repository="example/parakeet",
            upstream_revision="fixture",
            source_url="https://example.invalid/parakeet",
            precision=precision,
            expected_installed_bytes=sum(item.size_bytes for item in files),
            license_id="cc-by-4.0",
            license_url="https://example.invalid/license",
            usage_notice="test fixture",
            runtime_name="onnx-asr",
            runtime_version_constraint="==0.12.0",
            supported_os=("darwin",),
            supported_architectures=("arm64",),
            provenance=(ProvenanceClass.CHATBOOK_CURATED,),
            files=files,
        )

    service._descriptor_for = descriptor


def _make_db(tmp_path: Path, name: str = "library_ingest.db") -> MediaDatabase:
    return MediaDatabase(tmp_path / name, client_id="f3-runner-test")


def _write_text_file(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


class _RecordingAssociationScheduler:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def resume(self, operation_id: str) -> None:
        self.calls.append(operation_id)

    async def resume_incomplete(self) -> None:
        return None


def _exit_ingest_worker_abruptly() -> None:
    """Picklable spawn-pool target that simulates a hard worker crash."""
    os._exit(17)


async def _wait_for_job_state(
    app: _IngestRunnerHarness,
    pilot,
    job_id: str,
    state: IngestJobState,
    *,
    attempts: int = _POLL_ATTEMPTS,
) -> LibraryIngestJob:
    for _ in range(attempts):
        job = next(
            (j for j in app.library_ingest_jobs.jobs() if j.job_id == job_id), None
        )
        if job is not None and job.state == state:
            return job
        await pilot.pause(_POLL_INTERVAL)
    all_jobs = app.library_ingest_jobs.jobs()
    raise AssertionError(f"job {job_id} never reached {state}. Jobs: {all_jobs}")


async def _wait_for_runner_idle(
    app: _IngestRunnerHarness, pilot, *, attempts: int = _POLL_ATTEMPTS
) -> None:
    for _ in range(attempts):
        if not app.library_ingest_jobs.runner_active:
            return
        await pilot.pause(_POLL_INTERVAL)
    raise AssertionError("runner_active never returned to False")


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_submit_reaches_done_with_real_media_id(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    source = _write_text_file(
        tmp_path, "note-a.txt", "Tides are driven by the moon's gravity."
    )
    app = _IngestRunnerHarness(db)

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(source_path=str(source), title="Note A")
        assert job.state == IngestJobState.QUEUED

        done = await _wait_for_job_state(app, pilot, job.job_id, IngestJobState.DONE)

        assert done.media_id is not None
        row = db.get_media_by_id(done.media_id)
        assert row is not None
        assert row["title"] == "Note A"
        assert "moon's gravity" in row["content"]
        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_local_completion_schedules_research_association_after_mark_done(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "research-source.txt", "Captured source.")
    app = _IngestRunnerHarness(db)
    scheduler = _RecordingAssociationScheduler()
    app.research_source_association_scheduler = scheduler
    listener_observations: list[list[str]] = []
    app.library_ingest_jobs.add_listener(
        lambda: listener_observations.append(list(scheduler.calls))
    )

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(
            source_path=str(source),
            research_source_operation_id="research-op-app-local",
        )
        await _wait_for_job_state(app, pilot, job.job_id, IngestJobState.DONE)
        for _ in range(_POLL_ATTEMPTS):
            if scheduler.calls:
                break
            await pilot.pause(_POLL_INTERVAL)
        await _wait_for_runner_idle(app, pilot)

    assert scheduler.calls == ["research-op-app-local"]
    assert listener_observations
    assert all(observation == [] for observation in listener_observations)


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_research_operation_refuses_multi_file_folder_before_creating_jobs(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    folder = tmp_path / "research-folder"
    folder.mkdir()
    _write_text_file(folder, "first.txt", "First source.")
    _write_text_file(folder, "second.txt", "Second source.")
    app = _IngestRunnerHarness(db)

    async with app.run_test():
        with pytest.raises(
            ValueError,
            match="one Research source operation per catalog item",
        ):
            app.submit_library_ingest_job(
                source_path=str(folder),
                research_source_operation_id="research-op-folder",
            )

    assert app.library_ingest_jobs.jobs() == ()


def test_server_research_catalog_retry_stays_on_server_adapter(
    tmp_path: Path,
) -> None:
    app = _IngestRunnerHarness(_make_db(tmp_path))
    source = _write_text_file(tmp_path, "remote-retry.pdf", "PDF fixture")
    failed = app.library_ingest_jobs.submit(
        source_path=str(source),
        origin="server",
        detected_type="pdf",
        research_source_operation_id="research-op-server-retry",
    )
    app.library_ingest_jobs.mark_failed(failed.job_id, error="Temporary failure")

    with patch.object(app, "_send_server_ingest_job") as send:
        retried = app._retry_research_source_catalog_job(failed.job_id)

    assert retried is not None
    assert retried.origin == "server"
    assert retried.research_source_operation_id == "research-op-server-retry"
    assert app._pool_create_count == 0
    send.assert_called_once()


@pytest.mark.asyncio
async def test_terminal_failure_schedules_research_catalog_receipt(
    tmp_path: Path,
) -> None:
    app = _IngestRunnerHarness(_make_db(tmp_path))
    scheduler = _RecordingAssociationScheduler()
    app.research_source_association_scheduler = scheduler

    async with app.run_test() as pilot:
        job = app.library_ingest_jobs.submit(
            source_path=str(tmp_path / "missing.txt"),
            research_source_operation_id="research-op-failed",
        )
        app.library_ingest_jobs.mark_failed(job.job_id, error="missing")
        for _ in range(_POLL_ATTEMPTS):
            if scheduler.calls:
                break
            await pilot.pause(_POLL_INTERVAL)

    assert scheduler.calls == ["research-op-failed"]


@pytest.mark.asyncio
async def test_worker_exception_releases_terminal_job_for_later_resume(
    tmp_path: Path,
) -> None:
    app = _IngestRunnerHarness(_make_db(tmp_path))

    class FailOnceScheduler:
        def __init__(self) -> None:
            self.calls = 0

        async def resume(self, operation_id: str) -> None:
            assert operation_id == "research-op-worker-retry"
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("worker conflict")

    scheduler = FailOnceScheduler()
    app.research_source_association_scheduler = scheduler

    async with app.run_test() as pilot:
        job = app.library_ingest_jobs.submit(
            source_path=str(tmp_path / "missing.txt"),
            research_source_operation_id="research-op-worker-retry",
        )
        app.library_ingest_jobs.mark_failed(job.job_id, error="missing")
        for _ in range(_POLL_ATTEMPTS):
            if scheduler.calls == 1 and (
                job.job_id not in app._research_source_terminal_jobs_scheduled
            ):
                break
            await pilot.pause(_POLL_INTERVAL)

        app.library_ingest_jobs.submit(source_path=str(tmp_path / "unrelated.txt"))
        for _ in range(_POLL_ATTEMPTS):
            if scheduler.calls == 2:
                break
            await pilot.pause(_POLL_INTERVAL)

    assert scheduler.calls == 2


@pytest.mark.asyncio
async def test_clearing_terminal_history_prunes_research_schedule_dedupe(
    tmp_path: Path,
) -> None:
    app = _IngestRunnerHarness(_make_db(tmp_path))
    scheduler = _RecordingAssociationScheduler()
    app.research_source_association_scheduler = scheduler

    async with app.run_test() as pilot:
        job = app.library_ingest_jobs.submit(
            source_path=str(tmp_path / "missing.txt"),
            research_source_operation_id="research-op-cleared",
        )
        app.library_ingest_jobs.mark_failed(job.job_id, error="missing")
        for _ in range(_POLL_ATTEMPTS):
            if scheduler.calls:
                break
            await pilot.pause(_POLL_INTERVAL)
        assert job.job_id in app._research_source_terminal_jobs_scheduled

        app.library_ingest_jobs.clear_finished()

    assert app._research_source_terminal_jobs_scheduled == set()


def test_real_app_wires_research_association_and_restores_before_startup_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    workspace_db = _app_module.WorkspaceDB(
        tmp_path / "app-workspaces.sqlite",
        client_id="app-wiring-test",
    )
    request.addfinalizer(workspace_db.close)
    provider = SimpleNamespace(get_active_context=lambda: None)
    server_service = _app_module.ServerNotesWorkspaceService.from_server_context_provider(
        provider
    )
    app = _app_module.TldwCli.__new__(_app_module.TldwCli)
    app.local_workspace_db = workspace_db
    app.workspace_registry_service = _app_module.LocalWorkspaceRegistryService(
        workspace_db
    )
    app.library_ingest_jobs = LibraryIngestJobRegistry()
    app.server_context_provider = provider
    app.server_notes_workspace_service = server_service

    app._wire_research_source_association()

    scheduler = app.research_source_association_scheduler
    coordinator = app.research_source_association_coordinator

    assert scheduler._coordinator is coordinator
    assert scheduler._operation_store is app.research_source_operation_store
    assert coordinator._operation_store is app.research_source_operation_store
    assert coordinator._ingest_jobs is app.library_ingest_jobs
    assert coordinator._local_registry is app.workspace_registry_service
    assert coordinator._server_service is app.server_notes_workspace_service
    assert coordinator._server_context_provider is app.server_context_provider

    order: list[str] = []

    async def resume_incomplete() -> None:
        return None

    def queue_worker(awaitable, *, group: str):
        order.append(group)
        awaitable.close()

    monkeypatch.setattr(app, "_restore_ingest_jobs", lambda: order.append("restore"))
    monkeypatch.setattr(scheduler, "resume_incomplete", resume_incomplete)
    monkeypatch.setattr(app, "run_worker", queue_worker)

    app._restore_ingest_jobs_and_schedule_research_sources()

    assert order == ["restore", "research_source_association_startup"]


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_writer_passes_claimed_generate_embeddings_snapshot_to_persistence(
    tmp_path: Path,
) -> None:
    """Changing the writer's option forwarding must make this persistence call true."""
    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "embeddings-off.txt", "Persist this source.")
    app = _IngestRunnerHarness(db)

    with patch.object(
        _app_module, "persist_parsed_media", return_value=(777, "media-777", "saved")
    ) as persist:
        async with app.run_test() as pilot:
            job = app.submit_library_ingest_job(
                source_path=str(source),
                ingest_options={"generic": {"generate_embeddings": False}},
            )
            done = await _wait_for_job_state(
                app, pilot, job.job_id, IngestJobState.DONE
            )
            await _wait_for_runner_idle(app, pilot)

    assert done.media_id == 777
    assert persist.call_args.kwargs["generate_embeddings"] is False


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_writer_missing_generic_snapshot_uses_capability_defaults(
    tmp_path: Path,
) -> None:
    """Writer fallbacks must delegate to the capability schema, not literals."""
    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "defaults.txt", "Persist this source.")
    app = _IngestRunnerHarness(db)

    def schema_default(name: str, fallback: object = None) -> object:
        return {
            "overwrite_existing": True,
            "generate_embeddings": False,
        }.get(name, fallback)

    with (
        patch.object(_app_module, "generic_option_default", side_effect=schema_default),
        patch.object(
            _app_module,
            "persist_parsed_media",
            return_value=(778, "media-778", "saved"),
        ) as persist,
    ):
        async with app.run_test() as pilot:
            job = app.submit_library_ingest_job(source_path=str(source))
            done = await _wait_for_job_state(
                app, pilot, job.job_id, IngestJobState.DONE
            )
            await _wait_for_runner_idle(app, pilot)

    assert done.media_id == 778
    assert persist.call_args.kwargs["overwrite_existing"] is True
    assert persist.call_args.kwargs["generate_embeddings"] is False


@pytest.mark.asyncio
async def test_submitting_a_directory_queues_one_job_per_file(
    tmp_path: Path,
) -> None:
    """A folder must expand into per-file jobs, not one job for the folder.

    Pre-flight happily reports "4 plain text files" for a directory and lets
    the user start it, but the runner used to classify the directory itself
    and fail the whole submission with "Unsupported file type: ." -- so the
    batch import the UI advertises never worked at all (task-675).
    """
    db = _make_db(tmp_path)
    folder = tmp_path / "batch"
    folder.mkdir()
    for index in range(3):
        _write_text_file(folder, f"doc-{index}.txt", f"Body of document {index}.")
    app = _IngestRunnerHarness(db, worker_count=2)

    async with app.run_test() as pilot:
        app.submit_library_ingest_job(source_path=str(folder))

        jobs = app.library_ingest_jobs.jobs()
        assert len(jobs) == 3, f"expected one job per file, got {len(jobs)}"
        assert {Path(job.source_path).name for job in jobs} == {
            "doc-0.txt",
            "doc-1.txt",
            "doc-2.txt",
        }

        for job in jobs:
            await _wait_for_job_state(app, pilot, job.job_id, IngestJobState.DONE)

        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_directory_unsupported_file_is_skipped_alone(tmp_path: Path) -> None:
    """(task-2220 ruling) One unsupported file in a folder is SKIPPED on
    its own row -- a neutral outcome, never a failure -- and the supported
    siblings still reach DONE rather than being taken down with it."""
    db = _make_db(tmp_path)
    folder = tmp_path / "mixed"
    folder.mkdir()
    _write_text_file(folder, "good.txt", "A perfectly ingestible document.")
    # (task-3307: was cover.jpg -- images are a supported group now)
    (folder / "cover.xyz").write_bytes(b"no handler for this")
    app = _IngestRunnerHarness(db, worker_count=2)

    async with app.run_test() as pilot:
        app.submit_library_ingest_job(source_path=str(folder))

        jobs = {Path(j.source_path).name: j for j in app.library_ingest_jobs.jobs()}
        assert set(jobs) == {"good.txt", "cover.xyz"}

        await _wait_for_job_state(
            app, pilot, jobs["good.txt"].job_id, IngestJobState.DONE
        )
        skipped = await _wait_for_job_state(
            app, pilot, jobs["cover.xyz"].job_id, IngestJobState.SKIPPED
        )
        assert "Unsupported file type" in skipped.error

        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_directory_submission_honours_scan_limit(tmp_path: Path) -> None:
    """The folder expansion respects the configured directory scan limit."""
    db = _make_db(tmp_path)
    folder = tmp_path / "many"
    folder.mkdir()
    for index in range(6):
        _write_text_file(folder, f"doc-{index}.txt", f"Body {index}.")
    app = _IngestRunnerHarness(db, worker_count=2)

    from tldw_chatbook.app import get_cli_setting as _real_get_cli_setting

    def _limited(*args: Any, **kwargs: Any) -> Any:
        if args and args[0] == "library.ingest_directory_scan_limit":
            return 2
        return _real_get_cli_setting(*args, **kwargs)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_limited):
        async with app.run_test() as pilot:
            app.submit_library_ingest_job(source_path=str(folder))
            assert len(app.library_ingest_jobs.jobs()) == 2

            for job in app.library_ingest_jobs.jobs():
                await _wait_for_job_state(app, pilot, job.job_id, IngestJobState.DONE)
            await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_two_files_queued_both_reach_done_with_real_media_rows(
    tmp_path: Path,
) -> None:
    """(F3 pilot) Two files queued -> both reach DONE with real media rows,
    and the write stage (SQLite single-writer) never has two jobs WRITING
    at the same time -- even though both may PARSE concurrently (forced to
    N=2 here so both jobs' parses genuinely overlap)."""
    db = _make_db(tmp_path)
    source1 = _write_text_file(tmp_path, "note-1.txt", "First document body.")
    source2 = _write_text_file(tmp_path, "note-2.txt", "Second document body.")
    app = _IngestRunnerHarness(db, worker_count=2)

    async with app.run_test() as pilot:
        job1 = app.submit_library_ingest_job(source_path=str(source1))
        job2 = app.submit_library_ingest_job(source_path=str(source2))
        assert job1.job_id != job2.job_id

        max_writing_seen = 0
        for _ in range(_POLL_ATTEMPTS):
            counts = app.library_ingest_jobs.counts()
            max_writing_seen = max(max_writing_seen, counts["writing"])
            jobs_by_id = {j.job_id: j for j in app.library_ingest_jobs.jobs()}
            if (
                jobs_by_id[job1.job_id].state == IngestJobState.DONE
                and jobs_by_id[job2.job_id].state == IngestJobState.DONE
            ):
                break
            await pilot.pause(_POLL_INTERVAL)
        else:
            raise AssertionError(
                f"jobs never both completed: {app.library_ingest_jobs.jobs()}"
            )

        assert max_writing_seen <= 1, "two jobs were WRITING simultaneously"

        jobs_by_id = {j.job_id: j for j in app.library_ingest_jobs.jobs()}
        done1, done2 = jobs_by_id[job1.job_id], jobs_by_id[job2.job_id]
        assert done1.media_id is not None
        assert done2.media_id is not None
        assert db.get_media_by_id(done1.media_id) is not None
        assert db.get_media_by_id(done2.media_id) is not None

        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_failing_job_does_not_block_next_queued_job(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    missing = tmp_path / "does-not-exist.txt"
    ok_source = _write_text_file(tmp_path, "note-ok.txt", "This file exists just fine.")
    app = _IngestRunnerHarness(db)

    async with app.run_test() as pilot:
        failing_job = app.submit_library_ingest_job(source_path=str(missing))
        ok_job = app.submit_library_ingest_job(source_path=str(ok_source))

        failed = await _wait_for_job_state(
            app, pilot, failing_job.job_id, IngestJobState.FAILED
        )
        assert failed.error != ""
        assert len(failed.error) <= 200
        assert "\n" not in failed.error

        done = await _wait_for_job_state(app, pilot, ok_job.job_id, IngestJobState.DONE)
        assert done.media_id is not None

        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_retry_of_failed_job_succeeds_once_transient_error_clears(
    tmp_path: Path, monkeypatch
) -> None:
    """(M4 re-anchor, fix batch F1b; F3 re-anchor) A requeued job can reach
    DONE once whatever caused the first failure is gone. F3: the pool
    worker entry point (``run_parse_job``) never raises across the process
    boundary -- it always returns a structured result -- so a *transient*
    per-job parse failure is simulated the same way a real worker would
    report one: a structured ``{"ok": False, ...}`` result, not a raised
    exception (which the coordinator's ``error_callback`` path would
    instead treat as a POOL-level failure -- see the broken-pool pilot
    below -- since ``run_parse_job`` is contractually never supposed to
    raise)."""
    db = _make_db(tmp_path)
    target = _write_text_file(tmp_path, "arrives-later.txt", "Arrived just in time.")
    app = _IngestRunnerHarness(db)

    import tldw_chatbook.app as app_module

    real_run_parse_job = app_module.run_parse_job
    call_count = {"n": 0}

    def _flaky_run_parse_job(file_path, options, progress_context):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"ok": False, "error": "transient parse hiccup", "permanent": False}
        return real_run_parse_job(file_path, options, progress_context)

    monkeypatch.setattr(app_module, "run_parse_job", _flaky_run_parse_job)

    async with app.run_test() as pilot:
        failing_job = app.submit_library_ingest_job(source_path=str(target))
        failed = await _wait_for_job_state(
            app, pilot, failing_job.job_id, IngestJobState.FAILED
        )
        assert failed.permanent is False
        assert failed.error == "transient parse hiccup"
        await _wait_for_runner_idle(app, pilot)

        requeued = app.retry_library_ingest_job(failed.job_id)
        assert requeued is not None
        assert requeued.job_id != failed.job_id
        assert requeued.state == IngestJobState.QUEUED

        done = await _wait_for_job_state(
            app, pilot, requeued.job_id, IngestJobState.DONE
        )
        assert done.media_id is not None
        row = db.get_media_by_id(done.media_id)
        assert row is not None
        assert "Arrived just in time" in row["content"]

        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_missing_file_failure_is_permanent_and_refuses_retry(
    tmp_path: Path,
) -> None:
    """(M4, fix batch F1b) A ``FileNotFoundError`` from the parse worker
    fails the exact same way on every attempt -- classified ``permanent``
    inside ``run_parse_job`` (F3: the real exception type is only visible
    inside the worker) -- and ``retry_library_ingest_job`` must refuse it
    (defense in depth, on top of the canvas withholding the Retry button
    entirely for a permanent row)."""
    db = _make_db(tmp_path)
    missing = tmp_path / "does-not-exist.txt"
    app = _IngestRunnerHarness(db)

    async with app.run_test() as pilot:
        failing_job = app.submit_library_ingest_job(source_path=str(missing))
        failed = await _wait_for_job_state(
            app, pilot, failing_job.job_id, IngestJobState.FAILED
        )
        await _wait_for_runner_idle(app, pilot)

        assert failed.permanent is True
        assert app.retry_library_ingest_job(failed.job_id) is None


@pytest.mark.asyncio
async def test_unsupported_file_type_is_skipped_and_refuses_retry(
    tmp_path: Path,
) -> None:
    """(M4, contract revised by task-2220) An unsupported extension records
    as SKIPPED -- the pipeline never attempted it, so it is not a failure --
    and Retry stays refused (``requeue`` is FAILED-only)."""
    db = _make_db(tmp_path)
    unsupported = _write_text_file(tmp_path, "note.xyz", "irrelevant content")
    app = _IngestRunnerHarness(db)

    async with app.run_test() as pilot:
        submitted = app.submit_library_ingest_job(source_path=str(unsupported))
        skipped = await _wait_for_job_state(
            app, pilot, submitted.job_id, IngestJobState.SKIPPED
        )
        await _wait_for_runner_idle(app, pilot)

        assert "Unsupported file type" in skipped.error
        assert app.retry_library_ingest_job(skipped.job_id) is None


@pytest.mark.asyncio
async def test_submit_with_no_media_db_fails_immediately_without_starting_runner(
    tmp_path: Path,
) -> None:
    source = _write_text_file(tmp_path, "note.txt", "Irrelevant content.")
    app = _IngestRunnerHarness(None)

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(source_path=str(source))

        assert job.state == IngestJobState.FAILED
        assert job.error == "Media database is unavailable."

        # No runner should ever have started for this failure, and the
        # parse pool must never even have been created.
        await pilot.pause(_POLL_INTERVAL)
        await pilot.pause(_POLL_INTERVAL)
        assert app.library_ingest_jobs.runner_active is False
        assert app._pool_create_count == 0


@pytest.mark.asyncio
async def test_listener_fires_on_every_state_change(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "note-listener.txt", "Listener smoke content.")
    app = _IngestRunnerHarness(db)

    calls: list[int] = []
    app.library_ingest_jobs.add_listener(lambda: calls.append(1))

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(source_path=str(source))
        # (F3) submit_library_ingest_job's own top-up call synchronously
        # claims the job into PARSING before returning -- submit(1) +
        # mark_parsing(1) land inside this one call.
        assert len(calls) == 2

        await _wait_for_job_state(app, pilot, job.job_id, IngestJobState.DONE)
        await _wait_for_runner_idle(app, pilot)

        # (F3) submit -> mark_parsing -> mark_writing -> mark_done ==
        # 4 real, separate transitions (parsing and writing are now
        # distinct pipeline stages, not one aliased call).
        assert len(calls) == 4


# --- F3 Task 4: coordinator + writer pilots ---------------------------------


@pytest.mark.asyncio
async def test_broken_pool_fails_all_parsing_jobs_and_rebuilds_on_next_submit(
    tmp_path: Path,
) -> None:
    """(F3 pilot) A pool-level failure (e.g. a worker process died) must
    fail EVERY currently-``PARSING`` job as retryable and drop the pool --
    the next submission lazily rebuilds a fresh one."""
    db = _make_db(tmp_path)
    source1 = _write_text_file(tmp_path, "note-1.txt", "First body.")
    source2 = _write_text_file(tmp_path, "note-2.txt", "Second body.")

    pools: list[_FakeIngestParsePool] = []

    def _pool_factory() -> _FakeIngestParsePool:
        pool = _FakeIngestParsePool(auto_run=False)
        pools.append(pool)
        return pool

    app = _IngestRunnerHarness(db, pool_factory=_pool_factory, worker_count=2)

    async with app.run_test() as pilot:
        job1 = app.submit_library_ingest_job(source_path=str(source1))
        job2 = app.submit_library_ingest_job(source_path=str(source2))
        await pilot.pause()

        jobs_by_id = {j.job_id: j for j in app.library_ingest_jobs.jobs()}
        assert jobs_by_id[job1.job_id].state == IngestJobState.PARSING
        assert jobs_by_id[job2.job_id].state == IngestJobState.PARSING
        assert len(pools) == 1
        first_pool = pools[0]
        assert len(first_pool.calls) == 2

        # Simulate the pool dying: fire error_callback for ONE of the two
        # in-flight calls -- this must fail BOTH currently-PARSING jobs
        # (not just the one tied to this specific callback), since neither
        # can be trusted to ever complete on a broken pool.
        first_pool.trigger_error(0, RuntimeError("simulated worker death"))

        for _ in range(_POLL_ATTEMPTS):
            jobs_by_id = {j.job_id: j for j in app.library_ingest_jobs.jobs()}
            if (
                jobs_by_id[job1.job_id].state == IngestJobState.FAILED
                and jobs_by_id[job2.job_id].state == IngestJobState.FAILED
            ):
                break
            await pilot.pause(_POLL_INTERVAL)
        else:
            raise AssertionError(
                f"both jobs never reached FAILED: {app.library_ingest_jobs.jobs()}"
            )

        jobs_by_id = {j.job_id: j for j in app.library_ingest_jobs.jobs()}
        assert jobs_by_id[job1.job_id].permanent is False
        assert jobs_by_id[job2.job_id].permanent is False
        assert app._ingest_parse_pool is None

        # Retry -- the next submission must rebuild a fresh pool.
        requeued = app.retry_library_ingest_job(job1.job_id)
        assert requeued is not None
        assert len(pools) == 2, (
            "a fresh pool must be created lazily on the next submission"
        )


@pytest.mark.asyncio
async def test_stale_pool_generation_error_does_not_fail_replacement_job(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "stale-error.txt", "Generation isolation.")
    pools: list[_FakeIngestParsePool] = []

    def _pool_factory() -> _FakeIngestParsePool:
        pool = _FakeIngestParsePool(auto_run=False)
        pools.append(pool)
        return pool

    app = _IngestRunnerHarness(db, pool_factory=_pool_factory, worker_count=1)

    async with app.run_test() as pilot:
        original = app.submit_library_ingest_job(source_path=str(source))
        await pilot.pause()
        generation_a = pools[0]
        generation_a.trigger_error(0, RuntimeError("generation A died"))
        await _wait_for_job_state(app, pilot, original.job_id, IngestJobState.FAILED)

        replacement = app.retry_library_ingest_job(original.job_id)
        assert replacement is not None
        await _wait_for_job_state(
            app, pilot, replacement.job_id, IngestJobState.PARSING
        )
        assert len(pools) == 2

        generation_a.trigger_error(0, RuntimeError("late generation A error"))
        await pilot.pause(_POLL_INTERVAL)
        current = next(
            job
            for job in app.library_ingest_jobs.jobs()
            if job.job_id == replacement.job_id
        )
        assert current.state == IngestJobState.PARSING
        assert app._ingest_parse_pool is pools[1]


@pytest.mark.asyncio
async def test_stale_pool_generation_success_does_not_store_payload(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "stale-success.txt", "Generation isolation.")
    pools: list[_FakeIngestParsePool] = []

    def _pool_factory() -> _FakeIngestParsePool:
        pool = _FakeIngestParsePool(auto_run=False)
        pools.append(pool)
        return pool

    app = _IngestRunnerHarness(db, pool_factory=_pool_factory, worker_count=1)

    async with app.run_test() as pilot:
        original = app.submit_library_ingest_job(source_path=str(source))
        await pilot.pause()
        generation_a = pools[0]
        generation_a.trigger_error(0, RuntimeError("generation A died"))
        await _wait_for_job_state(app, pilot, original.job_id, IngestJobState.FAILED)

        replacement = app.retry_library_ingest_job(original.job_id)
        assert replacement is not None
        await _wait_for_job_state(
            app, pilot, replacement.job_id, IngestJobState.PARSING
        )

        generation_a.trigger_success(
            0,
            {"ok": True, "payload": {"file_type": "plaintext", "content": "stale"}},
        )
        await pilot.pause(_POLL_INTERVAL)

        assert original.job_id not in app._ingest_parsed_payloads
        assert replacement.job_id not in app._ingest_parsed_payloads
        assert app.library_ingest_jobs.runner_active is False
        current = next(
            job
            for job in app.library_ingest_jobs.jobs()
            if job.job_id == replacement.job_id
        )
        assert current.state == IngestJobState.PARSING


@pytest.mark.asyncio
async def test_real_pool_worker_exit_is_reported_for_owning_generation(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "hard-exit.txt", "Worker sentinel coverage.")
    app: _IngestRunnerHarness
    app = _IngestRunnerHarness(
        db,
        pool_factory=lambda: LibraryIngestQueueMixin._create_ingest_parse_pool(app),
        worker_count=1,
    )
    pool = None
    teardown_threads: list[threading.Thread] = []
    real_terminate = app._terminate_ingest_parse_pool_off_thread

    def _capture_production_teardown(
        target_pool: Any,
        progress_queue: Any | None = None,
        progress_thread: threading.Thread | None = None,
    ) -> threading.Thread:
        thread = real_terminate(target_pool, progress_queue, progress_thread)
        teardown_threads.append(thread)
        return thread

    app._terminate_ingest_parse_pool_off_thread = _capture_production_teardown

    try:
        async with app.run_test() as pilot:
            job = app.library_ingest_jobs.submit(source_path=str(source))
            claimed = app.library_ingest_jobs.mark_parsing(job.job_id)
            assert claimed is not None

            pool = app._ensure_ingest_parse_pool()
            generation = app._ingest_parse_pool_generation
            app._ingest_parse_jobs_by_generation[generation].add(job.job_id)
            pool.apply_async(_exit_ingest_worker_abruptly)

            failed = await _wait_for_job_state(
                app,
                pilot,
                job.job_id,
                IngestJobState.FAILED,
                attempts=500,
            )
            assert failed.permanent is False
            assert app._ingest_parse_pool is None
            assert generation not in app._ingest_parse_jobs_by_generation
    finally:
        if pool is not None:
            if teardown_threads:
                cleanup = teardown_threads[0]
            else:
                pool.terminate()
                cleanup = threading.Thread(target=pool.join, daemon=True)
                cleanup.start()
            cleanup.join(timeout=10.0)
            assert not cleanup.is_alive(), "real Pool cleanup exceeded 10 seconds"


def test_current_generation_sentinel_failure_retires_idle_pool(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(db, pool_factory=lambda: pool, worker_count=1)
    app._ensure_ingest_parse_pool()
    generation = app._ingest_parse_pool_generation
    assert app._ingest_parse_jobs_by_generation[generation] == set()

    app._handle_broken_ingest_parse_pool(
        generation,
        None,
        RuntimeError("idle worker exited"),
    )

    assert app._ingest_parse_pool is None
    assert generation not in app._ingest_parse_jobs_by_generation
    for _ in range(100):
        if pool.terminated:
            break
        threading.Event().wait(0.01)
    assert pool.terminated is True


def test_ensure_pool_initializes_generation_state_for_existing_host(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(db, pool_factory=lambda: pool, worker_count=1)
    del app._ingest_parse_pool_generation
    del app._ingest_parse_jobs_by_generation
    del app._ingest_parse_pool_stop_event

    assert app._ensure_ingest_parse_pool() is pool
    assert app._ingest_parse_pool_generation == 1
    assert app._ingest_parse_jobs_by_generation == {1: set()}
    assert isinstance(app._ingest_parse_pool_stop_event, threading.Event)

    teardown = app._shutdown_ingest_parse_pool()
    assert teardown is not None
    teardown.join(timeout=_FAKE_POOL_JOIN_TIMEOUT)
    assert not teardown.is_alive()


@pytest.mark.asyncio
async def test_submit_cap_backpressure_second_job_stays_queued_until_first_completes(
    tmp_path: Path,
) -> None:
    """(F3 pilot) The pool-size cap IS the backpressure: with N=1, a second
    submission must stay QUEUED until the first job's parse completes."""
    db = _make_db(tmp_path)
    source1 = _write_text_file(tmp_path, "note-1.txt", "First body.")
    source2 = _write_text_file(tmp_path, "note-2.txt", "Second body.")

    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(db, pool_factory=lambda: pool, worker_count=1)

    async with app.run_test() as pilot:
        job1 = app.submit_library_ingest_job(source_path=str(source1))
        job2 = app.submit_library_ingest_job(source_path=str(source2))
        await pilot.pause()

        jobs_by_id = {j.job_id: j for j in app.library_ingest_jobs.jobs()}
        assert jobs_by_id[job1.job_id].state == IngestJobState.PARSING
        assert jobs_by_id[job2.job_id].state == IngestJobState.QUEUED
        assert len(pool.calls) == 1

        # Manually complete job1's parse -- this must top up the pool and
        # promote job2 to PARSING.
        payload = {
            "media_type": "plaintext",
            "file_type": "plaintext",
            "title": "note-1",
            "author": "Unknown",
            "content": "First body.",
            "keywords": [],
            "url": f"file://{source1.absolute()}",
            "analysis_content": "",
            "chunks": None,
            "chunk_options": None,
            "metadata": {},
            "file_path": str(source1),
        }
        pool.trigger_success(0, {"ok": True, "payload": payload})

        for _ in range(_POLL_ATTEMPTS):
            jobs_by_id = {j.job_id: j for j in app.library_ingest_jobs.jobs()}
            if jobs_by_id[job2.job_id].state == IngestJobState.PARSING:
                break
            await pilot.pause(_POLL_INTERVAL)
        else:
            raise AssertionError(
                f"job2 never promoted to PARSING: {app.library_ingest_jobs.jobs()}"
            )

        assert len(pool.calls) == 2

        done1 = await _wait_for_job_state(app, pilot, job1.job_id, IngestJobState.DONE)
        assert done1.media_id is not None
        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_heavy_lane_caps_transcriptions_while_documents_fill_pool(
    tmp_path: Path,
) -> None:
    """(F3 pilot, heavy-lane cap) With worker_count=3 and heavy_lane=1, only
    one audio/video parse may run at a time -- a second transcription is
    skipped ahead of by queued documents, which fill the remaining pool
    slots. Completing the in-flight transcription frees the heavy slot and
    promotes the skipped one."""
    db = _make_db(tmp_path)
    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(
        db, pool_factory=lambda: pool, worker_count=3, heavy_lane=1
    )

    async with app.run_test() as pilot:
        paths = {}
        for name in ("a1.mp3", "a2.mp3", "d1.txt", "d2.txt", "d3.txt"):
            p = tmp_path / name
            p.write_text("x", encoding="utf-8")
            paths[name] = app.submit_library_ingest_job(source_path=str(p))
        await pilot.pause()

        # pool holds exactly 3: audio1 (heavy) + doc1 + doc2. audio2 is
        # skipped (heavy lane full); doc3 waits (pool full).
        assert len(pool.calls) == 3
        states = {j.job_id: j.state for j in app.library_ingest_jobs.jobs()}
        assert states[paths["a1.mp3"].job_id] == IngestJobState.PARSING
        assert states[paths["d1.txt"].job_id] == IngestJobState.PARSING
        assert states[paths["d2.txt"].job_id] == IngestJobState.PARSING
        assert states[paths["a2.mp3"].job_id] == IngestJobState.QUEUED
        assert states[paths["d3.txt"].job_id] == IngestJobState.QUEUED

        # completing audio1 frees the heavy slot -> audio2 is admitted next.
        pool.trigger_success(0, {"ok": True, "payload": {}})
        await _wait_for_job_state(
            app, pilot, paths["a2.mp3"].job_id, IngestJobState.PARSING
        )
        assert len(pool.calls) == 4


@pytest.mark.asyncio
async def test_retried_transcription_still_obeys_the_heavy_lane_cap(
    tmp_path: Path,
) -> None:
    """(task 160) The heavy-lane cap must hold on the retry path too. A
    requeued audio job carries its ``detected_type`` forward, so retrying a
    failed transcription while another transcription is already PARSING must
    leave the retry QUEUED -- not dispatch a second concurrent transcription
    (which the dropped-``detected_type`` bug allowed via the Home Retry
    control)."""
    db = _make_db(tmp_path)
    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(
        db, pool_factory=lambda: pool, worker_count=3, heavy_lane=1
    )

    async with app.run_test() as pilot:
        a1_path = tmp_path / "a1.mp3"
        a1_path.write_text("x", encoding="utf-8")
        a2_path = tmp_path / "a2.mp3"
        a2_path.write_text("x", encoding="utf-8")
        a1 = app.submit_library_ingest_job(source_path=str(a1_path))
        a2 = app.submit_library_ingest_job(source_path=str(a2_path))
        await pilot.pause()

        # Only one transcription parses at a time: a1 PARSING, a2 blocked.
        assert len(pool.calls) == 1
        states = {j.job_id: j.state for j in app.library_ingest_jobs.jobs()}
        assert states[a1.job_id] == IngestJobState.PARSING
        assert states[a2.job_id] == IngestJobState.QUEUED

        # Fail a1 (per-job structured failure, like the retry tests) -> its
        # heavy slot frees and a2 is admitted to PARSING.
        pool.trigger_success(0, {"ok": False, "error": "boom", "permanent": False})
        await _wait_for_job_state(app, pilot, a2.job_id, IngestJobState.PARSING)
        await _wait_for_job_state(app, pilot, a1.job_id, IngestJobState.FAILED)
        assert len(pool.calls) == 2

        # Retry a1 while a2 is still PARSING: the heavy lane is full, so the
        # requeued a1 must stay QUEUED (its detected_type='audio' is skipped),
        # NOT be dispatched as a second concurrent transcription.
        requeued = app.retry_library_ingest_job(a1.job_id)
        assert requeued is not None
        await pilot.pause()
        assert len(pool.calls) == 2
        states_after = {j.job_id: j.state for j in app.library_ingest_jobs.jobs()}
        assert states_after[requeued.job_id] == IngestJobState.QUEUED
        assert states_after[a2.job_id] == IngestJobState.PARSING


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["parakeet-onnx", "transcribe-cpp"])
async def test_eligible_local_stt_uses_executor_not_general_pool(
    tmp_path: Path,
    provider: str,
) -> None:
    pool = _FakeIngestParsePool(auto_run=False)
    executor = _FakeLocalSTTExecutor()
    app = _IngestRunnerHarness(
        _make_db(tmp_path),
        pool_factory=lambda: pool,
        local_stt_executor=executor,
        local_stt_dispatch_factory=_fake_local_stt_dispatch,
    )
    source = tmp_path / "speech.wav"
    source.write_bytes(b"fixture")

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(
            source_path=str(source),
            ingest_options={"audio_video": {"transcription_provider": provider}},
        )
        await pilot.pause()

        assert len(executor.calls) == 1
        assert executor.calls[0]["job_id"] == job.job_id
        assert executor.calls[0]["source"] == FileAudioSource(source)
        assert executor.calls[0]["options"]["transcription_provider"] == provider
        assert pool.calls == []


@pytest.mark.asyncio
async def test_library_retry_clears_executor_unhealthy_gate_explicitly(
    tmp_path: Path,
) -> None:
    executor = _FakeLocalSTTExecutor()
    app = _IngestRunnerHarness(
        _make_db(tmp_path),
        local_stt_executor=executor,
        local_stt_dispatch_factory=_fake_local_stt_dispatch,
    )
    source = tmp_path / "speech.wav"
    source.write_bytes(b"fixture")

    async with app.run_test() as pilot:
        original = app.submit_library_ingest_job(
            source_path=str(source),
            ingest_options={"audio_video": {"transcription_provider": "parakeet-onnx"}},
        )
        await pilot.pause()
        assert executor.calls[0]["explicit_retry"] is False
        executor.trigger_failure(
            0,
            ExecutorFailure(
                1,
                executor.calls[0]["attempt_id"],
                TranscriptionFailureCode.ENGINE_CRASHED,
                recovery_actions=("retry_faster_whisper",),
            ),
        )
        await _wait_for_job_state(
            app,
            pilot,
            original.job_id,
            IngestJobState.FAILED,
        )

        replacement = app.retry_library_ingest_job(original.job_id)
        assert replacement is not None
        for _ in range(_POLL_ATTEMPTS):
            if len(executor.calls) == 2:
                break
            await pilot.pause(_POLL_INTERVAL)
        assert len(executor.calls) == 2
        assert executor.calls[1]["explicit_retry"] is True


@pytest.mark.asyncio
async def test_parakeet_retry_keeps_job_local_override_and_scope_path_private(
    tmp_path: Path,
) -> None:
    executor = _FakeLocalSTTExecutor()
    app = _IngestRunnerHarness(
        _make_db(tmp_path),
        local_stt_executor=executor,
        local_stt_dispatch_factory=_fake_local_stt_dispatch,
    )
    source = tmp_path / "speech.wav"
    source.write_bytes(b"fixture")
    selected = str(tmp_path / "private-parakeet")
    scope_id = "library-external-retry-scope"

    async with app.run_test() as pilot:
        original = app.submit_library_ingest_job(
            source_path=str(source),
            ingest_options={
                "audio_video": {
                    "transcription_provider": "parakeet-onnx",
                    "transcription_model_dir": selected,
                    "transcription_external_scope_id": scope_id,
                }
            },
        )
        await pilot.pause()
        first_options = executor.calls[0]["options"]
        assert first_options["transcription_model_dir"] == selected
        assert first_options["transcription_context"]["external_scope_id"] == scope_id
        executor.trigger_failure(
            0,
            ExecutorFailure(
                1,
                executor.calls[0]["attempt_id"],
                TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
                recovery_actions=("retry_faster_whisper",),
            ),
        )
        failed = await _wait_for_job_state(
            app,
            pilot,
            original.job_id,
            IngestJobState.FAILED,
        )
        assert selected not in failed.error
        assert selected not in str(failed.error_detail)

        retry = app.retry_library_ingest_job(failed.job_id)
        assert retry is not None
        for _ in range(_POLL_ATTEMPTS):
            if len(executor.calls) == 2:
                break
            await pilot.pause(_POLL_INTERVAL)
        assert len(executor.calls) == 2
        retry_options = executor.calls[1]["options"]
        assert retry_options["transcription_model_dir"] == selected
        assert retry_options["transcription_context"]["external_scope_id"] == scope_id


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_local_stt_cancel_and_force_stop_are_exact_attempt_scoped(
    tmp_path: Path,
) -> None:
    pool = _FakeIngestParsePool(auto_run=False)
    executor = _FakeLocalSTTExecutor()
    app = _IngestRunnerHarness(
        _make_db(tmp_path),
        pool_factory=lambda: pool,
        local_stt_executor=executor,
        local_stt_dispatch_factory=_fake_local_stt_dispatch,
    )
    app._ingest_parse_pool = pool
    source = tmp_path / "speech.wav"
    source.write_bytes(b"fixture")

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(
            source_path=str(source),
            ingest_options={"audio_video": {"transcription_provider": "parakeet-onnx"}},
        )
        await pilot.pause()
        attempt_id = executor.calls[0]["attempt_id"]
        executor.trigger_event(
            0,
            ExecutorEvent(1, attempt_id, WorkerPhase.TRANSCRIBING),
        )
        await pilot.pause()

        assert app.cancel_local_ingest_job(job.job_id) is True
        assert executor.cancel_calls == [attempt_id]
        current = app.library_ingest_jobs.get_job(job.job_id)
        assert current is not None
        assert current.progress == {
            "phase": "transcribing",
            "message": "Transcribing audio",
            "cancel_requested": True,
        }
        app.library_ingest_jobs.update_progress(
            job.job_id,
            progress={**current.progress, "percent": 64.0},
            persist=False,
        )
        executor.trigger_event(
            0,
            ExecutorEvent(1, attempt_id, WorkerPhase.POST_PROCESSING),
        )
        await pilot.pause()
        current = app.library_ingest_jobs.get_job(job.job_id)
        assert current is not None
        assert current.progress == {
            "phase": "post-processing",
            "message": "Post-processing audio",
            "cancel_requested": True,
        }

        topups: list[str] = []
        app._top_up_ingest_parse_pool = lambda: topups.append("top-up")
        assert app.force_stop_local_ingest_job(job.job_id) is True
        for _ in range(_POLL_ATTEMPTS):
            if executor.force_stop_calls:
                break
            await pilot.pause(_POLL_INTERVAL)
        assert executor.force_stop_calls == [attempt_id]
        assert topups == []
        assert app._ingest_parse_pool is pool

        executor.complete_retirement()
        for _ in range(_POLL_ATTEMPTS):
            if topups:
                break
            await pilot.pause(_POLL_INTERVAL)
        assert topups == ["top-up"]


@pytest.mark.parametrize(
    ("cancel_requested", "expected_progress"),
    (
        (
            True,
            {
                "phase": "post-processing",
                "message": "Post-processing audio",
                "cancel_requested": True,
            },
        ),
        (
            "truthy-untrusted-value",
            {
                "phase": "post-processing",
                "message": "Post-processing audio",
            },
        ),
    ),
)
def test_local_stt_phase_replaces_untrusted_progress_and_preserves_only_true_cancel(
    cancel_requested: object,
    expected_progress: dict[str, Any],
) -> None:
    app = _IngestRunnerHarness(None)
    job = app.library_ingest_jobs.submit(source_path="speech.wav")
    assert app.library_ingest_jobs.mark_parsing(job.job_id) is not None
    attempt_id = "attempt-progress-replacement"
    app._ingest_local_stt_jobs[job.job_id] = (1, attempt_id)
    app.library_ingest_jobs.update_progress(
        job.job_id,
        progress={
            "phase": "transcribing",
            "message": "Stale message",
            "percent": 64.0,
            "cancel_requested": cancel_requested,
            "provider_private_detail": {"raw": "must not survive"},
        },
        persist=False,
    )

    app._on_ingest_local_stt_event(
        job.job_id,
        ExecutorEvent(1, attempt_id, WorkerPhase.POST_PROCESSING),
    )

    current = app.library_ingest_jobs.get_job(job.job_id)
    assert current is not None
    assert current.progress == expected_progress


@pytest.mark.asyncio
async def test_local_stt_cancel_rejects_stale_or_unbound_job(tmp_path: Path) -> None:
    executor = _FakeLocalSTTExecutor()
    app = _IngestRunnerHarness(
        _make_db(tmp_path),
        local_stt_executor=executor,
        local_stt_dispatch_factory=_fake_local_stt_dispatch,
    )

    async with app.run_test():
        assert app.cancel_local_ingest_job("missing-job") is False
        assert app.force_stop_local_ingest_job("missing-job") is False
        assert executor.cancel_calls == []
        assert executor.force_stop_calls == []


@pytest.mark.asyncio
async def test_executor_identity_build_and_submit_run_off_textual_thread(
    tmp_path: Path,
) -> None:
    executor = _FakeLocalSTTExecutor()
    dispatch_threads: list[int] = []

    def build_dispatch(job, options):
        dispatch_threads.append(threading.get_ident())
        return _fake_local_stt_dispatch(job, options)

    app = _IngestRunnerHarness(
        _make_db(tmp_path),
        local_stt_executor=executor,
        local_stt_dispatch_factory=build_dispatch,
    )
    source = tmp_path / "speech.wav"
    source.write_bytes(b"fixture")
    textual_thread = threading.get_ident()

    async with app.run_test() as pilot:
        app.submit_library_ingest_job(
            source_path=str(source),
            ingest_options={"audio_video": {"transcription_provider": "parakeet-onnx"}},
        )
        for _ in range(_POLL_ATTEMPTS):
            if executor.calls:
                break
            await pilot.pause(_POLL_INTERVAL)
        else:
            raise AssertionError("local STT dispatch never reached the executor")

        assert dispatch_threads
        assert dispatch_threads[0] != textual_thread
        assert executor.submit_thread_ident != textual_thread


@pytest.mark.asyncio
async def test_explicit_parakeet_directory_is_snapshotted_before_dispatch(
    tmp_path: Path,
) -> None:
    executor = _FakeLocalSTTExecutor()
    app = _IngestRunnerHarness(
        _make_db(tmp_path),
        local_stt_executor=executor,
    )
    source = tmp_path / "speech.wav"
    source.write_bytes(b"fixture")
    model_dir = tmp_path / "parakeet"
    model_dir.mkdir()
    for filename in (
        "config.json",
        "vocab.txt",
        "encoder-model.int8.onnx",
        "decoder_joint-model.int8.onnx",
    ):
        (model_dir / filename).write_bytes(filename.encode("utf-8"))
    _allow_test_external_root(app, model_dir)

    async with app.run_test() as pilot:
        app.submit_library_ingest_job(
            source_path=str(source),
            ingest_options={
                "audio_video": {
                    "transcription_provider": "parakeet-onnx",
                    "transcription_model_dir": str(model_dir),
                }
            },
        )
        await pilot.pause()

        assert len(executor.calls) == 1
        snapshot = executor.calls[0]["local_source"]
        identity = executor.calls[0]["identity"]
        assert snapshot is not None
        assert identity.local_snapshot_token == snapshot.token
        assert str(model_dir) not in repr(snapshot)


@pytest.mark.asyncio
async def test_explicit_parakeet_directory_uses_central_validated_path(
    tmp_path: Path,
) -> None:
    executor = _FakeLocalSTTExecutor()
    app = _IngestRunnerHarness(
        _make_db(tmp_path),
        local_stt_executor=executor,
    )
    source = tmp_path / "speech.wav"
    source.write_bytes(b"fixture")
    validated_dir = tmp_path / "validated-parakeet"
    validated_dir.mkdir()
    for filename in (
        "config.json",
        "vocab.txt",
        "encoder-model.int8.onnx",
        "decoder_joint-model.int8.onnx",
    ):
        (validated_dir / filename).write_bytes(filename.encode("utf-8"))
    _allow_test_external_root(app, validated_dir)

    with patch.object(
        _parakeet_external_module,
        "validate_path_simple",
        return_value=validated_dir,
    ):
        async with app.run_test() as pilot:
            app.submit_library_ingest_job(
                source_path=str(source),
                ingest_options={
                    "audio_video": {
                        "transcription_provider": "parakeet-onnx",
                        "transcription_model_dir": str(
                            tmp_path / "unvalidated-parakeet"
                        ),
                    }
                },
            )
            await pilot.pause()

    assert len(executor.calls) == 1
    assert executor.calls[0]["options"]["transcription_model_dir"] == str(validated_dir)


def test_parakeet_dispatch_delegates_to_shared_resolver_and_copies_updates(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Local_Ingestion.stt_batch_routing import PARAKEET_V2_MODEL
    from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey

    app = _IngestRunnerHarness(None)
    job = LibraryIngestJob(job_id="job-shared-resolver", source_path="speech.wav")
    identity = ModelIdentity(
        provider_id="parakeet-onnx",
        model_id=PARAKEET_V2_MODEL,
        root_revision="fixture-revision",
        closure_fingerprint="fixture-closure",
        precision="f32",
        device=ExecutionDevice.CPU,
    )
    resolved = ParakeetDispatch(
        identity=identity,
        local_source=None,
        managed_store_root=tmp_path / "managed",
        managed_artifact_ref=("artifact", "revision", "variant"),
        option_updates=MappingProxyType(
            {
                "transcription_model_dir": str(tmp_path / "resolved"),
                "_verify_legacy_parakeet_v2": True,
            }
        ),
        managed_dependency_refs=(("silero-vad", "vad-revision", "f32"),),
    )
    requested = tmp_path / "requested"
    requested.mkdir()
    for filename in (
        "config.json",
        "vocab.txt",
        "encoder-model.onnx",
        "encoder-model.onnx.data",
        "decoder_joint-model.onnx",
    ):
        (requested / filename).write_bytes(filename.encode())
    options = {
        "transcription_provider": "parakeet-onnx",
        "transcription_model": PARAKEET_V2_MODEL,
        "transcription_precision": "f32",
        "transcription_model_dir": str(requested),
    }

    resolver = SimpleNamespace(resolve=lambda *_args, **_kwargs: resolved)
    with patch.object(app, "_ensure_parakeet_source_service", return_value=resolver):
        with patch.object(resolver, "resolve", wraps=resolver.resolve) as resolve:
            dispatch = app._build_local_stt_dispatch(job, options)

    resolve.assert_called_once_with(
        ParakeetSourceKey.V2_F32,
        override=str(requested),
        scope_id="job-shared-resolver",
    )
    assert dispatch == {
        "attempt_id": "job-shared-resolver-attempt-1",
        "identity": identity,
        "local_source": None,
        "managed_store_root": tmp_path / "managed",
        "managed_artifact_ref": ("artifact", "revision", "variant"),
        "managed_dependency_refs": (("silero-vad", "vad-revision", "f32"),),
    }
    assert options["transcription_model_dir"] == str(tmp_path / "resolved")
    assert options["_verify_legacy_parakeet_v2"] is True


def test_transcribe_cpp_dispatch_stays_on_gguf_resolution(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    app = _IngestRunnerHarness(None)
    options = {
        "transcription_provider": "transcribe-cpp",
        "transcription_precision": "native",
        "transcription_context": {"model_path": str(model_path)},
    }
    admission = SimpleNamespace(
        path=model_path,
        metadata=SimpleNamespace(architecture="whisper"),
    )

    with (
        patch(
            "tldw_chatbook.Model_Artifacts.gguf_admission.validate_local_gguf",
            return_value=admission,
        ),
        patch.object(
            _parakeet_dispatch_module,
            "resolve_parakeet_dispatch",
            side_effect=AssertionError("Parakeet resolver used for transcribe.cpp"),
        ),
    ):
        dispatch = app._build_local_stt_dispatch(
            LibraryIngestJob(job_id="job-gguf", source_path="speech.wav"),
            options,
        )

    assert dispatch["identity"].provider_id == "transcribe-cpp"
    assert dispatch["identity"].model_id == "local-gguf:whisper"
    assert dispatch["identity"].device is ExecutionDevice.AUTO
    assert dispatch["local_source"].paths == (model_path,)
    assert "transcription_model_dir" not in options


def test_managed_parakeet_dispatch_selects_exact_model_and_precision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.Local_Ingestion import parakeet_v2_artifact as artifacts
    from tldw_chatbook.Local_Ingestion.stt_batch_routing import PARAKEET_V3_MODEL

    reference = artifacts.parakeet_reference(PARAKEET_V3_MODEL, "f32")
    handle = SimpleNamespace(
        root=reference,
        closure_fingerprint="exact-closure-fingerprint",
    )

    class _Lease:
        def __init__(self) -> None:
            self.handle = handle
            self.closed = False

        def close(self) -> None:
            self.closed = True

    lease = _Lease()
    service = SimpleNamespace(acquire=lambda selected: lease)
    monkeypatch.setattr(
        _parakeet_dispatch_module,
        "active_managed_parakeet_dir",
        lambda model, precision, service=None: tmp_path / "managed-root",
    )
    monkeypatch.setattr(
        _parakeet_dispatch_module,
        "parakeet_v2_managed_service",
        lambda: service,
    )
    monkeypatch.setattr(
        _parakeet_dispatch_module,
        "managed_model_artifact_root",
        lambda: tmp_path / "managed",
    )

    app = _IngestRunnerHarness(None)
    job = LibraryIngestJob(job_id="job-v3-f32", source_path="speech.wav")
    dispatch = app._build_local_stt_dispatch(
        job,
        {
            "transcription_provider": "parakeet-onnx",
            "transcription_model": PARAKEET_V3_MODEL,
            "transcription_precision": "f32",
        },
    )

    assert dispatch["identity"].model_id == PARAKEET_V3_MODEL
    assert dispatch["identity"].precision == "f32"
    assert dispatch["identity"].root_revision == reference.revision
    assert dispatch["identity"].closure_fingerprint == "exact-closure-fingerprint"
    assert dispatch["managed_artifact_ref"] == (
        reference.artifact_id,
        reference.revision,
        reference.variant,
    )
    assert lease.closed is True


def test_explicit_f32_folder_snapshots_the_f32_payload_files(tmp_path: Path) -> None:
    from tldw_chatbook.Local_Ingestion.stt_batch_routing import PARAKEET_V2_MODEL

    model_root = tmp_path / "parakeet-f32"
    model_root.mkdir()
    required = (
        "config.json",
        "vocab.txt",
        "encoder-model.onnx",
        "encoder-model.onnx.data",
        "decoder_joint-model.onnx",
    )
    for filename in required:
        (model_root / filename).write_bytes(filename.encode())
    app = _IngestRunnerHarness(None)
    _allow_test_external_root(app, model_root)
    job = LibraryIngestJob(job_id="job-v2-f32", source_path="speech.wav")

    dispatch = app._build_local_stt_dispatch(
        job,
        {
            "transcription_provider": "parakeet-onnx",
            "transcription_model": PARAKEET_V2_MODEL,
            "transcription_precision": "f32",
            "transcription_model_dir": str(model_root),
        },
    )

    assert dispatch["local_source"].paths == tuple(
        model_root / item for item in sorted(required)
    )


def test_unqualified_legacy_v2_folder_cannot_satisfy_a_v3_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.Local_Ingestion.stt_batch_routing import PARAKEET_V3_MODEL

    legacy_v2 = tmp_path / "legacy-v2-int8"
    legacy_v2.mkdir()
    for filename in (
        "config.json",
        "vocab.txt",
        "encoder-model.int8.onnx",
        "decoder_joint-model.int8.onnx",
    ):
        (legacy_v2 / filename).write_bytes(filename.encode())
    monkeypatch.setattr(
        _app_module,
        "get_cli_setting",
        lambda key, *args: str(legacy_v2)
        if key == "transcription.parakeet_onnx_model_dir"
        else args[0]
        if args
        else None,
    )
    monkeypatch.setattr(
        _parakeet_dispatch_module,
        "active_managed_parakeet_dir",
        lambda model, precision, service=None: None,
    )
    monkeypatch.setattr(
        _parakeet_dispatch_module,
        "parakeet_v2_managed_service",
        lambda: SimpleNamespace(),
    )
    app = _IngestRunnerHarness(None)

    with pytest.raises(FileNotFoundError, match="No installed Parakeet artifact"):
        app._build_local_stt_dispatch(
            LibraryIngestJob(job_id="job-v3", source_path="speech.wav"),
            {
                "transcription_provider": "parakeet-onnx",
                "transcription_model": PARAKEET_V3_MODEL,
                "transcription_precision": "int8",
            },
        )


@pytest.mark.asyncio
async def test_faster_whisper_stays_in_general_pool(tmp_path: Path) -> None:
    pool = _FakeIngestParsePool(auto_run=False)
    executor = _FakeLocalSTTExecutor()
    app = _IngestRunnerHarness(
        _make_db(tmp_path),
        pool_factory=lambda: pool,
        local_stt_executor=executor,
        local_stt_dispatch_factory=_fake_local_stt_dispatch,
    )
    source = tmp_path / "speech.wav"
    source.write_bytes(b"fixture")

    async with app.run_test() as pilot:
        app.submit_library_ingest_job(
            source_path=str(source),
            ingest_options={
                "audio_video": {"transcription_provider": "faster-whisper"}
            },
        )
        await pilot.pause()

        assert len(pool.calls) == 1
        assert executor.calls == []


@pytest.mark.asyncio
async def test_executor_heavy_job_leaves_remaining_pool_slots_for_documents(
    tmp_path: Path,
) -> None:
    pool = _FakeIngestParsePool(auto_run=False)
    executor = _FakeLocalSTTExecutor()
    app = _IngestRunnerHarness(
        _make_db(tmp_path),
        pool_factory=lambda: pool,
        worker_count=3,
        heavy_lane=1,
        local_stt_executor=executor,
        local_stt_dispatch_factory=_fake_local_stt_dispatch,
    )
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"fixture")
    documents = [
        _write_text_file(tmp_path, f"doc-{index}.txt", f"body {index}")
        for index in range(3)
    ]

    async with app.run_test() as pilot:
        app.submit_library_ingest_job(
            source_path=str(audio),
            ingest_options={"audio_video": {"transcription_provider": "parakeet-onnx"}},
        )
        for document in documents:
            app.submit_library_ingest_job(source_path=str(document))
        await pilot.pause()

        assert len(executor.calls) == 1
        assert len(pool.calls) == 2
        assert app.library_ingest_jobs.counts()["parsing"] == 3


# Windows Proactor event-loop setup owns an internal loopback socket pair.
@pytest.mark.allow_network
@pytest.mark.asyncio
async def test_dictation_reservation_gates_only_heavy_library_work(
    tmp_path: Path,
) -> None:
    pool = _FakeIngestParsePool(auto_run=False)
    executor = _FakeLocalSTTExecutor()
    app = _IngestRunnerHarness(
        _make_db(tmp_path),
        pool_factory=lambda: pool,
        worker_count=2,
        heavy_lane=1,
        local_stt_executor=executor,
        local_stt_dispatch_factory=_fake_local_stt_dispatch,
    )
    coordinator = app._ensure_local_stt_dispatch_coordinator()
    coordinator.begin_dictation(
        capture_generation=1,
        dispatch=_fake_parakeet_dispatch(),
        sample_rate=16_000,
        channels=1,
        sample_width=2,
        language="en",
        on_logical_segment=lambda _sequence, _text: None,
    )
    audio = tmp_path / "reserved.wav"
    audio.write_bytes(b"fixture")
    document = _write_text_file(tmp_path, "document.txt", "document body")

    async with app.run_test() as pilot:
        audio_job = app.submit_library_ingest_job(
            source_path=str(audio),
            ingest_options={"audio_video": {"transcription_provider": "parakeet-onnx"}},
        )
        document_job = app.submit_library_ingest_job(source_path=str(document))
        await pilot.pause()

        states = {job.job_id: job.state for job in app.library_ingest_jobs.jobs()}
        assert states[audio_job.job_id] is IngestJobState.QUEUED
        assert states[document_job.job_id] is IngestJobState.PARSING
        assert len(pool.calls) == 1
        assert pool.calls[0]["args"][0] == document_job.source_path
        assert executor.calls == []
        reserved_audio = app.library_ingest_jobs.get_job(audio_job.job_id)
        assert reserved_audio is not None
        assert reserved_audio.error == ""


@pytest.mark.asyncio
async def test_dictation_race_defers_claimed_library_job_without_failure(
    tmp_path: Path,
) -> None:
    pool = _FakeIngestParsePool(auto_run=False)
    executor = _FakeLocalSTTExecutor()
    dispatch_started = threading.Event()
    release_dispatch = threading.Event()
    dispatch_count = 0

    def build_dispatch(job, options):
        nonlocal dispatch_count
        dispatch_count += 1
        if dispatch_count == 1:
            dispatch_started.set()
            assert release_dispatch.wait(5.0)
        return _fake_local_stt_dispatch(job, options)

    app = _IngestRunnerHarness(
        _make_db(tmp_path),
        pool_factory=lambda: pool,
        worker_count=2,
        heavy_lane=1,
        local_stt_executor=executor,
        local_stt_dispatch_factory=build_dispatch,
    )
    audio = tmp_path / "raced.wav"
    audio.write_bytes(b"fixture")
    document = _write_text_file(tmp_path, "ordered.txt", "document body")
    later_document = _write_text_file(tmp_path, "later.txt", "later body")

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(
            source_path=str(audio),
            ingest_options={"audio_video": {"transcription_provider": "parakeet-onnx"}},
        )
        assert dispatch_started.wait(1.0)
        coordinator = app._ensure_local_stt_dispatch_coordinator()
        handle = coordinator.begin_dictation(
            capture_generation=11,
            dispatch=_fake_parakeet_dispatch(),
            sample_rate=16_000,
            channels=1,
            sample_width=2,
            language="en",
            on_logical_segment=lambda _sequence, _text: None,
        )
        document_job = app.submit_library_ingest_job(source_path=str(document))
        later_document_job = app.submit_library_ingest_job(
            source_path=str(later_document)
        )
        await pilot.pause()
        current_document = app.library_ingest_jobs.get_job(document_job.job_id)
        current_later_document = app.library_ingest_jobs.get_job(
            later_document_job.job_id
        )
        assert current_document is not None
        assert current_later_document is not None
        assert current_document.state is IngestJobState.PARSING
        assert current_later_document.state is IngestJobState.QUEUED
        assert len(pool.calls) == 1
        assert pool.calls[0]["args"][0] == document_job.source_path

        release_dispatch.set()
        for _ in range(_POLL_ATTEMPTS):
            if job.job_id not in app._ingest_local_stt_jobs:
                break
            await pilot.pause(_POLL_INTERVAL)

        deferred = app.library_ingest_jobs.get_job(job.job_id)
        assert deferred is not None
        assert deferred.state is IngestJobState.QUEUED
        assert deferred.retry_count == 0
        assert deferred.error == ""
        assert deferred.error_detail is None
        assert deferred.stt_failure_provenance is None
        assert [item.job_id for item in app.library_ingest_jobs.jobs()] == [
            later_document_job.job_id,
            document_job.job_id,
            job.job_id,
        ]
        assert executor.calls == []
        assert coordinator.dictation_reserved is True
        deferred_later_document = app.library_ingest_jobs.get_job(
            later_document_job.job_id
        )
        assert deferred_later_document is not None
        assert deferred_later_document.state is IngestJobState.PARSING
        assert len(pool.calls) == 2

        pool.trigger_success(
            0,
            {"ok": False, "error": "document failed", "permanent": False},
        )
        await _wait_for_job_state(
            app,
            pilot,
            document_job.job_id,
            IngestJobState.FAILED,
        )

        handle.append_segment(b"\x00\x00")
        handle.finish()
        assert [call["job_id"] for call in executor.calls] == [None]
        dictation_attempt = executor.calls[0]["attempt_id"]
        executor.trigger_result(
            0,
            ExecutorResult(
                1,
                dictation_attempt,
                {"logical_segments": ["dictated"]},
            ),
        )
        for _ in range(_POLL_ATTEMPTS):
            if len(executor.calls) >= 2:
                break
            await pilot.pause(_POLL_INTERVAL)

        resumed = app.library_ingest_jobs.get_job(job.job_id)
        assert resumed is not None
        assert resumed.state is IngestJobState.PARSING
        assert resumed.retry_count == 0
        assert resumed.error == ""
        assert resumed.error_detail is None
        assert [call["job_id"] for call in executor.calls] == [None, job.job_id]
        still_parsing = app.library_ingest_jobs.get_job(later_document_job.job_id)
        assert still_parsing is not None
        assert still_parsing.state is IngestJobState.PARSING


# Windows Proactor event-loop setup owns an internal loopback socket pair.
@pytest.mark.allow_network
@pytest.mark.asyncio
async def test_library_terminal_hands_executor_to_pending_dictation_before_top_up(
    tmp_path: Path,
) -> None:
    executor = _FakeLocalSTTExecutor()
    app = _IngestRunnerHarness(
        _make_db(tmp_path),
        worker_count=1,
        heavy_lane=1,
        local_stt_executor=executor,
        local_stt_dispatch_factory=_fake_local_stt_dispatch,
    )
    first_path = tmp_path / "first.wav"
    first_path.write_bytes(b"first")
    second_path = tmp_path / "second.wav"
    second_path.write_bytes(b"second")

    async with app.run_test() as pilot:
        first = app.submit_library_ingest_job(
            source_path=str(first_path),
            ingest_options={"audio_video": {"transcription_provider": "parakeet-onnx"}},
        )
        await pilot.pause()
        coordinator = app._ensure_local_stt_dispatch_coordinator()
        handle = coordinator.begin_dictation(
            capture_generation=2,
            dispatch=_fake_parakeet_dispatch(),
            sample_rate=16_000,
            channels=1,
            sample_width=2,
            language="en",
            on_logical_segment=lambda _sequence, _text: None,
        )
        handle.append_segment(b"\x00\x00")
        handle.finish()
        second = app.submit_library_ingest_job(
            source_path=str(second_path),
            ingest_options={"audio_video": {"transcription_provider": "parakeet-onnx"}},
        )
        await pilot.pause()
        first_attempt = executor.calls[0]["attempt_id"]

        executor.trigger_failure(
            0,
            ExecutorFailure(
                1,
                first_attempt,
                TranscriptionFailureCode.CANCELLED,
            ),
        )
        for _ in range(_POLL_ATTEMPTS):
            if len(executor.calls) >= 2:
                break
            await pilot.pause(_POLL_INTERVAL)

        assert [call["job_id"] for call in executor.calls] == [first.job_id, None]
        second_job = app.library_ingest_jobs.get_job(second.job_id)
        assert second_job is not None
        assert second_job.state is IngestJobState.QUEUED

        dictation_attempt = executor.calls[1]["attempt_id"]
        executor.trigger_result(
            1,
            ExecutorResult(
                1,
                dictation_attempt,
                {"logical_segments": ["dictated"]},
            ),
        )
        for _ in range(_POLL_ATTEMPTS):
            if len(executor.calls) >= 3:
                break
            await pilot.pause(_POLL_INTERVAL)

        assert [call["job_id"] for call in executor.calls] == [
            first.job_id,
            None,
            second.job_id,
        ]


@pytest.mark.asyncio
async def test_executor_admits_only_one_local_job_when_legacy_heavy_cap_is_higher(
    tmp_path: Path,
) -> None:
    pool = _FakeIngestParsePool(auto_run=False)
    executor = _FakeLocalSTTExecutor()
    app = _IngestRunnerHarness(
        _make_db(tmp_path),
        pool_factory=lambda: pool,
        worker_count=3,
        heavy_lane=2,
        local_stt_executor=executor,
        local_stt_dispatch_factory=_fake_local_stt_dispatch,
    )
    first_audio = tmp_path / "first.wav"
    first_audio.write_bytes(b"fixture")
    second_audio = tmp_path / "second.wav"
    second_audio.write_bytes(b"fixture")
    document = _write_text_file(tmp_path, "document.txt", "document body")

    async with app.run_test() as pilot:
        first_job = app.submit_library_ingest_job(
            source_path=str(first_audio),
            ingest_options={"audio_video": {"transcription_provider": "parakeet-onnx"}},
        )
        second_job = app.submit_library_ingest_job(
            source_path=str(second_audio),
            ingest_options={"audio_video": {"transcription_provider": "parakeet-onnx"}},
        )
        document_job = app.submit_library_ingest_job(source_path=str(document))
        await pilot.pause()

        states = {job.job_id: job.state for job in app.library_ingest_jobs.jobs()}
        assert len(executor.calls) == 1
        assert executor.calls[0]["job_id"] == first_job.job_id
        assert states[second_job.job_id] == IngestJobState.QUEUED
        assert states[document_job.job_id] == IngestJobState.PARSING
        assert len(pool.calls) == 1

        executor.trigger_failure(
            0,
            ExecutorFailure(
                1,
                executor.calls[0]["attempt_id"],
                TranscriptionFailureCode.CANCELLED,
            ),
        )
        await _wait_for_job_state(
            app, pilot, first_job.job_id, IngestJobState.CANCELLED
        )
        await _wait_for_job_state(app, pilot, second_job.job_id, IngestJobState.PARSING)
        assert len(executor.calls) == 2
        assert executor.calls[1]["job_id"] == second_job.job_id


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_executor_callbacks_are_fenced_and_progress_has_no_percentage(
    tmp_path: Path,
) -> None:
    executor = _FakeLocalSTTExecutor()
    app = _IngestRunnerHarness(
        _make_db(tmp_path),
        local_stt_executor=executor,
        local_stt_dispatch_factory=_fake_local_stt_dispatch,
    )
    store = _RecordingIngestJobStore()
    app.library_ingest_jobs.attach_store(store)
    source = tmp_path / "speech.wav"
    source.write_bytes(b"fixture")
    wakes: list[str] = []
    app._start_library_ingest_queue_if_idle = lambda: wakes.append("writer")

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(
            source_path=str(source),
            ingest_options={"audio_video": {"transcription_provider": "parakeet-onnx"}},
        )
        await pilot.pause()
        call = executor.calls[0]
        attempt_id = call["attempt_id"]
        persisted_before_tick = tuple(store.upserts)

        executor.trigger_event(
            0,
            ExecutorEvent(1, attempt_id, WorkerPhase.TRANSCRIBING),
        )
        await pilot.pause()
        current = app.library_ingest_jobs.get_job(job.job_id)
        assert current is not None
        assert current.progress == {
            "phase": "transcribing",
            "message": "Transcribing audio",
        }
        assert "percent" not in current.progress
        assert tuple(store.upserts) == persisted_before_tick

        executor.trigger_result(
            0,
            ExecutorResult(2, attempt_id, {"content": "stale"}),
        )
        await pilot.pause()
        assert job.job_id not in app._ingest_parsed_payloads

        payload = {"content": "accepted"}
        executor.trigger_result(0, ExecutorResult(1, attempt_id, payload))
        await pilot.pause()
        assert app._ingest_parsed_payloads[job.job_id] == payload
        assert wakes == ["writer"]

        executor.trigger_failure(
            0,
            ExecutorFailure(
                1,
                attempt_id,
                TranscriptionFailureCode.ENGINE_CRASHED,
            ),
        )
        await pilot.pause()
        current = app.library_ingest_jobs.get_job(job.job_id)
        assert current is not None and current.state == IngestJobState.PARSING


@pytest.mark.asyncio
async def test_executor_cpu_retry_generation_is_accepted_after_preparing_event(
    tmp_path: Path,
) -> None:
    executor = _FakeLocalSTTExecutor()
    app = _IngestRunnerHarness(
        _make_db(tmp_path),
        local_stt_executor=executor,
        local_stt_dispatch_factory=_fake_local_stt_dispatch,
    )
    source = tmp_path / "speech.wav"
    source.write_bytes(b"fixture")
    app._start_library_ingest_queue_if_idle = lambda: None

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(
            source_path=str(source),
            ingest_options={"audio_video": {"transcription_provider": "parakeet-onnx"}},
        )
        await pilot.pause()
        attempt_id = executor.calls[0]["attempt_id"]

        executor.trigger_event(
            0,
            ExecutorEvent(2, attempt_id, WorkerPhase.PREPARING),
        )
        await pilot.pause()
        executor.trigger_result(
            0,
            ExecutorResult(2, attempt_id, {"content": "cpu retry"}),
        )
        await pilot.pause()

        assert app._ingest_parsed_payloads[job.job_id] == {"content": "cpu retry"}


@pytest.mark.asyncio
async def test_executor_terminal_can_bind_before_submitted_callback(
    tmp_path: Path,
) -> None:
    executor = _FakeLocalSTTExecutor()
    app = _IngestRunnerHarness(
        _make_db(tmp_path),
        local_stt_executor=executor,
        local_stt_dispatch_factory=_fake_local_stt_dispatch,
    )
    source = tmp_path / "speech.wav"
    source.write_bytes(b"fixture")

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(
            source_path=str(source),
            ingest_options={"audio_video": {"transcription_provider": "parakeet-onnx"}},
        )
        await pilot.pause()
        attempt_id = executor.calls[0]["attempt_id"]
        app._ingest_local_stt_jobs[job.job_id] = (0, attempt_id)

        app._on_ingest_local_stt_failure(
            job.job_id,
            ExecutorFailure(
                1,
                attempt_id,
                TranscriptionFailureCode.ENGINE_CRASHED,
                recovery_actions=("retry_faster_whisper",),
            ),
        )
        app._on_ingest_local_stt_submitted(job.job_id, 1, attempt_id)

        terminal = app.library_ingest_jobs.get_job(job.job_id)
        assert terminal is not None
        assert terminal.state is IngestJobState.FAILED
        assert job.job_id not in app._ingest_local_stt_jobs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "expected_state"),
    [
        (TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE, IngestJobState.FAILED),
        (TranscriptionFailureCode.ENGINE_CRASHED, IngestJobState.FAILED),
        (TranscriptionFailureCode.CANCELLED, IngestJobState.CANCELLED),
    ],
)
async def test_executor_failure_uses_stable_job_terminal(
    tmp_path: Path,
    code: TranscriptionFailureCode,
    expected_state: IngestJobState,
) -> None:
    executor = _FakeLocalSTTExecutor()
    app = _IngestRunnerHarness(
        _make_db(tmp_path),
        local_stt_executor=executor,
        local_stt_dispatch_factory=_fake_local_stt_dispatch,
    )
    source = tmp_path / "speech.wav"
    source.write_bytes(b"fixture")

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(
            source_path=str(source),
            ingest_options={"audio_video": {"transcription_provider": "parakeet-onnx"}},
        )
        await pilot.pause()
        attempt_id = executor.calls[0]["attempt_id"]
        executor.trigger_failure(
            0,
            ExecutorFailure(
                1,
                attempt_id,
                code,
                recovery_actions=("retry_faster_whisper",),
            ),
        )
        terminal = await _wait_for_job_state(app, pilot, job.job_id, expected_state)

        assert terminal.error
        assert "fixture" not in terminal.error
        if code is not TranscriptionFailureCode.CANCELLED:
            assert terminal.error_detail == {
                "category": "stt_failure",
                "code": code.value,
                "message": terminal.error,
                "actions": ["retry_faster_whisper"],
            }


@pytest.mark.asyncio
async def test_parakeet_failed_attempt_reaches_faster_whisper_retry_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.Local_Ingestion.transcription_service import (
        TranscriptionService,
    )
    from tldw_chatbook.STT.persistence import (
        load_transcription_provenance_document,
    )

    monkeypatch.setattr(
        TranscriptionService,
        "transcribe",
        lambda self, audio_path, **kwargs: {
            "text": "Recovered with faster whisper.",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.5,
                    "text": "Recovered with faster whisper.",
                }
            ],
            "language": "en",
            "language_probability": 0.99,
            "duration": 1.5,
            "provider": "faster-whisper",
            "model": kwargs.get("model") or "base",
        },
    )
    pool = _FakeIngestParsePool()
    executor = _FakeLocalSTTExecutor()
    db = _make_db(tmp_path)
    app = _IngestRunnerHarness(
        db,
        pool_factory=lambda: pool,
        local_stt_executor=executor,
        local_stt_dispatch_factory=_fake_local_stt_dispatch,
    )
    source = tmp_path / "speech.wav"
    source.write_bytes(b"fixture")

    async with app.run_test() as pilot:
        original = app.submit_library_ingest_job(
            source_path=str(source),
            ingest_options={
                "audio_video": {"transcription_provider": "parakeet-onnx"}
            },
        )
        await pilot.pause()
        attempt_id = executor.calls[0]["attempt_id"]
        failed_attempt = {
            "attempt_id": attempt_id,
            "batch_id": original.batch_id,
            "job_id": original.job_id,
            "provider_id": "parakeet-onnx",
            "model_id": "nemo-parakeet-tdt-0.6b-v2",
            "artifact_root": {
                "artifact_id": "parakeet-v2",
                "revision": "root-revision",
                "variant": "int8",
            },
            "artifact_dependencies": [
                {
                    "artifact_id": "silero-vad",
                    "revision": "vad-revision",
                    "variant": "f32",
                }
            ],
            "precision": "int8",
            "requested_device": "cpu",
            "effective_device": "cpu",
            "requested_language": "en",
            "effective_language": "en",
            "detected_language": None,
            "task": "transcribe",
            "error_code": "artifact_incompatible",
        }
        executor.trigger_failure(
            0,
            ExecutorFailure(
                1,
                attempt_id,
                TranscriptionFailureCode.ARTIFACT_INCOMPATIBLE,
                recovery_actions=("retry_faster_whisper",),
                failed_attempt=failed_attempt,
            ),
        )
        failed = await _wait_for_job_state(
            app,
            pilot,
            original.job_id,
            IngestJobState.FAILED,
        )

        retry = app.retry_library_ingest_job_with_provider(
            failed.job_id,
            "faster-whisper",
        )
        await pilot.pause()

        assert retry is not None
        assert retry.retry_of_job_id == failed.job_id
        assert retry.retry_source_failure_provenance == failed_attempt
        assert retry.ingest_options["audio_video"]["transcription_provider"] == (
            "faster-whisper"
        )
        done = await _wait_for_job_state(
            app,
            pilot,
            retry.job_id,
            IngestJobState.DONE,
        )
        assert done.media_id is not None
        row = db.get_media_by_id(done.media_id)
        assert row is not None
        provenance = load_transcription_provenance_document(
            row["transcription_provenance_json"]
        )
        assert provenance["provider_id"] == "faster-whisper"
        assert provenance["model_id"] == "base"
        assert provenance["job_id"] == retry.job_id
        assert provenance["retry_of_attempt_id"] == failed_attempt["attempt_id"]
        assert provenance["retry_of_job_id"] == failed.job_id
        assert provenance["failed_attempt"] == failed_attempt


@pytest.mark.asyncio
async def test_executor_start_failure_does_not_retire_general_pool(
    tmp_path: Path,
) -> None:
    pool = _FakeIngestParsePool(auto_run=False)
    executor = _FakeLocalSTTExecutor(
        submit_error=ExecutorUnavailableError("executor unavailable")
    )
    app = _IngestRunnerHarness(
        _make_db(tmp_path),
        pool_factory=lambda: pool,
        worker_count=2,
        local_stt_executor=executor,
        local_stt_dispatch_factory=_fake_local_stt_dispatch,
    )
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"fixture")
    document = _write_text_file(tmp_path, "document.txt", "document body")

    async with app.run_test() as pilot:
        failed_job = app.submit_library_ingest_job(
            source_path=str(audio),
            ingest_options={"audio_video": {"transcription_provider": "parakeet-onnx"}},
        )
        document_job = app.submit_library_ingest_job(source_path=str(document))
        failed = await _wait_for_job_state(
            app, pilot, failed_job.job_id, IngestJobState.FAILED
        )

        assert failed.permanent is False
        assert len(pool.calls) == 1
        assert pool.calls[0]["args"][0] == document_job.source_path
        assert app._ingest_parse_pool is pool


def test_shutdown_closes_local_executor_off_caller_thread(tmp_path: Path) -> None:
    executor = _FakeLocalSTTExecutor()
    app = _IngestRunnerHarness(
        _make_db(tmp_path),
        local_stt_executor=executor,
    )
    caller_thread = threading.get_ident()

    teardown = app._shutdown_ingest_parse_pool()
    assert teardown is not None
    teardown.join(timeout=5.0)

    assert app._ingest_shutdown is True
    assert app._local_stt_executor is None
    assert executor.close_thread_ident is not None
    assert executor.close_thread_ident != caller_thread


def test_shutdown_closes_and_detaches_coordinator_before_executor_teardown(
    tmp_path: Path,
) -> None:
    class _BlockingExecutor(_FakeLocalSTTExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.close_started = threading.Event()
            self.close_release = threading.Event()

        def close(self) -> None:
            self.close_started.set()
            assert self.close_release.wait(5.0)
            super().close()

    executor = _BlockingExecutor()
    app = _IngestRunnerHarness(_make_db(tmp_path), local_stt_executor=executor)
    coordinator = app._ensure_local_stt_dispatch_coordinator()
    handle = coordinator.begin_dictation(
        capture_generation=3,
        dispatch=_fake_parakeet_dispatch(),
        sample_rate=16_000,
        channels=1,
        sample_width=2,
        language="en",
        on_logical_segment=lambda _sequence, _text: None,
    )

    started = time.monotonic()
    teardown = app._shutdown_ingest_parse_pool()
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert app._local_stt_dispatch_coordinator is None
    assert app._local_stt_executor is None
    with pytest.raises(RuntimeError, match="closed"):
        coordinator.begin_dictation(
            capture_generation=4,
            dispatch=_fake_parakeet_dispatch(),
            sample_rate=16_000,
            channels=1,
            sample_width=2,
            language="en",
            on_logical_segment=lambda _sequence, _text: None,
        )
    with pytest.raises(RuntimeError) as cancelled:
        handle.wait()
    assert cancelled.value.args == (TranscriptionFailureCode.CANCELLED,)
    assert executor.cancel_calls == []
    assert executor.close_started.wait(1.0)
    assert executor.close_thread_ident is None

    executor.close_release.set()
    assert teardown is not None
    teardown.join(timeout=5.0)
    assert not teardown.is_alive()
    assert executor.close_thread_ident is not None


def test_shutdown_cooperatively_cancels_active_dictation_before_executor_close(
    tmp_path: Path,
) -> None:
    executor = _FakeLocalSTTExecutor()
    app = _IngestRunnerHarness(_make_db(tmp_path), local_stt_executor=executor)
    coordinator = app._ensure_local_stt_dispatch_coordinator()
    handle = coordinator.begin_dictation(
        capture_generation=5,
        dispatch=_fake_parakeet_dispatch(),
        sample_rate=16_000,
        channels=1,
        sample_width=2,
        language="en",
        on_logical_segment=lambda _sequence, _text: None,
    )
    handle.append_segment(b"\x00\x00")
    attempt_id = executor.calls[0]["attempt_id"]

    teardown = app._shutdown_ingest_parse_pool()

    assert executor.cancel_calls == [attempt_id]
    assert app._local_stt_dispatch_coordinator is None
    assert app._local_stt_executor is None
    assert teardown is not None
    teardown.join(timeout=5.0)
    assert not teardown.is_alive()


def test_shutdown_thread_waits_for_executor_and_parse_pool(tmp_path: Path) -> None:
    class _BlockingExecutor(_FakeLocalSTTExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.close_started = threading.Event()
            self.close_release = threading.Event()

        def close(self) -> None:
            self.close_started.set()
            assert self.close_release.wait(5.0)
            super().close()

    executor = _BlockingExecutor()
    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(
        _make_db(tmp_path),
        pool_factory=lambda: pool,
        local_stt_executor=executor,
    )
    app._ingest_parse_pool = pool

    teardown = app._shutdown_ingest_parse_pool()
    assert teardown is not None
    assert executor.close_started.wait(1.0)
    teardown.join(timeout=0.05)
    assert teardown.is_alive()

    executor.close_release.set()
    teardown.join(timeout=_FAKE_POOL_JOIN_TIMEOUT)

    assert not teardown.is_alive()
    assert executor.close_thread_ident is not None
    assert pool.terminated is True


def test_local_stt_marshal_failure_logs_callback_context(tmp_path: Path) -> None:
    app = _IngestRunnerHarness(_make_db(tmp_path))

    def safe_callback() -> None:
        return None

    with (
        patch.object(app, "call_from_thread", side_effect=RuntimeError("closed")),
        patch("tldw_chatbook.app.logger") as logger,
    ):
        app._marshal_local_stt_call(safe_callback)

    logger.error.assert_called_once_with(
        "Library local STT callback could not be marshaled (callback={}).",
        "safe_callback",
    )


@pytest.mark.asyncio
async def test_shutdown_flag_stops_late_parse_completion_callbacks(
    tmp_path: Path,
) -> None:
    """(F3 pilot) Once ``_ingest_shutdown`` is set, a parse completion (or
    pool-level error) that lands afterward -- e.g. already in flight when
    the app started closing -- must be a pure no-op: no registry mutation,
    no pool top-up, no pool drop."""
    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "note.txt", "irrelevant")
    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(db, pool_factory=lambda: pool)

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(source_path=str(source))
        await pilot.pause()

        current = next(
            j for j in app.library_ingest_jobs.jobs() if j.job_id == job.job_id
        )
        assert current.state == IngestJobState.PARSING

        app._ingest_shutdown = True

        # Late success completion: must be a no-op.
        app._on_ingest_parse_complete(
            app._ingest_parse_pool_generation,
            job.job_id,
            {"ok": True, "payload": {"file_type": "plaintext"}},
        )
        current = next(
            j for j in app.library_ingest_jobs.jobs() if j.job_id == job.job_id
        )
        assert current.state == IngestJobState.PARSING
        assert job.job_id not in app._ingest_parsed_payloads
        assert app.library_ingest_jobs.runner_active is False

        # Late pool-level error: also a no-op -- the pool is not dropped.
        app._handle_broken_ingest_parse_pool(
            app._ingest_parse_pool_generation,
            job.job_id,
            RuntimeError("late failure"),
        )
        current = next(
            j for j in app.library_ingest_jobs.jobs() if j.job_id == job.job_id
        )
        assert current.state == IngestJobState.PARSING
        assert app._ingest_parse_pool is not None


@pytest.mark.asyncio
async def test_parse_completion_preserves_direct_local_stt_failure_provenance(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path)
    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(db, pool_factory=lambda: pool)
    source = tmp_path / "speech.wav"
    source.write_bytes(b"fixture")
    failed_attempt = {
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
        "error_code": "artifact_incompatible",
    }

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(source_path=str(source))
        await pilot.pause()
        pool.trigger_success(
            0,
            {
                "ok": False,
                "error": "The selected GGUF cannot be used by transcribe.cpp.",
                "permanent": False,
                "error_detail": {
                    "category": "stt_failure",
                    "code": "artifact_incompatible",
                    "message": "The selected GGUF cannot be used by transcribe.cpp.",
                    "actions": [
                        "choose_another_gguf",
                        "retry_faster_whisper",
                    ],
                },
                "stt_failure_provenance": failed_attempt,
            },
        )
        await _wait_for_job_state(app, pilot, job.job_id, IngestJobState.FAILED)

        failed = app.library_ingest_jobs.get_job(job.job_id)
        assert failed.error_detail["actions"] == [
            "choose_another_gguf",
            "retry_faster_whisper",
        ]
        assert failed.stt_failure_provenance == failed_attempt


# --- F3 Task 4 review fixes: quit-deadlock guard + payload-sparing ----------
#
# The Task 4 review found a quit-time deadlock race: Textual's
# `call_from_thread` blocks the calling thread on the marshaled call's
# result, and CPython's `Pool._terminate_pool` does an unbounded
# `result_handler.join()`. If a parse completed right as the user quit, the
# pool's result-handler thread could park inside `call_from_thread` while
# `on_unmount` (the loop thread) parked inside `pool.terminate()` waiting
# for that same result-handler thread -- mutual deadlock, app hangs on
# quit. The deadlock itself is race-timed, so these tests pin the two
# OBSERVABLE contracts of the fix instead: (a) the pool-side callbacks
# check `_ingest_shutdown` BEFORE marshaling (never entering
# `call_from_thread` at all once the flag is up), and (b) quit-path
# terminate/join runs on a detached daemon thread, never the caller's
# (loop) thread, so the loop stays free to drain any in-flight marshaled
# call.


def test_pool_callbacks_short_circuit_without_marshaling_when_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(Quit-deadlock guard, layer a) `_ingest_pool_callback`/
    `_ingest_pool_error_callback` -- which run on the pool's result-handler
    thread -- must return WITHOUT calling `call_from_thread` once
    `_ingest_shutdown` is set. The pre-fix lambdas marshaled
    unconditionally (the shutdown check only ran later, inside the
    already-marshaled UI-thread body -- too late to prevent the
    result-handler thread from blocking)."""
    db = _make_db(tmp_path)
    app = _IngestRunnerHarness(db)

    marshaled: list[tuple] = []
    monkeypatch.setattr(
        app, "call_from_thread", lambda *args, **kwargs: marshaled.append(args)
    )

    app._ingest_shutdown = True
    app._ingest_pool_callback(1, "ingest-job-1", {"ok": True, "payload": {}})
    app._ingest_pool_error_callback(
        1, "ingest-job-1", RuntimeError("late pool failure")
    )
    assert marshaled == []

    # Positive control: with the flag down, both callbacks marshal.
    app._ingest_shutdown = False
    app._ingest_pool_callback(1, "ingest-job-1", {"ok": True, "payload": {}})
    app._ingest_pool_error_callback(1, "ingest-job-1", RuntimeError("pool failure"))
    assert len(marshaled) == 2


@pytest.mark.parametrize(
    ("callback_name", "callback_value"),
    [
        ("_ingest_pool_callback", {"ok": True, "payload": {}}),
        ("_ingest_pool_error_callback", RuntimeError("pool failure")),
    ],
)
def test_pool_callback_ignores_cancelled_marshal_only_during_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    callback_name: str,
    callback_value: Any,
) -> None:
    """A callback already past the shutdown check may lose its UI future."""

    app = _IngestRunnerHarness(_make_db(tmp_path))
    marshal_entered = threading.Event()
    release_marshal = threading.Event()

    def cancelled_marshal(*_args: Any, **_kwargs: Any) -> None:
        marshal_entered.set()
        assert release_marshal.wait(5.0)
        raise concurrent.futures.CancelledError

    monkeypatch.setattr(app, "call_from_thread", cancelled_marshal)
    callback = getattr(app, callback_name)
    callback_errors: list[BaseException] = []

    def invoke_callback() -> None:
        try:
            callback(1, "ingest-job-1", callback_value)
        except BaseException as exc:  # noqa: BLE001 - assert thread outcome below
            callback_errors.append(exc)

    callback_thread = threading.Thread(target=invoke_callback, daemon=True)
    callback_thread.start()
    assert marshal_entered.wait(1.0)
    app._ingest_shutdown = True
    release_marshal.set()
    callback_thread.join(timeout=5.0)

    assert not callback_thread.is_alive()
    assert callback_errors == []

    app._ingest_shutdown = False
    with pytest.raises(concurrent.futures.CancelledError):
        callback(1, "ingest-job-2", callback_value)


def test_shutdown_terminates_pool_off_the_caller_thread(tmp_path: Path) -> None:
    """(Quit-deadlock guard, layer b) `_shutdown_ingest_parse_pool` must
    set the shutdown flag, detach the pool reference, and run
    `terminate()`/`join()` on a DIFFERENT thread than the caller's (in
    production the caller is `on_unmount`, i.e. the app's event-loop
    thread -- exactly the thread that must never block on the pool's
    result-handler join)."""
    db = _make_db(tmp_path)
    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(db, pool_factory=lambda: pool)
    app._ingest_parse_pool = pool

    caller_ident = threading.get_ident()
    teardown_thread = app._shutdown_ingest_parse_pool()

    # Synchronous effects, guaranteed before the method returns: flag up,
    # pool reference detached (nothing can submit to it anymore).
    assert app._ingest_shutdown is True
    assert app._ingest_parse_pool is None

    assert teardown_thread is not None
    assert teardown_thread.daemon is True
    teardown_thread.join(timeout=_FAKE_POOL_JOIN_TIMEOUT)
    assert not teardown_thread.is_alive()
    assert pool.terminated is True
    assert pool.terminate_thread_ident is not None
    assert pool.terminate_thread_ident != caller_ident


def test_shutdown_detaches_and_cleans_progress_resources_off_caller_thread(
    tmp_path: Path,
) -> None:
    pool = _FakeIngestParsePool(auto_run=False)
    progress_queue = _ClosableQueue()
    resources = _app_module._IngestParsePoolResources(pool, progress_queue)
    app = _IngestRunnerHarness(
        _make_db(tmp_path),
        pool_factory=lambda: resources,
    )
    app._ensure_ingest_parse_pool()
    stop_event = app._ingest_parse_pool_stop_event
    progress_thread = app._ingest_parse_progress_thread
    caller_ident = threading.get_ident()

    teardown_thread = app._shutdown_ingest_parse_pool()

    assert stop_event is not None and stop_event.is_set()
    assert app._ingest_parse_pool is None
    assert app._ingest_parse_progress_queue is None
    assert app._ingest_parse_progress_thread is None
    assert teardown_thread is not None
    teardown_thread.join(timeout=_FAKE_POOL_JOIN_TIMEOUT)
    assert not teardown_thread.is_alive()
    assert progress_thread is not None and not progress_thread.is_alive()
    assert pool.terminated is True
    assert pool.terminate_thread_ident not in {None, caller_ident}
    assert pool.join_thread_ident not in {None, caller_ident}
    assert progress_queue.closed is True
    assert progress_queue.cancelled_join is True
    assert progress_queue.close_thread_ident not in {None, caller_ident}
    assert progress_queue.cancel_thread_ident not in {None, caller_ident}


def test_broken_pool_detaches_and_cleans_progress_resources_off_caller_thread(
    tmp_path: Path,
) -> None:
    pool = _FakeIngestParsePool(auto_run=False)
    progress_queue = _ClosableQueue()
    resources = _app_module._IngestParsePoolResources(pool, progress_queue)
    app = _IngestRunnerHarness(
        _make_db(tmp_path),
        pool_factory=lambda: resources,
    )
    app._ensure_ingest_parse_pool()
    generation = app._ingest_parse_pool_generation
    stop_event = app._ingest_parse_pool_stop_event
    progress_thread = app._ingest_parse_progress_thread
    caller_ident = threading.get_ident()

    app._handle_broken_ingest_parse_pool(
        generation,
        None,
        RuntimeError("worker exited"),
    )

    assert stop_event is not None and stop_event.is_set()
    assert app._ingest_parse_pool is None
    assert app._ingest_parse_progress_queue is None
    assert app._ingest_parse_progress_thread is None
    deadline = time.monotonic() + 5.0
    while not progress_queue.closed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pool.terminated is True
    assert pool.terminate_thread_ident not in {None, caller_ident}
    assert pool.join_thread_ident not in {None, caller_ident}
    assert progress_queue.closed is True
    assert progress_queue.cancelled_join is True
    assert progress_queue.close_thread_ident not in {None, caller_ident}
    assert progress_queue.cancel_thread_ident not in {None, caller_ident}
    assert progress_thread is not None
    progress_thread.join(timeout=1.0)
    assert not progress_thread.is_alive()


def test_shutdown_with_no_pool_still_sets_flag_and_returns_none(tmp_path: Path) -> None:
    """`_shutdown_ingest_parse_pool` with no pool ever created: the flag
    still goes up (late callbacks must no-op regardless), no thread is
    spawned."""
    db = _make_db(tmp_path)
    app = _IngestRunnerHarness(db)

    assert app._shutdown_ingest_parse_pool() is None
    assert app._ingest_shutdown is True


@pytest.mark.asyncio
async def test_broken_pool_spares_payload_ready_job_and_writer_drains_it(
    tmp_path: Path,
) -> None:
    """(Task 4 review fix) A job whose parse already COMPLETED (payload
    sitting in `_ingest_parsed_payloads`, job still `PARSING` because the
    writer hasn't claimed it yet) needs nothing further from the pool --
    a pool-level failure must NOT fail it and throw the finished parse
    away. Only jobs still genuinely mid-parse fail (retryable); the
    handler wakes the writer so the surviving payload drains to DONE."""
    db = _make_db(tmp_path)
    source_a = _write_text_file(tmp_path, "note-a.txt", "Payload-ready body.")
    source_b = _write_text_file(tmp_path, "note-b.txt", "Mid-parse body.")
    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(db, pool_factory=lambda: pool, worker_count=2)

    async with app.run_test() as pilot:
        job_a = app.submit_library_ingest_job(source_path=str(source_a))
        job_b = app.submit_library_ingest_job(source_path=str(source_b))
        await pilot.pause()

        jobs_by_id = {j.job_id: j for j in app.library_ingest_jobs.jobs()}
        assert jobs_by_id[job_a.job_id].state == IngestJobState.PARSING
        assert jobs_by_id[job_b.job_id].state == IngestJobState.PARSING

        # Make job A payload-ready WITHOUT routing through
        # `_on_ingest_parse_complete` (which would wake the writer and let
        # it claim A out of PARSING before the pool breaks): stash the
        # payload directly -- exactly the state a real completion leaves
        # behind in the window before the writer's claim lands.
        payload_a = {
            "media_type": "plaintext",
            "file_type": "plaintext",
            "title": "note-a",
            "author": "Unknown",
            "content": "Payload-ready body.",
            "keywords": [],
            "url": f"file://{source_a.absolute()}",
            "analysis_content": "",
            "chunks": None,
            "chunk_options": None,
            "metadata": {},
            "file_path": str(source_a),
        }
        app._ingest_parsed_payloads[job_a.job_id] = payload_a

        # The pool dies while B is still genuinely mid-parse.
        pool.trigger_error(0, RuntimeError("simulated worker death"))

        failed_b = await _wait_for_job_state(
            app, pilot, job_b.job_id, IngestJobState.FAILED
        )
        assert failed_b.permanent is False

        # A must survive: never failed, and the handler's writer wake
        # drains its already-finished parse to DONE with a real media row.
        done_a = await _wait_for_job_state(
            app, pilot, job_a.job_id, IngestJobState.DONE
        )
        assert done_a.media_id is not None
        assert db.get_media_by_id(done_a.media_id) is not None

        assert app._ingest_parse_pool is None
        await _wait_for_runner_idle(app, pilot)


# --- Live-QA crash fix: Textual's fileno-less stderr vs. the resource tracker
#
# Served-TUI QA found the app dying on the FIRST ingest submission: under
# Textual (app mode / textual-serve), `sys.stderr` is replaced by a capture
# object whose `fileno()` returns -1 WITHOUT raising. CPython 3.12's
# `multiprocessing.resource_tracker._launch` appends `sys.stderr.fileno()`
# to the fds it passes to `util.spawnv_passfds` (its `except Exception`
# guard never fires because -1 is returned, not raised), and
# `spawnv_passfds` rejects the list with `ValueError: bad value(s) in
# fds_to_keep` -- so the very first `get_context("spawn").Pool(...)` (which
# ensure-runs the process-global resource tracker) exploded, propagated up
# the top-up path on the UI thread, and crashed the app.


class _ClosableQueue:
    """Queue double exposing multiprocessing queue cleanup observations."""

    def __init__(self) -> None:
        self.closed = False
        self.cancelled_join = False
        self.close_thread_ident: int | None = None
        self.cancel_thread_ident: int | None = None
        self._items: queue.Queue[Any] = queue.Queue()

    def get(self, timeout: float) -> Any:
        return self._items.get(timeout=timeout)

    def close(self) -> None:
        self.closed = True
        self.close_thread_ident = threading.get_ident()

    def cancel_join_thread(self) -> None:
        self.cancelled_join = True
        self.cancel_thread_ident = threading.get_ident()


def _bare_ingest_mixin() -> LibraryIngestQueueMixin:
    mixin = LibraryIngestQueueMixin()
    mixin._ingest_parse_worker_count = lambda: 1
    mixin._ingest_shutdown = False
    return mixin


def test_create_pool_returns_progress_resources_and_uses_combined_initializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _Context:
        def Queue(self, maxsize: int) -> _ClosableQueue:
            captured["maxsize"] = maxsize
            progress_queue = _ClosableQueue()
            captured["progress_queue"] = progress_queue
            return progress_queue

        def Pool(self, **kwargs: Any) -> _FakeIngestParsePool:
            captured.update(kwargs)
            return _FakeIngestParsePool(auto_run=False)

    monkeypatch.setattr(multiprocessing, "get_context", lambda _name: _Context())

    resources = LibraryIngestQueueMixin._create_ingest_parse_pool(
        _bare_ingest_mixin()
    )

    assert resources.progress_queue is captured["progress_queue"]
    assert captured["maxsize"] == INGEST_PARSE_PROGRESS_QUEUE_MAXSIZE
    assert captured["initializer"] is initialize_ingest_parse_worker
    assert captured["initargs"] == (resources.progress_queue,)


def test_create_pool_progress_resources_close_queue_when_pool_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress_queue = _ClosableQueue()

    class _Context:
        def Queue(self, maxsize: int) -> _ClosableQueue:
            assert maxsize == INGEST_PARSE_PROGRESS_QUEUE_MAXSIZE
            return progress_queue

        def Pool(self, **_kwargs: Any) -> Any:
            raise RuntimeError("pool construction failed")

    monkeypatch.setattr(multiprocessing, "get_context", lambda _name: _Context())

    with pytest.raises(RuntimeError, match="pool construction failed"):
        LibraryIngestQueueMixin._create_ingest_parse_pool(_bare_ingest_mixin())

    assert progress_queue.closed is True
    assert progress_queue.cancelled_join is True


def test_partial_pool_construction_cleanup_logs_operation_and_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed queue cleanup must identify its operation and resource type."""
    from loguru import logger

    class _FailingConstructionQueue:
        def close(self) -> None:
            raise RuntimeError("close failed")

        def cancel_join_thread(self) -> None:
            raise RuntimeError("cancel failed")

    progress_queue = _FailingConstructionQueue()

    class _Context:
        def Queue(self, maxsize: int) -> _FailingConstructionQueue:
            assert maxsize == INGEST_PARSE_PROGRESS_QUEUE_MAXSIZE
            return progress_queue

        def Pool(self, **_kwargs: Any) -> Any:
            raise RuntimeError("pool construction failed")

    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="ERROR",
    )
    monkeypatch.setattr(multiprocessing, "get_context", lambda _name: _Context())
    try:
        with pytest.raises(RuntimeError, match="pool construction failed"):
            LibraryIngestQueueMixin._create_ingest_parse_pool(_bare_ingest_mixin())
    finally:
        logger.remove(sink_id)

    assert messages == [
        "Error cleaning up a partially constructed Library ingest progress queue "
        "(operation=close, queue_type=_FailingConstructionQueue).",
        "Error cleaning up a partially constructed Library ingest progress queue "
        "(operation=cancel_join_thread, queue_type=_FailingConstructionQueue).",
    ]


def test_detached_progress_queue_cleanup_logs_operation_and_resource() -> None:
    """Detached cleanup failures must retain actionable queue context."""
    from loguru import logger

    class _FailingDetachedQueue:
        def close(self) -> None:
            raise RuntimeError("close failed")

        def cancel_join_thread(self) -> None:
            raise RuntimeError("cancel failed")

    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="ERROR",
    )
    try:
        teardown = LibraryIngestQueueMixin._shutdown_ingest_workers_off_thread(
            None,
            None,
            None,
            None,
            _FailingDetachedQueue(),
            None,
        )
        teardown.join(timeout=5.0)
        assert not teardown.is_alive()
    finally:
        logger.remove(sink_id)

    assert messages == [
        "Error cleaning up the Library ingest progress queue "
        "(operation=close, queue_type=_FailingDetachedQueue).",
        "Error cleaning up the Library ingest progress queue "
        "(operation=cancel_join_thread, queue_type=_FailingDetachedQueue).",
    ]


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_pool_submission_binds_generation_and_job_and_applies_transient_progress(
    tmp_path: Path,
) -> None:
    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(_make_db(tmp_path), pool_factory=lambda: pool)
    store = _RecordingIngestJobStore()
    app.library_ingest_jobs.attach_store(store)

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(
            source_path=str(_write_text_file(tmp_path, "progress.txt", "body"))
        )
        await pilot.pause()
        generation = app._ingest_parse_pool_generation
        persisted_before_tick = tuple(store.upserts)
        lifecycle_notifications: list[str] = []
        progress_notifications: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []
        app.library_ingest_jobs.add_listener(
            lambda: lifecycle_notifications.append("lifecycle")
        )
        app.library_ingest_jobs.add_progress_listener(
            lambda before, after: progress_notifications.append(
                (before.progress, after.progress)
            )
        )

        assert pool.calls[0]["args"][2] == (generation, job.job_id)
        app._on_ingest_parse_progress_batch(
            generation,
            (
                ParseProgressEvent(
                    generation,
                    job.job_id,
                    "extracting",
                    "Extracting page 1 of 4",
                    25.0,
                ),
            ),
        )

        current = app.library_ingest_jobs.get_job(job.job_id)
        assert current is not None
        assert current.progress == {
            "phase": "extracting",
            "message": "Extracting page 1 of 4",
            "percent": 25.0,
        }
        assert tuple(store.upserts) == persisted_before_tick
        assert lifecycle_notifications == []
        assert progress_notifications == [
            (
                None,
                {
                    "phase": "extracting",
                    "message": "Extracting page 1 of 4",
                    "percent": 25.0,
                },
            )
        ]


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_parse_progress_batch_revalidates_nominal_events_and_ignores_unknown_data(
    tmp_path: Path,
) -> None:
    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(_make_db(tmp_path), pool_factory=lambda: pool)

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(
            source_path=str(_write_text_file(tmp_path, "revalidate.txt", "body"))
        )
        await pilot.pause()
        generation = app._ingest_parse_pool_generation

        class _HostileQueueItem:
            @property
            def generation(self) -> int:
                raise RuntimeError("malformed IPC property")

        app._on_ingest_parse_progress_batch(
            generation,
            (
                object(),
                _HostileQueueItem(),
                ParseProgressEvent(
                    generation,
                    job.job_id,
                    "provider-private-stage",
                    "raw provider data",
                    90.0,
                ),
                ParseProgressEvent(
                    generation,
                    job.job_id,
                    "extracting",
                    "Extracting page 2\nof 4\x00",
                    float("inf"),
                ),
            ),
        )

        current = app.library_ingest_jobs.get_job(job.job_id)
        assert current is not None
        assert current.progress == {
            "phase": "extracting",
            "message": "Extracting page 2 of 4",
        }


@pytest.mark.parametrize(
    "fence",
    (
        "shutdown",
        "handler_generation",
        "event_generation",
        "generation_membership",
        "job_missing",
        "non_parsing",
        "terminal",
        "hidden",
        "payload_ready",
    ),
)
@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_parse_progress_batch_rejects_stale_or_ineligible_events(
    tmp_path: Path,
    fence: str,
) -> None:
    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(_make_db(tmp_path), pool_factory=lambda: pool)

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(
            source_path=str(_write_text_file(tmp_path, f"{fence}.txt", "body"))
        )
        await pilot.pause()
        generation = app._ingest_parse_pool_generation
        app.library_ingest_jobs.update_progress(
            job.job_id,
            progress={
                "phase": "inspecting",
                "message": "Before stale event",
                "percent": 5.0,
            },
            persist=False,
        )
        handler_generation = generation
        event_generation = generation
        event_job_id = job.job_id

        if fence == "shutdown":
            app._ingest_shutdown = True
        elif fence == "handler_generation":
            handler_generation += 1
            event_generation = handler_generation
            app._ingest_parse_jobs_by_generation[handler_generation] = {job.job_id}
        elif fence == "event_generation":
            event_generation += 1
        elif fence == "generation_membership":
            app._ingest_parse_jobs_by_generation[generation].remove(job.job_id)
        elif fence == "job_missing":
            event_job_id = "ingest-job-missing"
            app._ingest_parse_jobs_by_generation[generation].add(event_job_id)
        elif fence == "non_parsing":
            assert app.library_ingest_jobs.mark_writing(job.job_id) is not None
        elif fence == "terminal":
            assert app.library_ingest_jobs.mark_failed(
                job.job_id, error="settled"
            ) is not None
        elif fence == "hidden":
            assert app.library_ingest_jobs.mark_failed(
                job.job_id, error="hidden"
            ) is not None
            assert app.library_ingest_jobs.dismiss(job.job_id) is not None
        elif fence == "payload_ready":
            app._ingest_parsed_payloads[job.job_id] = {"content": "ready"}
        else:  # pragma: no cover - parameter table is exhaustive
            raise AssertionError(f"unknown fence: {fence}")

        before = app.library_ingest_jobs.get_job(job.job_id)
        assert before is not None
        app._on_ingest_parse_progress_batch(
            handler_generation,
            (
                ParseProgressEvent(
                    event_generation,
                    event_job_id,
                    "extracting",
                    "After stale event",
                    75.0,
                ),
            ),
        )

        after = app.library_ingest_jobs.get_job(job.job_id)
        assert after is not None
        assert after.progress == before.progress


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_late_progress_after_parse_completion_cannot_replace_payload_receipt(
    tmp_path: Path,
) -> None:
    pool = _FakeIngestParsePool(auto_run=False)
    app = _IngestRunnerHarness(_make_db(tmp_path), pool_factory=lambda: pool)
    app._start_library_ingest_queue_if_idle = lambda: None

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(
            source_path=str(_write_text_file(tmp_path, "complete.txt", "body"))
        )
        await pilot.pause()
        generation = app._ingest_parse_pool_generation
        app.library_ingest_jobs.update_progress(
            job.job_id,
            progress={"phase": "extracting", "message": "Parse receipt"},
            persist=False,
        )

        payload = {"content": "parsed"}
        app._on_ingest_parse_complete(
            generation,
            job.job_id,
            {"ok": True, "payload": payload},
        )
        app._on_ingest_parse_progress_batch(
            generation,
            (
                ParseProgressEvent(
                    generation,
                    job.job_id,
                    "extracting",
                    "Late extraction",
                    99.0,
                ),
            ),
        )

        current = app.library_ingest_jobs.get_job(job.job_id)
        assert current is not None
        assert app._ingest_parsed_payloads[job.job_id] == payload
        assert current.progress == {
            "phase": "extracting",
            "message": "Parse receipt",
        }


def test_progress_drain_coalesces_latest_event_with_injected_clock() -> None:
    first = ParseProgressEvent(4, "ingest-job-1", "extracting", "first", 10.0)
    latest = ParseProgressEvent(4, "ingest-job-1", "extracting", "latest", 30.0)

    class _ProgressQueue:
        def __init__(self) -> None:
            self.events = [first, latest]

        def get(self, timeout: float) -> ParseProgressEvent:
            assert timeout == 0.05
            if self.events:
                return self.events.pop(0)
            raise queue.Empty

    mixin = _bare_ingest_mixin()
    stop_event = threading.Event()

    def handler(*_args: Any) -> None:
        return None

    mixin._on_ingest_parse_progress_batch = handler
    marshaled: list[tuple[Any, ...]] = []

    def _capture_marshal(callback: Any, *args: Any) -> None:
        marshaled.append((callback, *args))
        stop_event.set()

    mixin._marshal_ingest_pool_call = _capture_marshal
    clock_values = iter((10.0, 10.1, 10.25))

    thread = mixin._start_ingest_parse_progress_drain(
        4,
        _ProgressQueue(),
        stop_event,
        clock=lambda: next(clock_values),
    )
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert marshaled == [(handler, 4, (latest,))]


def test_progress_drain_ignores_hostile_item_and_marshals_later_valid_event() -> None:
    valid = ParseProgressEvent(4, "ingest-job-1", "extracting", "valid", 30.0)

    class _HostileQueueItem:
        generation = 4

        @property
        def job_id(self) -> str:
            raise RuntimeError("hostile IPC attribute")

    class _ProgressQueue:
        def __init__(self) -> None:
            self.events: list[Any] = [_HostileQueueItem(), valid]

        def get(self, timeout: float) -> Any:
            assert timeout == 0.05
            if self.events:
                return self.events.pop(0)
            raise queue.Empty

    mixin = _bare_ingest_mixin()
    stop_event = threading.Event()

    def handler(*_args: Any) -> None:
        return None

    mixin._on_ingest_parse_progress_batch = handler
    marshaled: list[tuple[Any, ...]] = []

    def _capture_marshal(callback: Any, *args: Any) -> None:
        marshaled.append((callback, *args))
        stop_event.set()

    mixin._marshal_ingest_pool_call = _capture_marshal
    clock_values = iter((10.0, 10.1, 10.25))

    thread = mixin._start_ingest_parse_progress_drain(
        4,
        _ProgressQueue(),
        stop_event,
        clock=lambda: next(clock_values),
    )
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert stop_event.is_set()
    assert marshaled == [(handler, 4, (valid,))]


def test_progress_drain_does_not_marshal_event_released_after_generation_stop() -> None:
    event = ParseProgressEvent(4, "ingest-job-1", "extracting", "late", 40.0)
    get_entered = threading.Event()
    release_get = threading.Event()

    class _BlockingProgressQueue:
        def get(self, timeout: float) -> ParseProgressEvent:
            assert timeout == 0.05
            get_entered.set()
            assert release_get.wait(1.0)
            return event

    mixin = _bare_ingest_mixin()
    stop_event = threading.Event()
    marshaled: list[tuple[Any, ...]] = []

    def handler(*_args: Any) -> None:
        return None

    def _capture_marshal(callback: Any, *args: Any) -> None:
        marshaled.append((callback, *args))

    mixin._on_ingest_parse_progress_batch = handler
    mixin._marshal_ingest_pool_call = _capture_marshal
    clock_values = iter((10.0, 10.25))

    thread = mixin._start_ingest_parse_progress_drain(
        4,
        _BlockingProgressQueue(),
        stop_event,
        clock=lambda: next(clock_values),
    )
    assert get_entered.wait(1.0)
    stop_event.set()
    release_get.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert marshaled == []


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows spawn/resource-tracker boundary",
)
def test_create_pool_real_windows_spawn_progress_delivery_and_cleanup(
    tmp_path: Path,
) -> None:
    source = _write_text_file(
        tmp_path,
        "spawn-progress.txt",
        "Parsed inside a spawned worker with progress.",
    )
    mixin = _bare_ingest_mixin()
    resources = mixin._create_ingest_parse_pool()
    cleanup: threading.Thread | None = None
    cleaned = False
    try:
        result = resources.pool.apply_async(
            run_parse_job,
            (
                str(source),
                {"title": "Spawn progress"},
                (9, "ingest-job-windows-spawn"),
            ),
        ).get(timeout=120)
        event = resources.progress_queue.get(timeout=120)

        cleanup = mixin._shutdown_ingest_workers_off_thread(
            None,
            None,
            None,
            resources.pool,
            resources.progress_queue,
            None,
        )
        cleanup.join(timeout=10.0)
        cleaned = not cleanup.is_alive()

        assert result["ok"] is True
        assert result["payload"]["title"] == "Spawn progress"
        assert event.generation == 9
        assert event.job_id == "ingest-job-windows-spawn"
        assert event.phase == "inspecting"
        assert cleaned, "real parse-pool progress cleanup exceeded 10 seconds"
    finally:
        if not cleaned:
            resources.pool.terminate()
            resources.pool.join()
            resources.progress_queue.close()
            resources.progress_queue.cancel_join_thread()


def _run_isolated_python(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    """Run `code` in a FRESH interpreter (mirrors the Task 2 import-weight
    helper). Fresh matters here: the multiprocessing resource tracker is
    process-global and starts exactly once, so only a brand-new process is
    guaranteed to exercise its launch path (an earlier in-process test that
    touched multiprocessing would have already started it, silently turning
    the repro into a no-op)."""
    data_home = tmp_path / "data"
    config_home = tmp_path / "config"
    home = tmp_path / "home"
    for path in (data_home, config_home, home):
        path.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "XDG_DATA_HOME": str(data_home),
        "XDG_CONFIG_HOME": str(config_home),
        "HOME": str(home),
        "PYTHONPATH": str(_REPO_ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)

    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=_REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )


class _TextualLikeStderr:
    """Mimics Textual's stderr capture object: ``fileno()`` returns -1
    WITHOUT raising (the exact shape that defeats the resource tracker's
    ``except Exception`` guard)."""

    def fileno(self) -> int:
        return -1

    def write(self, *args: Any, **kwargs: Any) -> int:
        return 0

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


@pytest.mark.integration
def test_create_pool_survives_filenoless_stderr_real_spawn(tmp_path: Path) -> None:
    """Real repro, fresh interpreter: with ``sys.stderr`` swapped for a
    Textual-shaped capture object (fileno() == -1) BEFORE any
    multiprocessing use, the real ``_create_ingest_parse_pool`` must still
    construct a working spawn Pool (this is where the resource tracker
    launches) and round-trip a trivial ``apply_async``. RED pre-fix: the
    subprocess died with ``ValueError: bad value(s) in fds_to_keep``."""
    result = _run_isolated_python(
        tmp_path,
        """
        import sys


        class _TextualLikeStderr:
            def fileno(self):
                return -1

            def write(self, *args, **kwargs):
                return 0

            def flush(self):
                pass

            def isatty(self):
                return False


        if __name__ == "__main__":
            sys.stderr = _TextualLikeStderr()

            from tldw_chatbook.app import LibraryIngestQueueMixin

            mixin = LibraryIngestQueueMixin()
            # Instance shadow: one worker keeps the spawn cost bounded.
            mixin._ingest_parse_worker_count = lambda: 1

            resources = mixin._create_ingest_parse_pool()
            pool = resources.pool
            try:
                result = pool.apply_async(pow, (2, 3)).get(timeout=120)
                assert result == 8, result
            finally:
                pool.terminate()
                pool.join()
                resources.progress_queue.close()
                resources.progress_queue.cancel_join_thread()
            print("POOL_OK")
        """,
    )
    assert result.returncode == 0, (
        f"pool creation under fileno-less stderr failed:\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert "POOL_OK" in result.stdout


def test_create_pool_redirects_to_real_stderr_when_fileno_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-process anchor for the helper's swap behavior (the raw repro only
    triggers in a fresh interpreter -- see ``_run_isolated_python``'s
    docstring): with ``sys.stderr``'s fileno() invalid, the real mixin
    ``_create_ingest_parse_pool`` must have a valid, fd-backed stderr in
    effect AT POOL-CONSTRUCTION TIME (the resource tracker snapshots
    ``sys.stderr.fileno()`` during construction). Pool construction itself
    is faked (recording, not spawning) so this stays fast and
    deterministic."""
    import tldw_chatbook.app as app_module

    recorded: dict[str, Any] = {}

    class _RecordingPool:
        def __init__(self, processes=None):
            try:
                recorded["fd_during_construction"] = sys.stderr.fileno()
            except Exception:
                recorded["fd_during_construction"] = -1

    class _RecordingContext:
        def Queue(self, maxsize=None):
            recorded["maxsize"] = maxsize
            try:
                recorded["queue_fd_during_construction"] = sys.stderr.fileno()
            except Exception:
                recorded["queue_fd_during_construction"] = -1
            progress_queue = _ClosableQueue()
            recorded["progress_queue"] = progress_queue
            return progress_queue

        def Pool(self, processes=None, initializer=None, initargs=()):
            # The combined initializer retains worker-noise suppression and
            # installs the progress queue for this spawned generation.
            recorded["initializer"] = initializer
            recorded["initargs"] = initargs
            return _RecordingPool(processes)

    class _RecordingMultiprocessing:
        @staticmethod
        def get_context(method: str):
            assert method == "spawn"
            return _RecordingContext()

    monkeypatch.setattr(app_module, "multiprocessing", _RecordingMultiprocessing())
    monkeypatch.setattr(sys, "stderr", _TextualLikeStderr())

    mixin = LibraryIngestQueueMixin()
    mixin._ingest_parse_worker_count = lambda: 1  # instance shadow: skip config read
    resources = mixin._create_ingest_parse_pool()

    assert isinstance(resources.pool, _RecordingPool)
    assert recorded["queue_fd_during_construction"] >= 0
    assert recorded["fd_during_construction"] >= 0
    assert recorded["initializer"] is initialize_ingest_parse_worker
    assert recorded["initargs"] == (resources.progress_queue,)


def test_create_pool_leaves_stderr_alone_when_fileno_is_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: with a valid ``sys.stderr`` fileno, no redirect happens --
    the stream object seen during construction is the ambient one."""
    import tldw_chatbook.app as app_module

    recorded: dict[str, Any] = {}

    class _RecordingPool:
        def __init__(self, processes=None):
            recorded["stderr_during_construction"] = sys.stderr

    class _RecordingContext:
        def Queue(self, maxsize=None):
            recorded["maxsize"] = maxsize
            progress_queue = _ClosableQueue()
            recorded["progress_queue"] = progress_queue
            return progress_queue

        def Pool(self, processes=None, initializer=None, initargs=()):
            # The combined initializer retains worker-noise suppression and
            # installs the progress queue for this spawned generation.
            recorded["initializer"] = initializer
            recorded["initargs"] = initargs
            return _RecordingPool(processes)

    class _RecordingMultiprocessing:
        @staticmethod
        def get_context(method: str):
            assert method == "spawn"
            return _RecordingContext()

    monkeypatch.setattr(app_module, "multiprocessing", _RecordingMultiprocessing())

    ambient_stderr = sys.stderr
    assert ambient_stderr.fileno() >= 0  # pytest's capture stream is fd-backed

    mixin = LibraryIngestQueueMixin()
    mixin._ingest_parse_worker_count = lambda: 1
    resources = mixin._create_ingest_parse_pool()

    assert recorded["stderr_during_construction"] is ambient_stderr
    assert recorded["initializer"] is initialize_ingest_parse_worker
    assert recorded["initargs"] == (resources.progress_queue,)


@pytest.mark.asyncio
async def test_pool_creation_failure_fails_job_retryable_and_app_survives(
    tmp_path: Path,
) -> None:
    """(Containment) Pool creation raising must never crash the app: the
    triggering job lands FAILED retryable with the pool message, no
    exception escapes ``submit_library_ingest_job``, ``_ingest_parse_pool``
    stays ``None``, and a subsequent submit retries pool creation (and
    succeeds once the pool can be built again)."""
    db = _make_db(tmp_path)
    source1 = _write_text_file(tmp_path, "note-1.txt", "First body.")
    source2 = _write_text_file(tmp_path, "note-2.txt", "Second body.")

    boom = {"raise": True}

    def _flaky_factory():
        if boom["raise"]:
            raise RuntimeError("spawn machinery exploded")
        return _FakeIngestParsePool()

    app = _IngestRunnerHarness(db, pool_factory=_flaky_factory)

    async with app.run_test() as pilot:
        # Must not raise, despite pool creation exploding underneath.
        job1 = app.submit_library_ingest_job(source_path=str(source1))

        failed = await _wait_for_job_state(
            app, pilot, job1.job_id, IngestJobState.FAILED
        )
        assert failed.permanent is False
        assert failed.error.startswith("Parse pool could not start:")
        assert "spawn machinery exploded" in failed.error
        assert app._ingest_parse_pool is None
        assert app._pool_create_count == 1

        # The failure is retryable through the normal seam, and the next
        # submission retries pool creation from scratch.
        boom["raise"] = False
        job2 = app.submit_library_ingest_job(source_path=str(source2))
        done2 = await _wait_for_job_state(app, pilot, job2.job_id, IngestJobState.DONE)
        assert done2.media_id is not None
        assert app._pool_create_count == 2

        await _wait_for_runner_idle(app, pilot)


def test_top_up_abandons_pass_when_mark_parsing_rejects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(Whole-branch review, Minor 2) The defensive `mark_parsing`-returned-
    ``None`` branch must TERMINATE the top-up pass: ``next_queued()``
    always returns the OLDEST queued job, so a skip-and-``continue`` would
    be handed the exact same unclaimable job straight back -- an infinite
    loop on the UI thread. The branch is unreachable while UI-thread
    atomicity holds (see the surrounding docstring); forced here by
    stubbing ``mark_parsing``. The test completing at all (within the
    suite timeout) is the core assertion."""
    db = _make_db(tmp_path)
    app = _IngestRunnerHarness(db)
    app.library_ingest_jobs.submit(source_path="/tmp/whatever.txt")
    monkeypatch.setattr(
        app.library_ingest_jobs, "mark_parsing", lambda *args, **kwargs: None
    )

    app._top_up_ingest_parse_pool()  # must return, not loop forever

    # The pass was abandoned before ever reaching pool creation, and the
    # unclaimable job is left QUEUED (a later pass re-attempts it).
    assert app._pool_create_count == 0
    assert app.library_ingest_jobs.counts()["queued"] == 1


# --- Task 2 review fix regression tests -------------------------------------
#
# The review found a check-then-exit race in the queue-runner's old exit
# path: it called `next_queued()` and, only in a *separate later*
# `call_from_thread` (inside `finally`), cleared `runner_active`. A
# submission landing on the UI thread in the gap between those two calls
# would append a QUEUED job while `runner_active` was still (stale-)`True`,
# so `_start_library_ingest_queue_if_idle` would never start a new runner --
# stranding the job. The fix collapses the check-and-clear into one atomic
# UI-thread call, `_claim_next_ingest_job_or_release`, and adds a
# crash-recovery safety net (`_release_ingest_runner_after_crash`) for the
# case where something bypasses that atomic exit entirely.
#
# F3 re-anchor: the writer's claim now targets *payload-ready* jobs (an
# entry in `_ingest_parsed_payloads`) instead of *queued* ones -- the tests
# below are re-anchored to that new claim source but keep the same
# atomicity-proof structure:
#   (i)/(ii)  direct, synchronous calls to `_claim_next_ingest_job_or_release`
#             proving its atomic claim-or-release contract in isolation;
#   (iii)     an end-to-end test that forces the writer to hit its
#             crash-recovery `finally` path (not a per-job failure -- a
#             failure in the claim step itself, outside all per-job
#             isolation) with a second job's payload still pending,
#             proving the writer notices and restarts itself instead of
#             stranding the payload;
#   (iv)      a coarse end-to-end stress smoke across five rapid
#             submissions (some landing while the pool/writer are
#             genuinely mid-flight) as a gross-stranding catch-all.


@pytest.mark.asyncio
async def test_claim_next_job_returns_payload_ready_job_and_keeps_runner_active(
    tmp_path: Path,
) -> None:
    """(i) Direct-call contract: a payload-ready job is returned (job +
    payload, job now WRITING, payload popped) and ``runner_active`` is left
    untouched (``True``), so the writer keeps looping instead of exiting.
    """
    db = _make_db(tmp_path)
    source = _write_text_file(
        tmp_path, "note-claim.txt", "Body for the claim contract test."
    )
    app = _IngestRunnerHarness(db)

    async with app.run_test():
        # Simulate the writer already being active (as it always is by the
        # time anything calls the claim method for real) without actually
        # starting the background worker thread -- this keeps the test
        # fully synchronous and deterministic. Driving the registry
        # directly (bypassing submit/the pool) avoids triggering
        # `_start_library_ingest_queue_if_idle` for the same reason.
        app.library_ingest_jobs.runner_active = True
        job = app.library_ingest_jobs.submit(source_path=str(source))
        app.library_ingest_jobs.mark_parsing(job.job_id)
        fake_payload = {"file_type": "plaintext", "content": "irrelevant"}
        app._ingest_parsed_payloads[job.job_id] = fake_payload

        claimed = app._claim_next_ingest_job_or_release()

        assert claimed is not None
        claimed_job, claimed_payload = claimed
        assert claimed_job.job_id == job.job_id
        assert claimed_job.state == IngestJobState.WRITING
        assert claimed_payload == fake_payload
        assert job.job_id not in app._ingest_parsed_payloads
        assert app.library_ingest_jobs.runner_active is True


@pytest.mark.asyncio
async def test_claim_next_job_returns_none_and_clears_runner_active_when_empty(
    tmp_path: Path,
) -> None:
    """(ii) Direct-call contract: with no payload-ready jobs, ``None`` is
    returned and ``runner_active`` is cleared in that same call -- the
    exact atomicity the exit-race fix depends on.
    """
    db = _make_db(tmp_path)
    app = _IngestRunnerHarness(db)

    async with app.run_test():
        app.library_ingest_jobs.runner_active = True

        claimed = app._claim_next_ingest_job_or_release()

        assert claimed is None
        assert app.library_ingest_jobs.runner_active is False


@pytest.mark.asyncio
async def test_shutdown_refuses_to_claim_ready_payload(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "shutdown-claim.txt", "Do not persist me.")
    app = _IngestRunnerHarness(db)

    async with app.run_test():
        app.library_ingest_jobs.runner_active = True
        job = app.library_ingest_jobs.submit(source_path=str(source))
        app.library_ingest_jobs.mark_parsing(job.job_id)
        app._ingest_parsed_payloads[job.job_id] = {
            "file_type": "plaintext",
            "content": "ready before shutdown",
        }
        app._ingest_shutdown = True

        claimed = app._claim_next_ingest_job_or_release()

        assert claimed is None
        assert app.library_ingest_jobs.runner_active is False
        assert job.job_id in app._ingest_parsed_payloads
        current = next(
            current
            for current in app.library_ingest_jobs.jobs()
            if current.job_id == job.job_id
        )
        assert current.state == IngestJobState.PARSING


@pytest.mark.asyncio
async def test_finally_restarts_writer_after_unexpected_claim_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(iii) Regression test for the crash-recovery path -- the strongest
    deterministic proxy for the exit race available without depending on
    real thread-scheduling timing.

    Simulates a genuinely unexpected ("catastrophic") failure in the
    writer's own claim step, as opposed to a per-job write failure (already
    isolated by the inner try/except): the *first ever* call to
    ``LibraryIngestJobRegistry.jobs()`` -- made from inside
    ``_claim_next_ingest_job_or_release``, the only call site that reaches
    it in this harness's writer path -- raises once; every subsequent call
    behaves normally. The patch is installed before either submission.
    Polling below deliberately uses ``real_jobs`` (the pre-patch bound
    method) rather than going back through ``app.library_ingest_jobs.jobs``
    -- otherwise the test's own polling would be equally likely to be the
    call that trips the one-shot raise, instead of the writer's claim.

    Pre-fix (L3b), the writer's ``finally`` unconditionally set
    ``runner_active = False`` and exited for good -- it never rechecked for
    pending work and restarted itself. Post-fix,
    ``_release_ingest_runner_after_crash`` notices a payload is still
    waiting (the exception fires before the claim ever pops anything out of
    ``_ingest_parsed_payloads``) and restarts the writer, so both jobs
    still reach ``DONE``.
    """
    db = _make_db(tmp_path)
    source1 = _write_text_file(
        tmp_path, "note-crash-1.txt", "First body, present when the crash hits."
    )
    source2 = _write_text_file(
        tmp_path, "note-crash-2.txt", "Second body, queued behind the crash."
    )
    app = _IngestRunnerHarness(db, worker_count=2)

    real_jobs = app.library_ingest_jobs.jobs
    call_state = {"raised": False}

    def _flaky_jobs():
        if not call_state["raised"]:
            call_state["raised"] = True
            raise RuntimeError("simulated catastrophic writer failure")
        return real_jobs()

    async def _wait_for_done_via_real_jobs(job_id: str) -> LibraryIngestJob:
        for _ in range(_POLL_ATTEMPTS):
            job = next((j for j in real_jobs() if j.job_id == job_id), None)
            if job is not None and job.state == IngestJobState.DONE:
                return job
            await pilot.pause(_POLL_INTERVAL)
        raise AssertionError(f"job {job_id} never reached DONE: {real_jobs()}")

    async with app.run_test() as pilot:
        monkeypatch.setattr(app.library_ingest_jobs, "jobs", _flaky_jobs)

        job1 = app.submit_library_ingest_job(source_path=str(source1))
        job2 = app.submit_library_ingest_job(source_path=str(source2))

        done1 = await _wait_for_done_via_real_jobs(job1.job_id)
        assert done1.media_id is not None
        done2 = await _wait_for_done_via_real_jobs(job2.job_id)
        assert done2.media_id is not None

        assert call_state["raised"] is True, (
            "the simulated crash never fired -- test is not exercising the recovery path"
        )

        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_five_rapid_submissions_all_complete_no_stranding(tmp_path: Path) -> None:
    """(iv) End-to-end stress smoke: five jobs submitted in rapid
    succession -- some landing while the pool/writer are genuinely
    mid-flight -- must ALL reach DONE.

    This is a coarse, gross-stranding catch-all: if any submission landed
    in the (pre-fix) exit-race gap and got stuck behind a stale
    ``runner_active``, at least one job here would never leave QUEUED/
    PARSING and the wait loop below would time out.
    """
    db = _make_db(tmp_path)
    app = _IngestRunnerHarness(db, worker_count=2)

    async with app.run_test() as pilot:
        sources = [
            _write_text_file(
                tmp_path, f"note-stress-{i}.txt", f"Stress body number {i}."
            )
            for i in range(5)
        ]

        jobs = [
            app.submit_library_ingest_job(source_path=str(sources[0])),
            app.submit_library_ingest_job(source_path=str(sources[1])),
        ]
        # Give the pool/writer a chance to actually start draining before
        # the remaining submissions land -- the scenario most likely to hit
        # the exit-race gap is a submission arriving while the writer is
        # mid-loop, possibly right as it decides whether to exit.
        await pilot.pause(_POLL_INTERVAL)
        jobs.append(app.submit_library_ingest_job(source_path=str(sources[2])))
        await pilot.pause(_POLL_INTERVAL)
        jobs.append(app.submit_library_ingest_job(source_path=str(sources[3])))
        jobs.append(app.submit_library_ingest_job(source_path=str(sources[4])))

        for job in jobs:
            done = await _wait_for_job_state(
                app, pilot, job.job_id, IngestJobState.DONE
            )
            assert done.media_id is not None

        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_reingest_of_unchanged_file_still_resolves_media_id(
    tmp_path: Path,
) -> None:
    """Re-ingesting an already-present, unchanged file keeps Open usable.

    ``add_media_with_keywords`` takes its update path for a URL that already
    exists with identical content and returns ``media_id=None``; the writer
    must resolve the id via ``get_media_by_url`` (using the parsed payload's
    own ``url`` field) so the done job still carries a real ``media_id``
    (and the canvas keeps its Open action).
    """
    db = _make_db(tmp_path)
    source = _write_text_file(
        tmp_path, "note-b.txt", "Spring tides align sun and moon."
    )
    app = _IngestRunnerHarness(db)

    async with app.run_test() as pilot:
        first = app.submit_library_ingest_job(source_path=str(source))
        first_done = await _wait_for_job_state(
            app, pilot, first.job_id, IngestJobState.DONE
        )
        assert first_done.media_id is not None
        await _wait_for_runner_idle(app, pilot)

        second = app.submit_library_ingest_job(source_path=str(source))
        second_done = await _wait_for_job_state(
            app, pilot, second.job_id, IngestJobState.DONE
        )

        assert second_done.media_id == first_done.media_id
        await _wait_for_runner_idle(app, pilot)


# Windows Proactor event-loop setup owns an internal loopback socket pair.
@pytest.mark.allow_network
@pytest.mark.asyncio
async def test_local_writer_uses_claimed_generic_overwrite_option(tmp_path: Path) -> None:
    """A job's snapshot, rather than current form state, controls overwrite."""
    db = _make_db(tmp_path)
    source = _write_text_file(
        tmp_path, "overwrite.txt", "Unchanged document body for overwrite testing."
    )
    app = _IngestRunnerHarness(db)

    async with app.run_test() as pilot:
        first = app.submit_library_ingest_job(
            source_path=str(source),
            title="Original title",
            ingest_options={"generic": {"overwrite_existing": False}},
        )
        first_done = await _wait_for_job_state(
            app, pilot, first.job_id, IngestJobState.DONE
        )
        assert first_done.media_id is not None

        skipped = app.submit_library_ingest_job(
            source_path=str(source),
            title="Skipped title",
            ingest_options={"generic": {"overwrite_existing": False}},
        )
        await _wait_for_job_state(app, pilot, skipped.job_id, IngestJobState.DONE)
        row = db.execute_query(
            "SELECT title FROM Media WHERE id = ?", (first_done.media_id,)
        ).fetchone()
        assert row["title"] == "Original title"

        updated = app.submit_library_ingest_job(
            source_path=str(source),
            title="Updated title",
            ingest_options={"generic": {"overwrite_existing": True}},
        )
        updated_done = await _wait_for_job_state(
            app, pilot, updated.job_id, IngestJobState.DONE
        )
        assert updated_done.media_id == first_done.media_id
        row = db.execute_query(
            "SELECT title FROM Media WHERE id = ?", (first_done.media_id,)
        ).fetchone()
        assert row["title"] == "Updated title"
        await _wait_for_runner_idle(app, pilot)


# --- remote poller (task-684.2) ---------------------------------------------


class _FakeServerMediaService:
    """Records batch lookups and replays scripted status responses."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = responses
        self.batch_calls: list[str] = []

    async def list_media_ingest_jobs(self, batch_id: str, *, limit: int = 100):
        self.batch_calls.append(batch_id)
        if self._responses:
            return self._responses.pop(0)
        return {"batch_id": batch_id, "jobs": []}


def _queued_server_job(
    app,
    *,
    remote_job_id: str,
    batch_id: str = "batch-1",
    research_source_operation_id: str | None = None,
):
    job = app.library_ingest_jobs.submit(
        source_path="/tmp/a.mp3",
        origin="server",
        research_source_operation_id=research_source_operation_id,
    )
    return app.library_ingest_jobs.attach_remote(
        job.job_id, remote_job_id=remote_job_id, batch_id=batch_id
    )


@pytest.mark.asyncio
async def test_remote_poll_settles_a_server_job_then_stops(tmp_path: Path) -> None:
    """The poller applies a terminal status and then stops polling.

    A finished batch that kept being fetched would hit the server forever for an
    answer that cannot change.
    """
    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.REMOTE_INGEST_POLL_SECONDS = 0.01
    service = _FakeServerMediaService(
        [{"batch_id": "batch-1", "jobs": [{"id": 11, "status": "completed"}]}]
    )
    app.server_media_reading_service = service

    async with app.run_test() as pilot:
        _queued_server_job(app, remote_job_id="11")
        app.poll_remote_ingest_jobs()

        for _ in range(_POLL_ATTEMPTS):
            if app.library_ingest_jobs.jobs()[0].state == IngestJobState.DONE:
                break
            await pilot.pause(_POLL_INTERVAL)
        else:
            raise AssertionError("the server job never settled")

        calls_after_settle = len(service.batch_calls)
        for _ in range(5):
            await pilot.pause(_POLL_INTERVAL)
        assert len(service.batch_calls) == calls_after_settle, (
            "poller kept fetching a settled batch"
        )


@pytest.mark.asyncio
async def test_remote_completion_schedules_research_association_after_settle(
    tmp_path: Path,
) -> None:
    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.REMOTE_INGEST_POLL_SECONDS = 0.01
    scheduler = _RecordingAssociationScheduler()
    app.research_source_association_scheduler = scheduler
    app.server_media_reading_service = _FakeServerMediaService(
        [
            {
                "batch_id": "batch-1",
                "jobs": [
                    {
                        "id": 11,
                        "status": "completed",
                        "result": {"media_id": 884},
                    }
                ],
            }
        ]
    )

    async with app.run_test() as pilot:
        job = _queued_server_job(
            app,
            remote_job_id="11",
            research_source_operation_id="research-op-app-server",
        )
        app.poll_remote_ingest_jobs()
        await _wait_for_job_state(app, pilot, job.job_id, IngestJobState.DONE)
        for _ in range(_POLL_ATTEMPTS):
            if scheduler.calls:
                break
            await pilot.pause(_POLL_INTERVAL)

    assert scheduler.calls == ["research-op-app-server"]
    settled = app.library_ingest_jobs.get_job(job.job_id)
    assert settled is not None
    assert settled.media_id is None
    assert settled.remote_media_id == "884"


@pytest.mark.asyncio
async def test_remote_poll_does_not_start_without_outstanding_batches(
    tmp_path: Path,
) -> None:
    """A library with only local jobs must never touch the server."""
    app = _IngestRunnerHarness(_make_db(tmp_path))
    service = _FakeServerMediaService([])
    app.server_media_reading_service = service

    async with app.run_test() as pilot:
        app.library_ingest_jobs.submit(source_path="/tmp/local.txt")
        app.poll_remote_ingest_jobs()
        await pilot.pause(_POLL_INTERVAL)

    assert service.batch_calls == []


@pytest.mark.asyncio
async def test_remote_poll_survives_a_server_error(tmp_path: Path) -> None:
    """A transient failure must not kill the poller or the job."""
    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.REMOTE_INGEST_POLL_SECONDS = 0.01

    class _FlakyService:
        def __init__(self) -> None:
            self.calls = 0

        async def list_media_ingest_jobs(self, batch_id: str, *, limit: int = 100):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("connection reset")
            return {"batch_id": batch_id, "jobs": [{"id": 11, "status": "completed"}]}

    service = _FlakyService()
    app.server_media_reading_service = service

    async with app.run_test() as pilot:
        _queued_server_job(app, remote_job_id="11")
        app.poll_remote_ingest_jobs()

        for _ in range(_POLL_ATTEMPTS):
            if app.library_ingest_jobs.jobs()[0].state == IngestJobState.DONE:
                break
            await pilot.pause(_POLL_INTERVAL)
        else:
            raise AssertionError("the poller did not recover from the error")

    assert service.calls >= 2


@pytest.mark.asyncio
async def test_remote_poll_stops_on_shutdown(tmp_path: Path) -> None:
    """The quit flag ends the loop rather than leaving it fetching."""
    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.REMOTE_INGEST_POLL_SECONDS = 0.01
    service = _FakeServerMediaService([])
    app.server_media_reading_service = service

    async with app.run_test() as pilot:
        _queued_server_job(app, remote_job_id="11")
        app.poll_remote_ingest_jobs()
        await pilot.pause(_POLL_INTERVAL)

        app._ingest_shutdown = True
        await pilot.pause(_POLL_INTERVAL)
        calls_at_shutdown = len(service.batch_calls)
        for _ in range(5):
            await pilot.pause(_POLL_INTERVAL)

    assert len(service.batch_calls) == calls_at_shutdown, "poller ignored shutdown"


@pytest.mark.asyncio
async def test_remote_poll_without_a_server_service_is_a_noop(tmp_path: Path) -> None:
    """No configured server backend means nothing to poll, not a crash."""
    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.server_media_reading_service = None

    async with app.run_test() as pilot:
        _queued_server_job(app, remote_job_id="11")
        app.poll_remote_ingest_jobs()
        await pilot.pause(_POLL_INTERVAL)

        assert app.library_ingest_jobs.jobs()[0].state == IngestJobState.QUEUED


@pytest.mark.asyncio
async def test_cancel_remote_batch_asks_the_server_and_resumes_polling(
    tmp_path: Path,
) -> None:
    """Cancelling asks the server; the local job is NOT pre-emptively marked.

    The request is asynchronous and may be refused, so the queue must not claim
    an outcome the server has not confirmed. The poller records the real state.
    """
    cancelled_batches: list[str] = []

    class _CancellableService:
        async def cancel_media_ingest_jobs_batch(
            self, *, batch_id: str | None = None, session_id: str | None = None,
            reason: str | None = None,
        ):
            # Keyword-only, mirroring the real client/service signature --
            # a fake with a positional parameter would happily accept a
            # positional call that fails against the real one.
            cancelled_batches.append(batch_id)
            return {"batch_id": batch_id, "cancelled": 1}

        async def list_media_ingest_jobs(self, batch_id: str, *, limit: int = 100):
            return {
                "batch_id": batch_id,
                "jobs": [{"id": 11, "status": "cancelled",
                          "cancellation_reason": "user asked"}],
            }

    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.REMOTE_INGEST_POLL_SECONDS = 0.01
    app.server_media_reading_service = _CancellableService()

    async with app.run_test() as pilot:
        job = _queued_server_job(app, remote_job_id="11")
        app.cancel_remote_ingest_batch("batch-1")

        # The request alone must not move the local job.
        await pilot.pause(_POLL_INTERVAL)
        assert cancelled_batches == ["batch-1"]

        for _ in range(_POLL_ATTEMPTS):
            if app.library_ingest_jobs.jobs()[0].state == IngestJobState.CANCELLED:
                break
            await pilot.pause(_POLL_INTERVAL)
        else:
            raise AssertionError("the poller never recorded the cancellation")

        assert "user asked" in app.library_ingest_jobs.jobs()[0].error


@pytest.mark.asyncio
async def test_cancel_remote_batch_without_a_server_seam_is_quiet(
    tmp_path: Path,
) -> None:
    """No server backend means no cancel request, and no crash."""
    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.server_media_reading_service = None

    async with app.run_test() as pilot:
        _queued_server_job(app, remote_job_id="11")
        app.cancel_remote_ingest_batch("batch-1")
        await pilot.pause(_POLL_INTERVAL)

        assert app.library_ingest_jobs.jobs()[0].state == IngestJobState.QUEUED


@pytest.mark.asyncio
async def test_cancel_remote_batch_ignores_an_empty_batch_id(tmp_path: Path) -> None:
    calls: list[str] = []

    class _Service:
        async def cancel_media_ingest_jobs_batch(
            self, *, batch_id: str | None = None, session_id: str | None = None,
            reason: str | None = None,
        ):
            calls.append(batch_id)

        async def list_media_ingest_jobs(self, batch_id: str, *, limit: int = 100):
            return {"batch_id": batch_id, "jobs": []}

    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.server_media_reading_service = _Service()

    async with app.run_test() as pilot:
        app.cancel_remote_ingest_batch("")
        await pilot.pause(_POLL_INTERVAL)

    assert calls == []


# --- backend routing (task-684.1 slice 2) -----------------------------------


class _RecordingServerService:
    """Captures submissions and returns a scripted batch/job id pair."""

    def __init__(self, *, fail: bool = False) -> None:
        self.submissions: list[dict] = []
        self.fail = fail

    async def submit_ingest_jobs(self, **kwargs):
        self.submissions.append(kwargs)
        if self.fail:
            raise RuntimeError("server said no")
        return {"batch_id": "batch-7", "jobs": [{"id": 42, "source": "/tmp/a.mp3"}]}

    async def submit_media_ingest_jobs(self, **kwargs):
        return await self.submit_ingest_jobs(**kwargs)

    async def list_media_ingest_jobs(self, batch_id: str, *, limit: int = 100):
        return {"batch_id": batch_id, "jobs": []}


def _server_ingest_preference():
    """Patch the ingest backend preference to "server" for a test's duration.

    Sending an ingest to a server is an explicit opt-in stored under
    ``[library.ingest] backend``; the browse scope deliberately does not decide
    it. The opt-in alone is not enough, though -- runtime policy requires the
    runtime to be in server mode as well -- so tests that want a server route
    must ALSO call ``_use_server_runtime`` (see ``_resolve_ingest_backend``).
    """
    real = _app_module.get_cli_setting

    def _fake(*args, **kwargs):
        if args[:2] == ("library.ingest", "backend"):
            return "server"
        return real(*args, **kwargs)

    return patch("tldw_chatbook.app.get_cli_setting", side_effect=_fake)


def _use_server_runtime(app) -> None:
    """Put the harness's runtime policy in server mode.

    Required alongside the opt-in: ``media.ingestion_jobs.launch.server`` is
    declared ``required_source="server"``, so the service refuses the launch in
    local mode -- verified live, where it failed with "requires server mode".
    """
    from types import SimpleNamespace

    app.runtime_policy = SimpleNamespace(
        state=SimpleNamespace(active_source="server")
    )


@pytest.mark.asyncio
async def test_local_backend_still_ingests_locally(tmp_path: Path) -> None:
    """The default path must be untouched by server routing."""
    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "note.txt", "Local body.")
    app = _IngestRunnerHarness(db)
    service = _RecordingServerService()
    app.server_media_reading_service = service

    async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(source_path=str(source))
        assert job.origin == "local"
        done = await _wait_for_job_state(app, pilot, job.job_id, IngestJobState.DONE)
        assert done.media_id is not None
        await _wait_for_runner_idle(app, pilot)

    assert service.submissions == [], "local ingest must not contact the server"


@pytest.mark.asyncio
async def test_server_backend_submits_remotely_and_attaches_ids(
    tmp_path: Path,
) -> None:
    """A server-backend submit goes to the server and records its ids."""
    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.REMOTE_INGEST_POLL_SECONDS = 0.01
    service = _RecordingServerService()
    app.server_media_reading_service = service
    source = _write_text_file(tmp_path, "note.txt", "Body.")

    _use_server_runtime(app)
    with _server_ingest_preference():
      async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(source_path=str(source), title="A title")
        assert job.origin == "server"

        for _ in range(_POLL_ATTEMPTS):
            current = app.library_ingest_jobs.get_job(job.job_id)
            if current is not None and current.batch_id:
                break
            await pilot.pause(_POLL_INTERVAL)
        else:
            raise AssertionError("the remote ids were never attached")

    assert len(service.submissions) == 1
    submitted = service.submissions[0]
    # "document" not "plaintext": the live server accepts only video/audio/
    # document/pdf/ebook, enforced by a runtime validator not its OpenAPI spec.
    assert submitted["media_type"] == "document"
    assert submitted["file_paths"] == [str(source)]
    assert submitted["title"] == "A title"

    current = app.library_ingest_jobs.get_job(job.job_id)
    assert current.batch_id == "batch-7"
    assert current.remote_job_id == "42"


@pytest.mark.asyncio
async def test_server_backend_without_a_service_fails_the_job_clearly(
    tmp_path: Path,
) -> None:
    """Choosing a server with none configured must explain itself, not hang."""
    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.server_media_reading_service = None
    source = _write_text_file(tmp_path, "note.txt", "Body.")

    _use_server_runtime(app)
    with _server_ingest_preference():
      async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(source_path=str(source))
        failed = await _wait_for_job_state(
            app, pilot, job.job_id, IngestJobState.FAILED
        )

    assert "server" in failed.error.lower()


@pytest.mark.asyncio
async def test_server_backend_refuses_a_source_it_cannot_send(tmp_path: Path) -> None:
    """A plain web page belongs to the clipper, and says so rather than failing late."""
    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.server_media_reading_service = _RecordingServerService()

    _use_server_runtime(app)
    with _server_ingest_preference():
      async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(source_path="https://example.com/a-post")
        failed = await _wait_for_job_state(
            app, pilot, job.job_id, IngestJobState.FAILED
        )

    assert "clip" in failed.error.lower()


@pytest.mark.asyncio
async def test_server_submit_failure_marks_the_job_failed(tmp_path: Path) -> None:
    """A refused submission must surface, not sit queued forever."""
    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.server_media_reading_service = _RecordingServerService(fail=True)
    source = _write_text_file(tmp_path, "note.txt", "Body.")

    _use_server_runtime(app)
    with _server_ingest_preference():
      async with app.run_test() as pilot:
        job = app.submit_library_ingest_job(source_path=str(source))
        failed = await _wait_for_job_state(
            app, pilot, job.job_id, IngestJobState.FAILED
        )

    assert failed.error


@pytest.mark.asyncio
async def test_an_unrecognised_backend_falls_back_to_local(tmp_path: Path) -> None:
    """Anything that is not exactly "server" must mean local.

    Local is the backend that always works, so a typo'd or newly-added value
    must not silently start shipping the user's files to a server.

    Uses a raw stand-in rather than ``MediaRuntimeState``: that dataclass
    normalises in ``__post_init__``, so a test going through it would exercise
    the dataclass's guarantee instead of this fallback, and would pass even if
    the fallback were inverted.
    """
    from types import SimpleNamespace

    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "note.txt", "Body.")
    app = _IngestRunnerHarness(db)
    service = _RecordingServerService()
    app.server_media_reading_service = service

    async with app.run_test() as pilot:
        for value in ("", "  ", "remote", "Server-ish", "cloud", None):
            with patch(
                "tldw_chatbook.app.get_cli_setting",
                side_effect=lambda *a, v=value, **k: (
                    v if a[:2] == ("library.ingest", "backend") else None
                ),
            ):
                assert app._resolve_ingest_backend() == "local", repr(value)

        # And end-to-end: with no preference set, ingest stays local.
        job = app.submit_library_ingest_job(source_path=str(source))
        assert job.origin == "local"
        await _wait_for_job_state(app, pilot, job.job_id, IngestJobState.DONE)
        await _wait_for_runner_idle(app, pilot)

    assert service.submissions == []


@pytest.mark.asyncio
async def test_server_backend_is_matched_case_insensitively(tmp_path: Path) -> None:
    """A raw "Server"/" SERVER " value should still route remotely."""
    app = _IngestRunnerHarness(_make_db(tmp_path))
    _use_server_runtime(app)
    async with app.run_test():
        for value in ("server", "Server", " SERVER "):
            with patch(
                "tldw_chatbook.app.get_cli_setting",
                side_effect=lambda *a, v=value, **k: (
                    v if a[:2] == ("library.ingest", "backend") else None
                ),
            ):
                assert app._resolve_ingest_backend() == "server", repr(value)


@pytest.mark.asyncio
async def test_media_runtime_state_already_normalises_the_backend() -> None:
    """Defence in depth: the shared dataclass narrows the value first.

    Documented as its own test so the two layers are not confused -- this one
    is about ``MediaRuntimeState``'s guarantee, not the ingest fallback above.
    """
    from tldw_chatbook.UI.Screens.media_runtime_state import MediaRuntimeState

    assert MediaRuntimeState(runtime_backend="remote").runtime_backend == "local"
    assert MediaRuntimeState(runtime_backend="SERVER").runtime_backend == "server"


@pytest.mark.asyncio
async def test_browse_scope_alone_never_sends_files_to_a_server(
    tmp_path: Path,
) -> None:
    """Browsing in server scope must not silently upload a local file.

    ``build_library_ingest_state``'s own contract says ingest "always targets
    the local media store regardless of browsing scope". Letting the browse
    scope decide would mean a user who switched scope to look at server media
    then imported a file would have it leave their machine without ever asking
    for that -- so the ingest target is its own explicit preference, defaulting
    to local.
    """
    from types import SimpleNamespace

    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "note.txt", "Body.")
    app = _IngestRunnerHarness(db)
    service = _RecordingServerService()
    app.server_media_reading_service = service
    # Browsing server-side, with no ingest preference expressed.
    app.media_runtime_state = SimpleNamespace(runtime_backend="server")

    async with app.run_test() as pilot:
        assert app._resolve_ingest_backend() == "local"
        job = app.submit_library_ingest_job(source_path=str(source))
        assert job.origin == "local"
        await _wait_for_job_state(app, pilot, job.job_id, IngestJobState.DONE)
        await _wait_for_runner_idle(app, pilot)

    assert service.submissions == [], "browse scope must not route ingest remotely"


@pytest.mark.asyncio
async def test_an_explicit_server_preference_routes_remotely(tmp_path: Path) -> None:
    """Opting in -- and only opting in -- sends the ingest to the server."""
    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.REMOTE_INGEST_POLL_SECONDS = 0.01
    service = _RecordingServerService()
    app.server_media_reading_service = service
    _use_server_runtime(app)
    source = _write_text_file(tmp_path, "note.txt", "Body.")

    with patch(
        "tldw_chatbook.app.get_cli_setting",
        side_effect=lambda *a, **k: (
            "server" if a and a[0] == "library.ingest" and a[1:2] == ("backend",) else None
        ),
    ):
        async with app.run_test() as pilot:
            assert app._resolve_ingest_backend() == "server"
            job = app.submit_library_ingest_job(source_path=str(source))
            assert job.origin == "server"
            await pilot.pause(_POLL_INTERVAL)

    assert len(service.submissions) == 1


@pytest.mark.asyncio
async def test_remote_poll_follows_pagination(tmp_path: Path) -> None:
    """A batch larger than one page must be reconciled in full.

    The live server's MediaIngestJobListResponse carries has_more/next_offset
    (confirmed from its OpenAPI spec). Reading only the first page would leave
    later jobs unreconciled -- and because they stay unsettled, the poller would
    keep fetching that batch forever, which is exactly what the stop condition
    exists to prevent.
    """
    class _PagedService:
        def __init__(self) -> None:
            self.offsets: list[int] = []

        async def list_media_ingest_jobs(self, batch_id: str, *, limit: int = 100,
                                         offset: int = 0):
            self.offsets.append(offset)
            if offset == 0:
                return {
                    "batch_id": batch_id,
                    "jobs": [{"id": 11, "status": "completed"}],
                    "has_more": True,
                    "next_offset": 1,
                }
            return {
                "batch_id": batch_id,
                "jobs": [{"id": 12, "status": "completed"}],
                "has_more": False,
            }

    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.REMOTE_INGEST_POLL_SECONDS = 0.01
    service = _PagedService()
    app.server_media_reading_service = service

    async with app.run_test() as pilot:
        _queued_server_job(app, remote_job_id="11")
        _queued_server_job(app, remote_job_id="12")
        app.poll_remote_ingest_jobs()

        for _ in range(_POLL_ATTEMPTS):
            states = {j.state for j in app.library_ingest_jobs.jobs()}
            if states == {IngestJobState.DONE}:
                break
            await pilot.pause(_POLL_INTERVAL)
        else:
            raise AssertionError(
                f"second page never reconciled; states={states}, offsets={service.offsets}"
            )

    assert 1 in service.offsets, "the poller never asked for the second page"


def test_the_real_service_accepts_the_offset_the_poller_sends() -> None:
    """Pin the poller's paging against the REAL service, not a fake.

    ``test_remote_poll_follows_pagination`` passed while pagination was dead in
    production: its fake declared ``offset`` because I wrote it to match my call
    site, but ``ServerMediaReadingService.list_media_ingest_jobs`` took only
    ``limit``. Every real call raised ``TypeError`` and fell into a one-page
    fallback, so the poller never read past page one.

    A fake can agree with a wrong assumption; the real signature cannot. Same
    failure mode as the keyword-only ``cancel`` call whose fake took it
    positionally (task-684.2).
    """
    from tldw_chatbook.app import _accepts_keyword
    from tldw_chatbook.Media.server_media_reading_service import (
        ServerMediaReadingService,
    )

    for method_name in ("list_media_ingest_jobs", "list_ingest_jobs"):
        method = getattr(ServerMediaReadingService, method_name)
        assert _accepts_keyword(method, "offset"), (
            f"{method_name} takes no offset, so the poller cannot page"
        )


def test_accepts_keyword_does_not_mistake_an_internal_typeerror_for_absence() -> None:
    """The signature check must not be fooled the way ``except TypeError`` was.

    Catching ``TypeError`` around the call conflated "no such parameter" with
    "the callable raised TypeError internally", silently degrading a real bug
    into a missing feature.
    """
    from tldw_chatbook.app import _accepts_keyword

    async def raises_internally(batch_id: str, *, offset: int = 0):
        raise TypeError("something inside blew up")

    assert _accepts_keyword(raises_internally, "offset") is True

    async def no_offset(batch_id: str, *, limit: int = 100):
        return {}

    assert _accepts_keyword(no_offset, "offset") is False

    async def takes_kwargs(batch_id: str, **kwargs):
        return {}

    assert _accepts_keyword(takes_kwargs, "offset") is True


@pytest.mark.asyncio
async def test_a_service_without_offset_reads_one_page_instead_of_looping(
    tmp_path: Path,
) -> None:
    """An offset-less service must stop, not re-read page one to the cap.

    ``has_more`` stays true on a first page, so continuing the loop without a
    way to advance would re-fetch the same page ``REMOTE_INGEST_MAX_PAGES``
    times per pass, every pass.
    """
    class _UnpagedService:
        def __init__(self) -> None:
            self.calls = 0

        async def list_media_ingest_jobs(self, batch_id: str, *, limit: int = 100):
            self.calls += 1
            return {
                "batch_id": batch_id,
                "jobs": [{"id": 11, "status": "completed"}],
                "has_more": True,
                "next_offset": 1,
            }

    app = _IngestRunnerHarness(_make_db(tmp_path))
    service = _UnpagedService()
    app.server_media_reading_service = service

    async with app.run_test():
        _queued_server_job(app, remote_job_id="11")
        await app._reconcile_remote_batch(service, "batch-1")

    assert service.calls == 1, (
        f"re-read page one {service.calls} times with no way to advance"
    )
    assert {j.state for j in app.library_ingest_jobs.jobs()} == {IngestJobState.DONE}


@pytest.mark.asyncio
async def test_opting_in_without_server_mode_still_ingests_locally(
    tmp_path: Path,
) -> None:
    """The opt-in alone must not route remotely; policy requires server mode.

    Runtime policy declares ``media.ingestion_jobs.launch.server`` as
    ``required_source="server"``, so the service refuses the launch in local
    mode. Verified live: it failed with "media.ingestion_jobs.launch.server
    requires server mode". Falling back to a local ingest keeps the file where
    it is and lets the canvas explain the precondition, rather than handing the
    user a failed job.
    """
    db = _make_db(tmp_path)
    source = _write_text_file(tmp_path, "note.txt", "Body.")
    app = _IngestRunnerHarness(db)
    service = _RecordingServerService()
    app.server_media_reading_service = service
    # Opted in, but the runtime is still local.

    with _server_ingest_preference():
        async with app.run_test() as pilot:
            assert app._resolve_ingest_backend() == "local"
            job = app.submit_library_ingest_job(source_path=str(source))
            assert job.origin == "local"
            await _wait_for_job_state(app, pilot, job.job_id, IngestJobState.DONE)
            await _wait_for_runner_idle(app, pilot)

    assert service.submissions == []


class _RecordingClipService:
    """Captures web clips and returns a live-shaped clip response."""

    def __init__(self, *, extracted: bool = True, fail: bool = False) -> None:
        self.clips: list[dict] = []
        self.submissions: list[dict] = []
        self.extracted = extracted
        self.fail = fail

    async def ingest_web_content(self, **kwargs):
        self.clips.append(kwargs)
        if self.fail:
            raise RuntimeError("scraper unavailable")
        return {
            "status": "success",
            "message": "Web content processed",
            "count": 1,
            "results": [
                {
                    "url": kwargs["urls"][0],
                    "title": "Untitled",
                    "author": "Unknown",
                    "content": "Body" if self.extracted else "",
                    "extraction_successful": self.extracted,
                }
            ],
        }

    async def submit_ingest_jobs(self, **kwargs):
        self.submissions.append(kwargs)
        return {"batch_id": "batch-9", "jobs": [{"id": 1}]}

    async def list_media_ingest_jobs(self, batch_id: str, *, limit: int = 100, offset: int = 0):
        return {"batch_id": batch_id, "jobs": []}


@pytest.mark.asyncio
async def test_a_page_goes_to_the_clipper_and_a_file_does_not(tmp_path: Path) -> None:
    """Routing must split on what the source *is*, not on the backend alone.

    The ingest-jobs API has no media type for a web page -- ``server_media_type_for``
    refuses one deliberately -- so sending a page there fails. A page belongs to
    the clipper endpoint and a file to the jobs API, with one backend switch
    covering both (task-684.3).
    """
    app = _IngestRunnerHarness(_make_db(tmp_path))
    service = _RecordingClipService()
    app.server_media_reading_service = service
    audio = _write_text_file(tmp_path, "talk.mp3", "not really audio")

    with _server_ingest_preference():
        async with app.run_test() as pilot:
            _use_server_runtime(app)
            app.submit_library_ingest_job(source_path="https://example.com/post")
            app.submit_library_ingest_job(source_path=str(audio))
            for _ in range(_POLL_ATTEMPTS):
                if service.clips and service.submissions:
                    break
                await pilot.pause(_POLL_INTERVAL)

    assert [c["urls"] for c in service.clips] == [["https://example.com/post"]]
    assert service.submissions, "the file never reached the ingest-jobs API"
    assert "urls" not in service.submissions[0] or service.submissions[0].get(
        "file_paths"
    ), "a local file must be sent as a file, not a url"


@pytest.mark.asyncio
async def test_a_clipped_page_finishes_in_the_queue(tmp_path: Path) -> None:
    """A clip settles on the synchronous answer -- there is no job to poll."""
    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.server_media_reading_service = _RecordingClipService()

    with _server_ingest_preference():
        async with app.run_test() as pilot:
            _use_server_runtime(app)
            app.submit_library_ingest_job(source_path="https://example.com/post")
            for _ in range(_POLL_ATTEMPTS):
                states = {j.state for j in app.library_ingest_jobs.jobs()}
                if states == {IngestJobState.DONE}:
                    break
                await pilot.pause(_POLL_INTERVAL)
            else:
                raise AssertionError(f"clip never finished; states={states}")

    job = app.library_ingest_jobs.jobs()[0]
    assert job.origin == "server"
    # No media id comes back from the clipper, so "Open in Library" stays
    # withheld: the content is in the server's library, not this machine's.
    assert not job.media_id


@pytest.mark.asyncio
async def test_a_clip_that_extracted_nothing_fails_rather_than_reporting_done(
    tmp_path: Path,
) -> None:
    """A 200 is not a captured page.

    The endpoint answers 200 with its outcome in the body, so an extraction that
    found nothing arrives as transport-level success. Recording it as done would
    repeat the empty-ingest bug guarded against in task-677.
    """
    app = _IngestRunnerHarness(_make_db(tmp_path))
    app.server_media_reading_service = _RecordingClipService(extracted=False)

    with _server_ingest_preference():
        async with app.run_test() as pilot:
            _use_server_runtime(app)
            app.submit_library_ingest_job(source_path="https://example.com/post")
            for _ in range(_POLL_ATTEMPTS):
                states = {j.state for j in app.library_ingest_jobs.jobs()}
                if states == {IngestJobState.FAILED}:
                    break
                await pilot.pause(_POLL_INTERVAL)
            else:
                raise AssertionError(f"empty clip was not failed; states={states}")

    job = app.library_ingest_jobs.jobs()[0]
    assert "extracted" in (job.error or "").lower()


class _IdlessSubmitService:
    """Accepts a submission but answers without usable tracking ids."""

    def __init__(self, response: dict) -> None:
        self.response = response
        self.list_calls = 0

    async def submit_ingest_jobs(self, **kwargs):
        return self.response

    async def list_media_ingest_jobs(self, batch_id: str, *, limit: int = 100, offset: int = 0):
        self.list_calls += 1
        return {"batch_id": batch_id, "jobs": []}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "why"),
    [
        ({"jobs": [{"id": 7}], "errors": []}, "no batch_id to poll"),
        ({"batch_id": "b-1", "jobs": [], "errors": []}, "no job to reconcile against"),
        ({"batch_id": "b-1", "jobs": [{"source": "/tmp/a.mp3"}]}, "job carries no id"),
        ({}, "nothing usable at all"),
    ],
)
async def test_a_submission_we_cannot_track_fails_instead_of_queueing_forever(
    tmp_path: Path, response: dict, why: str
) -> None:
    """Reconciliation needs both ids, so without them the job can never settle.

    ``pending_remote_batches`` keys off ``batch_id`` and
    ``reconcile_remote_ingest_jobs`` keys off ``remote_job_id``. Missing the
    former means the job is never polled at all; missing the latter means the
    batch is polled forever while no status can ever be matched to the job.
    Either way the queue row sits at "queued" indefinitely -- the same
    never-resolves failure as the mistyped ``result`` field, which is exactly
    what a queue must not do silently.

    Failing says so out loud. The message admits the server may still be working,
    because it accepted the submission; what is lost is our ability to follow it.
    """
    app = _IngestRunnerHarness(_make_db(tmp_path))
    service = _IdlessSubmitService(response)
    app.server_media_reading_service = service
    source = _write_text_file(tmp_path, "note.txt", "Body.")

    with _server_ingest_preference():
        async with app.run_test() as pilot:
            _use_server_runtime(app)
            app.submit_library_ingest_job(source_path=str(source))
            for _ in range(_POLL_ATTEMPTS):
                states = {j.state for j in app.library_ingest_jobs.jobs()}
                if states == {IngestJobState.FAILED}:
                    break
                await pilot.pause(_POLL_INTERVAL)
            else:
                raise AssertionError(
                    f"{why}: job was not failed; states={states}"
                )

    job = app.library_ingest_jobs.jobs()[0]
    assert "track" in (job.error or "").lower(), job.error
    assert service.list_calls == 0, "an untrackable batch must not be polled"


@pytest.mark.asyncio
async def test_duplicate_content_at_different_path_resolves_existing_media_id(
    tmp_path: Path,
) -> None:
    """(task-2013) Byte-identical content at a DIFFERENT path takes the DB's
    duplicate-skip path (``add_media_with_keywords`` returns ``media_id=None``)
    and the writer's URL fallback misses (URLs differ). The job must still
    resolve the EXISTING item's id -- so the row keeps "Open in Library" --
    and must say it was a duplicate instead of impersonating a fresh ingest.
    """
    db = _make_db(tmp_path)
    body = "identical body " * 20
    first = _write_text_file(tmp_path, "report.txt", body)
    second = _write_text_file(tmp_path, "copy_of_report.txt", body)
    app = _IngestRunnerHarness(db)

    async with app.run_test() as pilot:
        first_job = app.submit_library_ingest_job(source_path=str(first))
        first_done = await _wait_for_job_state(
            app, pilot, first_job.job_id, IngestJobState.DONE
        )
        assert first_done.media_id is not None
        assert first_done.progress["message"].startswith("Imported ")

        second_job = app.submit_library_ingest_job(source_path=str(second))
        second_done = await _wait_for_job_state(
            app, pilot, second_job.job_id, IngestJobState.DONE
        )
        assert second_done.media_id == first_done.media_id, (
            "duplicate must resolve to the existing media item"
        )
        assert second_done.progress["message"].startswith("Already in Library"), (
            f"duplicate impersonated a fresh ingest: {second_done.progress}"
        )

        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_duplicate_hash_lookup_db_error_keeps_job_done_without_media_id(
    tmp_path: Path,
) -> None:
    """(task-2013, review follow-up) A ``DatabaseError`` from the hash-fallback
    lookup must not fail the job (the media row exists -- the DB deduped
    against it) and must not be swallowed into a fake match: the job stays
    DONE, unlinked, still labelled a duplicate."""
    from tldw_chatbook.DB.Client_Media_DB_v2 import DatabaseError

    db = _make_db(tmp_path)
    body = "identical body " * 20
    first = _write_text_file(tmp_path, "report.txt", body)
    second = _write_text_file(tmp_path, "copy_of_report.txt", body)
    app = _IngestRunnerHarness(db)

    async with app.run_test() as pilot:
        first_job = app.submit_library_ingest_job(source_path=str(first))
        await _wait_for_job_state(app, pilot, first_job.job_id, IngestJobState.DONE)

        def _boom(_content_hash, **_kwargs):
            raise DatabaseError("simulated lookup failure")

        db.get_media_by_hash = _boom

        second_job = app.submit_library_ingest_job(source_path=str(second))
        second_done = await _wait_for_job_state(
            app, pilot, second_job.job_id, IngestJobState.DONE
        )
        assert second_done.media_id is None
        assert second_done.progress["message"].startswith("Already in Library")

        await _wait_for_runner_idle(app, pilot)


@pytest.mark.asyncio
async def test_empty_file_failure_is_permanent_and_refuses_retry(
    tmp_path: Path,
) -> None:
    """(task-2015) A truly empty file fails identically on every attempt --
    offering Retry for it is dead bait (the UAT pressed it and got the same
    failure with '· attempt 2'). Same permanence family as missing-file and
    unsupported-type."""
    db = _make_db(tmp_path)
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    app = _IngestRunnerHarness(db)

    async with app.run_test() as pilot:
        failing_job = app.submit_library_ingest_job(source_path=str(empty))
        failed = await _wait_for_job_state(
            app, pilot, failing_job.job_id, IngestJobState.FAILED
        )
        await _wait_for_runner_idle(app, pilot)

        assert "empty" in (failed.error or "")
        assert failed.permanent is True
        assert app.retry_library_ingest_job(failed.job_id) is None


@pytest.mark.asyncio
async def test_directory_done_rows_all_carry_media_ids(tmp_path: Path) -> None:
    """(task-2015) Every done job from a folder submission resolves a real
    media id -- the UAT's 'folder rows lack Open in Library' observation was
    the duplicate-swallow case (task-2013); this pins the fresh-folder path.
    """
    db = _make_db(tmp_path)
    folder = tmp_path / "batch"
    folder.mkdir()
    (folder / "one.txt").write_text("first document body " * 10)
    (folder / "two.txt").write_text("second, different body " * 10)
    app = _IngestRunnerHarness(db)

    async with app.run_test() as pilot:
        jobs = app.submit_library_ingest_job(source_path=str(folder))
        submitted = jobs if isinstance(jobs, list) else [jobs]
        for job in submitted:
            done = await _wait_for_job_state(
                app, pilot, job.job_id, IngestJobState.DONE
            )
            assert done.media_id is not None, (
                f"folder done row without media id: {done.source_path}"
            )
        await _wait_for_runner_idle(app, pilot)


def test_worker_initializer_silences_import_noise(tmp_path: Path) -> None:
    """(task-2016) The pool initializer must stop loguru + warnings output
    from reaching stderr inside a worker process -- that stderr is the
    parent's real TTY under Textual, and import noise painted over the UI
    on every first submit."""
    result = _run_isolated_python(
        tmp_path,
        """
        import sys
        from tldw_chatbook.Local_Ingestion.ingest_parse_worker import (
            silence_ingest_worker_import_noise,
        )

        silence_ingest_worker_import_noise()

        import logging
        import warnings
        from loguru import logger

        logger.warning("should not reach stderr")
        warnings.warn("should not reach stderr either", UserWarning)
        logging.warning("should not reach stderr three")
        print("MARKER-OK")
        """,
    )
    assert result.returncode == 0, result.stderr
    assert "MARKER-OK" in result.stdout
    assert "should not reach stderr" not in result.stderr


@pytest.mark.asyncio
async def test_folder_submission_shares_one_batch_id(tmp_path: Path) -> None:
    """(task-2221 owner ruling) Every file expanded from one folder
    submission carries the same minted batch id, so the queue can group
    the run under one header; a single-file submission stays batchless."""
    db = _make_db(tmp_path)
    folder = tmp_path / "run"
    folder.mkdir()
    _write_text_file(folder, "a.txt", "Document A.")
    _write_text_file(folder, "b.txt", "Document B.")
    solo = _write_text_file(tmp_path, "solo.txt", "Solo document.")
    app = _IngestRunnerHarness(db, worker_count=2)

    async with app.run_test() as pilot:
        app.submit_library_ingest_job(source_path=str(folder))
        solo_job = app.submit_library_ingest_job(source_path=str(solo))

        jobs = {Path(j.source_path).name: j for j in app.library_ingest_jobs.jobs()}
        assert jobs["a.txt"].batch_id is not None
        assert jobs["a.txt"].batch_id == jobs["b.txt"].batch_id
        assert jobs["a.txt"].batch_id.startswith("local-")
        assert jobs["solo.txt"].batch_id is None
        assert solo_job.batch_id is None

        for submitted in jobs.values():
            await _wait_for_job_state(
                app,
                pilot,
                submitted.job_id,
                IngestJobState.DONE,
            )
        await _wait_for_runner_idle(app, pilot)
