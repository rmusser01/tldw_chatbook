"""Tests for reconciling server job statuses into the local ingest registry.

Exercised against a real ``LibraryIngestJobRegistry`` -- it is Textual-free, so
the whole reconciliation step is testable without an app, a worker or a server
(task-684.2). Only the polling loop that feeds this needs wiring.
"""

from __future__ import annotations

from types import SimpleNamespace

from tldw_chatbook.Library.library_ingest_jobs import (
    IngestJobState,
    LibraryIngestJobRegistry,
)
from tldw_chatbook.Library.server_ingest_reconcile import reconcile_remote_ingest_jobs


def _status(job_id, status, **kwargs):
    """A stand-in for ``MediaIngestJobStatus`` (extra="ignore" on the real one)."""
    return SimpleNamespace(
        id=job_id,
        status=status,
        error_message=kwargs.get("error_message"),
        progress_percent=kwargs.get("progress_percent"),
        progress_message=kwargs.get("progress_message"),
        cancellation_reason=kwargs.get("cancellation_reason"),
    )


def _server_job(registry, *, remote_job_id, source="/tmp/a.mp3"):
    job = registry.submit(source_path=source, origin="server")
    return registry.attach_remote(job.job_id, remote_job_id=remote_job_id, batch_id="b1")


def test_running_status_moves_the_job_to_parsing():
    registry = LibraryIngestJobRegistry()
    job = _server_job(registry, remote_job_id="11")

    applied = reconcile_remote_ingest_jobs(registry, [_status(11, "running")])

    assert applied == 1
    assert registry.jobs()[0].state is IngestJobState.PARSING


def test_completed_status_finishes_without_a_local_media_id():
    """A server completion has no local media row, and must not claim one.

    ``ReadingImportResponse`` carries only counts -- no media id -- so a server
    job legitimately finishes with ``media_id`` unset. The local invariant that
    a *local* completion has a media id stays intact.
    """
    registry = LibraryIngestJobRegistry()
    job = _server_job(registry, remote_job_id="11")

    reconcile_remote_ingest_jobs(registry, [_status(11, "completed")])

    done = registry.jobs()[0]
    assert done.state is IngestJobState.DONE
    assert done.media_id is None


def test_failed_status_carries_the_server_message():
    registry = LibraryIngestJobRegistry()
    _server_job(registry, remote_job_id="11")

    reconcile_remote_ingest_jobs(
        registry, [_status(11, "failed", error_message="transcoder exploded")]
    )

    failed = registry.jobs()[0]
    assert failed.state is IngestJobState.FAILED
    assert "transcoder exploded" in failed.error


def test_cancelled_status_carries_the_reason():
    registry = LibraryIngestJobRegistry()
    _server_job(registry, remote_job_id="11")

    reconcile_remote_ingest_jobs(
        registry, [_status(11, "cancelled", cancellation_reason="user asked")]
    )

    cancelled = registry.jobs()[0]
    assert cancelled.state is IngestJobState.CANCELLED
    assert "user asked" in cancelled.error


def test_unknown_status_leaves_the_job_untouched():
    """An unrecognised status must not move the job at all."""
    registry = LibraryIngestJobRegistry()
    _server_job(registry, remote_job_id="11")

    applied = reconcile_remote_ingest_jobs(registry, [_status(11, "reticulating")])

    assert applied == 0
    assert registry.jobs()[0].state is IngestJobState.QUEUED


def test_status_for_an_unknown_remote_id_is_ignored():
    """A status for a job this registry never submitted is not an error."""
    registry = LibraryIngestJobRegistry()
    _server_job(registry, remote_job_id="11")

    applied = reconcile_remote_ingest_jobs(registry, [_status(999, "completed")])

    assert applied == 0
    assert registry.jobs()[0].state is IngestJobState.QUEUED


def test_local_jobs_are_never_touched_by_a_server_status():
    """A remote id must not collide with a local job's identity."""
    registry = LibraryIngestJobRegistry()
    local = registry.submit(source_path="/tmp/local.txt")

    applied = reconcile_remote_ingest_jobs(registry, [_status(1, "completed")])

    assert applied == 0
    assert registry.jobs()[0].job_id == local.job_id
    assert registry.jobs()[0].state is IngestJobState.QUEUED


def test_reconciling_the_same_terminal_status_twice_is_idempotent():
    """Polling repeats; a settled job must not be rewritten on every pass."""
    registry = LibraryIngestJobRegistry()
    _server_job(registry, remote_job_id="11")

    first = reconcile_remote_ingest_jobs(registry, [_status(11, "completed")])
    second = reconcile_remote_ingest_jobs(registry, [_status(11, "completed")])

    assert first == 1
    assert second == 0, "a settled job should not be transitioned again"
    assert registry.jobs()[0].state is IngestJobState.DONE


def test_progress_is_recorded_for_an_in_flight_job():
    registry = LibraryIngestJobRegistry()
    _server_job(registry, remote_job_id="11")

    reconcile_remote_ingest_jobs(
        registry,
        [_status(11, "running", progress_percent=42.0, progress_message="transcoding")],
    )

    job = registry.jobs()[0]
    assert job.progress is not None
    assert job.progress.get("percent") == 42.0
    assert job.progress.get("message") == "transcoding"


def test_a_dict_shaped_status_is_accepted():
    """The client may hand back plain dicts as well as models."""
    registry = LibraryIngestJobRegistry()
    _server_job(registry, remote_job_id="11")

    applied = reconcile_remote_ingest_jobs(registry, [{"id": 11, "status": "completed"}])

    assert applied == 1
    assert registry.jobs()[0].state is IngestJobState.DONE


def test_a_local_job_carrying_a_remote_id_is_still_never_touched():
    """The origin filter must be load-bearing, not incidental.

    Local jobs have no remote id today, so a lookup miss would protect them by
    accident. This gives a *local* job a remote id that collides with a server
    status, so only the explicit origin check can keep it safe.
    """
    registry = LibraryIngestJobRegistry()
    local = registry.submit(source_path="/tmp/local.txt")
    registry.attach_remote(local.job_id, remote_job_id="11", batch_id="b1")

    # Deliberately a "failed" status, not "completed": mark_remote_done has its
    # own origin guard, so completed would be caught even without the filter
    # here. mark_failed has no such guard, so this path is only safe because
    # the reconciler refuses to match a non-server job at all.
    applied = reconcile_remote_ingest_jobs(
        registry, [_status(11, "failed", error_message="not yours")]
    )

    assert applied == 0
    assert registry.jobs()[0].state is IngestJobState.QUEUED
    assert registry.jobs()[0].error == ""


def test_a_settled_job_is_not_re_stamped_by_a_repeat_poll():
    """Re-polling a finished job must not move its finish timestamp.

    Enforced in two places -- the reconciler skips it, and the registry's own
    terminal guard refuses it -- so this asserts the observable outcome rather
    than which layer said no.
    """
    registry = LibraryIngestJobRegistry()
    _server_job(registry, remote_job_id="11")

    reconcile_remote_ingest_jobs(registry, [_status(11, "completed")])
    first_finish = registry.jobs()[0].finished_at_wall

    reconcile_remote_ingest_jobs(registry, [_status(11, "completed")])

    assert registry.jobs()[0].finished_at_wall == first_finish


# --- which batches still need watching -------------------------------------


def test_pending_batches_lists_only_unfinished_server_batches():
    """Polling must stop once a batch has nothing left to watch.

    A finished batch that kept being polled would hit the server forever for an
    answer that cannot change.
    """
    from tldw_chatbook.Library.server_ingest_reconcile import pending_remote_batches

    registry = LibraryIngestJobRegistry()
    # batch-1: one still running.
    a = registry.submit(source_path="/tmp/a.mp3", origin="server")
    registry.attach_remote(a.job_id, remote_job_id="1", batch_id="batch-1")
    # batch-2: everything settled.
    b = registry.submit(source_path="/tmp/b.mp3", origin="server")
    registry.attach_remote(b.job_id, remote_job_id="2", batch_id="batch-2")
    registry.mark_remote_done(b.job_id)
    # A local job has no batch at all.
    registry.submit(source_path="/tmp/c.txt")

    assert pending_remote_batches(registry) == ("batch-1",)


def test_pending_batches_is_empty_when_nothing_is_outstanding():
    from tldw_chatbook.Library.server_ingest_reconcile import pending_remote_batches

    registry = LibraryIngestJobRegistry()
    registry.submit(source_path="/tmp/local.txt")

    assert pending_remote_batches(registry) == ()


def test_pending_batches_ignores_hidden_jobs():
    """A dismissed job must not keep its batch alive."""
    from tldw_chatbook.Library.server_ingest_reconcile import pending_remote_batches

    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.mp3", origin="server")
    registry.attach_remote(job.job_id, remote_job_id="1", batch_id="batch-1")
    registry.mark_cancelled(job.job_id)
    registry.dismiss(job.job_id)

    assert pending_remote_batches(registry) == ()


def test_pending_batches_deduplicates_and_keeps_submission_order():
    from tldw_chatbook.Library.server_ingest_reconcile import pending_remote_batches

    registry = LibraryIngestJobRegistry()
    for index, batch in enumerate(("batch-2", "batch-1", "batch-2"), start=1):
        job = registry.submit(source_path=f"/tmp/{index}.mp3", origin="server")
        registry.attach_remote(job.job_id, remote_job_id=str(index), batch_id=batch)

    assert pending_remote_batches(registry) == ("batch-2", "batch-1")


def test_a_real_model_from_a_live_server_settles_the_job():
    """End-to-end on real data: the actual model, the actual registry.

    Every other test here feeds ``SimpleNamespace`` or a dict. Those exercise
    ``_field``'s two branches but cannot show that a genuine
    ``MediaIngestJobStatus`` -- as parsed from a live response -- carries the
    fields this layer reads under the names it reads them by.

    That mattered: ``result`` was typed as the reading-list import model, so a
    *completed* job's response did not validate at all. The poller logged the
    ``ValidationError`` as transient and retried, so this layer was never even
    reached and the job would have stayed "queued" in the UI indefinitely.

    Payload captured verbatim from a live server (2026-07-26), including the
    integer ``id`` -- remote ids are stored as strings, so the lookup depends on
    both sides normalising the same way.
    """
    from tldw_chatbook.tldw_api.media_reading_schemas import MediaIngestJobStatus

    status = MediaIngestJobStatus.model_validate(
        {
            "id": 281,
            "uuid": "aacbb0d9-111e-43d1-9a2a-3d57fcf97596",
            "status": "completed",
            "job_type": "media_ingest_item",
            "progress_percent": 100.0,
            "progress_message": "completed",
            "result": {
                "status": "Success",
                "media_id": 1125,
                "media_uuid": "bee6d9a0-931f-4c11-a87a-07ddebc66443",
                "error": None,
                "warnings": None,
                "db_message": "Media 'gettysburg' already exists.",
            },
            "media_type": "document",
            "source": "gettysburg.txt",
            "source_kind": "file",
            "batch_id": "1516ae79-58fd-46e8-834c-52c5a40bfbd8",
        }
    )

    registry = LibraryIngestJobRegistry()
    # The server issues an int id; the registry stores it as a string.
    _server_job(registry, remote_job_id="281", source="gettysburg.txt")

    assert reconcile_remote_ingest_jobs(registry, [status]) == 1
    assert registry.jobs()[0].state is IngestJobState.DONE
