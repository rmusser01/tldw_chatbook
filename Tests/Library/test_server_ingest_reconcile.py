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
