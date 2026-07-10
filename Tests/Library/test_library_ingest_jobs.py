"""Pure ingest job model and registry contracts (L3b Task 1)."""

from __future__ import annotations

import time

import pytest

from tldw_chatbook.Library.library_ingest_jobs import (
    IngestJobState,
    LibraryIngestJob,
    LibraryIngestJobRegistry,
)


def test_submit_assigns_sequential_ids_and_queued_state() -> None:
    registry = LibraryIngestJobRegistry()

    job1 = registry.submit(source_path="/tmp/a.txt")
    job2 = registry.submit(source_path="/tmp/b.txt")

    assert job1.job_id == "ingest-job-1"
    assert job2.job_id == "ingest-job-2"
    assert job1.state == IngestJobState.QUEUED
    assert job2.state == IngestJobState.QUEUED
    assert job1.source_path == "/tmp/a.txt"
    assert job1.submitted_at > 0.0


def test_next_queued_returns_fifo_order_and_skips_non_queued() -> None:
    registry = LibraryIngestJobRegistry()
    job1 = registry.submit(source_path="/tmp/a.txt")
    job2 = registry.submit(source_path="/tmp/b.txt")
    job3 = registry.submit(source_path="/tmp/c.txt")

    assert registry.next_queued().job_id == job1.job_id

    registry.mark_running(job1.job_id)
    assert registry.next_queued().job_id == job2.job_id

    registry.mark_running(job2.job_id)
    assert registry.next_queued().job_id == job3.job_id

    registry.mark_running(job3.job_id)
    assert registry.next_queued() is None


def test_mark_running_transitions_and_stamps_started_at() -> None:
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")

    running = registry.mark_running(job.job_id, detected_type="plaintext")

    assert running.state == IngestJobState.RUNNING
    assert running.detected_type == "plaintext"
    assert running.started_at is not None
    assert running.started_at >= job.submitted_at


def test_mark_done_transitions_and_fills_media_id() -> None:
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")
    running = registry.mark_running(job.job_id)

    done = registry.mark_done(job.job_id, media_id=42)

    assert done.state == IngestJobState.DONE
    assert done.media_id == 42
    assert done.finished_at is not None
    assert done.finished_at >= running.started_at


def test_mark_failed_transitions_and_fills_error() -> None:
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")
    registry.mark_running(job.job_id)

    failed = registry.mark_failed(job.job_id, error="file not found")

    assert failed.state == IngestJobState.FAILED
    assert failed.error == "file not found"
    assert failed.finished_at is not None


def test_unknown_job_id_returns_none_without_raising() -> None:
    registry = LibraryIngestJobRegistry()

    assert registry.mark_running("ingest-job-999") is None
    assert registry.mark_done("ingest-job-999", media_id=1) is None
    assert registry.mark_failed("ingest-job-999", error="x") is None
    assert registry.requeue("ingest-job-999") is None


def test_requeue_only_works_on_failed_jobs() -> None:
    registry = LibraryIngestJobRegistry()

    queued_job = registry.submit(source_path="/tmp/a.txt")
    assert registry.requeue(queued_job.job_id) is None

    running_job = registry.submit(source_path="/tmp/b.txt")
    registry.mark_running(running_job.job_id)
    assert registry.requeue(running_job.job_id) is None

    done_job = registry.submit(source_path="/tmp/c.txt")
    registry.mark_running(done_job.job_id)
    registry.mark_done(done_job.job_id, media_id=7)
    assert registry.requeue(done_job.job_id) is None


def test_requeue_failed_job_appends_fresh_queued_copy() -> None:
    registry = LibraryIngestJobRegistry()
    original = registry.submit(
        source_path="/tmp/a.txt",
        title="My Title",
        author="Someone",
        keywords=("k1", "k2"),
        perform_analysis=True,
        chunk_enabled=True,
        chunk_size=750,
    )
    registry.mark_running(original.job_id, detected_type="plaintext")
    failed = registry.mark_failed(original.job_id, error="boom")

    time.sleep(0.001)
    requeued = registry.requeue(failed.job_id)

    assert requeued is not None
    assert requeued.job_id != failed.job_id
    assert requeued.state == IngestJobState.QUEUED
    # Form fields preserved.
    assert requeued.source_path == "/tmp/a.txt"
    assert requeued.title == "My Title"
    assert requeued.author == "Someone"
    assert requeued.keywords == ("k1", "k2")
    assert requeued.perform_analysis is True
    assert requeued.chunk_enabled is True
    assert requeued.chunk_size == 750
    # Fresh state -- not a copy of the failed job's runtime fields.
    assert requeued.detected_type == ""
    assert requeued.media_id is None
    assert requeued.error == ""
    assert requeued.started_at is None
    assert requeued.finished_at is None
    assert requeued.submitted_at >= failed.submitted_at

    # The original failed job is left untouched in the registry.
    jobs_by_id = {j.job_id: j for j in registry.jobs()}
    assert jobs_by_id[failed.job_id].state == IngestJobState.FAILED


def test_jobs_returns_newest_first_immutable_snapshot() -> None:
    registry = LibraryIngestJobRegistry()
    job1 = registry.submit(source_path="/tmp/a.txt")
    job2 = registry.submit(source_path="/tmp/b.txt")
    job3 = registry.submit(source_path="/tmp/c.txt")

    snapshot = registry.jobs()

    assert isinstance(snapshot, tuple)
    assert [j.job_id for j in snapshot] == [job3.job_id, job2.job_id, job1.job_id]

    # Mutating a returned job must not corrupt registry state.
    snapshot[0].state = IngestJobState.FAILED
    snapshot[0].title = "tampered"

    fresh_snapshot = registry.jobs()
    assert fresh_snapshot[0].state == IngestJobState.QUEUED
    assert fresh_snapshot[0].title == ""


def test_counts_returns_per_state_counts() -> None:
    registry = LibraryIngestJobRegistry()
    job1 = registry.submit(source_path="/tmp/a.txt")
    job2 = registry.submit(source_path="/tmp/b.txt")
    job3 = registry.submit(source_path="/tmp/c.txt")
    registry.submit(source_path="/tmp/d.txt")

    registry.mark_running(job1.job_id)
    registry.mark_done(job1.job_id, media_id=1)
    registry.mark_running(job2.job_id)
    registry.mark_failed(job2.job_id, error="nope")
    registry.mark_running(job3.job_id)

    counts = registry.counts()

    assert counts == {"queued": 1, "running": 1, "done": 1, "failed": 1}


def test_listener_fires_once_per_successful_mutation() -> None:
    registry = LibraryIngestJobRegistry()
    calls: list[int] = []
    registry.add_listener(lambda: calls.append(1))

    job = registry.submit(source_path="/tmp/a.txt")
    assert len(calls) == 1

    registry.mark_running(job.job_id)
    assert len(calls) == 2

    registry.mark_done(job.job_id, media_id=1)
    assert len(calls) == 3

    # Unknown id -- no mutation, listener must not fire.
    registry.mark_running("ingest-job-999")
    assert len(calls) == 3

    failed_job = registry.submit(source_path="/tmp/b.txt")
    assert len(calls) == 4
    registry.mark_failed(failed_job.job_id, error="boom")
    assert len(calls) == 5
    registry.requeue(failed_job.job_id)
    assert len(calls) == 6


def test_listener_exception_is_swallowed_and_other_listeners_still_run() -> None:
    registry = LibraryIngestJobRegistry()
    calls: list[str] = []

    def bad_listener() -> None:
        raise RuntimeError("boom")

    def good_listener() -> None:
        calls.append("good")

    registry.add_listener(bad_listener)
    registry.add_listener(good_listener)

    registry.submit(source_path="/tmp/a.txt")  # Must not raise.

    assert calls == ["good"]


def test_remove_listener_stops_future_notifications() -> None:
    registry = LibraryIngestJobRegistry()
    calls: list[int] = []

    def listener() -> None:
        calls.append(1)

    registry.add_listener(listener)
    registry.submit(source_path="/tmp/a.txt")
    assert len(calls) == 1

    registry.remove_listener(listener)
    registry.submit(source_path="/tmp/b.txt")
    assert len(calls) == 1


def test_remove_listener_unknown_callback_does_not_raise() -> None:
    registry = LibraryIngestJobRegistry()

    def listener() -> None:
        pass

    registry.remove_listener(listener)  # Never added -- must not raise.


def test_keywords_stored_as_tuple() -> None:
    registry = LibraryIngestJobRegistry()

    job = registry.submit(source_path="/tmp/a.txt", keywords=["a", "b"])

    assert isinstance(job.keywords, tuple)
    assert job.keywords == ("a", "b")

    default_job = registry.submit(source_path="/tmp/b.txt")
    assert default_job.keywords == ()
    assert isinstance(default_job.keywords, tuple)


def test_runner_active_defaults_false_and_is_a_plain_attribute() -> None:
    registry = LibraryIngestJobRegistry()

    assert registry.runner_active is False

    registry.runner_active = True
    assert registry.runner_active is True
