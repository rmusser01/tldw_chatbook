"""Pure ingest job model and registry contracts (L3b Task 1)."""

from __future__ import annotations

import time
from datetime import datetime, timezone

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


# --- F3: mark_parsing / mark_writing (PARSING/WRITING replace RUNNING) -----


def test_mark_parsing_transitions_and_stamps_started_at() -> None:
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")

    parsing = registry.mark_parsing(job.job_id, detected_type="plaintext")

    assert parsing.state == IngestJobState.PARSING
    assert parsing.detected_type == "plaintext"
    assert parsing.started_at is not None
    assert parsing.started_at >= job.submitted_at


def test_mark_parsing_defaults_detected_type_to_empty_string() -> None:
    """The coordinator (Task 4) frequently calls this before the parse pool
    has even started the job, so the type is often still unknown."""
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")

    parsing = registry.mark_parsing(job.job_id)

    assert parsing.detected_type == ""


def test_mark_parsing_rejects_a_job_that_is_not_queued() -> None:
    """(F3) Unlike the old, unguarded ``mark_running``, ``mark_parsing`` IS
    guarded: with multiple jobs now able to be PARSING at once, silently
    re-parsing an already-active or terminal job would be a coordinator bug
    worth surfacing, not swallowing."""
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")
    registry.mark_parsing(job.job_id)

    assert registry.mark_parsing(job.job_id) is None  # already PARSING

    registry.mark_writing(job.job_id)
    assert registry.mark_parsing(job.job_id) is None  # already WRITING

    registry.mark_done(job.job_id, media_id=1)
    assert registry.mark_parsing(job.job_id) is None  # terminal


def test_mark_parsing_unknown_job_id_returns_none_without_raising() -> None:
    registry = LibraryIngestJobRegistry()

    assert registry.mark_parsing("ingest-job-999") is None


def test_mark_writing_transitions_from_parsing() -> None:
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")
    parsing = registry.mark_parsing(job.job_id, detected_type="plaintext")

    writing = registry.mark_writing(job.job_id)

    assert writing.state == IngestJobState.WRITING
    # started_at is untouched by the PARSING -> WRITING transition -- it
    # keeps measuring the job's total active time (parse + write combined).
    assert writing.started_at == parsing.started_at
    # detected_type persists across the transition too.
    assert writing.detected_type == "plaintext"


def test_mark_writing_rejects_a_job_that_is_not_parsing() -> None:
    registry = LibraryIngestJobRegistry()
    queued_job = registry.submit(source_path="/tmp/a.txt")
    assert registry.mark_writing(queued_job.job_id) is None  # still QUEUED

    writing_job = registry.submit(source_path="/tmp/b.txt")
    registry.mark_parsing(writing_job.job_id)
    registry.mark_writing(writing_job.job_id)
    assert registry.mark_writing(writing_job.job_id) is None  # already WRITING

    done_job = registry.submit(source_path="/tmp/c.txt")
    registry.mark_parsing(done_job.job_id)
    registry.mark_writing(done_job.job_id)
    registry.mark_done(done_job.job_id, media_id=1)
    assert registry.mark_writing(done_job.job_id) is None  # terminal


def test_mark_writing_unknown_job_id_returns_none_without_raising() -> None:
    registry = LibraryIngestJobRegistry()

    assert registry.mark_writing("ingest-job-999") is None


def test_mark_parsing_and_mark_writing_are_noop_for_a_superseded_job_id() -> None:
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")
    registry.mark_running(job.job_id)
    failed = registry.mark_failed(job.job_id, error="boom")
    registry.requeue(failed.job_id)

    calls: list[int] = []
    registry.add_listener(lambda: calls.append(1))

    assert registry.mark_parsing(failed.job_id) is None
    assert registry.mark_writing(failed.job_id) is None
    assert calls == []


# --- mark_running: DEPRECATED single-call alias -----------------------------
# TEMPORARY Task-4 removes this whole section alongside the mark_running
# alias itself once app.py's coordinator calls mark_parsing/mark_writing for
# real (see library_ingest_jobs.py's mark_running docstring).


def test_mark_running_alias_drives_queued_straight_to_writing() -> None:
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")

    running = registry.mark_running(job.job_id, detected_type="plaintext")

    assert running.state == IngestJobState.WRITING
    assert running.detected_type == "plaintext"
    assert running.started_at is not None
    assert running.started_at >= job.submitted_at


def test_mark_running_alias_fires_the_listener_twice() -> None:
    """Internally mark_parsing then mark_writing -- two real transitions,
    each firing the listener once."""
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")
    calls: list[int] = []
    registry.add_listener(lambda: calls.append(1))

    registry.mark_running(job.job_id)

    assert len(calls) == 2


def test_mark_running_alias_rejects_a_job_that_is_not_queued() -> None:
    """Inherits mark_parsing's QUEUED-only guard."""
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")
    registry.mark_running(job.job_id)

    assert registry.mark_running(job.job_id) is None


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


def test_mark_failed_permanent_defaults_false() -> None:
    """(M4, fix batch F1b) Every pre-existing caller of ``mark_failed`` --
    none of which pass ``permanent`` -- keeps today's always-retryable
    behavior."""
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")
    registry.mark_running(job.job_id)

    failed = registry.mark_failed(job.job_id, error="boom")

    assert failed.permanent is False


def test_mark_failed_permanent_true_is_stamped_on_the_job() -> None:
    """(M4) The queue-runner classifies validation-class failures (an
    unsupported file type, a missing source file) as ``permanent`` --
    ``mark_failed`` just stores whatever it's told."""
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")
    registry.mark_running(job.job_id)

    failed = registry.mark_failed(
        job.job_id, error="Unsupported file type: .xyz.", permanent=True
    )

    assert failed.permanent is True
    assert failed.state == IngestJobState.FAILED


def test_requeue_refuses_a_permanent_failed_job() -> None:
    """(M4) ``requeue`` is defense in depth alongside the queue row's own
    ``can_retry`` gating and Home's ``retry_available`` gating -- even a
    direct call against a permanently-failed job_id must be a no-op."""
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")
    registry.mark_running(job.job_id)
    failed = registry.mark_failed(
        job.job_id, error="Unsupported file type: .xyz.", permanent=True
    )

    assert registry.requeue(failed.job_id) is None
    # The permanently-failed job is untouched -- still visible, still FAILED.
    jobs_by_id = {j.job_id: j for j in registry.jobs()}
    assert jobs_by_id[failed.job_id].state == IngestJobState.FAILED


def test_requeue_still_works_for_a_non_permanent_failed_job() -> None:
    """(M4) An ordinary (non-permanent) failure stays retryable -- no
    regression from adding the ``permanent`` guard."""
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")
    registry.mark_running(job.job_id)
    failed = registry.mark_failed(job.job_id, error="boom", permanent=False)

    requeued = registry.requeue(failed.job_id)

    assert requeued is not None
    assert requeued.state == IngestJobState.QUEUED


def test_finished_at_wall_blank_until_a_terminal_transition() -> None:
    """(H1, fix batch F1b) ``finished_at_wall`` is the wall-clock ISO-8601
    UTC counterpart to the monotonic ``finished_at`` -- Home's Recent feed
    needs a real timestamp to sort/display by, since ``time.monotonic()``
    has no fixed epoch. It stays "" until a job reaches a terminal state."""
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")
    assert job.finished_at_wall == ""

    running = registry.mark_running(job.job_id)
    assert running.finished_at_wall == ""


def test_mark_done_stamps_finished_at_wall_iso_utc() -> None:
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")
    registry.mark_running(job.job_id)

    before = datetime.now(timezone.utc)
    done = registry.mark_done(job.job_id, media_id=42)
    after = datetime.now(timezone.utc)

    assert done.finished_at_wall != ""
    stamped = datetime.fromisoformat(done.finished_at_wall)
    assert before <= stamped <= after


def test_mark_failed_stamps_finished_at_wall_iso_utc() -> None:
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")
    registry.mark_running(job.job_id)

    before = datetime.now(timezone.utc)
    failed = registry.mark_failed(job.job_id, error="file not found")
    after = datetime.now(timezone.utc)

    assert failed.finished_at_wall != ""
    stamped = datetime.fromisoformat(failed.finished_at_wall)
    assert before <= stamped <= after


def test_unknown_job_id_returns_none_without_raising() -> None:
    registry = LibraryIngestJobRegistry()

    assert registry.mark_running("ingest-job-999") is None
    assert registry.mark_parsing("ingest-job-999") is None
    assert registry.mark_writing("ingest-job-999") is None
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

    # (L3b AB wave, B1) The original failed job is superseded -- retry
    # supersedes the failed original, so it no longer appears in jobs().
    jobs_by_id = {j.job_id: j for j in registry.jobs()}
    assert failed.job_id not in jobs_by_id


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
    job4 = registry.submit(source_path="/tmp/d.txt")
    registry.submit(source_path="/tmp/e.txt")

    registry.mark_parsing(job1.job_id)
    registry.mark_writing(job1.job_id)
    registry.mark_done(job1.job_id, media_id=1)
    registry.mark_parsing(job2.job_id)
    registry.mark_writing(job2.job_id)
    registry.mark_failed(job2.job_id, error="nope")
    registry.mark_parsing(job3.job_id)
    registry.mark_writing(job3.job_id)
    registry.mark_parsing(job4.job_id)

    counts = registry.counts()

    assert counts == {
        "queued": 1,
        "parsing": 1,
        "writing": 1,
        "done": 1,
        "failed": 1,
    }


def test_listener_fires_once_per_successful_mutation() -> None:
    registry = LibraryIngestJobRegistry()
    calls: list[int] = []
    registry.add_listener(lambda: calls.append(1))

    job = registry.submit(source_path="/tmp/a.txt")
    assert len(calls) == 1

    registry.mark_parsing(job.job_id)
    assert len(calls) == 2

    registry.mark_writing(job.job_id)
    assert len(calls) == 3

    registry.mark_done(job.job_id, media_id=1)
    assert len(calls) == 4

    # Unknown id -- no mutation, listener must not fire.
    registry.mark_parsing("ingest-job-999")
    assert len(calls) == 4

    failed_job = registry.submit(source_path="/tmp/b.txt")
    assert len(calls) == 5
    registry.mark_failed(failed_job.job_id, error="boom")
    assert len(calls) == 6
    registry.requeue(failed_job.job_id)
    assert len(calls) == 7


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


# --- L3b AB wave: B1 (retry supersedes) / B2 (dismiss + clear_finished) ---


def test_requeue_supersedes_original_from_jobs_and_counts() -> None:
    """B1: retrying a failed job hides the original from jobs()/counts() --
    the queue must show ONE row per retried file (the fresh copy), not two."""
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")
    registry.mark_running(job.job_id)
    failed = registry.mark_failed(job.job_id, error="boom")

    requeued = registry.requeue(failed.job_id)

    job_ids = [j.job_id for j in registry.jobs()]
    assert requeued.job_id in job_ids
    assert failed.job_id not in job_ids
    assert len(job_ids) == 1
    counts = registry.counts()
    assert counts["failed"] == 0
    assert counts["queued"] == 1


def test_requeue_twice_on_same_original_is_a_noop_second_time() -> None:
    """A superseded original must not be requeue-able again -- otherwise a
    stale Retry button (or a double click) could silently fork duplicate
    retries off the same dead job_id."""
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")
    registry.mark_running(job.job_id)
    failed = registry.mark_failed(job.job_id, error="boom")

    first = registry.requeue(failed.job_id)
    assert first is not None

    second = registry.requeue(failed.job_id)
    assert second is None
    assert len(registry.jobs()) == 1


def test_mark_methods_are_noop_for_a_superseded_job_id() -> None:
    """Once a failed job is superseded by retry, mark_running/mark_done/
    mark_failed against its (now-hidden) job_id must be safe no-ops -- they
    must not resurrect it into jobs()/counts() or fire the listener."""
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")
    registry.mark_running(job.job_id)
    failed = registry.mark_failed(job.job_id, error="boom")
    registry.requeue(failed.job_id)

    calls: list[int] = []
    registry.add_listener(lambda: calls.append(1))

    assert registry.mark_running(failed.job_id) is None
    assert registry.mark_done(failed.job_id, media_id=1) is None
    assert registry.mark_failed(failed.job_id, error="still boom") is None
    assert calls == []
    assert len(registry.jobs()) == 1


def test_dismiss_only_valid_for_failed_jobs() -> None:
    registry = LibraryIngestJobRegistry()

    queued_job = registry.submit(source_path="/tmp/a.txt")
    assert registry.dismiss(queued_job.job_id) is None

    running_job = registry.submit(source_path="/tmp/b.txt")
    registry.mark_running(running_job.job_id)
    assert registry.dismiss(running_job.job_id) is None

    done_job = registry.submit(source_path="/tmp/c.txt")
    registry.mark_running(done_job.job_id)
    registry.mark_done(done_job.job_id, media_id=7)
    assert registry.dismiss(done_job.job_id) is None

    # None of the above were actually dismissed.
    job_ids = [j.job_id for j in registry.jobs()]
    assert queued_job.job_id in job_ids
    assert running_job.job_id in job_ids
    assert done_job.job_id in job_ids


def test_dismiss_unknown_job_id_returns_none() -> None:
    registry = LibraryIngestJobRegistry()

    assert registry.dismiss("ingest-job-999") is None


def test_dismiss_failed_job_removes_from_jobs_and_counts() -> None:
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")
    registry.mark_running(job.job_id)
    failed = registry.mark_failed(job.job_id, error="boom")

    dismissed = registry.dismiss(failed.job_id)

    assert dismissed is not None
    assert dismissed.job_id == failed.job_id
    assert registry.jobs() == ()
    assert registry.counts()["failed"] == 0


def test_dismiss_twice_is_a_noop_second_time() -> None:
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")
    registry.mark_running(job.job_id)
    failed = registry.mark_failed(job.job_id, error="boom")
    registry.dismiss(failed.job_id)

    calls: list[int] = []
    registry.add_listener(lambda: calls.append(1))

    assert registry.dismiss(failed.job_id) is None
    assert calls == []


def test_dismiss_fires_listener_once_and_not_when_it_is_a_noop() -> None:
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")
    registry.mark_running(job.job_id)
    failed = registry.mark_failed(job.job_id, error="boom")
    queued_job = registry.submit(source_path="/tmp/b.txt")

    calls: list[int] = []
    registry.add_listener(lambda: calls.append(1))

    # No-op: not a FAILED job -- must not fire.
    registry.dismiss(queued_job.job_id)
    assert calls == []

    registry.dismiss(failed.job_id)
    assert len(calls) == 1


def test_clear_finished_removes_all_done_and_failed_returns_count() -> None:
    """(F3 re-anchor) Both new active states, PARSING and WRITING, must
    survive clear_finished untouched -- not just one of them."""
    registry = LibraryIngestJobRegistry()
    queued_job = registry.submit(source_path="/tmp/queued.txt")
    parsing_job = registry.submit(source_path="/tmp/parsing.txt")
    registry.mark_parsing(parsing_job.job_id)
    writing_job = registry.submit(source_path="/tmp/writing.txt")
    registry.mark_parsing(writing_job.job_id)
    registry.mark_writing(writing_job.job_id)
    done_job = registry.submit(source_path="/tmp/done.txt")
    registry.mark_running(done_job.job_id)
    registry.mark_done(done_job.job_id, media_id=1)
    failed_job = registry.submit(source_path="/tmp/failed.txt")
    registry.mark_running(failed_job.job_id)
    registry.mark_failed(failed_job.job_id, error="boom")

    removed = registry.clear_finished()

    assert removed == 2
    job_ids = [j.job_id for j in registry.jobs()]
    assert set(job_ids) == {queued_job.job_id, parsing_job.job_id, writing_job.job_id}
    counts = registry.counts()
    assert counts["done"] == 0
    assert counts["failed"] == 0
    assert counts["queued"] == 1
    assert counts["parsing"] == 1
    assert counts["writing"] == 1


def test_clear_finished_is_a_noop_when_nothing_to_clear() -> None:
    registry = LibraryIngestJobRegistry()
    registry.submit(source_path="/tmp/queued.txt")

    calls: list[int] = []
    registry.add_listener(lambda: calls.append(1))

    removed = registry.clear_finished()

    assert removed == 0
    assert calls == []


def test_clear_finished_fires_listener_once() -> None:
    registry = LibraryIngestJobRegistry()
    job = registry.submit(source_path="/tmp/a.txt")
    registry.mark_running(job.job_id)
    registry.mark_done(job.job_id, media_id=1)

    calls: list[int] = []
    registry.add_listener(lambda: calls.append(1))

    registry.clear_finished()

    assert len(calls) == 1
