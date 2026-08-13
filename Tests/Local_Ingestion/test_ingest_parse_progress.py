"""Tests for spawn-safe local-ingest parse progress telemetry."""

import queue

from tldw_chatbook.Local_Ingestion.ingest_parse_progress import (
    ParseProgressEvent,
    ParseProgressCoalescer,
    emit_parse_progress,
    install_parse_progress_sink,
    make_parse_progress_event,
)


def test_progress_event_is_bounded_plain_data_and_invalid_percent_is_omitted():
    """Reject unsafe percentages while normalizing emitted progress text."""
    event = make_parse_progress_event(
        generation=4,
        job_id="ingest-job-7",
        phase="extracting",
        message="Extracting page 2\nof 5\x00",
        percent=float("inf"),
    )

    assert event == ParseProgressEvent(
        generation=4,
        job_id="ingest-job-7",
        phase="extracting",
        message="Extracting page 2 of 5",
        percent=None,
    )
    assert make_parse_progress_event(
        generation=4,
        job_id="ingest-job-7",
        phase="provider-private-stage",
        message="raw",
    ) is None


def test_progress_event_rejects_oversized_ipc_identities():
    """Keep job and generation identities within their IPC-safe bounds."""
    assert make_parse_progress_event(
        generation=-1,
        job_id="ingest-job-7",
        phase="extracting",
        message="Extracting",
    ) is None
    assert make_parse_progress_event(
        generation=2**63,
        job_id="ingest-job-7",
        phase="extracting",
        message="Extracting",
    ) is None
    assert make_parse_progress_event(
        generation=4,
        job_id="a" * 65,
        phase="extracting",
        message="Extracting",
    ) is None


class _FullQueue:
    def put_nowait(self, _event):
        raise queue.Full


def test_full_progress_queue_is_best_effort():
    """Do not let a saturated worker IPC queue stop ingestion progress."""
    install_parse_progress_sink(_FullQueue())

    emit_parse_progress(1, "ingest-job-1", "extracting", "Extracting")


def test_coalescer_keeps_latest_event_per_job_until_due():
    """Flush the newest event for each job at the supplied deadline."""
    coalescer = ParseProgressCoalescer(interval=0.25, started_at=10.0)
    coalescer.accept(ParseProgressEvent(1, "a", "extracting", "first", 10.0))
    coalescer.accept(ParseProgressEvent(1, "a", "extracting", "latest", 30.0))

    assert coalescer.take_due(10.24) == ()
    assert coalescer.take_due(10.25) == (
        ParseProgressEvent(1, "a", "extracting", "latest", 30.0),
    )


def test_coalescer_force_flushes_pending_event_before_deadline():
    """Flush pending telemetry without waiting for the interval."""
    coalescer = ParseProgressCoalescer(interval=60.0, started_at=10.0)
    event = ParseProgressEvent(1, "a", "extracting", "latest", 30.0)
    coalescer.accept(event)

    assert coalescer.take_due(10.01, force=True) == (event,)
    assert coalescer.take_due(10.01, force=True) == ()
