"""Recovery copy for Library Import failures (critique #8 row 5, task-32054).

A failed row leaked a raw errno the user could not act on, offered no way
to see the underlying text, restated a "will skip" file as a failure, and a
six-file batch stacked six completion toasts.
"""

from __future__ import annotations

import errno
from unittest.mock import MagicMock

from tldw_chatbook.Library.library_ingest_jobs import IngestJobState, LibraryIngestJob
from tldw_chatbook.Library.library_ingest_state import (
    IngestFailureCopy,
    LibraryIngestFormState,
    build_library_ingest_state,
    map_ingest_failure,
)

_POOL_ERRNO_28 = "Parse pool could not start: [Errno 28] No space left on device"

_RESOURCE_LIMIT_SUMMARY = (
    "The import worker couldn't start on this machine (system resource limit)"
)
_RESOURCE_LIMIT_NEXT_STEP = "Restart the app, then Retry"


def _job(**overrides) -> LibraryIngestJob:
    defaults = dict(
        job_id="ingest-job-1",
        source_path="/tmp/example.txt",
        state=IngestJobState.QUEUED,
        submitted_at=100.0,
    )
    defaults.update(overrides)
    return LibraryIngestJob(**defaults)


def _row(**overrides):
    return build_library_ingest_state(
        (_job(**overrides),), form=LibraryIngestFormState()
    ).queue_rows[0]


# ---------------------------------------------------------------------------
# task-32054 AC#1/#3: map_ingest_failure
# ---------------------------------------------------------------------------


def test_pool_start_errno_28_maps_to_plain_language_with_a_next_step() -> None:
    """The host's semaphore exhaustion reads as a resource limit, not a disk."""
    copy = map_ingest_failure(_POOL_ERRNO_28)

    assert isinstance(copy, IngestFailureCopy)
    assert copy.summary == _RESOURCE_LIMIT_SUMMARY
    assert copy.next_step == _RESOURCE_LIMIT_NEXT_STEP
    assert copy.retryable is True
    # The raw text survives -- but only for "Show details".
    assert copy.detail == _POOL_ERRNO_28
    assert "Errno" not in copy.summary
    assert "Errno" not in copy.next_step


def test_pool_start_oserror_instance_maps_the_same_way() -> None:
    """The mapper takes the exception the pool actually raised, too."""
    copy = map_ingest_failure(
        OSError(errno.ENOSPC, "No space left on device"), context="pool_start"
    )

    assert copy.summary == _RESOURCE_LIMIT_SUMMARY
    assert copy.next_step == _RESOURCE_LIMIT_NEXT_STEP


def test_unsupported_file_keeps_its_own_reason_and_is_not_retryable() -> None:
    """An unsupported file is a skip, not a retryable failure."""
    copy = map_ingest_failure(
        "Unsupported file type: .xyz Supported types: .pdf, .txt"
    )

    assert copy.summary == "Unsupported file type: .xyz"
    assert copy.retryable is False
    assert copy.next_step == ""


def test_plain_database_error_passes_through_as_a_retryable_failure() -> None:
    """An ordinary failure keeps the reason the pipeline already wrote."""
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDBError as DatabaseError

    copy = map_ingest_failure(DatabaseError("database is locked"))

    assert copy.summary == "database is locked"
    assert copy.next_step == ""
    assert copy.retryable is True
    assert copy.detail == "database is locked"


# ---------------------------------------------------------------------------
# task-32054 AC#1/#2/#5: the queue row consumes the mapper
# ---------------------------------------------------------------------------


def test_failed_row_shows_the_mapped_summary_and_next_step() -> None:
    row = _row(
        state=IngestJobState.FAILED,
        source_path="/tmp/reading-notes.md",
        error=_POOL_ERRNO_28,
    )

    assert row.line == (
        f"✗ failed · reading-notes.md · {_RESOURCE_LIMIT_SUMMARY} · "
        f"{_RESOURCE_LIMIT_NEXT_STEP}"
    )
    assert "Errno" not in row.line


def test_every_failed_row_offers_show_details_with_the_raw_text() -> None:
    """AC#2: the underlying error is one press away even without error_detail."""
    row = _row(
        state=IngestJobState.FAILED,
        source_path="/tmp/reading-notes.md",
        error=_POOL_ERRNO_28,
    )
    assert row.can_show_details is True

    expanded = build_library_ingest_state(
        (
            _job(
                state=IngestJobState.FAILED,
                source_path="/tmp/reading-notes.md",
                error=_POOL_ERRNO_28,
            ),
        ),
        form=LibraryIngestFormState(),
        expanded_details={"ingest-job-1"},
    ).queue_rows[0]

    assert expanded.details_expanded is True
    assert any("[Errno 28]" in line for line in expanded.detail_lines)


def test_skipped_row_never_offers_retry_and_keeps_its_own_reason() -> None:
    """AC#3: an unsupported file stays a skip with its own copy."""
    row = _row(
        state=IngestJobState.SKIPPED,
        source_path="/tmp/photo.heic",
        error="Unsupported file type: .heic Supported types: .pdf, .txt",
    )

    assert row.line == "○ skipped · photo.heic · Unsupported file type: .heic"
    assert row.can_retry is False


def test_retry_suffix_reads_retry_not_attempt() -> None:
    """AC#5: the row copy matches the user guide's ' · retry 1'."""
    row = _row(
        state=IngestJobState.FAILED,
        source_path="/tmp/report.txt",
        error="bad codec",
        retry_count=1,
    )
    assert row.line.endswith("· retry 1")


# ---------------------------------------------------------------------------
# task-32054 AC#4: one completion toast per batch
# ---------------------------------------------------------------------------


def _settle_screen():
    """A minimal Library screen wired for the registry-changed listener."""
    from Tests.UI.test_library_ingest_inline_consent import _minimal_library_screen

    screen = _minimal_library_screen()
    screen._is_mounted = True
    screen._library_landing_attention_action = MagicMock(return_value=None)
    screen._library_landing_attention_signature = None
    screen._refresh_local_source_snapshot = MagicMock()
    return screen


def test_a_batch_of_instant_failures_produces_one_completion_toast(tmp_path) -> None:
    """AC#4: six files submitted in one loop settle into ONE toast.

    The host's parse pool fails at creation, so every file fails
    synchronously inside the folder-submission loop -- the active count
    crosses N -> 0 once per file, which used to fire one toast each.
    """
    screen = _settle_screen()
    toasts: list[str] = []
    screen.app_instance.notify = lambda message, **_kwargs: toasts.append(
        str(message)
    )
    registry = screen.app_instance.library_ingest_jobs
    listener = screen._ingest_controller._handle_library_ingest_registry_changed

    for index in range(6):
        job = registry.submit(source_path=f"/tmp/note-{index}.txt")
        listener()
        registry.mark_failed(job.job_id, error=_POOL_ERRNO_28, permanent=False)
        listener()

    # The settle is deferred by one turn of the event loop; run what the
    # controller scheduled.
    for call in screen.call_after_refresh.call_args_list:
        call.args[0]()

    assert toasts == ["Import finished — 6 failed"], toasts


def test_a_settled_single_import_still_reports_once(tmp_path) -> None:
    """The deferral must not lose the ordinary one-file completion toast."""
    screen = _settle_screen()
    toasts: list[str] = []
    screen.app_instance.notify = lambda message, **_kwargs: toasts.append(
        str(message)
    )
    registry = screen.app_instance.library_ingest_jobs
    listener = screen._ingest_controller._handle_library_ingest_registry_changed

    job = registry.submit(source_path="/tmp/note.txt")
    listener()
    registry.mark_failed(job.job_id, error="bad codec", permanent=False)
    listener()
    for call in screen.call_after_refresh.call_args_list:
        call.args[0]()

    assert toasts == ["Import finished — 1 failed"], toasts
