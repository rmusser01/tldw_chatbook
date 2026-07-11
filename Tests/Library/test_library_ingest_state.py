"""Pure display-state contracts for the Library ingest canvas (L3b Task 4)."""

from __future__ import annotations

from tldw_chatbook.Library.library_ingest_jobs import IngestJobState, LibraryIngestJob
from tldw_chatbook.Library.library_ingest_state import (
    INGEST_UNAVAILABLE_COPY,
    MEDIA_DB_UNAVAILABLE_COPY,
    SERVER_QUIET_LINE_COPY,
    LibraryIngestFormState,
    build_library_ingest_state,
    clamp_chunk_size,
    parse_keywords,
)


def _job(**overrides) -> LibraryIngestJob:
    defaults = dict(
        job_id="ingest-job-1",
        source_path="/tmp/example.txt",
        state=IngestJobState.QUEUED,
        submitted_at=100.0,
    )
    defaults.update(overrides)
    return LibraryIngestJob(**defaults)


def test_header_is_import_media():
    state = build_library_ingest_state((), form=LibraryIngestFormState())
    assert state.header == "Import media"


def test_queue_heading_is_queue():
    state = build_library_ingest_state((), form=LibraryIngestFormState())
    assert state.queue_heading == "Queue"


def test_server_runtime_shows_quiet_line():
    state = build_library_ingest_state(
        (), form=LibraryIngestFormState(), runtime_source="server"
    )
    assert state.server_quiet_line == SERVER_QUIET_LINE_COPY


def test_local_runtime_hides_quiet_line():
    state = build_library_ingest_state(
        (), form=LibraryIngestFormState(), runtime_source="local"
    )
    assert state.server_quiet_line == ""


def test_media_db_unavailable_blocks_start_with_exact_copy():
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp/a.txt"),
        media_db_available=False,
    )
    assert state.unavailable_line == MEDIA_DB_UNAVAILABLE_COPY
    assert state.start_enabled is False


def test_registry_unavailable_blocks_start_with_exact_copy_and_overrides_db_line():
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp/a.txt"),
        media_db_available=False,
        registry_available=False,
    )
    assert state.unavailable_line == INGEST_UNAVAILABLE_COPY
    assert state.start_enabled is False


def test_available_seams_and_blank_path_disable_start_with_no_blocking_line():
    state = build_library_ingest_state((), form=LibraryIngestFormState(path=""))
    assert state.unavailable_line == ""
    assert state.start_enabled is False


def test_available_seams_and_typed_path_enable_start():
    state = build_library_ingest_state(
        (), form=LibraryIngestFormState(path="/tmp/a.txt")
    )
    assert state.start_enabled is True


# --- start_quiet_line (L3b AB wave, A4) ------------------------------------


def test_start_quiet_line_shown_when_path_blank_and_seams_available():
    state = build_library_ingest_state((), form=LibraryIngestFormState(path=""))
    assert state.start_quiet_line == "Enter a file path to start."


def test_start_quiet_line_hidden_once_path_is_typed():
    state = build_library_ingest_state(
        (), form=LibraryIngestFormState(path="/tmp/a.txt")
    )
    assert state.start_quiet_line == ""


def test_start_quiet_line_shown_for_whitespace_only_path():
    state = build_library_ingest_state((), form=LibraryIngestFormState(path="   "))
    assert state.start_quiet_line == "Enter a file path to start."


def test_start_quiet_line_hidden_when_media_db_unavailable():
    """The db-unavailable line takes precedence -- never show two gate
    lines at once."""
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path=""),
        media_db_available=False,
    )
    assert state.unavailable_line == MEDIA_DB_UNAVAILABLE_COPY
    assert state.start_quiet_line == ""


def test_start_quiet_line_hidden_when_registry_unavailable():
    """The ingest-unavailable line takes precedence -- never show two gate
    lines at once."""
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path=""),
        media_db_available=False,
        registry_available=False,
    )
    assert state.unavailable_line == INGEST_UNAVAILABLE_COPY
    assert state.start_quiet_line == ""


def test_blank_path_with_whitespace_only_disables_start():
    state = build_library_ingest_state(
        (), form=LibraryIngestFormState(path="   ")
    )
    assert state.start_enabled is False


def test_queued_row_line_format():
    jobs = (_job(state=IngestJobState.QUEUED, source_path="/tmp/report.txt"),)
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    row = state.queue_rows[0]
    assert row.glyph == "●"
    assert row.line == "● queued · report.txt"
    assert row.can_open is False
    assert row.can_retry is False
    assert row.job_id == "ingest-job-1"


def test_running_row_line_format_without_detected_type():
    jobs = (
        _job(
            state=IngestJobState.RUNNING,
            source_path="/tmp/report.txt",
            started_at=100.0,
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    row = state.queue_rows[0]
    assert row.glyph == "●"
    assert row.line == "● running · report.txt"


def test_running_row_line_format_with_detected_type():
    jobs = (
        _job(
            state=IngestJobState.RUNNING,
            source_path="/tmp/report.txt",
            started_at=100.0,
            detected_type="plaintext",
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    row = state.queue_rows[0]
    assert row.line == "● running · report.txt · plaintext"


def test_done_row_line_format_seconds_only():
    jobs = (
        _job(
            state=IngestJobState.DONE,
            source_path="/tmp/report.txt",
            started_at=100.0,
            finished_at=107.0,
            media_id=42,
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    row = state.queue_rows[0]
    assert row.glyph == "✓"
    assert row.line == "✓ done · report.txt · 7s"
    assert row.can_open is True
    assert row.can_retry is False
    assert row.media_id == 42


def test_done_row_line_format_minutes_and_seconds():
    jobs = (
        _job(
            state=IngestJobState.DONE,
            source_path="/tmp/report.txt",
            started_at=0.0,
            finished_at=125.0,
            media_id=7,
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    row = state.queue_rows[0]
    assert row.line == "✓ done · report.txt · 2m 5s"


def test_done_row_without_media_id_cannot_open():
    """Defensive: the registry contract always sets media_id on DONE, but
    can_open must never be True without one -- a stray None must not crash
    the Open in Library handler downstream."""
    jobs = (
        _job(
            state=IngestJobState.DONE,
            source_path="/tmp/report.txt",
            started_at=100.0,
            finished_at=101.0,
            media_id=None,
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    assert state.queue_rows[0].can_open is False


def test_failed_row_line_format():
    jobs = (
        _job(
            state=IngestJobState.FAILED,
            source_path="/tmp/report.txt",
            started_at=100.0,
            finished_at=101.0,
            error="File not found",
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    row = state.queue_rows[0]
    assert row.glyph == "✗"
    assert row.line == "✗ failed · report.txt · File not found"
    assert row.can_open is False
    assert row.can_retry is True
    assert row.can_dismiss is True


# --- M4 (fix batch F1b): permanent failures don't offer Retry -------------


def test_permanent_failed_row_cannot_retry_but_can_dismiss():
    jobs = (
        _job(
            state=IngestJobState.FAILED,
            source_path="/tmp/report.xyz",
            started_at=100.0,
            finished_at=101.0,
            error="Unsupported file type: .xyz. Supported types: PDF, TXT",
            permanent=True,
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    row = state.queue_rows[0]
    assert row.can_retry is False
    assert row.can_dismiss is True


def test_non_permanent_failed_row_can_retry():
    jobs = (
        _job(
            state=IngestJobState.FAILED,
            source_path="/tmp/report.txt",
            started_at=100.0,
            finished_at=101.0,
            error="boom",
            permanent=False,
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    row = state.queue_rows[0]
    assert row.can_retry is True
    assert row.can_dismiss is True


# --- L4 (fix batch F1b): short row reason, supported list on the form -----


def test_failed_row_line_drops_supported_types_tail():
    jobs = (
        _job(
            state=IngestJobState.FAILED,
            source_path="/tmp/report.xyz",
            started_at=100.0,
            finished_at=101.0,
            error=(
                "Unsupported file type: .xyz. Supported types: PDF, DOCX, "
                "ODT, RTF, EPUB, MOBI, AZW, FB2, HTML, TXT, MD, MP3, M4A, "
                "WAV, FLAC, OGG, AAC, MP4, AVI, MKV, MOV, WEBM"
            ),
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    row = state.queue_rows[0]
    assert row.line == "✗ failed · report.xyz · Unsupported file type: .xyz."
    assert "Supported types:" not in row.line


def test_failed_row_line_without_marker_passes_through_whole():
    jobs = (
        _job(
            state=IngestJobState.FAILED,
            source_path="/tmp/report.txt",
            started_at=100.0,
            finished_at=101.0,
            error="Media database is unavailable.",
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    row = state.queue_rows[0]
    assert row.line == "✗ failed · report.txt · Media database is unavailable."


def test_supported_types_line_present_and_derived_from_live_registry():
    state = build_library_ingest_state((), form=LibraryIngestFormState())
    assert state.supported_types_line.startswith("Supported: ")
    assert "TXT" in state.supported_types_line
    assert "XML" not in state.supported_types_line


def test_supported_types_line_always_visible_regardless_of_jobs():
    empty_state = build_library_ingest_state((), form=LibraryIngestFormState())
    jobs = (_job(state=IngestJobState.QUEUED),)
    populated_state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    assert empty_state.supported_types_line == populated_state.supported_types_line
    assert populated_state.supported_types_line != ""


def test_basename_used_for_nested_path():
    jobs = (_job(state=IngestJobState.QUEUED, source_path="/a/b/c/deep.md"),)
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    assert state.queue_rows[0].line == "● queued · deep.md"


def test_row_order_mirrors_input_order():
    jobs = (
        _job(job_id="ingest-job-2", state=IngestJobState.QUEUED, source_path="/tmp/b.txt"),
        _job(job_id="ingest-job-1", state=IngestJobState.QUEUED, source_path="/tmp/a.txt"),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    assert [row.job_id for row in state.queue_rows] == ["ingest-job-2", "ingest-job-1"]


def test_queue_counts_line_lists_only_nonzero_states_in_fixed_order():
    """(L3b AB wave, A2) The counts line hides zero-count states entirely --
    segments are just ``{n} {state}`` (no "job"/"jobs" noun), joined by
    ` · `, always in queued -> running -> done -> failed order."""
    jobs = (
        _job(job_id="ingest-job-1", state=IngestJobState.QUEUED),
        _job(job_id="ingest-job-2", state=IngestJobState.RUNNING, started_at=1.0),
        _job(job_id="ingest-job-3", state=IngestJobState.DONE, started_at=1.0, finished_at=2.0, media_id=1),
        _job(job_id="ingest-job-4", state=IngestJobState.DONE, started_at=1.0, finished_at=2.0, media_id=2),
        _job(job_id="ingest-job-5", state=IngestJobState.FAILED, started_at=1.0, finished_at=2.0, error="x"),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    assert state.queue_counts_line == "1 queued · 1 running · 2 done · 1 failed"


def test_queue_counts_line_omits_zero_states():
    jobs = (
        _job(job_id="ingest-job-1", state=IngestJobState.DONE, started_at=1.0, finished_at=2.0, media_id=1),
        _job(job_id="ingest-job-2", state=IngestJobState.DONE, started_at=1.0, finished_at=2.0, media_id=2),
        _job(job_id="ingest-job-3", state=IngestJobState.FAILED, started_at=1.0, finished_at=2.0, error="x"),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    assert state.queue_counts_line == "2 done · 1 failed"


def test_queue_counts_line_hidden_with_no_jobs():
    state = build_library_ingest_state((), form=LibraryIngestFormState())
    assert state.queue_counts_line == ""


def test_empty_queue_has_no_rows():
    state = build_library_ingest_state((), form=LibraryIngestFormState())
    assert state.queue_rows == ()


# --- queue_show_clear_finished (L3b AB wave, B2) ---------------------------


def test_show_clear_finished_false_with_no_jobs():
    state = build_library_ingest_state((), form=LibraryIngestFormState())
    assert state.queue_show_clear_finished is False


def test_show_clear_finished_false_with_only_active_jobs():
    jobs = (
        _job(job_id="ingest-job-1", state=IngestJobState.QUEUED),
        _job(job_id="ingest-job-2", state=IngestJobState.RUNNING, started_at=1.0),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    assert state.queue_show_clear_finished is False


def test_show_clear_finished_true_with_a_done_job():
    jobs = (
        _job(state=IngestJobState.DONE, started_at=1.0, finished_at=2.0, media_id=1),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    assert state.queue_show_clear_finished is True


def test_show_clear_finished_true_with_a_failed_job():
    jobs = (
        _job(state=IngestJobState.FAILED, started_at=1.0, finished_at=2.0, error="x"),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    assert state.queue_show_clear_finished is True


def test_form_state_echoed_back_unchanged():
    form = LibraryIngestFormState(
        path="/tmp/a.txt",
        title="My title",
        author="An author",
        keywords="alpha, beta",
        analyze=True,
        chunk=True,
        chunk_size="750",
    )
    state = build_library_ingest_state((), form=form)
    assert state.form is form
    assert state.form.title == "My title"
    assert state.form.analyze is True


def test_done_row_missing_finished_at_falls_back_to_now():
    """Defensive: a malformed job missing finished_at still renders a sane
    elapsed value (via the ``now`` fallback) instead of crashing."""
    jobs = (
        _job(
            state=IngestJobState.DONE,
            source_path="/tmp/report.txt",
            started_at=100.0,
            finished_at=None,
            media_id=1,
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState(), now=110.0)
    assert state.queue_rows[0].line == "✓ done · report.txt · 10s"


# --- parse_keywords -----------------------------------------------------


def test_parse_keywords_splits_strips_and_drops_empties():
    assert parse_keywords("alpha, beta ,  , gamma") == ("alpha", "beta", "gamma")


def test_parse_keywords_empty_string_returns_empty_tuple():
    assert parse_keywords("") == ()
    assert parse_keywords("   ") == ()


# --- clamp_chunk_size -----------------------------------------------------


def test_clamp_chunk_size_within_range_unchanged():
    assert clamp_chunk_size("750") == 750


def test_clamp_chunk_size_clamps_below_minimum():
    assert clamp_chunk_size("10") == 100


def test_clamp_chunk_size_clamps_above_maximum():
    assert clamp_chunk_size("99999") == 5000


def test_clamp_chunk_size_defaults_on_garbage_input():
    assert clamp_chunk_size("not a number") == 500
    assert clamp_chunk_size("") == 500
