"""Pure display-state contracts for the Library ingest canvas (L3b Task 4)."""

from __future__ import annotations

from tldw_chatbook.Library.ingest_types import PreflightResult
from tldw_chatbook.Library.library_ingest_jobs import IngestJobState, LibraryIngestJob
from tldw_chatbook.Library.library_ingest_state import (
    INGEST_UNAVAILABLE_COPY,
    MEDIA_DB_UNAVAILABLE_COPY,
    SERVER_QUIET_LINE_COPY,
    LibraryIngestFormState,
    _human_size,
    build_estimate_line,
    build_library_ingest_state,
    build_type_breakdown_line,
    build_warning_lines,
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


def test_parsing_row_line_format_without_detected_type():
    jobs = (
        _job(
            state=IngestJobState.PARSING,
            source_path="/tmp/report.txt",
            started_at=100.0,
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    row = state.queue_rows[0]
    assert row.glyph == "●"
    assert row.line == "● parsing · report.txt"


def test_parsing_row_line_format_with_detected_type():
    jobs = (
        _job(
            state=IngestJobState.PARSING,
            source_path="/tmp/report.txt",
            started_at=100.0,
            detected_type="plaintext",
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    row = state.queue_rows[0]
    assert row.line == "● parsing · report.txt · plaintext"


def test_writing_row_line_format_without_detected_type():
    jobs = (
        _job(
            state=IngestJobState.WRITING,
            source_path="/tmp/report.txt",
            started_at=100.0,
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    row = state.queue_rows[0]
    assert row.glyph == "●"
    assert row.line == "● writing · report.txt"


def test_writing_row_line_format_with_detected_type():
    jobs = (
        _job(
            state=IngestJobState.WRITING,
            source_path="/tmp/report.txt",
            started_at=100.0,
            detected_type="plaintext",
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    row = state.queue_rows[0]
    assert row.line == "● writing · report.txt · plaintext"


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


def test_unsupported_file_type_category_cannot_retry():
    """A failed job whose error_detail category is unsupported_file_type
    cannot be retried even when ``permanent`` is False."""
    jobs = (
        _job(
            state=IngestJobState.FAILED,
            source_path="/tmp/report.xyz",
            started_at=100.0,
            finished_at=101.0,
            error="Unsupported file type",
            permanent=False,
            error_detail={
                "category": "unsupported_file_type",
                "message": "Unsupported extension",
            },
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    row = state.queue_rows[0]
    assert row.can_retry is False
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


def test_failed_row_line_appends_retry_suffix():
    """(Task 3, backlog 161) Mirrors Home's status_detail retry suffix --
    single source of truth per short_ingest_error's docstring."""
    jobs = (
        _job(
            state=IngestJobState.FAILED,
            source_path="/tmp/report.txt",
            started_at=100.0,
            finished_at=101.0,
            error="bad codec",
            retry_count=2,
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    row = state.queue_rows[0]
    assert row.line == "✗ failed · report.txt · bad codec · retry 2"


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
    """(L3b AB wave, A2; F3 re-anchor) The counts line hides zero-count
    states entirely -- segments are just ``{n} {state}`` (no "job"/"jobs"
    noun), joined by ` · `, always in parsing -> writing -> queued -> done
    -> failed order (the in-flight/"hot" stages first, per the F3 design
    spec's UI-impact example)."""
    jobs = (
        _job(job_id="ingest-job-1", state=IngestJobState.QUEUED),
        _job(job_id="ingest-job-2", state=IngestJobState.PARSING, started_at=1.0),
        _job(job_id="ingest-job-6", state=IngestJobState.WRITING, started_at=1.0),
        _job(job_id="ingest-job-3", state=IngestJobState.DONE, started_at=1.0, finished_at=2.0, media_id=1),
        _job(job_id="ingest-job-4", state=IngestJobState.DONE, started_at=1.0, finished_at=2.0, media_id=2),
        _job(job_id="ingest-job-5", state=IngestJobState.FAILED, started_at=1.0, finished_at=2.0, error="x"),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    assert state.queue_counts_line == "1 parsing · 1 writing · 1 queued · 2 done · 1 failed"


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
        _job(job_id="ingest-job-2", state=IngestJobState.PARSING, started_at=1.0),
        _job(job_id="ingest-job-3", state=IngestJobState.WRITING, started_at=1.0),
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


# --- Task 7: extended form state, pre-flight summary, recent jobs ----------


def test_form_state_has_new_preflight_fields():
    preflight = PreflightResult(
        type_groups={"pdf": ["/tmp/a.pdf"]},
        warnings=[],
        errors=[],
        total_size=1024,
        truncated=False,
        total_files=1,
    )
    form = LibraryIngestFormState(
        expanded_type_groups={"pdf"},
        type_options={"pdf": {"ocr": True}},
        preflight=preflight,
        preflight_checking=True,
    )
    assert form.expanded_type_groups == {"pdf"}
    assert form.type_options == {"pdf": {"ocr": True}}
    assert form.preflight is preflight
    assert form.preflight_checking is True


def test_form_state_defaults_are_sensible():
    form = LibraryIngestFormState()
    assert form.expanded_type_groups == set()
    assert form.type_options == {}
    assert form.preflight is None
    assert form.preflight_checking is False


# --- build_type_breakdown_line ---------------------------------------------


def test_build_type_breakdown_line_empty():
    assert build_type_breakdown_line({}) == ""


def test_build_type_breakdown_line_single_group_single_file():
    line = build_type_breakdown_line({"pdf": ["/tmp/a.pdf"]})
    assert line == "1 PDF document"


def test_build_type_breakdown_line_multiple_groups_and_counts():
    line = build_type_breakdown_line(
        {
            "pdf": ["/tmp/a.pdf", "/tmp/b.pdf"],
            "audio_video": ["/tmp/c.mp3"],
            "generic": ["/tmp/d.txt", "/tmp/e.txt", "/tmp/f.txt"],
        }
    )
    assert line == "2 PDF documents, 1 audio/video file, 3 plain text files"


def test_build_type_breakdown_line_unknown_group_uses_key():
    line = build_type_breakdown_line({"weird": ["/tmp/x.foo"]})
    assert line == "1 weird"


# --- build_estimate_line ---------------------------------------------------


def test_build_estimate_line_zero_files():
    assert build_estimate_line(0, 0, False) == "0 files"


def test_build_estimate_line_single_file_bytes():
    assert build_estimate_line(1, 512, False) == "1 file · 512 B"


def test_build_estimate_line_multiple_files_human_size():
    assert build_estimate_line(5, 1536, False) == "5 files · 1.5 KB"


def test_build_estimate_line_appends_truncated_note():
    line = build_estimate_line(1000, 1024 * 1024, True)
    assert line.startswith("1000 files · 1.0 MB")
    assert "more files not shown" in line


# --- build_warning_lines ---------------------------------------------------


def test_build_warning_lines_empty():
    assert build_warning_lines([]) == []


def test_build_warning_lines_label_and_hint():
    warnings = [{"label": "PDF processing", "hint": "PyMuPDF is not installed."}]
    assert build_warning_lines(warnings) == ["PDF processing: PyMuPDF is not installed."]


def test_build_warning_lines_falls_back_to_hint_only():
    warnings = [{"hint": "Something is missing."}]
    assert build_warning_lines(warnings) == ["Something is missing."]


def test_build_warning_lines_label_only():
    warnings = [{"label": "PDF processing"}]
    assert build_warning_lines(warnings) == ["PDF processing"]


def test_build_warning_lines_empty_dict():
    warnings = [{}]
    assert build_warning_lines(warnings) == ["{}"]


def test_build_warning_lines_ignores_command_key():
    warnings = [{"label": "PDF", "hint": "missing", "command": "pip install x"}]
    assert build_warning_lines(warnings) == ["PDF: missing"]


# --- _human_size -----------------------------------------------------------


def test_human_size_tb_midrange():
    assert _human_size(1024 ** 4 * 512) == "512.0 TB"


def test_human_size_pb_boundary():
    # 1024**5 bytes == 1 PB; the original bug reported the value in TB while
    # labeling it PB. After the fix it must read "1.0 PB".
    assert _human_size(1024 ** 5) == "1.0 PB"
    assert _human_size(1024 ** 6) == "1024.0 PB"


# --- canvas state pre-flight fields ----------------------------------------


def test_canvas_state_preflight_fields_populated_from_parameter():
    preflight = PreflightResult(
        type_groups={"pdf": ["/tmp/a.pdf", "/tmp/b.pdf"]},
        warnings=[{"label": "PDF", "hint": "missing"}],
        errors=["Path not found"],
        total_size=2048,
        truncated=False,
        total_files=2,
    )
    state = build_library_ingest_state((), form=LibraryIngestFormState(), preflight=preflight)
    assert state.type_breakdown_line == "2 PDF documents"
    assert state.estimate_line == "2 files · 2.0 KB"
    assert state.warning_lines == ["PDF: missing"]
    assert state.errors == ["Path not found"]
    assert state.type_groups == ["pdf"]
    assert state.unsupported_files == []
    assert state.preflight_checking is False


def test_canvas_state_preflight_fields_fallback_to_form():
    preflight = PreflightResult(
        type_groups={"generic": ["/tmp/a.txt"]},
        warnings=[],
        errors=[],
        total_size=100,
        truncated=False,
        total_files=1,
    )
    form = LibraryIngestFormState(preflight=preflight, preflight_checking=True)
    state = build_library_ingest_state((), form=form)
    assert state.type_breakdown_line == "1 plain text file"
    assert state.preflight_checking is True


def test_canvas_state_preflight_parameter_overrides_form():
    form_preflight = PreflightResult(
        type_groups={"generic": ["/tmp/form.txt"]},
        warnings=[],
        errors=[],
        total_size=100,
        truncated=False,
        total_files=1,
    )
    param_preflight = PreflightResult(
        type_groups={"pdf": ["/tmp/param.pdf"]},
        warnings=[],
        errors=[],
        total_size=200,
        truncated=False,
        total_files=1,
    )
    form = LibraryIngestFormState(preflight=form_preflight)
    state = build_library_ingest_state(
        (), form=form, preflight=param_preflight
    )
    assert state.type_breakdown_line == "1 PDF document"


def test_canvas_state_preflight_checking_parameter_overrides_form():
    form = LibraryIngestFormState(preflight_checking=True)
    state = build_library_ingest_state(
        (), form=form, preflight_checking=False
    )
    # Explicit ``False`` parameter wins over form flag.
    assert state.preflight_checking is False


def test_canvas_state_separates_unsupported_files():
    preflight = PreflightResult(
        type_groups={
            "pdf": ["/tmp/a.pdf"],
            "unsupported": ["/tmp/b.xyz", "/tmp/c.abc"],
        },
        warnings=[],
        errors=[],
        total_size=0,
        truncated=False,
        total_files=3,
    )
    state = build_library_ingest_state((), form=LibraryIngestFormState(), preflight=preflight)
    assert state.type_groups == ["pdf"]
    assert state.type_breakdown_line == "1 PDF document"
    assert state.unsupported_files == ["/tmp/b.xyz", "/tmp/c.abc"]


def test_canvas_state_preflight_none_gives_empty_summary():
    state = build_library_ingest_state((), form=LibraryIngestFormState())
    assert state.type_breakdown_line == ""
    assert state.estimate_line == ""
    assert state.warning_lines == []
    assert state.errors == []
    assert state.type_groups == []
    assert state.unsupported_files == []


def test_canvas_state_expanded_type_groups_copied_from_form():
    form = LibraryIngestFormState(expanded_type_groups={"audio_video", "ebook"})
    state = build_library_ingest_state((), form=form)
    assert state.expanded_type_groups == {"audio_video", "ebook"}


# --- recent_jobs -----------------------------------------------------------


def test_recent_jobs_includes_done_and_failed():
    done = _job(job_id="ingest-job-1", state=IngestJobState.DONE, started_at=1.0, finished_at=2.0, media_id=1)
    failed = _job(job_id="ingest-job-2", state=IngestJobState.FAILED, started_at=1.0, finished_at=2.0, error="boom")
    queued = _job(job_id="ingest-job-3", state=IngestJobState.QUEUED)
    state = build_library_ingest_state(
        (done, failed, queued), form=LibraryIngestFormState()
    )
    assert [j.job_id for j in state.recent_jobs] == ["ingest-job-1", "ingest-job-2"]


def test_recent_jobs_limits_to_ten():
    jobs = tuple(
        _job(
            job_id=f"ingest-job-{i}",
            state=IngestJobState.DONE,
            started_at=1.0,
            finished_at=2.0,
            media_id=i,
        )
        for i in range(15)
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    assert len(state.recent_jobs) == 10
