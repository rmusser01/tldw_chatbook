"""Pure display-state contracts for the Library ingest canvas (L3b Task 4)."""

from __future__ import annotations

import re

from tldw_chatbook.Library.ingest_types import PreflightResult
from tldw_chatbook.Library.library_ingest_jobs import IngestJobState, LibraryIngestJob
from tldw_chatbook.Library.library_ingest_state import (
    INGEST_UNAVAILABLE_COPY,
    MEDIA_DB_UNAVAILABLE_COPY,
    LibraryIngestFormState,
    _human_size,
    build_estimate_line,
    build_library_ingest_state,
    build_type_breakdown_line,
    build_warning_lines,
    clamp_chunk_size,
    parse_keywords,
    short_ingest_error,
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


def test_browse_scope_no_longer_drives_the_ingest_line():
    """Browsing server-side says nothing about where an import will run.

    This test previously asserted the old "ingest runs on Local" warning, which
    existed because ingest ignored the browse scope. Ingest now has its own
    explicit target and never keys off browse scope, so on a local-only install
    the line is silent rather than warning about a coupling that no longer
    exists (task-684.1).
    """
    state = build_library_ingest_state(
        (), form=LibraryIngestFormState(), runtime_source="server"
    )
    assert state.server_quiet_line == ""
    assert state.ingest_backend == "local"


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
    state = build_library_ingest_state((), form=LibraryIngestFormState(path="   "))
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
    # (task-2015) Elapsed measures from submission; the fixture's timeline is
    # submitted -> started -> finished, 125s of user-perceived wait.
    jobs = (
        _job(
            state=IngestJobState.DONE,
            source_path="/tmp/report.txt",
            submitted_at=100.0,
            started_at=110.0,
            finished_at=225.0,
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
        _job(
            job_id="ingest-job-2", state=IngestJobState.QUEUED, source_path="/tmp/b.txt"
        ),
        _job(
            job_id="ingest-job-1", state=IngestJobState.QUEUED, source_path="/tmp/a.txt"
        ),
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
        _job(
            job_id="ingest-job-3",
            state=IngestJobState.DONE,
            started_at=1.0,
            finished_at=2.0,
            media_id=1,
        ),
        _job(
            job_id="ingest-job-4",
            state=IngestJobState.DONE,
            started_at=1.0,
            finished_at=2.0,
            media_id=2,
        ),
        _job(
            job_id="ingest-job-5",
            state=IngestJobState.FAILED,
            started_at=1.0,
            finished_at=2.0,
            error="x",
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    assert (
        state.queue_counts_line
        == "1 parsing · 1 writing · 1 queued · 2 done · 1 failed — all ingests"
    )


def test_queue_counts_line_omits_zero_states():
    jobs = (
        _job(
            job_id="ingest-job-1",
            state=IngestJobState.DONE,
            started_at=1.0,
            finished_at=2.0,
            media_id=1,
        ),
        _job(
            job_id="ingest-job-2",
            state=IngestJobState.DONE,
            started_at=1.0,
            finished_at=2.0,
            media_id=2,
        ),
        _job(
            job_id="ingest-job-3",
            state=IngestJobState.FAILED,
            started_at=1.0,
            finished_at=2.0,
            error="x",
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    # (task-2043) The suffix says the totals span ALL ingests (the
    # registry restores prior sessions from the jobs DB).
    assert state.queue_counts_line == "2 done · 1 failed — all ingests"


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
    assert build_warning_lines(warnings) == [
        "PDF processing isn't installed — needed for PyMuPDF is not installed."
    ]


def test_build_warning_lines_falls_back_to_hint_only():
    warnings = [{"hint": "Something is missing."}]
    assert build_warning_lines(warnings) == ["Something is missing."]


def test_build_warning_lines_names_the_gap_and_the_fix():
    """The composed line says what is missing, why it matters, and the fix.

    The old shape pasted the label in front of a hint that already repeated
    it, producing "PDF processing: PDF processing is unavailable: PDF
    ingestion." (task-666).
    """
    warnings = [
        {
            "label": "PDF processing",
            "hint": "PDF ingestion",
            "command": 'pip install -e ".[pdf]"',
        }
    ]
    assert build_warning_lines(warnings) == [
        "PDF processing isn't installed — needed for PDF ingestion. "
        'Install it with: pip install -e ".[pdf]"'
    ]


def test_build_warning_lines_does_not_repeat_the_label():
    """When the label and the capability are the same words, say them once."""
    warnings = [{"label": "Audio processing", "hint": "Audio processing"}]
    assert build_warning_lines(warnings) == ["Audio processing isn't installed."]


def test_build_warning_lines_label_only():
    warnings = [{"label": "PDF processing"}]
    assert build_warning_lines(warnings) == ["PDF processing isn't installed."]


def test_build_warning_lines_empty_dict():
    warnings = [{}]
    assert build_warning_lines(warnings) == ["{}"]


def test_build_warning_lines_includes_the_install_command():
    """The command is the actionable half; it belongs in the line."""
    warnings = [{"label": "PDF", "hint": "missing", "command": "pip install x"}]
    assert build_warning_lines(warnings) == [
        "PDF isn't installed — needed for missing. Install it with: pip install x"
    ]


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
    # (task-2015) With errors present, the breakdown/estimate are suppressed
    # -- an estimate parked under an error is noise. Warnings still render.
    assert state.type_breakdown_line == ""
    assert state.estimate_line == ""
    assert state.warning_lines == ["PDF isn't installed — needed for missing."]
    assert state.errors == ["Path not found"]
    assert state.type_groups == ["pdf", "generic"]
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
    assert state.type_groups == ["pdf", "generic"]
    assert state.type_breakdown_line == "1 PDF document"
    assert state.unsupported_files == ["/tmp/b.xyz", "/tmp/c.abc"]


def test_canvas_state_preflight_none_gives_empty_summary():
    state = build_library_ingest_state((), form=LibraryIngestFormState())
    assert state.type_breakdown_line == ""
    assert state.estimate_line == ""
    assert state.warning_lines == []
    assert state.errors == []
    assert state.type_groups == ["generic"]
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


def test_form_defaults_come_from_the_capability_declaration() -> None:
    """One declaration of defaults, not two that disagree.

    The capability layer declared ``analyze=True, chunk_size=1000`` while the
    form shipped ``analyze=False, chunk=False, chunk_size=500``, and the two
    ingest surfaces disagreed about chunking on top of that. Whichever was
    authoritative, they could not both be (task-667).
    """
    from tldw_chatbook.Library.ingest_capabilities import get_capabilities

    generic = {f.name: f.default for f in get_capabilities("generic").fields}
    form = LibraryIngestFormState()

    assert form.analyze is generic["analyze"]
    assert form.chunk is generic["chunk"]
    assert form.chunk_size == str(generic["chunk_size"])


def test_chunking_is_on_by_default_and_analysis_is_off() -> None:
    """Imports are chunked for retrieval; nothing calls an LLM unasked.

    Chunking off by default meant imported documents were never chunked for
    retrieval, quietly undermining search and RAG for anyone who never opened
    the advanced panel. Analysis stays off because it costs an LLM call per
    document at ingest time.
    """
    form = LibraryIngestFormState()

    assert form.chunk is True
    assert form.chunk_size == "1000"
    assert form.analyze is False


# --- queue row origin (task-684.2) ------------------------------------------


def _row_for(job: LibraryIngestJob):
    from tldw_chatbook.Library.library_ingest_state import _build_queue_row

    return _build_queue_row(job, now=100.0)


def test_queue_row_marks_a_server_job():
    """A row has to say where the job runs, or two backends look identical.

    Once local and server ingests share one queue, "done · notes.txt" alone
    cannot tell the user which machine did the work (task-684.2).
    """
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/notes.txt",
        state=IngestJobState.QUEUED,
        origin="server",
    )

    row = _row_for(job)

    assert row.origin == "server"
    assert "server" in row.line.lower()


def test_queue_row_does_not_clutter_a_local_job():
    """Local is the overwhelmingly common case and stays unannotated."""
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/notes.txt",
        state=IngestJobState.QUEUED,
    )

    row = _row_for(job)

    assert row.origin == "local"
    assert "server" not in row.line.lower()
    assert row.line == "● queued · notes.txt"


def test_server_origin_is_marked_in_every_state():
    """The marker cannot depend on which state branch built the row."""
    for state in (
        IngestJobState.QUEUED,
        IngestJobState.PARSING,
        IngestJobState.WRITING,
        IngestJobState.DONE,
        IngestJobState.FAILED,
    ):
        job = LibraryIngestJob(
            job_id="ingest-job-1",
            source_path="/tmp/notes.txt",
            state=state,
            origin="server",
            media_id=1 if state == IngestJobState.DONE else None,
            error="boom" if state == IngestJobState.FAILED else "",
        )
        row = _row_for(job)
        assert "server" in row.line.lower(), f"{state.value} row lost the marker"
        assert row.origin == "server"


# --- cancelled row rendering (task-684.2) -----------------------------------


def test_cancelled_row_reads_as_stopped_not_failed():
    """A cancellation must not wear the failure glyph.

    The user stopped this deliberately; showing ✗ would read as an error they
    caused, and would sit beside a Retry the registry refuses anyway.
    """
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/talk.mp3",
        state=IngestJobState.CANCELLED,
        origin="server",
        error="Cancelled on the server.",
        finished_at=50.0,
    )

    row = _row_for(job)

    assert row.glyph not in {"✓", "✗"}
    assert "cancelled" in row.line.lower()
    assert "talk.mp3" in row.line
    assert row.can_open is False
    assert row.can_retry is False, "requeue is FAILED-only; Retry would be dead bait"
    assert row.can_dismiss is True, "a cancelled row is clearable"


def test_cancelled_counts_line_segment_is_rendered():
    """A cancelled job has to appear in the queue's per-state summary."""
    from tldw_chatbook.Library.library_ingest_state import _queue_counts_line

    jobs = [
        LibraryIngestJob(
            job_id="ingest-job-1",
            source_path="/tmp/a.mp3",
            state=IngestJobState.CANCELLED,
        ),
        LibraryIngestJob(
            job_id="ingest-job-2",
            source_path="/tmp/b.txt",
            state=IngestJobState.DONE,
            media_id=1,
        ),
    ]

    line = _queue_counts_line(jobs)

    assert "1 cancelled" in line
    assert "1 done" in line


# --- cancel affordance (task-684.2) -----------------------------------------


def test_a_running_server_job_can_be_cancelled():
    """Only a server job offers Cancel: the local pipeline has no cancel seam.

    ``cancel_media_ingest_jobs_batch`` exists on the server service; there is no
    local equivalent, so offering Cancel on a local job would be dead bait.
    """
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/a.mp3",
        state=IngestJobState.PARSING,
        origin="server",
        batch_id="batch-1",
        remote_job_id="11",
    )

    assert _row_for(job).can_cancel is True


def test_a_running_local_job_offers_no_cancel():
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/a.txt",
        state=IngestJobState.PARSING,
    )

    assert _row_for(job).can_cancel is False


def test_a_settled_server_job_offers_no_cancel():
    """There is nothing left to stop once the server has finished."""
    for state in (
        IngestJobState.DONE,
        IngestJobState.FAILED,
        IngestJobState.CANCELLED,
    ):
        job = LibraryIngestJob(
            job_id="ingest-job-1",
            source_path="/tmp/a.mp3",
            state=state,
            origin="server",
            batch_id="batch-1",
            remote_job_id="11",
            error="boom" if state is IngestJobState.FAILED else "",
        )
        assert _row_for(job).can_cancel is False, f"{state.value} offered Cancel"


def test_a_server_job_without_a_batch_offers_no_cancel():
    """Cancel goes by batch id; without one there is nothing to address."""
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/a.mp3",
        state=IngestJobState.QUEUED,
        origin="server",
    )

    assert _row_for(job).can_cancel is False


# --- backend target + switch (task-684.1 slice 3) ---------------------------


def test_local_only_install_says_nothing_about_backends():
    """With no server configured there is no choice to explain.

    The old "ingest runs on Local" line existed to warn that ingest ignored your
    browse scope. Ingest no longer keys off browse scope at all, so on a
    local-only install the line is just noise.
    """
    state = build_library_ingest_state(
        (), form=LibraryIngestFormState(), runtime_source="server"
    )

    assert state.server_quiet_line == ""
    assert state.show_backend_switch is False
    assert state.ingest_backend == "local"


def test_a_configured_server_names_the_target_and_offers_the_switch():
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(),
        ingest_backend="local",
        runtime_source="server",
        server_ingest_available=True,
    )

    assert "this machine" in state.server_quiet_line.lower()
    assert state.show_backend_switch is True
    assert state.ingest_backend == "local"


def test_targeting_the_server_says_so_plainly():
    """A user must be able to see that their files will leave the machine."""
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(),
        ingest_backend="server",
        runtime_source="server",
        server_ingest_available=True,
    )

    assert "server" in state.server_quiet_line.lower()
    assert state.ingest_backend == "server"
    assert state.show_backend_switch is True


def test_canvas_and_submit_agree_when_the_precondition_is_unmet():
    """An unmet precondition must show local, because submit will be local.

    This test previously asserted the opposite -- that a server target stays
    named even when unusable -- to stop the canvas claiming local while submit
    tried the server. That mismatch no longer exists: ``_resolve_ingest_backend``
    applies the same runtime-policy gate, so an opted-in user whose runtime is
    local really does get a local ingest. The canvas saying "this machine" is
    now the truth, not a lie, and the line explains how to enable server
    imports.
    """
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(),
        ingest_backend="server",
        runtime_source="local",
        server_ingest_available=True,
    )

    assert state.ingest_backend == "local"
    assert state.show_backend_switch is False
    assert "server mode" in state.server_quiet_line.lower()


def test_server_ingest_needs_server_mode_not_just_a_configured_server():
    """Server ingest is gated by runtime policy, so the UI must say so.

    ``media.ingestion_jobs.launch.server`` declares ``required_source="server"``
    in the runtime-policy registry, so the service refuses the launch outright
    when the Library runtime is local -- the same rule that makes the retired
    ingest window disable its server panels in local mode. Offering the switch
    regardless produced a job that failed with
    "media.ingestion_jobs.launch.server requires server mode" (seen live).
    """
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(),
        runtime_source="local",
        server_ingest_available=True,
    )

    assert state.show_backend_switch is False
    assert "server mode" in state.server_quiet_line.lower()


def test_server_mode_plus_a_configured_server_offers_the_switch():
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(),
        runtime_source="server",
        server_ingest_available=True,
    )

    assert state.show_backend_switch is True
    assert "this machine" in state.server_quiet_line.lower()


def test_a_page_source_offers_its_scope_settings_in_the_canvas_state():
    """AC#3: the clipper's scope settings survive the move into the canvas.

    The canvas renders one option group per pre-flight type group, reading each
    group's fields from the capability schema -- so a page source has to reach
    the state as the ``web`` group, and that group has to declare the scope
    settings the retired window exposed (scrape_method plus the page/depth
    limits). Asserting on the schema the canvas actually consults is what keeps
    this from passing while the screen shows nothing.
    """
    from tldw_chatbook.Library.ingest_capabilities import get_capabilities
    from tldw_chatbook.Library.ingest_preflight import analyze_path
    from unittest.mock import MagicMock, patch

    response = MagicMock()
    response.__enter__ = MagicMock(return_value=response)
    response.__exit__ = MagicMock(return_value=False)
    with patch(
        "tldw_chatbook.Library.ingest_preflight.urlopen", return_value=response
    ):
        preflight = analyze_path("https://example.com/some-post")

    assert list(preflight.type_groups) == ["web"], (
        "a page must reach the canvas as the web group, not as an unsupported file"
    )

    # get_capabilities is exactly what the canvas calls per group; a group it
    # cannot answer for raises rather than rendering (task-673).
    fields = {f.name for f in get_capabilities("web").fields}
    assert {"scrape_method", "max_pages", "max_depth"} <= fields


class TestOpenAServerIngestedItem:
    """A finished server job must offer a route to what it produced.

    Its content lives in the server's library, so "Open in Library" -- which
    resolves a local media row -- stays withheld. The server does report the id
    of the row it created, so the item is addressable; it just needs its own
    affordance (task-700).
    """

    def _server_job(self, **kwargs):
        from tldw_chatbook.Library.library_ingest_jobs import (
            IngestJobState,
            LibraryIngestJob,
        )

        defaults = dict(
            job_id="ingest-job-1",
            source_path="/tmp/paper.pdf",
            state=IngestJobState.DONE,
            origin="server",
            remote_media_id="1125",
            started_at=1.0,
            finished_at=2.0,
        )
        defaults.update(kwargs)
        return LibraryIngestJob(**defaults)

    def test_a_finished_server_job_offers_the_server_view(self):
        from tldw_chatbook.Library.library_ingest_state import (
            _build_queue_row as build_ingest_queue_row,
        )

        row = build_ingest_queue_row(self._server_job(), now=3.0)

        assert row.can_open_on_server is True
        assert row.remote_media_id == "1125"
        # The local action stays withheld: there is no local row to open.
        assert row.can_open is False

    def test_a_server_job_without_an_id_offers_nothing(self):
        """AC#3. The server does not always report an id, and an action that
        cannot resolve anything is worse than no action."""
        from tldw_chatbook.Library.library_ingest_state import (
            _build_queue_row as build_ingest_queue_row,
        )

        row = build_ingest_queue_row(self._server_job(remote_media_id=None), now=3.0)

        assert row.can_open_on_server is False

    def test_a_local_job_never_offers_the_server_view(self):
        """Even holding an id, a local job's content is local."""
        from tldw_chatbook.Library.library_ingest_state import (
            _build_queue_row as build_ingest_queue_row,
        )

        row = build_ingest_queue_row(
            self._server_job(origin="local", media_id=7, remote_media_id="1125"),
            now=3.0,
        )

        assert row.can_open_on_server is False
        assert row.can_open is True

    def test_an_unfinished_server_job_offers_nothing_yet(self):
        from tldw_chatbook.Library.library_ingest_jobs import IngestJobState
        from tldw_chatbook.Library.library_ingest_state import (
            _build_queue_row as build_ingest_queue_row,
        )

        row = build_ingest_queue_row(
            self._server_job(state=IngestJobState.PARSING), now=3.0
        )

        assert row.can_open_on_server is False


def test_the_view_on_server_action_cannot_be_caught_by_the_local_open_handler():
    """The two row actions must not collide on id prefix or class.

    "Open in Library" matches ``.library-ingest-open`` and recovers a job id by
    stripping the prefix ``library-ingest-open-``. An id like
    ``library-ingest-open-server-<job>`` would therefore be caught by that
    handler and parsed into the bogus job id ``server-<job>`` -- opening
    nothing, from the wrong handler. Both the id prefix and the class are
    deliberately distinct (task-700).
    """
    import inspect

    from tldw_chatbook.Widgets.Library import library_ingest_canvas

    source = inspect.getsource(library_ingest_canvas)
    assert 'id=f"library-ingest-view-server-{row.job_id}"' in source
    # Check the id CONSTRUCTIONS, not the raw source: the comment above that
    # button deliberately names the colliding form to explain why it is avoided.
    ids = re.findall(r'id=f"([a-z-]+)\{row\.job_id\}"', source)
    assert not [i for i in ids if i.startswith("library-ingest-open-") and i != "library-ingest-open-"], (
        f"an action id shadows the local open prefix: {ids}"
    )
    # The class the local handler selects on must not be on the server button.
    server_button = source[source.index('"View on server"'):]
    server_button = server_button[: server_button.index("compact=True")]
    assert "library-ingest-view-server" in server_button
    assert "library-ingest-open " not in server_button


# --- task-2015: P2 batch ----------------------------------------------------


def test_short_ingest_error_collapses_nested_failed_to_prefixes():
    """(task-2015) Wrapper-on-wrapper copy like the PDF failure chain must
    collapse to a single 'Failed to …' prefix on the queue-row surface."""
    nested = (
        "Failed to ingest pdf file: Failed to process pdf file: "
        "PDF Extraction Error."
    )
    assert (
        short_ingest_error(nested)
        == "Failed to process pdf file: PDF Extraction Error."
    )


def test_short_ingest_error_leaves_single_prefix_alone():
    single = "Failed to process pdf file: PDF Extraction Error."
    assert short_ingest_error(single) == single


def test_estimate_and_breakdown_suppressed_when_preflight_has_errors():
    """(task-2015) A '0 files' estimate parked under a path error is noise;
    error states render the error + recovery only."""
    preflight = PreflightResult(
        type_groups={},
        warnings=[],
        errors=["Path not found: /nope/missing.txt"],
        total_size=0,
        truncated=False,
        total_files=0,
        path_invalid=True,
    )
    state = build_library_ingest_state(
        (), form=LibraryIngestFormState(path="/nope/missing.txt"), preflight=preflight
    )
    assert state.errors == ["Path not found: /nope/missing.txt"]
    assert state.estimate_line == ""
    assert state.type_breakdown_line == ""


def test_start_disabled_when_every_staged_file_is_unsupported():
    """(task-2015) Pre-flight just promised every file will fail; Start must
    be disabled with the gate line explaining why."""
    preflight = PreflightResult(
        type_groups={"unsupported": ["/tmp/x.json", "/tmp/y.jpg"]},
        warnings=[],
        errors=[],
        total_size=51,
        truncated=False,
        total_files=2,
    )
    state = build_library_ingest_state(
        (), form=LibraryIngestFormState(path="/tmp/folder"), preflight=preflight
    )
    assert state.start_enabled is False
    assert "unsupported" in state.start_quiet_line
    assert "2" in state.start_quiet_line


def test_start_stays_enabled_for_mixed_selection():
    preflight = PreflightResult(
        type_groups={
            "generic": ["/tmp/a.txt"],
            "unsupported": ["/tmp/x.json"],
        },
        warnings=[],
        errors=[],
        total_size=300,
        truncated=False,
        total_files=2,
    )
    state = build_library_ingest_state(
        (), form=LibraryIngestFormState(path="/tmp/folder"), preflight=preflight
    )
    assert state.start_enabled is True


def test_done_row_elapsed_measures_from_submission():
    """(task-2015) Elapsed reflects what the user actually waited: submission
    to finish, not parse-start to finish."""
    jobs = (
        _job(
            state=IngestJobState.DONE,
            source_path="/tmp/report.txt",
            submitted_at=100.0,
            started_at=104.0,
            finished_at=105.0,
            media_id=1,
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState(), now=110.0)
    assert state.queue_rows[0].line == "✓ done · report.txt · 5s"


def test_done_row_subsecond_elapsed_renders_lt_one_second():
    jobs = (
        _job(
            state=IngestJobState.DONE,
            source_path="/tmp/report.txt",
            submitted_at=100.0,
            started_at=100.1,
            finished_at=100.4,
            media_id=1,
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState(), now=110.0)
    assert state.queue_rows[0].line == "✓ done · report.txt · <1s"


def test_done_row_without_timestamps_omits_elapsed_segment():
    """A restored/malformed job with no usable timestamps must not claim
    '0s'; the elapsed segment is dropped entirely."""
    jobs = (
        _job(
            state=IngestJobState.DONE,
            source_path="/tmp/report.txt",
            submitted_at=0.0,
            started_at=None,
            finished_at=None,
            media_id=1,
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState(), now=110.0)
    assert state.queue_rows[0].line == "✓ done · report.txt"


# --- task-2043: P2 batch ----------------------------------------------------


def test_unwrap_ingest_error_collapses_chain_keeping_tail():
    """(task-2043) The details surface shows the FULL message (tail kept),
    but never more than one 'Failed to …' prefix."""
    from tldw_chatbook.Library.library_ingest_state import unwrap_ingest_error

    nested = (
        "Failed to ingest pdf file: Failed to process pdf file: "
        "PDF Extraction Error."
    )
    assert (
        unwrap_ingest_error(nested)
        == "Failed to process pdf file: PDF Extraction Error."
    )
    single = "Failed to process pdf file: PDF Extraction Error."
    assert unwrap_ingest_error(single) == single


def test_expanded_details_render_unwrapped_lines_and_retry_hint():
    """(task-2043, contract revised by task-2130) An expanded failed row
    carries category and an honest retry hint -- and NEVER a Details line
    that repeats the row summary verbatim (the round-4 critique's
    "circular details" P1: the expansion click must add information)."""
    job = _job(
        state=IngestJobState.FAILED,
        source_path="/tmp/broken.pdf",
        error="Failed to ingest pdf file: Failed to process pdf file: PDF Extraction Error.",
        error_detail={
            "category": "parse_error",
            "exception_type": "RuntimeError",
            "message": (
                "Failed to ingest pdf file: Failed to process pdf file: "
                "PDF Extraction Error."
            ),
        },
    )
    state = build_library_ingest_state(
        (job,),
        form=LibraryIngestFormState(),
        expanded_details={"ingest-job-1"},
    )
    row = state.queue_rows[0]
    assert row.details_expanded is True
    # (task-2160) No parenthesized exception class -- it serves no user.
    assert row.detail_lines[0] == "Category: parse error"
    assert not any(
        line.startswith("Details:") for line in row.detail_lines
    ), "a Details line that repeats the summary is the round-4 P1"
    # (task-2140) Parse errors get corrupt-file advice, never network talk.
    assert any("repair or re-export" in line for line in row.detail_lines)
    assert not any("network" in line for line in row.detail_lines)
    assert not any("missing tooling" in line for line in row.detail_lines)


def test_expanded_details_surface_chain_and_name_missing_dependency():
    """(task-2130) The captured exception chain renders as Underlying
    lines, a genuinely-different structured message keeps its Details
    line, and a missing-module failure names the dependency in the
    retry advisory instead of "missing tooling"."""
    job = _job(
        state=IngestJobState.FAILED,
        source_path="/tmp/broken.pdf",
        error="Failed to process pdf file: PDF Extraction Error.",
        error_detail={
            "category": "parse_error",
            "exception_type": "ImportError",
            "message": "Text extraction failed at page 3.",
            "chain": [
                "ImportError: No module named 'pymupdf'",
                "OSError: cannot open shared object",
            ],
        },
    )
    state = build_library_ingest_state(
        (job,),
        form=LibraryIngestFormState(),
        expanded_details={"ingest-job-1"},
    )
    row = state.queue_rows[0]
    assert "Details: Text extraction failed at page 3." in row.detail_lines
    assert (
        "Underlying: ImportError: No module named 'pymupdf'"
        in row.detail_lines
    )
    assert (
        "Underlying: OSError: cannot open shared object" in row.detail_lines
    )
    assert (
        "Missing dependency: pymupdf. Install it, then Retry."
        in row.detail_lines
    )

    collapsed = build_library_ingest_state((job,), form=LibraryIngestFormState())
    assert collapsed.queue_rows[0].details_expanded is False
    assert collapsed.queue_rows[0].detail_lines == ()


def test_preflight_duplicate_line_renders_and_suppresses_under_errors():
    """(task-2043) The pre-flight duplicate forecast renders a quiet line;
    error states suppress it like the estimate."""
    preflight = PreflightResult(
        type_groups={"generic": ["/tmp/a.txt", "/tmp/b.txt"]},
        warnings=[],
        errors=[],
        total_size=200,
        truncated=False,
        total_files=2,
        already_in_library=2,
    )
    state = build_library_ingest_state(
        (), form=LibraryIngestFormState(path="/tmp"), preflight=preflight
    )
    assert state.duplicate_line == (
        "2 files appear to already be in your Library — "
        "they'll be matched, not re-imported."
    )

    single = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp"),
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/a.txt"]},
            warnings=[],
            errors=[],
            total_size=100,
            truncated=False,
            total_files=1,
            already_in_library=1,
        ),
    )
    assert single.duplicate_line == (
        "1 file appears to already be in your Library — "
        "it will be matched, not re-imported."
    )

    with_errors = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/nope"),
        preflight=PreflightResult(
            type_groups={},
            warnings=[],
            errors=["Path not found: /nope"],
            total_size=0,
            truncated=False,
            total_files=0,
            path_invalid=True,
            already_in_library=1,
        ),
    )
    assert with_errors.duplicate_line == ""


def test_unsupported_line_names_files_and_matches_gate():
    """(task-2100) The forecast names the files (count alone forced a
    submit-and-read round trip); when the gate blocks the whole selection
    the line stops promising failure rows a blocked submit never records."""
    mixed = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp/folder"),
        preflight=PreflightResult(
            type_groups={
                "generic": ["/tmp/a.txt"],
                "unsupported": ["/tmp/x.json"],
            },
            warnings=[],
            errors=[],
            total_size=100,
            truncated=False,
            total_files=2,
        ),
    )
    assert mixed.unsupported_line == (
        "1 unsupported file will be recorded as a failure: x.json."
    )

    blocked = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp/folder"),
        preflight=PreflightResult(
            type_groups={"unsupported": ["/tmp/x.json", "/tmp/y.jpg"]},
            warnings=[],
            errors=[],
            total_size=50,
            truncated=False,
            total_files=2,
        ),
    )
    assert blocked.start_enabled is False
    assert blocked.unsupported_line == (
        "Unsupported: x.json, y.jpg."
        " Supported: PDF documents, audio/video files, e-books, plain text files."
    )

    many = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp/folder"),
        preflight=PreflightResult(
            type_groups={
                "generic": ["/tmp/a.txt"],
                "unsupported": [f"/tmp/u{i}.bin" for i in range(5)],
            },
            warnings=[],
            errors=[],
            total_size=500,
            truncated=False,
            total_files=6,
        ),
    )
    assert many.unsupported_line.endswith("u0.bin, u1.bin, u2.bin, ....")


def test_invalid_option_values_gate_start_with_text_message():
    """(task-2130) 'abc' as a chunk size used to sail into a running job;
    invalid option values now gate Start like a bad path, with a text
    message (not a color-only border)."""
    form = LibraryIngestFormState(path="/tmp/report.txt")
    form.type_options["generic"] = {"chunk_size": "abc"}
    state = build_library_ingest_state((), form=form)
    assert not state.start_enabled
    assert ("generic", "chunk_size", "Chunk size must be a whole number.") in (
        state.option_errors
    )
    assert state.start_quiet_line == (
        "Fix the highlighted options to start: "
        "Chunk size must be a whole number."
    )

    form.type_options["generic"] = {"chunk_size": "0"}
    zero = build_library_ingest_state((), form=form)
    assert (
        "generic",
        "chunk_size",
        "Chunk size must be between 100 and 5000.",
    ) in zero.option_errors

    # (Qodo round) The UI validator mirrors the submit-time clamp bounds:
    # a value the gate blesses is never silently rewritten at submit.
    form.type_options["generic"] = {"chunk_size": "150000"}
    huge = build_library_ingest_state((), form=form)
    assert (
        "generic",
        "chunk_size",
        "Chunk size must be between 100 and 5000.",
    ) in huge.option_errors

    form.type_options["generic"] = {"chunk_size": "1000", "chunk_overlap": "-5"}
    negative = build_library_ingest_state((), form=form)
    assert ("generic", "chunk_overlap", "Chunk overlap must be at least 0.") in (
        negative.option_errors
    )

    form.type_options["generic"] = {"chunk_size": "1000", "chunk_overlap": "100"}
    valid = build_library_ingest_state((), form=form)
    assert valid.option_errors == ()
    assert valid.start_enabled


def test_recent_ledger_survives_registry_clear_and_empty_copy_is_honest():
    """(task-2130) Recent ingests is the durable session ledger: jobs
    snapshotted at Clear-finished time still render after the registry
    removal, and the empty-queue copy stops claiming "No ingest jobs
    yet." after a session with activity."""
    cleared = _job(
        state=IngestJobState.FAILED,
        source_path="/tmp/broken.pdf",
        error="Failed to process pdf file: PDF Extraction Error.",
    )
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(),
        recent_ledger=(cleared,),
    )
    assert [job.job_id for job in state.recent_jobs] == [cleared.job_id]
    assert state.queue_empty_line == "Queue is empty."

    untouched = build_library_ingest_state((), form=LibraryIngestFormState())
    assert untouched.queue_empty_line == "No ingest jobs yet."


def test_queue_counts_line_shows_in_flight_batch_work():
    """(task-2130 pin) The tally names queued/parsing work during a batch
    -- the round-4 live report of '3 done' with no in-flight signal was a
    sampling artifact, and this pins the contract that keeps it one."""
    jobs = (
        _job(job_id="ingest-job-1", state=IngestJobState.DONE),
        _job(job_id="ingest-job-2", state=IngestJobState.PARSING),
        _job(job_id="ingest-job-3", state=IngestJobState.QUEUED),
        _job(job_id="ingest-job-4", state=IngestJobState.QUEUED),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    assert state.queue_counts_line == (
        "1 parsing · 2 queued · 1 done — all ingests"
    )


def test_capped_duplicate_forecast_says_at_least():
    """(task-2130) When the duplicate check hits its 20-candidate cap the
    count is a floor -- an 80-duplicate folder used to read '20 files
    appear to already be…' presenting the cap as the total."""
    capped = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp/folder"),
        preflight=PreflightResult(
            type_groups={"generic": [f"/tmp/f{i}.txt" for i in range(20)]},
            warnings=[],
            errors=[],
            total_size=1000,
            truncated=False,
            total_files=80,
            already_in_library=20,
            already_in_library_capped=True,
        ),
    )
    assert capped.duplicate_line.startswith(
        "at least 20 files appear to already be in your Library"
    )
    assert "at least 20 will match" in capped.commit_summary_line


def test_option_errors_skip_hidden_groups_and_gated_fields():
    """(task-2130 Qodo round) A stale invalid value in a panel that is not
    rendered, or in a field whose enabled_when gate is off, must not block
    Start with nothing visible to fix."""
    form = LibraryIngestFormState(path="/tmp/report.txt")
    # Invalid value in a group whose panel is NOT rendered (no preflight
    # -> only the generic group is validated).
    form.type_options["web"] = {"max_pages": "abc"}
    form.type_options["generic"] = {"chunk_size": "1000"}
    state = build_library_ingest_state((), form=form)
    assert state.option_errors == ()
    assert state.start_enabled

    # Invalid chunk_size while its enabled_when gate (chunk) is OFF: the
    # field renders disabled, so it must not gate Start either.
    form.type_options["generic"] = {"chunk": False, "chunk_size": "abc"}
    gated_off = build_library_ingest_state((), form=form)
    assert gated_off.option_errors == ()


def test_underlying_lines_skip_prefixed_restatements_of_the_row_error():
    """(task-2140) A chain entry that merely restates the row's error
    behind a "ClassName: " prefix must not render -- round 5 saw
    "Underlying: FileIngestionError: <row error>" as the only detail."""
    job = _job(
        state=IngestJobState.FAILED,
        source_path="/tmp/broken.pdf",
        error="Failed to process pdf file: PDF Extraction Error.",
        error_detail={
            "category": "parse_error",
            "exception_type": "FileIngestionError",
            "message": "Failed to process pdf file: PDF Extraction Error.",
            "chain": [
                "FileIngestionError: Failed to process pdf file: PDF Extraction Error.",
                "ValueError: startxref not found",
            ],
        },
    )
    state = build_library_ingest_state(
        (job,),
        form=LibraryIngestFormState(),
        expanded_details={"ingest-job-1"},
    )
    row = state.queue_rows[0]
    assert not any(
        "FileIngestionError: Failed to process" in line
        for line in row.detail_lines
    ), "prefixed restatement of the row error leaked into the details"
    assert "Underlying: ValueError: startxref not found" in row.detail_lines


def test_empty_files_forecast_as_failures_not_imports():
    """(task-2160) A 0-byte file is pulled out of its type group at
    analysis time and forecast as a failure -- the forecast used to
    promise '1 will import' for a file it had measured at 0 B."""
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp/folder"),
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/real.txt"]},
            warnings=[],
            errors=[],
            total_size=100,
            truncated=False,
            total_files=2,
            empty_files=("/tmp/empty.txt",),
        ),
    )
    assert state.empty_line == "1 empty file will fail — empty.txt is 0 B."
    assert state.commit_summary_line == "1 will import · 1 will fail"

    solo = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp/empty.txt"),
        preflight=PreflightResult(
            type_groups={},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=1,
            empty_files=("/tmp/empty.txt",),
        ),
    )
    # A selection that is ONLY an empty file has nothing importable --
    # the gate blocks exactly like a solo-unsupported selection.
    assert not solo.start_enabled


def test_armed_clear_label_names_failed_rows():
    """(task-2160) 'finished' includes failed rows -- the armed label must
    say so at the moment of destruction."""
    jobs = (
        _job(job_id="ingest-job-1", state=IngestJobState.DONE),
        _job(job_id="ingest-job-2", state=IngestJobState.FAILED),
        _job(job_id="ingest-job-3", state=IngestJobState.FAILED),
    )
    armed = build_library_ingest_state(
        jobs, form=LibraryIngestFormState(), clear_finished_armed=True
    )
    assert armed.queue_clear_finished_label == (
        "Press again to clear 3 finished (incl. 2 failed)"
    )
    done_only = build_library_ingest_state(
        (_job(job_id="ingest-job-1", state=IngestJobState.DONE),),
        form=LibraryIngestFormState(),
        clear_finished_armed=True,
    )
    assert done_only.queue_clear_finished_label == (
        "Press again to clear 1 finished"
    )


def test_all_match_selection_gets_consent_line_and_stays_enabled():
    """(task-2223 ruling) Zero imports + >=1 predicted match keeps Start
    ENABLED (the dedup probe is capped best-effort) with an
    informed-consent quiet line saying what starting will actually do."""
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp/report.txt"),
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/report.txt"]},
            warnings=[],
            errors=[],
            total_size=100,
            truncated=False,
            total_files=1,
            already_in_library=1,
        ),
    )
    assert state.start_enabled
    assert state.start_quiet_line == (
        "Everything here appears to already be in your Library — "
        "starting will re-check and match, not re-import."
    )
    assert state.commit_summary_line == "0 will import · 1 will match"
