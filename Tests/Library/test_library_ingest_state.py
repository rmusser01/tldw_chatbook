"""Pure display-state contracts for the Library ingest canvas (L3b Task 4)."""

from __future__ import annotations

import re

import pytest

from tldw_chatbook.Library.ingest_capabilities import (
    field_available_for_backend,
    get_capabilities,
)
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
    format_ingest_progress_line,
    ingest_progress_action_signature,
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


def test_import_behavior_capabilities_share_controls_across_backends() -> None:
    """The capability schema is the one visibility/defaults source for modes."""
    fields = {field.name: field for field in get_capabilities("generic").fields}

    assert get_capabilities("generic").label == "Import behavior"
    for name in (
        "overwrite_existing",
        "custom_prompt",
        "system_prompt",
        "generate_embeddings",
    ):
        assert field_available_for_backend(fields[name], "local") is True
        assert field_available_for_backend(fields[name], "server") is True

    assert fields["generate_embeddings"].default is True
    assert field_available_for_backend(fields["keep_original_file"], "local") is False
    assert field_available_for_backend(fields["keep_original_file"], "server") is True


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
    assert state.start_quiet_line == "Enter a file path or URL to start."


def test_start_quiet_line_hidden_once_path_is_typed():
    state = build_library_ingest_state(
        (), form=LibraryIngestFormState(path="/tmp/a.txt")
    )
    assert state.start_quiet_line == ""


def test_start_quiet_line_shown_for_whitespace_only_path():
    state = build_library_ingest_state((), form=LibraryIngestFormState(path="   "))
    assert state.start_quiet_line == "Enter a file path or URL to start."


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


def test_failed_row_line_for_a_non_clippable_source_says_imported_not_ingested():
    """(task-2857 review, round 3) A local-file/media-URL source rejected by
    ``build_web_clip_kwargs`` (``NotAWebClipSource``) is queued and marked
    ``FAILED`` with that exception's own message (``app.py``'s
    ``_submit_web_clip_job``) -- the exact text a user sees in the failed
    queue row. It must say "imported", matching every other Import-flow
    string, not survive as the sole remaining "ingested"."""
    from tldw_chatbook.Library.web_clip_request import (
        NotAWebClipSource,
        build_web_clip_kwargs,
    )

    with pytest.raises(NotAWebClipSource) as excinfo:
        build_web_clip_kwargs("/tmp/notes.txt", options={})
    error_text = str(excinfo.value)
    assert "imported" in error_text
    assert "ingested" not in error_text

    jobs = (
        _job(
            state=IngestJobState.FAILED,
            source_path="/tmp/notes.txt",
            started_at=100.0,
            finished_at=101.0,
            error=error_text,
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    row = state.queue_rows[0]
    assert "imported, not clipped" in row.line
    assert "ingested" not in row.line


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
    assert row.line == "✗ failed · report.txt · bad codec · attempt 3"


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
        == "This queue: 1 parsing · 1 writing · 1 queued · 2 done · 1 failed"
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
    # (task-3305, MI-14) All jobs terminal: a trailing "— in queue" would
    # read as a contradiction over a finished run.
    # (task-2859 item 4) "This queue:" scopes the tally as a leading label
    # instead, without ever claiming a segment is still active.
    assert state.queue_counts_line == "This queue: 2 done · 1 failed"


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


def test_build_type_breakdown_line_document_group_has_honest_noun():
    """(task-3303 AC1) A .docx used to pre-flight as a "plain text file"."""
    line = build_type_breakdown_line({"document": ["/tmp/report.docx"]})
    assert line == "1 Word/Office document"


def test_build_type_breakdown_line_document_group_pluralizes():
    line = build_type_breakdown_line(
        {"document": ["/tmp/a.docx", "/tmp/b.odt", "/tmp/c.rtf"]}
    )
    assert line == "3 Word/Office documents"


# --- build_web_scope_note (task-3303 AC5) ----------------------------------


def test_web_scope_note_warns_local_multi_page_selection():
    """A local "sitemap" import silently fetched ONE page -- say so."""
    from tldw_chatbook.Library.library_ingest_state import build_web_scope_note

    note = build_web_scope_note("local", {"scrape_method": "sitemap"})
    assert note, "a local multi-page selection must carry the reason"
    assert "server" in note
    assert "one page" in note


def test_web_scope_note_covers_every_multi_page_method():
    from tldw_chatbook.Library.ingest_capabilities import (
        MULTI_PAGE_SCRAPE_METHODS,
    )
    from tldw_chatbook.Library.library_ingest_state import build_web_scope_note

    for method in MULTI_PAGE_SCRAPE_METHODS:
        assert build_web_scope_note("local", {"scrape_method": method})


def test_web_scope_note_silent_for_single_page_local():
    from tldw_chatbook.Library.library_ingest_state import build_web_scope_note

    assert build_web_scope_note("local", {"scrape_method": "individual"}) == ""
    # An untouched form defaults to the single-page method.
    assert build_web_scope_note("local", {}) == ""


def test_web_scope_note_silent_when_targeting_the_server():
    """Server behavior is unchanged: the clip path honors multi-page options."""
    from tldw_chatbook.Library.library_ingest_state import build_web_scope_note

    assert build_web_scope_note("server", {"scrape_method": "sitemap"}) == ""


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
    warnings = [
        # ``feature`` is what production always sets; without it this is
        # an advisory, not a missing component (see the producer guard in
        # test_ingest_capabilities.py).
        {
            "feature": "pdf_processing",
            "label": "PDF",
            "hint": "missing",
            "command": "pip install x",
        }
    ]
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
        warnings=[
            {"feature": "pdf_processing", "label": "PDF", "hint": "missing"}
        ],
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


@pytest.mark.parametrize(
    ("progress", "state", "expected"),
    [
        (
            {
                "phase": "extracting",
                "message": "Extracting page 21 of 50",
                "percent": 42.0,
            },
            IngestJobState.PARSING,
            "42% · Extracting page 21 of 50",
        ),
        (
            {"phase": "transcribing"},
            IngestJobState.PARSING,
            "Transcribing audio",
        ),
        (None, IngestJobState.PARSING, "Preparing import"),
        (
            {"phase": "writing", "message": "Saving to Library"},
            IngestJobState.WRITING,
            "Saving to Library",
        ),
        (
            {"message": "Imported report.txt"},
            IngestJobState.DONE,
            "Imported report.txt",
        ),
    ],
)
def test_format_ingest_progress_line(progress, state, expected):
    """State prefixes or invented values would make the reserved detail lie."""
    rendered = format_ingest_progress_line(progress, state=state)
    assert rendered == expected
    assert "Â·" not in rendered


@pytest.mark.parametrize(
    "percent",
    [float("nan"), float("inf"), -0.1, 100.1],
)
def test_format_ingest_progress_line_omits_invalid_percentages(percent) -> None:
    """Clamping invalid values would turn broken telemetry into false precision."""
    assert format_ingest_progress_line(
        {"phase": "extracting", "percent": percent},
        state=IngestJobState.PARSING,
    ) == "Extracting"


def test_format_ingest_progress_line_does_not_round_incomplete_work_to_100() -> None:
    """A fractional measurement below 100 must not look complete."""
    assert format_ingest_progress_line(
        {"phase": "extracting", "percent": 99.5},
        state=IngestJobState.PARSING,
    ) == "99% · Extracting"
    assert format_ingest_progress_line(
        {"phase": "extracting", "percent": 100.0},
        state=IngestJobState.PARSING,
    ) == "100% · Extracting"


def test_format_ingest_progress_line_normalizes_and_bounds_message() -> None:
    """An unbounded server message would flood the queue detail line."""
    message = "Extracting\n" + ("x" * 200)

    rendered = format_ingest_progress_line(
        {"message": message}, state=IngestJobState.PARSING
    )

    assert rendered == "Extracting " + ("x" * 149)
    assert "\n" not in rendered


def test_format_ingest_progress_line_omits_enormous_integer_percent() -> None:
    """Converting invalid giant telemetry to float must not crash formatting."""
    assert format_ingest_progress_line(
        {"phase": "extracting", "percent": 10**400},
        state=IngestJobState.PARSING,
    ) == "Extracting"


@pytest.mark.parametrize(
    ("progress", "expected"),
    [
        ({"phase": "transcribing"}, (True, False)),
        ({"phase": "transcribing", "cancel_requested": True}, (False, True)),
        ({"phase": "extracting"}, (False, False)),
    ],
)
def test_ingest_progress_action_signature_uses_local_stt_progress_rules(
    progress, expected
) -> None:
    """Drifting the shared predicate would make stable row updates miss actions."""
    job = _job(state=IngestJobState.PARSING, progress=progress)

    assert ingest_progress_action_signature(job) == expected


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


def test_a_running_local_stt_attempt_can_be_cancelled():
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/a.wav",
        state=IngestJobState.PARSING,
        progress={"phase": "transcribing"},
    )

    row = _row_for(job)
    assert row.can_cancel is True
    assert row.can_force_stop is False


def test_a_cancel_requested_local_stt_attempt_offers_force_stop_only():
    job = LibraryIngestJob(
        job_id="ingest-job-1",
        source_path="/tmp/a.wav",
        state=IngestJobState.PARSING,
        progress={"phase": "transcribing", "cancel_requested": True},
    )

    row = _row_for(job)
    assert row.can_cancel is False
    assert row.can_force_stop is True


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

    # (TASK-19556) The pre-flight no longer probes a URL by default -- that
    # network call fired from the ingest field's typing debounce and made it
    # an internal-host scanning oracle. Classification, which is what this
    # test is about, needs no probe at all now.
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


# --- task-3312 (#2): egress-blocked receipts read as plain language ----------

# The exact text an egress-blocked URL ingest produces today:
# web_article_ingestion wraps ``EgressBlockedError``'s message, whose remedy
# tail carries the raw config-key brackets that leaked into the queue row
# (live 2026-08-08: rendered a literal "\[web_security]" and clipped
# mid-sentence at "config.toml,").
_EGRESS_RAW_ERROR = (
    "URL blocked by egress policy (SSRF guard): Egress blocked (private) "
    "for http://127.0.0.1:8000 [remedy: add the host to [web_security] "
    "allowed_hosts in config.toml, or set [web_security] enabled = false]"
)


def test_short_ingest_error_maps_egress_block_to_plain_language():
    """task-3312 (#2): the queue receipt must match the pre-flight line's
    plain-language register (task-3305) -- a complete sentence, no policy
    jargon, no markup-hostile brackets -- while keeping the remedy.

    (xhigh review round) The register is unchanged; the receipt now also
    NAMES the refused origin, which the fixed sentence never did.
    """
    short = short_ingest_error(_EGRESS_RAW_ERROR)
    assert "http://127.0.0.1:8000" in short
    # No bracketed config-key syntax to fight the renderer with.
    assert "[" not in short and "\\" not in short
    # The remedy survives in plain words.
    assert "allowed_hosts" in short
    assert "web_security" in short
    assert "config.toml" in short
    # A complete sentence -- the live receipt ended "config.toml,".
    assert short.endswith(".")


def test_short_ingest_error_maps_egress_block_under_pipeline_wrappers():
    """The pipeline may wrap the egress text in its historical
    'Failed to … file:' layers; the mapping keys on the egress marker, not
    on position."""
    wrapped = f"Failed to ingest web file: {_EGRESS_RAW_ERROR}"
    assert short_ingest_error(wrapped) == short_ingest_error(
        _EGRESS_RAW_ERROR
    )
    assert "http://127.0.0.1:8000" in short_ingest_error(wrapped)


def test_failed_queue_row_for_egress_block_carries_the_plain_receipt():
    """The FAILED queue row (and Home's failed-item line, same helper)
    renders the plain-language receipt, never the raw policy text."""
    job = _job(
        job_id="ingest-job-egress",
        source_path="http://127.0.0.1:8000/page",
        state=IngestJobState.FAILED,
        error=_EGRESS_RAW_ERROR,
    )
    state = build_library_ingest_state((job,), form=LibraryIngestFormState())
    (row,) = state.queue_rows
    assert short_ingest_error(_EGRESS_RAW_ERROR) in row.line
    assert "SSRF" not in row.line
    assert "[web_security]" not in row.line
    assert "[remedy" not in row.line


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
        type_groups={"unsupported": ["/tmp/x.json", "/tmp/y.srt"]},
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
    # (task-14821) ...and no raw internal token either: the reason reads
    # as a sentence, not as the pipeline's own category name.
    assert row.detail_lines[0] == "Reason: The file couldn't be read."
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
        "1 unsupported file will be skipped: x.json."
    )

    blocked = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp/folder"),
        preflight=PreflightResult(
            type_groups={"unsupported": ["/tmp/x.json", "/tmp/y.srt"]},
            warnings=[],
            errors=[],
            total_size=50,
            truncated=False,
            total_files=2,
        ),
    )
    assert blocked.start_enabled is False
    assert blocked.unsupported_line == (
        "Unsupported: x.json, y.srt."
        " Supported: PDF documents, Word/Office documents, audio/video files,"
        " e-books, images, plain text files, web pages (by URL)."
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
    """(task-2130) Recent imports is the durable session ledger: jobs
    snapshotted at Clear-finished time still render after the registry
    removal, and the empty-queue copy stops claiming "No import jobs
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
    assert untouched.queue_empty_line == "No import jobs yet."


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
        "This queue: 1 parsing · 2 queued · 1 done"
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


def test_trim_time_validation_accepts_ffmpeg_forms_and_rejects_garbage():
    """(task-3306) Start/Stop trim inputs are format-gated at the shared
    validator seam: the values travel verbatim to ffmpeg's -ss/-to/-t (and
    yt-dlp's postprocessor args), which accept plain seconds or
    [HH:]MM:SS[.fraction] -- anything else fails the job only at run time,
    long after the form could have said so."""
    from tldw_chatbook.Library.ingest_capabilities import get_capabilities
    from tldw_chatbook.Library.library_ingest_state import (
        validate_ingest_option_value,
    )

    fields = {f.name: f for f in get_capabilities("audio_video").fields}
    start_field = fields["start_time"]
    end_field = fields["end_time"]

    for good in ("", "  ", "90", "90.5", "0:30", "1:30", "01:02:03",
                 "1:02:03.5", "10:5"):
        assert validate_ingest_option_value(start_field, good) == "", good
        assert validate_ingest_option_value(end_field, good) == "", good

    for bad in ("abc", "1:75", "12:34:56:78", "-5", ":30", "1:2:3:4", "1h30"):
        message = validate_ingest_option_value(start_field, bad)
        assert message, bad
        assert "Start at" in message
        assert "HH:MM:SS" in message and "seconds" in message

    bad_end = validate_ingest_option_value(end_field, "nope")
    assert "Stop at" in bad_end


def test_invalid_trim_time_gates_start_when_audio_panel_rendered(monkeypatch):
    """(task-3306) End-to-end through the state gate: a malformed trim
    value in a RENDERED audio/video panel blocks Start with a message."""
    import tldw_chatbook.Library.library_ingest_state as state_mod

    monkeypatch.setattr(state_mod, "_dependency_installed", lambda feature: True)
    form = LibraryIngestFormState(path="/tmp/talk.mp3")
    form.type_options["audio_video"] = {"start_time": "abc"}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=PreflightResult(
            type_groups={"audio_video": ["/tmp/talk.mp3"]},
            warnings=[],
            errors=[],
            total_size=100,
            truncated=False,
            total_files=1,
        ),
    )
    assert not state.start_enabled
    assert any(
        group == "audio_video" and name == "start_time"
        for group, name, _message in state.option_errors
    )


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


def test_skipped_jobs_render_neutral_and_count_separately():
    """(task-2220 ruling) A skipped job renders with the neutral glyph, no
    Retry, dismiss offered; the tally counts skips in their own segment,
    never as failures; the armed clear label counts them as finished but
    not as failed."""
    skipped = _job(
        job_id="ingest-job-1",
        state=IngestJobState.SKIPPED,
        source_path="/tmp/photo.xyz",
        error="Unsupported file type: .xyz.",
    )
    done = _job(job_id="ingest-job-2", state=IngestJobState.DONE)
    state = build_library_ingest_state(
        (skipped, done),
        form=LibraryIngestFormState(),
        clear_finished_armed=True,
    )
    row = next(r for r in state.queue_rows if r.job_id == "ingest-job-1")
    assert row.glyph == "○"
    assert row.line.startswith("○ skipped · photo.xyz")
    assert row.can_retry is False
    assert row.can_dismiss is True
    assert state.queue_counts_line == "This queue: 1 done · 1 skipped"
    assert state.queue_clear_finished_label == (
        "Press again to clear 2 finished"
    )
    assert [j.job_id for j in state.recent_jobs] == [
        "ingest-job-1",
        "ingest-job-2",
    ]


def test_commit_summary_splits_skip_from_fail():
    """(task-2220) Unsupported files forecast as 'will skip'; empty files
    keep 'will fail' (they are enqueued and genuinely fail)."""
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp/folder"),
        preflight=PreflightResult(
            type_groups={
                "generic": ["/tmp/a.txt", "/tmp/b.txt"],
                "unsupported": ["/tmp/pic.srt"],
            },
            warnings=[],
            errors=[],
            total_size=100,
            truncated=False,
            total_files=4,
            empty_files=("/tmp/zero.txt",),
        ),
    )
    assert state.commit_summary_line == (
        "2 will import · 1 will skip · 1 will fail"
    )
    assert "will be skipped: pic.srt." in state.unsupported_line


def test_skips_only_queue_still_offers_clear_finished():
    """(task-2220 Qodo round) A queue holding ONLY skipped rows must show
    the Clear finished control -- skips count as finished everywhere."""
    skipped = _job(
        job_id="ingest-job-1",
        state=IngestJobState.SKIPPED,
        source_path="/tmp/photo.jpg",
    )
    state = build_library_ingest_state((skipped,), form=LibraryIngestFormState())
    assert state.queue_show_clear_finished is True


def test_queue_groups_batches_with_headers_and_latest_line():
    """(task-2221 owner ruling) Contiguous same-batch runs become one
    headed group (source, count, age, outcomes); singles stay bare; the
    latest-batch line leads with the newest batch's outcomes."""
    single = _job(
        job_id="ingest-job-1",
        state=IngestJobState.DONE,
        source_path="/tmp/solo.txt",
    )
    b1 = _job(
        job_id="ingest-job-2",
        state=IngestJobState.DONE,
        source_path="/data/folder_a/one.txt",
        batch_id="local-aaa",
        submitted_at=100.0,
    )
    b2 = _job(
        job_id="ingest-job-3",
        state=IngestJobState.SKIPPED,
        source_path="/data/folder_a/pic.jpg",
        batch_id="local-aaa",
        submitted_at=101.0,
    )
    state = build_library_ingest_state(
        (single, b1, b2), form=LibraryIngestFormState()
    )
    assert len(state.queue_groups) == 2
    bare, headed = state.queue_groups
    assert bare.header_line == ""
    assert bare.job_ids == ("ingest-job-1",)
    assert headed.batch_id == "local-aaa"
    assert headed.job_ids == ("ingest-job-2", "ingest-job-3")
    # Whole-branch review M-D (pre-existing conformance): no leading "▸ " --
    # the task-4023 AC#5 convention reserves that prefix for the selected
    # row of a list, and this header is a plain grouping Static.
    assert headed.header_line.startswith("folder_a — 2 files")
    assert not headed.header_line.startswith("▸")
    assert "1 done" in headed.header_line
    assert "1 skipped" in headed.header_line
    # The batch carries the newer submitted_at here, so it is the latest
    # run (the single job's default timestamp is older).
    assert state.latest_batch_line == "Latest run: 1 done · 1 skipped"


def test_active_batch_header_uses_exact_states_without_running_synonym():
    jobs = (
        _job(
            job_id="ingest-job-1",
            state=IngestJobState.QUEUED,
            source_path="/data/folder/a.txt",
            batch_id="local-active",
        ),
        _job(
            job_id="ingest-job-2",
            state=IngestJobState.DONE,
            source_path="/data/folder/b.txt",
            batch_id="local-active",
        ),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    header = state.queue_groups[0].header_line
    assert "active" in header
    assert "1 queued" in header
    assert "1 done" in header
    assert "running" not in header


def test_single_file_submission_reads_naturally_without_header():
    """(task-2221) A batchless single job renders exactly as before: one
    bare group, no header, no latest-batch line."""
    solo = _job(job_id="ingest-job-1", state=IngestJobState.DONE)
    state = build_library_ingest_state((solo,), form=LibraryIngestFormState())
    assert len(state.queue_groups) == 1
    assert state.queue_groups[0].header_line == ""
    assert state.latest_batch_line == ""


def test_latest_run_line_follows_a_single_file_submission() -> None:
    """The latest-run line reports a single-file submission.

    (task-2230) THE round-7 regression: the line was computed only from
    groups carrying a batch_id, so a single-file run left it reporting
    the previous multi-file batch. Every submission is a run.
    """
    b1 = _job(
        job_id="ingest-job-1",
        state=IngestJobState.DONE,
        source_path="/data/folder_a/one.txt",
        batch_id="local-aaa",
        submitted_at=100.0,
    )
    b2 = _job(
        job_id="ingest-job-2",
        state=IngestJobState.SKIPPED,
        source_path="/data/folder_a/pic.jpg",
        batch_id="local-aaa",
        submitted_at=101.0,
    )
    later_single = _job(
        job_id="ingest-job-3",
        state=IngestJobState.DONE,
        source_path="/tmp/solo.txt",
        submitted_at=500.0,
    )
    state = build_library_ingest_state(
        (b1, b2, later_single), form=LibraryIngestFormState()
    )
    assert state.latest_batch_line == "Latest run: 1 done", (
        "the single-file run is the latest submission and must be reported"
    )


def test_latest_run_line_hidden_when_the_queue_holds_one_run() -> None:
    """The latest-run line hides when the queue holds a single run.

    (task-2230) The group header already reports it, so the line would
    just repeat itself.
    """
    only = _job(job_id="ingest-job-1", state=IngestJobState.DONE)
    state = build_library_ingest_state((only,), form=LibraryIngestFormState())
    assert state.latest_batch_line == ""


def test_unresolvable_path_gates_start_with_an_explanation() -> None:
    """An unresolvable path gates Start and explains itself.

    (task-2230) It used to leave Start styled exactly like a valid
    selection with a BLANK gate line, and pressing it left no queue
    record at all.
    """
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp/nope_does_not_exist.txt"),
        preflight=PreflightResult(
            type_groups={},
            warnings=[],
            errors=["Path not found: /tmp/nope_does_not_exist.txt"],
            total_size=0,
            truncated=False,
            total_files=0,
            path_invalid=True,
        ),
    )
    assert not state.start_enabled
    assert state.start_quiet_line == (
        "Can't find that path — check it, or use Browse… to pick a file "
        "or folder."
    )


def test_matched_rows_and_tallies_use_the_forecast_vocabulary() -> None:
    """A dedup match reports as "matched", distinct from an import.

    (task-2837) The forecast promises "will import · will match · will
    skip"; the receipt used to fold match into "done" and render the two
    outcomes as byte-identical rows, so the promise could not be audited.
    """
    imported = _job(
        job_id="ingest-job-1",
        state=IngestJobState.DONE,
        source_path="/tmp/fresh.txt",
        progress={"message": "Imported fresh.txt"},
        batch_id="local-aaa",
        submitted_at=10.0,
    )
    matched = _job(
        job_id="ingest-job-2",
        state=IngestJobState.DONE,
        source_path="/tmp/twin.txt",
        progress={
            "message": (
                "Already in Library — matched an existing item; nothing "
                "new was imported."
            )
        },
        batch_id="local-aaa",
        submitted_at=11.0,
    )
    other = _job(
        job_id="ingest-job-3",
        state=IngestJobState.DONE,
        source_path="/tmp/solo.txt",
        submitted_at=99.0,
    )
    state = build_library_ingest_state(
        (imported, matched, other), form=LibraryIngestFormState()
    )
    rows = {row.job_id: row for row in state.queue_rows}
    assert rows["ingest-job-1"].line.startswith("✓ done · fresh.txt")
    assert rows["ingest-job-2"].line.startswith("≡ matched · twin.txt")
    assert rows["ingest-job-2"].glyph == "≡"

    headed = next(g for g in state.queue_groups if g.header_line)
    assert "1 done" in headed.header_line
    assert "1 matched" in headed.header_line


def test_active_rows_show_the_attempt_number_after_a_retry() -> None:
    """A re-attempt is visible while it runs, not only once it ends.

    (task-2837) Requeue creates a new QUEUED job with an incremented
    count, but the in-flight rows never showed it — so pressing Retry
    looked identical to nothing happening.
    """
    # (Qodo round) detected_type is appended by the parsing/writing
    # branches, so the marker must be the row's TRAILING element -- with a
    # type present it used to read "… · attempt 2 · pdf".
    for state_value, word, detected in (
        (IngestJobState.QUEUED, "queued", ""),
        (IngestJobState.PARSING, "parsing", "pdf"),
        (IngestJobState.WRITING, "writing", "pdf"),
    ):
        job = _job(
            job_id="ingest-job-1",
            state=state_value,
            source_path="/tmp/broken.pdf",
            retry_count=1,
            detected_type=detected,
        )
        row = build_library_ingest_state(
            (job,), form=LibraryIngestFormState()
        ).queue_rows[0]
        assert row.line.startswith(f"● {word} · broken.pdf")
        assert row.line.endswith("· attempt 2"), (
            f"{word} row must show the attempt number: {row.line!r}"
        )

    first = _job(job_id="ingest-job-2", state=IngestJobState.PARSING)
    first_row = build_library_ingest_state(
        (first,), form=LibraryIngestFormState()
    ).queue_rows[0]
    assert "attempt" not in first_row.line


def test_consent_line_requires_every_importable_file_to_match() -> None:
    """"Everything here" only renders when it is true.

    (task-2837) The line rendered on a selection where only some files
    were predicted matches.
    """
    form = LibraryIngestFormState(path="/tmp/folder")
    partial = build_library_ingest_state(
        (),
        form=form,
        preflight=PreflightResult(
            type_groups={
                "generic": ["/tmp/a.txt", "/tmp/b.txt"],
                "unsupported": ["/tmp/pic.srt"],
            },
            warnings=[],
            errors=[],
            total_size=100,
            truncated=False,
            total_files=3,
            already_in_library=2,
        ),
    )
    assert "Everything here" not in partial.start_quiet_line

    total = build_library_ingest_state(
        (),
        form=form,
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
    assert total.start_quiet_line.startswith("Everything here")


def test_queue_tally_and_group_header_agree_on_matched() -> None:
    """The tally buckets the way the headers and rows do.

    (task-2837 Qodo round) The top-level counts line bucketed purely by
    state, so it read "2 done" while a group header directly below it
    read "1 done · 1 matched" — two contradictory summaries on one
    screen.
    """
    imported = _job(
        job_id="ingest-job-1",
        state=IngestJobState.DONE,
        source_path="/tmp/fresh.txt",
        progress={"message": "Imported fresh.txt"},
        batch_id="local-aaa",
        submitted_at=10.0,
    )
    matched = _job(
        job_id="ingest-job-2",
        state=IngestJobState.DONE,
        source_path="/tmp/twin.txt",
        progress={
            "message": (
                "Already in Library — matched an existing item; nothing "
                "new was imported."
            )
        },
        batch_id="local-aaa",
        submitted_at=11.0,
    )
    state = build_library_ingest_state(
        (imported, matched), form=LibraryIngestFormState()
    )
    assert state.queue_counts_line == "This queue: 1 done · 1 matched"
    headed = next(g for g in state.queue_groups if g.header_line)
    assert "1 done" in headed.header_line
    assert "1 matched" in headed.header_line


# ---------------------------------------------------------------------------
# task-3301: Analyze-after-import readiness hint
# ---------------------------------------------------------------------------


def test_analysis_hint_renders_when_analyze_on_and_provider_unready() -> None:
    form = LibraryIngestFormState(analyze=True)
    state = build_library_ingest_state(
        (),
        form=form,
        analysis_unready_hint=(
            "Analyze after import is on, but OpenAI is not ready: Missing "
            "API key. Imports will run without analysis."
        ),
    )
    assert "OpenAI" in state.analysis_hint_line
    assert "without analysis" in state.analysis_hint_line


def test_analysis_hint_empty_when_analyze_off() -> None:
    form = LibraryIngestFormState(analyze=False)
    state = build_library_ingest_state(
        (),
        form=form,
        analysis_unready_hint="Analyze after import is on, but nothing is ready.",
    )
    assert state.analysis_hint_line == ""


def test_analysis_hint_empty_when_provider_ready() -> None:
    form = LibraryIngestFormState(analyze=True)
    state = build_library_ingest_state((), form=form, analysis_unready_hint="")
    assert state.analysis_hint_line == ""


def test_analysis_hint_does_not_block_start() -> None:
    """The hint informs; analysis is optional, so Start stays available."""
    form = LibraryIngestFormState(path="/tmp/file.txt", analyze=True)
    state = build_library_ingest_state(
        (),
        form=form,
        analysis_unready_hint="Analyze after import is on, but X is not ready.",
    )
    assert state.start_enabled is True


# ---------------------------------------------------------------------------
# task-28007 Task 3 (AC#1/AC#2): batch-analyze a run's analysis-skipped items
# ---------------------------------------------------------------------------


def _skipped_job(**overrides) -> LibraryIngestJob:
    defaults = dict(
        job_id="ingest-job-1",
        source_path="/tmp/notes.txt",
        state=IngestJobState.DONE,
        media_id=7,
        submitted_at=100.0,
        finished_at=101.0,
        progress={
            "message": (
                "Imported notes.txt — analysis skipped: no analysis "
                "provider is configured"
            ),
            "analysis_skipped": "no analysis provider is configured",
        },
    )
    defaults.update(overrides)
    return LibraryIngestJob(**defaults)


def test_analyze_skipped_action_hidden_without_skipped_items():
    """No skipped rows -- the action never appears, ready or not."""
    done = _job(
        state=IngestJobState.DONE,
        media_id=1,
        progress={"message": "Imported x.txt"},
    )
    state = build_library_ingest_state(
        (done,), form=LibraryIngestFormState(), analysis_action_ready=True
    )
    assert state.analyze_skipped_media_ids == ()
    assert state.show_analyze_skipped is False


def test_analyze_skipped_action_hidden_while_the_provider_is_not_ready():
    """AC#1's gate: skipped items alone are not enough -- Task 1's reason
    must ALSO be empty (re-offering the action would repeat the exact
    failure it exists to fix)."""
    state = build_library_ingest_state(
        (_skipped_job(),),
        form=LibraryIngestFormState(),
        analysis_action_ready=False,
    )
    assert state.analyze_skipped_media_ids == ("7",)
    assert state.show_analyze_skipped is False


def test_analyze_skipped_action_shows_with_skipped_items_and_a_ready_provider():
    state = build_library_ingest_state(
        (_skipped_job(),),
        form=LibraryIngestFormState(),
        analysis_action_ready=True,
    )
    assert state.analyze_skipped_media_ids == ("7",)
    assert state.show_analyze_skipped is True


def test_analyze_skipped_ids_span_the_whole_visible_queue_not_one_batch():
    """The action id is fixed/singular ("library-ingest-analyze-skipped"),
    so it is ONE canvas-wide control over every skipped id currently in
    the queue -- never one per batch (which could mount the same id
    twice and crash)."""
    first = _skipped_job(job_id="ingest-job-1", media_id=1, batch_id="local-aaa")
    second = _skipped_job(job_id="ingest-job-2", media_id=2, batch_id="local-bbb")
    state = build_library_ingest_state(
        (first, second),
        form=LibraryIngestFormState(),
        analysis_action_ready=True,
    )
    assert set(state.analyze_skipped_media_ids) == {"1", "2"}


def test_analyze_skipped_excludes_an_id_this_action_already_fixed():
    """N is the count of skipped rows that STILL have no analysis: an id
    the screen's own outcomes map already marked ok=True drops out."""
    state = build_library_ingest_state(
        (_skipped_job(media_id=7),),
        form=LibraryIngestFormState(),
        analysis_action_ready=True,
        analyze_outcomes={"7": (True, "")},
    )
    assert state.analyze_skipped_media_ids == ()
    assert state.show_analyze_skipped is False


def test_analyze_skipped_keeps_an_id_this_action_failed_on():
    """A failed re-attempt still has no analysis -- it stays offered."""
    state = build_library_ingest_state(
        (_skipped_job(media_id=7),),
        form=LibraryIngestFormState(),
        analysis_action_ready=True,
        analyze_outcomes={"7": (False, "analysis did not persist")},
    )
    assert state.analyze_skipped_media_ids == ("7",)


def test_analyze_skipped_action_disabled_while_a_run_is_active():
    state = build_library_ingest_state(
        (_skipped_job(),),
        form=LibraryIngestFormState(),
        analysis_action_ready=True,
        analyze_running=True,
    )
    assert state.show_analyze_skipped is True
    assert state.analyze_skipped_running is True


def test_analyze_outcome_paints_a_success_receipt_on_its_own_row():
    """AC#2: rows ARE individually addressable in the Import canvas -- the
    outcome overlays the row's OWN progress line, replacing the stale
    "analysis skipped: ..." note with the receipt grammar (same glyphs
    Task 2 used on the Media canvas)."""
    job = _skipped_job(media_id=7, source_path="/tmp/notes.txt")
    state = build_library_ingest_state(
        (job,),
        form=LibraryIngestFormState(),
        analyze_outcomes={"7": (True, "")},
    )
    row = state.queue_rows[0]
    assert row.progress is not None
    assert row.progress["message"] == "✓ analyzed · notes.txt"


def test_analyze_outcome_paints_a_failure_receipt_with_its_reason():
    job = _skipped_job(media_id=7, source_path="/tmp/notes.txt")
    state = build_library_ingest_state(
        (job,),
        form=LibraryIngestFormState(),
        analyze_outcomes={"7": (False, "analysis did not persist")},
    )
    row = state.queue_rows[0]
    assert row.progress["message"] == (
        "✗ analysis failed · notes.txt · analysis did not persist"
    )


def test_analyze_outcome_is_a_no_op_for_a_job_without_a_media_id():
    job = _skipped_job(media_id=None)
    state = build_library_ingest_state(
        (job,),
        form=LibraryIngestFormState(),
        analyze_outcomes={"7": (True, "")},
    )
    row = state.queue_rows[0]
    assert "analysis skipped" in row.progress["message"]


# --- task-3304 (MI-17): install commands recoverable at the warning ----------


def _preflight_with_warnings(warnings):
    return PreflightResult(
        type_groups={"audio_video": ["/tmp/talk.mp3"]},
        warnings=warnings,
        errors=[],
        total_size=0,
        truncated=False,
        total_files=1,
    )


def test_state_exposes_deduped_warning_commands_in_order() -> None:
    """The summary's copy affordance needs the commands themselves, not
    just the composed warning prose; duplicates (several features sharing
    one extra) collapse to one button."""
    form = LibraryIngestFormState(path="/tmp/talk.mp3")
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=_preflight_with_warnings(
            [
                {
                    "feature": "faster_whisper",
                    "label": "Faster Whisper",
                    "hint": "audio transcription",
                    "command": 'pip install -e ".[transcription_faster_whisper]"',
                },
                {
                    "feature": "audio_processing",
                    "label": "Audio processing",
                    "hint": "audio ingestion",
                    "command": 'pip install -e ".[audio]"',
                },
                {
                    "feature": "scipy",
                    "label": "SciPy",
                    "hint": "audio ingestion",
                    "command": 'pip install -e ".[audio]"',
                },
                {
                    "feature": "commandless",
                    "label": "No command",
                    "hint": "whatever",
                },
            ]
        ),
    )
    assert state.warning_commands == (
        'pip install -e ".[transcription_faster_whisper]"',
        'pip install -e ".[audio]"',
    )


def test_warning_commands_empty_without_preflight() -> None:
    state = build_library_ingest_state((), form=LibraryIngestFormState())
    assert state.warning_commands == ()


# --- task-3305: copy & labels batch -----------------------------------------


def test_counts_line_drops_in_queue_suffix_when_every_job_is_terminal():
    """(task-3305, MI-14) "1 done — in queue" over a finished run read as a
    contradiction; the queue-scope suffix belongs only while something is
    still actually queued/working.

    (rebase note, task-2859 item 4) The fix landed as an unconditional
    "This queue:" leading label rather than a conditional trailing suffix
    -- it never claims a segment is active, so the check below no longer
    needs to branch on job state at all.
    """
    jobs = (
        _job(job_id="ingest-job-1", state=IngestJobState.DONE),
        _job(job_id="ingest-job-2", state=IngestJobState.FAILED, error="x"),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    assert state.queue_counts_line == "This queue: 1 done · 1 failed"


def test_counts_line_keeps_in_queue_suffix_while_any_job_is_active():
    jobs = (
        _job(job_id="ingest-job-1", state=IngestJobState.DONE),
        _job(job_id="ingest-job-2", state=IngestJobState.QUEUED),
    )
    state = build_library_ingest_state(jobs, form=LibraryIngestFormState())
    assert state.queue_counts_line == "This queue: 1 queued · 1 done"


def test_type_breakdown_names_web_pages():
    """(task-3305, MI-18) A URL selection used to read "1 web" -- the label
    table had no ``web`` entry, so the fallback pluralised the group id."""
    assert build_type_breakdown_line({"web": ["https://a.example"]}) == "1 web page"
    assert (
        build_type_breakdown_line({"web": ["https://a.example", "https://b.example"]})
        == "2 web pages"
    )


def test_intro_line_promises_web_pages():
    from tldw_chatbook.Library.library_ingest_state import build_intro_lines

    what_line = build_intro_lines()[0]
    assert "web pages" in what_line


def test_supported_copy_and_start_gate_name_urls():
    """(task-3305, MI-12) The surface accepts URLs, so the supported list
    and the blank-path nudge must say so."""
    from tldw_chatbook.Library.library_ingest_state import (
        START_QUIET_LINE_COPY,
        SUPPORTED_FORMATS_COPY,
    )

    assert "web pages (by URL)" in SUPPORTED_FORMATS_COPY
    assert START_QUIET_LINE_COPY == "Enter a file path or URL to start."


def test_estimate_line_omitted_for_url_sources():
    """(task-3305, MI-19) A URL is not a 0-byte file: the estimate line
    ("1 file · 0 B") is dropped for URL sources -- the breakdown line
    already names what the URL is."""
    preflight = PreflightResult(
        type_groups={"web": ["https://example.com/article"]},
        warnings=[],
        errors=[],
        total_size=0,
        truncated=False,
        total_files=1,
        source_is_url=True,
    )
    state = build_library_ingest_state(
        (), form=LibraryIngestFormState(path="https://example.com/article"),
        preflight=preflight,
    )
    assert state.estimate_line == ""
    assert state.type_breakdown_line == "1 web page"


def test_failed_row_detail_drops_leading_basename_echo():
    """(task-3305) "✗ failed · empty.txt · empty.txt is empty…" repeated the
    name; the detail drops the echo when it starts with the row's own
    basename."""
    job = _job(
        job_id="ingest-job-9",
        source_path="/tmp/empty.txt",
        state=IngestJobState.FAILED,
        error="empty.txt is empty; there was nothing to ingest.",
    )
    state = build_library_ingest_state((job,), form=LibraryIngestFormState())
    assert state.queue_rows[0].line == (
        "✗ failed · empty.txt · is empty; there was nothing to ingest."
    )


def test_failed_row_detail_without_basename_echo_passes_through():
    job = _job(
        job_id="ingest-job-9",
        source_path="/tmp/broken.pdf",
        state=IngestJobState.FAILED,
        error="PDF Extraction Error.",
    )
    state = build_library_ingest_state((job,), form=LibraryIngestFormState())
    assert state.queue_rows[0].line == (
        "✗ failed · broken.pdf · PDF Extraction Error."
    )


# --- task-3308: .xml defers honestly (owner ruling in task-3310's notes) -----


def test_xml_only_selection_gates_start_with_honest_copy():
    """task-3308: an ``.xml``-only staging closes the Start gate and says
    so in plain terms -- the "XML processing is not yet implemented" raise
    must stay unreachable from the queue."""
    preflight = PreflightResult(
        type_groups={"unsupported": ["/tmp/feed.xml"]},
        warnings=[],
        errors=[],
        total_size=512,
        truncated=False,
        total_files=1,
    )
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp/feed.xml", preflight=preflight),
    )
    assert state.start_enabled is False
    assert state.start_quiet_line == (
        "Nothing in this selection can be imported — 1 unsupported file."
    )
    assert "feed.xml" in state.unsupported_line
    assert "Unsupported" in state.unsupported_line


def test_xml_in_a_mixed_selection_renders_the_will_skip_line():
    """task-3308: alongside importable files, the ``.xml`` is named on the
    will-skip line (task-2220's "skipped, never attempted" ruling) and the
    commit forecast counts it as a skip."""
    preflight = PreflightResult(
        type_groups={
            "generic": ["/tmp/notes.txt"],
            "unsupported": ["/tmp/feed.xml"],
        },
        warnings=[],
        errors=[],
        total_size=1024,
        truncated=False,
        total_files=2,
    )
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp", preflight=preflight),
    )
    assert state.start_enabled is True
    assert state.unsupported_line == (
        "1 unsupported file will be skipped: feed.xml."
    )
    assert "1 will skip" in state.commit_summary_line


# --- task-3307: images join the supported set --------------------------------


def test_breakdown_line_counts_images_task_3307():
    from tldw_chatbook.Library.library_ingest_state import build_type_breakdown_line

    assert build_type_breakdown_line({"image": ["/tmp/a.png"]}) == "1 image"
    assert (
        build_type_breakdown_line({"image": ["/tmp/a.png", "/tmp/b.jpg"]})
        == "2 images"
    )


def test_intro_line_promises_images_task_3307():
    from tldw_chatbook.Library.library_ingest_state import build_intro_lines

    assert "images" in build_intro_lines()[0]


def test_supported_copy_names_images_task_3307():
    from tldw_chatbook.Library.library_ingest_state import SUPPORTED_FORMATS_COPY

    assert "images" in SUPPORTED_FORMATS_COPY


# --- xhigh review round: the egress receipt never named the host ------------


def _egress_raw(origin: str, reason: str = "private") -> str:
    """The exact shape ``EgressBlockedError`` produces (Utils/egress.py)."""
    return (
        f"URL blocked by egress policy (SSRF guard): Egress blocked "
        f"({reason}) for {origin} [remedy: add the host to [web_security] "
        "allowed_hosts in config.toml, or set [web_security] enabled = "
        "false]"
    )


def test_egress_receipts_for_different_hosts_are_distinguishable():
    """task-3312 flattened EVERY egress refusal into one fixed sentence
    that never names the refused address, so a queue of blocked URLs read
    as N copies of the same receipt and the expanded details could not
    recover the host either. Keep the plain-language register; keep the
    host."""
    first = short_ingest_error(_egress_raw("http://127.0.0.1:8000"))
    second = short_ingest_error(_egress_raw("https://internal.example.com"))

    assert first != second, f"both receipts read {first!r}"
    assert "127.0.0.1:8000" in first, first
    assert "internal.example.com" in second, second
    # Still plain language, still bracket-free (the live markup incident).
    for receipt in (first, second):
        assert "SSRF" not in receipt
        assert "[" not in receipt and "\\" not in receipt
        assert "allowed_hosts" in receipt
        assert "web_security" in receipt
        assert receipt.endswith(".")


def test_egress_receipt_without_a_recoverable_host_keeps_the_generic_copy():
    """A refusal whose origin cannot be parsed out (or renders with
    markup-hostile IPv6 brackets) falls back to the host-less sentence
    rather than inventing or leaking one."""
    from tldw_chatbook.Library.library_ingest_state import (
        INGEST_EGRESS_BLOCKED_COPY,
    )

    assert (
        short_ingest_error("Egress blocked (dns_failure) for <invalid-url>")
        == INGEST_EGRESS_BLOCKED_COPY
    )
    assert (
        short_ingest_error(_egress_raw("http://[::1]:8000"))
        == INGEST_EGRESS_BLOCKED_COPY
    )


def test_failed_queue_row_names_the_blocked_host(tmp_path=None):
    """The receipt reaches the queue row (and, via the same helper, Home's
    failed-item line)."""
    job = _job(
        job_id="ingest-job-egress-host",
        source_path="http://127.0.0.1:8000/page",
        state=IngestJobState.FAILED,
        error=_egress_raw("http://127.0.0.1:8000"),
    )
    state = build_library_ingest_state((job,), form=LibraryIngestFormState())
    (row,) = state.queue_rows
    assert "127.0.0.1:8000" in row.line
    assert "SSRF" not in row.line
    assert "[web_security]" not in row.line


# --- task-14820: ONE truthful forecast --------------------------------------


_PDF_WARNING = {
    "feature": "pdf_processing",
    "label": "PDF processing",
    "hint": "PDF ingestion",
    "command": 'pip install -e ".[pdf]"',
}
_DOCLING_WARNING = {
    "feature": "docling",
    "label": "Docling",
    "hint": "richer document extraction",
    "command": 'pip install -e ".[pdf]"',
}


def _mixed_preflight(**overrides) -> PreflightResult:
    defaults = dict(
        type_groups={
            "pdf": ["/tmp/a.pdf", "/tmp/b.pdf"],
            "generic": ["/tmp/notes.txt"],
            "unsupported": ["/tmp/weird.xyz"],
        },
        warnings=[dict(_PDF_WARNING)],
        errors=[],
        total_size=400,
        truncated=False,
        total_files=5,
        empty_files=("/tmp/zero.txt",),
    )
    defaults.update(overrides)
    return PreflightResult(**defaults)


def test_files_needing_missing_required_tooling_forecast_as_failures():
    """(task-14820 AC#2) The pre-flight already warned that PDF processing
    is absent; the commit line used to promise those two PDFs would
    import anyway (``will_import = supported_total - will_match``)."""
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp/folder"),
        preflight=_mixed_preflight(),
    )
    assert state.commit_summary_line == (
        "1 will import · 1 will skip · 3 will fail (2 need tooling, 1 empty)"
    )


def test_forecast_and_consent_line_state_the_same_number():
    """(task-14820 AC#1) Both lines read one forecast object, so they
    cannot disagree: live saw "15 will import" two rows above "7 files
    may fail"."""
    from tldw_chatbook.Library.library_ingest_state import (
        build_ingest_forecast,
    )

    preflight = _mixed_preflight()
    forecast = build_ingest_forecast(preflight)
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp/folder"),
        preflight=preflight,
        start_confirm_armed=True,
    )
    assert forecast.will_fail_tooling == 2
    assert state.start_confirm_armed is True
    assert "2 files will fail without more tooling" in state.start_quiet_line
    # The commit line's own tooling component is the SAME number.
    assert "2 need tooling" in state.commit_summary_line


def test_optional_only_tooling_gap_stays_an_import_but_reads_as_at_risk():
    """A group whose OPTIONAL feature is missing still imports -- the
    consent line says "may fail", never "will fail"."""
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp/folder"),
        preflight=PreflightResult(
            type_groups={"document": ["/tmp/a.docx", "/tmp/b.docx"]},
            warnings=[dict(_DOCLING_WARNING)],
            errors=[],
            total_size=200,
            truncated=False,
            total_files=2,
        ),
        start_confirm_armed=True,
    )
    assert state.commit_summary_line == "2 will import"
    assert "2 files may fail" in state.start_quiet_line
    assert "will fail without" not in state.start_quiet_line


def test_forecast_stays_visible_while_a_gate_blocks_start():
    """(task-14820 AC#4) A blocked user must not lose the numbers they
    were reasoning about. Supersedes task-3305 MI-16's hide-on-block
    rule -- the real defect there was a STALE line, not a visible one."""
    form = LibraryIngestFormState(path="/tmp/folder")
    form.type_options["generic"] = {"chunk": True, "chunk_size": "abc"}
    state = build_library_ingest_state(
        (),
        form=form,
        preflight=PreflightResult(
            type_groups={"generic": ["/tmp/notes.txt"]},
            warnings=[],
            errors=[],
            total_size=10,
            truncated=False,
            total_files=1,
        ),
    )
    assert state.start_enabled is False
    assert "Fix the highlighted options" in state.start_quiet_line
    assert state.commit_summary_line == "1 will import"


def test_forecast_is_silent_without_a_selection_or_under_path_errors():
    """The line still clears with the selection and stays out of the way
    of a path error (the error + its recovery own that state)."""
    empty = build_library_ingest_state((), form=LibraryIngestFormState())
    assert empty.commit_summary_line == ""
    errored = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp/nope"),
        preflight=PreflightResult(
            type_groups={},
            warnings=[],
            errors=["Path not found: /tmp/nope"],
            total_size=0,
            truncated=False,
            total_files=0,
            path_invalid=True,
        ),
    )
    assert errored.commit_summary_line == ""


# --- task-14823: gate a selection with nothing importable -------------------


def test_empty_folder_gates_start_with_its_own_reason():
    """(task-14823) An empty directory left Start ENABLED with an empty
    gate line; pressing it manufactured "✗ failed · emptydir · No files
    to import were found in this folder."."""
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp/emptydir"),
        preflight=PreflightResult(
            type_groups={},
            warnings=[],
            errors=[],
            total_size=0,
            truncated=False,
            total_files=0,
        ),
    )
    assert state.start_enabled is False
    assert state.start_quiet_line == (
        "This folder is empty — there's nothing to import. Choose a folder "
        "with files, or a single file."
    )


def test_all_unsupported_folder_keeps_its_own_distinct_reason():
    """(task-14823 AC#2) The recovery differs from an empty folder, so
    the reason must too."""
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp/folder"),
        preflight=PreflightResult(
            type_groups={"unsupported": ["/tmp/a.xyz", "/tmp/b.xyz"]},
            warnings=[],
            errors=[],
            total_size=20,
            truncated=False,
            total_files=2,
        ),
    )
    assert state.start_enabled is False
    assert state.start_quiet_line == (
        "Nothing in this selection can be imported — 2 unsupported files."
    )


# --- task-14821: retry advice derives from the failure reason ---------------


_OCR_MESSAGE = (
    "No text was found in diagram.png. An image import stores the text OCR "
    "extracts; turn Extract text (OCR) on and install an OCR backend "
    "(docling, tesseract, easyocr, paddleocr, or docext)."
)


def _expanded_detail_lines(**error_detail) -> tuple[str, ...]:
    job = _job(
        job_id="ingest-job-1",
        state=IngestJobState.FAILED,
        source_path=error_detail.pop("source_path", "/tmp/diagram.png"),
        error=error_detail.pop("error", _OCR_MESSAGE),
        error_detail=dict(error_detail),
    )
    state = build_library_ingest_state(
        (job,),
        form=LibraryIngestFormState(),
        expanded_details={"ingest-job-1"},
    )
    return state.queue_rows[0].detail_lines


def test_no_content_failure_is_never_described_as_transient():
    """(task-14821 AC#1) A missing-OCR failure is deterministic. Its
    message names an install remedy the ``_MISSING_DEPENDENCY_RE`` does
    not match, so the optimistic ELSE branch fired -- on the COMMON case."""
    lines = _expanded_detail_lines(
        category="no_content",
        message=_OCR_MESSAGE,
        exception_type="FileIngestionError",
    )
    joined = " ".join(lines)
    assert "transient" not in joined
    assert "network hiccup" not in joined
    assert "install the tooling named above" in joined


def test_no_content_reason_is_user_readable_and_not_a_write_error():
    """(task-14821 AC#3) Nothing was written -- extraction produced no
    content -- and the line must not show the raw internal token."""
    lines = _expanded_detail_lines(
        category="no_content",
        message=_OCR_MESSAGE,
        exception_type="FileIngestionError",
    )
    assert lines[0] == "Reason: No text could be extracted."
    assert not any("write error" in line for line in lines)
    assert not any(line.startswith("Category:") for line in lines)


def test_unknown_failure_category_stays_silent_rather_than_encouraging():
    """(task-14821 AC#2) The unknown case gets no advice at all."""
    lines = _expanded_detail_lines(
        category="something_new",
        message="Ingest stopped for an unrecognised reason.",
        exception_type="RuntimeError",
        error="Ingest stopped for an unrecognised reason (row).",
    )
    joined = " ".join(lines)
    assert "Retry" not in joined
    assert "transient" not in joined


def test_write_error_keeps_the_one_genuinely_retryable_advisory():
    """A real database write failure IS worth retrying as-is."""
    lines = _expanded_detail_lines(
        category="write_error",
        message="database is locked",
        exception_type="MediaDatabaseError",
        error="Failed to ingest txt file: database is locked",
    )
    joined = " ".join(lines)
    assert "A retry can succeed" in joined
    assert "network hiccup" not in joined


def test_underlying_tool_output_is_printed_once_not_twice():
    """(task-14821 AC#4) The real ffmpeg failure: the row message carries
    a "Failed to ingest audio file: " wrapper the chain entry lacks, so
    the exact-equality dedup missed it and the ~40-line banner printed
    under BOTH Details and Underlying."""
    banner = (
        "ffmpeg version 8.1 Copyright (c) 2000-2026 the FFmpeg developers\n"
        "  built with Apple clang version 17.0.0\n"
        "  configuration: --prefix=/opt/homebrew/Cellar/ffmpeg/8.1\n"
        "  libavutil      60. 26.100 / 60. 26.100\n"
        "Error opening input files: Invalid data found when processing input"
    )
    inner = f"Audio processing failed: FFmpeg conversion failed: {banner}"
    lines = _expanded_detail_lines(
        source_path="/tmp/song.mp3",
        category="parse_error",
        message=f"Failed to ingest audio file: {inner}",
        exception_type="FileIngestionError",
        chain=[f"FileIngestionError: {inner}"],
        # ``job.error`` is the sanitized single line (app.py caps at 200).
        error="Failed to ingest audio file: Audio processing failed: FFmpeg "
        "conversion failed: ffmpeg version 8.1 Copyright (c) 2000-2026 the "
        "FFmpeg developers",
    )
    banner_lines = [line for line in lines if "libavutil" in line]
    assert len(banner_lines) == 1, (
        f"the tool banner appeared {len(banner_lines)} times: {lines}"
    )


def test_real_canvas_state_carries_the_forecast_the_fold_reads():
    """(task-14820/14822 seam) The folded tooling summary reads
    ``state.forecast.consent_affected``/``.staged_total``. Every unit test
    of that line stubs the state, so nothing proved the REAL builder
    supplies those fields -- and while it didn't, the fold silently
    rendered its degraded "N optional components aren't installed"
    fallback instead of the file count it exists to show. This drives the
    real builder and then the real render function.
    """
    from tldw_chatbook.Widgets.Library.library_ingest_canvas import (
        ingest_tooling_summary_line,
    )

    preflight = PreflightResult(
        type_groups={
            "pdf": ["/tmp/a.pdf", "/tmp/b.pdf"],
            "generic": ["/tmp/c.txt"],
        },
        # A missing REQUIRED pdf feature: both PDFs are doomed, and the
        # pre-flight's own warning is what the forecast keys off.
        warnings=[{"feature": "pdf_processing", "label": "PDF processing"}],
        errors=[],
        total_size=3072,
        truncated=False,
        total_files=3,
    )
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp/mixed", preflight=preflight),
    )

    assert state.forecast is not None, (
        "the canvas state must carry the forecast; without it the fold has "
        "no honest file count"
    )
    assert state.forecast.staged_total == 3, state.forecast
    assert state.forecast.consent_affected == 2, state.forecast

    line = ingest_tooling_summary_line(state)
    # The seam under test is the DATA the fold reads, not its wording (the
    # canvas owns that and states required/optional gaps differently), so
    # assert the file scope it can only produce from a real forecast.
    assert "2 of 3 files" in line, line
    assert "components" not in line, (
        f"the fold fell back to counting components, not files: {line}"
    )


# ===========================================================================
# xhigh review round of the 14820-14826 arc: the regressions the arc shipped
# ===========================================================================


def _audio_preflight(count: int = 5) -> PreflightResult:
    """A server-shaped selection: N .mp3 with NO local audio tooling.

    ``audio_processing`` is the ``audio_video`` group's REQUIRED feature,
    so on this install the pre-flight always warns about it.
    """
    files = [f"/tmp/talk{index}.mp3" for index in range(count)]
    return PreflightResult(
        type_groups={"audio_video": files},
        warnings=[
            {
                "feature": "audio_processing",
                "label": "Audio processing",
                "hint": "audio transcription",
                "command": 'pip install -e ".[audio]"',
            }
        ],
        errors=[],
        total_size=count * 4096,
        truncated=False,
        total_files=count,
    )


def test_local_forecast_still_counts_missing_local_tooling_as_failures():
    """Guard: the LOCAL forecast is what task-14820 fixed — keep it."""
    from tldw_chatbook.Library.library_ingest_state import build_ingest_forecast

    forecast = build_ingest_forecast(_audio_preflight())
    assert forecast is not None
    assert (forecast.will_import, forecast.will_fail_tooling) == (0, 5)
    assert forecast.tooling_groups == ("audio_video",)


def test_server_forecast_does_not_subtract_local_tooling_gaps():
    """(xhigh F1) A server import never touches a local parser.

    ``build_ingest_forecast`` subtracted LOCAL tooling gaps unconditionally,
    so server mode + 5 .mp3 + no local audio extra forecast "0 will import ·
    5 will fail (need tooling)" for a run the server would transcribe in
    full. The deleted ``will_import = supported_total - will_match`` was at
    least backend-agnostic; this regression is worse than the defect it
    replaced for every server user.
    """
    from tldw_chatbook.Library.library_ingest_state import build_ingest_forecast

    forecast = build_ingest_forecast(_audio_preflight(), targets_server=True)
    assert forecast is not None
    assert forecast.will_fail_tooling == 0, (
        "a LOCAL tooling gap was forecast as a certain failure of a SERVER "
        "run that never loads a local parser"
    )
    assert forecast.will_import == 5
    assert forecast.tooling_groups == ()
    assert forecast.at_risk == 0
    assert forecast.consent_affected == 0


def test_server_forecast_line_claims_only_what_it_can_know():
    """(xhigh F1) The server's own capabilities are not knowable from here
    (task-3309 is open precisely because forwarded extras are unverified),
    so the line states what WILL happen — the files are sent — and says
    outright that the server's tooling was not checked."""
    from tldw_chatbook.Library.library_ingest_state import (
        build_ingest_forecast,
        forecast_summary_line,
    )

    line = forecast_summary_line(
        build_ingest_forecast(_audio_preflight(), targets_server=True)
    )
    assert "will fail" not in line, line
    assert line == (
        "5 will be sent to the server · server tooling isn't checked "
        "from here"
    ), line


def _server_state(*, armed: bool = False, count: int = 5):
    preflight = _audio_preflight(count)
    return build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp/podcasts", preflight=preflight),
        runtime_source="server",
        ingest_backend="server",
        server_ingest_available=True,
        start_confirm_armed=armed,
    )


def test_server_mode_state_stops_forecasting_certain_local_failures():
    """(xhigh F1) End to end through the real builder: ``targets_server``
    is computed there and must reach the forecast."""
    state = _server_state()
    assert state.ingest_backend == "server"
    assert state.forecast is not None
    assert state.forecast.will_fail_tooling == 0
    assert "need tooling" not in state.commit_summary_line
    assert "will fail" not in state.commit_summary_line
    assert "5 will be sent to the server" in state.commit_summary_line


def test_server_mode_never_arms_consent_for_local_only_warnings():
    """(xhigh F1) The consent line's blast radius is the forecast's. With
    nothing at stake locally there is nothing to consent to, so the armed
    flag must not paint a reasonless "import anyway"."""
    state = _server_state(armed=True)
    assert state.start_confirm_armed is False
    assert "Press Start again" not in state.start_quiet_line


def test_local_mode_state_keeps_the_tooling_failure_forecast():
    """Guard for the same builder path in local mode."""
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(
            path="/tmp/podcasts", preflight=_audio_preflight()
        ),
    )
    assert state.ingest_backend == "local"
    assert "5 will fail (need tooling)" in state.commit_summary_line


# --- F8: a capped duplicate probe makes the import count an upper bound -----


def test_capped_duplicate_probe_hedges_the_import_count_too():
    """(xhigh F8) ``will_import = supported - will_match``: when
    ``will_match`` is a capped FLOOR the import count is a CEILING, and
    stating it exactly beside "at least 20 will match" is a contradiction
    the user can do arithmetic on."""
    from tldw_chatbook.Library.library_ingest_state import (
        build_ingest_forecast,
        forecast_summary_line,
    )

    preflight = PreflightResult(
        type_groups={"generic": [f"/tmp/n{i}.txt" for i in range(25)]},
        warnings=[],
        errors=[],
        total_size=25 * 100,
        truncated=False,
        total_files=25,
        already_in_library=20,
        already_in_library_capped=True,
    )
    forecast = build_ingest_forecast(preflight)
    assert (forecast.will_import, forecast.will_match) == (5, 20)
    assert forecast_summary_line(forecast) == (
        "at most 5 will import · at least 20 will match"
    )


def test_uncapped_duplicate_probe_still_states_both_counts_exactly():
    """Guard: the hedge is carried only when the probe was capped."""
    from tldw_chatbook.Library.library_ingest_state import (
        build_ingest_forecast,
        forecast_summary_line,
    )

    preflight = PreflightResult(
        type_groups={"generic": [f"/tmp/n{i}.txt" for i in range(25)]},
        warnings=[],
        errors=[],
        total_size=2500,
        truncated=False,
        total_files=25,
        already_in_library=20,
    )
    assert forecast_summary_line(build_ingest_forecast(preflight)) == (
        "5 will import · 20 will match"
    )


# --- F7: the forecast must not promise imports a dead runtime cannot make ---


def _one_text_file_preflight(path: str = "/tmp/report.txt") -> PreflightResult:
    return PreflightResult(
        type_groups={"generic": [path]},
        warnings=[],
        errors=[],
        total_size=11,
        truncated=False,
        total_files=1,
    )


@pytest.mark.parametrize(
    "seam",
    [
        {"registry_available": False},
        {"media_db_available": False},
    ],
    ids=["no-registry", "no-media-db"],
)
def test_forecast_is_withheld_when_the_runtime_cannot_import_at_all(seam):
    """(xhigh F7) Un-gating the commit line (task-14820 AC#4) also un-gated
    it for the seam-missing case, so "1 will import" rendered beside a
    Start that can never run. AC#4 is about a BLOCKED user keeping their
    numbers; a runtime with no import path at all is not that case."""
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(
            path="/tmp/report.txt", preflight=_one_text_file_preflight()
        ),
        **seam,
    )
    assert state.unavailable_line
    assert state.commit_summary_line == "", (
        "the forecast promised an import the runtime cannot perform: "
        f"{state.commit_summary_line!r}"
    )


def test_option_error_gating_keeps_the_forecast_visible_task_14820_ac4():
    """(task-14820 AC#4, preserved) The selection is still real — only the
    OPTIONS are wrong — so the blocked user keeps their numbers."""
    form = LibraryIngestFormState(
        path="/tmp/report.txt", preflight=_one_text_file_preflight()
    )
    form.type_options = {"generic": {"chunk_size": "abc"}}
    state = build_library_ingest_state((), form=form)
    assert state.start_enabled is False
    assert state.option_errors
    assert "1 will import" in state.commit_summary_line


# --- F5: "this folder is empty" must not be said about folders that aren't --


def test_a_folder_whose_entries_are_all_skipped_is_not_called_empty(tmp_path):
    """(xhigh F5) ``_collect_files`` skips symlinks and dot-entries, so a
    folder full of symlinked media pre-flights as ``total_files == 0`` —
    which task-14823 words as "This folder is empty" AND (since its new
    submit gate) hard-blocks. The diagnosis is false and the block turns it
    into a dead end."""
    from tldw_chatbook.Library.ingest_preflight import (
        analyze_path,
        collect_directory_files,
    )

    target = tmp_path / "real.txt"
    target.write_text("hello world")
    folder = tmp_path / "links"
    folder.mkdir()
    (folder / "linked.txt").symlink_to(target)
    (folder / ".hidden.txt").write_text("hidden")

    result = analyze_path(str(folder))
    assert result.total_files == 0
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path=str(folder), preflight=result),
    )

    assert "This folder is empty" not in state.start_quiet_line, (
        f"a folder holding 2 entries was called empty: "
        f"{state.start_quiet_line!r}"
    )
    assert "2 entries" in state.start_quiet_line, state.start_quiet_line
    # The gate is honest only because the SUBMIT path walks the folder with
    # the very same collector -- it would queue nothing either.
    files, _truncated = collect_directory_files(folder, 1000)
    assert files == []
    assert state.selection_has_nothing_importable is True


def test_a_genuinely_empty_folder_keeps_the_empty_sentence(tmp_path):
    """Guard: the empty-folder recovery ("put files in it") is different
    from the skipped-entries one, so the two sentences stay distinct."""
    from tldw_chatbook.Library.ingest_preflight import analyze_path

    folder = tmp_path / "nothing"
    folder.mkdir()
    result = analyze_path(str(folder))
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path=str(folder), preflight=result),
    )
    assert "This folder is empty" in state.start_quiet_line
    assert state.selection_has_nothing_importable is True


# --- F2/F3/F9/F10: retry advice that is true of THIS failure -----------------


def test_a_transient_pool_teardown_is_not_told_it_will_always_fail():
    """(xhigh F2) ``_TOOLING_REMEDY_RE``'s ``is (?:not|un)available``
    alternative matched ``TranscriptionError("The shared local executor is
    unavailable.")`` — a pool-teardown that clears on the next attempt —
    and answered it with "Retrying now will fail the same way — install the
    tooling named above first", naming no tooling anywhere."""
    message = (
        "Failed to ingest audio file: The shared local executor is "
        "unavailable."
    )
    lines = _expanded_detail_lines(
        source_path="/tmp/talk.mp3",
        category="parse_error",
        message=message,
        exception_type="TranscriptionError",
        error=message,
    )
    joined = " ".join(lines)
    assert "install the tooling named above" not in joined, joined
    assert "will fail the same way" not in joined, joined


def test_the_generic_no_content_advice_names_no_phantom_remedy():
    """(xhigh F3) The generic extraction refusal names no dependency, yet
    the deterministic branch told the user to "install the tooling named
    above" — there is nothing above."""
    message = (
        "No text could be extracted from doc.pdf. The pdf content may be "
        "scanned images, or the tooling for this file type may not be "
        "installed."
    )
    lines = _expanded_detail_lines(
        source_path="/tmp/doc.pdf",
        category="no_content",
        message=message,
        exception_type="FileIngestionError",
        error=message,
    )
    joined = " ".join(lines)
    assert "named above" not in joined, joined
    # ...but the DETERMINISM is still stated: this is not a retry-and-hope.
    assert "transient" not in joined
    assert "Retrying" in joined, joined


def test_a_remedy_that_names_its_tooling_still_says_install_it():
    """Guard (task-14821): the OCR refusal DOES name its backends."""
    lines = _expanded_detail_lines(
        category="no_content",
        message=_OCR_MESSAGE,
        exception_type="FileIngestionError",
    )
    assert "install the tooling named above" in " ".join(lines)


def test_a_tooling_remedy_carried_only_by_the_chain_is_honoured():
    """(xhigh F9) A real pdf failure on an install without pdf tooling
    reports ``'NoneType' object has no attribute 'FileDataError'`` as its
    MESSAGE and carries the remedy two links down the chain. The advice
    must read the same text the user reads."""
    lines = _expanded_detail_lines(
        source_path="/tmp/scan.png",
        category="parse_error",
        message="Failed to ingest image file: image extraction produced nothing.",
        exception_type="FileIngestionError",
        chain=[
            "NoContentExtractedError: No text was found in scan.png. An image "
            "import stores the text OCR extracts; turn Extract text (OCR) on "
            "and install an OCR backend (docling, tesseract, easyocr, "
            "paddleocr, or docext)."
        ],
        error="Failed to ingest image file: image extraction produced nothing.",
    )
    assert "install the tooling named above" in " ".join(lines), lines


def test_missing_dependency_name_does_not_keep_the_sentence_period():
    """(xhigh F10, live) ``pip install (\\S+)`` swallowed the sentence's
    full stop, and the template added another: "Missing dependency:
    tldw_chatbook[pdf]..". Verbatim shape of the real chain captured from
    ``run_parse_job`` on this install."""
    message = (
        "Failed to ingest pdf file: 'NoneType' object has no attribute "
        "'FileDataError'"
    )
    lines = _expanded_detail_lines(
        source_path="/tmp/doc.pdf",
        category="parse_error",
        message=message,
        exception_type="FileIngestionError",
        chain=[
            "AttributeError: 'NoneType' object has no attribute 'FileDataError'",
            "ImportError: PDF processing libraries not available. Install "
            "with: pip install tldw_chatbook[pdf]. Error: No module named "
            "'pymupdf'",
        ],
        error=message,
    )
    advice = [line for line in lines if line.startswith("Missing dependency")]
    assert advice == [
        "Missing dependency: tldw_chatbook[pdf]. Install it, then Retry."
    ], advice


# --- F6: the chain dedup must keep what ADDS to the summary -----------------


def test_a_chain_entry_that_adds_the_root_cause_survives_the_dedup():
    """(xhigh F6) ``_restates_known_text``'s ``text in candidate``
    direction dropped every chain entry that quotes the row summary AND
    appends the underlying cause — exactly the entry the chain exists to
    surface."""
    summary = "PDF Extraction Error."
    lines = _expanded_detail_lines(
        source_path="/tmp/doc.pdf",
        category="parse_error",
        message=summary,
        exception_type="FileIngestionError",
        chain=[
            "ImportError: PDF Extraction Error. Caused by: no pdf backend "
            "could be loaded"
        ],
        error=summary,
    )
    underlying = [line for line in lines if line.startswith("Underlying:")]
    assert underlying == [
        "Underlying: ImportError: PDF Extraction Error. Caused by: no pdf "
        "backend could be loaded"
    ], lines


def test_a_chain_entry_that_only_restates_the_summary_is_still_dropped():
    """Guard: the duplicate-banner fix (task-14821 AC#4) stays fixed.

    The real ffmpeg shape: the row message carries a "Failed to ingest
    <type> file: " wrapper the chain entry lacks, so only containment (not
    equality) sees that the two say the same thing.
    """
    summary = "Failed to ingest pdf file: PDF Extraction Error."
    lines = _expanded_detail_lines(
        source_path="/tmp/doc.pdf",
        category="parse_error",
        message=summary,
        exception_type="FileIngestionError",
        chain=["FileIngestionError: PDF Extraction Error."],
        error=summary,
    )
    assert not [line for line in lines if line.startswith("Underlying:")], lines


# ===========================================================================
# task-14827: the SERVER path refuses a different set than the LOCAL one
# ===========================================================================


def _mixed_server_preflight() -> PreflightResult:
    """One file per fate on the SERVER path.

    ``notes.txt`` maps to the server's ``document`` type; ``diagram.png``
    has no server media type at all (task-3307 deliberately left images
    server-unmapped); ``weird.xyz`` is unclassifiable anywhere.
    """
    return PreflightResult(
        type_groups={
            "generic": ["/tmp/mixed/notes.txt"],
            "image": ["/tmp/mixed/diagram.png"],
            "unsupported": ["/tmp/mixed/weird.xyz"],
        },
        warnings=[],
        errors=[],
        total_size=3 * 128,
        truncated=False,
        total_files=3,
    )


def test_server_forecast_counts_a_refused_file_as_a_failure_not_a_skip():
    """(task-14827 AC#1) ``build_server_ingest_kwargs`` raises
    ``ServerIngestUnsupported`` for both of these, and
    ``_submit_server_ingest_job`` turns that into an immediately FAILED,
    permanent row -- so forecasting them as "will skip" contradicted the
    receipt the same way task-14820 existed to stop."""
    from tldw_chatbook.Library.library_ingest_state import build_ingest_forecast

    forecast = build_ingest_forecast(
        _mixed_server_preflight(), targets_server=True
    )
    assert forecast is not None
    assert forecast.will_skip == 0, (
        "the server path skips nothing -- every source it cannot map is "
        "failed with a reason"
    )
    assert forecast.will_fail_refused == 2
    assert forecast.will_fail == 2
    assert forecast.will_import == 1


def test_server_forecast_line_says_the_server_refused_them_not_that_nothing_can():
    """(task-14827 AC#1) A file the SERVER will not take is not a file
    nothing can read: ``diagram.png`` imports fine locally with an OCR
    backend. The copy must name the backend that is refusing."""
    from tldw_chatbook.Library.library_ingest_state import (
        build_ingest_forecast,
        forecast_summary_line,
    )

    line = forecast_summary_line(
        build_ingest_forecast(_mixed_server_preflight(), targets_server=True)
    )
    assert line == (
        "1 will be sent to the server · 2 will fail (unsupported by the "
        "server) · server tooling isn't checked from here"
    ), line
    assert "will skip" not in line, line


def test_local_forecast_still_skips_what_only_the_server_refuses():
    """Guard: the two backends refuse DIFFERENT sets, so the local
    forecast must not inherit the server's verdict. Locally the image is
    an import candidate and only the unrecognised extension is skipped."""
    from tldw_chatbook.Library.library_ingest_state import (
        build_ingest_forecast,
        forecast_summary_line,
    )

    forecast = build_ingest_forecast(_mixed_server_preflight())
    assert forecast is not None
    assert (forecast.will_import, forecast.will_skip) == (2, 1)
    assert forecast.will_fail_refused == 0
    assert forecast_summary_line(forecast) == "2 will import · 1 will skip"


def test_server_forecast_counts_a_page_url_as_sent_because_it_goes_to_the_clipper():
    """A page has no ingest-jobs media type, but ``submit_library_ingest_job``
    routes it to the web clipper before ``build_server_ingest_kwargs`` is
    ever asked -- so it is sent, not refused. A forecast that consulted
    ``server_media_type_for`` alone would condemn every server-mode URL
    import."""
    from tldw_chatbook.Library.library_ingest_state import build_ingest_forecast

    preflight = PreflightResult(
        type_groups={"web": ["https://example.com/post"]},
        warnings=[],
        errors=[],
        total_size=0,
        truncated=False,
        total_files=1,
        source_is_url=True,
    )
    forecast = build_ingest_forecast(preflight, targets_server=True)
    assert forecast is not None
    assert (forecast.will_import, forecast.will_fail_refused) == (1, 0)


def test_server_mode_names_refused_files_as_failing_not_skipped():
    """(task-14827 AC#1) The named-files line is part of the forecast the
    user reads. "will be skipped" is the local pipeline's promise; the
    server records a failure row for the same file."""
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(
            path="/tmp/mixed", preflight=_mixed_server_preflight()
        ),
        runtime_source="server",
        ingest_backend="server",
        server_ingest_available=True,
    )
    assert state.ingest_backend == "server"
    assert "skipped" not in state.unsupported_line, state.unsupported_line
    assert state.unsupported_line == (
        "1 unsupported file will fail: weird.xyz."
    ), state.unsupported_line


def test_local_mode_keeps_the_skipped_wording_for_unsupported_files():
    """Guard for the local half of the same line (task-2220 owner ruling:
    skipped, not "recorded as failures" -- the pipeline never attempts
    these)."""
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(
            path="/tmp/mixed", preflight=_mixed_server_preflight()
        ),
    )
    assert state.unsupported_line == (
        "1 unsupported file will be skipped: weird.xyz."
    ), state.unsupported_line


def test_server_mode_stops_presenting_local_tooling_gaps_as_blockers():
    """(task-14827 AC#3) The tooling wall, its ⚠ summary and its "Copy
    install command" button all describe THIS machine's inventory. During
    a server-targeted import that machine does no work, so installing
    those extras changes nothing about the run -- the wall becomes one
    quiet line that says so."""
    state = _server_state()
    assert state.warning_lines == [], state.warning_lines
    assert state.warning_commands == (), state.warning_commands
    assert state.advisory_lines == (
        "1 local component isn't installed — that affects imports on this "
        "machine only; this one runs on the server.",
    ), state.advisory_lines


def test_local_mode_keeps_the_tooling_wall():
    """Guard: the same selection on the local backend still gets the full
    warning wall and its install command."""
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(
            path="/tmp/podcasts", preflight=_audio_preflight()
        ),
    )
    assert state.warning_lines
    assert state.warning_commands == ('pip install -e ".[audio]"',)
    assert state.advisory_lines == ()


# ===========================================================================
# task-14911: the Start gate must ask the backend the run is aimed at
# ===========================================================================


def _images_only_preflight(count: int = 3) -> PreflightResult:
    """A folder the LOCAL pipeline can import and the SERVER cannot.

    Images are a real local capability (the ``image`` group, OCR) that
    task-3307 deliberately left server-unmapped, so this is the selection
    on which the two backends' verdicts diverge completely.
    """
    files = [f"/tmp/shots/photo{i}.png" for i in range(count)]
    return PreflightResult(
        type_groups={"image": files},
        warnings=[],
        errors=[],
        total_size=count * 2048,
        truncated=False,
        total_files=count,
    )


def _server_gate_state(preflight: PreflightResult):
    return build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp/shots", preflight=preflight),
        runtime_source="server",
        ingest_backend="server",
        server_ingest_available=True,
    )


def test_server_mode_gates_start_when_the_server_refuses_everything():
    """(task-14911 AC#1) task-14823's gate asked a LOCAL question -- "did
    the pre-flight find a supported type group" -- so a folder of nothing
    but images forecast "0 will be sent to the server · 3 will fail
    (unsupported by the server)" with Start still ENABLED: the guaranteed
    failure submit that gate exists to prevent, one backend over."""
    state = _server_gate_state(_images_only_preflight())

    assert state.ingest_backend == "server"
    assert state.forecast is not None
    assert (state.forecast.will_import, state.forecast.will_fail_refused) == (
        0,
        3,
    )
    assert state.start_enabled is False, (
        "Start stayed live for a selection the server refuses entirely: "
        f"{state.commit_summary_line!r}"
    )
    assert state.selection_has_nothing_importable is True
    assert state.start_quiet_line, "the gate closed without stating why"


def test_the_server_gate_names_the_backend_that_is_refusing():
    """(task-14911 AC#1) A file nothing can read and a file this server
    won't take need different sentences: the image imports fine on this
    machine. The arc already settled the vocabulary for the second case
    ("unsupported by the server"), and the recovery differs too -- switch
    the target, or stage something the server accepts."""
    from tldw_chatbook.Library.library_ingest_state import (
        INGEST_SERVER_REFUSED_COPY,
    )

    line = _server_gate_state(_images_only_preflight()).start_quiet_line

    assert "sent to the server" in line, line
    assert INGEST_SERVER_REFUSED_COPY in line, line
    assert "can be imported" not in line, (
        "the local gate sentence claims nothing can read these files; they "
        f"import fine on this machine: {line!r}"
    )
    assert "on this machine" in line, (
        f"the gate stated no way forward: {line!r}"
    )


def test_the_same_selection_is_untouched_in_local_mode():
    """(task-14911 AC#2) The images import on this machine, so nothing
    about this selection is doomed locally -- the gate must stay open."""
    state = build_library_ingest_state(
        (),
        form=LibraryIngestFormState(
            path="/tmp/shots", preflight=_images_only_preflight()
        ),
    )

    assert state.ingest_backend == "local"
    assert state.start_enabled is True
    assert state.selection_has_nothing_importable is False
    assert state.start_quiet_line == "", state.start_quiet_line


def test_the_server_gate_counts_come_from_the_forecast():
    """(task-14911 AC#3) The gate reads the existing ``IngestForecast``
    rather than deriving a second notion of what is importable -- the
    same "one computation" move task-14820 made for the commit and
    consent lines. If it re-derived, the two lines could disagree on
    screen, which is the defect that arc exists to remove."""
    state = _server_gate_state(_images_only_preflight(count=7))

    assert state.forecast is not None
    assert f"{state.forecast.will_fail_refused} files" in (
        state.start_quiet_line
    ), state.start_quiet_line
    assert f"{state.forecast.will_fail}" in state.commit_summary_line


def test_the_server_gate_names_empty_files_separately():
    """A 0-byte file is not "unsupported by the server" (task-14910: the
    client refuses it before sending), so a selection blocked by both
    states both reasons."""
    preflight = PreflightResult(
        type_groups={"image": ["/tmp/shots/photo0.png"]},
        warnings=[],
        errors=[],
        total_size=2048,
        truncated=False,
        total_files=2,
        empty_files=("/tmp/shots/blank.txt",),
    )
    state = _server_gate_state(preflight)

    assert state.start_enabled is False
    line = state.start_quiet_line
    assert "1 file unsupported by the server" in line, line
    assert "1 empty file" in line, line


def test_a_server_selection_with_one_sendable_file_is_not_gated():
    """Guard: the gate is "the server will take NOTHING here", not "the
    server will refuse something here" -- a mixed folder still starts."""
    preflight = PreflightResult(
        type_groups={
            "generic": ["/tmp/shots/notes.txt"],
            "image": ["/tmp/shots/photo0.png"],
        },
        warnings=[],
        errors=[],
        total_size=4096,
        truncated=False,
        total_files=2,
    )
    state = _server_gate_state(preflight)

    assert state.forecast is not None
    assert state.forecast.will_import == 1
    assert state.start_enabled is True
    assert state.selection_has_nothing_importable is False


def test_a_server_selection_that_is_all_duplicates_still_starts():
    """Guard (task-2223 ruling, preserved): zero imports plus predicted
    matches keeps Start ENABLED -- the duplicate probe is capped
    best-effort, never a blocker -- so the new gate must key off "the
    backend accepts nothing", not "will_import == 0"."""
    preflight = PreflightResult(
        type_groups={"generic": [f"/tmp/dupes/n{i}.txt" for i in range(3)]},
        warnings=[],
        errors=[],
        total_size=3 * 128,
        truncated=False,
        total_files=3,
        already_in_library=3,
    )
    state = _server_gate_state(preflight)

    assert state.forecast is not None
    assert (state.forecast.will_import, state.forecast.will_match) == (0, 3)
    assert state.start_enabled is True
    assert state.selection_has_nothing_importable is False


def test_an_all_unsupported_folder_keeps_the_local_sentence_on_both_backends():
    """Guard: a file nothing can read is diagnosed the same way whichever
    target is selected -- switching to the server would not help, so the
    server's own vocabulary would be misleading here."""
    preflight = PreflightResult(
        type_groups={"unsupported": ["/tmp/junk/a.xyz", "/tmp/junk/b.xyz"]},
        warnings=[],
        errors=[],
        total_size=64,
        truncated=False,
        total_files=2,
    )
    state = _server_gate_state(preflight)

    assert state.start_enabled is False
    assert state.start_quiet_line == (
        "Nothing in this selection can be imported — 2 unsupported files."
    ), state.start_quiet_line
