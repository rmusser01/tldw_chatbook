"""Pure display-state contracts for the Library media viewer canvas."""

from __future__ import annotations

from datetime import datetime, timezone

from tldw_chatbook.Library.library_media_viewer_state import (
    LibraryMediaViewerState,
    build_library_media_viewer_state,
)

NOW = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)


def test_full_detail_builds_all_metadata_lines_in_order():
    """All metadata sources present -> ordered Type/Author/URL/Keywords/Ingested lines."""
    detail = {
        "media_id": "media-1",
        "title": "Alpha Video",
        "type": "video",
        "author": "A. Author",
        "url": "http://example.com/alpha",
        "keywords": ["a", "b"],
        "ingestion_date": "2026-07-06T10:00:00+00:00",  # 2h old
        "content": "full transcript text",
        "analysis_content": "summary text",
        "version": 3,
    }

    state = build_library_media_viewer_state(detail, now=NOW)

    assert isinstance(state, LibraryMediaViewerState)
    assert state.media_id == "media-1"
    assert state.title == "Alpha Video"
    assert state.metadata_lines == (
        "Type: video",
        "Author: A. Author",
        "URL: http://example.com/alpha",
        "Keywords: a, b",
        "Ingested: 2h",
    )
    assert state.content == "full transcript text"
    assert state.analysis == "summary text"
    assert state.has_content is True
    assert state.has_analysis is True
    assert state.version == 3
    assert state.edit_fields == {
        "title": "Alpha Video",
        "author": "A. Author",
        "url": "http://example.com/alpha",
        "keywords": "a, b",
    }


def test_media_type_key_fallback():
    """Falls back to media_type when type is absent."""
    detail = {"media_id": "1", "title": "T", "media_type": "pdf"}

    state = build_library_media_viewer_state(detail, now=NOW)

    assert state.metadata_lines == ("Type: pdf",)


def test_missing_type_uses_unknown_fallback_and_line_always_present():
    """Type line is always present; falls back to 'unknown' when missing."""
    detail = {"media_id": "1", "title": "No Type"}

    state = build_library_media_viewer_state(detail, now=NOW)

    assert state.metadata_lines == ("Type: unknown",)


def test_author_url_keywords_omitted_when_absent():
    """Author/URL/Keywords lines are omitted entirely when their source is absent."""
    detail = {"media_id": "1", "title": "Sparse", "type": "video"}

    state = build_library_media_viewer_state(detail, now=NOW)

    assert state.metadata_lines == ("Type: video",)
    assert state.edit_fields == {
        "title": "Sparse",
        "author": "",
        "url": "",
        "keywords": "",
    }


def test_keywords_list_joined_with_comma_space():
    """Keywords list is joined with ', ' preserving order."""
    detail = {"media_id": "1", "title": "T", "type": "video", "keywords": ["zeta", "alpha", "mid"]}

    state = build_library_media_viewer_state(detail, now=NOW)

    assert "Keywords: zeta, alpha, mid" in state.metadata_lines
    assert state.edit_fields["keywords"] == "zeta, alpha, mid"


def test_empty_keywords_list_omits_keywords_line():
    """An empty keywords list omits the Keywords line and yields empty edit field."""
    detail = {"media_id": "1", "title": "T", "type": "video", "keywords": []}

    state = build_library_media_viewer_state(detail, now=NOW)

    assert state.metadata_lines == ("Type: video",)
    assert state.edit_fields["keywords"] == ""


def test_ingested_age_falls_back_to_last_modified():
    """When ingestion_date is absent, Ingested age is derived from last_modified."""
    detail = {
        "media_id": "1",
        "title": "T",
        "type": "video",
        "last_modified": "2026-07-06T11:57:00+00:00",  # 3m old
    }

    state = build_library_media_viewer_state(detail, now=NOW)

    assert "Ingested: 3m" in state.metadata_lines


def test_ingested_date_takes_priority_over_last_modified():
    """When both timestamps are present, ingestion_date is preferred."""
    detail = {
        "media_id": "1",
        "title": "T",
        "type": "video",
        "ingestion_date": "2026-07-06T10:00:00+00:00",  # 2h old
        "last_modified": "2026-07-06T11:57:00+00:00",  # 3m old
    }

    state = build_library_media_viewer_state(detail, now=NOW)

    assert "Ingested: 2h" in state.metadata_lines
    assert "Ingested: 3m" not in state.metadata_lines


def test_ingested_line_omitted_when_both_timestamps_missing():
    """Ingested line is omitted entirely when no timestamp source is present."""
    detail = {"media_id": "1", "title": "T", "type": "video"}

    state = build_library_media_viewer_state(detail, now=NOW)

    assert all(not line.startswith("Ingested:") for line in state.metadata_lines)


def test_ingested_line_omitted_when_timestamp_unparseable():
    """An unparseable timestamp yields a blank age, so the line is omitted."""
    detail = {
        "media_id": "1",
        "title": "T",
        "type": "video",
        "ingestion_date": "not-a-timestamp",
    }

    state = build_library_media_viewer_state(detail, now=NOW)

    assert all(not line.startswith("Ingested:") for line in state.metadata_lines)


def test_content_and_analysis_absent_yields_false_flags_and_empty_strings():
    """Missing content/analysis_content -> has_* False and blank strings."""
    detail = {"media_id": "1", "title": "T", "type": "video"}

    state = build_library_media_viewer_state(detail, now=NOW)

    assert state.content == ""
    assert state.analysis == ""
    assert state.has_content is False
    assert state.has_analysis is False


def test_whitespace_only_content_and_analysis_treated_as_blank():
    """Whitespace-only content/analysis strings are stripped to blank -> has_* False."""
    detail = {
        "media_id": "1",
        "title": "T",
        "type": "video",
        "content": "   ",
        "analysis_content": "\n\t",
    }

    state = build_library_media_viewer_state(detail, now=NOW)

    assert state.content == ""
    assert state.analysis == ""
    assert state.has_content is False
    assert state.has_analysis is False


def test_version_passthrough_and_missing_defaults_to_none():
    """Version is passed through when present, and defaults to None when absent."""
    with_version = build_library_media_viewer_state(
        {"media_id": "1", "title": "T", "type": "video", "version": 7}, now=NOW
    )
    assert with_version.version == 7

    without_version = build_library_media_viewer_state(
        {"media_id": "1", "title": "T", "type": "video"}, now=NOW
    )
    assert without_version.version is None


def test_media_id_key_fallback_to_id():
    """media_id falls back to the 'id' key when 'media_id' is absent."""
    detail = {"id": "row-42", "title": "T", "type": "video"}

    state = build_library_media_viewer_state(detail, now=NOW)

    assert state.media_id == "row-42"


def test_none_detail_yields_empty_state():
    """A None detail yields a fully empty, safe-default state."""
    state = build_library_media_viewer_state(None, now=NOW)

    assert state.media_id == ""
    assert state.title == ""
    assert state.metadata_lines == ()
    assert state.content == ""
    assert state.analysis == ""
    assert state.has_content is False
    assert state.has_analysis is False
    assert state.version is None
    assert state.edit_fields == {"title": "", "author": "", "url": "", "keywords": ""}


def test_non_mapping_detail_tolerated_like_none():
    """Non-mapping input (e.g. a string) is tolerated and yields the empty state."""
    state = build_library_media_viewer_state("not-a-mapping", now=NOW)  # type: ignore[arg-type]

    assert state.media_id == ""
    assert state.metadata_lines == ()
    assert state.edit_fields == {"title": "", "author": "", "url": "", "keywords": ""}


def test_default_now_uses_current_time_when_not_supplied():
    """When now is not supplied, the function still returns a valid state (uses real clock)."""
    detail = {
        "media_id": "1",
        "title": "T",
        "type": "video",
        "ingestion_date": "2020-01-01T00:00:00+00:00",
    }

    state = build_library_media_viewer_state(detail)

    assert state.media_id == "1"
    # Some plausible age line should be present (years old relative to real "now").
    assert any(line.startswith("Ingested:") for line in state.metadata_lines)
