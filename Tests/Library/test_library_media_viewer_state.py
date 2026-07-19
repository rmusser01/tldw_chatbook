"""Pure display-state contracts for the Library media viewer canvas."""

from __future__ import annotations

from datetime import datetime, timezone

from tldw_chatbook.Library.library_media_viewer_state import (
    LibraryMediaHighlightRow,
    LibraryMediaViewerState,
    build_library_media_highlight_rows,
    build_library_media_viewer_state,
    find_content_matches,
)

NOW = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)


def test_full_detail_builds_all_metadata_lines_in_order():
    """All metadata sources present -> ordered Type/Author/URL/Keywords/Updated lines."""
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
        "is_read_it_later": True,
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
        "Updated: 2h",
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
    assert state.read_later is True


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


def test_url_line_omitted_for_local_scheme():
    """UX wave L3: a ``local://`` URL is an internal identifier, not a
    user-meaningful link -- the metadata line is hidden, but the edit-form
    field still prefills the real stored value.
    """
    detail = {"media_id": "1", "title": "T", "type": "video", "url": "local://media/1"}

    state = build_library_media_viewer_state(detail, now=NOW)

    assert state.metadata_lines == ("Type: video",)
    assert state.edit_fields["url"] == "local://media/1"


def test_url_line_present_for_https_scheme():
    """A real https:// URL still renders its metadata line as before."""
    detail = {"media_id": "1", "title": "T", "type": "video", "url": "https://example.com/a"}

    state = build_library_media_viewer_state(detail, now=NOW)

    assert "URL: https://example.com/a" in state.metadata_lines


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


def test_updated_age_from_ingestion_date_when_last_modified_absent():
    """When last_modified is absent, the Updated age falls back to ingestion_date."""
    detail = {
        "media_id": "1",
        "title": "T",
        "type": "video",
        "ingestion_date": "2026-07-06T11:57:00+00:00",  # 3m old
    }

    state = build_library_media_viewer_state(detail, now=NOW)

    assert "Updated: 3m" in state.metadata_lines


def test_updated_prefers_last_modified_over_ingestion_date():
    """When both timestamps are present, last_modified is preferred (matches the list)."""
    detail = {
        "media_id": "1",
        "title": "T",
        "type": "video",
        "ingestion_date": "2026-07-06T10:00:00+00:00",  # 2h old
        "last_modified": "2026-07-06T11:57:00+00:00",  # 3m old
    }

    state = build_library_media_viewer_state(detail, now=NOW)

    assert "Updated: 3m" in state.metadata_lines
    assert "Updated: 2h" not in state.metadata_lines


def test_updated_line_omitted_when_both_timestamps_missing():
    """Updated line is omitted entirely when no timestamp source is present."""
    detail = {"media_id": "1", "title": "T", "type": "video"}

    state = build_library_media_viewer_state(detail, now=NOW)

    assert all(not line.startswith("Updated:") for line in state.metadata_lines)


def test_updated_line_omitted_when_timestamp_unparseable():
    """An unparseable timestamp yields a blank age, so the line is omitted."""
    detail = {
        "media_id": "1",
        "title": "T",
        "type": "video",
        "last_modified": "not-a-timestamp",
    }

    state = build_library_media_viewer_state(detail, now=NOW)

    assert all(not line.startswith("Updated:") for line in state.metadata_lines)


def test_content_and_analysis_absent_yields_false_flags_and_empty_strings():
    """Missing content/analysis_content -> has_* False and blank strings."""
    detail = {"media_id": "1", "title": "T", "type": "video"}

    state = build_library_media_viewer_state(detail, now=NOW)

    assert state.content == ""
    assert state.analysis == ""
    assert state.has_content is False
    assert state.has_analysis is False


def test_analysis_falls_back_to_latest_document_version_when_top_level_absent():
    """Local media details never carry top-level analysis_content -- it lives on
    DocumentVersions only, surfaced via get_media_item's ``versions`` list
    (newest-first). The newest version's analysis_content must be surfaced.
    """
    detail = {
        "media_id": "1",
        "title": "T",
        "type": "article",
        "versions": [
            {"version_number": 2, "analysis_content": "Latest analysis"},
            {"version_number": 1, "analysis_content": None},
        ],
    }

    state = build_library_media_viewer_state(detail, now=NOW)

    assert state.analysis == "Latest analysis"
    assert state.has_analysis is True


def test_analysis_prefers_top_level_analysis_content_over_versions():
    """When a top-level analysis_content IS present, it wins over versions."""
    detail = {
        "media_id": "1",
        "title": "T",
        "type": "article",
        "analysis_content": "Top-level analysis",
        "versions": [{"version_number": 1, "analysis_content": "Version analysis"}],
    }

    state = build_library_media_viewer_state(detail, now=NOW)

    assert state.analysis == "Top-level analysis"


def test_analysis_blank_when_latest_version_has_no_analysis_even_if_older_one_does():
    """The latest version's analysis is authoritative -- an older version's
    analysis is not used as a fallback when the newest version is blank
    (e.g. after an explicit clear or a content-only rollback version).
    """
    detail = {
        "media_id": "1",
        "title": "T",
        "type": "article",
        "versions": [
            {"version_number": 2, "analysis_content": None},
            {"version_number": 1, "analysis_content": "Old analysis"},
        ],
    }

    state = build_library_media_viewer_state(detail, now=NOW)

    assert state.analysis == ""
    assert state.has_analysis is False


def test_analysis_blank_when_versions_absent_or_not_a_list():
    """Missing/non-list versions field is tolerated -> blank analysis."""
    assert build_library_media_viewer_state(
        {"media_id": "1", "title": "T", "type": "article"}, now=NOW
    ).analysis == ""
    assert build_library_media_viewer_state(
        {"media_id": "1", "title": "T", "type": "article", "versions": "not-a-list"}, now=NOW
    ).analysis == ""
    assert build_library_media_viewer_state(
        {"media_id": "1", "title": "T", "type": "article", "versions": [{"analysis_content": None}, "not-a-mapping"]},
        now=NOW,
    ).analysis == ""


def test_read_later_true_when_detail_flag_set():
    """read_later reflects a truthy is_read_it_later flag on the detail."""
    detail = {"media_id": "1", "title": "T", "type": "article", "is_read_it_later": True}

    state = build_library_media_viewer_state(detail, now=NOW)

    assert state.read_later is True


def test_read_later_false_when_flag_absent_or_falsy():
    """read_later defaults to False when the flag is absent or falsy.

    Mirrors the real local backend: after
    LocalMediaReadingService.remove_from_read_it_later, the
    MediaReadItLaterState row is deleted entirely, so a re-fetched detail
    has no is_read_it_later key at all (not even False).
    """
    assert build_library_media_viewer_state(
        {"media_id": "1", "title": "T", "type": "article"}, now=NOW
    ).read_later is False
    assert build_library_media_viewer_state(
        {"media_id": "1", "title": "T", "type": "article", "is_read_it_later": False}, now=NOW
    ).read_later is False


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
    assert state.read_later is False


def test_non_mapping_detail_tolerated_like_none():
    """Non-mapping input (e.g. a string) is tolerated and yields the empty state."""
    state = build_library_media_viewer_state("not-a-mapping", now=NOW)  # type: ignore[arg-type]

    assert state.media_id == ""
    assert state.metadata_lines == ()
    assert state.edit_fields == {"title": "", "author": "", "url": "", "keywords": ""}


def test_highlight_rows_include_quote_note_and_color():
    """A highlight with quote/note/color renders all three on its display text."""
    rows = build_library_media_highlight_rows(
        [{"id": 5, "quote": "Important sentence", "note": "Check this", "color": "yellow"}]
    )

    assert rows == (
        LibraryMediaHighlightRow(
            highlight_id="5",
            quote="Important sentence",
            note="Check this",
            color="yellow",
            display_text="“Important sentence”\nColor: yellow · Note: Check this",
        ),
    )


def test_highlight_row_quote_only_omits_extras_line():
    """A highlight with only a quote renders a single-line display text."""
    rows = build_library_media_highlight_rows([{"id": 1, "quote": "Just the quote"}])

    assert rows == (
        LibraryMediaHighlightRow(
            highlight_id="1",
            quote="Just the quote",
            note="",
            color="",
            display_text="“Just the quote”",
        ),
    )


def test_highlight_rows_preserve_order():
    """Multiple highlights render in the given order."""
    rows = build_library_media_highlight_rows(
        [
            {"id": 1, "quote": "First"},
            {"id": 2, "quote": "Second"},
        ]
    )

    assert [row.quote for row in rows] == ["First", "Second"]


def test_highlight_rows_skip_blank_quote_and_non_mapping_entries():
    """Entries with a blank/missing quote, or non-mapping entries, are skipped."""
    rows = build_library_media_highlight_rows(
        [
            {"id": 1, "quote": "   "},
            {"id": 2},
            "not-a-mapping",
            {"id": 3, "quote": "Kept"},
        ]
    )

    assert [row.quote for row in rows] == ["Kept"]


def test_highlight_rows_none_input_yields_empty_tuple():
    """None input yields an empty tuple of rows."""
    assert build_library_media_highlight_rows(None) == ()


def test_highlight_row_missing_id_yields_empty_highlight_id():
    """A highlight missing its id yields an empty highlight_id string."""
    rows = build_library_media_highlight_rows([{"quote": "No id here"}])

    assert rows[0].highlight_id == ""


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
    assert any(line.startswith("Updated:") for line in state.metadata_lines)


def test_find_content_matches_returns_matching_line_indices_in_order():
    """Multi-line content: only the lines containing the query are returned, in order."""
    content = "alpha line\nbravo line\ncharlie line\nbravo again"

    matches = find_content_matches(content, "bravo")

    assert matches == (1, 3)


def test_find_content_matches_is_case_insensitive():
    """Query matching ignores case on both sides."""
    content = "The Quick Fox\nlazy dog\nQUICK step"

    assert find_content_matches(content, "quick") == (0, 2)
    assert find_content_matches(content, "QUICK") == (0, 2)


def test_find_content_matches_no_match_returns_empty_tuple():
    """A query with no hits in the content returns an empty tuple."""
    content = "one\ntwo\nthree"

    assert find_content_matches(content, "nonexistent") == ()


def test_find_content_matches_empty_query_returns_empty_tuple():
    """A blank query returns no matches even when content is present."""
    content = "one\ntwo\nthree"

    assert find_content_matches(content, "") == ()


def test_find_content_matches_strips_surrounding_whitespace_in_query():
    """A padded query matches the same lines as its trimmed form.

    The in-content highlighter strips the query, so match-finding/scrolling
    must strip too or the status/scroll would disagree with the highlights.
    """
    content = "alpha line\nbudget line\ncharlie"

    assert find_content_matches(content, "  budget  ") == (1,)


def test_find_content_matches_whitespace_only_query_returns_empty_tuple():
    """A whitespace-only query is treated as blank (no matches)."""
    content = "one\ntwo\nthree"

    assert find_content_matches(content, "   ") == ()


def test_find_content_matches_empty_content_returns_empty_tuple():
    """Blank content returns no matches even when a query is given."""
    assert find_content_matches("", "anything") == ()


def test_find_content_matches_none_content_or_query_returns_empty_tuple():
    """None content/query is tolerated the same as an empty string."""
    assert find_content_matches(None, "anything") == ()  # type: ignore[arg-type]
    assert find_content_matches("some content", None) == ()  # type: ignore[arg-type]


def test_find_content_matches_multiple_occurrences_on_one_line_counts_once():
    """A line with the query appearing more than once is only reported once."""
    content = "bravo bravo bravo\ncharlie"

    assert find_content_matches(content, "bravo") == (0,)


def test_find_content_matches_multiple_matches_on_different_lines():
    """Every distinct matching line is reported, preserving document order."""
    content = "\n".join(
        [
            "no match here",
            "TARGET found",
            "still nothing",
            "another target line",
            "target once more",
        ]
    )

    assert find_content_matches(content, "target") == (1, 3, 4)
