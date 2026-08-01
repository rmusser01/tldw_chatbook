"""Tests for the markdown half of briefing export (spec #2 phase 3, Task 1).

`briefing_export.py` turns one `briefings` row into a markdown document a
user can save wherever they like, plus the two pure helpers around it:
`safe_export_stem` (a filesystem-safe name from arbitrary, possibly
adversarial text -- reused verbatim by Task 4's feed-directory writer) and
`default_briefing_filename` (the seed the `FileSave` dialog opens with).

All three functions here are pure: no DB, no filesystem, no Textual. The UI
half (the toolbar button and the screen's dialog/write flow) is exercised
separately in `Tests/Watchlists/test_watchlists_artifacts_pane.py`.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Subscriptions.briefing_export import (
    BriefingExportError,
    briefing_markdown_document,
    default_briefing_filename,
    safe_export_stem,
)

pytestmark = pytest.mark.unit


def _complete_briefing(**overrides: object) -> dict:
    row = {
        "id": 42,
        "watchlist_id": 7,
        "watchlist_name": "Morning AI Brief",
        "status": "complete",
        "error": None,
        "covers_from_ts": "2026-07-25T00:00:00+00:00",
        "covers_through_item_id": 99,
        "model_used": "gpt-test",
        "body_markdown": "## This week\n\nAcme shipped a thing.\n",
        "item_count": 3,
        "featured_count": 1,
        "overflow_count": 0,
        "created_at": "2026-08-01 10:00:00",
    }
    row.update(overrides)
    return row


# --- briefing_markdown_document -----------------------------------------


def test_document_carries_the_body_verbatim():
    """The stored `body_markdown` must appear byte-for-byte in the exported
    document -- no reformatting, escaping, or truncation of the model- or
    user-authored content the export exists to hand over."""
    document = briefing_markdown_document(_complete_briefing())

    assert "## This week\n\nAcme shipped a thing.\n" in document


def test_document_front_matter_carries_watchlist_status_window_and_created():
    """All four front-matter fields (watchlist name, status, coverage
    window, created_at) must actually render, not just be accepted as
    arguments -- a document missing one silently loses the context a
    reader needs to place it without the app around it."""
    briefing = _complete_briefing()

    document = briefing_markdown_document(briefing)

    assert "Morning AI Brief" in document
    assert "complete" in document
    assert "2026-07-25T00:00:00+00:00" in document
    assert "99" in document
    assert "2026-08-01 10:00:00" in document


def test_document_front_matter_precedes_the_body():
    """The front matter must come BEFORE the body, not merely be present
    somewhere in the document -- a reader (or a markdown front-matter
    parser) reads top-down, so context has to precede content."""
    briefing = _complete_briefing()

    document = briefing_markdown_document(briefing)

    assert document.index("Morning AI Brief") < document.index(
        "Acme shipped a thing."
    )


def test_null_body_raises_naming_the_briefing():
    """A `body_markdown` of `None` must raise, and the exception message
    must name the specific briefing id (`42`) -- a toast built from
    `str(exc)` is the only way the user learns WHICH row failed to
    export."""
    briefing = _complete_briefing(body_markdown=None)

    with pytest.raises(BriefingExportError, match="42"):
        briefing_markdown_document(briefing)


def test_empty_body_raises_naming_the_briefing():
    """Whitespace-only counts as empty too -- an empty file is not an
    export, whatever whitespace it is made of."""
    briefing = _complete_briefing(body_markdown="   \n  ")

    with pytest.raises(BriefingExportError, match="42"):
        briefing_markdown_document(briefing)


# --- safe_export_stem -----------------------------------------------------


def test_stem_keeps_ordinary_characters():
    """The whitelist must not over-filter: ordinary alnum/space/hyphen text
    (a realistic watchlist name) must survive completely unchanged,
    including its spaces and casing -- this is the control case the
    stripping tests below are contrasted against."""
    assert safe_export_stem("Morning Brief 2026-08-01", fallback="x") == (
        "Morning Brief 2026-08-01"
    )


def test_stem_strips_path_separators():
    """Both `/` and `\\` must be dropped from the output -- a watchlist or
    briefing title containing a path separator must never be able to
    escape the destination directory a caller builds a filename inside."""
    stem = safe_export_stem("../../etc/passwd", fallback="fallback")
    assert "/" not in stem
    assert "\\" not in stem


def test_stem_strips_dot_dot():
    """A bare `".."` contains no whitelisted character at all (`.` is not
    in the whitelist), so it must fall back to the caller's `fallback`
    rather than surviving as a literal `".."` stem -- which, unlike the
    ordinary path-separator case above, contains no separator for a naive
    filter to catch."""
    stem = safe_export_stem("..", fallback="fallback")
    assert stem == "fallback"


def test_stem_strips_markup_shaped_text():
    """Bracket/markup punctuation (`[`, `]`) must be dropped -- a crafted
    title must not be able to produce a filename that some other surface
    later misinterprets as markup."""
    stem = safe_export_stem("[bold red]Evening Brief[/]", fallback="x")
    assert "[" not in stem
    assert "]" not in stem


def test_stem_falls_back_when_nothing_survives():
    """When every character in the input is outside the whitelist, the
    function must return the caller-supplied `fallback` verbatim rather
    than an empty string or a stem made of leftover punctuation."""
    assert safe_export_stem("###???", fallback="fallback-name") == "fallback-name"


def test_stem_falls_back_on_empty_input():
    """An empty string is a degenerate case of "nothing survives" and must
    hit the same fallback path as filtered-out text, not raise or return
    an empty stem."""
    assert safe_export_stem("", fallback="fallback-name") == "fallback-name"


# --- default_briefing_filename --------------------------------------------


def test_default_filename_ends_in_md():
    """The `FileSave` dialog this seeds always expects a markdown file --
    the suggested filename must carry a `.md` extension every time, not
    just for the input this particular briefing happens to have."""
    briefing = _complete_briefing()
    filename = default_briefing_filename(briefing, watchlist_name="Morning AI Brief")
    assert filename.endswith(".md")


def test_default_filename_never_contains_a_separator():
    """A hostile or path-shaped watchlist name must not survive into the
    suggested filename as a separator -- this is the end-to-end version of
    `safe_export_stem`'s own guarantee, through this caller specifically."""
    briefing = _complete_briefing(watchlist_name="../../evil")
    filename = default_briefing_filename(briefing, watchlist_name="../../evil")
    assert "/" not in filename
    assert "\\" not in filename


def test_default_filename_falls_back_when_watchlist_name_is_unusable():
    """When the watchlist name has nothing left after the whitelist, the
    result must still be a real, non-degenerate filename (via the
    `briefing-<id>` fallback) -- not just a bare `.md` with an empty
    stem, which would look wrong in a save dialog and could collide with
    every other briefing that hits the same fallback."""
    briefing = _complete_briefing()
    filename = default_briefing_filename(briefing, watchlist_name="###???")
    assert filename.endswith(".md")
    assert filename != ".md"
