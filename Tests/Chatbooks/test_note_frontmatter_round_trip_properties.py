"""TASK-34000.31: Hypothesis properties for the note front matter round trip.

Kept out of the fast PR lane on purpose: that lane installs no Hypothesis
(the ``test_note_links_typed_titles_properties.py`` pattern), so the table
tests live in ``test_note_frontmatter_round_trip.py`` and only the
generated-input properties live here. Pure: the renderer and the splitter,
no database, no app import.
"""

from __future__ import annotations

import pytest

hypothesis = pytest.importorskip("hypothesis")

import yaml  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from tldw_chatbook.Library.library_notes_state import (  # noqa: E402
    build_note_export_content,
)
from tldw_chatbook.Notes.note_frontmatter import render_note_frontmatter  # noqa: E402
from tldw_chatbook.Notes.note_import_parsers import (  # noqa: E402
    _frontmatter_keywords,
    split_frontmatter,
)
from tldw_chatbook.Notes.note_import_plan_models import ImportBounds  # noqa: E402

# A title that is only whitespace falls back to "Untitled Note" -- pinned by
# ``test_export_content_blank_title_falls_back_to_untitled`` -- and a line
# break cannot be a title at all, so both are out of the strategy. The
# exporter strips the title, so a padded title is compared stripped.
_TITLES = st.text(min_size=1, max_size=120).filter(
    lambda s: s.strip() and "\n" not in s and "\r" not in s
)
# Keywords arrive as one comma-separated field, so a keyword cannot hold a
# comma or a line break; the exporter strips each one.
_KEYWORDS = st.lists(
    st.text(min_size=1, max_size=40).filter(
        lambda s: s.strip() and "," not in s and "\n" not in s and "\r" not in s
    ),
    max_size=6,
)

_BOUNDS = ImportBounds(
    max_files=200,
    max_file_bytes=100_000,
    max_total_bytes=1_000_000,
    max_depth=6,
)


@settings(max_examples=300, deadline=None)
@given(_TITLES, _KEYWORDS)
def test_rendered_block_reads_back_exactly_with_a_standard_parser(title, keywords):
    text = render_note_frontmatter({"title": title, "tags": keywords, "note_id": "n-1"})
    assert text.startswith("---\n") and text.endswith("---\n")
    data = yaml.safe_load(text.split("---\n", 2)[1])
    assert data == {"title": title, "tags": keywords, "note_id": "n-1"}


@settings(max_examples=300, deadline=None)
@given(_TITLES, _KEYWORDS)
def test_single_note_export_round_trips_through_the_app_importer(title, keywords):
    stripped = [keyword.strip() for keyword in keywords]
    text = build_note_export_content(
        title, "body", ", ".join(keywords), "n-1", "markdown"
    )
    metadata, body = split_frontmatter(text)
    assert metadata is not None
    assert metadata["title"] == title.strip()
    assert metadata["tags"] == stripped
    assert body.startswith(f"# {title.strip()}\n")
    # The importer's keyword read: bounded, deduplicated case-insensitively.
    expected: list[str] = []
    seen: set[str] = set()
    for keyword in stripped:
        if keyword.casefold() in seen:
            continue
        seen.add(keyword.casefold())
        expected.append(keyword)
    assert list(_frontmatter_keywords(metadata, _BOUNDS)) == expected
