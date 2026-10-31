"""TASK-34000.11: Hypothesis properties for the typed-link helpers.

Kept out of the fast PR lane on purpose: that lane installs no Hypothesis
(the P1-A ``test_library_note_field_tidy_properties.py`` pattern), so the
table tests live in ``test_note_links_typed_titles.py`` and only the
generated-input properties live here.
"""

from __future__ import annotations

import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from tldw_chatbook.Notes.note_import_plan_models import (  # noqa: E402
    WIKILINK_SCAN,
    bare_wikilink_titles,
    note_link_entries,
    stored_note_link,
    wikilink_completion_context,
)

_LINKISH = st.text(alphabet="[]|#^`!\n (a)note:/", min_size=0, max_size=40)


@settings(max_examples=300, deadline=None)
@given(st.one_of(st.text(max_size=60), _LINKISH), st.integers(-3, 70), st.integers(-3, 70))
def test_completion_context_never_raises_and_starts_with_the_opener(text, row, col):
    result = wikilink_completion_context(text, (row, col))
    if result is None:
        return
    query, (start_row, start_col) = result
    lines = text.split("\n")
    assert 0 <= start_row < len(lines)
    line = lines[start_row]
    assert line[start_col : start_col + 2] == "[["
    assert line[start_col + 2 : start_col + 2 + len(query)] == query
    assert not any(ch in query for ch in "[]|\n")


@settings(max_examples=300, deadline=None)
@given(_LINKISH)
def test_bare_titles_never_come_from_a_code_span_or_a_tailed_link(text):
    titles = bare_wikilink_titles(text)
    assert len(titles) == len(set(titles))
    for title in titles:
        assert title == title.strip() and title
        assert not any(ch in title for ch in "[]|#^")
    # Every title is the target of SOME bare, non-code match.
    bare_targets = {
        (m.group("target") or "").strip()
        for m in WIKILINK_SCAN.finditer(text)
        if m.group("code") is None and not m.group("link")
    }
    assert set(titles) <= bare_targets


@settings(max_examples=300, deadline=None)
@given(st.text(max_size=40), st.from_regex(r"[A-Za-z0-9-]{1,36}", fullmatch=True))
def test_stored_note_link_always_parses_back_as_one_link_carrying_its_id(title, note_id):
    link = stored_note_link(title, note_id)
    matches = [m for m in WIKILINK_SCAN.finditer(link) if m.group("code") is None]
    assert len(matches) == 1
    assert matches[0].group(0) == link
    assert matches[0].group("link") == f"(note://{note_id})"
    assert note_link_entries(link) == ((matches[0].group("target").strip(), note_id),)
