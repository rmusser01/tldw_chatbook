"""Hypothesis properties of the note-field tidy (TASK-34000.10).

The deterministic contract lives in ``test_library_note_field_tidy.py``,
which runs in the PR Fast Lane. That lane installs no Hypothesis (final
review, CI: ``ModuleNotFoundError: No module named 'hypothesis'`` on
collection), so the properties live here, behind ``importorskip``, and
this file is NOT in ``FAST_LANE_TARGETS``.

Each tidy is the inverse of exactly its veto: a tidied title never draws
the whitespace veto, tidied keywords never draw the duplicate veto, and a
title or keyword list the save already accepts is left exactly as typed.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from tldw_chatbook.Library.library_note_field_tidy import (  # noqa: E402
    tidy_note_keywords,
    tidy_note_title,
)
from tldw_chatbook.Library.library_notes_state import (  # noqa: E402
    DatabaseNoteDraft,
    NoteValidationVeto,
    validate_database_note_draft,
)

pytestmark = pytest.mark.unit

_WHITESPACE_VETO = "Title begins or ends with whitespace — remove it to save."
_DUPLICATE_VETO = "Keywords contain a case-insensitive duplicate — remove one to save."


def _draft(*, title: str = "Note", keywords_text: str = "") -> DatabaseNoteDraft:
    return DatabaseNoteDraft("n-1", title, "Body", keywords_text, 0)


@settings(max_examples=200, deadline=None)
@given(st.text())
def test_a_tidied_title_never_draws_the_whitespace_veto(title: str) -> None:
    outcome = validate_database_note_draft(
        replace(_draft(), title=tidy_note_title(title))
    )
    if isinstance(outcome, NoteValidationVeto):
        assert outcome.message != _WHITESPACE_VETO, title


@settings(max_examples=200, deadline=None)
@given(st.text())
def test_tidied_keywords_never_draw_the_duplicate_veto(keywords_text: str) -> None:
    outcome = validate_database_note_draft(
        replace(_draft(), keywords_text=tidy_note_keywords(keywords_text))
    )
    if isinstance(outcome, NoteValidationVeto):
        assert outcome.message != _DUPLICATE_VETO, keywords_text


@settings(max_examples=200, deadline=None)
@given(st.text())
def test_a_title_the_save_accepts_is_left_exactly_as_typed(title: str) -> None:
    # The tidy changes only what the whitespace veto would refuse.
    outcome = validate_database_note_draft(replace(_draft(), title=title))
    if not isinstance(outcome, NoteValidationVeto):
        assert tidy_note_title(title) == title


@settings(max_examples=200, deadline=None)
@given(st.lists(st.text(min_size=1, alphabet=st.characters(blacklist_characters=",")), max_size=6))
def test_keywords_with_nothing_to_drop_are_left_exactly_as_typed(tokens: list[str]) -> None:
    # Final review M-2: no duplicate identity and no empty segment means no
    # rewrite at all -- not even the ", " re-spelling.
    stripped = [token.strip() for token in tokens]
    if any(not token for token in stripped):
        return
    identities = [token.casefold() for token in stripped]
    if len(set(identities)) != len(identities):
        return
    text = ",".join(tokens)
    assert tidy_note_keywords(text) == text
