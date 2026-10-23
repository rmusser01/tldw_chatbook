"""Leaving a note field tidies exactly what the save would refuse (TASK-34000.10).

Review N-07: a title ending in a space, or keywords with a case-insensitive
duplicate ("ok, x, X"), vetoes every autosave -- ADR-027 forbids the save
from stripping or deduplicating on the user's behalf. The tidy runs when the
user LEAVES the field, as a visible draft edit, and it is the inverse of
exactly those two vetoes and nothing more: no truncation, no markup change,
no rewrite of a title or keyword list the save would already accept.

Deterministic examples only: this file runs in the PR Fast Lane, which
installs no Hypothesis. The properties (a tidied value never draws its
veto; an accepted value is left exactly as typed) are in
``test_library_note_field_tidy_properties.py``, behind ``importorskip``.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tldw_chatbook.Library.library_note_field_tidy import (
    tidy_note_keywords,
    tidy_note_title,
)
from tldw_chatbook.Library.library_notes_state import (
    DatabaseNoteDraft,
    NoteValidationVeto,
    validate_database_note_draft,
)

pytestmark = pytest.mark.unit

_WHITESPACE_VETO = "Title begins or ends with whitespace — remove it to save."
_DUPLICATE_VETO = "Keywords contain a case-insensitive duplicate — remove one to save."


def _draft(*, title: str = "Note", keywords_text: str = "") -> DatabaseNoteDraft:
    return DatabaseNoteDraft("n-1", title, "Body", keywords_text, 0)


def _veto_message(**fields) -> str | None:
    outcome = validate_database_note_draft(replace(_draft(), **fields))
    return outcome.message if isinstance(outcome, NoteValidationVeto) else None


# --- title ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "tidied"),
    [
        ("Groceries ", "Groceries"),
        ("  Groceries", "Groceries"),
        ("\tGroceries \n", "Groceries"),
        ("Groceries", "Groceries"),
        ("Grocery  list", "Grocery  list"),  # inner spacing is content
        ("", ""),
        ("   ", ""),
    ],
)
def test_tidy_note_title_strips_only_the_edges(raw: str, tidied: str) -> None:
    assert tidy_note_title(raw) == tidied


def test_tidy_note_title_is_idempotent() -> None:
    once = tidy_note_title(" A title ")
    assert tidy_note_title(once) == once


# --- keywords ---------------------------------------------------------------


def test_tidy_note_keywords_drops_the_later_case_insensitive_duplicate() -> None:
    assert tidy_note_keywords("ok, x, X") == "ok, x"


def test_tidy_note_keywords_keeps_order_and_the_first_spelling() -> None:
    assert tidy_note_keywords("Zulu, alpha, ALPHA") == "Zulu, alpha"
    assert tidy_note_keywords("b, A, a, B, c") == "b, A, c"


def test_tidy_note_keywords_normalises_delimiter_whitespace_and_empty_tokens() -> None:
    # Delimiter-adjacent whitespace and empty segments are syntax the save
    # already ignores; the tidy spells the same keyword set the one way.
    assert tidy_note_keywords(" ok ,x,, X ,") == "ok, x"
    assert tidy_note_keywords("one") == "one"


@pytest.mark.parametrize("raw", ["", "   ", ",", ", ,", "\t,\n"])
def test_tidy_note_keywords_returns_empty_for_all_empty_input(raw: str) -> None:
    assert tidy_note_keywords(raw) == ""


def test_tidy_note_keywords_is_idempotent() -> None:
    once = tidy_note_keywords("ok, x, X, Ok")
    assert once == "ok, x"
    assert tidy_note_keywords(once) == once


def test_tidy_note_keywords_does_not_touch_a_keyword_the_save_would_refuse_for_length() -> None:
    # Not the tidy's veto: a 101-character keyword stays as typed, so the
    # save still refuses it with its own message (no truncation, ADR-027).
    long = "k" * 101
    assert tidy_note_keywords(long) == long


# --- final review M-2: no rewrite unless the save would refuse something ---


@pytest.mark.parametrize(
    "raw",
    ["a,b", "a, b", "a,b,c", "ok, x", " a ,b", "one two, three"],
)
def test_tidy_note_keywords_leaves_an_accepted_list_exactly_as_typed(raw: str) -> None:
    # No duplicate identity and no empty segment: the save accepts it, so
    # the blur does not even re-spell the delimiters ("a,b" stays "a,b").
    assert _veto_message(keywords_text=raw) != _DUPLICATE_VETO
    assert tidy_note_keywords(raw) == raw


@pytest.mark.parametrize(
    ("raw", "tidied"),
    [
        ("a, A", "a"),  # a case-insensitive duplicate: the save refuses it
        ("a,,b", "a, b"),  # an empty segment: dropped, the rest re-spelled
        ("a,b,", "a, b"),  # a trailing comma is an empty segment too
    ],
)
def test_tidy_note_keywords_rewrites_only_when_there_is_something_to_drop(
    raw: str, tidied: str
) -> None:
    assert tidy_note_keywords(raw) == tidied
    assert _veto_message(keywords_text=tidied) != _DUPLICATE_VETO


# --- the tidy is the inverse of exactly the two vetoes (deterministic) --------


@pytest.mark.parametrize("title", ["Groceries ", " Groceries", "\tGroceries\n", "   "])
def test_a_tidied_title_never_draws_the_whitespace_veto(title: str) -> None:
    assert _veto_message(title=title) == _WHITESPACE_VETO
    assert _veto_message(title=tidy_note_title(title)) != _WHITESPACE_VETO


@pytest.mark.parametrize("keywords_text", ["ok, x, X", "a, A", "b, A, a, B, c"])
def test_tidied_keywords_never_draw_the_duplicate_veto(keywords_text: str) -> None:
    assert _veto_message(keywords_text=keywords_text) == _DUPLICATE_VETO
    assert _veto_message(keywords_text=tidy_note_keywords(keywords_text)) != (
        _DUPLICATE_VETO
    )


@pytest.mark.parametrize("title", ["Groceries", "Grocery  list", "A title"])
def test_a_title_the_save_accepts_is_left_exactly_as_typed(title: str) -> None:
    # The tidy changes only what the whitespace veto would refuse.
    assert _veto_message(title=title) != _WHITESPACE_VETO
    assert tidy_note_title(title) == title
