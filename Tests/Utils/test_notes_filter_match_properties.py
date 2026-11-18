"""Hypothesis properties of the Notes list filter rule (TASK-34000.12).

The table lives in ``test_notes_filter_match.py``, which runs in the PR
Fast Lane. That lane installs no Hypothesis, so the generated-input
properties live here behind ``importorskip`` and this file is NOT in
``FAST_LANE_TARGETS`` (the TASK-34000.10 ``_properties`` pattern).

Two properties. The first is syntactic: whatever the user types, every
expression the rule binds is a doubled-quote-balanced quoted literal,
optionally followed by one ``*``, and the rule answers ``None`` exactly
when the primitive's searchability check says so. The second is the one
that matters: every generated expression is executed against a real
in-memory ``notes_fts`` and never raises -- an ``OperationalError`` here
would mean an unescaped operator reached FTS5.
"""

from __future__ import annotations

import re
import sqlite3

import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from tldw_chatbook.Utils.fts5_match_forms import (  # noqa: E402
    NOTES_FILTER_MAX_TOKENS,
    build_notes_filter_match,
    fts5_query_is_searchable,
)

pytestmark = pytest.mark.unit

#: One FTS5 string literal: a quoted run in which every embedded quote is
#: doubled, optionally a prefix.
_LITERAL = re.compile(r'^"(?:[^"]|"")*"\*?$', re.DOTALL)


def _phrase_wrapped(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) >= 3 and (stripped[0], stripped[-1]) in (
        ('"', '"'),
        ("“", "”"),
    )


@given(st.text(max_size=300))
@settings(max_examples=300, deadline=None)
def test_every_expression_is_one_balanced_quoted_literal(query: str) -> None:
    match = build_notes_filter_match(query)

    if match is None:
        # None exactly when the primitive refuses the text -- or when the
        # user quoted nothing searchable.
        assert not fts5_query_is_searchable(query) or (
            _phrase_wrapped(query)
            and not fts5_query_is_searchable(query.strip()[1:-1].strip())
        )
        return
    assert fts5_query_is_searchable(query)
    assert 1 <= len(match.expressions) <= NOTES_FILTER_MAX_TOKENS
    for expression in match.expressions:
        assert _LITERAL.match(expression), expression
    assert match.fts_expression == " ".join(match.expressions)
    assert match.phrase == _phrase_wrapped(query)
    if match.phrase:
        assert len(match.expressions) == 1


@pytest.fixture(scope="module")
def notes_fts() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE VIRTUAL TABLE notes_fts USING fts5(title, content)")
    connection.executemany(
        "INSERT INTO notes_fts(title, content) VALUES (?, ?)",
        [
            ("Trip planning: Lisbon", "itinerary and chunk size notes"),
            ("Meeting notes", 'team meeting about "size" of chunk'),
            ("Zebra", "nothing here, O'Brien said"),
        ],
    )
    return connection


@given(query=st.text(max_size=300))
@settings(max_examples=300, deadline=None)
def test_every_expression_executes_against_a_real_fts5_table(
    query: str, notes_fts: sqlite3.Connection
) -> None:
    match = build_notes_filter_match(query)
    if match is None:
        return
    for expression in (*match.expressions, match.fts_expression):
        # An unescaped operator, column filter or unterminated string would
        # raise sqlite3.OperationalError here.
        notes_fts.execute(
            "SELECT rowid FROM notes_fts WHERE notes_fts MATCH ?", (expression,)
        ).fetchall()
