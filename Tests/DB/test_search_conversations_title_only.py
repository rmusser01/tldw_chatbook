"""TASK-33622.6: a titles-only conversation search for the Console rail filter.

The rail's 'Filter visible titles' input counted message-content and
conversation-id matches at the database but showed only rows whose title
matched, so filtering 'zebra' read '1 match. Showing 0 of 1' over an empty
list. The owner's rule (2026-10-27): that filter matches titles only, with
its count and rows on the same rule. ``title_only=True`` is the database
half: the count and the page both match the title alone, as a literal
substring, case-insensitively for ASCII as the rail's own matcher does.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, InputError


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(tmp_path / "chacha.db", "test-client")
    yield database
    database.close_connection()


def _conversation(db, title: str, content: str = "") -> str:
    conversation_id = db.add_conversation({"title": title})
    if content:
        db.add_message(
            {"conversation_id": conversation_id, "sender": "user", "content": content}
        )
    return conversation_id


def _titles(rows) -> set[str]:
    return {row["title"] for row in rows}


def test_title_only_ignores_message_content_and_ids(db):
    titled = _conversation(db, "Zebra crossing plan", "nothing here")
    _conversation(db, "Quarterly notes", "the zebra crossing outside")

    rows, total, _ = db.search_conversations_page("zebra", title_only=True)
    assert _titles(rows) == {"Zebra crossing plan"}
    assert total == 1

    # The default (Library, Ctrl+K) search still matches content.
    _rows, content_total, _ = db.search_conversations_page("zebra")
    assert content_total == 2

    # An exact id is not a title.
    _rows, id_total, _ = db.search_conversations_page(titled, title_only=True)
    assert id_total == 0


def test_title_only_count_equals_rows_and_ignores_case(db):
    for title in ("OK 9 rollout plan", "ok computer", "Okra", "Unrelated"):
        _conversation(db, title)

    rows, total, _ = db.search_conversations_page("OK", title_only=True)
    assert _titles(rows) == {"OK 9 rollout plan", "ok computer", "Okra"}
    assert total == len(rows) == 3


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("a_b", {"a_b plan"}),
        ("50%", {"50% off"}),
        ("c\\d", {"c\\d path"}),
    ],
)
def test_title_only_treats_like_wildcards_as_literal_text(db, query, expected):
    """'_' and '%' are what the user typed, not SQL wildcards.

    Unescaped, 'a_b' also counted 'axb plan', which the rail's substring
    matcher never shows -- the count and the rows would disagree again.
    """
    for title in ("a_b plan", "axb plan", "50% off", "500 items", "c\\d path", "cd"):
        _conversation(db, title)

    rows, total, _ = db.search_conversations_page(query, title_only=True)
    assert _titles(rows) == expected
    assert total == len(expected)


def test_title_only_refuses_per_term_queries(db):
    with pytest.raises(InputError):
        db.search_conversations_page(
            None, scope_type="all", query_terms=("alpha",), title_only=True
        )
