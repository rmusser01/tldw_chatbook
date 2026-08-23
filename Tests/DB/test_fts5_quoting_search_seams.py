"""TASK-19558: every FTS5 search seam binds a quoted literal, not raw input.

Driven through the real search methods against real throwaway SQLite
databases -- never a hand-built SQL string -- because the defect was not "the
wrong string was formatted", it was "the string that was formatted never
reached the query". Only running the method proves which value SQLite saw.

The two user-visible consequences named in the task, and reproduced here as
born-red cases before the fix:

* **Library notes filter** (`UI/Screens/library_screen.py` ▸
  `_run_library_notes_filter` -> `notes_scope_service.search_notes` ->
  `CharactersRAGDB.search_notes`). Base behaviour, measured:
  `alpha" OR title:"Other` returned **2 rows** against a corpus where only
  one note contained "alpha" -- the closing quote ended the intended
  literal and the rest became a live FTS5 column filter. And `foo"bar`
  raised `unterminated string`, which that screen swallows
  (`except Exception: logger.warning(...); return`), so the filter box
  silently did nothing.
* **Study flashcard search box** (`UI/Study_Modules/flashcards_handler.py`
  ▸ `refresh_cards` -> `list_flashcards(q=...)`). Base behaviour: `foo"bar`
  raised a bare `sqlite3.OperationalError` out of an `await` that nothing on
  that path catches.

The equivalent Evals and Media seams are covered here too, in the same
sweep, for the same reason.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Evals_DB import EvalsDB

#: A query whose closing quote would end the intended literal and hand the
#: rest of the string to FTS5 as a column filter.
COLUMN_FILTER_INJECTION = 'alpha" OR title:"Other'
#: A query that is merely ordinary text with a quote in it.
ORDINARY_QUOTED = 'foo"bar'


@pytest.fixture()
def db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(tmp_path / "t19558.db", client_id="t19558")
    yield database
    database.close_connection()


@pytest.fixture()
def seeded(db: CharactersRAGDB) -> CharactersRAGDB:
    db.add_note(title="Quarterly report", content='the alpha "beta" gamma note')
    db.add_note(title="Other note", content="unrelated content entirely")
    return db


# ---------------------------------------------------------------------------
# Seam 1: the Library notes filter.
# ---------------------------------------------------------------------------


def test_notes_filter_does_not_execute_a_typed_column_filter(
    seeded: CharactersRAGDB,
) -> None:
    """The headline: base returned BOTH notes; neither contains this text."""
    rows = seeded.search_notes(COLUMN_FILTER_INJECTION, limit=10)
    assert [row["title"] for row in rows] == []


def test_notes_filter_survives_an_ordinary_typed_quote(
    seeded: CharactersRAGDB,
) -> None:
    """Base raised `unterminated string`, which the screen swallowed."""
    assert seeded.search_notes(ORDINARY_QUOTED, limit=10) == []


def test_notes_filter_finds_content_that_really_contains_a_quoted_word(
    seeded: CharactersRAGDB,
) -> None:
    """A clean, correct result -- not merely "no longer raises"."""
    rows = seeded.search_notes('"beta"', limit=10)
    assert [row["title"] for row in rows] == ["Quarterly report"]


def test_notes_filter_plain_query_is_unchanged(seeded: CharactersRAGDB) -> None:
    rows = seeded.search_notes("alpha", limit=10)
    assert [row["title"] for row in rows] == ["Quarterly report"]


def test_notes_filter_caller_built_expression_still_passes_through(
    seeded: CharactersRAGDB,
) -> None:
    """`fts_match_query` is the seam for the Library's widened keyword form."""
    rows = seeded.search_notes("ignored", limit=10, fts_match_query='"alpha"*')
    assert [row["title"] for row in rows] == ["Quarterly report"]


# ---------------------------------------------------------------------------
# Seam 2: the Study flashcard search box.
# ---------------------------------------------------------------------------


@pytest.fixture()
def deck(db: CharactersRAGDB) -> tuple[CharactersRAGDB, str]:
    deck_id = db.create_deck(name="Deck A")
    db.create_flashcard(
        {"deck_id": deck_id, "front": 'what is alpha "beta"?', "back": "gamma"}
    )
    db.create_flashcard({"deck_id": deck_id, "front": "plain card", "back": "delta"})
    return db, deck_id


def test_flashcard_search_box_does_not_raise_operationalerror(
    deck: tuple[CharactersRAGDB, str],
) -> None:
    """Base raised a bare sqlite3.OperationalError into the Study screen."""
    database, deck_id = deck
    assert database.list_flashcards(deck_id=deck_id, q=ORDINARY_QUOTED) == []


def test_flashcard_search_box_finds_a_card_whose_front_contains_a_quote(
    deck: tuple[CharactersRAGDB, str],
) -> None:
    database, deck_id = deck
    rows = database.list_flashcards(deck_id=deck_id, q="beta")
    assert [row["front"] for row in rows] == ['what is alpha "beta"?']


def test_flashcard_search_box_ignores_a_typed_column_filter(
    deck: tuple[CharactersRAGDB, str],
) -> None:
    database, deck_id = deck
    assert database.list_flashcards(deck_id=deck_id, q='alpha" OR back:"delta') == []


def test_search_flashcards_is_swept_with_its_sibling(
    deck: tuple[CharactersRAGDB, str],
) -> None:
    database, _deck_id = deck
    assert database.search_flashcards(ORDINARY_QUOTED) == []
    assert len(database.search_flashcards("alpha")) == 1


# ---------------------------------------------------------------------------
# The three dead stores: the value that was quoted is the value that is bound.
# ---------------------------------------------------------------------------


def _seed_for_dead_store(db: CharactersRAGDB) -> str:
    db.add_character_card({"name": "Zed the Hunter", "description": "a hunter"})
    conversation_id = db.add_conversation(
        {"title": "Talk about dragons", "character_id": 1}
    )
    db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "hello world",
        }
    )
    return conversation_id


def test_character_card_search_binds_the_quoted_term(db: CharactersRAGDB) -> None:
    """`safe_search_term` used to be computed and then NOT bound.

    Proven at base by mutation: replacing the computed value with an absurd
    string left all three methods' results byte-identical. The observable
    consequence of binding it is that a typed FTS5 operator stops being
    executed -- `Zed OR dragons` matched on the raw path and must not now.
    """
    _seed_for_dead_store(db)
    assert [row["name"] for row in db.search_character_cards("Hunter")] == [
        "Zed the Hunter"
    ]
    assert db.search_character_cards("Zed OR Nobody") == []
    assert db.search_character_cards('Zed" OR name:"x') == []


def test_character_card_search_still_serves_the_ccp_prefix_expression(
    db: CharactersRAGDB,
) -> None:
    """CCP's picker builds `"term"*`; it now goes through `fts_match_query`."""
    _seed_for_dead_store(db)
    rows = db.search_character_cards("Hun", fts_match_query='"Hun"*')
    assert [row["name"] for row in rows] == ["Zed the Hunter"]


def test_conversation_title_search_binds_the_quoted_term(
    db: CharactersRAGDB,
) -> None:
    _seed_for_dead_store(db)
    assert [
        row["title"] for row in db.search_conversations_by_title("dragons")
    ] == ["Talk about dragons"]
    assert db.search_conversations_by_title("dragons OR nothing") == []
    assert db.search_conversations_by_title(ORDINARY_QUOTED) == []


def test_message_content_search_binds_the_quoted_term(db: CharactersRAGDB) -> None:
    _seed_for_dead_store(db)
    assert [
        row["content"] for row in db.search_messages_by_content("world")
    ] == ["hello world"]
    assert db.search_messages_by_content("world OR nothing") == []
    assert db.search_messages_by_content(ORDINARY_QUOTED) == []


# ---------------------------------------------------------------------------
# The remaining ChaChaNotes seams that raised on ordinary input.
# ---------------------------------------------------------------------------


def test_keyword_search_finds_a_keyword_that_contains_a_quote(
    db: CharactersRAGDB,
) -> None:
    """Base raised `unterminated string` on the keyword's OWN spelling."""
    db.add_keyword('alpha"beta')
    rows = db.search_keywords('alpha"beta')
    assert [row["keyword"] for row in rows] == ['alpha"beta']


def test_keyword_collection_search_is_swept_too(db: CharactersRAGDB) -> None:
    db.add_keyword_collection('Set "A"')
    rows = db.search_keyword_collections('Set "A"')
    assert [row["name"] for row in rows] == ['Set "A"']
    assert db.search_keyword_collections(ORDINARY_QUOTED) == []


def test_conversation_content_search_no_longer_raises(db: CharactersRAGDB) -> None:
    _seed_for_dead_store(db)
    assert db.search_conversations_by_content('hello"x') == []
    assert len(db.search_conversations_by_content("hello")) == 1


def test_every_chachanotes_search_seam_survives_a_quote(
    db: CharactersRAGDB,
) -> None:
    """One sweep over the whole family: none of them raises on `"`.

    A quote is ordinary text a user types (an apostrophe-styled quotation, a
    pasted title). Before this task, six of these eight either raised or
    silently reinterpreted the query.
    """
    _seed_for_dead_store(db)
    db.add_note(title="n", content="c")
    db.add_keyword("k")
    deck_id = db.create_deck(name="D")
    db.create_flashcard({"deck_id": deck_id, "front": "f", "back": "b"})
    probes = (
        lambda q: db.search_notes(q),
        lambda q: db.search_character_cards(q),
        lambda q: db.search_conversations_by_title(q),
        lambda q: db.search_conversations_by_content(q),
        lambda q: db.search_messages_by_content(q),
        lambda q: db.search_keywords(q),
        lambda q: db.search_keyword_collections(q),
        lambda q: db.list_flashcards(q=q),
        lambda q: db.search_flashcards(q),
    )
    for probe in probes:
        assert probe(ORDINARY_QUOTED) == []
        assert probe(COLUMN_FILTER_INJECTION) == []


def test_library_note_and_conversation_search_pages_survive_a_quote(
    db: CharactersRAGDB,
) -> None:
    """The `_library_*_fts_query` token builders, through their real pages."""
    db.add_note(title="Quarterly report", content='alpha "beta" gamma')
    page = db.search_library_notes_page(query=ORDINARY_QUOTED, limit=10, offset=0)
    assert page["items"] == [] and page["total"] == 0
    hit = db.search_library_notes_page(query="beta", limit=10, offset=0)
    assert [item["title"] for item in hit["items"]] == ["Quarterly report"]


# ---------------------------------------------------------------------------
# The Evals seam.
# ---------------------------------------------------------------------------


@pytest.fixture()
def evals_db() -> EvalsDB:
    return EvalsDB(db_path=":memory:", client_id="t19558")


def test_dataset_search_no_longer_raises_on_a_quote(evals_db: EvalsDB) -> None:
    """`search_datasets` wrapped the RAW query in quotes with no doubling."""
    evals_db.create_dataset(
        name='Bench "gold" set', format="json", source_path="/tmp/x.json"
    )
    assert evals_db.search_datasets(ORDINARY_QUOTED) == []
    assert [row["name"] for row in evals_db.search_datasets("gold")] == [
        'Bench "gold" set'
    ]


def test_dataset_search_does_not_execute_a_typed_column_filter(
    evals_db: EvalsDB,
) -> None:
    evals_db.create_dataset(name="Alpha set", format="json", source_path="/tmp/a.json")
    evals_db.create_dataset(name="Other set", format="json", source_path="/tmp/o.json")
    assert evals_db.search_datasets('Alpha" OR name:"Other') == []


def test_task_search_still_finds_plain_terms(evals_db: EvalsDB) -> None:
    evals_db.create_task(
        name="math evaluation",
        description="arithmetic benchmark",
        task_type="question_answer",
        config_format="custom",
        config_data={},
    )
    assert [row["name"] for row in evals_db.search_tasks("math")] == [
        "math evaluation"
    ]


# ---------------------------------------------------------------------------
# The Media seam.
# ---------------------------------------------------------------------------


@pytest.fixture()
def media_db(tmp_path: Path):
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

    database = MediaDatabase(db_path=tmp_path / "media.db", client_id="t19558")
    yield database
    database.close_connection()


def test_media_search_no_longer_raises_on_a_quote(media_db) -> None:
    media_db.add_media_with_keywords(
        title='The "gold" standard',
        media_type="document",
        content="alpha beta gamma",
        keywords=["kw"],
        url="https://example.invalid/a",
    )
    rows, total = media_db.search_media_db(
        search_query=ORDINARY_QUOTED,
        search_fields=["title", "content"],
        page=1,
        results_per_page=10,
    )
    assert (rows, total) == ([], 0)


def test_media_search_still_finds_a_plain_term(media_db) -> None:
    media_db.add_media_with_keywords(
        title='The "gold" standard',
        media_type="document",
        content="alpha beta gamma",
        keywords=["kw"],
        url="https://example.invalid/a",
    )
    rows, total = media_db.search_media_db(
        search_query="gold",
        search_fields=["title", "content"],
        page=1,
        results_per_page=10,
    )
    assert total == 1 and rows[0]["title"] == 'The "gold" standard'


def test_media_search_short_term_prefix_widening_is_preserved(media_db) -> None:
    """The 1-2 character prefix branch survives the quoting change.

    It is measured on the RAW length, not on the quoted string's -- quoting
    adds two characters, so testing the quoted length would have silently
    retired the branch.
    """
    media_db.add_media_with_keywords(
        title="Gorgonzola notes",
        media_type="document",
        content="cheese",
        keywords=["kw"],
        url="https://example.invalid/g",
    )
    rows, total = media_db.search_media_db(
        search_query="Go",
        search_fields=["title"],
        page=1,
        results_per_page=10,
    )
    assert total == 1 and rows[0]["title"] == "Gorgonzola notes"


def test_no_search_seam_leaks_a_raw_operationalerror(db: CharactersRAGDB) -> None:
    """Whatever a seam does with bad input, it is never a bare sqlite error.

    `sqlite3.OperationalError` reaching a UI handler is what the Study
    screen shipped; the DB layer's own wrapper (`CharactersRAGDBError`) is
    the contract every caller is written against.
    """
    db.add_note(title="n", content="c")
    for query in (ORDINARY_QUOTED, COLUMN_FILTER_INJECTION, '"', '""', 'a"b"c'):
        try:
            db.search_notes(query)
            db.list_flashcards(q=query)
        except sqlite3.OperationalError as exc:  # pragma: no cover - guard
            pytest.fail(f"raw sqlite error for {query!r}: {exc}")
