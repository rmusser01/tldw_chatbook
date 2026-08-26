"""Behavioural witnesses for the FTS soft-delete guards (task-19567 B).

Why this exists -- the incident, not the rule. Lane 5 of the 2026-08-21
holistic review mutated ``WHERE new.deleted = 0`` out of the ``messages_au``
trigger and **475 tests across every FTS-adjacent file still passed**. Tests
named for the behaviour exist (``Tests/DB/test_search_conversations_fts.py``
``test_soft_deleted_message_does_not_match``), and none of them could catch it,
because all four production ``messages_fts`` consumers redundantly re-filter at
query level -- so the trigger regression is invisible through every shipped
API. The only tests that touched ``messages_fts`` directly
(``test_chachanotes_provider_continuation_migration.py``) compare a before and
an after snapshot for **equality**, and a uniform mutation moves both sides
together.

So every assertion here queries the FTS index **directly**, never through a
consumer, and every assertion is an **absolute** expectation (``== []`` /
``== [id]``), never a before/after equality snapshot.

Mutation proof for these tests is recorded in task-19567: deleting the
``WHERE new.deleted = 0`` line from ``messages_au`` in the shipped schema turns
the messages cases red. The equivalent holds for each sibling.

The census at the bottom is the "any FTS consumer added later inherits the
guarantee" net: the coverage lives with the trigger, so a NEW soft-deletable
FTS-backed table cannot ship without a guard even if nobody thinks to write a
behavioural test for it.
"""

from __future__ import annotations

import re
import sqlite3
import uuid
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture()
def db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(tmp_path / "witness.db", "witness_client")
    try:
        yield database
    finally:
        database.close_connection()


def _needle() -> str:
    """A token no other row can contain, safe for an FTS MATCH."""
    return "zqx" + uuid.uuid4().hex[:12]


def _fts_rowids(db: CharactersRAGDB, table: str, needle: str) -> list[int]:
    """Rowids the FTS index itself returns -- no join, no ``deleted`` filter.

    This is the whole point of the module: a consumer's ``WHERE m.deleted = 0``
    cannot mask a regression here, because there is no consumer.
    """
    return [
        row[0]
        for row in db.execute_query(
            f"SELECT rowid FROM {table} WHERE {table} MATCH ?", (needle,)
        ).fetchall()
    ]


# ---------------------------------------------------------------------------
# messages -- the trigger Lane 5 actually mutated
# ---------------------------------------------------------------------------
def test_messages_fts_index_drops_a_soft_deleted_message(db: CharactersRAGDB):
    needle = _needle()
    conversation_id = db.add_conversation({"title": "witness", "character_id": 1})
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": f"body carrying {needle} inside",
        }
    )
    rowid = db.execute_query(
        "SELECT rowid FROM messages WHERE id = ?", (message_id,)
    ).fetchone()[0]
    assert _fts_rowids(db, "messages_fts", needle) == [rowid]

    db.soft_delete_message(message_id, expected_version=1)

    assert _fts_rowids(db, "messages_fts", needle) == []


def test_messages_fts_index_drops_a_message_deleted_by_raw_update(
    db: CharactersRAGDB,
):
    """The guard, not the Python method, is what has to hold.

    ``soft_delete_message`` could grow an explicit index maintenance step and
    keep the previous test green with the trigger broken. A bare ``UPDATE``
    leaves only the trigger in the path.
    """
    needle = _needle()
    conversation_id = db.add_conversation({"title": "witness", "character_id": 1})
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": f"raw path {needle}",
        }
    )
    with db.transaction() as conn:
        conn.execute(
            "UPDATE messages SET deleted = 1, version = version + 1 WHERE id = ?",
            (message_id,),
        )

    assert _fts_rowids(db, "messages_fts", needle) == []


def test_messages_fts_index_restores_an_undeleted_message(db: CharactersRAGDB):
    """The guard must not be satisfiable by never indexing anything."""
    needle = _needle()
    conversation_id = db.add_conversation({"title": "witness", "character_id": 1})
    message_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": f"round trip {needle}",
        }
    )
    rowid = db.execute_query(
        "SELECT rowid FROM messages WHERE id = ?", (message_id,)
    ).fetchone()[0]
    db.soft_delete_message(message_id, expected_version=1)
    assert _fts_rowids(db, "messages_fts", needle) == []

    with db.transaction() as conn:
        conn.execute(
            "UPDATE messages SET deleted = 0, version = version + 1 WHERE id = ?",
            (message_id,),
        )

    assert _fts_rowids(db, "messages_fts", needle) == [rowid]


# ---------------------------------------------------------------------------
# siblings -- confirmed to share the shape, and to have shared the gap
# ---------------------------------------------------------------------------
def test_notes_fts_index_drops_a_soft_deleted_note(db: CharactersRAGDB):
    needle = _needle()
    note_id = db.add_note(title="witness note", content=f"note body {needle}")
    rowid = db.execute_query(
        "SELECT rowid FROM notes WHERE id = ?", (note_id,)
    ).fetchone()[0]
    assert _fts_rowids(db, "notes_fts", needle) == [rowid]

    db.soft_delete_note(note_id, expected_version=1)

    assert _fts_rowids(db, "notes_fts", needle) == []


def test_conversations_fts_index_drops_a_soft_deleted_conversation(
    db: CharactersRAGDB,
):
    needle = _needle()
    conversation_id = db.add_conversation(
        {"title": f"title {needle}", "character_id": 1}
    )
    rowid = db.execute_query(
        "SELECT rowid FROM conversations WHERE id = ?", (conversation_id,)
    ).fetchone()[0]
    assert _fts_rowids(db, "conversations_fts", needle) == [rowid]

    db.soft_delete_conversation(conversation_id, expected_version=1)

    assert _fts_rowids(db, "conversations_fts", needle) == []


def test_character_cards_fts_index_drops_a_soft_deleted_card(db: CharactersRAGDB):
    needle = _needle()
    card_id = db.add_character_card(
        {"name": f"witness {needle}", "description": "witness card"}
    )
    assert _fts_rowids(db, "character_cards_fts", needle) == [card_id]

    db.soft_delete_character_card(card_id, expected_version=1)

    assert _fts_rowids(db, "character_cards_fts", needle) == []


def test_keywords_fts_index_drops_a_soft_deleted_keyword(db: CharactersRAGDB):
    needle = _needle()
    keyword_id = db.add_keyword(needle)
    assert _fts_rowids(db, "keywords_fts", needle) == [keyword_id]

    db.soft_delete_keyword(keyword_id, expected_version=1)

    assert _fts_rowids(db, "keywords_fts", needle) == []


def test_keyword_collections_fts_index_drops_a_soft_deleted_collection(
    db: CharactersRAGDB,
):
    needle = _needle()
    collection_id = db.add_keyword_collection(f"collection {needle}")
    assert _fts_rowids(db, "keyword_collections_fts", needle) == [collection_id]

    db.soft_delete_keyword_collection(collection_id, expected_version=1)

    assert _fts_rowids(db, "keyword_collections_fts", needle) == []


def test_re_adding_a_soft_deleted_keyword_collection_keeps_the_index_usable(
    db: CharactersRAGDB,
):
    """Regression: the unguarded DELETE half corrupted the index (task-19567).

    On the shipped code this raised ``sqlite3.DatabaseError: database disk
    image is malformed`` inside ``_add_generic_item``'s undelete UPDATE --
    ``keyword_collections_au`` issued the FTS ``'delete'`` for a row that was
    not in the index. Reached entirely through the public API.
    """
    needle = _needle()
    name = f"collection {needle}"
    collection_id = db.add_keyword_collection(name)
    db.soft_delete_keyword_collection(collection_id, expected_version=1)

    assert db.add_keyword_collection(name) == collection_id

    assert _fts_rowids(db, "keyword_collections_fts", needle) == [collection_id]


# `chat_dictionaries` and `world_books` are the two soft-deletable FTS-backed
# tables the filing did not list; the census below is what found them. They
# have no soft-delete method on this class, so the raw UPDATE is the honest
# shape of what production does to them.
def test_chat_dictionaries_fts_index_drops_a_soft_deleted_dictionary(
    db: CharactersRAGDB,
):
    needle = _needle()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO chat_dictionaries (name, description, content, client_id) "
            "VALUES (?, ?, ?, ?)",
            (f"witness {needle}", "witness dictionary", "entries", "witness_client"),
        )
        dictionary_id = conn.execute(
            "SELECT id FROM chat_dictionaries WHERE name = ?", (f"witness {needle}",)
        ).fetchone()[0]
    assert _fts_rowids(db, "chat_dictionaries_fts", needle) == [dictionary_id]

    with db.transaction() as conn:
        conn.execute(
            "UPDATE chat_dictionaries SET deleted = 1, version = version + 1 "
            "WHERE id = ?",
            (dictionary_id,),
        )

    assert _fts_rowids(db, "chat_dictionaries_fts", needle) == []


def test_world_books_fts_index_drops_a_soft_deleted_world_book(db: CharactersRAGDB):
    needle = _needle()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO world_books (name, description, client_id) VALUES (?, ?, ?)",
            (f"witness {needle}", "witness world book", "witness_client"),
        )
        book_id = conn.execute(
            "SELECT id FROM world_books WHERE name = ?", (f"witness {needle}",)
        ).fetchone()[0]
    assert _fts_rowids(db, "world_books_fts", needle) == [book_id]

    with db.transaction() as conn:
        conn.execute(
            "UPDATE world_books SET deleted = 1, version = version + 1 WHERE id = ?",
            (book_id,),
        )

    assert _fts_rowids(db, "world_books_fts", needle) == []

    # `world_books_au` had the same unguarded DELETE half as
    # `keyword_collections_au` (task-19567). It had no undelete path to reach
    # it, which made it latent rather than safe -- one restore API away.
    with db.transaction() as conn:
        conn.execute(
            "UPDATE world_books SET deleted = 0, version = version + 1 WHERE id = ?",
            (book_id,),
        )

    assert _fts_rowids(db, "world_books_fts", needle) == [book_id]


# ---------------------------------------------------------------------------
# the census -- so a table added later inherits the guarantee
# ---------------------------------------------------------------------------
_CONTENT_TABLE_RE = re.compile(r"content\s*=\s*'([^']+)'", re.IGNORECASE)

# Every external-content FTS5 table whose base table carries `deleted`, as of
# schema v45. The list is asserted against the live schema below, so adding a
# new one is a test failure until it also gets a behavioural witness above.
EXPECTED_SOFT_DELETABLE_FTS = {
    "character_cards_fts": "character_cards",
    "chat_dictionaries_fts": "chat_dictionaries",
    "conversations_fts": "conversations",
    "keyword_collections_fts": "keyword_collections",
    "keywords_fts": "keywords",
    "messages_fts": "messages",
    "notes_fts": "notes",
    "world_books_fts": "world_books",
}


def _soft_deletable_fts_tables(conn: sqlite3.Connection) -> dict[str, str]:
    found: dict[str, str] = {}
    for name, sql in conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND sql LIKE '%fts5%'"
    ).fetchall():
        match = _CONTENT_TABLE_RE.search(sql or "")
        if match is None:
            continue  # standalone FTS index; it owns its own rows
        base = match.group(1)
        columns = {
            row[1] for row in conn.execute(f"PRAGMA table_info({base})").fetchall()
        }
        if "deleted" in columns:
            found[name] = base
    return found


def test_the_set_of_soft_deletable_fts_tables_is_the_witnessed_set(
    db: CharactersRAGDB,
):
    """A new soft-deletable FTS table fails here until it gets a witness."""
    assert _soft_deletable_fts_tables(db.get_connection()) == (
        EXPECTED_SOFT_DELETABLE_FTS
    )


def test_every_soft_deletable_fts_table_has_a_guarded_after_update_trigger(
    db: CharactersRAGDB,
):
    """Structural backstop for the behavioural witnesses above.

    Both halves are checked. The INSERT half's ``new.deleted = 0`` is the leak
    guard. The DELETE half's ``old.deleted = 0`` is the corruption guard --
    issuing an FTS5 ``'delete'`` for a row that is not in an external-content
    index corrupts it, which is how ``add_keyword_collection`` on a
    soft-deleted name raised ``database disk image is malformed`` before
    task-19567.

    Deliberately secondary to the behavioural witnesses: a source-string
    assertion is what let ``test_uses_messages_fts_match`` pass while the guard
    was mutated away. Its job is to catch a NEW table whose author forgot a
    guard before anyone writes it a behavioural test.
    """
    conn = db.get_connection()
    findings: list[str] = []
    for fts_table, base in sorted(_soft_deletable_fts_tables(conn).items()):
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'trigger' AND tbl_name = ?",
            (base,),
        ).fetchall()
        after_update = [
            (name, sql)
            for name, sql in rows
            if sql and "AFTER UPDATE" in sql.upper() and fts_table in sql
        ]
        assert after_update, f"{fts_table} has no AFTER UPDATE trigger on {base}"
        for name, sql in after_update:
            normalized = " ".join(sql.lower().split())
            if "new.deleted = 0" not in normalized:
                findings.append(f"{name}: insert half unguarded")
            if "old.deleted = 0" not in normalized:
                findings.append(f"{name}: delete half unguarded")
    assert findings == []
