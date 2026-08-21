"""v31 -> v32: enrich the seeded 'Default Assistant' character card (id=1)
with documentation-grade content (task-2451), plus the pre-existing FTS5
shadow-index defect this task's own enrichment write surfaced.

Two things are under test:

1. Enrichment itself: a fresh database seeds the rich content directly; an
   existing database is promoted from bare to rich ONLY when every
   user-editable content field on row 1 is still byte-identical to the
   original bare-seed literals -- any single edited field (anywhere on the
   row, not just the fields this migration writes) leaves it untouched.

2. A regression fix discovered while building (1): row 1's INSERT in
   ``_FULL_SCHEMA_SQL_V4`` used to run BEFORE ``character_cards_fts`` and its
   ``character_cards_ai`` trigger existed, so row 1 was never indexed into
   the FTS5 shadow tables on ANY database ever created by this schema. The
   first UPDATE to that row (this migration's own enrichment write, or an
   ordinary user edit via ``update_character_card``) made the
   ``character_cards_au`` trigger ask FTS5 to remove index entries that were
   never inserted, which raises ``sqlite3.DatabaseError: database disk image
   is malformed`` (SQLITE_CORRUPT_VTAB). Fixed by reordering the schema SQL
   (fresh databases) and rebuilding the FTS5 index inside the shared
   enrichment routine (existing databases).
"""

import sqlite3
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


# Matches CharactersRAGDB._SCHEMA_NAME, per the sibling migration tests
# (Tests/DB/test_chachanotes_message_metadata_migration.py and friends).
SCHEMA_NAME = "rag_char_chat_schema"

BARE_NAME = "Default Assistant"
BARE_DESCRIPTION = "A general-purpose assistant."
BARE_FIRST_MESSAGE = "Hello! How can I help you today?"
BARE_ALTERNATE_GREETINGS = "[]"
BARE_TAGS = "[]"
BARE_CREATOR = "System"
BARE_CHARACTER_VERSION = "1.0"
BARE_EXTENSIONS = "{}"


def _version(connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (SCHEMA_NAME,),
    ).fetchone()
    return int(row[0])


def _force_row1_bare(connection) -> None:
    """Rewrite row 1 back to the ORIGINAL bare-seed literals via a raw
    connection, simulating a database created before task-2451's fresh-seed
    enrichment existed. Necessary because, post-fix, even a database
    monkeypatched to schema version 31 gets the rich content directly at
    creation time (``_apply_schema_v4`` always enriches a fresh bare row,
    independent of ``_CURRENT_SCHEMA_VERSION``) -- so the constructor alone
    can no longer produce the legacy "existing bare v31 database" state this
    migration exists to handle.
    """
    connection.execute(
        """
        UPDATE character_cards
           SET name = ?,
               description = ?,
               personality = NULL,
               scenario = NULL,
               system_prompt = NULL,
               image = NULL,
               post_history_instructions = NULL,
               first_message = ?,
               message_example = NULL,
               creator_notes = NULL,
               alternate_greetings = ?,
               tags = ?,
               creator = ?,
               character_version = ?,
               extensions = ?,
               version = 1
         WHERE id = 1
        """,
        (
            BARE_NAME,
            BARE_DESCRIPTION,
            BARE_FIRST_MESSAGE,
            BARE_ALTERNATE_GREETINGS,
            BARE_TAGS,
            BARE_CREATOR,
            BARE_CHARACTER_VERSION,
            BARE_EXTENSIONS,
        ),
    )
    connection.commit()


def _seed_v31_database_with_bare_row(
    path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as v31_patch:
        v31_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 31)
        db = CharactersRAGDB(path, client_id="migration-seed")
        connection = db.get_connection()
        assert _version(connection) == 31
        _force_row1_bare(connection)
        row = db.get_character_card_by_id(1)
        assert row["description"] == BARE_DESCRIPTION
        assert row["version"] == 1
        db.close_connection()


# --------------------------------------------------------------------------
# Fresh database: seeds rich content directly
# --------------------------------------------------------------------------


def test_fresh_database_seeds_rich_content(tmp_path):
    db = CharactersRAGDB(tmp_path / "fresh.db", client_id="fresh-test")
    row = db.get_character_card_by_id(1)

    assert row["name"] == BARE_NAME  # FK anchor stays stable
    assert row["creator"] == BARE_CREATOR  # provenance stays stable
    assert row["description"] != BARE_DESCRIPTION
    assert row["description"].startswith("The built-in Default Assistant character")
    assert row["personality"]
    assert row["system_prompt"]
    assert "{{char}}" in row["system_prompt"]
    assert row["first_message"] != BARE_FIRST_MESSAGE
    assert "Roleplay" in row["first_message"]
    assert "Voice & Speech" in row["first_message"]
    assert row["creator_notes"]
    assert row["alternate_greetings"] != BARE_ALTERNATE_GREETINGS

    connection = db.get_connection()
    assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
    db.close_connection()


def test_fresh_database_alternate_greetings_is_a_two_item_json_list(tmp_path):
    # get_character_card_by_id deserializes JSON columns (_CHARACTER_CARD_JSON_FIELDS)
    # into native Python objects, so `alternate_greetings` comes back as a list already.
    db = CharactersRAGDB(tmp_path / "fresh.db", client_id="fresh-test")
    row = db.get_character_card_by_id(1)
    greetings = row["alternate_greetings"]
    assert isinstance(greetings, list)
    assert len(greetings) == 2
    assert all(isinstance(g, str) and g.strip() for g in greetings)
    db.close_connection()


# --------------------------------------------------------------------------
# Migration: enriches an untouched bare row
# --------------------------------------------------------------------------


def test_migration_enriches_untouched_bare_row_and_bumps_version(tmp_path, monkeypatch):
    db_path = tmp_path / "chachanotes.db"
    _seed_v31_database_with_bare_row(db_path, monkeypatch)

    db = CharactersRAGDB(db_path, client_id="migration-test")
    connection = db.get_connection()
    assert _version(connection) == CharactersRAGDB._CURRENT_SCHEMA_VERSION

    row = db.get_character_card_by_id(1)
    assert row["name"] == BARE_NAME
    assert row["creator"] == BARE_CREATOR
    assert row["description"] != BARE_DESCRIPTION
    assert row["first_message"] != BARE_FIRST_MESSAGE
    assert row["personality"]
    assert row["system_prompt"]
    assert row["creator_notes"]
    # The enrichment write itself bumps version (1 -> 2), matching how a
    # real update_character_card() call represents a content change.
    assert row["version"] == 2
    db.close_connection()


def test_migration_untouched_fields_not_written_stay_bare(tmp_path, monkeypatch):
    """scenario, message_example, tags, character_version, extensions, image,
    and post_history_instructions are not part of the authored rich content
    (task-2451 step 2) and must stay exactly as the bare seed left them."""
    db_path = tmp_path / "chachanotes.db"
    _seed_v31_database_with_bare_row(db_path, monkeypatch)

    db = CharactersRAGDB(db_path, client_id="migration-test")
    row = db.get_character_card_by_id(1)
    assert row["scenario"] is None
    assert row["message_example"] is None
    assert row["image"] is None
    assert row["post_history_instructions"] is None
    # tags/extensions are JSON columns; get_character_card_by_id deserializes them.
    assert row["tags"] == []
    assert row["character_version"] == BARE_CHARACTER_VERSION
    assert row["extensions"] == {}
    db.close_connection()


# --------------------------------------------------------------------------
# Migration: preserves a row with any single field edited
# --------------------------------------------------------------------------


_EDIT_CASES = [
    ("name", "My Assistant"),
    ("description", "Something I wrote myself."),
    ("personality", "Grumpy but helpful."),
    ("scenario", "A cozy cabin in the woods."),
    ("system_prompt", "Always answer in haiku."),
    ("image", b"\x89PNG\r\n\x1a\n" + b"fake-avatar-bytes"),
    ("post_history_instructions", "Always end with a one-line summary."),
    ("first_message", "Yo, what's up?"),
    ("message_example", "<START>\n{{user}}: Hi\n{{char}}: Hi!"),
    ("creator_notes", "I made this the way I like it."),
    ("alternate_greetings", '["Hey there!"]'),
    ("tags", '["custom"]'),
    ("creator", "SomeUser"),
    ("character_version", "2.0"),
    ("extensions", '{"custom_key": true}'),
]


# get_character_card_by_id deserializes these three JSON columns into native
# Python objects (_CHARACTER_CARD_JSON_FIELDS), so the value read back through
# the API differs in representation from the raw TEXT written via SQL.
_JSON_FIELDS = {"alternate_greetings", "tags", "extensions"}


@pytest.mark.parametrize(
    "field, edited_value", _EDIT_CASES, ids=[c[0] for c in _EDIT_CASES]
)
def test_migration_preserves_row_with_single_field_edited(
    tmp_path, monkeypatch, field, edited_value
):
    import json

    expected_value = json.loads(edited_value) if field in _JSON_FIELDS else edited_value

    db_path = tmp_path / "chachanotes.db"
    _seed_v31_database_with_bare_row(db_path, monkeypatch)

    with monkeypatch.context() as v31_patch:
        v31_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 31)
        db = CharactersRAGDB(db_path, client_id="edit-seed")
        connection = db.get_connection()
        # `image` is the one BLOB column here (everything else is TEXT).
        # sqlite3 binds a Python `bytes` object as BLOB, but that's an
        # implementation detail worth making explicit rather than relying
        # on -- the two branches deliberately mirror the two SQLite storage
        # classes this fixture actually exercises.
        if isinstance(edited_value, bytes):
            connection.execute(
                f"UPDATE character_cards SET {field} = ? WHERE id = 1",
                (sqlite3.Binary(edited_value),),
            )
        else:
            connection.execute(
                f"UPDATE character_cards SET {field} = ? WHERE id = 1", (edited_value,)
            )
        connection.commit()
        pre_migration_row = db.get_character_card_by_id(1)
        assert pre_migration_row[field] == expected_value
        pre_migration_version = pre_migration_row["version"]
        db.close_connection()

    db2 = CharactersRAGDB(db_path, client_id="migration-test")
    connection2 = db2.get_connection()
    assert _version(connection2) == CharactersRAGDB._CURRENT_SCHEMA_VERSION

    post_migration_row = db2.get_character_card_by_id(1)
    # The edited field survives untouched.
    assert post_migration_row[field] == expected_value
    # Every OTHER field is exactly what it was before the migration ran --
    # the whole row is left alone, not just the edited field.
    for key in pre_migration_row.keys():
        if key in ("last_modified",):
            continue
        assert post_migration_row[key] == pre_migration_row[key], (
            f"field {key!r} changed even though the row had an edit to "
            f"{field!r} and should have been left completely untouched"
        )
    assert post_migration_row["version"] == pre_migration_version
    db2.close_connection()


def test_migration_preserves_deleted_row(tmp_path, monkeypatch):
    """A soft-deleted row 1 (deleted=1) must never be silently un-deleted or
    enriched by the migration."""
    db_path = tmp_path / "chachanotes.db"
    _seed_v31_database_with_bare_row(db_path, monkeypatch)

    with monkeypatch.context() as v31_patch:
        v31_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 31)
        db = CharactersRAGDB(db_path, client_id="delete-seed")
        connection = db.get_connection()
        connection.execute("UPDATE character_cards SET deleted = 1 WHERE id = 1")
        connection.commit()
        db.close_connection()

    db2 = CharactersRAGDB(db_path, client_id="migration-test")
    connection2 = db2.get_connection()
    assert _version(connection2) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
    row = connection2.execute(
        "SELECT description, deleted FROM character_cards WHERE id = 1"
    ).fetchone()
    assert row["deleted"] == 1
    assert row["description"] == BARE_DESCRIPTION  # untouched, still bare
    db2.close_connection()


# --------------------------------------------------------------------------
# Idempotence
# --------------------------------------------------------------------------


def test_migration_is_idempotent_on_already_rich_row(tmp_path):
    """A fresh database is already rich by the time it reaches current schema (the
    enrichment happens at seed time, not via the migration). Re-running the
    same conditional UPDATE (as a second migration attempt would) must be a
    no-op: the WHERE clause no longer matches an enriched row."""
    db = CharactersRAGDB(tmp_path / "fresh.db", client_id="idempotence-test")
    row_before = db.get_character_card_by_id(1)

    conn = db.get_connection()
    db._enrich_default_assistant_card_if_bare(conn)
    conn.commit()

    row_after = db.get_character_card_by_id(1)
    assert row_after == row_before
    db.close_connection()


def test_reopening_a_migrated_database_does_not_touch_row_again(tmp_path, monkeypatch):
    db_path = tmp_path / "chachanotes.db"
    _seed_v31_database_with_bare_row(db_path, monkeypatch)

    db = CharactersRAGDB(db_path, client_id="migration-test")
    row_after_migration = db.get_character_card_by_id(1)
    db.close_connection()

    db2 = CharactersRAGDB(db_path, client_id="reopen-test")
    row_after_reopen = db2.get_character_card_by_id(1)
    assert row_after_reopen == row_after_migration
    db2.close_connection()


# --------------------------------------------------------------------------
# Regression: the FTS5 shadow-index defect this migration's write surfaced
# --------------------------------------------------------------------------


def test_first_edit_of_row_1_succeeds_on_a_fresh_database(tmp_path):
    """Before the fix, ANY update to row 1 raised SQLITE_CORRUPT_VTAB
    ("database disk image is malformed") because row 1's INSERT ran before
    character_cards_fts/its triggers existed, so it was never indexed."""
    db = CharactersRAGDB(tmp_path / "fresh.db", client_id="edit-test")
    row = db.get_character_card_by_id(1)

    result = db.update_character_card(
        1, {"description": "my own description"}, expected_version=row["version"]
    )
    assert result is True

    updated = db.get_character_card_by_id(1)
    assert updated["description"] == "my own description"
    db.close_connection()


def test_first_edit_of_row_1_succeeds_on_a_migrated_pre_existing_database(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "chachanotes.db"
    _seed_v31_database_with_bare_row(db_path, monkeypatch)

    db = CharactersRAGDB(db_path, client_id="migration-test")
    row = db.get_character_card_by_id(1)

    result = db.update_character_card(
        1,
        {"description": "second edit after migration"},
        expected_version=row["version"],
    )
    assert result is True
    db.close_connection()


def test_fts_search_finds_row_1_after_enrichment(tmp_path):
    """Row 1 must be genuinely indexed (not just readable via a plain
    SELECT, which FTS5 can satisfy from the content table alone even when
    the index itself has no entry for that docid)."""
    db = CharactersRAGDB(tmp_path / "fresh.db", client_id="search-test")
    hits = db.search_character_cards("worked example")
    assert hits, "expected the enriched Default Assistant card to be findable via FTS"
    assert any(hit["id"] == 1 for hit in hits)
    db.close_connection()


def test_fts_search_finds_row_1_after_migration_from_bare(tmp_path, monkeypatch):
    db_path = tmp_path / "chachanotes.db"
    _seed_v31_database_with_bare_row(db_path, monkeypatch)

    db = CharactersRAGDB(db_path, client_id="migration-test")
    hits = db.search_character_cards("worked example")
    assert hits
    assert any(hit["id"] == 1 for hit in hits)
    db.close_connection()


def test_fts_rebuild_does_not_disturb_other_character_cards(tmp_path, monkeypatch):
    """The enrichment routine's defensive `rebuild` reconstructs the WHOLE
    FTS5 index, not just row 1's entry -- verify a second, independently
    inserted character card is still searchable afterward."""
    db_path = tmp_path / "chachanotes.db"
    _seed_v31_database_with_bare_row(db_path, monkeypatch)

    with monkeypatch.context() as v31_patch:
        v31_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 31)
        db = CharactersRAGDB(db_path, client_id="second-card-seed")
        new_id = db.add_character_card(
            {"name": "Zorblax", "description": "A xenophobic space pirate."}
        )
        assert new_id is not None
        db.close_connection()

    db2 = CharactersRAGDB(db_path, client_id="migration-test")
    hits = db2.search_character_cards("space pirate")
    assert any(hit["id"] == new_id for hit in hits)
    db2.close_connection()


# --------------------------------------------------------------------------
# Regular chat / FK behavior unchanged
# --------------------------------------------------------------------------


def test_conversation_and_message_against_character_id_1_still_works(tmp_path):
    db = CharactersRAGDB(tmp_path / "fresh.db", client_id="fk-test")
    conv_id = db.add_conversation({"title": "chat with default", "character_id": 1})
    assert conv_id
    msg_id = db.add_message(
        {"conversation_id": conv_id, "sender": "user", "content": "hi there"}
    )
    assert msg_id

    conv = db.get_conversation_by_id(conv_id)
    assert conv["character_id"] == 1
    db.close_connection()


# --------------------------------------------------------------------------
# Schema version arithmetic
# --------------------------------------------------------------------------


def test_current_schema_version_is_current():
    # Pinned dynamically: this sibling test exists to catch an accidental
    # DOWNGRADE, not to be re-edited on every legitimate bump (it has been
    # re-hardcoded at 35, 39, 40, 41 ...).
    assert CharactersRAGDB._CURRENT_SCHEMA_VERSION >= 41


def test_migrate_from_v31_to_v32_requires_version_31(tmp_path):
    db = CharactersRAGDB(tmp_path / "fresh.db", client_id="version-test")
    conn = db.get_connection()
    from tldw_chatbook.DB.ChaChaNotes_DB import SchemaError

    with pytest.raises(SchemaError):
        db._migrate_from_v31_to_v32(conn)
    db.close_connection()
