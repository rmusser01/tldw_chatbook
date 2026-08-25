"""v42 -> v43 private Research Quick Note owner-proof migration contracts."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService


SCHEMA_NAME = "rag_char_chat_schema"
TABLE_NAME = "research_quick_note_owner_proofs"
LEGACY_PROOF_PREFIX = "research-receipt-proof:"
EXACT_LEGACY_PROOF_KEYWORD = LEGACY_PROOF_PREFIX + "c" * 64
ORDINARY_NEAR_PROOF_TAGS = (
    "Research-receipt-proof:" + "d" * 64,
    LEGACY_PROOF_PREFIX + "E" * 64,
    LEGACY_PROOF_PREFIX + "f" * 63,
    LEGACY_PROOF_PREFIX + "1" * 65,
    LEGACY_PROOF_PREFIX + "2" * 63 + "g",
    LEGACY_PROOF_PREFIX + "project-alpha",
)


def _version(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (SCHEMA_NAME,),
    ).fetchone()
    return int(row[0])


def _seed_v42_database(path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    """Create a genuine v42 owner database before the private proof table exists."""

    with monkeypatch.context() as v42_patch:
        v42_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 42)
        db = CharactersRAGDB(path, client_id="migration-seed")
        note_id = str(db.add_note("Preserve note", "Preserve body"))
        conversation_id = str(db.add_conversation({"title": "Preserve chat"}))
        legacy_keyword_id = db.add_keyword(EXACT_LEGACY_PROOF_KEYWORD)
        assert legacy_keyword_id is not None
        assert db.link_note_to_keyword(note_id, legacy_keyword_id)
        for ordinary_tag in ORDINARY_NEAR_PROOF_TAGS:
            ordinary_keyword_id = db.add_keyword(ordinary_tag)
            assert ordinary_keyword_id is not None
            assert db.link_note_to_keyword(note_id, ordinary_keyword_id)
        connection = db.get_connection()
        # The canonical full bootstrap is forward-shaped even when its target
        # version is patched for a historical fixture; remove the v43 table to
        # restore the genuine v42 artifact before reopening it.
        connection.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        connection.commit()
        assert _version(connection) == 42
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (TABLE_NAME,),
            ).fetchone()
            is None
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sync_log WHERE payload LIKE ?",
                (f"%{EXACT_LEGACY_PROOF_KEYWORD}%",),
            ).fetchone()[0]
            == 1
        )
        db.close_connection()
    return note_id, conversation_id


@pytest.mark.asyncio
async def test_v42_to_v43_purges_only_exact_proof_and_preserves_near_prefix_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "proof-migration.sqlite"
    note_id, conversation_id = _seed_v42_database(path, monkeypatch)

    db = CharactersRAGDB(path, client_id="migration-test")
    connection = db.get_connection()

    assert _version(connection) == 43
    assert (
        connection.execute(
            "SELECT title FROM notes WHERE id = ?", (note_id,)
        ).fetchone()[0]
        == "Preserve note"
    )
    assert (
        connection.execute(
            "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()[0]
        == "Preserve chat"
    )
    columns = {
        str(row[1]): row
        for row in connection.execute(f"PRAGMA table_info({TABLE_NAME})").fetchall()
    }
    assert set(columns) == {"note_id", "owner_proof", "created_at"}
    assert columns["note_id"][5] == 1
    assert columns["owner_proof"][3] == 1
    assert connection.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0] == 1
    assert (
        connection.execute(
            f"SELECT owner_proof FROM {TABLE_NAME} WHERE note_id = ?", (note_id,)
        ).fetchone()[0]
        == "c" * 64
    )
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM keywords WHERE keyword = ? COLLATE BINARY",
            (EXACT_LEGACY_PROOF_KEYWORD,),
        ).fetchone()[0]
        == 0
    )
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM sync_log WHERE payload LIKE ?",
            (f"%{EXACT_LEGACY_PROOF_KEYWORD}%",),
        ).fetchone()[0]
        == 0
    )
    raw_keywords = {str(row["keyword"]) for row in db.get_keywords_for_note(note_id)}
    assert raw_keywords == set(ORDINARY_NEAR_PROOF_TAGS)
    raw_listed_keywords = {
        str(row["keyword"]) for row in db.list_keywords(limit=100, offset=0)
    }
    assert set(ORDINARY_NEAR_PROOF_TAGS) <= raw_listed_keywords
    for ordinary_tag in ORDINARY_NEAR_PROOF_TAGS:
        ordinary_row = connection.execute(
            "SELECT id FROM keywords WHERE keyword = ? COLLATE BINARY",
            (ordinary_tag,),
        ).fetchone()
        assert ordinary_row is not None
        ordinary_keyword_id = int(ordinary_row[0])
        assert (
            connection.execute(
                "SELECT 1 FROM note_keywords WHERE note_id = ? AND keyword_id = ?",
                (note_id, ordinary_keyword_id),
            ).fetchone()
            is not None
        )
        assert (
            connection.execute(
                "SELECT 1 FROM sync_log WHERE entity = 'keywords' "
                "AND entity_id = ? AND payload LIKE ?",
                (str(ordinary_keyword_id), f"%{ordinary_tag}%"),
            ).fetchone()
            is not None
        )
        assert (
            connection.execute(
                "SELECT 1 FROM sync_log WHERE entity = 'note_keywords' "
                "AND entity_id = ?",
                (f"{note_id}_{ordinary_keyword_id}",),
            ).fetchone()
            is not None
        )

    interop = NotesInteropService(
        base_db_directory=tmp_path,
        api_client_id="research-client",
        global_db_to_use=db,
    )
    scope = NotesScopeService(
        local_notes_service=interop,
        server_service=None,
        policy_enforcer=None,
    )
    assert {
        str(row["keyword"])
        for row in interop.get_keywords_for_note("notes-user", note_id)
    } == set(ORDINARY_NEAR_PROOF_TAGS)
    assert set(
        await scope.get_note_keywords(
            scope="local_note", note_id=note_id, user_id="notes-user"
        )
    ) == set(ORDINARY_NEAR_PROOF_TAGS)
    library_page = interop.list_library_notes("notes-user", limit=20, offset=0)
    assert set(library_page["items"][0]["keywords"]) == set(ORDINARY_NEAR_PROOF_TAGS)
    search_page = interop.search_library_notes(
        "notes-user", query="project-alpha", limit=20, offset=0
    )
    assert search_page["total"] == 1
    assert ORDINARY_NEAR_PROOF_TAGS[-1] in search_page["items"][0]["matched_keywords"]

    new_note_id = "research-note-123e4567e89b42d3a456426614174000"
    uppercase_prefix_tag = "Research-receipt-proof:" + "9" * 64
    await scope.save_note(
        scope="local_note",
        note_id=None,
        create_note_id=new_note_id,
        title="Case-sensitive ordinary tag",
        content="Body",
        keywords=[uppercase_prefix_tag],
        version=None,
        user_id="notes-user",
    )
    assert await scope.get_note_keywords(
        scope="local_note", note_id=new_note_id, user_id="notes-user"
    ) == [uppercase_prefix_tag]
    db.close_connection()


def test_v42_to_v43_inline_sql_matches_artifact_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    migration_path = (
        Path(__file__).parents[2]
        / "tldw_chatbook/DB/migrations/chachanotes_v42_to_v43_research_quick_note_proofs.sql"
    )
    assert (
        migration_path.read_text(encoding="utf-8")
        == CharactersRAGDB._MIGRATE_V42_TO_V43_SQL
    )

    path = tmp_path / "proof-rollback.sqlite"
    _seed_v42_database(path, monkeypatch)
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TRIGGER block_v43_version_update
        BEFORE UPDATE OF version ON db_schema_version
        WHEN OLD.schema_name = 'rag_char_chat_schema'
          AND OLD.version = 42
          AND NEW.version = 43
        BEGIN
            SELECT RAISE(ABORT, 'blocked version update');
        END
        """
    )
    connection.commit()
    connection.close()

    with pytest.raises(CharactersRAGDBError, match="Migration from V42 to V43 failed"):
        CharactersRAGDB(path, client_id="failed-migration")

    with sqlite3.connect(path) as connection:
        assert _version(connection) == 42
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (TABLE_NAME,),
            ).fetchone()
            is None
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM keywords WHERE keyword = ? COLLATE BINARY",
                (EXACT_LEGACY_PROOF_KEYWORD,),
            ).fetchone()[0]
            == 1
        )
        assert connection.execute(
            "SELECT COUNT(*) FROM keywords WHERE keyword <> ? COLLATE BINARY "
            "AND keyword LIKE ?",
            (EXACT_LEGACY_PROOF_KEYWORD, f"{LEGACY_PROOF_PREFIX}%"),
        ).fetchone()[0] == len(ORDINARY_NEAR_PROOF_TAGS)


@pytest.mark.parametrize("invalid_proof", ["a" * 63, "A" * 64, "g" * 64])
def test_private_proof_table_rejects_noncanonical_hashes(
    tmp_path: Path, invalid_proof: str
) -> None:
    db = CharactersRAGDB(tmp_path / "proof-check.sqlite", client_id="proof-test")
    note_id = str(db.add_note("Owner", "Body"))
    with pytest.raises(sqlite3.IntegrityError):
        with db.transaction() as connection:
            connection.execute(
                f"INSERT INTO {TABLE_NAME} (note_id, owner_proof) VALUES (?, ?)",
                (note_id, invalid_proof),
            )
    db.close_connection()


def test_fresh_schema_contains_private_proof_table_without_sync_triggers(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(tmp_path / "proof-fresh.sqlite", client_id="proof-test")
    connection = db.get_connection()

    assert CharactersRAGDB._CURRENT_SCHEMA_VERSION == 43
    assert _version(connection) == 43
    assert (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (TABLE_NAME,),
        ).fetchone()
        is not None
    )
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger' AND tbl_name = ?",
            (TABLE_NAME,),
        ).fetchone()[0]
        == 0
    )
    db.close_connection()


def test_private_proof_crud_is_exact_payload_free_and_cascades(tmp_path: Path) -> None:
    db = CharactersRAGDB(tmp_path / "proof-crud.sqlite", client_id="proof-test")
    note_id = str(db.add_note("Owner", "Body"))
    proof = "a" * 64
    other_proof = "b" * 64
    connection = db.get_connection()
    try:
        assert db.add_research_quick_note_owner_proof(note_id, proof)
        assert db.has_research_quick_note_owner_proof(note_id, proof)
        assert not db.has_research_quick_note_owner_proof(note_id, other_proof)
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sync_log WHERE payload LIKE ?", (f"%{proof}%",)
            ).fetchone()[0]
            == 0
        )
        assert not db.remove_research_quick_note_owner_proof(note_id, other_proof)
        assert db.remove_research_quick_note_owner_proof(note_id, proof)
        assert not db.has_research_quick_note_owner_proof(note_id, proof)

        assert db.add_research_quick_note_owner_proof(note_id, proof)
        with db.transaction() as cursor:
            cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        assert not db.has_research_quick_note_owner_proof(note_id, proof)
    finally:
        db.close_connection()
