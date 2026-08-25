"""v42 -> v43 private Research Quick Note owner-proof migration contracts."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError


SCHEMA_NAME = "rag_char_chat_schema"
TABLE_NAME = "research_quick_note_owner_proofs"


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
        legacy_keyword_id = db.add_keyword("research-receipt-proof:" + "c" * 64)
        assert legacy_keyword_id is not None
        assert db.link_note_to_keyword(note_id, legacy_keyword_id)
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
                ("%research-receipt-proof:%",),
            ).fetchone()[0]
            == 1
        )
        db.close_connection()
    return note_id, conversation_id


def test_v42_to_v43_adds_private_proof_table_and_preserves_owner_rows(
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
            "SELECT COUNT(*) FROM keywords WHERE keyword LIKE ?",
            ("research-receipt-proof:%",),
        ).fetchone()[0]
        == 0
    )
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM sync_log WHERE payload LIKE ?",
            ("%research-receipt-proof:%",),
        ).fetchone()[0]
        == 0
    )
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
                "SELECT COUNT(*) FROM keywords WHERE keyword LIKE ?",
                ("research-receipt-proof:%",),
            ).fetchone()[0]
            == 1
        )


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
