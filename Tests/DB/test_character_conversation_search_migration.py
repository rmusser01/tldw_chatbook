from pathlib import Path

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

OWNED_OBJECTS = {
    "character_conversation_search_documents",
    "character_conversation_fts",
    "character_conversation_search_generations",
    "character_conversation_search_dirty",
    "character_conversation_search_revision",
    "character_conversation_search_state",
}


def test_v66_dispatches_versioned_sql_artifact(monkeypatch, tmp_path: Path) -> None:
    artifact = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "DB"
        / "migrations"
        / "chachanotes_v65_to_v66_character_conversation_search.sql"
    )
    assert artifact.is_file()
    reads = []
    read_text = Path.read_text

    def tracked_read(path, *args, **kwargs):
        if path == artifact:
            reads.append(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracked_read)
    db = CharactersRAGDB(tmp_path / "artifact.sqlite", client_id="artifact")
    assert reads == [artifact]
    assert db._get_db_version(db.get_connection()) == 68


def _owned_schema(db: CharactersRAGDB) -> dict[str, tuple[tuple[object, ...], ...]]:
    connection = db.get_connection()
    result: dict[str, tuple[tuple[object, ...], ...]] = {}
    for name in sorted(OWNED_OBJECTS):
        master = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master WHERE name = ?",
            (name,),
        ).fetchone()
        assert master is not None
        columns = connection.execute(f'PRAGMA table_info("{name}")').fetchall()
        result[name] = (tuple(master), *(tuple(column) for column in columns))
    result["indexes"] = tuple(
        tuple(row)
        for row in connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE (type = 'index' OR type = 'trigger') "
            "AND name LIKE 'character_conversation_%' ORDER BY type, name"
        ).fetchall()
    )
    return result


def test_v66_migrates_genuine_v65_once_without_starting_backfill(
    tmp_path: Path,
) -> None:
    path = tmp_path / "character-search-v65.sqlite"
    with chachanotes_db_at_version(path, 65) as historical:
        authority = historical.get_local_authority_id()
        character_id = historical.add_character_card({"name": "Legacy"})
        assert character_id is not None
        for conversation_id in ("unique", "ambiguous"):
            assert (
                historical.add_conversation(
                    {
                        "id": conversation_id,
                        "character_id": character_id,
                        "assistant_kind": "character",
                        "assistant_id": str(character_id),
                        "assistant_authority_id": authority,
                        "title": conversation_id,
                    }
                )
                == conversation_id
            )
        with historical.transaction() as connection:
            connection.execute(
                "UPDATE conversations SET assistant_authority_id = NULL "
                "WHERE id IN ('unique', 'ambiguous')"
            )
            connection.execute(
                "UPDATE conversations SET assistant_id = 'not-provable' "
                "WHERE id = 'ambiguous'"
            )

    upgraded = CharactersRAGDB(path, client_id="character-search-v66")
    try:
        connection = upgraded.get_connection()
        assert upgraded._get_db_version(connection) == 68
        assert {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE name IN (?, ?, ?, ?, ?, ?)",
                tuple(sorted(OWNED_OBJECTS)),
            )
        } == OWNED_OBJECTS
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM character_conversation_search_documents"
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM character_conversation_search_generations"
            ).fetchone()[0]
            == 0
        )
        links = {
            row["id"]: row["assistant_authority_id"]
            for row in connection.execute(
                "SELECT id, assistant_authority_id FROM conversations "
                "WHERE id IN ('unique', 'ambiguous')"
            )
        }
        assert links == {"unique": authority, "ambiguous": None}
    finally:
        upgraded.close_connection()

    reopened = CharactersRAGDB(path, client_id="character-search-v66-reopen")
    try:
        assert reopened._get_db_version(reopened.get_connection()) == 68
        assert (
            reopened.get_connection()
            .execute(
                "SELECT data_revision FROM character_conversation_search_revision "
                "WHERE singleton_id = 1"
            )
            .fetchone()[0]
            == 1
        )
    finally:
        reopened.close_connection()


def test_v66_fresh_schema_matches_migrated_tables_columns_indexes_and_triggers(
    tmp_path: Path,
) -> None:
    historical_path = tmp_path / "migrated.sqlite"
    with chachanotes_db_at_version(historical_path, 65):
        pass
    migrated = CharactersRAGDB(historical_path, client_id="migrated")
    fresh = CharactersRAGDB(tmp_path / "fresh.sqlite", client_id="fresh")
    try:
        assert CharactersRAGDB._CURRENT_SCHEMA_VERSION == 68
        assert _owned_schema(fresh) == _owned_schema(migrated)
        document_columns = {
            row[1]
            for row in fresh.get_connection().execute(
                "PRAGMA table_info(character_conversation_search_documents)"
            )
        }
        assert document_columns == {
            "document_id",
            "data_authority_id",
            "conversation_id",
            "character_id",
            "character_label",
            "title",
            "body",
            "eligibility_digest",
            "validated_eligibility_digest",
            "source_revision",
            "generation_id",
        }
        generation_columns = {
            row[1]
            for row in fresh.get_connection().execute(
                "PRAGMA table_info(character_conversation_search_generations)"
            )
        }
        assert {"lease_expires_at", "updated_at"} <= generation_columns
    finally:
        fresh.close_connection()
        migrated.close_connection()


def test_canvas_migrations_preserve_character_search_state_from_genuine_v66(
    tmp_path: Path,
) -> None:
    """The integrated 66→67→68 chain must not reset canonical search state."""

    path = tmp_path / "character-search-state-v66.sqlite"
    with chachanotes_db_at_version(path, 66) as historical:
        authority = historical.get_local_authority_id()
        character_id = historical.add_character_card({"name": "Preserved"})
        assert character_id is not None
        conversation_id = historical.add_conversation(
            {
                "id": "preserved-search-conversation",
                "character_id": character_id,
                "assistant_kind": "character",
                "assistant_id": str(character_id),
                "assistant_authority_id": authority,
                "title": "Preserved search state",
            }
        )
        assert conversation_id == "preserved-search-conversation"
        with historical.transaction() as connection:
            connection.execute(
                "UPDATE character_conversation_search_revision "
                "SET data_revision = 7, updated_at = '2026-09-05T10:00:00+00:00' "
                "WHERE singleton_id = 1"
            )
            connection.execute(
                "UPDATE character_conversation_search_state "
                "SET data_authority_id = ?, active_policy_version = 3, "
                "activated = 1, updated_at = '2026-09-05T10:01:00+00:00' "
                "WHERE singleton_id = 1",
                (authority,),
            )
            connection.execute(
                "INSERT INTO character_conversation_search_generations "
                "(generation_id, data_authority_id, status, policy_version, "
                "source_revision, processed_conversations, created_at, "
                "completed_at, updated_at) "
                "VALUES ('preserved-generation', ?, 'ready', 3, 7, 1, "
                "'2026-09-05T10:02:00+00:00', "
                "'2026-09-05T10:03:00+00:00', "
                "'2026-09-05T10:03:00+00:00')",
                (authority,),
            )
            connection.execute(
                "INSERT INTO character_conversation_search_dirty "
                "(conversation_id, data_authority_id, source_revision, enqueued_at) "
                "VALUES (?, ?, 7, '2026-09-05T10:04:00+00:00')",
                (conversation_id, authority),
            )
        expected = {
            "revision": tuple(
                historical.get_connection()
                .execute(
                    "SELECT singleton_id, data_revision, updated_at "
                    "FROM character_conversation_search_revision"
                )
                .fetchone()
            ),
            "state": tuple(
                historical.get_connection()
                .execute(
                    "SELECT singleton_id, data_authority_id, active_policy_version, "
                    "activated, updated_at FROM character_conversation_search_state"
                )
                .fetchone()
            ),
            "generation": tuple(
                historical.get_connection()
                .execute(
                    "SELECT generation_id, data_authority_id, status, policy_version, "
                    "source_revision, processed_conversations, created_at, "
                    "completed_at, lease_expires_at, updated_at "
                    "FROM character_conversation_search_generations"
                )
                .fetchone()
            ),
            "dirty": tuple(
                historical.get_connection()
                .execute(
                    "SELECT conversation_id, data_authority_id, source_revision, "
                    "enqueued_at FROM character_conversation_search_dirty"
                )
                .fetchone()
            ),
        }

    upgraded = CharactersRAGDB(path, client_id="character-search-state-v68")
    try:
        connection = upgraded.get_connection()
        assert upgraded._get_db_version(connection) == 68
        assert (
            tuple(
                connection.execute(
                    "SELECT singleton_id, data_revision, updated_at "
                    "FROM character_conversation_search_revision"
                ).fetchone()
            )
            == expected["revision"]
        )
        assert (
            tuple(
                connection.execute(
                    "SELECT singleton_id, data_authority_id, active_policy_version, "
                    "activated, updated_at FROM character_conversation_search_state"
                ).fetchone()
            )
            == expected["state"]
        )
        assert (
            tuple(
                connection.execute(
                    "SELECT generation_id, data_authority_id, status, policy_version, "
                    "source_revision, processed_conversations, created_at, completed_at, "
                    "lease_expires_at, updated_at "
                    "FROM character_conversation_search_generations"
                ).fetchone()
            )
            == expected["generation"]
        )
        assert (
            tuple(
                connection.execute(
                    "SELECT conversation_id, data_authority_id, source_revision, "
                    "enqueued_at FROM character_conversation_search_dirty"
                ).fetchone()
            )
            == expected["dirty"]
        )
    finally:
        upgraded.close_connection()


@pytest.mark.parametrize(
    ("index", "query", "params"),
    [
        (
            "character_conversation_search_dirty_authority_revision",
            (
                "SELECT conversation_id FROM character_conversation_search_dirty "
                "WHERE data_authority_id = ? AND source_revision <= ?"
            ),
            ("authority", 1),
        ),
        (
            "character_conversation_search_documents_character",
            (
                "SELECT conversation_id FROM character_conversation_search_documents "
                "WHERE data_authority_id = ? AND character_id = ? AND generation_id = ?"
            ),
            ("authority", 1, "generation"),
        ),
        (
            "character_conversation_search_documents_revision",
            (
                "SELECT conversation_id FROM character_conversation_search_documents "
                "WHERE data_authority_id = ? AND source_revision = ?"
            ),
            ("authority", 1),
        ),
        (
            "character_conversation_search_generations_authority_status",
            (
                "SELECT generation_id FROM character_conversation_search_generations "
                "WHERE data_authority_id = ? AND status = ?"
            ),
            ("authority", "failed"),
        ),
        (
            "character_conversation_search_one_ready_generation",
            (
                "SELECT data_authority_id FROM character_conversation_search_generations "
                "WHERE status = 'ready'"
            ),
            (),
        ),
    ],
)
def test_keyword_indexes_are_selected_without_statistics(
    tmp_path: Path, index: str, query: str, params: tuple
) -> None:
    database = CharactersRAGDB(tmp_path / "keyword-plans.sqlite", client_id="plans")
    try:
        connection = database.get_connection()
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name = 'sqlite_stat1'"
            ).fetchone()
            is None
        )
        plan = " ".join(
            row[3] for row in connection.execute("EXPLAIN QUERY PLAN " + query, params)
        )
        assert index in plan
        assert "TEMP B-TREE" not in plan
    finally:
        database.close_connection()
