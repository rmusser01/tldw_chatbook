"""Schema v6 (spec §8): chunk_engine_version column, NULL backfill."""
import pytest
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

from Tests.DB.historical_bootstrap_v6 import media_db_at_version


@pytest.fixture()
def fresh_db(tmp_path):
    return MediaDatabase(str(tmp_path / "media.db"), client_id="test")


def test_schema_version_is_current(fresh_db):
    # AC 28 (chunking-template-parity spec): the pin is
    # ``== _CURRENT_SCHEMA_VERSION``, never a literal — later migrations
    # (v7+) must not break this file. What this test adds beyond
    # test_media_db_schema_v7.py is that a fresh DB reaches the code's
    # version at all.
    version = fresh_db.get_connection().execute(
        "SELECT version FROM schema_version LIMIT 1"
    ).fetchone()
    assert version["version"] == MediaDatabase._CURRENT_SCHEMA_VERSION


def test_column_exists(fresh_db):
    cols = [r["name"] for r in fresh_db.get_connection().execute(
        "PRAGMA table_info(UnvectorizedMediaChunks)").fetchall()]
    assert "chunk_engine_version" in cols


def test_v5_upgrade_leaves_rows_null(tmp_path):
    # Build a v5 database by hand: fresh DB, then drop to v5 semantics by
    # removing the column is impossible — instead create a DB with the OLD
    # code path: write one chunk row, NULL its version, and verify a re-open
    # keeps it readable and NULL (migration must not backfill).
    db = MediaDatabase(str(tmp_path / "m.db"), client_id="test")
    db.add_media_with_keywords(
        title="t", media_type="document", content="text", keywords=None,
        url=None, analysis_content=None, author=None,
        chunks=[{"text": "old chunk", "metadata": {}}], chunk_options={},
    )
    db.get_connection().execute(
        # ``version = version + 1`` is required by the
        # unvectorizedmediachunks_validate_sync_update trigger (every UPDATE
        # on this table must increment version by exactly 1); the brief's
        # bare UPDATE would ABORT on that trigger before ever reaching the
        # column under test.
        "UPDATE UnvectorizedMediaChunks SET chunk_engine_version = NULL, "
        "version = version + 1")
    db.get_connection().commit()
    # simulate upgrade: re-open the DB (runs migrations)
    db2 = MediaDatabase(str(tmp_path / "m.db"), client_id="test")
    rows = db2.get_connection().execute(
        "SELECT chunk_engine_version FROM UnvectorizedMediaChunks").fetchall()
    assert rows and rows[0]["chunk_engine_version"] is None


def test_genuine_v5_db_upgrades_to_v6_without_backfill(tmp_path):
    # Genuine v5→v6 upgrade path via the patched-_CURRENT_SCHEMA_VERSION
    # bootstrap (AC 19 pattern): the production chain builds a real v5 DB,
    # which by construction lacks chunk_engine_version. Re-opening replays
    # v5→v6 (and whatever follows): the column must appear and the existing
    # row's value must stay NULL (stamp + report only — no backfill), and
    # the version must equal the code's current version (AC 28: never a
    # literal). The hand-rewound fixture this replaced (drop column, stamp
    # version back) rotted the moment v7 rebuilt ChunkingTemplates — the
    # exact failure mode that banned the style.
    path = str(tmp_path / "upgrade.db")
    with media_db_at_version(path, 5) as db:
        conn = db.get_connection()
        conn.execute(
            "INSERT INTO Media (title, type, content_hash, uuid, "
            "last_modified, client_id) "
            "VALUES ('t', 'document', 'hash-v5', 'media-uuid-v5', "
            "'2020-01-01 00:00:00', 'test')"
        )
        conn.execute(
            "INSERT INTO UnvectorizedMediaChunks "
            "(media_id, chunk_text, chunk_index, uuid, last_modified, "
            "client_id) "
            "VALUES (1, 'old chunk', 0, 'chunk-uuid-v5', "
            "'2020-01-01 00:00:00', 'test')"
        )
        conn.commit()

    db2 = MediaDatabase(path, client_id="test")
    version = db2.get_connection().execute(
        "SELECT version FROM schema_version LIMIT 1").fetchone()["version"]
    assert version == MediaDatabase._CURRENT_SCHEMA_VERSION
    cols = [r["name"] for r in db2.get_connection().execute(
        "PRAGMA table_info(UnvectorizedMediaChunks)").fetchall()]
    assert "chunk_engine_version" in cols
    rows = db2.get_connection().execute(
        "SELECT chunk_engine_version FROM UnvectorizedMediaChunks").fetchall()
    assert rows and rows[0]["chunk_engine_version"] is None
