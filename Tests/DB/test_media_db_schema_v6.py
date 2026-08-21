"""Schema v6 (spec §8): chunk_engine_version column, NULL backfill."""
import sqlite3
import pytest
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase


@pytest.fixture()
def fresh_db(tmp_path):
    return MediaDatabase(str(tmp_path / "media.db"), client_id="test")


def test_schema_version_is_6(fresh_db):
    # The DB tracks its version in the ``schema_version`` table (PRAGMA
    # user_version stays 0); read that and assert 6 either way.
    version = fresh_db.get_connection().execute(
        "SELECT version FROM schema_version LIMIT 1"
    ).fetchone()
    assert version["version"] == 6
    assert fresh_db._CURRENT_SCHEMA_VERSION == 6


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
    # Genuine v5→v6 upgrade path (strengthens the approximation above):
    # build a v6 DB with a chunk row, rewind it to v5 shape by dropping the
    # column and resetting the version table, then re-open. The migration
    # must re-add the column, bump the version to 6, and leave the existing
    # row's chunk_engine_version NULL (stamp + report only — no backfill).
    path = str(tmp_path / "upgrade.db")
    db = MediaDatabase(path, client_id="test")
    db.add_media_with_keywords(
        title="t", media_type="document", content="text", keywords=None,
        url=None, analysis_content=None, author=None,
        chunks=[{"text": "old chunk", "metadata": {}}], chunk_options={},
    )
    db.close_connection()

    conn = sqlite3.connect(path)
    conn.execute(
        "ALTER TABLE UnvectorizedMediaChunks DROP COLUMN chunk_engine_version")
    conn.execute("UPDATE schema_version SET version = 5")
    conn.commit()
    conn.close()

    db2 = MediaDatabase(path, client_id="test")
    version = db2.get_connection().execute(
        "SELECT version FROM schema_version LIMIT 1").fetchone()["version"]
    assert version == 6
    cols = [r["name"] for r in db2.get_connection().execute(
        "PRAGMA table_info(UnvectorizedMediaChunks)").fetchall()]
    assert "chunk_engine_version" in cols
    rows = db2.get_connection().execute(
        "SELECT chunk_engine_version FROM UnvectorizedMediaChunks").fetchall()
    assert rows and rows[0]["chunk_engine_version"] is None
