"""Tests for TASK-896: site_configs relocated into SubscriptionsDB._initialize_schema.

Mirrors the local_watchlist_alert_rules precedent from TASK-690
(Tests/DB/test_subscriptions_db_watchlists.py): a fresh database gets the
table from ``_initialize_schema`` with no service call needed, reopening an
already-migrated database is a silent no-op, and a "legacy" database that
already has the table (created the old way, on a CharactersRAGDB connection
pointed at this same file) survives untouched, rows included.
"""

import sqlite3
from contextlib import closing

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB


@pytest.fixture
def db(tmp_path):
    return SubscriptionsDB(str(tmp_path / "subs.db"), client_id="test")


def _columns(db, table):
    cursor = db.conn.cursor()
    return {row[1] for row in cursor.execute(f"PRAGMA table_info({table})")}


def _tables(db):
    cursor = db.conn.cursor()
    return {
        row[0]
        for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def test_site_configs_table_owned_by_db(db):
    # Fresh database: no SiteConfigManager call needed -- SubscriptionsDB owns
    # this table now via _initialize_schema, same as local_watchlist_runs and
    # local_watchlist_alert_rules.
    assert "site_configs" in _tables(db)
    cols = _columns(db, "site_configs")
    assert {"id", "domain", "config_data", "created_at", "updated_at"} <= cols


def test_site_configs_index_created(db):
    cursor = db.conn.cursor()
    indices = {
        row[0]
        for row in cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='site_configs'"
        )
    }
    assert "idx_site_configs_domain" in indices


def test_site_configs_schema_creation_is_idempotent_across_reopen(tmp_path):
    # "Already migrated" case: opening the same file a second time must be a
    # silent no-op, not an error, since _initialize_schema always runs and
    # uses CREATE TABLE IF NOT EXISTS.
    path = tmp_path / "subs.db"
    first = SubscriptionsDB(str(path), client_id="test")
    assert "site_configs" in _tables(first)

    second = SubscriptionsDB(str(path), client_id="test")
    assert "site_configs" in _tables(second)


def test_site_configs_table_and_rows_survive_legacy_lazy_creation(tmp_path):
    # A database created before this relocation: it already has site_configs
    # (created on demand by the old SiteConfigManager._create_tables path, on
    # a CharactersRAGDB connection pointed at this same file), with an
    # existing row.
    path = tmp_path / "legacy.db"
    with closing(sqlite3.connect(path)) as legacy_conn:
        legacy_conn.executescript(
            """
            CREATE TABLE site_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT UNIQUE NOT NULL,
                config_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX idx_site_configs_domain ON site_configs(domain);
            """
        )
        legacy_conn.execute(
            "INSERT INTO site_configs (domain, config_data) VALUES (?, ?)",
            ("example.com", '{"rate_limit_requests": 60}'),
        )
        legacy_conn.commit()

    migrated = SubscriptionsDB(str(path), client_id="test")
    assert "site_configs" in _tables(migrated)
    rows = migrated.conn.execute(
        "SELECT domain, config_data FROM site_configs"
    ).fetchall()
    assert [(row[0], row[1]) for row in rows] == [
        ("example.com", '{"rate_limit_requests": 60}')
    ]


def test_lazy_site_config_table_creation_helper_is_gone():
    from tldw_chatbook.Subscriptions.site_config_manager import SiteConfigManager

    assert not hasattr(SiteConfigManager, "_create_tables")
