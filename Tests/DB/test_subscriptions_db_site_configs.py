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


def test_ensure_site_configs_schema_creates_only_that_table(tmp_path):
    """Qodo #4 on PR #989: guaranteeing one table must not build a database.

    `SiteConfigManager` takes a caller-supplied `db_path`. Opening a full
    `SubscriptionsDB` there to make sure `site_configs` exists would run the
    whole subscriptions schema against that file -- around fifteen unrelated
    tables plus indices and triggers -- a side effect no caller asked for.

    `ensure_site_configs_schema` shares its DDL with `_initialize_schema`, so
    there is still exactly one definition of the table; it just applies that
    one and stops.
    """
    from tldw_chatbook.DB.Subscriptions_DB import ensure_site_configs_schema

    path = tmp_path / "somebody_elses.db"
    ensure_site_configs_schema(path)

    with closing(sqlite3.connect(str(path))) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        }

    # `sqlite_sequence` is excluded above: SQLite creates it itself for any
    # AUTOINCREMENT column, so it is not a table this helper chose to make.
    assert "site_configs" in tables
    assert tables == {"site_configs"}, (
        f"only site_configs may be created on a caller-supplied path; "
        f"found {sorted(tables)}"
    )


def test_ensure_site_configs_schema_is_idempotent_and_keeps_rows(tmp_path):
    """Runs on every `SiteConfigManager` construction, so it must be a no-op
    the second time and must never disturb existing configs."""
    from tldw_chatbook.DB.Subscriptions_DB import ensure_site_configs_schema

    path = tmp_path / "configs.db"
    ensure_site_configs_schema(path)
    with closing(sqlite3.connect(str(path))) as conn:
        conn.execute(
            "INSERT INTO site_configs (domain, config_data) VALUES (?, ?)",
            ("example.com", '{"kept": true}'),
        )
        conn.commit()

    ensure_site_configs_schema(path)

    with closing(sqlite3.connect(str(path))) as conn:
        rows = conn.execute(
            "SELECT domain, config_data FROM site_configs"
        ).fetchall()
    assert rows == [("example.com", '{"kept": true}')]
