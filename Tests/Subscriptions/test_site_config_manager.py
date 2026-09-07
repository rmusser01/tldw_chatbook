"""Tests for TASK-896: SiteConfigManager no longer lazily creates site_configs.

SiteConfigManager reads/writes site_configs through its own CharactersRAGDB
connection (unchanged), but the table itself is now declared in
SubscriptionsDB._initialize_schema (tldw_chatbook/DB/Subscriptions_DB.py).
These tests exercise that end-to-end through the public SiteConfigManager
API across fresh, already-migrated, and legacy databases, all against
tmp_path files -- never the real user config/data dirs.
"""

import sqlite3
from contextlib import closing

from tldw_chatbook.Subscriptions.site_config_manager import SiteConfig, SiteConfigManager


def test_fresh_db_manager_can_save_and_load_config(tmp_path):
    db_path = str(tmp_path / "subs.db")
    manager = SiteConfigManager(db_path)

    config = manager.get_config("https://example.com/page")
    assert config.domain == "example.com"

    config.rate_limit_requests = 30
    assert manager.save_config(config) is True

    reloaded = manager.get_config("https://example.com/other-page")
    assert reloaded.rate_limit_requests == 30


def test_idempotent_across_multiple_manager_constructions(tmp_path):
    # "Already migrated" case at the manager level: a second SiteConfigManager
    # (fresh Python object, fresh CharactersRAGDB/SubscriptionsDB instances)
    # opening the same file must see the first manager's data and must not
    # error on re-running _initialize_schema.
    db_path = str(tmp_path / "subs.db")

    first = SiteConfigManager(db_path)
    config = SiteConfig("example.com")
    config.rate_limit_requests = 42
    assert first.save_config(config) is True

    second = SiteConfigManager(db_path)
    domains = {row["domain"] for row in second.list_configs()}
    assert "example.com" in domains

    reloaded = second.get_config("https://example.com/")
    assert reloaded.rate_limit_requests == 42

    # A third construction, after the second has also written, stays
    # consistent -- covers "fresh, already-migrated, and legacy" all landing
    # on the same code path with no drift between opens.
    config2 = SiteConfig("other.example.com")
    assert second.save_config(config2) is True
    third = SiteConfigManager(db_path)
    domains = {row["domain"] for row in third.list_configs()}
    assert domains == {"example.com", "other.example.com"}


def test_manager_survives_legacy_db_with_existing_site_configs_rows(tmp_path):
    # A database created before this relocation: site_configs already exists
    # (created on demand by the old SiteConfigManager._create_tables path)
    # and already has a row in it. Existing site configs must survive.
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
            (
                "legacy.example.com",
                '{"rate_limit_requests": 15, "domain": "legacy.example.com"}',
            ),
        )
        legacy_conn.commit()

    manager = SiteConfigManager(str(path))

    domains = {row["domain"] for row in manager.list_configs()}
    assert "legacy.example.com" in domains

    existing = manager.get_config("https://legacy.example.com/")
    assert existing.rate_limit_requests == 15

    # New writes alongside the pre-existing legacy row keep working too.
    new_config = SiteConfig("new.example.com")
    assert manager.save_config(new_config) is True
    domains = {row["domain"] for row in manager.list_configs()}
    assert domains == {"legacy.example.com", "new.example.com"}


def test_apply_preset_and_delete_still_work(tmp_path):
    # Smoke-covers the public surface SiteConfigSettings/web_scraping_pipelines
    # exercise, so a regression in either would show up here too.
    manager = SiteConfigManager(str(tmp_path / "subs.db"))

    assert manager.apply_preset("github.com", "github.com") is True
    config = manager.get_config("https://github.com/")
    assert config.content_selector == ".markdown-body"

    assert manager.delete_config("github.com") is True
    domains = {row["domain"] for row in manager.list_configs()}
    assert "github.com" not in domains
