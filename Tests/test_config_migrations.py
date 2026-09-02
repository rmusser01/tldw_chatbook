"""TASK-26040: versioned config with stepwise forward migrations."""

from __future__ import annotations

import pytest

from tldw_chatbook import config as config_module
from tldw_chatbook.config import (
    CONFIG_SCHEMA_VERSION_KEY,
    migrate_config_forward,
)


def test_unversioned_config_is_treated_as_baseline_and_stamped():
    """AC#6 + AC#1."""
    old = {"general": {"users_name": "Alice"}}
    migrated, changed, conflict = migrate_config_forward(old)
    assert conflict is None
    assert changed is True
    assert migrated[CONFIG_SCHEMA_VERSION_KEY] == config_module._CURRENT_CONFIG_SCHEMA_VERSION
    assert migrated["general"]["users_name"] == "Alice", "values preserved"


def test_current_version_config_is_unchanged():
    current = {CONFIG_SCHEMA_VERSION_KEY: config_module._CURRENT_CONFIG_SCHEMA_VERSION, "x": 1}
    migrated, changed, conflict = migrate_config_forward(current)
    assert changed is False
    assert conflict is None
    assert migrated == current


def test_newer_version_is_detected_not_mangled():
    """AC#5."""
    future = {CONFIG_SCHEMA_VERSION_KEY: config_module._CURRENT_CONFIG_SCHEMA_VERSION + 5, "x": 1}
    migrated, changed, conflict = migrate_config_forward(future)
    assert conflict is not None
    assert "newer" in conflict.lower()
    assert changed is False
    assert migrated == future, "a newer config must not be transformed/mangled"


def test_stepwise_migrations_preserve_values(monkeypatch):
    """AC#2 + AC#7: a realistic old config reaches the current shape with
    values preserved, through each numbered step in order."""
    calls = []

    def _v1_to_v2(cfg):
        calls.append(2)
        cfg = dict(cfg)
        # a realistic rename: [old_section] moved to [new_section]
        if "old_section" in cfg:
            cfg["new_section"] = cfg.pop("old_section")
        return cfg

    def _v2_to_v3(cfg):
        calls.append(3)
        cfg = dict(cfg)
        cfg.setdefault("new_section", {})["added_default"] = True
        return cfg

    monkeypatch.setattr(config_module, "_CURRENT_CONFIG_SCHEMA_VERSION", 3)
    monkeypatch.setattr(
        config_module, "_CONFIG_MIGRATIONS", {2: _v1_to_v2, 3: _v2_to_v3}
    )

    old = {
        CONFIG_SCHEMA_VERSION_KEY: 1,
        "old_section": {"kept_value": 42},
        "general": {"users_name": "Bob"},
    }
    migrated, changed, conflict = migrate_config_forward(old)

    assert conflict is None and changed is True
    assert calls == [2, 3], "migrations run in stepwise numeric order"
    assert migrated[CONFIG_SCHEMA_VERSION_KEY] == 3
    assert "old_section" not in migrated
    assert migrated["new_section"]["kept_value"] == 42, "values preserved across rename"
    assert migrated["new_section"]["added_default"] is True
    assert migrated["general"]["users_name"] == "Bob"


# --- load-path + persist integration (AC#1/#2/#3/#4/#5) ---

def _clear():
    config_module._CONFIG_CACHE = None
    config_module._CONFIG_CACHE_SOURCE = None
    config_module._SETTINGS_CACHE = None
    config_module._SETTINGS_CACHE_SOURCE = None
    config_module._CONFIG_SCHEMA_CONFLICT = None


def test_load_stamps_and_migrates_unversioned_file_in_memory(tmp_path, monkeypatch):
    """AC#2/#6: an unversioned file on disk is read as the baseline and the
    running config carries the current version, without rewriting the file."""
    target = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    _clear()
    target.write_text('[general]\nusers_name = "Alice"\n')
    original = target.read_text()

    result = config_module._load_cli_config_bootstrap_unlocked(force_reload=True)

    assert result.succeeded is True
    assert result.config[CONFIG_SCHEMA_VERSION_KEY] == config_module._CURRENT_CONFIG_SCHEMA_VERSION
    assert result.config["general"]["users_name"] == "Alice"
    assert config_module.get_config_schema_conflict() is None
    # a bare stamp must NOT rewrite the file (would strip user comments)
    assert target.read_text() == original


def test_load_reports_newer_version_and_does_not_mangle(tmp_path, monkeypatch):
    """AC#5: a config from a newer version is served untouched with a warning."""
    target = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    _clear()
    future = config_module._CURRENT_CONFIG_SCHEMA_VERSION + 3
    target.write_text(
        f"{CONFIG_SCHEMA_VERSION_KEY} = {future}\n"
        '[general]\nusers_name = "FromFuture"\n'
        "[brand_new_section]\nkept = true\n"
    )

    result = config_module._load_cli_config_bootstrap_unlocked(force_reload=True)

    conflict = config_module.get_config_schema_conflict()
    assert conflict is not None and "newer" in conflict.lower()
    # newer keys are preserved, not dropped
    assert result.config[CONFIG_SCHEMA_VERSION_KEY] == future
    assert result.config["brand_new_section"]["kept"] is True


def test_persist_migration_backs_up_and_rewrites(tmp_path, monkeypatch):
    """AC#3/#4/#7: a real forward migration backs up the original and
    atomically rewrites the migrated result with values preserved."""
    target = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    _clear()
    target.write_text(
        f"{CONFIG_SCHEMA_VERSION_KEY} = 1\n"
        "[old_section]\nkept_value = 42\n"
    )

    def _v1_to_v2(cfg):
        cfg = dict(cfg)
        cfg["new_section"] = cfg.pop("old_section")
        return cfg

    monkeypatch.setattr(config_module, "_CURRENT_CONFIG_SCHEMA_VERSION", 2)
    monkeypatch.setattr(config_module, "_CONFIG_MIGRATIONS", {2: _v1_to_v2})

    backup = config_module.migrate_config_file_if_needed()

    assert backup is not None and backup.exists()
    assert "old_section" in backup.read_text(), "backup keeps the pre-migration file"
    import tomllib
    rewritten = tomllib.loads(target.read_text())
    assert rewritten[CONFIG_SCHEMA_VERSION_KEY] == 2
    assert "old_section" not in rewritten
    assert rewritten["new_section"]["kept_value"] == 42


def test_failed_migration_leaves_original_untouched(tmp_path, monkeypatch):
    """AC#3: a migration that raises must not corrupt or replace the file."""
    target = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    _clear()
    target.write_text(
        f"{CONFIG_SCHEMA_VERSION_KEY} = 1\n"
        "[general]\nusers_name = \"Safe\"\n"
    )
    original = target.read_text()

    def _boom(cfg):
        raise RuntimeError("migration blew up")

    monkeypatch.setattr(config_module, "_CURRENT_CONFIG_SCHEMA_VERSION", 2)
    monkeypatch.setattr(config_module, "_CONFIG_MIGRATIONS", {2: _boom})

    with pytest.raises(RuntimeError):
        config_module.migrate_config_file_if_needed()

    assert target.read_text() == original, "the original file is untouched on failure"


def test_bare_stamp_does_not_persist(tmp_path, monkeypatch):
    """No migration function in range => no file rewrite (comments preserved)."""
    target = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    _clear()
    target.write_text("[general]\nusers_name = \"Alice\"  # a comment\n")
    original = target.read_text()

    assert config_module.migrate_config_file_if_needed() is None
    assert target.read_text() == original
