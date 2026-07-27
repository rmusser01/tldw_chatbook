from __future__ import annotations

import inspect
import os
import stat
from pathlib import Path

import pytest
import toml

from tldw_chatbook import config
from tldw_chatbook.Utils.config_encryption import config_encryption


def _reset_config_state() -> None:
    config._CONFIG_CACHE = None
    config._CONFIG_CACHE_SOURCE = None
    config._SETTINGS_CACHE = None
    config._SETTINGS_CACHE_SOURCE = None
    config.clear_encryption_password()


@pytest.fixture(autouse=True)
def reset_config_state():
    _reset_config_state()
    yield
    _reset_config_state()


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_serialized_raw_replace_backup_and_downgrade_guard_use_effective_path(
    tmp_path,
    monkeypatch,
):
    password = "owner-boundary-password"
    config_path = tmp_path / "override" / "config.toml"
    config_path.parent.mkdir(mode=0o700)
    ignored_default = tmp_path / "default" / "config.toml"
    encrypted = config.encrypt_api_keys_in_config(
        {
            "encryption": {"enabled": True},
            "api_settings": {"openai": {"api_key": "old-SENTINEL"}},
        },
        password,
    )
    config_path.write_text(toml.dumps(encrypted), encoding="utf-8")
    config_path.chmod(0o644)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(config, "DEFAULT_CONFIG_PATH", ignored_default)
    config.set_encryption_password(password)

    serialized = config.read_cli_config_serialized()
    loaded, backup = config.replace_cli_config_serialized(
        '[encryption]\nenabled = true\n'
        '[api_settings.openai]\napi_key = "new-SENTINEL"\n',
        create_backup=True,
    )

    assert "old-SENTINEL" not in serialized
    assert loaded["api_settings"]["openai"]["api_key"] == "new-SENTINEL"
    assert backup == config_path.with_suffix(".toml.bak")
    assert backup is not None
    assert "old-SENTINEL" not in backup.read_text(encoding="utf-8")
    assert config_encryption.is_encrypted(
        toml.load(backup)["api_settings"]["openai"]["api_key"]
    )
    assert config_encryption.is_encrypted(
        toml.load(config_path)["api_settings"]["openai"]["api_key"]
    )
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    assert not ignored_default.exists()

    before = config_path.read_bytes()
    with pytest.raises(ValueError, match="disable encryption explicitly"):
        config.replace_cli_config_serialized(
            '[api_settings.openai]\napi_key = "plaintext-downgrade-SENTINEL"\n'
        )
    assert config_path.read_bytes() == before


@pytest.mark.skipif(os.name != "posix", reason="POSIX shutdown contract")
def test_shutdown_persistence_uses_only_effective_path(tmp_path, monkeypatch):
    password = "shutdown-password"
    target = tmp_path / "override" / "config.toml"
    target.parent.mkdir(mode=0o700)
    ignored_default = tmp_path / "ignored" / "config.toml"
    encrypted = config.encrypt_api_keys_in_config(
        {
            "encryption": {"enabled": True},
            "API": {"openai_api_key": "shutdown-SENTINEL"},
        },
        password,
    )
    target.write_text(toml.dumps(encrypted), encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    monkeypatch.setattr(config, "DEFAULT_CONFIG_PATH", ignored_default)
    config.set_encryption_password(password)
    config.load_cli_config_and_ensure_existence(force_reload=True)

    assert config.persist_cli_config_for_shutdown() is True

    assert config_encryption.is_encrypted(toml.load(target)["API"]["openai_api_key"])
    assert "shutdown-SENTINEL" not in target.read_text(encoding="utf-8")
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not ignored_default.exists()


def test_production_config_persistence_has_one_owner_and_no_mutable_imports():
    package_root = Path(config.__file__).parent
    config_source = Path(config.__file__).read_text(encoding="utf-8")
    app_source = (package_root / "app.py").read_text(encoding="utf-8")
    settings_source = (
        package_root / "UI" / "Screens" / "settings_screen.py"
    ).read_text(encoding="utf-8")

    assert "atomic_private_write_text(" in config_source
    assert "atomic_write_text(\n        config_path" not in config_source
    assert "atomic_write_text(DEFAULT_CONFIG_PATH" not in app_source
    assert 'open(DEFAULT_CONFIG_PATH, "w"' not in app_source
    assert "DEFAULT_CONFIG_PATH.parent.mkdir" not in app_source
    assert "backup_path.write_text(" not in settings_source
    assert "tmp_path.replace(config_path)" not in settings_source
    assert "config_path.read_text(" not in settings_source

    offenders = []
    for source_path in package_root.rglob("*.py"):
        if source_path == Path(config.__file__):
            continue
        source = source_path.read_text(encoding="utf-8")
        if "config import settings" in source:
            offenders.append(source_path.relative_to(package_root).as_posix())
    assert offenders == []

    assert "persist_cli_config_for_shutdown" in app_source
    assert "load_cli_config_and_ensure_existence" in app_source
    assert inspect.isfunction(config.replace_cli_config_serialized)
