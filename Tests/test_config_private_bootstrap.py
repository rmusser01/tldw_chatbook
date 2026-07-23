import os
import stat
from pathlib import Path

import pytest

from tldw_chatbook import config as config_module
import tldw_chatbook.Utils.private_paths as private_paths
from tldw_chatbook.Utils.private_paths import PrivatePathError


def _clear_config_cache():
    config_module._CONFIG_CACHE = None
    config_module._CONFIG_CACHE_SOURCE = None
    config_module._SETTINGS_CACHE = None
    config_module._SETTINGS_CACHE_SOURCE = None


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_first_config_creation_is_private(tmp_path, monkeypatch):
    target = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    _clear_config_cache()
    previous = os.umask(0o022)
    try:
        loaded = config_module.load_cli_config_and_ensure_existence(force_reload=True)
    finally:
        os.umask(previous)

    assert loaded["_first_run"] is True
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_default_application_config_directory_is_created_as_0700(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "application-config" / "config.toml"
    monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", target)
    _clear_config_cache()

    config_module.load_cli_config_and_ensure_existence(force_reload=True)

    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_existing_default_config_directory_is_hardened_before_read(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "application-config" / "config.toml"
    target.parent.mkdir()
    target.parent.chmod(0o755)
    target.write_text("[chat_defaults]\ntemperature = 0.17\n", encoding="utf-8")
    target.chmod(0o644)
    monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", target)
    _clear_config_cache()

    loaded = config_module.load_cli_config_and_ensure_existence(force_reload=True)

    assert loaded["chat_defaults"]["temperature"] == 0.17
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_existing_config_is_hardened_before_read(tmp_path, monkeypatch):
    target = tmp_path / "config.toml"
    target.write_text("[chat_defaults]\nstreaming = false\n", encoding="utf-8")
    target.chmod(0o644)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    _clear_config_cache()

    loaded = config_module.load_cli_config_and_ensure_existence(force_reload=True)

    assert loaded["chat_defaults"]["streaming"] is False
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX link contract")
def test_config_loader_rejects_final_symlink_without_reading_outside(
    tmp_path,
    monkeypatch,
):
    outside = tmp_path / "outside.toml"
    outside.write_text("[chat_defaults]\nstreaming = false\n", encoding="utf-8")
    outside.chmod(0o644)
    selected = tmp_path / "config.toml"
    selected.symlink_to(outside)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selected))
    _clear_config_cache()

    with pytest.raises(PrivatePathError):
        config_module.load_cli_config_and_ensure_existence(force_reload=True)

    assert stat.S_IMODE(outside.stat().st_mode) == 0o644


@pytest.mark.skipif(os.name != "posix", reason="POSIX namespace contract")
def test_config_loader_rejects_missing_file_in_shared_sticky_parent(
    tmp_path,
    monkeypatch,
):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o1777)
    selected = shared / "config.toml"
    fallback = tmp_path / ".tldw_cli_config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selected))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    _clear_config_cache()

    with pytest.raises(PrivatePathError):
        config_module.load_cli_config_and_ensure_existence(force_reload=True)

    assert not selected.exists()
    assert not fallback.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX namespace contract")
def test_config_loader_does_not_create_custom_config_parent(
    tmp_path,
    monkeypatch,
):
    selected = tmp_path / "custom" / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selected))
    _clear_config_cache()

    with pytest.raises(PrivatePathError):
        config_module.load_cli_config_and_ensure_existence(force_reload=True)

    assert not selected.parent.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX namespace contract")
def test_failed_private_creation_clears_existing_config_cache(
    tmp_path,
    monkeypatch,
):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o1777)
    selected = shared / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selected))
    config_module._CONFIG_CACHE = {"stale": True}
    config_module._CONFIG_CACHE_SOURCE = selected.absolute()

    with pytest.raises(PrivatePathError):
        config_module.load_cli_config_and_ensure_existence(force_reload=True)

    assert config_module._CONFIG_CACHE is None
    assert config_module._CONFIG_CACHE_SOURCE is None


def test_malformed_config_defaults_are_not_cached_and_repaired_file_is_reloaded(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    target.write_text("[chat_defaults\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    _clear_config_cache()

    loaded = config_module.load_cli_config_and_ensure_existence(force_reload=True)

    assert loaded["chat_defaults"]["temperature"] == 0.6
    assert config_module._CONFIG_CACHE is None
    assert config_module._CONFIG_CACHE_SOURCE is None

    target.write_text("[chat_defaults]\ntemperature = 0.17\n", encoding="utf-8")

    repaired = config_module.load_cli_config_and_ensure_existence()

    assert repaired["chat_defaults"]["temperature"] == 0.17


@pytest.mark.skipif(os.name != "posix", reason="POSIX link contract")
def test_failed_forced_settings_reload_clears_normalized_cache(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    target.write_text("[chat_defaults]\ntemperature = 0.17\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    _clear_config_cache()

    loaded = config_module.load_settings(force_reload=True)
    assert loaded["chat_defaults"]["temperature"] == 0.17

    outside = tmp_path / "outside.toml"
    outside.write_text("[chat_defaults]\ntemperature = 0.99\n", encoding="utf-8")
    target.unlink()
    target.symlink_to(outside)

    with pytest.raises(PrivatePathError):
        config_module.load_settings(force_reload=True)

    assert config_module._SETTINGS_CACHE is None
    assert config_module._SETTINGS_CACHE_SOURCE is None

    with pytest.raises(PrivatePathError):
        config_module.load_settings()


def test_malformed_config_defaults_are_not_cached_by_load_settings(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    target.write_text("[chat_defaults\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    _clear_config_cache()

    loaded = config_module.load_settings(force_reload=True)

    assert loaded["chat_defaults"]["temperature"] == 0.6
    assert config_module._SETTINGS_CACHE is None
    assert config_module._SETTINGS_CACHE_SOURCE is None

    target.write_text("[chat_defaults]\ntemperature = 0.17\n", encoding="utf-8")

    repaired = config_module.load_settings()

    assert repaired["chat_defaults"]["temperature"] == 0.17


def test_effective_path_preserves_symlink_spelling(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    selected = alias / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selected))

    assert config_module._get_effective_config_path() == selected


def test_config_loader_reports_unverified_platform_without_claiming_acl_safety(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    target.write_text("[chat_defaults]\nstreaming = true\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    monkeypatch.setattr(
        private_paths,
        "_posix_guards_available",
        lambda: False,
    )
    monkeypatch.setattr(private_paths, "_WINDOWS_PLATFORM", True)
    messages = []
    sink = config_module.logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )
    _clear_config_cache()
    try:
        config_module.load_cli_config_and_ensure_existence(force_reload=True)
    finally:
        config_module.logger.remove(sink)

    text = "\n".join(messages).lower()
    assert "permission posture is unverified" in text
    assert "owner-only" not in text
    assert "acl-secure" not in text
