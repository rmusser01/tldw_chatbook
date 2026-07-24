import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
import toml

from tldw_chatbook import config as config_module
import tldw_chatbook.Utils.private_paths as private_paths
from tldw_chatbook.Utils.config_encryption import config_encryption
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


@pytest.mark.skipif(os.name != "posix", reason="POSIX creation contract")
def test_first_config_creation_is_serialized_across_workers(tmp_path, monkeypatch):
    target = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    _clear_config_cache()
    worker_count = 5
    ready = Barrier(worker_count)

    def load_config():
        ready.wait()
        return config_module.load_cli_config_and_ensure_existence()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        loaded = list(executor.map(lambda _index: load_config(), range(worker_count)))

    assert target.exists()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert all(config["general"]["users_name"] == "default_user" for config in loaded)


@pytest.mark.skipif(os.name != "posix", reason="POSIX replacement contract")
def test_whole_config_replacement_uses_effective_private_path(tmp_path, monkeypatch):
    target = tmp_path / "custom" / "config.toml"
    target.parent.mkdir(mode=0o700)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    _clear_config_cache()
    previous = os.umask(0)
    try:
        loaded = config_module.replace_cli_config(
            {"general": {"users_name": "private-user"}}
        )
    finally:
        os.umask(previous)

    assert config_module.get_cli_config_path() == target
    assert loaded["general"]["users_name"] == "private-user"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX export contract")
def test_config_snapshot_is_private_and_uses_effective_parent(tmp_path, monkeypatch):
    target = tmp_path / "custom" / "config.toml"
    target.parent.mkdir(mode=0o700)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    previous = os.umask(0)
    try:
        snapshot = config_module.export_cli_config_snapshot(
            {"API": {"openai_api_key": "secret"}},
            timestamp="20260723_120000",
        )
    finally:
        os.umask(previous)

    assert snapshot == target.parent / "config_backup_20260723_120000.toml"
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o600
    assert "secret" in snapshot.read_text(encoding="utf-8")


@pytest.mark.skipif(os.name != "posix", reason="POSIX encryption lifecycle contract")
def test_encryption_lifecycle_uses_effective_private_config_path(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "custom" / "config.toml"
    target.parent.mkdir(mode=0o700)
    target.write_text(
        '[API]\nopenai_api_key = "secret"\n',
        encoding="utf-8",
    )
    target.chmod(0o644)
    ignored_default = tmp_path / "ignored-default" / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", ignored_default)
    _clear_config_cache()
    previous = os.umask(0)
    try:
        assert config_module.enable_config_encryption("old-password") is True
        encrypted_value = toml.load(target)["API"]["openai_api_key"]
        assert config_encryption.is_encrypted(encrypted_value)
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        assert not ignored_default.exists()

        assert (
            config_module.change_encryption_password(
                "old-password",
                "new-password",
            )
            is True
        )
        rekeyed_value = toml.load(target)["API"]["openai_api_key"]
        assert config_encryption.is_encrypted(rekeyed_value)
        assert rekeyed_value != encrypted_value
        assert stat.S_IMODE(target.stat().st_mode) == 0o600

        assert config_module.disable_config_encryption("new-password") is True
        decrypted = toml.load(target)
        assert decrypted["API"]["openai_api_key"] == "secret"
        assert "encryption" not in decrypted
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
    finally:
        os.umask(previous)
        config_module.clear_encryption_password()
        _clear_config_cache()


@pytest.mark.skipif(os.name != "posix", reason="POSIX encrypted persistence contract")
def test_whole_replacement_and_snapshot_preserve_encryption_at_rest(
    tmp_path,
    monkeypatch,
):
    password = "replacement-password"
    target = tmp_path / "custom" / "config.toml"
    target.parent.mkdir(mode=0o700)
    encrypted = config_module.encrypt_api_keys_in_config(
        {
            "encryption": {"enabled": True},
            "API": {"openai_api_key": "old-secret"},
        },
        password,
    )
    target.write_text(toml.dumps(encrypted), encoding="utf-8")
    target.chmod(0o600)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    _clear_config_cache()
    config_module.set_encryption_password(password)
    try:
        loaded = config_module.load_cli_config_and_ensure_existence(force_reload=True)
        loaded["API"]["openai_api_key"] = "updated-secret"

        reloaded = config_module.replace_cli_config(loaded)
        snapshot = config_module.export_cli_config_snapshot(
            reloaded,
            timestamp="20260723_130000",
        )

        assert reloaded["API"]["openai_api_key"] == "updated-secret"
        on_disk = toml.load(target)["API"]["openai_api_key"]
        in_snapshot = toml.load(snapshot)["API"]["openai_api_key"]
        assert config_encryption.is_encrypted(on_disk)
        assert config_encryption.is_encrypted(in_snapshot)
        assert "updated-secret" not in target.read_text(encoding="utf-8")
        assert "updated-secret" not in snapshot.read_text(encoding="utf-8")
    finally:
        config_module.clear_encryption_password()
        _clear_config_cache()


@pytest.mark.skipif(os.name != "posix", reason="POSIX incremental write contract")
def test_incremental_config_writes_harden_existing_effective_file(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "custom" / "config.toml"
    target.parent.mkdir(mode=0o700)
    target.write_text(
        '[general]\nusers_name = "before"\n',
        encoding="utf-8",
    )
    target.chmod(0o644)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    _clear_config_cache()

    assert config_module.save_settings_to_cli_config(
        {"general": {"users_name": "after", "temporary": True}}
    )
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert config_module.delete_settings_from_cli_config(
        "general",
        ["temporary"],
    )
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_locked_encrypted_config_rejects_plaintext_incremental_secret(
    tmp_path,
    monkeypatch,
):
    password = "locked-config-password"
    target = tmp_path / "custom" / "config.toml"
    target.parent.mkdir(mode=0o700)
    encrypted = config_module.encrypt_api_keys_in_config(
        {
            "encryption": {"enabled": True},
            "api_settings": {"openai": {"api_key": "old-secret"}},
        },
        password,
    )
    target.write_text(toml.dumps(encrypted), encoding="utf-8")
    target.chmod(0o600)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    config_module.clear_encryption_password()
    _clear_config_cache()

    assert (
        config_module.save_setting_to_cli_config(
            "api_settings.openai",
            "api_key",
            "new-plaintext-secret",
        )
        is False
    )
    after_rejected_write = target.read_text(encoding="utf-8")
    assert "new-plaintext-secret" not in after_rejected_write
    assert config_encryption.is_encrypted(
        toml.loads(after_rejected_write)["api_settings"]["openai"]["api_key"]
    )

    assert config_module.save_setting_to_cli_config(
        "general",
        "default_tab",
        "chat",
    )
    after_safe_write = toml.load(target)
    assert after_safe_write["general"]["default_tab"] == "chat"
    assert config_encryption.is_encrypted(
        after_safe_write["api_settings"]["openai"]["api_key"]
    )


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


def test_decryptor_failure_returns_uncached_defaults_and_retries(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    target.write_text(
        "[encryption]\nenabled = true\n[chat_defaults]\ntemperature = 0.17\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    monkeypatch.setattr(config_module, "_ENCRYPTION_PASSWORD", "test-password")

    class EncryptionModule:
        failing = True

        def decrypt_config(self, config_data, password):
            assert password == "test-password"
            if self.failing:
                raise RuntimeError("decrypt failed")
            return config_data

        def decrypt_config_strict(self, config_data, password):
            return self.decrypt_config(config_data, password)

    encryption_module = EncryptionModule()
    monkeypatch.setattr(config_module, "_ENCRYPTION_MODULE", encryption_module)
    _clear_config_cache()

    ciphertext = {
        "encryption": {"enabled": True},
        "chat_defaults": {"temperature": 0.17},
    }
    assert config_module.decrypt_config_section(ciphertext) is ciphertext

    raw = config_module.load_cli_config_and_ensure_existence(force_reload=True)

    assert raw["chat_defaults"]["temperature"] == 0.6
    assert config_module._CONFIG_CACHE is None
    assert config_module._CONFIG_CACHE_SOURCE is None

    normalized = config_module.load_settings(force_reload=True)

    assert normalized["chat_defaults"]["temperature"] == 0.6
    assert config_module._CONFIG_CACHE is None
    assert config_module._CONFIG_CACHE_SOURCE is None
    assert config_module._SETTINGS_CACHE is None
    assert config_module._SETTINGS_CACHE_SOURCE is None

    encryption_module.failing = False

    repaired = config_module.load_settings()

    assert repaired["chat_defaults"]["temperature"] == 0.17
    assert config_module._CONFIG_CACHE_SOURCE == target.absolute()
    assert config_module._SETTINGS_CACHE_SOURCE == target.absolute()


def test_corrupt_encrypted_value_fails_bootstrap_without_poisoning_caches(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "config.toml"
    target.write_text(
        "[encryption]\nenabled = true\n"
        '[chat_defaults]\nsystem_prompt = "enc:not-valid-ciphertext"\n'
        "temperature = 0.17\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    monkeypatch.setattr(config_module, "_ENCRYPTION_PASSWORD", "test-password")
    monkeypatch.setattr(config_module, "_ENCRYPTION_MODULE", config_encryption)
    _clear_config_cache()

    failed = config_module._load_cli_config_bootstrap(force_reload=True)

    assert failed.succeeded is False
    assert failed.config["chat_defaults"]["temperature"] == 0.6
    assert config_module._CONFIG_CACHE is None
    assert config_module._CONFIG_CACHE_SOURCE is None

    normalized = config_module.load_settings(force_reload=True)

    assert normalized["chat_defaults"]["temperature"] == 0.6
    assert config_module._CONFIG_CACHE is None
    assert config_module._CONFIG_CACHE_SOURCE is None
    assert config_module._SETTINGS_CACHE is None
    assert config_module._SETTINGS_CACHE_SOURCE is None

    encrypted_prompt = config_encryption.encrypt_value(
        "decrypted system prompt",
        "test-password",
    )
    target.write_text(
        "[encryption]\nenabled = true\n"
        f'[chat_defaults]\nsystem_prompt = "{encrypted_prompt}"\n'
        "temperature = 0.17\n",
        encoding="utf-8",
    )

    repaired = config_module.load_settings()

    assert repaired["chat_defaults"]["system_prompt"] == "decrypted system prompt"
    assert repaired["chat_defaults"]["temperature"] == 0.17
    assert config_module._CONFIG_CACHE is not None
    assert (
        config_module._CONFIG_CACHE["chat_defaults"]["system_prompt"]
        == "decrypted system prompt"
    )
    assert config_module._SETTINGS_CACHE is not None
    assert (
        config_module._SETTINGS_CACHE["chat_defaults"]["system_prompt"]
        == "decrypted system prompt"
    )
    assert config_module._CONFIG_CACHE_SOURCE == target.absolute()
    assert config_module._SETTINGS_CACHE_SOURCE == target.absolute()


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
