"""Config saves must fail hard rather than persist plaintext secrets."""

import tomllib

import toml

from tldw_chatbook import config as config_module


class _BrokenEncryptionModule:
    def encrypt_value(self, value: str, password: str) -> str:
        raise RuntimeError("encryption backend exploded")


def _write_config(config_path, data: dict) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(toml.dumps(data), encoding="utf-8")


def test_encrypt_failure_blocks_save_and_leaves_file_unchanged(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        {"encryption": {"enabled": True}, "chat_defaults": {"streaming": True}},
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(config_module, "get_encryption_password", lambda: "pw")
    monkeypatch.setattr(
        config_module, "get_encryption_module", lambda: _BrokenEncryptionModule()
    )

    saved = config_module.save_settings_to_cli_config(
        {"tldw_api": {"base_url": "https://s.example.com", "auth_token": "secret-1"}}
    )

    assert saved is False
    on_disk = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert "tldw_api" not in on_disk
    assert on_disk["chat_defaults"] == {"streaming": True}


def test_locked_encryption_blocks_plaintext_secret_save(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        {"encryption": {"enabled": True}, "chat_defaults": {"streaming": True}},
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(config_module, "get_encryption_password", lambda: None)

    saved = config_module.save_settings_to_cli_config(
        {"tldw_api": {"base_url": "https://s.example.com", "auth_token": "secret-2"}}
    )

    assert saved is False
    on_disk = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert "secret-2" not in config_path.read_text(encoding="utf-8")
    assert on_disk["chat_defaults"] == {"streaming": True}
