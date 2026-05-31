"""Console configuration defaults."""

import tomllib

from tldw_chatbook import config as config_module


def test_console_large_paste_collapse_defaults_enabled():
    assert config_module.DEFAULT_CONFIG_FROM_TOML["console"]["collapse_large_pastes"] is True
    assert config_module.DEFAULT_CONFIG_FROM_TOML["console"]["paste_collapse_threshold"] == 50


def test_load_settings_exposes_console_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "missing-config.toml"))

    settings = config_module.load_settings(force_reload=True)

    assert settings["console"]["collapse_large_pastes"] is True
    assert settings["console"]["paste_collapse_threshold"] == 50


def test_load_settings_coerces_console_paste_threshold(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[console]\npaste_collapse_threshold = "120"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    settings = config_module.load_settings(force_reload=True)

    assert settings["console"]["paste_collapse_threshold"] == 120


def test_load_settings_rejects_boolean_console_paste_threshold(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    for raw_value in ("true", "false"):
        config_path.write_text(
            f"[console]\npaste_collapse_threshold = {raw_value}\n",
            encoding="utf-8",
        )

        settings = config_module.load_settings(force_reload=True)

        assert (
            settings["console"]["paste_collapse_threshold"]
            == config_module.DEFAULT_CONSOLE_PASTE_COLLAPSE_THRESHOLD
        )


def test_load_settings_coerces_console_string_false(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[console]\ncollapse_large_pastes = "false"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    settings = config_module.load_settings(force_reload=True)

    assert settings["console"]["collapse_large_pastes"] is False


def test_save_setting_respects_tldw_config_path_override(tmp_path, monkeypatch):
    override_config = tmp_path / "override" / "config.toml"
    default_config = tmp_path / "default" / "config.toml"
    override_config.parent.mkdir()
    override_config.write_text(
        '[console]\ncollapse_large_pastes = true\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(override_config))
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", default_config)
    config_module.load_cli_config_and_ensure_existence(force_reload=True)

    assert config_module.save_setting_to_cli_config(
        "console",
        "collapse_large_pastes",
        False,
    )

    saved_override = tomllib.loads(override_config.read_text(encoding="utf-8"))
    assert saved_override["console"]["collapse_large_pastes"] is False
    assert not default_config.exists()


def test_save_settings_batches_multiple_sections(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[console]\ncollapse_large_pastes = true\n[chat_defaults]\nstreaming = true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    assert config_module.save_settings_to_cli_config(
        {
            "console": {"collapse_large_pastes": False},
            "chat_defaults": {
                "streaming": False,
                "temperature": 0.33,
            },
        }
    )

    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert saved["console"]["collapse_large_pastes"] is False
    assert saved["chat_defaults"]["streaming"] is False
    assert saved["chat_defaults"]["temperature"] == 0.33


def test_chat_defaults_streaming_prefers_canonical_key(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[chat_defaults]\nstreaming = true\nenable_streaming = false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    assert config_module.get_chat_defaults_streaming(default=False) is True


def test_chat_defaults_streaming_uses_legacy_fallback(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[chat_defaults]\nenable_streaming = false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    assert config_module.get_chat_defaults_streaming(default=True) is False
