"""Console configuration defaults."""

from tldw_chatbook import config as config_module


def test_console_large_paste_collapse_defaults_enabled():
    assert config_module.DEFAULT_CONFIG_FROM_TOML["console"]["collapse_large_pastes"] is True


def test_load_settings_exposes_console_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "missing-config.toml"))

    settings = config_module.load_settings(force_reload=True)

    assert settings["console"]["collapse_large_pastes"] is True


def test_load_settings_coerces_console_string_false(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[console]\ncollapse_large_pastes = "false"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    settings = config_module.load_settings(force_reload=True)

    assert settings["console"]["collapse_large_pastes"] is False
