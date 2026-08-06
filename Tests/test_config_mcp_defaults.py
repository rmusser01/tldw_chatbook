"""MCP configuration defaults and coercion."""

from tldw_chatbook import config as config_module


def test_mcp_expose_local_tools_defaults_false(tmp_path, monkeypatch):
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "missing-config.toml"))

    settings = config_module.load_settings(force_reload=True)

    assert settings["mcp"]["expose_local_tools"] is False


def test_mcp_expose_local_tools_coerces_string_yes(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[mcp]\nexpose_local_tools = "yes"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    settings = config_module.load_settings(force_reload=True)

    assert settings["mcp"]["expose_local_tools"] is True
