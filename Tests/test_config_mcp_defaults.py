"""MCP configuration defaults and coercion."""

import pytest

from Tests.Backup_Recovery.config_test_support import select_config_source
from tldw_chatbook import config as config_module

SWITCHES = ("expose_local_tools", "expose_character_tools")


@pytest.mark.parametrize("switch", SWITCHES)
def test_mcp_exposure_switch_defaults_false(tmp_path, monkeypatch, switch):
    select_config_source(monkeypatch, str(tmp_path / "missing-config.toml"), globals())

    settings = config_module.load_settings(force_reload=True)

    assert settings["mcp"][switch] is False


@pytest.mark.parametrize("switch", SWITCHES)
def test_mcp_exposure_switch_coerces_string_yes(tmp_path, monkeypatch, switch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'[mcp]\n{switch} = "yes"\n',
        encoding="utf-8",
    )
    select_config_source(monkeypatch, str(config_path), globals())

    settings = config_module.load_settings(force_reload=True)

    assert settings["mcp"][switch] is True


def test_character_exposure_gate_treats_quoted_false_as_off(monkeypatch):
    from tldw_chatbook.MCP import local_server_tools

    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda s, k, d=None: (
            "false" if (s, k) == ("mcp", "expose_character_tools") else d
        ),
    )
    assert local_server_tools.character_tools_exposure_enabled() is False
    monkeypatch.setattr(config_module, "get_cli_setting", lambda s, k, d=None: "yes")
    assert local_server_tools.character_tools_exposure_enabled() is True
