"""TASK-545 P3: the Settings tools section must control the real runtime.

Before this, the section was dead in four directions: its config read
returned {}, its executor had no callers, its save raised KeyError: 'None',
and it had no tests.
"""

import pytest

from tldw_chatbook.Agents.tool_catalog import (
    BuiltinToolProvider,
    gateable_builtin_tools,
)


def test_saving_a_gate_key_round_trips_to_the_provider(tmp_path, monkeypatch):
    """config -> save -> provider: the round trip the UI depends on.

    Drives the real save helper the UI uses, not a mock, because the bug
    being fixed was IN that call shape.
    """
    cfg = tmp_path / "config.toml"
    cfg.write_text('[general]\nusers_name = "t"\n', encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(cfg))

    import tldw_chatbook.config as config_module

    config_module.load_settings(force_reload=True)
    try:
        assert config_module.save_settings_to_cli_config(
            {"tools": {"write_file_enabled": True}}
        )
        config_module.load_settings(force_reload=True)
        names = {e.name for e in BuiltinToolProvider().list_catalog()}
        assert "write_file" in names
    finally:
        config_module._SETTINGS_CACHE = None
        config_module._SETTINGS_CACHE_SOURCE = None


def test_saving_leaves_unrendered_tools_keys_alone(tmp_path, monkeypatch):
    """A save must never silently disable a hand-edited flag."""
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[general]\nusers_name = "t"\n\n[tools]\ncreate_note_enabled = true\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(cfg))

    import tldw_chatbook.config as config_module

    config_module.load_settings(force_reload=True)
    try:
        config_module.save_settings_to_cli_config({"tools": {"read_file_enabled": True}})
        config_module.load_settings(force_reload=True)
        assert config_module.get_cli_setting("tools", "create_note_enabled") is True
    finally:
        config_module._SETTINGS_CACHE = None
        config_module._SETTINGS_CACHE_SOURCE = None


def test_the_broken_save_shape_is_gone():
    """`save_setting_to_cli_config(section, None, dict)` raises KeyError:
    'None'. Pin that the UI no longer uses it."""
    import pathlib

    src = pathlib.Path("tldw_chatbook/UI/Tools_Settings_Window.py").read_text(
        encoding="utf-8"
    )
    assert 'save_setting_to_cli_config("tools", None' not in src


def test_settings_window_no_longer_touches_system_a():
    import pathlib

    src = pathlib.Path("tldw_chatbook/UI/Tools_Settings_Window.py").read_text(
        encoding="utf-8"
    )
    assert "get_tool_executor" not in src
    assert "reload_tool_executor" not in src


def test_orphaned_executor_controls_are_gone():
    """Timeout/worker/cache controls configured only the deleted executor."""
    import pathlib

    src = pathlib.Path("tldw_chatbook/UI/Tools_Settings_Window.py").read_text(
        encoding="utf-8"
    )
    for widget_id in (
        "tool-timeout-input",
        "tool-max-workers-input",
        "tool-cache-enabled",
        "tool-cache-max-size-input",
        "tool-cache-ttl-input",
        "tool-cache-persist",
    ):
        assert widget_id not in src, f"{widget_id} still present"


def test_risk_tags_are_not_rendered_as_textual_markup():
    """A Label containing [reads] would be parsed as markup, not shown."""
    import pathlib

    src = pathlib.Path("tldw_chatbook/UI/Tools_Settings_Window.py").read_text(
        encoding="utf-8"
    )
    assert '[{tags}]' not in src and '[{", ".join' not in src


def test_every_gateable_tool_gets_a_switch_id():
    """The compose loop must cover the whole table, not a subset."""
    import pathlib

    src = pathlib.Path("tldw_chatbook/UI/Tools_Settings_Window.py").read_text(
        encoding="utf-8"
    )
    assert "gateable_builtin_tools()" in src
    assert 'f"tool-switch-{entry.tool_name}"' in src
