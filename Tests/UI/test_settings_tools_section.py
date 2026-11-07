"""TASK-545 P3: a saved tool gate must reach the real runtime.

The legacy Settings tools section these round trips were written for is gone
with its window (TASK-33244), and the tests that read that window's source
went with it. Tool gates are switched in the MCP hub's Tool gates pane now
(Tests/Agents/test_builtin_tool_gate.py, Tests/UI/test_mcp_servers_mode.py);
what stays here is the config round trip every gate surface depends on.
"""

from tldw_chatbook.Agents.tool_catalog import (
    BuiltinToolProvider,
)


class _NoopWorkspaceExecutor:
    def execute(self, operation: str, arguments: dict, *, intent: str) -> str:
        raise AssertionError(f"unexpected workspace operation: {operation}")


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
        config_module.save_settings_to_cli_config(
            {"tools": {"read_file_enabled": True}}
        )
        config_module.load_settings(force_reload=True)
        assert config_module.get_cli_setting("tools", "create_note_enabled") is True
    finally:
        config_module._SETTINGS_CACHE = None
        config_module._SETTINGS_CACHE_SOURCE = None







def test_web_deep_search_gate_key_round_trips_to_config(tmp_path, monkeypatch):
    """task-3222: web_deep_search is a LocalToolSpec, not a _GATEABLE_
    BUILTINS entry, so it isn't covered by the round trip above -- but it
    shares the same [tools] table and must round-trip the same way.
    """
    cfg = tmp_path / "config.toml"
    cfg.write_text('[general]\nusers_name = "t"\n', encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(cfg))

    import tldw_chatbook.config as config_module
    from tldw_chatbook.Agents.local_tool_provider import _default_specs

    config_module.load_settings(force_reload=True)
    try:
        assert config_module.save_settings_to_cli_config(
            {"tools": {"web_deep_search_enabled": True}}
        )
        config_module.load_settings(force_reload=True)
        specs = _default_specs(tmp_path, workspace_executor=_NoopWorkspaceExecutor())
        assert "web_deep_search" in {s.name for s in specs}
    finally:
        config_module._SETTINGS_CACHE = None
        config_module._SETTINGS_CACHE_SOURCE = None
