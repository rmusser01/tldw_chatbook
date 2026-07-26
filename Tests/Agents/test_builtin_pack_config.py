import pytest


@pytest.fixture
def cli_setting(monkeypatch):
    """Drive get_cli_setting the way the app really reads it."""
    values = {}
    import tldw_chatbook.Agents.builtin_pack_config as mod

    def fake(section, key=None, default=None):
        return values.get((section, key), default)

    monkeypatch.setattr(mod, "get_cli_setting", fake)
    return values


def test_defaults_to_no_packs_enabled(cli_setting):
    """TASK-584 shipped these tools OFF. Restructuring must not turn them on."""
    from tldw_chatbook.Agents.builtin_pack_config import enabled_packs

    assert enabled_packs() == frozenset()


def test_reads_the_pack_list(cli_setting):
    from tldw_chatbook.Agents.builtin_pack_config import enabled_packs

    cli_setting[("agent_tools", "enabled_packs")] = ["files"]
    assert enabled_packs() == frozenset({"files"})


def test_legacy_tools_flags_enable_the_files_pack(cli_setting):
    """A user who already set read_file_enabled must not be switched off."""
    from tldw_chatbook.Agents.builtin_pack_config import enabled_packs

    cli_setting[("tools", "read_file_enabled")] = True
    assert enabled_packs() == frozenset({"files"})


def test_explicit_pack_list_wins_over_legacy_flags(cli_setting):
    from tldw_chatbook.Agents.builtin_pack_config import enabled_packs

    cli_setting[("agent_tools", "enabled_packs")] = []
    cli_setting[("tools", "read_file_enabled")] = True
    assert enabled_packs() == frozenset()


def test_non_list_pack_setting_is_ignored(cli_setting):
    """Hand-edited config must never crash a run."""
    from tldw_chatbook.Agents.builtin_pack_config import enabled_packs

    cli_setting[("agent_tools", "enabled_packs")] = "files"
    assert enabled_packs() == frozenset()


def test_enabled_packs_reads_a_real_config_file(tmp_path, monkeypatch):
    """No mocks: prove the key is reachable the way the app reads it.

    TASK-547 shipped a config section no reader could reach, and every
    mocked test of it would have passed. This test would have failed.
    """
    import tldw_chatbook.config as config_module
    from tldw_chatbook.Agents.builtin_pack_config import enabled_packs

    config_path = tmp_path / "config.toml"
    config_path.write_text('[agent_tools]\nenabled_packs = ["files"]\n')
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    # Force a real reload from the new path, and make sure this test's
    # config state cannot leak into any other test: the module-level cache
    # is reset back to whatever it held before we touched it, regardless of
    # pass/fail, once this test's TLDW_CONFIG_PATH override (reverted by
    # monkeypatch at teardown) is no longer in effect.
    original_cache = config_module._CONFIG_CACHE
    original_cache_source = config_module._CONFIG_CACHE_SOURCE
    try:
        config_module.load_cli_config_and_ensure_existence(force_reload=True)
        assert enabled_packs() == frozenset({"files"})
    finally:
        config_module._CONFIG_CACHE = original_cache
        config_module._CONFIG_CACHE_SOURCE = original_cache_source
