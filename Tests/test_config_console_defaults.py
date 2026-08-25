"""Console configuration defaults."""

import os
from pathlib import Path
import tomllib

import pytest
from loguru import logger

CONFIG_PATH_BEFORE_CONFIG_IMPORT = os.environ.get("TLDW_CONFIG_PATH")
from tldw_chatbook import config as config_module  # noqa: E402


LOCAL_STREAMING_PROVIDER_SECTIONS = (
    "llama_cpp",
    "oobabooga",
    "koboldcpp",
    "ollama",
    "vllm",
    "aphrodite",
    "tabbyapi",
    "local-llm",
    "local_llamafile",
    "local_llamacpp",
    "local_vllm",
    "local_ollama",
    "local_onnx",
    "local_transformers",
    "local_mlx_lm",
)


def test_config_template_defaults_local_provider_streaming_on():
    """Local providers stream by default so slow generations do not appear hung.

    Regression guard for the Console UAT failure where the generated template's
    ``streaming = false`` forced llama.cpp onto the slow non-streamed path.
    """
    template = tomllib.loads(config_module.CONFIG_TOML_CONTENT)
    api_settings = template["api_settings"]

    for provider in LOCAL_STREAMING_PROVIDER_SECTIONS:
        assert api_settings[provider]["streaming"] is True, provider


def test_config_template_keeps_cloud_provider_streaming_opt_in():
    template = tomllib.loads(config_module.CONFIG_TOML_CONTENT)
    api_settings = template["api_settings"]

    for provider in (
        "openai",
        "anthropic",
        "google",
        "mistralai",
        "openrouter",
        "groq",
    ):
        assert api_settings[provider]["streaming"] is False, provider


def test_config_template_does_not_claim_provider_setup_confirmation():
    template = tomllib.loads(config_module.CONFIG_TOML_CONTENT)

    confirmed = template.get("provider_setup", {}).get("confirmed", {})
    assert confirmed == {}


def test_console_large_paste_collapse_defaults_enabled():
    assert (
        config_module.DEFAULT_CONFIG_FROM_TOML["console"]["collapse_large_pastes"]
        is True
    )
    assert (
        config_module.DEFAULT_CONFIG_FROM_TOML["console"]["paste_collapse_threshold"]
        == 50
    )


def test_console_assistant_library_access_default_is_false():
    """Fresh Console settings default assistant Library access to blocked."""
    template = tomllib.loads(config_module.CONFIG_TOML_CONTENT)

    assert template["console"]["assistant_library_access_default"] is False
    assert (
        config_module.DEFAULT_CONFIG_FROM_TOML["console"][
            "assistant_library_access_default"
        ]
        is False
    )


def test_legacy_console_rag_auto_retrieve_is_not_a_standing_template_default():
    """The obsolete setting is read only from an existing migration input."""
    template = tomllib.loads(config_module.CONFIG_TOML_CONTENT)

    assert "rag_auto_retrieve_on_send" not in template["chat_defaults"]
    assert (
        "rag_auto_retrieve_on_send"
        not in config_module.DEFAULT_CONFIG_FROM_TOML["chat_defaults"]
    )


@pytest.mark.parametrize("value", [True, False])
def test_legacy_console_rag_auto_retrieve_round_trips_as_a_strict_bool(
    tmp_path, monkeypatch, value
):
    """A valid saved boolean must be available unchanged to the migration seed."""
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    assert config_module.save_setting_to_cli_config(
        "chat_defaults", "rag_auto_retrieve_on_send", value
    )

    settings = config_module.load_settings(force_reload=True)
    seed = config_module.load_console_library_migration_seed(settings)

    assert settings["chat_defaults"]["rag_auto_retrieve_on_send"] is value
    assert seed.auto_retrieve_on_send is value


@pytest.mark.parametrize("raw_value", ['"sideways"', "42", '"true"'])
def test_malformed_legacy_console_rag_auto_retrieve_value_falls_back_safely(
    tmp_path, monkeypatch, raw_value
):
    """Invalid config cannot make new sessions automatic by accident."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"[chat_defaults]\\nrag_auto_retrieve_on_send = {raw_value}\\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    settings = config_module.load_settings(force_reload=True)
    seed = config_module.load_console_library_migration_seed(settings)

    assert seed.auto_retrieve_on_send is False


def test_migration_seed_rejects_string_boolean_from_an_already_loaded_config():
    """Config-looking strings cannot bypass the strict migration boundary."""
    seed = config_module.load_console_library_migration_seed(
        {"chat_defaults": {"rag_auto_retrieve_on_send": "true"}}
    )

    assert seed.auto_retrieve_on_send is False


def test_console_rail_labels_ship_horizontal_by_default():
    """The generated config keeps the established horizontal rail handles."""
    assert (
        config_module.DEFAULT_CONFIG_FROM_TOML["console"]["stack_collapsed_rail_labels"]
        is False
    )


def test_console_rail_layout_scope_ships_global_by_default():
    """One shared arrangement is the continuity-first generated default."""
    assert (
        config_module.DEFAULT_CONFIG_FROM_TOML["console"]["rail_layout_scope"]
        == "global"
    )


def test_console_background_effect_defaults_disabled():
    background = config_module.DEFAULT_CONFIG_FROM_TOML["console"]["background_effects"]

    assert background == {
        "enabled": False,
        "effect": "none",
        "scope": "transcript",
        "intensity": "low",
        "fps": 6,
    }


def test_load_settings_exposes_console_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "missing-config.toml"))

    settings = config_module.load_settings(force_reload=True)

    assert settings["console"]["collapse_large_pastes"] is True
    assert settings["console"]["paste_collapse_threshold"] == 50
    assert settings["console"]["stack_collapsed_rail_labels"] is False
    assert settings["console"]["rail_layout_scope"] == "global"
    assert settings["console"]["conversation_budget_mode"] == "automatic"
    assert settings["console"]["compaction_mode"] == "ask"


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("false", False),
        ("true", True),
        ('"false"', False),
        ('"true"', True),
        ('"sideways"', False),
    ],
)
def test_load_settings_normalizes_console_rail_label_style(
    tmp_path, monkeypatch, raw_value, expected
):
    """Only valid boolean-like values can opt into stacked rail labels."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"[console]\nstack_collapsed_rail_labels = {raw_value}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    settings = config_module.load_settings(force_reload=True)

    assert settings["console"]["stack_collapsed_rail_labels"] is expected


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ('"workspace"', "workspace"),
        ('"  WoRkSpAcE  "', "workspace"),
        ('"global"', "global"),
        ('"session"', "global"),
        ("123", "global"),
        ("true", "global"),
    ],
)
def test_load_settings_normalizes_console_rail_layout_scope(
    tmp_path, monkeypatch, raw_value, expected
):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"[console]\nrail_layout_scope = {raw_value}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    settings = config_module.load_settings(force_reload=True)

    assert settings["console"]["rail_layout_scope"] == expected


def test_console_sidechat_model_default_is_empty_string():
    from tldw_chatbook.config import get_cli_setting

    assert config_module.DEFAULT_CONFIG_FROM_TOML["console"]["sidechat_model"] == ""
    assert get_cli_setting("console", "sidechat_model", "") == ""


def test_console_sidechat_prompt_template_default():
    from tldw_chatbook.config import get_cli_setting

    assert (
        config_module.DEFAULT_CONFIG_FROM_TOML["console"]["sidechat_prompt_template"]
        == "Give me more details about: {selection}"
    )
    assert (
        get_cli_setting("console", "sidechat_prompt_template", "")
        == "Give me more details about: {selection}"
    )


def test_load_settings_exposes_console_sidechat_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "missing-config.toml"))

    settings = config_module.load_settings(force_reload=True)

    assert settings["console"]["sidechat_model"] == ""
    assert (
        settings["console"]["sidechat_prompt_template"]
        == "Give me more details about: {selection}"
    )


def test_console_sidechat_keys_survive_loader_coercion(tmp_path, monkeypatch):
    """String side-chat keys round-trip through the loader as strings."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[console]\n"
        'sidechat_model = "openai/gpt-5-mini"\n'
        'sidechat_prompt_template = "Summarize this simply: {selection}"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    settings = config_module.load_settings(force_reload=True)

    assert settings["console"]["sidechat_model"] == "openai/gpt-5-mini"
    assert (
        settings["console"]["sidechat_prompt_template"]
        == "Summarize this simply: {selection}"
    )


def test_console_sidechat_non_string_values_fall_back_to_defaults(
    tmp_path, monkeypatch
):
    """Presence-validation only: non-string values reset to the defaults."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[console]\nsidechat_model = 123\nsidechat_prompt_template = 789\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    settings = config_module.load_settings(force_reload=True)

    assert settings["console"]["sidechat_model"] == ""
    assert (
        settings["console"]["sidechat_prompt_template"]
        == "Give me more details about: {selection}"
    )


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


def test_load_settings_normalizes_console_background_effects(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[console.background_effects]",
                'enabled = "true"',
                'effect = "fire"',
                'scope = "everywhere"',
                'intensity = "extreme"',
                "fps = 99",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    settings = config_module.load_settings(force_reload=True)

    assert settings["console"]["background_effects"] == {
        "enabled": True,
        "effect": "none",
        "scope": "transcript",
        "intensity": "low",
        "fps": 12,
    }


def test_load_settings_coerces_console_string_false(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[console]\ncollapse_large_pastes = "false"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    settings = config_module.load_settings(force_reload=True)

    assert settings["console"]["collapse_large_pastes"] is False


def test_console_local_tools_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "missing-config.toml"))

    settings = config_module.load_settings(force_reload=True)
    console = settings["console"]

    assert console["local_tools_enabled"] is True
    assert console["workspace_root"] == ""


def test_config_template_enables_local_and_standard_web_tools():
    """Fresh profiles persist the same default the missing-key loader uses."""
    template = tomllib.loads(config_module.CONFIG_TOML_CONTENT)

    assert template["console"]["local_tools_enabled"] is True


def test_config_template_exposes_conversation_memory_defaults():
    template = tomllib.loads(config_module.CONFIG_TOML_CONTENT)
    console = template["console"]

    assert console["conversation_budget_mode"] == "automatic"
    assert "conversation_budget_tokens" not in console
    assert console["compaction_mode"] == "ask"
    assert console["compaction_trigger_ratio"] == 0.80
    assert console["compaction_target_ratio"] == 0.55
    assert console["compaction_summary_max_tokens"] == 1024
    assert console["compaction_failure_behavior"] == "stop_and_ask"
    assert console["compaction_carry_forward_mode"] == "memory_with_recent_turns"


def test_console_project_instruction_byte_limits_default_to_32_kib(
    tmp_path, monkeypatch
):
    template_console = config_module.DEFAULT_CONFIG_FROM_TOML["console"]
    assert template_console["project_instructions_startup_max_bytes"] == 32768
    assert template_console["project_instructions_nested_max_bytes"] == 32768

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "missing-config.toml"))

    settings = config_module.load_settings(force_reload=True)
    console = settings["console"]

    assert console["project_instructions_startup_max_bytes"] == 32768
    assert console["project_instructions_nested_max_bytes"] == 32768


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ('"4096"', 4096),
        ("1", 1),
        ("1048576", 1048576),
        ("0", 32768),
        ("1048577", 32768),
        ("true", 32768),
    ],
)
def test_console_project_instruction_byte_limits_are_bounded(
    tmp_path, monkeypatch, raw_value, expected
):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[console]",
                f"project_instructions_startup_max_bytes = {raw_value}",
                f"project_instructions_nested_max_bytes = {raw_value}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    console = config_module.load_settings(force_reload=True)["console"]

    assert console["project_instructions_startup_max_bytes"] == expected
    assert console["project_instructions_nested_max_bytes"] == expected


def test_console_local_tools_coerced(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[console]\nlocal_tools_enabled = "yes"\nworkspace_root = 123\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    settings = config_module.load_settings(force_reload=True)
    console = settings["console"]

    assert console["local_tools_enabled"] is True
    assert console["workspace_root"] == ""


def test_save_setting_respects_tldw_config_path_override(tmp_path, monkeypatch):
    override_config = tmp_path / "override" / "config.toml"
    default_config = tmp_path / "default" / "config.toml"
    override_config.parent.mkdir()
    override_config.write_text(
        "[console]\ncollapse_large_pastes = true\n",
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


def test_config_path_is_bootstrapped_before_config_import():
    assert CONFIG_PATH_BEFORE_CONFIG_IMPORT is not None
    bootstrap_config = Path(CONFIG_PATH_BEFORE_CONFIG_IMPORT)
    assert bootstrap_config != config_module.DEFAULT_CONFIG_PATH
    assert bootstrap_config.parent.is_dir()


def test_autouse_fixture_isolates_config_saves(tmp_path):
    isolated_config = tmp_path / "test_data" / "config" / "config.toml"
    default_config = config_module.DEFAULT_CONFIG_PATH
    default_contents = default_config.read_bytes() if default_config.exists() else None

    assert config_module._get_effective_config_path() == isolated_config.absolute()
    assert config_module.save_setting_to_cli_config(
        "console",
        "collapse_large_pastes",
        False,
    )

    saved = tomllib.loads(isolated_config.read_text(encoding="utf-8"))
    assert saved["console"]["collapse_large_pastes"] is False
    assert (
        default_config.read_bytes() if default_config.exists() else None
    ) == default_contents


def test_save_setting_redacts_sensitive_value_in_attempt_log(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    secret = "sk-review-secret-redaction-source"
    messages = []
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    sink_id = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="INFO",
    )
    try:
        assert config_module.save_setting_to_cli_config(
            "api_settings.openai",
            "api_key",
            secret,
        )
    finally:
        logger.remove(sink_id)

    joined_messages = "\n".join(messages)
    assert secret not in joined_messages
    assert "[api_settings.openai].api_key = '<redacted>'" in joined_messages


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


def test_chat_display_name_uses_chat_defaults_not_general_users_name(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[general]\nusers_name = 'storage-owner'\n"
        "[chat_defaults]\nuser_display_name = 'Rowan'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    assert config_module.get_chat_defaults_user_display_name() == "Rowan"


def test_blank_chat_display_name_falls_back_to_user(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[chat_defaults]\nuser_display_name = '   '\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    assert config_module.get_chat_defaults_user_display_name() == "User"


def test_invalid_chat_display_name_warns_without_echoing_value(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    invalid_value = "unsafe-secret\u202e"
    config_path.write_text(
        f'[chat_defaults]\nuser_display_name = "{invalid_value}"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    messages = []
    sink_id = logger.add(
        lambda message: messages.append(message.record["message"]),
        level="WARNING",
    )

    try:
        assert config_module.get_chat_defaults_user_display_name() == "User"
    finally:
        logger.remove(sink_id)

    joined_messages = "\n".join(messages)
    assert "chat display name" in joined_messages.lower()
    assert invalid_value not in joined_messages


def test_config_template_has_neutral_chat_display_name_default():
    template = tomllib.loads(config_module.CONFIG_TOML_CONTENT)

    assert template["chat_defaults"]["user_display_name"] == "User"
