"""Raw bootstrap and precedence contracts for Console reasoning replay settings."""

from __future__ import annotations

import json

from loguru import logger

from tldw_chatbook import config as config_module
from tldw_chatbook.Chat.local_reasoning import REASONING_HISTORY_OPTIONS
from tldw_chatbook.Utils.reasoning_config import REASONING_HISTORY_MODES


def _write_console_config(path, body: str) -> None:
    path.write_text(f"[console]\n{body}", encoding="utf-8")


def test_raw_bootstrap_migrates_legacy_replay_opt_out(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    _write_console_config(config_path, "replay_thinking = false\n")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    bootstrap = config_module.load_cli_config_and_ensure_existence(force_reload=True)
    settings = config_module.load_settings(force_reload=True)

    assert bootstrap["console"]["reasoning_history"] == "off"
    assert settings["console"]["reasoning_history"] == "off"


def test_raw_bootstrap_preserves_explicit_mode_over_legacy_opt_out(
    tmp_path, monkeypatch
) -> None:
    config_path = tmp_path / "config.toml"
    _write_console_config(
        config_path,
        'replay_thinking = false\nreasoning_history = "current"\n',
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    bootstrap = config_module.load_cli_config_and_ensure_existence(force_reload=True)

    assert bootstrap["console"]["reasoning_history"] == "current"


def test_reasoning_environment_values_override_toml(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    _write_console_config(
        config_path,
        'reasoning_history = "current"\n'
        'reasoning_history_overrides = { saved = "off" }\n'
        "reasoning_native_tool_overrides = { saved = false }\n",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("TLDW_CONSOLE_REASONING_HISTORY", "all")
    monkeypatch.setenv(
        "TLDW_CONSOLE_REASONING_HISTORY_OVERRIDES",
        json.dumps({"environment": "current"}),
    )
    monkeypatch.setenv(
        "TLDW_CONSOLE_REASONING_NATIVE_TOOL_OVERRIDES",
        json.dumps({"environment": True}),
    )

    console = config_module.load_settings(force_reload=True)["console"]

    assert console["reasoning_history"] == "all"
    assert console["reasoning_history_overrides"] == {"environment": "current"}
    assert console["reasoning_native_tool_overrides"] == {"environment": True}


def test_blank_reasoning_environment_values_leave_toml_in_effect(
    tmp_path, monkeypatch
) -> None:
    config_path = tmp_path / "config.toml"
    _write_console_config(config_path, 'reasoning_history = "off"\n')
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("TLDW_CONSOLE_REASONING_HISTORY", "")

    console = config_module.load_settings(force_reload=True)["console"]

    assert console["reasoning_history"] == "off"


def test_malformed_override_reports_only_sanitized_field_name(
    tmp_path, monkeypatch
) -> None:
    config_path = tmp_path / "config.toml"
    private_canary = "user:secret@private.invalid"
    _write_console_config(
        config_path,
        f'reasoning_history = "current"\n'
        f'reasoning_history_overrides = "{private_canary}"\n',
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        console = config_module.load_settings(force_reload=True)["console"]
    finally:
        logger.remove(sink)

    diagnostic = "\n".join(messages)
    assert console["reasoning_history"] == "current"
    assert console["reasoning_history_overrides"] == {}
    assert "reasoning_history_overrides" in diagnostic
    assert private_canary not in diagnostic


def test_invalid_environment_map_is_diagnostic_and_does_not_use_toml_map(
    tmp_path, monkeypatch
) -> None:
    config_path = tmp_path / "config.toml"
    _write_console_config(
        config_path,
        'reasoning_history_overrides = { saved = "all" }\n',
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setenv(
        "TLDW_CONSOLE_REASONING_HISTORY_OVERRIDES",
        '{"private.invalid":',
    )
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        console = config_module.load_settings(force_reload=True)["console"]
    finally:
        logger.remove(sink)

    assert console["reasoning_history_overrides"] == {}
    assert "reasoning_history_overrides" in "\n".join(messages)


def test_nested_override_values_are_strictly_validated(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    _write_console_config(
        config_path,
        'reasoning_history_overrides = { target = "sideways" }\n'
        'reasoning_native_tool_overrides = { target = "true" }\n',
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        console = config_module.load_settings(force_reload=True)["console"]
    finally:
        logger.remove(sink)

    diagnostic = "\n".join(messages)
    assert console["reasoning_history_overrides"] == {}
    assert console["reasoning_native_tool_overrides"] == {}
    assert "reasoning_history_overrides" in diagnostic
    assert "reasoning_native_tool_overrides" in diagnostic


def test_generated_config_documents_reasoning_environment_names() -> None:
    for name in (
        "TLDW_CONSOLE_REASONING_HISTORY",
        "TLDW_CONSOLE_REASONING_HISTORY_OVERRIDES",
        "TLDW_CONSOLE_REASONING_NATIVE_TOOL_OVERRIDES",
    ):
        assert name in config_module.CONFIG_TOML_CONTENT


def test_config_model_modes_match_runtime_policy_options() -> None:
    assert REASONING_HISTORY_MODES == {
        value for _label, value in REASONING_HISTORY_OPTIONS
    }
