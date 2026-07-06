"""Tests for config_module.delete_settings_from_cli_config."""

import os

import toml
import tomllib

from tldw_chatbook import config as config_module


def _write_config(config_path, data: dict) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(toml.dumps(data), encoding="utf-8")


def test_deletes_existing_keys_from_nested_section(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        {
            "console": {
                "rail_state": {
                    "console_rail_state:ws:orphan-1": {
                        "left_open": True,
                        "right_open": False,
                    },
                    "console_rail_state:ws:live-conv": {
                        "left_open": False,
                        "right_open": True,
                    },
                }
            }
        },
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    assert config_module.delete_settings_from_cli_config(
        "console.rail_state",
        ["console_rail_state:ws:orphan-1"],
    )

    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    rail_state = saved["console"]["rail_state"]
    assert "console_rail_state:ws:orphan-1" not in rail_state
    assert rail_state["console_rail_state:ws:live-conv"] == {
        "left_open": False,
        "right_open": True,
    }


def test_missing_section_is_a_noop_returning_true(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    _write_config(config_path, {"chat_defaults": {"streaming": True}})
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    original_content = config_path.read_text(encoding="utf-8")
    original_mtime_ns = config_path.stat().st_mtime_ns

    assert config_module.delete_settings_from_cli_config(
        "console.rail_state",
        ["whatever-key"],
    )

    assert config_path.read_text(encoding="utf-8") == original_content
    assert config_path.stat().st_mtime_ns == original_mtime_ns


def test_missing_file_returns_true(tmp_path, monkeypatch):
    config_path = tmp_path / "does-not-exist" / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    assert config_module.delete_settings_from_cli_config(
        "console.rail_state",
        ["some-key"],
    )

    assert not config_path.exists()


def test_non_matching_delete_leaves_file_byte_identical(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        {
            "console": {
                "rail_state": {
                    "console_rail_state:ws:live-conv": {
                        "left_open": False,
                        "right_open": True,
                    },
                }
            }
        },
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    original_bytes = config_path.read_bytes()
    original_mtime_ns = config_path.stat().st_mtime_ns

    assert config_module.delete_settings_from_cli_config(
        "console.rail_state",
        ["console_rail_state:ws:key-that-does-not-exist"],
    )

    # No key was actually removed, so the file must not be rewritten at all.
    assert config_path.read_bytes() == original_bytes
    assert config_path.stat().st_mtime_ns == original_mtime_ns


def test_other_sections_and_keys_are_untouched(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        {
            "console": {
                "rail_state": {
                    "console_rail_state:ws:orphan-1": {"left_open": True},
                    "console_rail_state:ws:live-conv": {"left_open": False},
                },
                "collapse_large_pastes": True,
            },
            "chat_defaults": {"streaming": True, "temperature": 0.33},
        },
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    assert config_module.delete_settings_from_cli_config(
        "console.rail_state",
        ["console_rail_state:ws:orphan-1"],
    )

    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert "console_rail_state:ws:orphan-1" not in saved["console"]["rail_state"]
    assert saved["console"]["rail_state"]["console_rail_state:ws:live-conv"] == {
        "left_open": False,
    }
    assert saved["console"]["collapse_large_pastes"] is True
    assert saved["chat_defaults"] == {"streaming": True, "temperature": 0.33}
