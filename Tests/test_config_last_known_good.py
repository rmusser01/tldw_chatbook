"""TASK-26036: serve last-known-good config on a parse failure.

A mid-edit break in config.toml must not silently revert security-relevant
settings to built-in defaults. The loader serves the last successfully
loaded configuration instead, preserves the corrupt file aside, and records
a failure the app surfaces. First run (no prior good load) keeps the default
fallback, and a later good load clears everything.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook import config as config_module


def _clear():
    config_module._CONFIG_CACHE = None
    config_module._CONFIG_CACHE_SOURCE = None
    config_module._SETTINGS_CACHE = None
    config_module._SETTINGS_CACHE_SOURCE = None
    config_module._LAST_CONFIG_LOAD_FAILURE = None


_GOOD = '[general]\nusers_name = "Alice"\n'
_BAD = '[general]\nusers_name = "Alice'  # unterminated string


def test_parse_failure_after_a_good_load_serves_last_known_good(tmp_path, monkeypatch):
    target = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    _clear()

    target.write_text(_GOOD)
    good = config_module._load_cli_config_bootstrap_unlocked(force_reload=True)
    assert good.succeeded is True
    assert good.config["general"]["users_name"] == "Alice"

    # corrupt the file and force a reload
    target.write_text(_BAD)
    result = config_module._load_cli_config_bootstrap_unlocked(force_reload=True)

    # AC#1: the last good config is served, NOT built-in defaults
    assert result.config["general"]["users_name"] == "Alice"
    # AC#3: the failure is recorded, naming the file
    failure = config_module.get_config_load_failure()
    assert failure is not None
    assert failure.path == target
    # AC#2: the corrupt file is preserved aside, and the good file restored
    aside = list(tmp_path.glob("config.toml.corrupt*"))
    assert aside, "the unparseable file must be preserved under a distinct name"
    assert aside[0].read_text() == _BAD


def test_parse_failure_on_first_run_uses_defaults(tmp_path, monkeypatch):
    target = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    _clear()

    target.write_text(_BAD)
    result = config_module._load_cli_config_bootstrap_unlocked(force_reload=True)

    # AC#4: no prior good load -> the existing default fallback applies
    assert result.succeeded is False
    assert result.config.get("general", {}).get("users_name") != "Alice"
    assert config_module.get_config_load_failure() is not None


def test_recovery_after_the_file_is_fixed_clears_everything(tmp_path, monkeypatch):
    target = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    _clear()

    target.write_text(_GOOD)
    config_module._load_cli_config_bootstrap_unlocked(force_reload=True)
    target.write_text(_BAD)
    config_module._load_cli_config_bootstrap_unlocked(force_reload=True)
    assert config_module.get_config_load_failure() is not None

    # AC#5: a subsequent successful load replaces the retained copy and
    # clears the warning
    target.write_text('[general]\nusers_name = "Bob"\n')
    fixed = config_module._load_cli_config_bootstrap_unlocked(force_reload=True)
    assert fixed.succeeded is True
    assert fixed.config["general"]["users_name"] == "Bob"
    assert config_module.get_config_load_failure() is None
