"""TASK-26038: pick up external config edits without a manual reload.

Detection is a THROTTLED inline metadata (mtime+size) check on the read
path -- no filesystem watcher, no polling thread -- so an external edit is
picked up on the next read (after the throttle window) while the read hot
path keeps TASK-21124's near-zero cost.
"""

from __future__ import annotations

import pytest

from tldw_chatbook import config as config_module


def _clear():
    config_module._CONFIG_CACHE = None
    config_module._CONFIG_CACHE_SOURCE = None
    config_module._SETTINGS_CACHE = None
    config_module._SETTINGS_CACHE_SOURCE = None
    config_module._LAST_CONFIG_LOAD_FAILURE = None
    config_module._CONFIG_FILE_STAMP = None
    config_module._CONFIG_STAT_CHECKED_MONOTONIC = 0.0


def test_external_edit_is_picked_up_on_next_read(tmp_path, monkeypatch):
    target = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    monkeypatch.setattr(config_module, "_CONFIG_STAT_THROTTLE_SECONDS", 0.0)
    _clear()

    target.write_text('[general]\nusers_name = "Alice"\n')
    first = config_module.load_cli_config_and_ensure_existence()
    assert first["general"]["users_name"] == "Alice"

    # edit the file externally (bump size + content); no manual reload
    target.write_text('[general]\nusers_name = "Bob-the-longer-name"\n')
    second = config_module.load_cli_config_and_ensure_existence()
    assert second["general"]["users_name"] == "Bob-the-longer-name", (
        "an external edit must be picked up on the next read (AC#1)"
    )


def test_unchanged_file_is_not_re_read(tmp_path, monkeypatch):
    target = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    monkeypatch.setattr(config_module, "_CONFIG_STAT_THROTTLE_SECONDS", 0.0)
    _clear()
    target.write_text('[general]\nusers_name = "Alice"\n')

    first = config_module.load_cli_config_and_ensure_existence()
    # identity: an unchanged file returns the SAME cached object (no re-parse)
    second = config_module.load_cli_config_and_ensure_existence()
    assert second is first


def test_throttle_suppresses_the_stat_within_the_window(tmp_path, monkeypatch):
    """AC#6: within the throttle window no stat happens, so the hot path
    keeps its near-zero cost even if the file changed."""
    target = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    # a large window: the second read must NOT stat, so the edit is not seen
    monkeypatch.setattr(config_module, "_CONFIG_STAT_THROTTLE_SECONDS", 3600.0)
    _clear()
    target.write_text('[general]\nusers_name = "Alice"\n')
    first = config_module.load_cli_config_and_ensure_existence()
    assert first["general"]["users_name"] == "Alice"

    target.write_text('[general]\nusers_name = "Bob"\n')
    second = config_module.load_cli_config_and_ensure_existence()
    assert second["general"]["users_name"] == "Alice", (
        "within the throttle window the file is not re-statted"
    )


def test_manual_reload_still_works(tmp_path, monkeypatch):
    """AC#5."""
    target = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    monkeypatch.setattr(config_module, "_CONFIG_STAT_THROTTLE_SECONDS", 3600.0)
    _clear()
    target.write_text('[general]\nusers_name = "Alice"\n')
    config_module.load_cli_config_and_ensure_existence()

    target.write_text('[general]\nusers_name = "Bob"\n')
    forced = config_module.load_cli_config_and_ensure_existence(force_reload=True)
    assert forced["general"]["users_name"] == "Bob"
