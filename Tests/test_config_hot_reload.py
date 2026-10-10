"""TASK-26038: pick up external config edits without a manual reload.

Detection is a THROTTLED inline metadata (mtime+size) check on the read
path -- no filesystem watcher, no polling thread -- so an external edit is
picked up on the next read (after the throttle window) while the read hot
path keeps TASK-21124's near-zero cost.
"""

from __future__ import annotations

import pytest

from Tests.Backup_Recovery.test_bound_config_companions import _SCRIPT
from Tests.Backup_Recovery.test_home_citation_retirement import _run
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


def test_self_write_is_not_a_phantom_external_edit_across_throttle(tmp_path, monkeypatch):
    """TASK-26038 regression (lane-7 review Important #1): after the app writes
    the config itself, a read occurring AFTER the stat-throttle window must NOT
    treat its own write as an external edit and force a locked re-read. The
    earlier test only read within the throttle window and missed this."""
    target = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    config_module._CONFIG_CACHE = None
    config_module._CONFIG_CACHE_SOURCE = None
    config_module._SETTINGS_CACHE = None
    config_module._SETTINGS_CACHE_SOURCE = None
    config_module._CONFIG_FILE_STAMP = None

    config_module.load_settings(force_reload=True)
    config_module.save_setting_to_cli_config("probe", "k", "v")

    # simulate the throttle window having elapsed since the write
    config_module._CONFIG_STAT_CHECKED_MONOTONIC = 0.0

    real_open = config_module.open_private_binary
    reads = {"n": 0}

    def counting_open(path, *a, **k):
        reads["n"] += 1
        return real_open(path, *a, **k)

    monkeypatch.setattr(config_module, "open_private_binary", counting_open)
    config_module.get_cli_setting("general", "users_name")

    assert reads["n"] == 0, (
        f"{reads['n']} file re-read(s) after a self-write across the throttle "
        "window; the app's own write must refresh the stamp so it is not seen "
        "as an external edit"
    )


_SHIPPED_THROTTLE = (
    _SCRIPT.split("assert config.get_cli_setting")[0]
    + r"""
import time
assert config._CONFIG_STAT_THROTTLE_SECONDS==1.0
assert config.get_cli_setting('general','users_name')=='fixture'
stamp=config._CONFIG_FILE_STAMP
selected.write_text(selected.read_text().replace('"fixture"','"edited-externally"'))
seen=[]
for _window in range(2):
 time.sleep(1.1)
 seen.append(config.get_cli_setting('general','users_name'))
report=f'seen={seen} stamp_updated={config._CONFIG_FILE_STAMP!=stamp}'
print(report)
assert seen[-1]=='edited-externally','external_edit_swallowed: '+report
print('retired and reopened')
"""
)


class ExternalEditSwallowed(AssertionError):
    """The TASK-34700 failure, as distinct from any other harness failure."""


@pytest.mark.xfail(
    raises=ExternalEditSwallowed,
    strict=True,
    reason="TASK-34700: the warm read's throttled stat consumes the window",
)
def test_external_edit_is_picked_up_at_the_shipped_throttle(tmp_path):
    """An external edit read through ``get_cli_setting`` at the real 1 s throttle.

    The tests above zero the throttle. At the shipped value the warm read's
    own throttled stat sees the edit and consumes the window, so the guarded
    bootstrap's check inside it finds nothing and serves the stale cache
    (observed: ``seen=['fixture', 'fixture'] stamp_updated=False``). Remove
    the xfail when TASK-34700 lands.
    """
    try:
        _run(tmp_path, "throttle", "hot-reload", script=_SHIPPED_THROTTLE, timeout=60)
    except AssertionError as failure:
        if "external_edit_swallowed" not in str(failure):
            raise RuntimeError("the isolated profile run failed otherwise") from failure
        raise ExternalEditSwallowed(str(failure)[-300:]) from None
