"""[console_ssh] defaults and typed accessors (Phase 2a, Task 10).

The section is the config surface of the SSH ControlMaster lifecycle
(``Tools/remote_workspace_transport.py``): ``control_persist`` rides the
``-MNf`` master command verbatim, ``enable_multiplexing`` is the Windows /
opt-out kill switch, ``connect_timeout_s`` bounds both master spawn and
per-call connects, and ``max_concurrent_calls`` caps in-flight per-binding
calls (consumed by the executor task, landed here so the section ships
complete).

Accessor behaviour is exercised against a stubbed ``get_cli_setting``
rather than a force-reloaded scratch config: the guarded loader's
backup-recovery participants reject a mid-process config-source swap on
machines carrying participant state, so the swap-based idiom is not a
reliable isolation here (the sibling model-catalog tests fail the same
way on such machines, independent of this change).
"""

import tomllib

import pytest

from tldw_chatbook import config as config_module
from tldw_chatbook.config import (
    CONFIG_TOML_CONTENT,
    ConsoleSshSettings,
    get_console_ssh_settings,
)


@pytest.fixture()
def raw_section(monkeypatch: pytest.MonkeyPatch):
    """Route ``get_cli_setting("console_ssh", ...)`` at a plain dict."""

    def install(values: dict) -> None:
        def fake_get_cli_setting(section, key=None, default=None):
            if section == "console_ssh":
                return values.get(key, default)
            return default

        monkeypatch.setattr(config_module, "get_cli_setting", fake_get_cli_setting)

    return install


def test_console_ssh_defaults_exist_in_the_default_toml():
    parsed = tomllib.loads(CONFIG_TOML_CONTENT)
    section = parsed["console_ssh"]
    assert section == {
        "control_persist": "10m",
        "enable_multiplexing": True,
        "connect_timeout_s": 3,
        "max_concurrent_calls": 8,
        "session_worker": True,
        "session_idle_s": 60,
        "bundle_cache": True,
    }


def test_console_ssh_settings_defaults_when_section_absent(raw_section):
    raw_section({})

    settings = get_console_ssh_settings()

    assert settings == ConsoleSshSettings(
        control_persist="10m",
        enable_multiplexing=True,
        connect_timeout_s=3,
        max_concurrent_calls=8,
    )


def test_console_ssh_settings_honour_overrides(raw_section):
    raw_section(
        {
            "control_persist": "45s",
            "enable_multiplexing": False,
            "connect_timeout_s": 7,
            "max_concurrent_calls": 2,
        }
    )

    settings = get_console_ssh_settings()

    assert settings == ConsoleSshSettings(
        control_persist="45s",
        enable_multiplexing=False,
        connect_timeout_s=7,
        max_concurrent_calls=2,
    )


def test_console_ssh_settings_reject_unusable_values(raw_section):
    raw_section(
        {
            "control_persist": 123,  # not a string: cannot ride argv verbatim
            "enable_multiplexing": "maybe",
            "connect_timeout_s": "soon",
            "max_concurrent_calls": 0,  # below the floor of 1
        }
    )

    settings = get_console_ssh_settings()

    # Falls back to the shipped defaults — never to a silently disabled or
    # zero-sized posture.
    assert settings == ConsoleSshSettings()


@pytest.mark.parametrize("value", ["10 minutes", "10m;", " 10m", "-5", "1h 30m", "forever"])
def test_console_ssh_control_persist_rejects_malformed_durations(raw_section, value):
    raw_section({"control_persist": value})
    assert get_console_ssh_settings().control_persist == ConsoleSshSettings().control_persist


@pytest.mark.parametrize("value", ["yes", "no", "600", "10m", "1h30m", "2d"])
def test_console_ssh_control_persist_accepts_ssh_time_formats(raw_section, value):
    raw_section({"control_persist": value})
    assert get_console_ssh_settings().control_persist == value


def test_console_ssh_session_worker_defaults(raw_section):
    raw_section({})
    settings = get_console_ssh_settings()
    assert (settings.session_worker, settings.session_idle_s, settings.bundle_cache) == (
        True,
        60,
        True,
    )


def test_console_ssh_session_worker_overrides(raw_section):
    raw_section({"session_worker": False, "session_idle_s": 5, "bundle_cache": False})
    settings = get_console_ssh_settings()
    assert (settings.session_worker, settings.session_idle_s, settings.bundle_cache) == (
        False,
        5,
        False,
    )


def test_console_ssh_session_worker_rejects_unusable_values(raw_section):
    raw_section({"session_worker": "no", "session_idle_s": 0, "bundle_cache": 1})
    settings = get_console_ssh_settings()
    assert (settings.session_worker, settings.session_idle_s, settings.bundle_cache) == (
        True,
        60,
        True,
    )
