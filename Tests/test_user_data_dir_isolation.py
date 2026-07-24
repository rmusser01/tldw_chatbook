"""Regression tests for task-519: get_user_data_dir()'s default-dir fallback
must resolve HOME/XDG_DATA_HOME at CALL time, not at module-import time.

Background: `tldw_chatbook.config.BASE_DATA_DIR_CLI` used to be a module-level
`Path.home()` constant, frozen the first time `config.py` was imported into
the process (i.e. before any per-test HOME/XDG_DATA_HOME monkeypatch could
possibly run). Any test that exercised the *default* data-dir fallback (no
`paths.data_dir` configured) therefore silently read/wrote the real
developer/CI-runner home directory -- this bit the RAG settings+profiles
program three separate times. See backlog task-519 for the full history.
"""

import pytest

from tldw_chatbook import config


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Ensure load_settings()/get_cli_setting() re-read from the (monkeypatched)
    active config path in every test in this module, instead of serving a
    stale cross-test cache."""
    config._SETTINGS_CACHE = None
    config._SETTINGS_CACHE_SOURCE = None
    config._CONFIG_CACHE = None
    config._CONFIG_CACHE_SOURCE = None
    yield
    config._SETTINGS_CACHE = None
    config._SETTINGS_CACHE_SOURCE = None
    config._CONFIG_CACHE = None
    config._CONFIG_CACHE_SOURCE = None


def _isolate(monkeypatch, tmp_path, *, home_name="scratch_home", xdg_data=None):
    """Point HOME (and optionally XDG_DATA_HOME) at a scratch dir, and give
    this test its own, guaranteed-empty CLI config file (so `paths.data_dir`
    is never configured and the default fallback is actually exercised)."""
    home_dir = tmp_path / home_name
    home_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home_dir))
    if xdg_data is not None:
        monkeypatch.setenv("XDG_DATA_HOME", str(xdg_data))
    else:
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    # Give this test a private, empty config file so get_cli_setting("paths",
    # "data_dir", None) genuinely falls through to the default-dir branch,
    # instead of picking up a data_dir configured by an earlier test/session.
    scratch_config = tmp_path / "config" / "config.toml"
    scratch_config.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(scratch_config))
    return home_dir


def test_get_user_data_dir_honors_home_set_after_import(monkeypatch, tmp_path):
    """The default data dir must live under the CURRENT (test-scoped) HOME,
    not whatever HOME happened to be true the first time config.py was
    imported into this process."""
    home_dir = _isolate(monkeypatch, tmp_path)

    user_dir = config.get_user_data_dir()

    assert str(user_dir).startswith(str(home_dir)), (
        f"get_user_data_dir() returned {user_dir!r}, which is not under the "
        f"test-scoped HOME {home_dir!r} -- the default data dir fallback is "
        f"still resolving against a stale, import-time-frozen home."
    )


def test_get_user_data_dir_honors_xdg_data_home_precedence(monkeypatch, tmp_path):
    """XDG_DATA_HOME, when set, must take precedence over HOME/.local/share."""
    xdg_dir = tmp_path / "xdg_data"
    xdg_dir.mkdir(parents=True, exist_ok=True)
    _isolate(monkeypatch, tmp_path, xdg_data=xdg_dir)

    user_dir = config.get_user_data_dir()

    assert str(user_dir).startswith(str(xdg_dir)), (
        f"get_user_data_dir() returned {user_dir!r}, expected a path under "
        f"XDG_DATA_HOME {xdg_dir!r}."
    )
    assert "tldw_cli" in user_dir.parts


def test_default_base_data_dir_helper_uses_home_when_no_xdg(monkeypatch, tmp_path):
    home_dir = tmp_path / "plain_home"
    home_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    resolved = config._default_base_data_dir()

    assert resolved == home_dir / ".local" / "share" / "tldw_cli"


def test_default_base_data_dir_helper_prefers_xdg(monkeypatch, tmp_path):
    home_dir = tmp_path / "plain_home"
    home_dir.mkdir(parents=True, exist_ok=True)
    xdg_dir = tmp_path / "xdg_data"
    xdg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_dir))

    resolved = config._default_base_data_dir()

    assert resolved == xdg_dir / "tldw_cli"
