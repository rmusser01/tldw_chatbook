"""Regression tests for task-519: get_user_data_dir()'s default-dir fallback
must resolve HOME at CALL time, not at module-import time.

Background: `tldw_chatbook.config.BASE_DATA_DIR_CLI` used to be a module-level
`Path.home()` constant, frozen the first time `config.py` was imported into
the process (i.e. before any per-test HOME monkeypatch could possibly run).
Any test that exercised the *default* data-dir fallback (no `paths.data_dir`
configured) therefore silently read/wrote the real developer/CI-runner home
directory -- this bit the RAG settings+profiles program three separate
times. See backlog task-519 for the full history.

Note on XDG_DATA_HOME: an earlier version of this fix also made the default
fallback honor XDG_DATA_HOME (taking precedence over HOME). That was reverted
after review: the pre-existing default NEVER consulted XDG_DATA_HOME, so
honoring it would silently relocate a real XDG-configured user's data dir on
upgrade -- their entire existing data tree under ~/.local/share/tldw_cli
would appear to have vanished, with no migration and no warning. The default
is therefore deliberately HOME-only; the tests below assert XDG_DATA_HOME is
ignored.
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


def test_get_user_data_dir_ignores_xdg_data_home(monkeypatch, tmp_path):
    """XDG_DATA_HOME must NOT override the default data dir, even when set.

    The pre-task-519 default fallback (BASE_DATA_DIR_CLI) never consulted
    XDG_DATA_HOME -- it was always ~/.local/share/tldw_cli. A real user who
    has XDG_DATA_HOME exported (common on Linux desktops) already has their
    entire tldw_cli data tree under ~/.local/share/tldw_cli from every prior
    run. If the default fallback started honoring XDG_DATA_HOME, that user's
    very next launch would silently resolve to a brand-new, empty
    $XDG_DATA_HOME/tldw_cli directory -- with no migration and no warning,
    their conversations/notes/media would appear to have vanished. So this
    default is deliberately HOME-only; see task-519 review notes.
    """
    home_dir = _isolate(
        monkeypatch, tmp_path, xdg_data=tmp_path / "xdg_data_should_be_ignored"
    )

    user_dir = config.get_user_data_dir()

    assert str(user_dir).startswith(str(home_dir)), (
        f"get_user_data_dir() returned {user_dir!r}, expected it to stay "
        f"under HOME {home_dir!r} -- XDG_DATA_HOME must be ignored by the "
        f"default fallback to avoid orphaning an existing XDG user's data "
        f"on upgrade (task-519 review)."
    )
    assert "tldw_cli" in user_dir.parts


def test_default_base_data_dir_helper_uses_home_when_no_xdg(monkeypatch, tmp_path):
    home_dir = tmp_path / "plain_home"
    home_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    resolved = config._default_base_data_dir()

    assert resolved == home_dir / ".local" / "share" / "tldw_cli"


def test_default_base_data_dir_helper_ignores_xdg(monkeypatch, tmp_path):
    """XDG_DATA_HOME must be ignored even when it points somewhere real and
    HOME is also set -- honoring it here would silently relocate an existing
    XDG user's data dir on upgrade with no migration (task-519 review)."""
    home_dir = tmp_path / "plain_home"
    home_dir.mkdir(parents=True, exist_ok=True)
    xdg_dir = tmp_path / "xdg_data"
    xdg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_dir))

    resolved = config._default_base_data_dir()

    assert resolved == home_dir / ".local" / "share" / "tldw_cli"
