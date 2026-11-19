"""Config defaults for the quit-time session summary (issue #365)."""

import tomllib

import pytest

from tldw_chatbook import config as config_module
from tldw_chatbook.config import CONFIG_TOML_CONTENT

# Importing tldw_chatbook.app (for the TldwCli method binding) pulls the
# guarded config bootstrap, which under the per-test sandbox fails closed
# with RecoveryRequired("raw_source_selection_changed") -- the signature
# Tests/conftest.py's keep-list documents. Keep the bootstrap profile.
pytestmark = pytest.mark.bootstrap_profile


def test_session_summary_defaults_exist():
    parsed = tomllib.loads(CONFIG_TOML_CONTENT)
    section = parsed["session_summary"]
    assert section["enabled"] is False
    assert section["duration_seconds"] == 3


def test_duration_loader_clamps_and_defaults(monkeypatch):
    from tldw_chatbook import app as app_module
    from tldw_chatbook import app_lifecycle

    class _Harness:
        _session_summary_duration_seconds = (
            app_module.TldwCli._session_summary_duration_seconds
        )

    harness = _Harness()

    def _make_setting(duration):
        def setting(section, key, default=None):
            if (section, key) == ("session_summary", "duration_seconds"):
                return duration
            return default

        return setting

    for duration, expected in [(0, 1), (1, 1), (3, 3), (30, 30), (99, 30), (-5, 1), (2.9, 3)]:
        monkeypatch.setattr(app_lifecycle, "get_cli_setting", _make_setting(duration))
        assert harness._session_summary_duration_seconds() == expected

    for bad in ["abc", None, "", "inf", "nan"]:
        monkeypatch.setattr(app_lifecycle, "get_cli_setting", _make_setting(bad))
        assert harness._session_summary_duration_seconds() == 3
