"""The splash config loads in one section read, not one per key (TASK-32804.8).

_load_splash_config called get_cli_setting once per key (9 reads) before first
paint. It now reads the splash_screen section once and indexes it.
"""

import tldw_chatbook.Widgets.splash_screen as sp


def test_splash_config_reads_the_section_once(monkeypatch):
    calls = {"n": 0, "sections": []}

    def _fake(section, key=None, default=None):
        calls["n"] += 1
        calls["sections"].append(section)
        return {}  # empty section -> the loader falls back to its defaults

    monkeypatch.setattr(sp, "get_cli_setting", _fake)

    config = sp.SplashScreen._load_splash_config(object())

    assert calls["n"] == 1, f"read config {calls['n']} times: {calls['sections']}"
    assert calls["sections"] == ["splash_screen"]
    # Defaults are still merged in for every expected key.
    assert isinstance(config, dict)
    assert "fade_in_duration" in config
