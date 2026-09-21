"""RecentLocations must load config lazily, not in __init__ (TASK-32804.12).

Opening any enhanced file picker constructed RecentLocations, which read config
synchronously in __init__ (~5.5 ms on the click handler), while its sibling
BookmarksManager in the same file was made lazy by task-261. It now defers the
read to first use, matching the sibling.
"""

import tldw_chatbook.Widgets.enhanced_file_picker as efp


def test_recent_locations_does_not_read_config_on_construction(monkeypatch):
    calls = {"n": 0}

    def _fake(section, key=None, default=None):
        calls["n"] += 1
        return []

    monkeypatch.setattr(efp, "get_cli_setting", _fake)

    rl = efp.RecentLocations()
    assert calls["n"] == 0, "RecentLocations read config in __init__"

    rl.get_recent()
    assert calls["n"] == 1, "first use should load exactly once"

    rl.get_recent()
    assert calls["n"] == 1, "subsequent use must not re-read config"
