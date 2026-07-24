"""BookmarksManager constructor must be I/O-free (task-261).

`BookmarksManager.__init__` used to run five synchronous ``Path.exists()``
probes (a stall hazard when $HOME directories live on a cloud mount) plus, on
first run, a full TOML config write — on EVERY picker construction. These
tests prove construction now touches neither the config module nor the
filesystem, and that the first actual use performs exactly the old eager
behavior (defaults computed, first-run defaults written once, saved lists
honored).
"""

from pathlib import Path

import pytest

from tldw_chatbook.Widgets import enhanced_file_picker as efp
from tldw_chatbook.Widgets.enhanced_file_picker import BookmarksManager

REAL_HOME = Path.home()


@pytest.fixture
def config_recorder(monkeypatch):
    """Record config reads/writes; serve first-run (nothing saved) by default."""
    state = {"reads": [], "writes": [], "saved": None}

    def fake_get(section, key, default=None):
        state["reads"].append((section, key))
        return state["saved"] if state["saved"] is not None else default

    def fake_save(section, key, value):
        state["writes"].append((section, key, value))

    monkeypatch.setattr(efp, "get_cli_setting", fake_get)
    monkeypatch.setattr(efp, "save_setting_to_cli_config", fake_save)
    return state


@pytest.fixture
def home_call_counter(monkeypatch):
    """Count Path.home() calls — the gateway to the default-bookmark stats."""
    calls = []

    def counting_home(cls):
        calls.append(1)
        return REAL_HOME

    monkeypatch.setattr(Path, "home", classmethod(counting_home))
    return calls


def test_constructor_performs_no_config_or_filesystem_io(
    config_recorder, home_call_counter
):
    BookmarksManager(context="lazy_test")

    assert config_recorder["reads"] == [], "constructor must not read config"
    assert config_recorder["writes"] == [], "constructor must not write config"
    assert home_call_counter == [], (
        "constructor must not compute defaults (Path.home()/Path.exists())"
    )


def test_first_use_loads_defaults_and_writes_them_once(
    config_recorder, home_call_counter
):
    manager = BookmarksManager(context="lazy_test")

    bookmarks = manager.get_bookmarks()

    # Old eager behavior, now at first use: defaults computed and persisted.
    assert config_recorder["reads"] == [("filepicker", "bookmarks_lazy_test")]
    assert len(config_recorder["writes"]) == 1
    assert len(home_call_counter) >= 1
    assert any(entry["name"] == "Home" for entry in bookmarks)
    assert any(entry["path"] == str(REAL_HOME) for entry in bookmarks)

    # Second use: already loaded — no further reads or writes.
    manager.get_bookmarks()
    assert config_recorder["reads"] == [("filepicker", "bookmarks_lazy_test")]
    assert len(config_recorder["writes"]) == 1


def test_saved_bookmarks_are_honored_without_a_defaults_write(config_recorder):
    saved = [{"name": "Projects", "path": "/tmp/projects", "icon": "📁"}]
    config_recorder["saved"] = saved
    manager = BookmarksManager(context="lazy_test")

    assert manager.get_bookmarks() == saved
    assert config_recorder["writes"] == [], (
        "a saved list must not trigger the first-run defaults write"
    )


def test_add_and_is_bookmarked_trigger_lazy_load(config_recorder, tmp_path):
    manager = BookmarksManager(context="lazy_test")

    assert manager.add(tmp_path, name="Scratch") is True
    assert manager.is_bookmarked(tmp_path) is True
    assert manager.add(tmp_path) is False, "duplicate add must be rejected"

    # One load (which wrote first-run defaults) + one write for the add.
    assert config_recorder["reads"] == [("filepicker", "bookmarks_lazy_test")]
    assert len(config_recorder["writes"]) == 2
    saved_paths = [b["path"] for b in config_recorder["writes"][-1][2]]
    assert str(tmp_path.resolve()) in saved_paths


def test_save_on_unloaded_manager_is_a_noop(config_recorder):
    manager = BookmarksManager(context="lazy_test")

    manager.save_to_config()

    assert config_recorder["writes"] == [], (
        "saving an unloaded manager must not persist anything"
    )
