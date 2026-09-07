"""Config persistence for the Lab frame's rail collapse state."""

from __future__ import annotations

import pytest

from tldw_chatbook.UI.Lab_Modules import lab_rail_store
from tldw_chatbook.UI.Lab_Modules.lab_rail_layout import (
    LAB_RAIL_INSPECTOR,
    LAB_RAIL_LEFT,
    LabRailLayout,
)


@pytest.fixture
def fake_config(monkeypatch):
    """Capture reads and writes without touching the user's config file."""
    store = {}

    def fake_get(section, key=None, default=None):
        assert section == lab_rail_store.LAB_CONFIG_SECTION
        assert key == lab_rail_store.LAB_COLLAPSED_RAILS_KEY
        return store.get("value", default)

    def fake_save(section, key, value):
        assert section == lab_rail_store.LAB_CONFIG_SECTION
        assert key == lab_rail_store.LAB_COLLAPSED_RAILS_KEY
        store["value"] = value
        return True

    monkeypatch.setattr(lab_rail_store, "get_cli_setting", fake_get)
    monkeypatch.setattr(lab_rail_store, "save_setting_to_cli_config", fake_save)
    return store


def test_unset_config_yields_the_first_run_layout(fake_config):
    """Never-set must give the first-run default: inspector collapsed."""
    layout = lab_rail_store.load_rail_layout()
    assert layout.is_collapsed(LAB_RAIL_INSPECTOR) is True
    assert layout.is_collapsed(LAB_RAIL_LEFT) is False


def test_explicitly_empty_is_not_the_first_run_default(fake_config):
    """A user who expanded everything must not get the default re-imposed.

    This is why the sentinel passed to get_cli_setting is None, not [].
    """
    fake_config["value"] = []
    layout = lab_rail_store.load_rail_layout()
    assert layout.is_collapsed(LAB_RAIL_INSPECTOR) is False
    assert layout.is_collapsed(LAB_RAIL_LEFT) is False


def test_round_trip(fake_config):
    saved = LabRailLayout(collapsed=frozenset({LAB_RAIL_LEFT}))
    lab_rail_store.save_rail_layout(saved)
    loaded = lab_rail_store.load_rail_layout()
    assert loaded.is_collapsed(LAB_RAIL_LEFT) is True
    assert loaded.is_collapsed(LAB_RAIL_INSPECTOR) is False


def test_saved_value_is_a_plain_sorted_list(fake_config):
    """TOML cannot hold a frozenset; sorted keeps the file diff stable."""
    lab_rail_store.save_rail_layout(
        LabRailLayout(collapsed=frozenset({LAB_RAIL_INSPECTOR, LAB_RAIL_LEFT}))
    )
    assert fake_config["value"] == sorted([LAB_RAIL_INSPECTOR, LAB_RAIL_LEFT])


def test_unknown_names_in_config_are_ignored(fake_config):
    """A hand-edited or stale config must not crash the screen."""
    fake_config["value"] = ["inspector", "sidebar", 17, None]
    layout = lab_rail_store.load_rail_layout()
    assert layout.is_collapsed(LAB_RAIL_INSPECTOR) is True
    assert layout.is_collapsed(LAB_RAIL_LEFT) is False


def test_a_non_list_config_value_falls_back_to_first_run(fake_config):
    fake_config["value"] = "inspector"
    layout = lab_rail_store.load_rail_layout()
    assert layout == lab_rail_store.LAB_FIRST_RUN_LAYOUT
