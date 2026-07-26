from tldw_chatbook.UI.Watchlists_Modules import region_layout_store
from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region, RegionLayout


def test_round_trips_collapsed_regions(monkeypatch):
    saved = {}

    def fake_save(section, key, value):
        saved[(section, key)] = value
        return True

    monkeypatch.setattr(region_layout_store, "save_setting_to_cli_config", fake_save)
    region_layout_store.save_region_layout(
        RegionLayout(collapsed=frozenset({Region.CONTENT, Region.RIGHT_RAIL}))
    )

    # Flat section, not "watchlists.layout" — a dotted section silently no-ops.
    assert ("watchlists", "collapsed_regions") in saved
    assert sorted(saved[("watchlists", "collapsed_regions")]) == ["content", "right_rail"]

    monkeypatch.setattr(
        region_layout_store, "get_cli_setting",
        lambda section, key, default=None: saved.get((section, key), default),
    )
    loaded = region_layout_store.load_region_layout()
    assert loaded.collapsed == frozenset({Region.CONTENT, Region.RIGHT_RAIL})


def test_load_defaults_to_everything_expanded(monkeypatch):
    monkeypatch.setattr(
        region_layout_store, "get_cli_setting", lambda section, key, default=None: default
    )
    assert region_layout_store.load_region_layout() == RegionLayout()


def test_load_ignores_unknown_region_names(monkeypatch):
    # A config hand-edited or written by a newer version must not crash the screen.
    monkeypatch.setattr(
        region_layout_store, "get_cli_setting",
        lambda section, key, default=None: ["content", "nonsense", "left_rail"],
    )
    loaded = region_layout_store.load_region_layout()
    assert loaded.collapsed == frozenset({Region.CONTENT, Region.LEFT_RAIL})


def test_load_tolerates_a_non_list_value(monkeypatch):
    monkeypatch.setattr(
        region_layout_store, "get_cli_setting", lambda section, key, default=None: "content"
    )
    assert region_layout_store.load_region_layout().collapsed == frozenset({Region.CONTENT})


def test_save_never_persists_solo(monkeypatch):
    saved = {}
    monkeypatch.setattr(
        region_layout_store, "save_setting_to_cli_config",
        lambda section, key, value: saved.__setitem__((section, key), value) or True,
    )
    region_layout_store.save_region_layout(RegionLayout().solo(Region.ITEMS))
    assert sorted(saved[("watchlists", "collapsed_regions")]) == ["content", "feeds"]
    assert ("watchlists", "solo_region") not in saved
