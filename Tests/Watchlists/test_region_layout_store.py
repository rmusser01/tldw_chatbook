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


def test_load_applies_first_run_default_when_key_was_never_saved(monkeypatch):
    """`get_cli_setting` returns its `default` argument only when the key is
    absent — i.e. this is a genuine "never saved" case, not merely "saved as
    empty." That must apply the first-run default (CONTENT collapsed), not
    `RegionLayout()`. See `load_region_layout`'s docstring for why the two
    cases cannot be collapsed into one."""
    monkeypatch.setattr(
        region_layout_store, "get_cli_setting", lambda section, key, default=None: default
    )
    loaded = region_layout_store.load_region_layout()
    assert loaded == RegionLayout(collapsed=frozenset({Region.CONTENT}))
    assert loaded != RegionLayout()


def test_load_honors_an_explicit_empty_layout_as_everything_expanded(monkeypatch):
    """A key that was explicitly SAVED as `[]` (the user re-expanded
    everything, including CONTENT, and that was persisted) must be honored
    exactly — not silently overridden back to the first-run default, or a
    user's deliberate choice to keep CONTENT expanded could never survive a
    restart."""
    monkeypatch.setattr(
        region_layout_store, "get_cli_setting", lambda section, key, default=None: []
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
