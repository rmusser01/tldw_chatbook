import pytest

from tldw_chatbook.UI.Watchlists_Modules import region_layout
from tldw_chatbook.UI.Watchlists_Modules.region_layout import (
    Region,
    RegionLayout,
)


def test_default_layout_has_everything_visible():
    layout = RegionLayout()
    assert layout.collapsed == frozenset()
    assert layout.visible() == (
        Region.LEFT_RAIL, Region.ITEMS, Region.CONTENT, Region.RIGHT_RAIL,
    )


def test_feeds_is_no_longer_a_region():
    # Persisted "feeds" collapse strings from before the removal are dropped
    # by `region_layout_store.load_region_layout`'s unknown-region guard
    # (ADR-042) — which only works because this lookup raises.
    with pytest.raises(ValueError):
        Region("feeds")


def test_only_side_panes_are_collapsible_preferences():
    assert region_layout.COLLAPSIBLE_REGIONS == (
        Region.LEFT_RAIL,
        Region.ITEMS,
        Region.RIGHT_RAIL,
    )


def test_preferred_toggle_rejects_the_permanent_reader():
    with pytest.raises(ValueError, match="collapsible"):
        RegionLayout().toggle_preferred(Region.CONTENT)


def test_navigation_feed_items_and_inspector_toggle_independently():
    layout = RegionLayout()
    for region in region_layout.COLLAPSIBLE_REGIONS:
        layout = layout.toggle_preferred(region)
        assert layout.is_collapsed(region)

    layout = layout.toggle_preferred(Region.ITEMS)
    assert layout.collapsed == frozenset({Region.LEFT_RAIL, Region.RIGHT_RAIL})


def test_preferred_toggle_returns_a_new_instance_and_leaves_the_original_alone():
    original = RegionLayout()
    changed = original.toggle_preferred(Region.ITEMS)
    assert original.collapsed == frozenset()
    assert changed is not original


@pytest.mark.parametrize(
    "member",
    ["solo_region", "_pre_solo", "solo", "toggle", "collapsed_for_persistence"],
)
def test_transitional_layout_compatibility_is_removed(member: str):
    assert not hasattr(RegionLayout, member)
