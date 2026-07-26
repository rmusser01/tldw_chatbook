import pytest

from tldw_chatbook.UI.Watchlists_Modules.region_layout import (
    CENTRE_REGIONS,
    Region,
    RegionLayout,
)


def test_default_layout_has_everything_visible():
    layout = RegionLayout()
    assert layout.collapsed == frozenset()
    assert layout.solo_region is None
    assert layout.visible() == (
        Region.LEFT_RAIL, Region.FEEDS, Region.ITEMS, Region.CONTENT, Region.RIGHT_RAIL,
    )


def test_toggle_collapses_then_expands():
    layout = RegionLayout().toggle(Region.CONTENT)
    assert layout.is_collapsed(Region.CONTENT)
    assert Region.CONTENT not in layout.visible()

    layout = layout.toggle(Region.CONTENT)
    assert not layout.is_collapsed(Region.CONTENT)


def test_toggle_returns_a_new_instance_and_leaves_the_original_alone():
    original = RegionLayout()
    changed = original.toggle(Region.ITEMS)
    assert original.collapsed == frozenset()
    assert changed is not original


def test_rails_collapse_independently_of_the_centre():
    layout = RegionLayout().toggle(Region.LEFT_RAIL).toggle(Region.RIGHT_RAIL)
    assert layout.is_collapsed(Region.LEFT_RAIL)
    assert layout.is_collapsed(Region.RIGHT_RAIL)
    for region in CENTRE_REGIONS:
        assert not layout.is_collapsed(region)


def test_solo_collapses_the_other_centre_regions_only():
    layout = RegionLayout().solo(Region.ITEMS)
    assert layout.solo_region == Region.ITEMS
    assert not layout.is_collapsed(Region.ITEMS)
    assert layout.is_collapsed(Region.FEEDS)
    assert layout.is_collapsed(Region.CONTENT)
    # Rails are untouched by solo.
    assert not layout.is_collapsed(Region.LEFT_RAIL)
    assert not layout.is_collapsed(Region.RIGHT_RAIL)


def test_solo_twice_restores_the_prior_layout():
    before = RegionLayout().toggle(Region.FEEDS).toggle(Region.LEFT_RAIL)
    after = before.solo(Region.ITEMS).solo(Region.ITEMS)
    assert after.collapsed == before.collapsed
    assert after.solo_region is None


def test_solo_on_a_different_region_re_solos_without_stacking():
    layout = RegionLayout().solo(Region.ITEMS).solo(Region.CONTENT)
    assert layout.solo_region == Region.CONTENT
    assert layout.is_collapsed(Region.ITEMS)
    assert not layout.is_collapsed(Region.CONTENT)
    # Restoring from here returns to the ORIGINAL pre-solo layout, not to the ITEMS solo.
    restored = layout.solo(Region.CONTENT)
    assert restored.collapsed == frozenset()
    assert restored.solo_region is None


def test_manual_toggle_while_soloed_clears_solo():
    # Otherwise a later Z would "restore" a layout the user has since edited by hand.
    layout = RegionLayout().solo(Region.ITEMS).toggle(Region.FEEDS)
    assert layout.solo_region is None
    assert not layout.is_collapsed(Region.FEEDS)


def test_solo_rejects_rails():
    with pytest.raises(ValueError, match="centre region"):
        RegionLayout().solo(Region.LEFT_RAIL)


def test_all_three_centre_regions_may_collapse_at_once():
    # Legal: each collapses to a one-line header that stays clickable, so this is recoverable.
    layout = RegionLayout()
    for region in CENTRE_REGIONS:
        layout = layout.toggle(region)
    assert all(layout.is_collapsed(region) for region in CENTRE_REGIONS)
    assert layout.visible() == (Region.LEFT_RAIL, Region.RIGHT_RAIL)


def test_collapsed_for_persistence_is_the_collapsed_set_when_not_soloed():
    # Regression coverage for PR #926 review, Bug 1: when nothing is soloed
    # there is no pre-solo baseline to prefer, so this must just be `collapsed`.
    layout = RegionLayout().toggle(Region.RIGHT_RAIL).toggle(Region.LEFT_RAIL)
    assert layout.collapsed_for_persistence() == layout.collapsed


def test_collapsed_for_persistence_returns_the_pre_solo_baseline_while_soloed():
    # Regression coverage for PR #926 review, Bug 1: `collapsed` while soloed
    # is the solo-DERIVED view (the other centre panes collapsed around the
    # soloed one) — not something the user configured. Persisting THAT would
    # strand a restart in a layout with no `_pre_solo` baseline left to
    # recover from, so the accessor used for persistence must return the
    # baseline instead.
    pre_solo = RegionLayout().toggle(Region.LEFT_RAIL)
    soloed = pre_solo.solo(Region.ITEMS)
    assert soloed.collapsed != pre_solo.collapsed  # sanity: solo did derive a new view
    assert soloed.collapsed_for_persistence() == pre_solo.collapsed
