from __future__ import annotations

import pytest

from tldw_chatbook.UI.Watchlists_Modules import region_layout
from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region, RegionLayout


def resolve(
    preferred: RegionLayout,
    width: int,
    *,
    read_mode: bool = True,
    article_focus: bool = False,
    priority_target: Region | None = None,
) -> RegionLayout:
    return region_layout.resolve_effective_layout(
        preferred,
        width=width,
        read_mode=read_mode,
        article_focus=article_focus,
        priority_target=priority_target,
    )


def test_declared_widths_orders_and_default_priorities():
    assert region_layout.PANE_GRIP_WIDTH == 5
    assert region_layout.PANE_MINIMUM_WIDTHS == {
        Region.LEFT_RAIL: 24,
        Region.ITEMS: 32,
        Region.RIGHT_RAIL: 30,
    }
    assert region_layout.CENTRE_COMFORT_WIDTH == 44
    assert region_layout.READ_SIDE_PANE_ORDER == (
        Region.LEFT_RAIL,
        Region.ITEMS,
        Region.RIGHT_RAIL,
    )
    assert region_layout.MANAGEMENT_SIDE_PANE_ORDER == (
        Region.LEFT_RAIL,
        Region.RIGHT_RAIL,
    )
    assert region_layout.READ_COLLAPSE_PRIORITY == (
        Region.RIGHT_RAIL,
        Region.LEFT_RAIL,
        Region.ITEMS,
    )
    assert region_layout.MANAGEMENT_COLLAPSE_PRIORITY == (
        Region.RIGHT_RAIL,
        Region.LEFT_RAIL,
    )


def test_read_all_open_boundary_collapses_inspector_first():
    preferred = RegionLayout()
    assert resolve(preferred, 145).collapsed == frozenset()
    assert resolve(preferred, 144).collapsed == frozenset({Region.RIGHT_RAIL})


def test_read_navigation_and_feed_items_boundary_collapses_navigation_next():
    preferred = RegionLayout().toggle_preferred(Region.RIGHT_RAIL)
    assert resolve(preferred, 115).collapsed == frozenset({Region.RIGHT_RAIL})
    assert resolve(preferred, 114).collapsed == frozenset(
        {Region.LEFT_RAIL, Region.RIGHT_RAIL}
    )


def test_read_feed_items_only_boundary_collapses_every_side_pane():
    preferred = (
        RegionLayout()
        .toggle_preferred(Region.LEFT_RAIL)
        .toggle_preferred(Region.RIGHT_RAIL)
    )
    assert resolve(preferred, 91).collapsed == frozenset(
        {Region.LEFT_RAIL, Region.RIGHT_RAIL}
    )
    assert resolve(preferred, 90).collapsed == frozenset(
        region_layout.COLLAPSIBLE_REGIONS
    )


def test_management_boundaries_exclude_feed_items():
    preferred = RegionLayout()
    assert resolve(preferred, 108, read_mode=False).collapsed == frozenset()
    assert resolve(preferred, 107, read_mode=False).collapsed == frozenset(
        {Region.RIGHT_RAIL}
    )
    assert resolve(preferred, 77, read_mode=False).collapsed == frozenset(
        {Region.LEFT_RAIL, Region.RIGHT_RAIL}
    )


def test_management_parks_feed_items_preference_but_keeps_mounted_rail_preferences():
    preferred = RegionLayout(
        collapsed=frozenset({Region.ITEMS, Region.RIGHT_RAIL})
    )
    effective = resolve(preferred, 200, read_mode=False)
    assert effective.collapsed == frozenset({Region.RIGHT_RAIL})


@pytest.mark.parametrize(
    ("read_mode", "mounted"),
    [
        (True, frozenset(region_layout.COLLAPSIBLE_REGIONS)),
        (False, frozenset({Region.LEFT_RAIL, Region.RIGHT_RAIL})),
    ],
)
def test_article_focus_collapses_mounted_side_panes_without_mutating_preferred(
    read_mode: bool,
    mounted: frozenset[Region],
):
    preferred = RegionLayout()
    effective = resolve(
        preferred,
        200,
        read_mode=read_mode,
        article_focus=True,
    )
    assert effective.collapsed == mounted
    assert preferred == RegionLayout()


def test_priority_target_is_protected_until_every_other_eligible_pane_collapses():
    preferred = RegionLayout()
    protected = resolve(
        preferred,
        114,
        priority_target=Region.RIGHT_RAIL,
    )
    assert protected.collapsed == frozenset({Region.LEFT_RAIL, Region.ITEMS})

    too_narrow = resolve(
        preferred,
        88,
        priority_target=Region.RIGHT_RAIL,
    )
    assert too_narrow.collapsed == frozenset(region_layout.COLLAPSIBLE_REGIONS)


def test_preferred_closed_panes_stay_closed():
    preferred = RegionLayout().toggle_preferred(Region.RIGHT_RAIL)
    assert resolve(preferred, 200).is_collapsed(Region.RIGHT_RAIL)


def test_repeated_resolution_is_idempotent():
    preferred = RegionLayout()
    effective = resolve(preferred, 114)
    assert resolve(effective, 114) == effective


@pytest.mark.parametrize("width", [59, 30, 1, 0, -1])
@pytest.mark.parametrize(
    ("read_mode", "mounted"),
    [
        (True, frozenset(region_layout.COLLAPSIBLE_REGIONS)),
        (False, frozenset({Region.LEFT_RAIL, Region.RIGHT_RAIL})),
    ],
)
def test_sub_sixty_widths_collapse_all_mounted_side_panes_without_raising(
    width: int,
    read_mode: bool,
    mounted: frozenset[Region],
):
    effective = resolve(RegionLayout(), width, read_mode=read_mode)
    assert effective.collapsed == mounted


def test_resolver_discards_a_retired_reader_collapse_value():
    preferred = RegionLayout(
        collapsed=frozenset({Region.LEFT_RAIL, Region.CONTENT})
    )
    effective = resolve(preferred, 145)
    assert effective.collapsed == frozenset({Region.LEFT_RAIL})
    assert not effective.is_collapsed(Region.CONTENT)
