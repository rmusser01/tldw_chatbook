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
    previous: RegionLayout | None = None,
) -> RegionLayout:
    return region_layout.resolve_effective_layout(
        preferred,
        width=width,
        read_mode=read_mode,
        article_focus=article_focus,
        priority_target=priority_target,
        previous=previous,
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


# --- Hysteresis at the collapse boundaries (TASK-22211) -------------------
#
# Boundary map (read mode, everything preferred open):
#   145 = 44 (centre) + 3*5 (grips) + 24 + 32 + 30  -> RIGHT_RAIL collapses < 145
#   115 = 145 - 30                                  -> LEFT_RAIL collapses < 115
#    91 = 115 - 24                                  -> ITEMS collapses < 91
# Management mode: 108 -> RIGHT_RAIL, 78 -> LEFT_RAIL.
# A collapsed pane re-expands only once the width clears its boundary by
# LAYOUT_HYSTERESIS_WIDTH (the Library reader precedent).


def test_hysteresis_constant_matches_the_library_reader_precedent():
    from tldw_chatbook.Library.library_media_reader_state import (
        LAYOUT_HYSTERESIS_WIDTH as READER_HYSTERESIS,
    )

    assert region_layout.LAYOUT_HYSTERESIS_WIDTH == READER_HYSTERESIS == 4


def test_one_cell_oscillation_at_the_read_inspector_boundary_is_stable():
    preferred = RegionLayout()
    layout = resolve(preferred, 145)
    assert layout.collapsed == frozenset()
    for _ in range(5):
        layout = resolve(preferred, 144, previous=layout)
        assert layout.collapsed == frozenset({Region.RIGHT_RAIL})
        layout = resolve(preferred, 145, previous=layout)
        assert layout.collapsed == frozenset({Region.RIGHT_RAIL})


def test_expansion_requires_clearing_the_boundary_by_the_hysteresis_width():
    preferred = RegionLayout()
    collapsed = resolve(preferred, 144, previous=resolve(preferred, 145))
    assert collapsed.collapsed == frozenset({Region.RIGHT_RAIL})

    still_held = resolve(preferred, 148, previous=collapsed)
    assert still_held.collapsed == frozenset({Region.RIGHT_RAIL})

    reopened = resolve(preferred, 149, previous=collapsed)
    assert reopened.collapsed == frozenset()

    # The expand boundary is itself stable in both directions: once open,
    # dropping one cell below it does not re-collapse (the bare threshold,
    # 145, governs collapse).
    assert resolve(preferred, 148, previous=reopened).collapsed == frozenset()


def test_crossing_by_at_least_the_hysteresis_width_still_flips_both_ways():
    preferred = RegionLayout()
    open_layout = resolve(preferred, 149)
    assert open_layout.collapsed == frozenset()

    collapsed = resolve(preferred, 144, previous=open_layout)
    assert collapsed.collapsed == frozenset({Region.RIGHT_RAIL})

    reopened = resolve(preferred, 149, previous=collapsed)
    assert reopened.collapsed == frozenset()


def test_one_cell_oscillation_at_the_management_boundary_is_stable():
    preferred = RegionLayout()
    layout = resolve(preferred, 108, read_mode=False)
    assert layout.collapsed == frozenset()
    for _ in range(5):
        layout = resolve(preferred, 107, read_mode=False, previous=layout)
        assert layout.collapsed == frozenset({Region.RIGHT_RAIL})
        layout = resolve(preferred, 108, read_mode=False, previous=layout)
        assert layout.collapsed == frozenset({Region.RIGHT_RAIL})
    assert resolve(preferred, 112, read_mode=False, previous=layout).collapsed == (
        frozenset()
    )


def test_hysteresis_composes_per_region_when_two_boundaries_are_near():
    preferred = RegionLayout()
    prev = RegionLayout(
        collapsed=frozenset({Region.RIGHT_RAIL, Region.LEFT_RAIL})
    )

    # LEFT_RAIL's expand boundary (115 + 4) is evaluated with RIGHT_RAIL's
    # suppressed width already deducted -- per-region state, not one flag.
    held = resolve(preferred, 118, previous=prev)
    assert held.collapsed == frozenset({Region.RIGHT_RAIL, Region.LEFT_RAIL})

    left_back = resolve(preferred, 119, previous=prev)
    assert left_back.collapsed == frozenset({Region.RIGHT_RAIL})

    # RIGHT_RAIL's own expand boundary is the all-open requirement + 4.
    assert resolve(preferred, 148, previous=prev).collapsed == frozenset(
        {Region.RIGHT_RAIL}
    )
    assert resolve(preferred, 149, previous=prev).collapsed == frozenset()


def test_hysteresis_applies_to_the_priority_target_too():
    preferred = RegionLayout()
    prev = resolve(preferred, 88, priority_target=Region.RIGHT_RAIL)
    assert prev.collapsed == frozenset(region_layout.COLLAPSIBLE_REGIONS)

    # Protected target's own requirement once LEFT/ITEMS are collapsed:
    # 44 + 15 + 30 = 89, so its expand boundary is 93.
    held = resolve(preferred, 92, priority_target=Region.RIGHT_RAIL, previous=prev)
    assert held.collapsed == frozenset(region_layout.COLLAPSIBLE_REGIONS)

    reopened = resolve(
        preferred, 93, priority_target=Region.RIGHT_RAIL, previous=prev
    )
    assert reopened.collapsed == frozenset({Region.LEFT_RAIL, Region.ITEMS})


def test_article_focus_still_collapses_everything_regardless_of_previous():
    preferred = RegionLayout()
    effective = resolve(
        preferred, 200, article_focus=True, previous=RegionLayout()
    )
    assert effective.collapsed == frozenset(region_layout.COLLAPSIBLE_REGIONS)


def test_resolution_with_previous_reaches_a_fixed_point():
    preferred = RegionLayout()
    layout = resolve(preferred, 144, previous=resolve(preferred, 145))
    assert resolve(preferred, 144, previous=layout) == layout
    layout = resolve(preferred, 118, previous=layout)
    assert resolve(preferred, 118, previous=layout) == layout


@pytest.mark.parametrize("read_mode", [True, False])
@pytest.mark.parametrize(
    "previous_collapsed",
    [
        frozenset(),
        frozenset({Region.RIGHT_RAIL}),
        frozenset({Region.RIGHT_RAIL, Region.LEFT_RAIL}),
        frozenset(region_layout.COLLAPSIBLE_REGIONS),
    ],
)
def test_hysteresis_never_holds_a_pane_open_the_bare_resolver_collapses(
    read_mode: bool,
    previous_collapsed: frozenset[Region],
):
    """Overflow safety: hysteresis only ever suppresses expansion.

    The resolved collapsed set with ``previous`` threaded is always a
    superset of the bare resolution's, so ``required_width`` can never
    exceed what the bare resolver already fit into ``width`` -- a pane is
    never stuck open at a width where it cannot fit.
    """
    preferred = RegionLayout()
    previous = RegionLayout(collapsed=previous_collapsed)
    for width in range(0, 201):
        bare = resolve(preferred, width, read_mode=read_mode)
        with_hysteresis = resolve(
            preferred, width, read_mode=read_mode, previous=previous
        )
        assert with_hysteresis.collapsed >= bare.collapsed


def test_no_previous_state_resolves_exactly_as_before():
    preferred = RegionLayout()
    for width in (90, 91, 114, 115, 144, 145):
        assert resolve(preferred, width) == resolve(
            preferred, width, previous=None
        )
    # A far shrink with previous threaded still collapses everything the
    # bare resolver would (convergence at genuinely narrow widths).
    assert resolve(
        preferred, 60, previous=resolve(preferred, 200)
    ).collapsed == frozenset(region_layout.COLLAPSIBLE_REGIONS)
