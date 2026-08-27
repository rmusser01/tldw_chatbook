"""Preferred and effective pane state for the Watchlists workbench.

Pure state: no Textual import, no I/O. The user's preferred side-pane state
and the transient responsive/Article Focus result can therefore be tested
without a Textual pilot.

Every mutator returns a new instance; the type is frozen and hashable, so a
Textual reactive can hold it and equality comparison decides whether to
re-render.

The FEEDS region was removed in task-2513 (reader-first IA, ADR-042) with no
migration code: a persisted ``"feeds"`` string from before the removal is an
unknown region name now, dropped with a debug log by
`region_layout_store.load_region_layout`'s existing guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Region(str, Enum):
    """One collapsible region of the Watchlists workbench."""

    LEFT_RAIL = "left_rail"
    ITEMS = "items"
    CONTENT = "content"
    RIGHT_RAIL = "right_rail"


#: Display order, left rail through right rail.
REGION_ORDER: tuple[Region, ...] = (
    Region.LEFT_RAIL,
    Region.ITEMS,
    Region.CONTENT,
    Region.RIGHT_RAIL,
)

#: Side panes whose preferred open/collapsed state may be changed by the user.
COLLAPSIBLE_REGIONS: tuple[Region, ...] = (
    Region.LEFT_RAIL,
    Region.ITEMS,
    Region.RIGHT_RAIL,
)

#: Side panes mounted in Read, in display order.
READ_SIDE_PANE_ORDER: tuple[Region, ...] = COLLAPSIBLE_REGIONS

#: Side panes mounted around a management canvas, in display order.
MANAGEMENT_SIDE_PANE_ORDER: tuple[Region, ...] = (
    Region.LEFT_RAIL,
    Region.RIGHT_RAIL,
)

#: Fixed width of every mounted side-pane grip.
PANE_GRIP_WIDTH = 5

#: Minimum expanded width of each side pane.
PANE_MINIMUM_WIDTHS: dict[Region, int] = {
    Region.LEFT_RAIL: 24,
    Region.ITEMS: 32,
    Region.RIGHT_RAIL: 30,
}

#: Preferred comfort width for the permanent Reader or management canvas.
CENTRE_COMFORT_WIDTH = 44

#: Extra columns a responsively collapsed pane must clear before it
#: re-expands (TASK-22211). Same value and role as the Library reader's
#: `LAYOUT_HYSTERESIS_WIDTH` (`Library/library_media_reader_state.py`):
#: collapse happens exactly at a pane's bare width boundary, but expansion
#: waits until the width clears that boundary by this margin, so a +/-1-cell
#: resize oscillation (or a 2-cell scrollbar toggle) at the boundary cannot
#: mount/remove a whole pane per event. Must stay wider than the default
#: vertical scrollbar (2 cells) for the scrollbar-toggle guard to hold.
LAYOUT_HYSTERESIS_WIDTH = 4

#: Default collapse order when Read becomes too narrow.
READ_COLLAPSE_PRIORITY: tuple[Region, ...] = (
    Region.RIGHT_RAIL,
    Region.LEFT_RAIL,
    Region.ITEMS,
)

#: Default collapse order when a management tab becomes too narrow.
MANAGEMENT_COLLAPSE_PRIORITY: tuple[Region, ...] = (
    Region.RIGHT_RAIL,
    Region.LEFT_RAIL,
)

@dataclass(frozen=True)
class RegionLayout:
    """A preferred or effective set of collapsed regions.

    Attributes:
        collapsed: Regions currently collapsed. Preferred layouts contain
            only `COLLAPSIBLE_REGIONS`.
    """

    collapsed: frozenset[Region] = frozenset()

    def is_collapsed(self, region: Region) -> bool:
        """Whether ``region`` is currently collapsed to its header.

        Args:
            region: The region to check.

        Returns:
            `True` if `region` is in `collapsed`, `False` otherwise.
        """
        return region in self.collapsed

    def visible(self) -> tuple[Region, ...]:
        """Expanded regions, in display order.

        Returns:
            The regions not in `collapsed`, in `REGION_ORDER` order.
        """
        return tuple(r for r in REGION_ORDER if r not in self.collapsed)

    def toggle_preferred(self, region: Region) -> RegionLayout:
        """Flip one side pane's preferred collapse state.

        Args:
            region: The collapsible side pane to update.

        Returns:
            A new preferred layout with ``region`` flipped.

        Raises:
            ValueError: If ``region`` is the permanent centre content.
        """
        if region not in COLLAPSIBLE_REGIONS:
            raise ValueError(f"{region!r} is not a collapsible side pane")

        collapsed = set(self.collapsed).intersection(COLLAPSIBLE_REGIONS)
        collapsed.symmetric_difference_update({region})
        return RegionLayout(collapsed=frozenset(collapsed))


def resolve_effective_layout(
    preferred: RegionLayout,
    *,
    width: int,
    read_mode: bool,
    article_focus: bool,
    priority_target: Region | None,
    previous: RegionLayout | None = None,
) -> RegionLayout:
    """Derive mounted side-pane collapses without changing ``preferred``.

    Args:
        preferred: The user's preferred side-pane layout.
        width: Available workbench width in terminal columns.
        read_mode: Whether Read's Feed Items pane is mounted.
        article_focus: Whether every mounted side pane is temporarily hidden.
        priority_target: An expanded mounted pane to collapse last, if any.
        previous: Previously resolved effective layout used for hysteresis.

    Returns:
        A new effective layout with no solo or restore state.
    """
    mounted = READ_SIDE_PANE_ORDER if read_mode else MANAGEMENT_SIDE_PANE_ORDER
    priority = READ_COLLAPSE_PRIORITY if read_mode else MANAGEMENT_COLLAPSE_PRIORITY
    preferred_collapsed = set(preferred.collapsed).intersection(mounted)

    if article_focus:
        return RegionLayout(
            collapsed=frozenset(preferred_collapsed.union(mounted))
        )

    required_width = (
        CENTRE_COMFORT_WIDTH
        + len(mounted) * PANE_GRIP_WIDTH
        + sum(
            PANE_MINIMUM_WIDTHS[region]
            for region in mounted
            if region not in preferred_collapsed
        )
    )
    candidates = [
        region for region in priority if region not in preferred_collapsed
    ]
    if priority_target in candidates:
        candidates.remove(priority_target)
        candidates.append(priority_target)

    nominal_collapsed = set(preferred_collapsed)
    for region in candidates:
        if required_width <= width:
            break
        nominal_collapsed.add(region)
        required_width -= PANE_MINIMUM_WIDTHS[region]

    if previous is None:
        return RegionLayout(collapsed=frozenset(nominal_collapsed))

    nominally_open = set(mounted).difference(nominal_collapsed)
    previously_open = set(mounted).difference(previous.collapsed)
    accepted_open = nominally_open.intersection(previously_open)
    accepted_width = (
        CENTRE_COMFORT_WIDTH
        + len(mounted) * PANE_GRIP_WIDTH
        + sum(PANE_MINIMUM_WIDTHS[region] for region in accepted_open)
    )

    for region in reversed(candidates):
        if region not in nominally_open or region not in previous.collapsed:
            continue
        reopened_width = accepted_width + PANE_MINIMUM_WIDTHS[region]
        if width < reopened_width + LAYOUT_HYSTERESIS_WIDTH:
            break
        accepted_open.add(region)
        accepted_width = reopened_width

    return RegionLayout(
        collapsed=frozenset(set(mounted).difference(accepted_open))
    )
