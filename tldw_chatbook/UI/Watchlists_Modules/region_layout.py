"""Collapse and solo state for the Watchlists workbench's five regions.

Pure state: no Textual import, no I/O. The screen's fiddliest interaction —
five independently collapsible regions plus a solo/restore toggle — lives
here so it can be tested without a Textual pilot.

Every mutator returns a new instance; the type is frozen and hashable, so a
Textual reactive can hold it and equality comparison decides whether to
re-render.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Region(str, Enum):
    """One collapsible region of the Watchlists workbench."""

    LEFT_RAIL = "left_rail"
    FEEDS = "feeds"
    ITEMS = "items"
    CONTENT = "content"
    RIGHT_RAIL = "right_rail"


#: Display order, left rail through right rail.
REGION_ORDER: tuple[Region, ...] = (
    Region.LEFT_RAIL,
    Region.FEEDS,
    Region.ITEMS,
    Region.CONTENT,
    Region.RIGHT_RAIL,
)

#: The vertically stacked centre panes. Only these may be soloed.
CENTRE_REGIONS: tuple[Region, ...] = (Region.FEEDS, Region.ITEMS, Region.CONTENT)


@dataclass(frozen=True)
class RegionLayout:
    """Which regions are collapsed, and whether one centre pane is soloed.

    Attributes:
        collapsed: Regions currently collapsed to their header. While
            `solo_region` is set, this is the solo-DERIVED view (the other
            centre panes collapsed around the soloed one) rather than a
            layout the user configured directly — see
            `collapsed_for_persistence`.
        solo_region: The centre region currently isolated via `solo`, or
            `None` when no pane is soloed.
        _pre_solo: The collapsed set as it was immediately before the most
            recent `solo` call, kept so a second `solo` call (or config
            persistence, via `collapsed_for_persistence`) can recover it.
            `None` when the layout has never been soloed.
    """

    collapsed: frozenset[Region] = frozenset()
    solo_region: Region | None = None
    _pre_solo: frozenset[Region] | None = None

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

    def toggle(self, region: Region) -> RegionLayout:
        """Collapse ``region`` if expanded, expand it if collapsed.

        A manual toggle clears any solo: the user has edited the layout by
        hand, so a later solo-restore must not resurrect a stale snapshot.

        Args:
            region: The region to collapse or expand.

        Returns:
            A new layout with `region`'s collapse state flipped and no
            solo in effect (`solo_region` is always `None` afterwards).
        """
        collapsed = set(self.collapsed)
        if region in collapsed:
            collapsed.discard(region)
        else:
            collapsed.add(region)
        return RegionLayout(collapsed=frozenset(collapsed))

    def solo(self, region: Region) -> RegionLayout:
        """Collapse the other centre panes around ``region``; call again to restore.

        Rails are unaffected — solo is about the centre stack only.

        Args:
            region: The centre region to isolate.

        Returns:
            A layout with the other centre regions collapsed, or the
            pre-solo layout if ``region`` is already soloed.

        Raises:
            ValueError: If ``region`` is a rail rather than a centre region.
        """
        if region not in CENTRE_REGIONS:
            raise ValueError(f"{region!r} is not a centre region; solo applies to {CENTRE_REGIONS}")

        if self.solo_region == region:
            return RegionLayout(collapsed=self._pre_solo or frozenset())

        # Re-soloing a different pane keeps the ORIGINAL pre-solo snapshot, so
        # restore always returns to what the user had before soloing at all.
        baseline = self._pre_solo if self.solo_region is not None else self.collapsed
        rails = {r for r in self.collapsed if r not in CENTRE_REGIONS}
        others = {r for r in CENTRE_REGIONS if r != region}
        return RegionLayout(
            collapsed=frozenset(rails | others),
            solo_region=region,
            _pre_solo=baseline,
        )

    def collapsed_for_persistence(self) -> frozenset[Region]:
        """The collapsed set that should survive a restart.

        While soloed, `collapsed` is the solo-DERIVED view — the other
        centre panes collapsed to isolate `solo_region` — not a layout the
        user configured. Persisting that verbatim would strand a restart in
        a view the user never chose, with no `_pre_solo` baseline left to
        restore from. Returning the pre-solo baseline instead means a
        restart reproduces what the user actually set up, and solo itself
        genuinely does not survive a restart, matching what
        `region_layout_store`'s module docstring promises.

        Returns:
            `_pre_solo` when `solo_region` is set, otherwise `collapsed`
            unchanged.
        """
        return self._pre_solo if self.solo_region is not None else self.collapsed
