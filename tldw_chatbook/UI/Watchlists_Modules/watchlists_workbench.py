"""Watchlists workbench with a permanent horizontal centre canvas."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from .pane_grip import RegionToggled, WatchlistsPaneGrip
from .region_layout import Region, RegionLayout

__all__ = ["REGION_TITLES", "RegionToggled", "WatchlistsWorkbench"]

REGION_TITLES: dict[Region, str] = {
    Region.LEFT_RAIL: "Watchlists",
    Region.ITEMS: "Items",
    Region.CONTENT: "Content",
    Region.RIGHT_RAIL: "Inspector",
}

SELF_HEADED_REGIONS: frozenset[Region] = frozenset(
    {Region.ITEMS, Region.RIGHT_RAIL}
)

_READ_GRIP_REGIONS: tuple[Region, ...] = (
    Region.LEFT_RAIL,
    Region.ITEMS,
    Region.RIGHT_RAIL,
)
_MANAGEMENT_GRIP_REGIONS: tuple[Region, ...] = (
    Region.LEFT_RAIL,
    Region.RIGHT_RAIL,
)


class WatchlistsWorkbench(Vertical):
    """Render optional chrome above a permanent horizontal centre.

    Read permanently anchors :class:`Region.CONTENT`; Navigation, Feed Items,
    and Inspector are independently collapsible side panes. Management tabs
    permanently anchor :class:`Region.ITEMS` and omit the Feed Items grip and
    Reader.
    """

    region_layout: reactive[RegionLayout] = reactive(RegionLayout())

    def __init__(
        self,
        layout: RegionLayout,
        content: Mapping[Region, Callable[[], Widget]] | None = None,
        hidden: frozenset[Region] = frozenset(),
        header: Callable[[], Widget] | None = None,
        collapsed_suffixes: Mapping[Region, str] | None = None,
        *,
        read_mode: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Build the workbench from reusable region factories.

        Args:
            layout: Effective side-pane collapse state.
            content: Per-region factories. A remounted pane always receives a
                fresh widget instance.
            hidden: Temporary screen compatibility adapter. CONTENT hidden
                selects management mode; otherwise it selects Read.
            header: Optional factory for the chrome above the horizontal body.
            collapsed_suffixes: Temporary no-op compatibility argument.
            read_mode: Explicit mode. When omitted, ``hidden`` selects it.
        """
        del collapsed_suffixes
        super().__init__(**kwargs)
        self.add_class("watchlists-workbench")
        self._content: dict[Region, Callable[[], Widget]] = dict(content or {})
        self._header = header
        self.read_mode = (
            Region.CONTENT not in hidden if read_mode is None else read_mode
        )
        self.set_reactive(WatchlistsWorkbench.region_layout, layout)

    def compose(self) -> ComposeResult:
        """Mount the header first and the horizontal workbench body second."""
        if self._header is not None:
            yield self._header()
        with Horizontal(id="wl-workbench-body"):
            for node in self._desired_body_nodes():
                yield node

    def _desired_body_nodes(self) -> list[Widget]:
        """Construct the currently desired body children in display order."""
        nodes: list[Widget] = []
        if not self.region_layout.is_collapsed(Region.LEFT_RAIL):
            nodes.append(self._region_body(Region.LEFT_RAIL))
        nodes.append(self._grip(Region.LEFT_RAIL))

        if self.read_mode:
            if not self.region_layout.is_collapsed(Region.ITEMS):
                nodes.append(self._region_body(Region.ITEMS))
            nodes.append(self._grip(Region.ITEMS))
            nodes.append(self._region_body(Region.CONTENT))
        else:
            nodes.append(self._region_body(Region.ITEMS))

        nodes.append(self._grip(Region.RIGHT_RAIL))
        if not self.region_layout.is_collapsed(Region.RIGHT_RAIL):
            nodes.append(self._region_body(Region.RIGHT_RAIL))
        return nodes

    def _region_body(self, region: Region) -> Vertical:
        """Build one expanded body around its current factory output."""
        factory = self._content.get(region)
        supplied = factory() if factory is not None else None
        children: list[Widget] = []
        if region not in SELF_HEADED_REGIONS:
            children.append(
                Static(REGION_TITLES[region], classes="watchlists-region-title")
            )
        if supplied is not None:
            children.append(supplied)
        body = Vertical(
            *children,
            id=f"wl-region-{region.value}",
            classes=f"watchlists-region watchlists-region-{region.value}",
        )
        body.can_focus = True
        return body

    def _grip(self, region: Region) -> WatchlistsPaneGrip:
        """Build a side-pane grip with its effective state."""
        return WatchlistsPaneGrip(
            region,
            expanded=not self.region_layout.is_collapsed(region),
            id=f"wl-grip-{region.value}",
        )

    async def watch_region_layout(
        self, previous: RegionLayout, layout: RegionLayout
    ) -> None:
        """Mount or remove only side bodies whose collapse state changed."""
        if not self.is_mounted:
            return
        body = self._body()
        if body is None:
            return
        for region in self._grip_regions:
            if previous.is_collapsed(region) == layout.is_collapsed(region):
                continue
            grip = self.query_one(f"#wl-grip-{region.value}", WatchlistsPaneGrip)
            expanded = not layout.is_collapsed(region)
            grip.expanded = expanded
            mounted = self._mounted_region_body(region)
            if not expanded:
                if mounted is not None:
                    await mounted.remove()
                continue
            if mounted is None:
                node = self._region_body(region)
                if region is Region.RIGHT_RAIL:
                    await body.mount(node, after=grip)
                else:
                    await body.mount(node, before=grip)

    @property
    def _grip_regions(self) -> tuple[Region, ...]:
        return _READ_GRIP_REGIONS if self.read_mode else _MANAGEMENT_GRIP_REGIONS

    def _body(self) -> Horizontal | None:
        try:
            return self.query_one("#wl-workbench-body", Horizontal)
        except NoMatches:
            return None

    def _mounted_region_body(self, region: Region) -> Widget | None:
        try:
            return self.query_one(f"#wl-region-{region.value}")
        except NoMatches:
            return None

    async def apply_section_view(
        self,
        *,
        layout: RegionLayout,
        read_mode: bool | None = None,
        hidden: frozenset[Region] | None = None,
        rebuild_regions: tuple[Region, ...] = (),
        rebuild_header: bool = False,
    ) -> None:
        """Incrementally apply a Read/management section view.

        ``hidden`` remains a temporary adapter for the live screen through
        Task 6. Explicit ``read_mode`` wins when both are provided.
        """
        next_read_mode = read_mode
        if next_read_mode is None:
            next_read_mode = (
                self.read_mode
                if hidden is None
                else Region.CONTENT not in hidden
            )
        if not self.is_mounted:
            self.read_mode = next_read_mode
            self.set_reactive(WatchlistsWorkbench.region_layout, layout)
            return

        with self.app.batch_update():
            self.read_mode = next_read_mode
            self.set_reactive(WatchlistsWorkbench.region_layout, layout)
            created = await self._reconcile_body()
            for region in rebuild_regions:
                if region not in created:
                    await self.refresh_region_content(region)
            if rebuild_header:
                await self.refresh_header_content()

    async def _reconcile_body(self) -> set[Region]:
        """Reconcile direct children while retaining every reusable node."""
        body = self._body()
        if body is None:
            return set()
        desired_ids = self._desired_body_ids()
        for child in list(body.children):
            if child.id not in desired_ids:
                await child.remove()

        created: set[Region] = set()
        for index, node_id in enumerate(desired_ids):
            try:
                self.query_one(f"#{node_id}")
            except NoMatches:
                node = self._node_for_id(node_id)
                region = self._region_from_body_id(node_id)
                if region is not None:
                    created.add(region)
                if index >= len(body.children):
                    await body.mount(node)
                else:
                    await body.mount(node, before=index)

        for region in self._grip_regions:
            grip = self.query_one(f"#wl-grip-{region.value}", WatchlistsPaneGrip)
            grip.expanded = not self.region_layout.is_collapsed(region)
        return created

    def _desired_body_ids(self) -> list[str]:
        ids: list[str] = []
        if not self.region_layout.is_collapsed(Region.LEFT_RAIL):
            ids.append("wl-region-left_rail")
        ids.append("wl-grip-left_rail")
        if self.read_mode:
            if not self.region_layout.is_collapsed(Region.ITEMS):
                ids.append("wl-region-items")
            ids.extend(("wl-grip-items", "wl-region-content"))
        else:
            ids.append("wl-region-items")
        ids.append("wl-grip-right_rail")
        if not self.region_layout.is_collapsed(Region.RIGHT_RAIL):
            ids.append("wl-region-right_rail")
        return ids

    def _node_for_id(self, node_id: str) -> Widget:
        if node_id.startswith("wl-region-"):
            return self._region_body(Region(node_id.removeprefix("wl-region-")))
        return self._grip(Region(node_id.removeprefix("wl-grip-")))

    @staticmethod
    def _region_from_body_id(node_id: str) -> Region | None:
        if not node_id.startswith("wl-region-"):
            return None
        return Region(node_id.removeprefix("wl-region-"))

    async def refresh_region_content(self, region: Region) -> None:
        """Replace only one mounted body's factory output, failure-safely."""
        factory = self._content.get(region)
        if factory is None:
            return
        container = self._mounted_region_body(region)
        if container is None:
            return
        replacement = factory()
        stale = [
            child
            for child in container.children
            if not child.has_class("watchlists-region-title")
        ]
        for child in stale:
            await child.remove()
        await container.mount(replacement)

    async def refresh_header_content(self) -> None:
        """Replace only header factory output, leaving the body untouched."""
        if self._header is None:
            return
        try:
            stale = self.query_one("#wl-centre-status")
        except NoMatches:
            stale = None
        replacement = self._header()
        if stale is not None:
            await stale.remove()
        await self.mount(replacement, before=0)

    def set_collapsed_suffixes(self, suffixes: Mapping[Region, str]) -> None:
        """Compatibility no-op retained until the live screen stops calling it."""
        del suffixes
