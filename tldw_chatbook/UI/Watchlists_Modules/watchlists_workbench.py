"""Watchlists workbench with a permanent horizontal centre canvas."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from loguru import logger
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from .pane_grip import RegionToggled, WatchlistsPaneGrip
from .region_layout import Region, RegionLayout

__all__ = [
    "REGION_TITLES",
    "RegionLayoutApplied",
    "RegionLayoutApplyFailed",
    "RegionToggled",
    "WatchlistsWorkbench",
]

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
_EXPANDED_SIDE_PANE_CLASS = "watchlists-has-expanded-side-pane"


class RegionLayoutApplyFailed(Message):
    """Report a rejected layout so the screen can roll back its preference."""

    def __init__(
        self,
        *,
        attempted: RegionLayout,
        fallback: RegionLayout,
    ) -> None:
        super().__init__()
        self.attempted = attempted
        self.fallback = fallback


class RegionLayoutApplied(Message):
    """Acknowledge a successful effective-layout transition."""

    def __init__(self, *, previous: RegionLayout, layout: RegionLayout) -> None:
        super().__init__()
        self.previous = previous
        self.layout = layout


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
        header: Callable[[], Widget] | None = None,
        *,
        read_mode: bool,
        **kwargs: Any,
    ) -> None:
        """Build the workbench from reusable region factories.

        Args:
            layout: Effective side-pane collapse state.
            content: Per-region factories. A remounted pane always receives a
                fresh widget instance.
            header: Optional factory for the chrome above the horizontal body.
            read_mode: Whether the permanent Reader view is active.
        """
        super().__init__(**kwargs)
        self.add_class("watchlists-workbench")
        self._content: dict[Region, Callable[[], Widget]] = dict(content or {})
        self._header = header
        self.read_mode = read_mode
        self.set_class(self.read_mode, "watchlists-read-mode")
        self.set_reactive(WatchlistsWorkbench.region_layout, layout)
        self._sync_expanded_side_pane_class(layout=layout)

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
        changed = [
            region
            for region in self._grip_regions
            if previous.is_collapsed(region) != layout.is_collapsed(region)
        ]
        prepared: dict[Region, Widget] = {}
        try:
            for region in changed:
                if (
                    not layout.is_collapsed(region)
                    and self._mounted_region_body(region) is None
                ):
                    prepared[region] = self._region_body(region)
        except Exception:
            self.set_reactive(WatchlistsWorkbench.region_layout, previous)
            self._sync_expanded_side_pane_class(layout=previous)
            logger.exception("Watchlists pane expansion factory failed")
            self.post_message(
                RegionLayoutApplyFailed(attempted=layout, fallback=previous)
            )
            return

        restore_focus = {
            region
            for region in prepared
            if self.query_one(
                f"#wl-grip-{region.value}", WatchlistsPaneGrip
            ).has_focus
        }
        try:
            for region, node in prepared.items():
                grip = self.query_one(
                    f"#wl-grip-{region.value}", WatchlistsPaneGrip
                )
                if region is Region.RIGHT_RAIL:
                    await body.mount(node, after=grip)
                else:
                    await body.mount(node, before=grip)
        except Exception:
            for node in prepared.values():
                if node.is_mounted:
                    await node.remove()
            self.set_reactive(WatchlistsWorkbench.region_layout, previous)
            self._sync_expanded_side_pane_class(layout=previous)
            logger.exception("Watchlists pane expansion mount failed")
            self.post_message(
                RegionLayoutApplyFailed(attempted=layout, fallback=previous)
            )
            return

        for region in changed:
            expanded = not layout.is_collapsed(region)
            if not expanded:
                mounted = self._mounted_region_body(region)
                if mounted is not None:
                    grip = self.query_one(
                        f"#wl-grip-{region.value}", WatchlistsPaneGrip
                    )
                    if self._contains_focus(mounted):
                        grip.focus()
                    await mounted.remove()
            grip = self.query_one(
                f"#wl-grip-{region.value}", WatchlistsPaneGrip
            )
            grip.expanded = expanded
        for region in restore_focus:
            mounted = self._mounted_region_body(region)
            if mounted is not None:
                mounted.focus()
        self._sync_expanded_side_pane_class(layout=layout)
        self.post_message(RegionLayoutApplied(previous=previous, layout=layout))

    @property
    def _grip_regions(self) -> tuple[Region, ...]:
        return _READ_GRIP_REGIONS if self.read_mode else _MANAGEMENT_GRIP_REGIONS

    def _sync_expanded_side_pane_class(
        self,
        *,
        read_mode: bool | None = None,
        layout: RegionLayout | None = None,
    ) -> None:
        """Expose whether the effective mode has an expanded side body."""
        read_mode = self.read_mode if read_mode is None else read_mode
        layout = self.region_layout if layout is None else layout
        side_regions = (
            _READ_GRIP_REGIONS if read_mode else _MANAGEMENT_GRIP_REGIONS
        )
        self.set_class(
            any(not layout.is_collapsed(region) for region in side_regions),
            _EXPANDED_SIDE_PANE_CLASS,
        )

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
        read_mode: bool,
        rebuild_regions: tuple[Region, ...] = (),
        rebuild_header: bool = False,
    ) -> None:
        """Incrementally apply a Read/management section view."""
        next_read_mode = read_mode
        if not self.is_mounted:
            self.read_mode = next_read_mode
            self.set_class(self.read_mode, "watchlists-read-mode")
            self.set_reactive(WatchlistsWorkbench.region_layout, layout)
            self._sync_expanded_side_pane_class(
                read_mode=next_read_mode, layout=layout
            )
            return

        replacement_header = (
            self._header()
            if rebuild_header and self._header is not None
            else None
        )
        with self.app.batch_update():
            await self._reconcile_body(
                read_mode=next_read_mode,
                layout=layout,
                rebuild_regions=rebuild_regions,
            )
            if replacement_header is not None:
                await self._replace_header(replacement_header)
            self.read_mode = next_read_mode
            self.set_class(self.read_mode, "watchlists-read-mode")
            self.set_reactive(WatchlistsWorkbench.region_layout, layout)
            self._sync_expanded_side_pane_class(
                read_mode=next_read_mode, layout=layout
            )

    async def _reconcile_body(
        self,
        *,
        read_mode: bool,
        layout: RegionLayout,
        rebuild_regions: tuple[Region, ...],
    ) -> None:
        """Prepare factories, then atomically reconcile direct children."""
        body = self._body()
        if body is None:
            return
        desired_ids = self._desired_body_ids(
            read_mode=read_mode,
            layout=layout,
        )
        mounted_ids = {child.id for child in body.children}
        prepared_nodes: dict[str, Widget] = {}
        created: set[Region] = set()
        for node_id in desired_ids:
            if node_id in mounted_ids:
                continue
            prepared_nodes[node_id] = self._node_for_id(node_id, layout=layout)
            region = self._region_from_body_id(node_id)
            if region is not None:
                created.add(region)

        prepared_content: dict[Region, Widget] = {}
        for region in rebuild_regions:
            if region in created or f"wl-region-{region.value}" not in desired_ids:
                continue
            factory = self._content.get(region)
            if factory is not None:
                prepared_content[region] = factory()

        restore_focus_ids = {
            node_id
            for node_id in prepared_nodes
            if node_id.startswith("wl-region-")
            and self._grip_has_focus(
                Region(node_id.removeprefix("wl-region-"))
            )
        }
        for child in list(body.children):
            if child.id not in desired_ids:
                region = self._region_from_body_id(child.id or "")
                if region is not None and self._contains_focus(child):
                    self.query_one(f"#wl-grip-{region.value}").focus()
                await child.remove()

        for index, node_id in enumerate(desired_ids):
            node = prepared_nodes.get(node_id)
            if node is not None:
                if index >= len(body.children):
                    await body.mount(node)
                else:
                    await body.mount(node, before=index)

        for node_id in restore_focus_ids:
            mounted = self.query_one(f"#{node_id}")
            mounted.focus()

        for region, replacement in prepared_content.items():
            container = self._mounted_region_body(region)
            if container is not None:
                await self._replace_region_content(container, replacement)

        grip_regions = (
            _READ_GRIP_REGIONS if read_mode else _MANAGEMENT_GRIP_REGIONS
        )
        for region in grip_regions:
            grip = self.query_one(f"#wl-grip-{region.value}", WatchlistsPaneGrip)
            grip.expanded = not layout.is_collapsed(region)

    def _desired_body_ids(
        self,
        *,
        read_mode: bool | None = None,
        layout: RegionLayout | None = None,
    ) -> list[str]:
        read_mode = self.read_mode if read_mode is None else read_mode
        layout = self.region_layout if layout is None else layout
        ids: list[str] = []
        if not layout.is_collapsed(Region.LEFT_RAIL):
            ids.append("wl-region-left_rail")
        ids.append("wl-grip-left_rail")
        if read_mode:
            if not layout.is_collapsed(Region.ITEMS):
                ids.append("wl-region-items")
            ids.extend(("wl-grip-items", "wl-region-content"))
        else:
            ids.append("wl-region-items")
        ids.append("wl-grip-right_rail")
        if not layout.is_collapsed(Region.RIGHT_RAIL):
            ids.append("wl-region-right_rail")
        return ids

    def _node_for_id(self, node_id: str, *, layout: RegionLayout) -> Widget:
        if node_id.startswith("wl-region-"):
            return self._region_body(Region(node_id.removeprefix("wl-region-")))
        region = Region(node_id.removeprefix("wl-grip-"))
        return WatchlistsPaneGrip(
            region,
            expanded=not layout.is_collapsed(region),
            id=node_id,
        )

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
        await self._replace_region_content(container, replacement)

    async def _replace_region_content(
        self, container: Widget, replacement: Widget
    ) -> None:
        """Commit one already-built factory replacement."""
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
        replacement = self._header()
        await self._replace_header(replacement)

    async def _replace_header(self, replacement: Widget) -> None:
        """Commit one already-built header replacement."""
        try:
            stale = self.query_one("#wl-centre-status")
        except NoMatches:
            stale = None
        if stale is not None:
            await stale.remove()
        await self.mount(replacement, before=0)

    def _contains_focus(self, widget: Widget) -> bool:
        """Whether the current focus lives at or below ``widget``."""
        focused = getattr(self.screen, "focused", None)
        while focused is not None:
            if focused is widget:
                return True
            focused = focused.parent
        return False

    def _grip_has_focus(self, region: Region) -> bool:
        try:
            return self.query_one(f"#wl-grip-{region.value}").has_focus
        except NoMatches:
            return False
