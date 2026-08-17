"""Watchlists workbench: two collapsible rails around a stacked centre.

The shared ``DestinationWorkbench`` cannot express this layout — it is a
fixed ``Horizontal`` of equal-width panes composed once from a frozen tuple,
with no collapse, resize, or vertical stacking. If the collapse behaviour
here proves useful to a second screen, it graduates into the shared widget
then; generalising ahead of a second consumer is not worth it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Static

from .region_layout import CENTRE_REGIONS, REGION_ORDER, Region, RegionLayout


#: Human-readable titles, used for both expanded bodies and collapsed headers.
REGION_TITLES: dict[Region, str] = {
    Region.LEFT_RAIL: "Watchlists",
    Region.ITEMS: "Items",
    Region.CONTENT: "Content",
    Region.RIGHT_RAIL: "Inspector",
}

#: Regions whose supplied pane draws its own heading, so adding the generic
#: region title from `REGION_TITLES` would produce a *second* heading.
#:
#: This is deliberately NOT the same signal as "a content factory was
#: supplied for this region". The two diverge at LEFT_RAIL: it supplies
#: content too, but that content (`WatchlistTree`) composes only navigation
#: `Button`s and no heading widget at all (see `watchlist_tree.py::compose`),
#: so keying suppression on factory-presence -- as this module did before --
#: rendered the expanded left rail as an unlabelled bordered box while its
#: *collapsed* header still read "▸ Watchlists". Membership here means
#: "the pane supplies its own heading", which is the actual rule.
#:
#: Task 4 wired `ContentPane` into CONTENT (`watchlists_collections_screen.py`
#: `_build_content_pane`) and deliberately did NOT add `Region.CONTENT` here:
#: `ContentPane.compose()` yields only a bare `Static` (the placeholder or the
#: rendered item), no heading widget of its own -- the same shape as
#: LEFT_RAIL's `WatchlistTree` above, which is excluded for the identical
#: reason. CONTENT gets the generic "Content" title like LEFT_RAIL gets
#: "Watchlists".
SELF_HEADED_REGIONS: frozenset[Region] = frozenset(
    {Region.ITEMS, Region.RIGHT_RAIL}
)


class RegionToggled(Message):
    """A collapsed region's header or rail handle was activated."""

    def __init__(self, region: Region) -> None:
        super().__init__()
        self.region = region


class WatchlistsWorkbench(Horizontal):
    """Renders a :class:`RegionLayout` as rails plus a stacked centre.

    The reactive is named ``region_layout``, not ``layout``: ``Widget.layout``
    is an existing, unsettable Textual property (``textual/widget.py``) that
    the compositor reads on every arrange pass (``widget.layout.arrange(...)``
    in ``textual/_arrange.py``). A same-named reactive here shadows it, so
    Textual ends up calling ``.arrange()`` on our ``RegionLayout`` domain
    object instead of the real layout strategy and crashes with
    ``AttributeError: 'RegionLayout' object has no attribute 'arrange'`` on
    first render. Verified with a minimal repro using an unrelated reactive
    also named ``layout``, so this is a general Widget-subclass constraint,
    not specific to this dataclass.

    Attributes:
        region_layout: The current collapse/solo state. Setting it swaps ONLY
            the regions whose rendered form actually changed — a region that
            flipped between its collapsed one-line header and its expanded
            body. Regions that stayed expanded keep their live widget
            instance and are patched in place (the sole-centre CSS marker is
            the only other thing a layout change can move for them).

            This was `recompose=True` until task-15461, which meant every
            `z`/`Z`/`[`/`]`/chevron rebuilt all four regions: measured on a
            seeded screen, one `]` tore down and rebuilt 76–99 widgets
            (the tree with its per-expanded-watchlist synchronous source
            query, the tab strip, the article list, the Inspector) to
            collapse a rail. `content` still holds FACTORIES rather than
            instances — see `__init__` — because a swapped-out region's
            replacement must still be a brand new widget.
    """

    # task-16843: a shared *instance* default (`reactive(RegionLayout())`
    # installs the SAME `RegionLayout` object on every workbench instance
    # until `__init__`'s `set_reactive` calls below overwrite it) -- but
    # harmless: `RegionLayout` is `frozen=True` and every field is itself
    # immutable (`frozenset`, `Region | None`), so there is no mutable
    # container underneath to mutate in place. Allowlisted in
    # `Tests/Architecture/test_reactive_mutable_default_inventory.py`'s
    # `IMMUTABLE_INSTANCE_ALLOWLIST` rather than rewritten into a factory.
    region_layout: reactive[RegionLayout] = reactive(RegionLayout())

    def __init__(
        self,
        layout: RegionLayout,
        content: Mapping[Region, Callable[[], Widget]] | None = None,
        hidden: frozenset[Region] = frozenset(),
        header: Callable[[], Widget] | None = None,
        collapsed_suffixes: Mapping[Region, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Build the workbench, seeding `region_layout` without a region sync.

        Args:
            layout: Initial collapse/solo state.
            content: Per-region **factories**, not widget instances — a
                region whose rendered form changes (collapsed header <->
                expanded body) is swapped for a freshly built one, so the
                factory must be callable more than once.
                Passing already-constructed instances was tried first and
                verified broken empirically: a container widget's
                constructor-supplied children (e.g. ``Vertical(Static(...),
                Static(...))``) are consumed on that widget's *first* mount
                only. Once such an instance has been unmounted (as part of
                a recompose) and the *same* instance is handed back to
                `compose()` again, it remounts with zero children — its
                grandchildren do not come back. A leaf widget with no
                children of its own (e.g. a bare ``Label``) happens to
                survive remounting, and a widget with an *overridden*
                ``compose()`` (which regenerates its children from scratch
                every call) also survives — which is exactly why this was
                easy to miss with a single-recompose test. A factory
                sidesteps the whole class of bug by handing back a brand
                new instance on every region rebuild, matching how
                ``WatchlistsTabStrip`` (an overridden-``compose()``
                widget) already behaves.
            hidden: Centre regions to omit from `compose()` entirely — no
                collapsed header, no body (TASK-1344 AC#4: gated regions
                UNMOUNT rather than keep a one-row header). The caller
                (`WatchlistsCollectionsScreen._hidden_centre_regions`)
                decides which regions this is, keyed on `active_section`;
                the workbench itself has no opinion about tabs. The initial
                value only: since task-15461 a section switch no longer
                rebuilds this widget, so the caller pushes the new set
                through `apply_section_view` instead, which mounts or
                unmounts exactly the centre regions that crossed the
                boundary. A plain toggle/solo (`_apply_layout` pushing a new
                `region_layout` onto the ALREADY-mounted instance) never
                changes which tab is active, so `hidden` is untouched there.
            header: An optional factory for a widget rendered as the FIRST
                child of the centre stack, unconditionally — regardless of
                `hidden`. TASK-1344: the section tab strip and the
                snapshot's own loading/error/empty markers are cross-
                cutting chrome, not region content, so they must survive
                CONTENT being hidden on every non-Read tab. Since
                task-2513 removed the FEEDS region (whose own inline copy
                used to carry that chrome on Read), the screen wires this
                on EVERY tab — Read included; `None` is for callers (and
                tests) that genuinely have no header to show. This class
                stays a generic building block with no opinion about tabs,
                and any two factories that both mount an id must never be
                combined by a caller, same as ever.
        """
        super().__init__(**kwargs)
        self.add_class("watchlists-workbench")
        self._content: dict[Region, Callable[[], Widget]] = dict(content or {})
        self._hidden = frozenset(hidden)
        self._header = header
        # Extra text appended to a collapsed region's header (task-2513 Task
        # 9): "▸ Watchlists  12 unread". Mutable via `set_collapsed_suffixes`
        # because counts change while the region stays collapsed.
        self._collapsed_suffixes: dict[Region, str] = dict(collapsed_suffixes or {})
        self.set_reactive(WatchlistsWorkbench.region_layout, layout)

    def compose(self) -> ComposeResult:
        """Render the left rail, the stacked centre, and the right rail.

        Runs once per mount. A later `region_layout` change no longer
        re-runs this (task-15461): `watch_region_layout` swaps only the
        regions whose form actually changed — see that watcher.

        Returns:
            The left-rail region, the centre `VerticalScroll` (an optional
            header, then ITEMS/CONTENT minus anything in `self._hidden`), and
            the right-rail region, in that order.
        """
        yield self._region_widget(Region.LEFT_RAIL)

        with VerticalScroll(id="wl-centre", classes="watchlists-centre"):
            if self._header is not None:
                yield self._header()
            for region in CENTRE_REGIONS:
                if region in self._hidden:
                    continue
                yield self._region_widget(region)

        yield self._region_widget(Region.RIGHT_RAIL)

    def _region_widget(self, region: Region) -> Widget:
        """Build one region: a titled body, or a focusable one-line header.

        Returns a constructed widget rather than yielding, so `compose` stays
        the single place that mounts anything. Building children positionally
        avoids the `with container: ... ; yield container` shape, which
        double-mounts — Textual's `with` already adds the container.

        Args:
            region: The region to build, per `self.region_layout`'s current
                collapse state.

        Returns:
            A focusable `Button` header when `region` is collapsed,
            otherwise a focusable `Vertical` body holding the region's
            title and its supplied content (or the placeholder stub).
        """
        if self.region_layout.is_collapsed(region):
            # A Button, not a Static: a collapsed region must stay focusable
            # and clickable, or collapsing it is one-way.
            suffix = self._collapsed_suffixes.get(region, "")
            header = Button(
                f"▸ {REGION_TITLES[region]}" + (f"  {suffix}" if suffix else ""),
                id=f"wl-header-{region.value}",
                compact=True,
            )
            header.add_class("watchlists-region-header")
            header.tooltip = f"Expand {REGION_TITLES[region]}"
            return header

        factory = self._content.get(region)
        supplied = factory() if factory is not None else None
        # Two independent questions, deliberately kept apart:
        #
        #   1. Does this region draw the generic `REGION_TITLES` heading?
        #      Only when its pane does NOT supply one — `SELF_HEADED_REGIONS`
        #      (see that constant for why this is not the factory check).
        #   2. Does this region render the placeholder stub, or real content?
        #      That IS the factory check.
        #
        # They looked identical while ITEMS/RIGHT_RAIL were the only wired
        # self-headed regions, which is how LEFT_RAIL — wired, but with a
        # heading-less `WatchlistTree` inside — ended up as an unlabelled box.
        children: list[Widget] = []
        if region not in SELF_HEADED_REGIONS:
            children.append(
                Static(REGION_TITLES[region], classes="watchlists-region-title")
            )
        # Whole-branch review (Minor): there used to be a `REGION_PLACEHOLDERS`
        # branch here ("Reader arrives in the next slice.") for a region with
        # no factory. Task 4 wired the last unwired region, so every region the
        # screen builds supplies content; the branch could only ever be reached
        # by a test that constructed a workbench with no content at all, which
        # made a "coming soon" string look like live product copy to a grep.
        if supplied is not None:
            children.append(supplied)
        classes = ["watchlists-region", f"watchlists-region-{region.value}"]
        if self._is_sole_expanded_centre_region(region, self.region_layout):
            # A CSS hook for the solo case. `.watchlists-region-content`
            # carries a `max-height` cap so a long article cannot crowd
            # ITEMS out of the centre stack — but when ITEMS is collapsed
            # to its one-line header there is nothing left to crowd, and
            # the cap turns solo-CONTENT into a short scrolling window with
            # blank rows under it. Nothing in the DOM distinguished that
            # state before this class: `RegionLayout.solo` only collapses
            # the *siblings*, so the soloed region itself is
            # indistinguishable from an ordinarily-expanded one.
            #
            # Keyed on "sole expanded centre region" rather than on
            # `solo_region` because the two produce the same DOM and want the
            # same layout: `Z` on CONTENT and manually collapsing ITEMS
            # with `z` both leave CONTENT alone in the centre.
            classes.append("watchlists-region-sole-centre")
        body = Vertical(
            *children,
            id=f"wl-region-{region.value}",
            classes=" ".join(classes),
        )
        # Regions must be keyboard-reachable, or `z` cannot target them.
        body.can_focus = True
        return body

    def _is_sole_expanded_centre_region(
        self, region: Region, layout: RegionLayout
    ) -> bool:
        """Whether ``region`` is the only centre region still expanded.

        True exactly when the centre stack shows one real pane and, for
        every OTHER centre region that is not hidden outright, a one-line
        header — the state `RegionLayout.solo` produces, and the state a
        user reaches by collapsing the other one by hand. A region in
        `self._hidden` (TASK-1344: CONTENT off the Read tab) is never
        rendered at all, not even as a header, so it is excluded from
        "expanded" the same way a rail is — without this, ITEMS would never
        read as sole-expanded on a non-Read tab (CONTENT's real
        `region_layout.collapsed` membership is whatever the user left it
        at on Read, not "hidden", so counting it unfiltered would make
        `expanded` include a region that in fact never mounted).

        Args:
            region: The region to test. Rails always answer `False`; solo
                applies to the centre stack only (`RegionLayout.solo`).
            layout: The layout to answer against. Passed explicitly rather
                than read off `self` (task-15461) so `watch_region_layout`
                can ask the same question of the OUTGOING layout and tell
                whether this marker is what actually moved.

        Returns:
            `True` if `region` is a centre region and every other
            non-hidden centre region is collapsed, `False` otherwise.
        """
        if region not in CENTRE_REGIONS:
            return False
        expanded = [
            r
            for r in CENTRE_REGIONS
            if r not in self._hidden and not layout.is_collapsed(r)
        ]
        return expanded == [region]

    # --- scoped layout/section updates (task-15461) -----------------------

    async def watch_region_layout(
        self, previous: RegionLayout, layout: RegionLayout
    ) -> None:
        """Swap only the regions whose rendered form the new layout moves.

        The reactive used to be `recompose=True`, which rebuilt all four
        regions for every `z`/`Z`/`[`/`]`/chevron. A layout change can only
        move two things about a region: whether it renders as its collapsed
        one-line header or its expanded body, and whether it carries the
        `watchlists-region-sole-centre` marker. The first needs a widget
        swap; the second is a class toggle on the live widget, which keeps
        the pane instance (and everything mounted inside it) alive.

        Args:
            previous: The layout being replaced.
            layout: The layout now in effect (already stored on `self`, so
                `_region_widget` builds against it).
        """
        if not self.is_mounted:
            return
        await self._sync_regions(previous)

    async def apply_section_view(
        self,
        *,
        hidden: frozenset[Region],
        layout: RegionLayout,
        rebuild_regions: tuple[Region, ...] = (),
        rebuild_header: bool = False,
    ) -> None:
        """Move the workbench onto a different section, region by region.

        The caller's `active_section` feeds exactly three things here: which
        centre regions exist at all (`hidden`), the tab-adjusted collapse
        state (`layout`), and the content of the region whose pane is routed
        by the section (passed in `rebuild_regions`). Everything else --
        both rails above all -- is left standing, which is the whole point:
        rebuilding the left rail means re-running `WatchlistTree.compose`,
        and that runs one synchronous source-row query per expanded
        watchlist.

        Args:
            hidden: Centre regions that must have no DOM presence on the new
                section. Newly hidden regions are unmounted; newly shown ones
                are mounted back into their `CENTRE_REGIONS` position.
            layout: The tab-adjusted layout (see
                `WatchlistsCollectionsScreen._rendered_region_layout`).
                Applied without firing `watch_region_layout`, so the
                hidden-set change and the layout change are reconciled in
                one pass rather than two.
            rebuild_regions: Regions whose supplied content must be rebuilt
                even though their form did not change -- the section's own
                pane lives in one of these.
            rebuild_header: Whether to rebuild the centre header too (the
                tab strip lives there, so a real section switch always does).
        """
        if not self.is_mounted:
            self._hidden = frozenset(hidden)
            self.set_reactive(WatchlistsWorkbench.region_layout, layout)
            return
        previous_layout = self.region_layout
        previous_hidden = self._hidden
        self._hidden = frozenset(hidden)
        # `set_reactive`, not assignment: `watch_region_layout` would run a
        # SECOND reconcile pass against a `_hidden` this method has already
        # moved, so the two changes would be applied in the wrong order and
        # the layout half would be done twice.
        self.set_reactive(WatchlistsWorkbench.region_layout, layout)
        # One layout/paint pass for the whole section move (task-15778),
        # as an explicit contract rather than a scheduling accident. The
        # region sync, the section pane rebuild and the header rebuild
        # below are each an awaited remove/mount cycle, and every await
        # between them is in principle a window for the screen's update
        # timer to paint a half-moved workbench. Measured at HEAD it never
        # actually does -- the whole sequence runs inside the screen's one
        # `_drain_surface_refresh` `call_next` callback (task-15461's own
        # move off `run_worker`), so the pump never goes idle mid-swap and
        # the paused update timer never resumes: 0 in-swap layout passes
        # and 0 compositor refreshes with and without this batch, on a
        # cold Read switch. `batch_update` makes that one-pass property
        # structural: it survives a future factory that awaits, or the
        # drain moving off a single callback, instead of depending on
        # them never happening. It defers repaints only -- it does not
        # reorder or coalesce the DOM work itself, so the raising-factory
        # guarantees inside `refresh_region_content`/`_swap_region_widget`
        # are unchanged.
        with self.app.batch_update():
            await self._sync_regions(previous_layout)
            for region in rebuild_regions:
                if region in self._hidden:
                    continue
                # A region the sync above just swapped is already built from
                # the current factory; rebuilding it again would be the
                # second of the two rebuilds this task exists to remove.
                if self._region_form_changed(
                    previous_layout, previous_hidden, region
                ):
                    continue
                await self.refresh_region_content(region)
            if rebuild_header:
                await self.refresh_header_content()

    def _region_form_changed(
        self,
        previous_layout: RegionLayout,
        previous_hidden: frozenset[Region],
        region: Region,
    ) -> bool:
        """Whether ``region``'s mounted widget has to be replaced outright."""
        was_present = region not in previous_hidden
        is_present = region not in self._hidden
        if was_present != is_present:
            return True
        if not is_present:
            return False
        return previous_layout.is_collapsed(region) != self.region_layout.is_collapsed(
            region
        )

    async def _sync_regions(self, previous_layout: RegionLayout) -> None:
        """Bring the mounted regions in line with `region_layout`/`_hidden`.

        Rails first (they are direct children of this `Horizontal` and are
        never hidden), then the centre stack, which is rebuilt positionally
        because a region can be absent from it entirely.
        """
        for region in (Region.LEFT_RAIL, Region.RIGHT_RAIL):
            if previous_layout.is_collapsed(region) == self.region_layout.is_collapsed(
                region
            ):
                continue
            index = 0 if region is Region.LEFT_RAIL else len(self.children) - 1
            await self._swap_region_widget(region, self, index)
        await self._sync_centre_regions(previous_layout)

    async def _sync_centre_regions(self, previous_layout: RegionLayout) -> None:
        """Mount, unmount, swap or repaint each centre region as required.

        Reads the CURRENT `self._hidden` rather than needing the previous one:
        a region that is newly shown simply has no mounted node, and a region
        that is newly hidden is removed by the first loop below -- both cases
        fall out of comparing the DOM against the desired set.
        """
        try:
            centre = self.query_one("#wl-centre")
        except NoMatches:
            return
        for region in CENTRE_REGIONS:
            if region not in self._hidden:
                continue
            node = self._mounted_region_node(region)
            if node is not None:
                await node.remove()

        desired = [region for region in CENTRE_REGIONS if region not in self._hidden]
        for position, region in enumerate(desired):
            index = self._centre_content_offset(centre) + position
            node = self._mounted_region_node(region)
            if node is None:
                await centre.mount(self._region_widget(region), before=index)
                continue
            if previous_layout.is_collapsed(region) != self.region_layout.is_collapsed(
                region
            ):
                await self._swap_region_widget(region, centre, index)
                continue
            if self.region_layout.is_collapsed(region):
                continue
            # Still an expanded body, same instance: the only thing a layout
            # change can have moved for it is the solo marker.
            node.set_class(
                self._is_sole_expanded_centre_region(region, self.region_layout),
                "watchlists-region-sole-centre",
            )

    def _centre_content_offset(self, centre: Widget) -> int:
        """How many non-region children the centre stack starts with.

        The optional `header` factory's widget is mounted first and carries
        whatever id the caller gave it, so it is identified by exclusion
        rather than by name.
        """
        region_ids = {f"wl-region-{r.value}" for r in REGION_ORDER}
        region_ids |= {f"wl-header-{r.value}" for r in REGION_ORDER}
        offset = 0
        for child in centre.children:
            if (child.id or "") in region_ids:
                break
            offset += 1
        return offset

    def _mounted_region_node(self, region: Region) -> Widget | None:
        """The widget currently rendering ``region``, in either form."""
        for prefix in ("wl-region-", "wl-header-"):
            try:
                return self.query_one(f"#{prefix}{region.value}")
            except NoMatches:
                continue
        return None

    async def _swap_region_widget(
        self, region: Region, parent: Widget, index: int
    ) -> None:
        """Replace ``region``'s mounted widget with a freshly built one.

        Built before the old one is detached, for the reason
        `refresh_region_content` states: a factory that raises must leave
        what is on screen standing rather than emptying the slot. The
        remove-then-mount pair still has an await between its halves --
        `NodeList._ensure_unique_id` refuses to mount a second
        `#wl-region-<r>`.
        """
        node = self._mounted_region_node(region)
        if node is None:
            return
        replacement = self._region_widget(region)
        await node.remove()
        if index >= len(parent.children):
            await parent.mount(replacement)
        else:
            await parent.mount(replacement, before=index)

    async def refresh_region_content(self, region: Region) -> None:
        """Rebuild one expanded region's supplied content in place.

        Task 7: a region's content can go stale without `region_layout`
        itself changing (the tree scope moving under the rail, a background
        load landing), so nothing would otherwise call its factory again.
        Pushing a new `region_layout` would not work at all: since task-15461
        that only swaps regions whose COLLAPSE state moved, so a layout equal
        to the current one is a no-op and one that differs changes what the
        user is looking at. (Before task-15461 it worked for the wrong
        reason -- it recomposed every region -- at the cost of replacing
        widgets whose whole design point is staying the same instance across
        an unrelated change: the Inspector is pushed new
        `scope`/`selected_entity` values in place for exactly that reason,
        see `WatchlistsCollectionsScreen.watch_selected_scope`.)

        A no-op when `region` is collapsed (nothing mounted to replace) or
        was not given a content factory (nothing to refresh either).

        Replaces only the *supplied content*, never the generic
        `REGION_TITLES` heading `_region_widget` prepends for regions
        outside `SELF_HEADED_REGIONS` (fix round 1, Finding 3). The first
        version removed every child, so refreshing LEFT_RAIL -- which
        supplies content but is not self-headed -- stripped its
        "Watchlists" heading and left an unlabelled bordered rail until the
        next region toggle rebuilt it. That is the same defect
        `SELF_HEADED_REGIONS`' own comment records having shipped once.

        Args:
            region: The region whose supplied content should be rebuilt.
        """
        if self.region_layout.is_collapsed(region):
            return
        factory = self._content.get(region)
        if factory is None:
            return
        try:
            container = self.query_one(f"#wl-region-{region.value}")
        except NoMatches:
            return
        # Build the replacement BEFORE detaching anything: a factory that
        # raises (or a worker cancelled while it runs) then leaves the
        # mounted pane standing rather than a bordered empty box. The
        # remove-then-mount pair below still has one await boundary --
        # Textual's `NodeList._ensure_unique_id` rejects mounting the new
        # pane while the old one (same id, e.g. `watchlists-items-pane`) is
        # still attached, so there is no single-await atomic swap available
        # without changing that guarded id.
        replacement = factory()
        # The heading, when present, is `_region_widget`'s first child and
        # is not ours to replace.
        stale = [
            child
            for child in container.children
            if not child.has_class("watchlists-region-title")
        ]
        for child in stale:
            await child.remove()
        await container.mount(replacement)

    async def refresh_header_content(self) -> None:
        """Rebuild the header in place from a fresh call to its factory.

        The header's twin of `refresh_region_content` above (task-1344 fix
        wave, Qodo correctness), and reached the same way: nothing else calls
        the `header` factory again, so a header-only change (the tree scope
        moving -- see `WatchlistsCollectionsScreen.watch_tree_scope`) has no
        other route onto the screen. Pushing a new `region_layout` is not one
        either: since task-15461 that touches only the regions whose collapse
        state moved, and never the header. The header is the only surface
        carrying the tab strip and the snapshot's scoped markers (since
        task-2513, on every tab), so this is what keeps that readout current
        between rebuilds.

        A no-op when this workbench was built with no `header` factory:
        nothing to refresh, and no `#wl-centre-status` to query for either.
        """
        if self._header is None:
            return
        try:
            centre = self.query_one("#wl-centre")
        except NoMatches:
            return
        try:
            stale = self.query_one("#wl-centre-status")
        except NoMatches:
            stale = None
        # Build the replacement before detaching the old header, for the
        # identical reason `refresh_region_content` does: a factory that
        # raises must leave the previously mounted header standing rather
        # than removing it and never replacing it.
        replacement = self._header()
        if stale is not None:
            await stale.remove()
        await centre.mount(replacement, before=0)

    def set_collapsed_suffixes(self, suffixes: Mapping[Region, str]) -> None:
        """Update collapsed-header suffixes in place (no recompose).

        Counts refresh while the rail stays collapsed; tearing the workbench
        down for a number is exactly what `refresh_region_content` exists to
        avoid for bodies. A no-op for regions not currently collapsed — they
        have no header mounted to repaint.

        Args:
            suffixes: The new suffix per region, replacing the current map.
        """
        self._collapsed_suffixes = dict(suffixes)
        for region, suffix in self._collapsed_suffixes.items():
            if not self.region_layout.is_collapsed(region):
                continue
            try:
                header = self.query_one(f"#wl-header-{region.value}", Button)
            except NoMatches:
                continue
            header.label = f"▸ {REGION_TITLES[region]}" + (f"  {suffix}" if suffix else "")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Turn a collapsed-region header click into a `RegionToggled` message.

        Ignores presses from any other button (e.g. content the caller
        supplied via `content=`) by checking the `wl-header-` id prefix.

        Args:
            event: The button-press event to inspect and, if it targets a
                region header, stop from bubbling further.
        """
        button_id = event.button.id or ""
        prefix = "wl-header-"
        if not button_id.startswith(prefix):
            return
        event.stop()
        self.post_message(RegionToggled(Region(button_id[len(prefix):])))
