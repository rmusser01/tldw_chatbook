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
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Static

from .region_layout import CENTRE_REGIONS, Region, RegionLayout


#: Human-readable titles, used for both expanded bodies and collapsed headers.
REGION_TITLES: dict[Region, str] = {
    Region.LEFT_RAIL: "Watchlists",
    Region.FEEDS: "Feeds",
    Region.ITEMS: "Items",
    Region.CONTENT: "Content",
    Region.RIGHT_RAIL: "Inspector",
}

#: Regions whose supplied pane draws its own heading, so adding the generic
#: region title from `REGION_TITLES` would produce a *second* heading (and,
#: for FEEDS -- whose pane is the "Sources" list -- an inaccurate one).
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
    {Region.FEEDS, Region.ITEMS, Region.RIGHT_RAIL}
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
        region_layout: The current collapse/solo state. Setting it
            (`recompose=True`) unmounts and rebuilds every region, not just
            the one that changed — see `__init__`'s note on `content` for
            why that requires factories rather than widget instances.
    """

    region_layout: reactive[RegionLayout] = reactive(RegionLayout(), recompose=True)

    def __init__(
        self,
        layout: RegionLayout,
        content: Mapping[Region, Callable[[], Widget]] | None = None,
        hidden: frozenset[Region] = frozenset(),
        header: Callable[[], Widget] | None = None,
        **kwargs: Any,
    ) -> None:
        """Build the workbench, seeding `region_layout` without triggering a recompose.

        Args:
            layout: Initial collapse/solo state.
            content: Per-region **factories**, not widget instances —
                ``region_layout`` is ``recompose=True``, so *any* collapse/
                solo/rail toggle fully unmounts and rebuilds every region,
                not just the one that changed (that blast radius is
                inherited from the reactive design, not introduced here).
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
                the workbench itself has no opinion about tabs. Constructor-
                only, not a reactive: the caller fully reconstructs this
                widget (via `compose_content`'s own recompose) whenever
                `active_section` changes, so nothing here needs to react to
                that change directly. A plain toggle/solo (`_apply_layout`
                pushing a new `region_layout` onto the ALREADY-mounted
                instance) never changes which tab is active, so a stale
                `hidden` is never observable.
            header: An optional factory for a widget rendered as the FIRST
                child of the centre stack, unconditionally — regardless of
                `hidden`. TASK-1344: the section tab strip and the
                snapshot's own loading/error/empty markers are cross-
                cutting chrome, not FEEDS content, so they must survive
                FEEDS being hidden on every non-Read tab. TASK-2312: the
                current screen caller (`WatchlistsCollectionsScreen.
                compose_content`) now passes this on EVERY tab, including
                Read — an earlier version passed `None` there in favour of
                an inline copy in `content[Region.FEEDS]`'s own factory,
                which visibly moved the tab strip's screen position
                between sections (UAT F2/F22/F23) and is why `None` is
                still accepted here: this class stays a generic building
                block with no opinion about tabs, and any two factories
                that both mount an id must never be combined by a caller,
                same as ever.
        """
        super().__init__(**kwargs)
        self.add_class("watchlists-workbench")
        self._content: dict[Region, Callable[[], Widget]] = dict(content or {})
        self._hidden = frozenset(hidden)
        self._header = header
        self.set_reactive(WatchlistsWorkbench.region_layout, layout)

    def compose(self) -> ComposeResult:
        """Render the left rail, the stacked centre, and the right rail.

        Re-runs in full on every `region_layout` change (`recompose=True`),
        rebuilding all five regions from `self.region_layout` and
        `self._content` regardless of which single region actually changed.

        Returns:
            The left-rail region, the centre `Vertical` (an optional header,
            then FEEDS/ITEMS/CONTENT minus anything in `self._hidden`), and
            the right-rail region, in that order.
        """
        yield self._region_widget(Region.LEFT_RAIL)

        with Vertical(id="wl-centre", classes="watchlists-centre"):
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
            header = Button(
                f"▸ {REGION_TITLES[region]}",
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
        # They looked identical while FEEDS/ITEMS/RIGHT_RAIL were the only
        # wired regions, which is how LEFT_RAIL — wired, but with a heading-
        # less `WatchlistTree` inside — ended up as an unlabelled box.
        #
        # FEEDS's pane needs a companion CSS fix for its own title removal
        # (see the `#watchlists-list-pane` rule in `_watchlists.tcss`): FEEDS
        # is the one region styled `height: auto`, so its supplied pane can
        # no longer lean on the (now-removed) title `Static` as a `height: 1`
        # sibling to anchor that auto-sizing.
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
        if self._is_sole_expanded_centre_region(region):
            # A CSS hook for the solo case. `.watchlists-region-feeds` carries
            # a `max-height` cap so a long feeds list cannot crowd ITEMS and
            # CONTENT out of the centre stack — but when those two are
            # collapsed to their one-line headers there is nothing left to
            # crowd, and the cap turns solo-FEEDS into a short scrolling
            # window with 25 blank rows under it. Nothing in the DOM
            # distinguished that state before this class: `RegionLayout.solo`
            # only collapses the *siblings*, so the soloed region itself is
            # indistinguishable from an ordinarily-expanded one.
            #
            # Keyed on "sole expanded centre region" rather than on
            # `solo_region` because the two produce the same DOM and want the
            # same layout: `Z` on FEEDS and manually collapsing ITEMS+CONTENT
            # with `z` both leave FEEDS alone in the centre.
            classes.append("watchlists-region-sole-centre")
        body = Vertical(
            *children,
            id=f"wl-region-{region.value}",
            classes=" ".join(classes),
        )
        # Regions must be keyboard-reachable, or `z` cannot target them.
        body.can_focus = True
        return body

    def _is_sole_expanded_centre_region(self, region: Region) -> bool:
        """Whether ``region`` is the only centre region still expanded.

        True exactly when the centre stack shows one real pane and, for
        every OTHER centre region that is not hidden outright, a one-line
        header — the state `RegionLayout.solo` produces, and the state a
        user reaches by collapsing the other two by hand. A region in
        `self._hidden` (TASK-1344: FEEDS/CONTENT off the Read tab) is never
        rendered at all, not even as a header, so it is excluded from
        "expanded" the same way a rail is — without this, ITEMS would never
        read as sole-expanded on a non-Read tab (FEEDS/CONTENT's real
        `region_layout.collapsed` membership is whatever the user left it
        at on Read, not "hidden", so counting it unfiltered would make
        `expanded` include a region that in fact never mounted).

        Args:
            region: The region to test. Rails always answer `False`; solo
                applies to the centre stack only (`RegionLayout.solo`).

        Returns:
            `True` if `region` is a centre region and every other
            non-hidden centre region is collapsed, `False` otherwise.
        """
        if region not in CENTRE_REGIONS:
            return False
        expanded = [
            r
            for r in CENTRE_REGIONS
            if r not in self._hidden and not self.region_layout.is_collapsed(r)
        ]
        return expanded == [region]

    async def refresh_region_content(self, region: Region) -> None:
        """Rebuild one expanded region's supplied content in place.

        Task 7: the tree scope can change what FEEDS should show without
        `region_layout` itself changing, so nothing would otherwise call
        FEEDS's factory again. Setting `region_layout` (`recompose=True`)
        would work too, but at the cost of tearing down and remounting
        *every* region, including ones whose whole design point is staying
        the same instance across an unrelated change -- the Inspector is
        pushed new `scope`/`selected_entity` values in place for exactly
        that reason (see `WatchlistsCollectionsScreen.watch_selected_scope`),
        and a full recompose would silently replace it with a fresh
        instance instead, breaking any caller holding a reference to the
        old one.

        A no-op when `region` is collapsed (nothing mounted to replace) or
        was not given a content factory (the placeholder stub has nothing
        to refresh either).

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
        # pane while the old one (same id, e.g. `watchlists-list-pane`) is
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
        wave, Qodo correctness): `region_layout` is `recompose=True`, so
        picking up a header-only change (the tree scope moving while FEEDS
        is hidden -- see `WatchlistsCollectionsScreen.watch_tree_scope`) by
        pushing a new layout would tear down and remount every region,
        including the Inspector, which `watch_tree_scope` deliberately
        avoids (see its own docstring). Before TASK-2312, the current
        screen caller wired `header=` only off the Read tab (exactly where
        FEEDS was also in `hidden`), so a header-only tab had no OTHER path
        that picked up a scope change; the header kept showing the
        PREVIOUS scope's summary until some unrelated recompose came along
        and rebuilt it for a different reason. That caller now wires
        `header=` on every tab, so this runs everywhere the header exists.

        A no-op when this workbench was built with no `header` factory at
        all (any caller may still pass `None`, per `__init__`'s own
        docstring): nothing to refresh, and no `#wl-centre-status` to
        query for either.
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
