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
#: Phase D wires a real reader pane into CONTENT; whoever does that must add
#: `Region.CONTENT` here *if and only if* that pane draws its own heading.
SELF_HEADED_REGIONS: frozenset[Region] = frozenset(
    {Region.FEEDS, Region.ITEMS, Region.RIGHT_RAIL}
)

#: Placeholder body copy. Phase C and D replace these with real panes.
REGION_PLACEHOLDERS: dict[Region, str] = {
    Region.LEFT_RAIL: "Watchlist tree arrives in the next slice.",
    Region.FEEDS: "Feeds table arrives in the next slice.",
    Region.ITEMS: "Items table arrives in the next slice.",
    Region.CONTENT: "Reader arrives in the next slice.",
    Region.RIGHT_RAIL: "Inspector arrives in the next slice.",
}


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
        """
        super().__init__(**kwargs)
        self.add_class("watchlists-workbench")
        self._content: dict[Region, Callable[[], Widget]] = dict(content or {})
        self.set_reactive(WatchlistsWorkbench.region_layout, layout)

    def compose(self) -> ComposeResult:
        """Render the left rail, the stacked centre, and the right rail.

        Re-runs in full on every `region_layout` change (`recompose=True`),
        rebuilding all five regions from `self.region_layout` and
        `self._content` regardless of which single region actually changed.

        Returns:
            The left-rail region, the centre `Vertical` of FEEDS/ITEMS/
            CONTENT, and the right-rail region, in that order.
        """
        yield self._region_widget(Region.LEFT_RAIL)

        with Vertical(id="wl-centre", classes="watchlists-centre"):
            for region in CENTRE_REGIONS:
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
        if supplied is None:
            children.append(
                Static(
                    REGION_PLACEHOLDERS[region], classes="watchlists-region-placeholder"
                )
            )
        else:
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

        True exactly when the centre stack shows one real pane and two
        one-line headers — the state `RegionLayout.solo` produces, and the
        state a user reaches by collapsing the other two by hand.

        Args:
            region: The region to test. Rails always answer `False`; solo
                applies to the centre stack only (`RegionLayout.solo`).

        Returns:
            `True` if `region` is a centre region and every other centre
            region is collapsed, `False` otherwise.
        """
        if region not in CENTRE_REGIONS:
            return False
        expanded = [r for r in CENTRE_REGIONS if not self.region_layout.is_collapsed(r)]
        return expanded == [region]

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
