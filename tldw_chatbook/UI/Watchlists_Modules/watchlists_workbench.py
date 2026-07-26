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
    """

    region_layout: reactive[RegionLayout] = reactive(RegionLayout(), recompose=True)

    def __init__(
        self,
        layout: RegionLayout,
        content: Mapping[Region, Callable[[], Widget]] | None = None,
        **kwargs: Any,
    ) -> None:
        """
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
                ``WatchlistsNavigator`` (an overridden-``compose()``
                widget) already behaves.
        """
        super().__init__(**kwargs)
        self.add_class("watchlists-workbench")
        self._content: dict[Region, Callable[[], Widget]] = dict(content or {})
        self.set_reactive(WatchlistsWorkbench.region_layout, layout)

    def compose(self) -> ComposeResult:
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
        body = Vertical(
            Static(REGION_TITLES[region], classes="watchlists-region-title"),
            supplied
            if supplied is not None
            else Static(
                REGION_PLACEHOLDERS[region], classes="watchlists-region-placeholder"
            ),
            id=f"wl-region-{region.value}",
            classes=f"watchlists-region watchlists-region-{region.value}",
        )
        # Regions must be keyboard-reachable, or `z` cannot target them.
        body.can_focus = True
        return body

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        prefix = "wl-header-"
        if not button_id.startswith(prefix):
            return
        event.stop()
        self.post_message(RegionToggled(Region(button_id[len(prefix):])))
