import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region, RegionLayout
from tldw_chatbook.UI.Watchlists_Modules.watchlists_workbench import (
    RegionToggled,
    WatchlistsWorkbench,
)


class _WorkbenchApp(App):
    def __init__(self, layout: RegionLayout) -> None:
        super().__init__()
        self._layout = layout
        self.toggles: list[Region] = []

    def compose(self) -> ComposeResult:
        yield WatchlistsWorkbench(self._layout, id="wl-workbench")

    def on_region_toggled(self, message: RegionToggled) -> None:
        self.toggles.append(message.region)


@pytest.mark.asyncio
async def test_all_regions_render_expanded_by_default():
    app = _WorkbenchApp(RegionLayout())
    async with app.run_test():
        for region in Region:
            assert app.query(f"#wl-region-{region.value}")
            assert not app.query(f"#wl-header-{region.value}")


@pytest.mark.asyncio
async def test_collapsed_region_renders_a_header_instead_of_a_body():
    app = _WorkbenchApp(RegionLayout(collapsed=frozenset({Region.CONTENT})))
    async with app.run_test():
        assert app.query("#wl-header-content")
        assert not app.query("#wl-region-content")


@pytest.mark.asyncio
async def test_collapsed_header_is_focusable_so_collapse_is_not_one_way():
    app = _WorkbenchApp(RegionLayout(collapsed=frozenset({Region.ITEMS})))
    async with app.run_test():
        header = app.query_one("#wl-header-items")
        assert header.focusable, "a collapsed region must be reachable by keyboard"


@pytest.mark.asyncio
async def test_clicking_a_collapsed_header_posts_region_toggled():
    app = _WorkbenchApp(RegionLayout(collapsed=frozenset({Region.FEEDS})))
    async with app.run_test() as pilot:
        await pilot.click("#wl-header-feeds")
        await pilot.pause()
        assert app.toggles == [Region.FEEDS]


@pytest.mark.asyncio
async def test_updating_the_layout_reactive_re_renders():
    # NOTE: the reactive is exposed as `region_layout`, not `layout`. Widget
    # already defines a read-only `layout` property (the compositor's arrange
    # strategy; see textual/widget.py and textual/_arrange.py:97) that a
    # same-named reactive here would shadow, crashing every render with
    # AttributeError: '...' object has no attribute 'arrange'. Verified
    # empirically with a minimal Horizontal subclass before renaming.
    app = _WorkbenchApp(RegionLayout())
    async with app.run_test() as pilot:
        workbench = app.query_one(WatchlistsWorkbench)
        workbench.region_layout = RegionLayout(collapsed=frozenset({Region.LEFT_RAIL}))
        await pilot.pause()
        assert app.query("#wl-header-left_rail")
        assert not app.query("#wl-region-left_rail")


@pytest.mark.asyncio
async def test_every_centre_region_may_be_collapsed_at_once():
    app = _WorkbenchApp(RegionLayout(collapsed=frozenset(CENTRE := {Region.FEEDS, Region.ITEMS, Region.CONTENT})))
    async with app.run_test():
        for region in CENTRE:
            assert app.query(f"#wl-header-{region.value}")
        # The rails survive, so the screen is never empty.
        assert app.query("#wl-region-left_rail")
        assert app.query("#wl-region-right_rail")


@pytest.mark.asyncio
async def test_supplied_content_replaces_the_stub_placeholder():
    from textual.widgets import Label

    # `content` holds FACTORIES, not instances — see the empirical finding
    # documented on `WatchlistsWorkbench.__init__` and
    # `test_supplied_content_with_nested_children_survives_recompose` below.
    class _App(App):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(),
                content={Region.FEEDS: lambda: Label("real feeds table", id="my-real-feeds")},
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test():
        assert app.query("#my-real-feeds"), "supplied content should be mounted"
        # The stub placeholder for that region must be gone.
        placeholders = [
            str(node.renderable) for node in app.query(".watchlists-region-placeholder")
        ]
        assert not any("Feeds table arrives" in text for text in placeholders)


@pytest.mark.asyncio
async def test_regions_without_supplied_content_keep_their_stubs():
    class _App(App):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(RegionLayout(), content={}, id="wl-workbench")

    app = _App()
    async with app.run_test():
        assert app.query(".watchlists-region-placeholder")


@pytest.mark.asyncio
async def test_supplied_content_with_nested_children_survives_recompose():
    """Regression test for an empirically-found bug: passing an already-
    constructed CONTAINER widget (with its own constructor-supplied children,
    e.g. `Vertical(Static(...), Static(...))`) as `content` works on the
    FIRST render, but its grandchildren vanish after ANY subsequent
    recompose — `region_layout` is `recompose=True`, so toggling even an
    unrelated region unmounts and rebuilds every region. A widget's
    constructor-supplied children are only mounted on that instance's first
    mount; the same instance remounted a second time comes back childless.
    `content` must therefore hold factories that build a fresh instance on
    every call, not pre-built instances — this test would fail against the
    pre-fix, instance-based implementation.
    """
    from textual.containers import Vertical as _Vertical
    from textual.widgets import Static as _Static

    def build_nested():
        return _Vertical(
            _Static("outer-title", id="outer-title"),
            _Static("inner-content", id="inner-content"),
            id="my-nested-pane",
        )

    class _App(App):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(), content={Region.FEEDS: build_nested}, id="wl-workbench"
            )

    app = _App()
    async with app.run_test() as pilot:
        assert app.query("#inner-content")

        workbench = app.query_one(WatchlistsWorkbench)
        # Toggle an UNRELATED region. Because the reactive recomposes the
        # whole workbench, this rebuilds FEEDS too, even though FEEDS itself
        # was never toggled.
        workbench.region_layout = RegionLayout(collapsed=frozenset({Region.LEFT_RAIL}))
        await pilot.pause()

        assert app.query("#my-nested-pane"), "the region body should still be rebuilt"
        assert app.query(
            "#inner-content"
        ), "nested content must survive an unrelated region's recompose"


@pytest.mark.asyncio
async def test_a_region_with_supplied_content_does_not_double_title():
    from textual.widgets import Static

    def factory():
        return Static("Sources", classes="pane-title")

    class _App(App):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(), content={Region.FEEDS: factory}, id="wl-workbench"
            )

    app = _App()
    async with app.run_test():
        titles = [str(n.renderable) for n in app.query(".watchlists-region-title")]
        assert "Feeds" not in titles, (
            "a region whose content supplies its own heading should not add a second one"
        )


@pytest.mark.asyncio
async def test_left_rail_keeps_its_heading_when_its_content_supplies_none():
    """Fix round 3, Finding 1: title suppression was keyed on
    factory-presence, but the rule is "suppress it where the pane supplies
    its own heading". LEFT_RAIL is where those two signals diverge — it
    supplies content (`WatchlistTree`) that composes only navigation buttons
    and no heading widget — so the expanded rail rendered as an unlabelled
    bordered box while its collapsed header still read "▸ Watchlists".
    `SELF_HEADED_REGIONS` now carries the real rule.
    """
    from textual.widgets import Label

    class _App(App):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(),
                content={
                    # Stands in for `WatchlistTree`: real content, no heading.
                    Region.LEFT_RAIL: lambda: Label("All sources  0", id="headingless"),
                    Region.FEEDS: lambda: Label("Sources", id="self-headed"),
                },
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test():
        assert app.query("#headingless"), "supplied rail content should be mounted"
        titles = [str(n.renderable) for n in app.query(".watchlists-region-title")]
        assert "Watchlists" in titles, (
            "the left rail's content supplies no heading of its own, so the "
            f"region must still label it: {titles}"
        )
        assert "Feeds" not in titles, (
            "a region whose pane DOES supply its own heading must not add a "
            f"second one: {titles}"
        )


@pytest.mark.asyncio
async def test_a_soloed_centre_region_is_marked_for_css():
    """Fix round 3, Finding 3: `RegionLayout.solo` only collapses the soloed
    region's *siblings*, so nothing in the DOM distinguished a soloed region
    from an ordinarily-expanded one — and FEEDS, the one capped region, stayed
    pinned at its `max-height` with the rest of the centre blank. This class
    is the hook `.watchlists-region-sole-centre` keys off (see
    `_watchlists.tcss`); the geometry it produces is asserted against the real
    stylesheet in
    `Tests/UI/test_destination_visual_parity_correction.py::
    test_watchlists_soloed_feeds_fills_the_centre`.
    """
    app = _WorkbenchApp(RegionLayout().solo(Region.FEEDS))
    async with app.run_test():
        feeds = app.query_one("#wl-region-feeds")
        assert feeds.has_class("watchlists-region-sole-centre"), sorted(feeds.classes)
        # The rails are still expanded, and solo never applies to them.
        for rail in ("left_rail", "right_rail"):
            rail_region = app.query_one(f"#wl-region-{rail}")
            assert not rail_region.has_class("watchlists-region-sole-centre")

    # Reaching the same DOM by hand must get the same treatment: `z` on ITEMS
    # and CONTENT leaves FEEDS just as alone as `Z` on FEEDS does.
    manual = _WorkbenchApp(
        RegionLayout(collapsed=frozenset({Region.ITEMS, Region.CONTENT}))
    )
    async with manual.run_test():
        assert manual.query_one("#wl-region-feeds").has_class(
            "watchlists-region-sole-centre"
        )

    # ... and an ordinary expanded layout must NOT get it, or the cap never
    # applies at all.
    ordinary = _WorkbenchApp(RegionLayout())
    async with ordinary.run_test():
        for region in Region:
            assert not ordinary.query_one(f"#wl-region-{region.value}").has_class(
                "watchlists-region-sole-centre"
            )


@pytest.mark.asyncio
async def test_feeds_height_is_capped_and_scrollable_when_content_overflows():
    """Fix round 1 (human-ruled): removing FEEDS's duplicate title (the test
    above) exposed a pre-existing bug where FEEDS's `height: auto` grew
    without any limit. That is fine for the empty state, but Task 7 gives
    FEEDS the real, scope-driven feeds list -- potentially dozens of
    sources -- which would push ITEMS/CONTENT clean off the bottom of the
    viewport. `_watchlists.tcss` now caps `.watchlists-region-feeds` at
    `max-height: 12` with `overflow-y: auto`: it grows to fit small lists,
    stops at the cap, and scrolls past it rather than either clipping
    silently or displacing its neighbours.

    (Fix round 2 raised the cap from round 1's `10`, which sat one row
    BELOW the real empty state's own 11-row need, so even zero sources
    scrolled. The shipped `12` is one under the maximum that holds at
    160x42, chosen so the invariant also survives a 41-row terminal. See
    `Tests/UI/test_destination_visual_parity_correction.py::
    test_watchlists_feeds_empty_state_fits_without_scrolling` (which checks
    this against the real production screen, not synthetic content) and the
    derivation comment on `.watchlists-region-feeds` in `_watchlists.tcss`.)

    Unlike this file's other tests, this one loads the REAL production
    stylesheet (`CSS_PATH`) -- a bare `App` with no CSS at all would never
    exercise `max-height`/`overflow-y`, so the assertions below would be
    vacuous against the pre-fix code as much as the post-fix code.
    """
    from pathlib import Path

    from textual.containers import Vertical
    from textual.widgets import Static

    css_path = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )

    def overflowing_feeds() -> Vertical:
        # 40 rows: far more than any reasonable cap, standing in for
        # Task 7's real feeds list once a watchlist has dozens of sources.
        body = Vertical(
            *[Static(f"source-{i:02d}") for i in range(40)],
            id="feeds-overflow-probe",
        )
        # Mirrors the production companion fix on `#watchlists-list-pane`
        # in `_watchlists.tcss`: a bare `Vertical` defaults to
        # `height: 1fr`, which is circular inside FEEDS's `height: auto`
        # region -- it must size to its own content instead.
        body.styles.height = "auto"
        return body

    class _App(App):
        CSS_PATH = css_path

        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(),
                content={Region.FEEDS: overflowing_feeds},
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test(size=(100, 40)) as pilot:
        feeds = app.query_one("#wl-region-feeds")
        items = app.query_one("#wl-region-items")

        assert feeds.region.height <= 12, (
            f"FEEDS should stop growing at the cap once its content "
            f"overflows it: {feeds.region}"
        )
        assert items.region.height > feeds.region.height, (
            f"ITEMS must stay the taller reading area even when FEEDS's "
            f"content would otherwise dwarf it: items={items.region} "
            f"feeds={feeds.region}"
        )

        # Confirm it SCROLLS rather than clips: all 40 supplied rows must
        # be reachable, not silently cut off past the cap.
        def painted_rows() -> list[str]:
            strips = feeds.screen._compositor.render_strips()
            region = feeds.region
            return [
                "".join(segment.text for segment in strips[y])[
                    region.x : region.x + region.width
                ]
                for y in range(region.y, region.y + region.height)
            ]

        rows = painted_rows()
        assert any("source-00" in row for row in rows), (
            f"expected the first supplied row on screen initially: {rows!r}"
        )
        # Fix round 3, Finding 2: the region owns the border AND the scroll,
        # so the box must stay closed at both scroll extremes. When the pane
        # inside owned the border instead, the border rows were part of the
        # scrolled content — at scroll top the bottom edge was off-screen and
        # at scroll end the top edge was.
        assert rows[0].startswith("╭") and rows[0].endswith("╮"), rows[0]
        assert rows[-1].startswith("╰") and rows[-1].endswith("╯"), rows[-1]

        feeds.scroll_end(animate=False)
        await pilot.pause()
        await pilot.pause()

        rows = painted_rows()
        assert any("source-39" in row for row in rows), (
            f"the last supplied row should be reachable by scrolling, not "
            f"lost past the cap: {rows!r}"
        )
        assert rows[0].startswith("╭") and rows[0].endswith("╮"), (
            f"scrolling must not carry the region's own top border away: {rows[0]!r}"
        )
        assert rows[-1].startswith("╰") and rows[-1].endswith("╯"), (
            f"scrolling must not carry the region's own bottom border away: "
            f"{rows[-1]!r}"
        )


# --- Task 7: `refresh_region_content` ---------------------------------------
#
# `WatchlistsCollectionsScreen.watch_selected_scope` needs FEEDS to follow a
# tree-scope change without recomposing the whole workbench (a full
# recompose would also replace the Inspector, breaking its "same instance,
# updated in place" contract -- see that method's docstring). These pin the
# primitive it relies on, independent of the screen.


@pytest.mark.asyncio
async def test_refresh_region_content_rebuilds_only_the_named_region():
    from textual.widgets import Label

    calls = {"feeds": 0}

    def feeds_factory():
        calls["feeds"] += 1
        return Label(f"feeds-{calls['feeds']}", id="feeds-content")

    items_widget_ids: list[int] = []

    def items_factory():
        label = Label("items", id="items-content")
        items_widget_ids.append(id(label))
        return label

    class _App(App):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(),
                content={Region.FEEDS: feeds_factory, Region.ITEMS: items_factory},
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test() as pilot:
        workbench = app.query_one(WatchlistsWorkbench)
        first_feeds = app.query_one("#feeds-content", Label)
        first_items = app.query_one("#items-content", Label)
        assert str(first_feeds.renderable) == "feeds-1"

        await workbench.refresh_region_content(Region.FEEDS)
        await pilot.pause()

        refreshed_feeds = app.query_one("#feeds-content", Label)
        assert str(refreshed_feeds.renderable) == "feeds-2", (
            "the factory should run again, reflecting whatever changed"
        )
        assert refreshed_feeds is not first_feeds, (
            "the old content widget should be replaced, not mutated in place"
        )

        still_items = app.query_one("#items-content", Label)
        assert still_items is first_items, (
            "an unrelated region's content must not be touched"
        )
        assert calls["feeds"] == 2
        assert len(items_widget_ids) == 1, "ITEMS's factory must not run again"


@pytest.mark.asyncio
async def test_refresh_region_content_is_a_noop_when_the_region_is_collapsed():
    from textual.widgets import Label

    calls = {"feeds": 0}

    def feeds_factory():
        calls["feeds"] += 1
        return Label("feeds", id="feeds-content")

    class _App(App):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(collapsed=frozenset({Region.FEEDS})),
                content={Region.FEEDS: feeds_factory},
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test():
        workbench = app.query_one(WatchlistsWorkbench)
        assert calls["feeds"] == 0, "a collapsed region should not build its content at all"

        await workbench.refresh_region_content(Region.FEEDS)

        assert calls["feeds"] == 0, "refreshing a collapsed region must be a no-op"
