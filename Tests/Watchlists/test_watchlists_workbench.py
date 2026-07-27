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
async def test_feeds_height_is_capped_and_scrollable_when_content_overflows():
    """Fix round 1 (human-ruled): removing FEEDS's duplicate title (the test
    above) exposed a pre-existing bug where FEEDS's `height: auto` grew
    without any limit. That is fine for the empty state, but Task 7 gives
    FEEDS the real, scope-driven feeds list -- potentially dozens of
    sources -- which would push ITEMS/CONTENT clean off the bottom of the
    viewport. `_watchlists.tcss` now caps `.watchlists-region-feeds` at
    `max-height: 13` with `overflow-y: auto`: it grows to fit small lists,
    stops at the cap, and scrolls past it rather than either clipping
    silently or displacing its neighbours.

    (Fix round 2 raised the cap from round 1's `10` to `13` -- round 1's
    value sat one row BELOW the real empty state's own 11-row need, so even
    zero sources scrolled. See
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

        assert feeds.region.height <= 13, (
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
        strips = feeds.screen._compositor.render_strips()
        top_row_text = "".join(segment.text for segment in strips[feeds.region.y])
        assert "source-00" in top_row_text, (
            f"expected the first supplied row on screen initially: {top_row_text!r}"
        )

        feeds.scroll_end(animate=False)
        await pilot.pause()
        await pilot.pause()

        strips = feeds.screen._compositor.render_strips()
        bottom_y = feeds.region.y + feeds.region.height - 1
        bottom_row_text = "".join(segment.text for segment in strips[bottom_y])
        assert "source-39" in bottom_row_text, (
            f"the last supplied row should be reachable by scrolling, not "
            f"lost past the cap: {bottom_row_text!r}"
        )
