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
