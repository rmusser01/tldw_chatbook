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
