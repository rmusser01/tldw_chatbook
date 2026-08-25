"""TASK-22211 screen-level probe: resize oscillation must not churn panes.

`WatchlistsCollectionsScreen.on_resize` re-derives the effective layout per
Textual Resize event. Without hysteresis, a +/-1-cell width oscillation at a
collapse boundary re-runs a region factory and mounts/removes the whole pane
body on every crossing (`watchlists_workbench.py`'s
`_apply_effective_layout_request`). This probe counts region-body builds
across such an oscillation at the management-mode RIGHT_RAIL boundary
(108 = 44 centre + 2*5 grips + 24 + 30), driving the real `on_resize` ->
`_recompute_effective_layout` -> `request_region_layout` -> mount/remove
pipeline with only the width *measurement* stubbed. The screen opens on its
default Read section (`active_section = "items"`), whose all-open
requirement is 145 = 44 centre + 3*5 grips + 24 + 32 + 30; RIGHT_RAIL is
the first collapse candidate below it.

The conftest `isolate_test_environment` fixture patches
`load_region_layout` to `RegionLayout()` (nothing collapsed), so the
preferred layout here is all-open and every collapse below is responsive.
"""

from __future__ import annotations

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import DestinationHarness
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)
from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region
from tldw_chatbook.UI.Watchlists_Modules.watchlists_workbench import (
    WatchlistsWorkbench,
)

pytestmark = pytest.mark.asyncio

#: Read mode all-open requirement: RIGHT_RAIL collapses below this.
READ_BOUNDARY_WIDTH = 145


class _RegionBuildCounter:
    """Count pane-body factory construction (same idiom as the cold-open
    suite's counter: every mount of a region body passes through
    `WatchlistsWorkbench._region_body`)."""

    def __init__(self) -> None:
        self.regions: list[Region] = []
        self._original = WatchlistsWorkbench._region_body

    def __enter__(self) -> "_RegionBuildCounter":
        counter = self
        original = self._original

        def _counting_build(widget_self, region, content=None):
            counter.regions.append(region)
            return original(widget_self, region, content)

        WatchlistsWorkbench._region_body = _counting_build
        return self

    def __exit__(self, *exc_info) -> None:
        WatchlistsWorkbench._region_body = self._original


async def _resize_to(screen, pilot, width_box, width: int) -> None:
    width_box["value"] = width
    screen.on_resize(None)
    await pilot.pause(0.05)


async def test_one_cell_resize_oscillation_at_the_boundary_causes_no_churn():
    """+/-1-cell oscillation at the RIGHT_RAIL boundary: after the first
    legitimate collapse, zero further factory builds and zero
    mount/remove churn -- and a width that genuinely clears the boundary
    by the hysteresis margin still re-expands the pane."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        assert isinstance(screen, WatchlistsCollectionsScreen)
        await host.workers.wait_for_complete()
        await pilot.pause()

        workbench = screen.query_one(WatchlistsWorkbench)
        assert workbench.read_mode is True, (
            "probe assumes the default Read section (items)"
        )
        assert workbench._mounted_region_body(Region.RIGHT_RAIL) is not None, (
            "all-open preferred layout at 180 columns must mount RIGHT_RAIL"
        )

        width_box = {"value": READ_BOUNDARY_WIDTH}
        screen._available_layout_width = lambda: width_box["value"]

        # Settle exactly at the boundary: everything still fits at 145.
        await _resize_to(screen, pilot, width_box, READ_BOUNDARY_WIDTH)
        assert workbench._mounted_region_body(Region.RIGHT_RAIL) is not None

        with _RegionBuildCounter() as builds:
            for _ in range(5):
                await _resize_to(
                    screen, pilot, width_box, READ_BOUNDARY_WIDTH - 1
                )
                await _resize_to(
                    screen, pilot, width_box, READ_BOUNDARY_WIDTH
                )

            # The first 144 collapses the rail; every later +/-1 step must
            # be absorbed: NO region body is ever rebuilt during the
            # oscillation, and the rail stays collapsed (no re-mount at
            # 145, which is inside the hysteresis band).
            assert builds.regions == [], (
                "a +/-1-cell oscillation must cause zero region-body "
                f"rebuilds; got {builds.regions!r}"
            )
            assert workbench._mounted_region_body(Region.RIGHT_RAIL) is None, (
                "RIGHT_RAIL must stay collapsed while the width oscillates "
                "inside the hysteresis band"
            )

        # Convergence: clearing the boundary by the hysteresis width is a
        # real expand and must still work (hysteresis never sticks a pane).
        with _RegionBuildCounter() as reopen_builds:
            await _resize_to(
                screen, pilot, width_box, READ_BOUNDARY_WIDTH + 4
            )
            assert (
                workbench._mounted_region_body(Region.RIGHT_RAIL) is not None
            ), "clearing the boundary by the hysteresis width must re-expand"
            assert reopen_builds.regions == [Region.RIGHT_RAIL]
