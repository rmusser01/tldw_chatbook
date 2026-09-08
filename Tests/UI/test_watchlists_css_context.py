"""Full-app Watchlists fixtures must honor the real route's CSS boundary."""

from pathlib import Path

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.full_app_destination_context import FullAppDestinationContext
from Tests.UI.test_watchlists_destination_shell import (
    DestinationHarness,
    WatchlistsContextHarness,
)
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)
from tldw_chatbook.css import build_css


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("host_type", "size"),
    (
        (FullAppDestinationContext, (160, 45)),
        (DestinationHarness, (235, 52)),
        (WatchlistsContextHarness, (80, 24)),
    ),
)
async def test_full_app_context_has_watchlists_css_before_mount(
    monkeypatch, host_type, size
):
    app = _build_test_app()
    sheet = str(Path(build_css.__file__).parent / "screen_feature_watchlists.tcss")
    observed = []
    original_mount = WatchlistsCollectionsScreen.on_mount

    def on_mount(screen):
        observed.append(screen.app.stylesheet.has_source(sheet, ""))
        return original_mount(screen)

    monkeypatch.setattr(WatchlistsCollectionsScreen, "on_mount", on_mount)
    if host_type is WatchlistsContextHarness:
        host = host_type(WatchlistsCollectionsScreen(app))
    else:
        host = host_type(app, "watchlists_collections")
    async with host.run_test(size=size) as pilot:
        await pilot.pause()
        assert observed == [True]
        screen = host.context_screen
        # The initial Read action is the compact layout's primary control;
        # also exercise the later Overview tab at both wider supported sizes.
        section, label = (
            ("items", "Read") if size[0] == 80 else ("overview", "Overview")
        )
        button = screen.query_one(f"#wl-tab-{section}")
        button.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        x = button.region.x + button.region.width // 2
        y = button.region.y + button.region.height // 2
        hit, _ = screen._compositor.get_widget_at(x, y)
        assert hit is button
        painted = "\n".join(strip.text for strip in screen._compositor.render_strips())
        assert label in painted
