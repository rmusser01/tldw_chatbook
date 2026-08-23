"""TEMP probe (not committed): widget-recreation + recompose counts for a media row open."""
from unittest.mock import patch

import pytest
from textual.widget import Widget
from textual.widgets import Button

from Tests.UI.test_library_per_click_recompose_t21116 import (
    _boot_media_library,
    _media_app_host,
)
from Tests.UI.test_library_shell import LIBRARY_TEST_SIZE, _wait_for_selector
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen


@pytest.mark.asyncio
async def test_probe_media_open_recreation_counts() -> None:
    host = _media_app_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _boot_media_library(host, pilot)

        recomposes: list[object] = []
        created: list[str] = []
        original_refresh = BaseAppScreen.refresh
        original_init = Widget.__init__

        def refresh_spy(self, *args, **kwargs):
            if kwargs.get("recompose"):
                recomposes.append(self)
            return original_refresh(self, *args, **kwargs)

        def init_spy(self, *args, **kwargs):
            created.append(type(self).__name__)
            return original_init(self, *args, **kwargs)

        with patch.object(BaseAppScreen, "refresh", refresh_spy), patch.object(
            Widget, "__init__", init_spy
        ):
            screen.query_one("#library-media-row-0", Button).press()
            await _wait_for_selector(screen, pilot, "#library-media-content-search")
            for _ in range(10):
                await pilot.pause(0.02)
        print(
            f"PROBE media-open: whole_screen_recomposes={len(recomposes)} "
            f"widgets_created={len(created)}"
        )

        # Same probe for the back-to-list exit.
        recomposes.clear()
        created.clear()
        with patch.object(BaseAppScreen, "refresh", refresh_spy), patch.object(
            Widget, "__init__", init_spy
        ):
            screen.query_one("#library-media-back", Button).press()
            await _wait_for_selector(screen, pilot, "#library-media-row-0")
            for _ in range(10):
                await pilot.pause(0.02)
        print(
            f"PROBE viewer-back: whole_screen_recomposes={len(recomposes)} "
            f"widgets_created={len(created)}"
        )
