"""The palette keeps gallery imports deferred until the command executes."""

import sys

import pytest

from Tests.private_profile import private_profile_test


@pytest.mark.parametrize("query", [None, "Pattern Gallery"])
@pytest.mark.asyncio
@private_profile_test
async def test_gallery_palette_command_loads_and_opens_on_demand(query, request):
    from tldw_chatbook.app import PatternGalleryProvider, TldwCli

    from .consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp

    module = "tldw_chatbook.Widgets.pattern_gallery"
    assert PatternGalleryProvider in TldwCli.COMMANDS
    assert module not in sys.modules
    app = ConsolidatedCSSApp(css_path=BUNDLED_STYLESHEET)
    async with app.run_test(size=(120, 40)) as pilot:
        host = app.screen
        provider = PatternGalleryProvider(screen=host)
        stream = provider.discover() if query is None else provider.search(query)
        hits = [hit async for hit in stream]
        assert len(hits) == 1
        assert hits[0].text == "Design System: Pattern Gallery"
        assert module not in sys.modules

        hits[0].command()
        await pilot.pause()
        assert module in sys.modules
        assert isinstance(app.screen, sys.modules[module].PatternGalleryScreen)
        assert app.screen.query_one("#pg-root .section-title").region.height > 0

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is host
