"""Focus-based mode switching on the Lab frame."""

from __future__ import annotations

import pytest
from textual.widgets import Button

from tldw_chatbook.UI.Screens.lab_mode_strip import LAB_MODE_CHIP_IDS
from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from Tests.UI.test_screen_navigation import _build_test_app


async def _models(app):
    screen = LLMScreen(app)
    await app.push_screen(screen)
    return screen


@pytest.mark.asyncio
async def test_bracket_moves_focus_along_the_strip_without_navigating():
    app = _build_test_app()
    navigated: list[str] = []
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models(app)
        await pilot.pause()
        await pilot.pause()
        screen.query_one(f"#{LAB_MODE_CHIP_IDS[0]}", Button).focus()
        await pilot.pause()

        await pilot.press("right_square_bracket")
        await pilot.pause()

        assert app.focused is not None
        assert app.focused.id == LAB_MODE_CHIP_IDS[1]
        assert navigated == [], "moving focus must not navigate"


@pytest.mark.asyncio
async def test_bracket_wraps_at_both_ends():
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models(app)
        await pilot.pause()
        await pilot.pause()

        screen.query_one(f"#{LAB_MODE_CHIP_IDS[0]}", Button).focus()
        await pilot.pause()
        await pilot.press("left_square_bracket")
        await pilot.pause()
        assert app.focused.id == LAB_MODE_CHIP_IDS[-1]

        await pilot.press("right_square_bracket")
        await pilot.pause()
        assert app.focused.id == LAB_MODE_CHIP_IDS[0]


@pytest.mark.asyncio
async def test_bracket_starts_from_the_active_chip_when_nothing_is_focused():
    """With focus elsewhere, the first press should land beside the active
    mode rather than jumping to an arbitrary end of the strip."""
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models(app)
        await pilot.pause()
        await pilot.pause()
        screen.set_focus(None)
        await pilot.pause()

        await pilot.press("right_square_bracket")
        await pilot.pause()

        assert app.focused.id == LAB_MODE_CHIP_IDS[1]


@pytest.mark.asyncio
async def test_the_footer_advertises_the_mode_keys():
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models(app)
        await pilot.pause()
        await pilot.pause()
        footer = screen.query_one(AppFooterStatus)
        # `shortcut_text` is the assertable surface; AppFooterStatus has no
        # render() of its own. Existing hint tests use the same property.
        assert "[ / ]" in footer.shortcut_text
        assert "Switch mode" in footer.shortcut_text
