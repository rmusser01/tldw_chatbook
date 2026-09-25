import pytest
from textual.app import ComposeResult

from Tests.private_profile import private_profile_test
from Tests.textual_test_harness import IsolatedWidgetTestApp
from tldw_chatbook.Widgets.theme_preview import ThemePreview


def _app(*widgets):
    def compose() -> ComposeResult:
        yield from widgets

    return IsolatedWidgetTestApp(compose)


@pytest.mark.asyncio
@private_profile_test
async def test_theme_preview_paints_rows_from_colours(request):
    preview = ThemePreview("pv")
    async with _app(preview).run_test(size=(80, 20)) as pilot:
        preview.paint({"panel": "#112233", "foreground": "#EEEEEE", "accent": "#FF8800", "background": "#000000"})
        await pilot.pause()
        rail = preview.query_one("#pv-rail")
        assert rail.styles.background.hex.upper() == "#112233"
        assert "[ Send ]" in str(preview.query_one("#pv-accent").render())


@pytest.mark.asyncio
@private_profile_test
async def test_compact_preview_has_two_rows(request):
    preview = ThemePreview("pv", compact=True)
    async with _app(preview).run_test(size=(80, 20)):
        assert [w.id for w in preview.children] == ["pv-rail", "pv-accent"]
