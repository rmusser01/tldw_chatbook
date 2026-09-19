"""Explicit action alignment must survive the shared dialog stylesheet cascade."""

import pytest
from textual.containers import Container
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Widgets.pattern_gallery import PatternGalleryScreen


def assert_painted(screen, button):
    region, clip = screen._compositor.visible_widgets[button]
    assert region.intersection(clip) == region
    content = button.content_region
    painted = "\n".join(
        strip.crop(content.x, content.right).text
        for strip in screen._compositor.render_strips()[content.y : content.bottom]
    )
    assert str(button.label) in painted


def assert_alignment(row, buttons, expected):
    first, last = buttons
    assert first.region.y == last.region.y
    assert first.region.right < last.region.x
    left = first.region.x - first.styles.margin.left - row.content_region.x
    right = row.content_region.right - last.region.right - last.styles.margin.right
    assert left >= 0 and right >= 0
    if expected == "left":
        assert left == 0 and right > 0
    elif expected == "right":
        assert right == 0 and left > 0
    else:
        assert abs(left - right) <= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("alignment", [None, "left", "center", "right"])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(80, 24), (170, 48)])
@private_profile_test
async def test_explicit_dialog_alignment_wins_without_moving_default_rows(
    request, alignment, theme, size
):
    app = ConsolidatedCSSApp(css_path=list(APP_STYLESHEETS))
    app.theme = theme
    async with app.run_test(size=size) as pilot:
        classes = "dialog-buttons button-group"
        if alignment:
            classes += f" button-group-{alignment}"
        row = Container(Button("Cancel"), Button("Save"), classes=classes)
        await app.screen.mount(row)
        await pilot.pause()
        buttons = list(row.query(Button))
        assert_alignment(row, buttons, alignment or "center")
        for button in buttons:
            assert_painted(app.screen, button)
        buttons[0].focus()
        await pilot.press("tab")
        await pilot.pause()
        assert app.focused is buttons[1]
        assert_alignment(row, buttons, alignment or "center")
        assert_painted(app.screen, buttons[1])


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(80, 24), (170, 48)])
@private_profile_test
async def test_gallery_dialog_actions_use_the_documented_trailing_edge(
    request, theme, size
):
    app = ConsolidatedCSSApp(css_path=list(APP_STYLESHEETS))
    app.theme = theme
    async with app.run_test(size=size) as pilot:
        host = app.screen
        gallery = PatternGalleryScreen()
        app.push_screen(gallery)
        await pilot.pause()
        row = gallery.query_one(".pg-dialog-frame .dialog-buttons")
        buttons = list(row.query(Button))
        buttons[0].focus()
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        assert app.focused is buttons[1]
        assert_alignment(row, buttons, "right")
        for button in buttons:
            assert_painted(gallery, button)
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is host
