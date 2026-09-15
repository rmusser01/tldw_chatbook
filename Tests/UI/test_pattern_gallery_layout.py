"""Canonical inline form examples must paint inside their available row."""

import pytest
from textual.widgets import Input, Label

from tldw_chatbook.Widgets.pattern_gallery import PatternGalleryScreen

from .test_pattern_gallery_snapshots import _GalleryApp


@pytest.mark.parametrize("width", [80, 120])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("focused", [False, True], ids=["rest", "focus"])
@pytest.mark.asyncio
async def test_temperature_field_and_edges_stay_inside_the_inline_row(
    width: int, theme: str, focused: bool
) -> None:
    """A full-row input beside its label must not clip its value or border."""
    app = _GalleryApp(theme)
    async with app.run_test(size=(width, 60)) as pilot:
        await app.push_screen(PatternGalleryScreen())
        temperature = next(
            field for field in app.screen.query(Input) if field.placeholder == "0.7"
        )
        temperature.value = "0.7"
        app.screen.set_focus(temperature if focused else None)
        await pilot.pause()

        row = temperature.parent
        assert row is not None
        label = row.query_one(Label)
        assert row.content_region.contains_region(label.region)
        assert row.content_region.contains_region(temperature.region), (
            f"Temperature input {temperature.region} overflows row "
            f"{row.content_region} at {width} columns ({theme}, focused={focused})"
        )
        assert label.region.right <= temperature.region.x
        assert temperature.has_focus is focused
        assert temperature.value == "0.7"

        strips = app.screen._compositor.render_strips()
        region = temperature.region
        painted = [
            strip.crop(region.x, region.right).text
            for strip in strips[region.y : region.bottom]
        ]
        assert len(painted) >= 3
        assert painted[0].strip() and painted[-1].strip()
        assert all(line[0].strip() and line[-1].strip() for line in painted[1:-1])
        assert "0.7" in "\n".join(painted[1:-1])
