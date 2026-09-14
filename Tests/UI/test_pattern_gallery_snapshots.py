"""Gallery SVG snapshots (spec 3.4): fixed size, textual-dark + textual-light.

The harness mirrors the production cascade exactly (``ConsolidatedCSSApp``
with the app bundle as ``CSS_PATH``), so the fixtures pin the canonical
sheets' real output -- colors, borders, and geometry -- not Textual
defaults. Regenerate deliberately: ``UPDATE_SNAPSHOTS=1`` then review the
diff; spec 5.9 requires token-value changes to be live-verified.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from textual.widgets import Static

from tldw_chatbook.Widgets.pattern_gallery import PatternGalleryScreen

from .consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp

FIXTURES = Path(__file__).parent / "snapshots" / "pattern_gallery"

#: Includes the bounded utility samples added during the Python migration.
SIZE = (120, 360)


def _normalize(svg: str) -> str:
    """Strip run-varying SVG chrome (xml header, svg attrs, font names)."""
    svg = re.sub(r"<\?xml[^>]*\?>", "", svg)
    svg = re.sub(r"<svg[^>]*>", "<svg>", svg)
    svg = re.sub(r'font-family="[^"]*"', 'font-family="X"', svg)
    return svg.strip()


class _GalleryApp(ConsolidatedCSSApp):
    """Mount the gallery with the production stylesheet stack and a theme."""

    CSS_PATH = BUNDLED_STYLESHEET

    def __init__(self, theme: str) -> None:
        super().__init__()
        self._theme = theme

    def on_mount(self) -> None:
        self.theme = self._theme

    def compose(self):
        yield Static("host")


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.asyncio
async def test_gallery_snapshot(theme: str) -> None:
    app = _GalleryApp(theme)
    async with app.run_test(size=SIZE) as pilot:
        # The gallery must be the active SCREEN (a Screen yielded from
        # App.compose mounts as a zero-size child of the default screen).
        await app.push_screen(PatternGalleryScreen())
        await pilot.pause()
        svg = _normalize(app.export_screenshot(simplify=True))
    fixture = FIXTURES / f"{theme.removeprefix('textual-')}.svg"
    if os.environ.get("UPDATE_SNAPSHOTS"):
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(svg, encoding="utf-8")
    assert svg == fixture.read_text(encoding="utf-8").strip(), (
        f"Pattern rendering changed under {theme}. If intended: UPDATE_SNAPSHOTS=1 "
        "and review the diff; spec 5.9 requires token-value changes to be live-verified."
    )
