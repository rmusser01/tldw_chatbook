"""Command-palette entry for the throwaway Image Gen (dev) panel (Phase 1)."""
from __future__ import annotations

from textual.command import Hit, Hits, Provider

from .Screens.image_gen_demo_screen import ImageGenDemoScreen


class ImageGenCommandProvider(Provider):
    """Yield a single "Image Gen (dev)" command that opens the demo panel."""

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        label = "Image Gen (dev)"
        score = matcher.match(label)
        if score > 0:
            yield Hit(
                score,
                matcher.highlight(label),
                lambda: self.app.push_screen(ImageGenDemoScreen(self.app)),
                help="Open the throwaway image-generation demo panel",
            )
