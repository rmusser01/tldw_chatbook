"""Command-palette entry for the throwaway Image Gen (dev) panel (Phase 1).

The demo screen is imported inside ``search`` (TASK-21103): this provider is
imported by ``app.py`` at module scope, and a module-level screen import
dragged ``Image_Generation.worker`` -> ``request_validation`` -> PIL onto
the app boot path.
"""
from __future__ import annotations

from textual.command import Hit, Hits, Provider


class ImageGenCommandProvider(Provider):
    """Yield a single "Image Gen (dev)" command that opens the demo panel."""

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        label = "Image Gen (dev)"
        score = matcher.match(label)
        if score > 0:
            from .Screens.image_gen_demo_screen import (  # noqa: PLC0415 - keeps PIL off the boot path (TASK-21103)
                ImageGenDemoScreen,
            )

            yield Hit(
                score,
                matcher.highlight(label),
                lambda: self.app.push_screen(ImageGenDemoScreen(self.app)),
                help="Open the throwaway image-generation demo panel",
            )
