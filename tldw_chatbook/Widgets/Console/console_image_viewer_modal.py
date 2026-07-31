"""Full-size image viewer modal for character portraits (task-1534).

Clicking an avatar thumb (Console rail "Character" box, Roleplay Inspector
portrait) opens the image as large as the viewport allows: the graphics
path (TGP/Sixel terminals) shows the true raster, everything else gets a
viewport-sized quadrant mosaic -- roughly 10x the pixel budget of the
16x8-cell thumbnails. Escape or a click anywhere closes it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual import events
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Static

if TYPE_CHECKING:  # pragma: no cover
    from PIL import Image as PILImage


class AvatarViewRequested(Message):
    """A clickable avatar box asking its screen to open the full-size view.

    Carries no payload on purpose: each screen resolves the CURRENT
    portrait itself (selection can change between mount and click).
    """


class ClickableAvatarBox(Container):
    """Avatar thumb holder that requests the full-size viewer on click."""

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.post_message(AvatarViewRequested())


class ConsoleImageViewerModal(ModalScreen[None]):
    """Show one image as large as the current viewport allows."""

    DEFAULT_CSS = """
    ConsoleImageViewerModal {
        align: center middle;
    }

    #console-image-viewer {
        width: auto;
        height: auto;
        max-width: 100%;
        max-height: 100%;
        border: tall gray;
        background: black;
        padding: 0 1;
    }

    #console-image-viewer-title {
        width: 100%;
        height: 1;
        text-align: center;
        color: $text-muted;
    }

    #console-image-viewer-body {
        width: auto;
        height: auto;
    }

    #console-image-viewer-hint {
        width: 100%;
        height: 1;
        text-align: center;
        color: $text-muted;
    }
    """

    BINDINGS = [("escape", "dismiss_viewer", "Close")]

    def __init__(self, image: "PILImage.Image", *, title: str = "") -> None:
        super().__init__()
        self._image = image
        self._title = title

    def compose(self) -> ComposeResult:
        with Vertical(id="console-image-viewer"):
            yield Static(self._title, id="console-image-viewer-title", markup=False)
            yield Container(id="console-image-viewer-body")
            yield Static(
                "Esc / click to close", id="console-image-viewer-hint", markup=False
            )

    async def on_mount(self) -> None:
        body = self.query_one("#console-image-viewer-body", Container)
        await body.mount(self._build_full_size_widget())

    def _build_full_size_widget(self) -> Static:
        """Build the largest renderable the viewport allows.

        Mirrors the avatar fallback ladder: graphics widget when the
        session default mode resolves to graphics, quadrant mosaic
        otherwise. Any graphics-path failure degrades to the mosaic.
        The viewer always uses fit="contain" -- the point is seeing the
        WHOLE image, unlike the cover-cropped thumbs.
        """
        cols = max(20, self.app.size.width - 8)
        lines = max(10, self.app.size.height - 6)
        mode = self._resolve_mode()
        if mode == "graphics":
            try:
                from textual_image.widget import Image as _GraphicsImage

                from ...Chat.console_image_view import fit_image_cell_size

                widget: Any = _GraphicsImage(
                    self._image, id="console-image-viewer-image"
                )
                w, h = fit_image_cell_size(
                    self._image.width, self._image.height, cols, lines
                )
                widget.styles.width = w
                widget.styles.height = h
                return widget
            except Exception:
                pass
        from ...Utils.mosaic_render import mosaic_from_image

        return Static(
            mosaic_from_image(self._image, cols, lines),
            id="console-image-viewer-image",
        )

    def _resolve_mode(self) -> str:
        try:
            from ...Chat.console_image_view import resolve_default_mode

            app_config = getattr(self.app, "app_config", {}) or {}
            return resolve_default_mode(app_config)
        except Exception:
            return "pixels"

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.dismiss(None)

    def action_dismiss_viewer(self) -> None:
        self.dismiss(None)
