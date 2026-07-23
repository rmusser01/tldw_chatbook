"""THROWAWAY dev panel (Phase 1) for the image-generation feature.

Proves the backend engine (`Image_Generation.worker` + `.listing`) end to end
from inside the Textual app: pick a configured backend, type a prompt, hit
Generate, see the image. Replaced by the real chat card in Phase 2 -- this
screen persists nothing and is reachable only via the command palette.

Renders the generated image with the low-level `rich_pixels`/PIL primitives
directly (NOT the Console transcript/attachment rendering path), since this
panel has no message list to attach an image to.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Input, Label, Select, Static, TextArea

from ..Navigation.base_app_screen import BaseAppScreen
from ...Image_Generation.worker import build_request, run_generation
from ...Image_Generation.listing import list_image_models_for_catalog

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class ImageGenDemoScreen(BaseAppScreen):
    """Throwaway proof-of-life panel for the image-generation backends."""

    def __init__(self, app_instance: "TldwCli", **kwargs):
        super().__init__(app_instance, "imagegen_demo", **kwargs)

    def compose_content(self) -> ComposeResult:
        with Vertical(id="imagegen-demo"):
            yield Label("Image Gen (dev) — throwaway Phase-1 panel")
            opts = [
                (
                    f'{entry["name"]}{"" if entry.get("is_configured") else "  (not configured)"}',
                    entry["name"],
                )
                for entry in list_image_models_for_catalog()
            ]
            select_options = opts or [("(no backends enabled)", "")]
            # Pre-select the first entry so the panel is generate-ready
            # without an extra manual pick -- this is a one-backend-at-a-time
            # proof surface, not a form the user is expected to leave blank.
            yield Select(
                select_options,
                id="imagegen-backend",
                value=select_options[0][1],
                allow_blank=False,
            )
            yield TextArea(id="imagegen-prompt")
            yield TextArea(id="imagegen-negative")
            yield Input(placeholder="seed (blank = -1)", id="imagegen-seed")
            yield Button("Generate", id="imagegen-generate", variant="primary")
            yield Static(id="imagegen-status")
            yield Static(id="imagegen-image")
            yield Static(id="imagegen-meta")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "imagegen-generate":
            self._generate()

    @work(thread=True, exclusive=True, group="imagegen-demo")
    def _generate(self) -> None:
        backend = self.query_one("#imagegen-backend", Select).value
        prompt = self.query_one("#imagegen-prompt", TextArea).text
        negative = self.query_one("#imagegen-negative", TextArea).text or None
        seed_raw = self.query_one("#imagegen-seed", Input).value.strip()
        seed = int(seed_raw) if seed_raw.lstrip("-").isdigit() else -1
        self.app.call_from_thread(
            self.query_one("#imagegen-status", Static).update, "Generating…"
        )
        try:
            req = build_request(
                backend=backend,
                prompt=prompt,
                negative_prompt=negative,
                seed=seed,
                image_format="png",
            )
            res = run_generation(req)  # blocking; we are in a worker thread
        except Exception as exc:  # surface the error clearly (incl. inline_max_bytes cap)
            logger.warning("Image Gen (dev) generation failed: {}", exc)
            self.app.call_from_thread(
                self.query_one("#imagegen-status", Static).update, f"Error: {exc}"
            )
            return
        self.app.call_from_thread(self._render_result, res, req)

    def _render_result(self, res, req) -> None:
        # Low-level render (rich-pixels), decoupled from the Console
        # transcript/attachment path -- this panel has no message list.
        from rich_pixels import Pixels
        from PIL import Image as PILImage
        import io

        img = PILImage.open(io.BytesIO(res.content))
        img.thumbnail((80, 40))
        self.query_one("#imagegen-status", Static).update("Done")
        self.query_one("#imagegen-image", Static).update(Pixels.from_image(img))
        self.query_one("#imagegen-meta", Static).update(
            f"backend={req.backend} format={res.content_type} bytes={res.bytes_len} "
            f"seed={req.seed} prompt={req.prompt!r}"
        )
