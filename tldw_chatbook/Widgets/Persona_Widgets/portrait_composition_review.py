"""Static paired-frame review built only from original normalized PNGs."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from io import BytesIO
from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static

from ...Character_Chat.portrait_compositing import (
    PortraitCompositionRecipe,
    _workspace,
    validate_recipe_geometry,
)
from ...Character_Chat.portrait_performance import (
    FRAME_ROLES,
    MAX_FRAME_BYTES,
    MAX_FRAME_PIXELS,
    PerformanceFrame,
)
from ...Character_Chat.portrait_performance_authoring import PortraitPerformanceDraft
from ...Character_Chat.visual_identity import MAX_EXPRESSION_IMAGE_DIMENSION
from ...Chat.character_expression_playback import _adjust_budget
from ...Utils.mosaic_render import mosaic_from_image
from ..modal_dismissal import SafeModalDismissMixin

CELL_ROLES = (FRAME_ROLES[:3], FRAME_ROLES[3:])


@dataclass(frozen=True, slots=True)
class ReviewRaster:
    """One bounded matrix; detail crops contain unresampled source RGBA pixels."""

    png: bytes = field(repr=False)
    width: int
    height: int
    crop_box: tuple[int, int, int, int] | None
    roles: tuple[tuple[str, ...], ...] = CELL_ROLES


def build_review_raster(
    frames: tuple[PerformanceFrame, ...],
    recipe: PortraitCompositionRecipe,
    box: tuple[int, int],
    *,
    region: str = "overview",
    pan: tuple[int, int] = (0, 0),
) -> ReviewRaster:
    """Decode/copy/close one original at a time, never six full canvases together."""
    from PIL import Image

    validate_recipe_geometry(recipe)
    validated = PortraitPerformanceDraft._validated(frames)
    if set(validated) != set(FRAME_ROLES):
        raise ValueError("portrait_performance_requires_six_roles")
    if region not in {"overview", "left_eye", "right_eye", "mouth"}:
        raise ValueError("portrait_review_region_invalid")
    if any(type(value) is not int or value < 1 for value in box) or any(
        type(value) is not int for value in pan
    ):
        raise ValueError("portrait_review_viewport_invalid")
    rect = getattr(recipe, region) if region != "overview" else None
    width = min(
        rect.width if rect else recipe.width,
        max(2, box[0] * 2 // 3),
        MAX_EXPRESSION_IMAGE_DIMENSION // 3,
    )
    width -= width % 2
    height = min(
        rect.height if rect else recipe.height,
        max(1, box[1]),
        MAX_EXPRESSION_IMAGE_DIMENSION // 2,
    )
    height = min(height, MAX_FRAME_PIXELS // (6 * width))
    crop = None
    if rect:
        x = rect.x + min(max(0, pan[0]), rect.width - width)
        y = rect.y + min(max(0, pan[1]), rect.height - height)
        crop = (x, y, x + width, y + height)
    raster_width, raster_height = 3 * width, 2 * height
    owned = raster_width * raster_height * 12
    _adjust_budget(owned)
    try:
        with Image.new("RGBA", (raster_width, raster_height)) as matrix:
            try:
                for index, role in enumerate(FRAME_ROLES):
                    with _workspace(validated[role].png, 24) as (stack, image):
                        if image.size != (recipe.width, recipe.height):
                            raise ValueError("portrait_composition_geometry_mismatch")
                        if crop:
                            tile = image.crop(crop)
                        else:
                            # Overview compensates for physical terminal-cell aspect.
                            scale = min(
                                width / (2 * image.width), height / image.height
                            )
                            tile = image.resize(
                                (
                                    max(1, round(2 * image.width * scale)),
                                    max(1, round(image.height * scale)),
                                )
                            )
                        stack.callback(tile.close)
                        matrix.paste(
                            tile,
                            (
                                (index % 3) * width + (width - tile.width) // 2,
                                (index // 3) * height + (height - tile.height) // 2,
                            ),
                        )
                with BytesIO() as output:
                    matrix.save(output, format="PNG")
                    png = output.getvalue()
                if len(png) > MAX_FRAME_BYTES:
                    raise ValueError("portrait_review_bytes_exceeded")
                return ReviewRaster(png, raster_width, raster_height, crop)
            finally:
                matrix.close()
    finally:
        _adjust_budget(-owned)


def _render(frames, recipe, box, region, pan, monochrome):
    raster = build_review_raster(frames, recipe, box, region=region, pan=pan)
    with _workspace(raster.png, 24) as (_, image):
        # This exact 2×2 grid adds no fitted thumbnail resampling to detail data.
        text = mosaic_from_image(
            image,
            raster.width // 2,
            raster.height // 2,
            fit="native",
            monochrome=monochrome,
        )
    return raster, text


class _ReviewCanvas(Static, can_focus=True):
    pass


class PortraitCompositionReview(SafeModalDismissMixin, ModalScreen[None]):
    """Review open/closed eye rows and common native source crops without side effects."""

    BINDINGS: ClassVar = [
        ("escape", "close", "Close"),
        ("left,h", "pan(-1,0)", "Pan left"),
        ("right,l", "pan(1,0)", "Pan right"),
        ("up,k", "pan(0,-1)", "Pan up"),
        ("down,j", "pan(0,1)", "Pan down"),
    ]
    BUNDLED_CSS = """
    PortraitCompositionReview { align: center middle; }
    PortraitCompositionReview > VerticalScroll.portrait-pairs-dialog {
        width: $ds-inspector-modal-width; max-width: $ds-width-full;
        height: $ds-height-full; max-height: $ds-height-full;
        padding: $ds-space-inset $ds-space-section;
        background: $ds-surface-panel; border: tall $ds-action-focus;
    }
    PortraitCompositionReview .pairs-copy { height: $ds-height-auto; color: $ds-text-muted; }
    PortraitCompositionReview #portrait-pairs-canvas { width: $ds-width-full; height: $ds-height-fill; min-height: $ds-portrait-review-lines; }
    PortraitCompositionReview .pairs-pan { height: $ds-height-auto; width: $ds-width-full; }
    PortraitCompositionReview .pairs-coordinate { width: $ds-width-fill; height: $ds-height-auto; }
    PortraitCompositionReview Input.portrait-pairs-input { width: $ds-width-full; }
    """

    def __init__(
        self,
        frames: tuple[PerformanceFrame, ...],
        recipe: PortraitCompositionRecipe,
        is_current: Callable[[], bool],
    ) -> None:
        super().__init__()
        self._frames, self._recipe, self._is_current = frames, recipe, is_current
        self._retired = False
        self._epoch = 0
        self._job: asyncio.Task | None = None
        self._raster: ReviewRaster | None = None
        self._region = "overview"
        self._pan = (0, 0)
        self._loading = False

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="portrait-pairs-dialog"):
            yield Static("Portrait pairs", classes="dialog-title")
            yield Static(
                "Compare each mouth across open/closed eyes. Detail copies original RGBA pixels without resizing; terminal glyph colors are approximations. Arrow keys or h/j/k/l pan while the image is focused; offsets are source pixels.",
                classes="pairs-copy",
                markup=False,
            )
            yield Select(
                [
                    ("Scaled overview", "overview"),
                    ("Image-left eye detail", "left_eye"),
                    ("Image-right eye detail", "right_eye"),
                    ("Mouth detail", "mouth"),
                ],
                value="overview",
                allow_blank=False,
                id="portrait-pairs-region",
                classes="form-select",
            )
            yield Static(
                "Open eyes: rest | small | open\nClosed eyes: blink_rest | blink_small | blink_open",
                id="portrait-pairs-labels",
                classes="pairs-copy",
                markup=False,
            )
            yield _ReviewCanvas(
                "Preparing original-frame review…",
                id="portrait-pairs-canvas",
                markup=False,
            )
            yield Static(
                "", id="portrait-pairs-bounds", classes="pairs-copy", markup=False
            )
            with Horizontal(classes="pairs-pan"):
                with Vertical(classes="pairs-coordinate"):
                    yield Static("Pan X (pixels)", classes="form-label")
                    yield Input(
                        "0",
                        type="integer",
                        id="portrait-pairs-x",
                        classes="form-input portrait-pairs-input",
                    )
                with Vertical(classes="pairs-coordinate"):
                    yield Static("Pan Y (pixels)", classes="form-label")
                    yield Input(
                        "0",
                        type="integer",
                        id="portrait-pairs-y",
                        classes="form-input portrait-pairs-input",
                    )
            yield Button("Close", id="portrait-pairs-close")

    def on_mount(self) -> None:
        self.query_one("#portrait-pairs-canvas").focus()
        self.call_after_refresh(self._request_render)

    def on_resize(self) -> None:
        if self.is_mounted:
            self.call_after_refresh(self._request_render)

    def _current(self) -> bool:
        return not self._retired and self.is_mounted and self._is_current()

    def _request_render(self) -> None:
        self._epoch += 1
        if self._current() and (self._job is None or self._job.done()):
            self._job = asyncio.create_task(self._refresh())

    async def _refresh(self) -> None:
        while self._current():
            epoch = self._epoch
            canvas = self.query_one("#portrait-pairs-canvas", Static)
            box = (
                max(1, canvas.content_size.width),
                max(1, canvas.content_size.height),
            )
            try:
                raster, text = await asyncio.to_thread(
                    _render,
                    self._frames,
                    self._recipe,
                    box,
                    self._region,
                    self._pan,
                    self.app.console.no_color,
                )
                if not self._current():
                    return
                if epoch != self._epoch:
                    continue
                self._raster = raster
                canvas.update(text)
                self.query_one("#portrait-pairs-bounds", Static).update(
                    f"Common source crop: {raster.crop_box}; {raster.width}×{raster.height} review raster."
                    if raster.crop_box
                    else "Scaled overview. Choose an eye or mouth detail to inspect original source pixels."
                )
                return
            except (ValueError, OSError, ImportError, MemoryError):
                if self._current():
                    canvas.update(
                        "Review unavailable within the shared image limits. Close and retry when other previews finish."
                    )
                return

    async def wait_for_render(self) -> None:
        """Drain the current coalesced render without canceling its native image work."""
        if self._job is not None:
            await asyncio.shield(self._job)

    @on(Select.Changed, "#portrait-pairs-region")
    def _region_changed(self, event: Select.Changed) -> None:
        event.stop()
        self._region = str(event.value)
        self._pan = (0, 0)
        self._sync_pan()
        self._request_render()

    def _sync_pan(self) -> None:
        self._loading = True
        try:
            self.query_one("#portrait-pairs-x", Input).value = str(self._pan[0])
            self.query_one("#portrait-pairs-y", Input).value = str(self._pan[1])
        finally:
            self._loading = False

    @on(Input.Changed)
    def _offset_changed(self, event: Input.Changed) -> None:
        event.stop()
        if self._loading:
            return
        try:
            self._pan = tuple(
                max(0, int(self.query_one(f"#portrait-pairs-{axis}", Input).value))
                for axis in ("x", "y")
            )
        except ValueError:
            return
        self._request_render()

    def action_pan(self, x: int, y: int) -> None:
        if not self._current() or self._region == "overview":
            return
        self._pan = (max(0, self._pan[0] + x), max(0, self._pan[1] + y))
        self._sync_pan()
        self._request_render()

    @on(Button.Pressed, "#portrait-pairs-close")
    def _close(self, event: Button.Pressed) -> None:
        event.stop()
        self.retire()

    def action_close(self) -> None:
        self.retire()

    def retire(self) -> None:
        """Explicitly remove displayed stale artwork as well as fencing future jobs."""
        self._retired = True
        self._raster = None
        if self.is_mounted:
            self.query_one("#portrait-pairs-canvas", Static).update("Review retired.")
            self.dismiss_safe_once_when_on_top(None)

    def on_unmount(self) -> None:
        self._retired = True
        self._raster = None
