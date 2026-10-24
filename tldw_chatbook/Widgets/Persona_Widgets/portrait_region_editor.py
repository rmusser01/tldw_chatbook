"""One-time source-space region selection; the caller owns settings authority."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import ClassVar

from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Select, Static

from ...Character_Chat.portrait_compositing import (
    PixelRect,
    PortraitCompositionRecipe,
    _mask,
    _workspace,
    validate_recipe_geometry,
)
from ...Utils.mosaic_render import mosaic_contain_cell_size, mosaic_from_image
from ..modal_dismissal import SafeModalDismissMixin

REGIONS = (
    ("Image-left eye", "left_eye"),
    ("Image-right eye", "right_eye"),
    ("Mouth", "mouth"),
)


@dataclass(frozen=True, slots=True)
class RegionTransform:
    """The exact centered mosaic grid, with source endpoints at its corner cells."""

    width: int
    height: int
    x: int
    y: int
    cols: int
    lines: int

    @classmethod
    def contain(cls, source: tuple[int, int], box: tuple[int, int]) -> RegionTransform:
        """Use the renderer's rounding and terminal-cell aspect, never image stretching."""
        cols, lines = mosaic_contain_cell_size(*source, *box)
        return cls(*source, (box[0] - cols) // 2, (box[1] - lines) // 2, cols, lines)

    def source_point(self, x: int, y: int) -> tuple[int, int] | None:
        """Reject letterboxing and map a selectable cell to a prepared-source pixel."""
        x, y = x - self.x, y - self.y
        if not (0 <= x < self.cols and 0 <= y < self.lines):
            return None
        return (
            round(x * (self.width - 1) / max(1, self.cols - 1)),
            round(y * (self.height - 1) / max(1, self.lines - 1)),
        )


def _render(
    base: bytes,
    recipe: PortraitCompositionRecipe | None,
    rectangles: dict[str, PixelRect],
    role: str,
    box: tuple[int, int],
    monochrome: bool,
) -> tuple[Text, Text, RegionTransform]:
    """Close all decoded pixels inside the shared reservation before returning text."""
    from PIL import ImageDraw

    with _workspace(base, 24) as (stack, image):
        transform = RegionTransform.contain(image.size, box)
        outlined = image.copy()
        stack.callback(outlined.close)
        draw = ImageDraw.Draw(outlined)
        for key, rect in rectangles.items():
            draw.rectangle(
                (rect.x, rect.y, rect.x + rect.width - 1, rect.y + rect.height - 1),
                outline="white",
                width=2 if key == role else 1,
            )
        mosaic = mosaic_from_image(outlined, *box, monochrome=monochrome)
        lines = mosaic.split("\n")
        for label, key in REGIONS:
            rect = rectangles.get(key)
            if rect is not None:
                x = min(transform.cols - 1, rect.x * transform.cols // image.width)
                y = min(transform.lines - 1, rect.y * transform.lines // image.height)
                line = lines[y]
                marker = {"left_eye": "L", "right_eye": "R", "mouth": "M"}[key]
                lines[y] = Text.assemble(
                    line[:x], (marker, "reverse bold"), line[x + 1 :]
                )
        overview = Text("\n" * transform.y)
        for index, line in enumerate(lines):
            if index:
                overview.append("\n")
            overview.append(" " * transform.x)
            overview.append_text(line)
        rect = rectangles.get(role)
        if rect is None:
            return (
                overview,
                Text("Select two corners or enter x, y, width and height."),
                transform,
            )
        bounds = (rect.x, rect.y, rect.x + rect.width, rect.y + rect.height)
        crop = image.crop(bounds)
        stack.callback(crop.close)
        detail = Text(
            f"Source crop: ({rect.x}, {rect.y}) · {rect.width}×{rect.height} pixels\n"
        )
        detail.append_text(mosaic_from_image(crop, *box, monochrome=monochrome))
        if recipe is not None:
            mask = _mask(recipe, eyes=role != "mouth", stack=stack)
            crop_mask = mask.crop(bounds)
            stack.callback(crop_mask.close)
            detail.append("\nFeather mask: dark = preserved; light = edited\n")
            detail.append_text(
                mosaic_from_image(crop_mask, *box, monochrome=monochrome)
            )
        else:
            detail.append(
                "\nComplete three valid, separate regions to preview the feather mask."
            )
        return overview, detail, transform


class _RegionSource(Static):
    def on_click(self, event: events.Click) -> None:
        event.stop()
        offset = event.get_content_offset(self)
        if offset is not None:
            self.screen.select_corner(offset.x, offset.y)


class PortraitRegionEditor(SafeModalDismissMixin, ModalScreen[bool]):
    """Edit three rectangles with explicit confirmation and an independently saved result."""

    BINDINGS: ClassVar = [("escape", "cancel", "Cancel")]
    BUNDLED_CSS = """
    PortraitRegionEditor { align: center middle; }
    PortraitRegionEditor > VerticalScroll {
        width: $ds-inspector-modal-width; max-width: $ds-width-full;
        height: $ds-height-full; max-height: $ds-height-full;
        padding: $ds-space-inset $ds-space-section;
        background: $ds-surface-panel; border: tall $ds-action-focus;
    }
    PortraitRegionEditor .region-copy { height: $ds-height-auto; color: $ds-text-muted; }
    PortraitRegionEditor #portrait-regions-source { height: $ds-portrait-review-lines; width: $ds-width-full; }
    PortraitRegionEditor #portrait-regions-detail { height: $ds-height-auto; width: $ds-width-full; }
    PortraitRegionEditor .region-fields { height: $ds-height-auto; width: $ds-width-full; }
    PortraitRegionEditor .region-field { width: $ds-width-fill; height: $ds-height-auto; }
    PortraitRegionEditor .region-fields Input { width: $ds-width-full; }
    PortraitRegionEditor .dialog-buttons { height: $ds-height-auto; }
    PortraitRegionEditor.region-compact .region-fields,
    PortraitRegionEditor.region-compact .dialog-buttons { layout: vertical; }
    """

    def __init__(
        self,
        base: bytes,
        *,
        width: int,
        height: int,
        pixel_sha256: str,
        recipe: PortraitCompositionRecipe | None,
        save: Callable[[PortraitCompositionRecipe], Awaitable[bool]],
        is_current: Callable[[], bool],
    ) -> None:
        super().__init__()
        self._base = base
        self._width, self._height, self._pixel_sha256 = width, height, pixel_sha256
        self._rectangles = (
            {key: getattr(recipe, key) for _, key in REGIONS} if recipe else {}
        )
        self._confirmed = recipe is not None
        self._save, self._is_current = save, is_current
        self._retired = False
        self._saving = False
        self._loading = False
        self._corner: tuple[int, int] | None = None
        self._transform: RegionTransform | None = None
        self._preview_task: asyncio.Task | None = None
        self._preview_epoch = 0

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static("Portrait expression regions", classes="dialog-title")
            yield Static(
                f"Prepared base: {self._width}×{self._height} pixels. Choose a region, then click two opposite corners or edit its coordinates. Image-left/right refer to the picture. Markers: L = left eye, R = right eye, M = mouth.",
                classes="region-copy",
                markup=False,
            )
            yield Select(
                REGIONS,
                value="left_eye",
                allow_blank=False,
                id="portrait-regions-role",
                classes="form-select",
            )
            yield _RegionSource(
                "Preparing source…", id="portrait-regions-source", markup=False
            )
            with Horizontal(classes="region-fields"):
                for key, label in (
                    ("x", "X"),
                    ("y", "Y"),
                    ("width", "Width"),
                    ("height", "Height"),
                ):
                    with VerticalScroll(classes="region-field"):
                        yield Static(f"{label} (pixels)", classes="form-label")
                        yield Input(
                            type="integer",
                            id=f"portrait-region-{key}",
                            classes="form-input",
                        )
            yield Static(
                "",
                id="portrait-regions-validation",
                classes="region-copy",
                markup=False,
            )
            yield Static("", id="portrait-regions-detail", markup=False)
            yield Checkbox(
                "Base: open eyes, closed mouth",
                value=self._confirmed,
                id="portrait-regions-neutral",
                classes="form-checkbox",
            )
            yield Static(
                "If the base differs, save a suitable portrait or import six ready-made frames. Save regions keeps the currently applied performance.",
                classes="region-copy",
                markup=False,
            )
            with Horizontal(classes="dialog-buttons"):
                yield Button(
                    "Save regions",
                    id="portrait-regions-save",
                    variant="primary",
                    disabled=True,
                )
                yield Button("Cancel", id="portrait-regions-cancel")

    def on_mount(self) -> None:
        self._load_fields()
        self.call_after_refresh(self._request_preview)
        self.query_one("#portrait-regions-role").focus()

    def on_resize(self) -> None:
        if self.is_mounted:
            # Stack when four labeled pixel fields cannot fit their text and edges.
            labels = self.query(".region-field .form-label")
            required = sum(
                len(str(label.renderable)) + label.styles.gutter.width
                for label in labels
            )
            self.set_class(self.size.width < required, "region-compact")
            self._corner = self._transform = None
            self._request_preview()

    def _current(self) -> bool:
        return not self._retired and self.is_mounted and self._is_current()

    def _recipe(self, *, confirmed: bool = True) -> PortraitCompositionRecipe:
        recipe = PortraitCompositionRecipe(
            1,
            self._width,
            self._height,
            self._pixel_sha256,
            *(self._rectangles[key] for _, key in REGIONS),
            confirmed,
        )
        validate_recipe_geometry(recipe)
        return recipe

    def _validate(self) -> None:
        try:
            self._recipe(
                confirmed=self.query_one("#portrait-regions-neutral", Checkbox).value
            )
            message = (
                "Three regions ready. Save reuses them for this exact prepared source."
            )
            valid = True
        except (ValueError, KeyError):
            message = "Set three non-overlapping regions, at least 3×3 pixels each, within the base; confirm open eyes and closed mouth."
            valid = False
        self.query_one("#portrait-regions-validation", Static).update(message)
        self.query_one("#portrait-regions-save", Button).disabled = (
            not valid or self._saving or not self._current()
        )

    def _load_fields(self) -> None:
        self._loading = True
        try:
            role = str(self.query_one("#portrait-regions-role", Select).value)
            rect = self._rectangles.get(role)
            for key in ("x", "y", "width", "height"):
                self.query_one(f"#portrait-region-{key}", Input).value = (
                    str(getattr(rect, key)) if rect else ""
                )
        finally:
            self._loading = False
        self._validate()

    @on(Select.Changed, "#portrait-regions-role")
    def _role_changed(self, event: Select.Changed) -> None:
        event.stop()
        self._corner = None
        self._load_fields()
        self._request_preview()

    @on(Input.Changed)
    def _coordinates_changed(self, event: Input.Changed) -> None:
        event.stop()
        if self._loading:
            return
        role = str(self.query_one("#portrait-regions-role", Select).value)
        try:
            rect = PixelRect(
                *(
                    int(self.query_one(f"#portrait-region-{key}", Input).value)
                    for key in ("x", "y", "width", "height")
                )
            )
            if (
                min(rect.x, rect.y) < 0
                or min(rect.width, rect.height) < 3
                or rect.x + rect.width > self._width
                or rect.y + rect.height > self._height
            ):
                raise ValueError
            self._rectangles[role] = rect
        except ValueError:
            self._rectangles.pop(role, None)
        self._validate()
        self._request_preview()

    @on(Checkbox.Changed)
    def _confirmation_changed(self, event: Checkbox.Changed) -> None:
        event.stop()
        self._validate()

    def select_corner(self, x: int, y: int) -> None:
        """Two clicks select inclusive endpoint pixels; no letterbox coordinates escape."""
        if not self._current() or self._saving or self._transform is None:
            return
        point = self._transform.source_point(x, y)
        if point is None:
            return
        if self._corner is None:
            self._corner = point
            self.query_one("#portrait-regions-validation", Static).update(
                f"First corner: {point}. Select the opposite corner."
            )
            return
        first, self._corner = self._corner, None
        role = str(self.query_one("#portrait-regions-role", Select).value)
        self._rectangles[role] = PixelRect(
            min(first[0], point[0]),
            min(first[1], point[1]),
            abs(first[0] - point[0]) + 1,
            abs(first[1] - point[1]) + 1,
        )
        self._load_fields()
        self._request_preview()

    def _request_preview(self) -> None:
        self._preview_epoch += 1
        if self._current() and (
            self._preview_task is None or self._preview_task.done()
        ):
            self._preview_task = asyncio.create_task(self._refresh_preview())

    async def _refresh_preview(self) -> None:
        while self._current():
            epoch = self._preview_epoch
            source = self.query_one("#portrait-regions-source", Static)
            box = (
                max(1, source.content_size.width),
                max(1, source.content_size.height),
            )
            try:
                try:
                    recipe = self._recipe()
                except (ValueError, KeyError):
                    recipe = None
                result = await asyncio.to_thread(
                    _render,
                    self._base,
                    recipe,
                    dict(self._rectangles),
                    str(self.query_one("#portrait-regions-role", Select).value),
                    box,
                    self.app.console.no_color,
                )
                if not self._current():
                    return
                if epoch == self._preview_epoch:
                    overview, detail, self._transform = result
                    source.update(overview)
                    self.query_one("#portrait-regions-detail", Static).update(detail)
                    return
            except (ValueError, OSError, ImportError, MemoryError):
                if self._current():
                    source.update(
                        "Preview unavailable. Numeric coordinates remain available."
                    )
                return

    @on(Button.Pressed)
    async def _pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "portrait-regions-cancel":
            self.action_cancel()
        elif (
            event.button.id == "portrait-regions-save"
            and not self._saving
            and self._current()
        ):
            self._saving = True
            self._validate()
            for control in self.query("Input, Select, Checkbox, Button"):
                control.disabled = True
            saved = False
            try:
                saved = await self._save(
                    self._recipe(
                        confirmed=self.query_one(
                            "#portrait-regions-neutral", Checkbox
                        ).value
                    )
                )
                if self._current() and saved:
                    self.dismiss(True)
            finally:
                self._saving = False
                if self._current():
                    for control in self.query("Input, Select, Checkbox, Button"):
                        control.disabled = False
                    self._validate()
                    if not saved:
                        self.query_one("#portrait-regions-validation", Static).update(
                            "Regions could not be saved. If the source or settings changed, Cancel and Prepare again. Current artwork kept."
                        )

    def action_cancel(self) -> None:
        if not self._saving:
            self.retire()

    def retire(self) -> None:
        """Fence asynchronous preview completion before retiring its exact owner."""
        self._retired = True
        self._transform = None
        if self.is_mounted:
            self.dismiss(False)

    def on_unmount(self) -> None:
        self._retired = True
        self._base = b""
