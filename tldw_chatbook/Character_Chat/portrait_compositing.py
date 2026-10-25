"""Bounded source-preserving component composition; no provider or storage authority."""

from __future__ import annotations

import hashlib
import re
import warnings
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Literal

from .portrait_performance import (
    FRAME_ROLES,
    MAX_FRAME_BYTES,
    MAX_FRAME_PIXELS,
    MAX_SET_BYTES,
    FrameRole,
    PerformanceFrame,
    normalize_performance_frame,
)

if TYPE_CHECKING:
    from PIL.Image import Image

ComponentRole = Literal["eyelids", "small", "open"]
COMPONENT_ROLES: tuple[ComponentRole, ...] = ("eyelids", "small", "open")
_PIXEL_TAG = b"tldw-portrait-base-rgba-v1\0"


@dataclass(frozen=True, slots=True)
class PixelRect:
    """One rectangle in prepared-source pixels, with exclusive right/bottom bounds."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class PortraitCompositionRecipe:
    """Versioned source-pixel geometry; no image bytes or profile authority."""

    algorithm_version: int
    width: int
    height: int
    base_pixel_sha256: str
    left_eye: PixelRect
    right_eye: PixelRect
    mouth: PixelRect
    neutral_base_confirmed: bool


def validate_recipe_geometry(recipe: PortraitCompositionRecipe) -> None:
    """Refuse unsupported/ambiguous geometry before any decode or state change."""
    if (
        type(recipe) is not PortraitCompositionRecipe
        or type(recipe.algorithm_version) is not int
        or recipe.algorithm_version != 1
        or recipe.neutral_base_confirmed is not True
        or type(recipe.width) is not int
        or type(recipe.height) is not int
        or min(recipe.width, recipe.height) < 1
        or recipe.width * recipe.height > MAX_FRAME_PIXELS
        or not isinstance(recipe.base_pixel_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", recipe.base_pixel_sha256)
    ):
        raise ValueError("portrait_composition_recipe_invalid")
    rectangles = (recipe.left_eye, recipe.right_eye, recipe.mouth)
    for rect in rectangles:
        if (
            type(rect) is not PixelRect
            or any(
                type(value) is not int
                for value in (rect.x, rect.y, rect.width, rect.height)
            )
            or min(rect.x, rect.y) < 0
            or min(rect.width, rect.height) < 3
            or rect.x + rect.width > recipe.width
            or rect.y + rect.height > recipe.height
        ):
            raise ValueError("portrait_composition_region_invalid")
    for index, a in enumerate(rectangles):
        for b in rectangles[index + 1 :]:
            if (
                a.x < b.x + b.width
                and b.x < a.x + a.width
                and a.y < b.y + b.height
                and b.y < a.y + a.height
            ):
                raise ValueError("portrait_composition_regions_overlap")


def _open(raw: bytes, stack: ExitStack) -> Image:
    from PIL import Image

    from . import visual_identity as visual

    if (
        type(raw) is not bytes
        or not raw
        or len(raw) > min(MAX_FRAME_BYTES, visual.MAX_EXPRESSION_ASSET_BYTES)
    ):
        raise ValueError("portrait_composition_image_bytes_invalid")
    image = Image.open(stack.enter_context(BytesIO(raw)))
    stack.callback(image.close)
    if (
        image.format not in {"PNG", "JPEG", "WEBP"}
        or getattr(image, "n_frames", 1) != 1
        or getattr(image, "is_animated", False)
        or min(image.size) < 1
        or image.width * image.height
        > min(MAX_FRAME_PIXELS, visual.MAX_EXPRESSION_ASSET_DECODED_PIXELS)
        or max(image.size) > visual.MAX_EXPRESSION_IMAGE_DIMENSION
    ):
        raise ValueError("portrait_composition_image_limits_exceeded")
    return image


def _rgba(image: Image, stack: ExitStack) -> Image:
    converted = image.convert("RGBA")
    stack.callback(converted.close)
    image.close()
    return converted


@contextmanager
def _workspace(
    base_png: bytes, bytes_per_pixel: int
) -> Iterator[tuple[ExitStack, Image]]:
    from PIL import Image

    from ..Chat.character_expression_playback import _adjust_budget

    try:
        with ExitStack() as stack, warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = _open(base_png, stack)
            owned = image.width * image.height * bytes_per_pixel
            _adjust_budget(owned)
            try:
                yield stack, _rgba(image, stack)
            finally:
                stack.close()
                _adjust_budget(-owned)
    except (
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise ValueError("portrait_composition_image_invalid") from error


def _digest(image: Image) -> str:
    header = (
        _PIXEL_TAG + image.width.to_bytes(4, "big") + image.height.to_bytes(4, "big")
    )
    digest = hashlib.sha256(header)
    digest.update(image.tobytes())
    return digest.hexdigest()


def prepared_base_pixel_sha256(base_png: bytes) -> str:
    """Identify exact bounded RGBA pixels, independent of PNG encoding/metadata."""
    with _workspace(base_png, 12) as (_, image):
        return _digest(image)


def _check_base(image: Image, recipe: PortraitCompositionRecipe) -> None:
    if (
        image.size != (recipe.width, recipe.height)
        or _digest(image) != recipe.base_pixel_sha256
    ):
        raise ValueError("portrait_composition_base_changed")


def validate_recipe(base_png: bytes, recipe: PortraitCompositionRecipe) -> None:
    """Validate geometry and the exact canonical prepared-base pixel identity."""
    validate_recipe_geometry(recipe)
    with _workspace(base_png, 12) as (_, image):
        _check_base(image, recipe)


def _load(raw: bytes, size: tuple[int, int], stack: ExitStack) -> Image:
    image = _open(raw, stack)
    if image.size != size:
        raise ValueError("portrait_composition_geometry_mismatch")
    return _rgba(image, stack)


def _mask(recipe: PortraitCompositionRecipe, *, eyes: bool, stack: ExitStack) -> Image:
    from PIL import Image, ImageDraw

    mask = Image.new("L", (recipe.width, recipe.height))
    stack.callback(mask.close)
    draw = ImageDraw.Draw(mask)
    for rect in (recipe.left_eye, recipe.right_eye) if eyes else (recipe.mouth,):
        feather = min(3, max(1, (min(rect.width, rect.height) - 1) // 2))
        for depth in range(1, feather + 1):
            draw.rectangle(
                (
                    rect.x + depth,
                    rect.y + depth,
                    rect.x + rect.width - 1 - depth,
                    rect.y + rect.height - 1 - depth,
                ),
                fill=255 * depth // feather,
            )
    return mask


def _support(mask: Image, stack: ExitStack) -> Image:
    binary = mask.point([0] + [255] * 255)
    stack.callback(binary.close)
    return binary


def _equal(a: Image, b: Image) -> bool:
    from PIL import ImageChops

    difference = ImageChops.difference(a, b)
    try:
        return all(high == 0 for _, high in difference.getextrema())
    finally:
        difference.close()


def _validate_plane(base: Image, plane: Image, support: Image) -> None:
    from PIL import Image

    with ExitStack() as stack:
        alpha = base.getchannel("A")
        other_alpha = plane.getchannel("A")
        stack.callback(alpha.close)
        stack.callback(other_alpha.close)
        if alpha.tobytes() != other_alpha.tobytes():
            raise ValueError("portrait_composition_alpha_changed")
        outside = Image.composite(base, plane, support)
        stack.callback(outside.close)
        if not _equal(base, outside):
            raise ValueError("portrait_composition_outside_changed")


def _frame(image: Image, role: FrameRole) -> PerformanceFrame:
    with BytesIO() as output:
        image.save(output, format="PNG")
        return normalize_performance_frame(role, output.getvalue())


def compose_component(
    base_png: bytes,
    recipe: PortraitCompositionRecipe,
    role: ComponentRole,
    edit_png: bytes,
) -> bytes:
    """Feather a raw edit onto the base once, preserving all base alpha pixels."""
    from PIL import Image

    if role not in COMPONENT_ROLES:
        raise ValueError("portrait_composition_component_invalid")
    validate_recipe_geometry(recipe)
    with _workspace(base_png, 32) as (stack, base):
        _check_base(base, recipe)
        edit = _load(edit_png, base.size, stack)
        mask = _mask(recipe, eyes=role == "eyelids", stack=stack)
        result = Image.composite(edit, base, mask)
        stack.callback(result.close)
        alpha = base.getchannel("A")
        stack.callback(alpha.close)
        result.putalpha(alpha)
        return _frame(result, "blink_rest" if role == "eyelids" else role).png


def assemble_components(
    base_png: bytes,
    recipe: PortraitCompositionRecipe,
    components: Mapping[ComponentRole, bytes],
) -> tuple[PerformanceFrame, ...]:
    """Derive available roles from finished planes; never feather them again."""
    from PIL import Image

    if not isinstance(components, Mapping) or any(
        role not in COMPONENT_ROLES for role in components
    ):
        raise ValueError("portrait_composition_components_invalid")
    validate_recipe_geometry(recipe)
    with _workspace(base_png, 32) as (stack, base):
        _check_base(base, recipe)
        eyes = _support(_mask(recipe, eyes=True, stack=stack), stack)
        mouth = _support(_mask(recipe, eyes=False, stack=stack), stack)
        planes = {}
        for role, raw in components.items():
            planes[role] = plane = _load(raw, base.size, stack)
            _validate_plane(base, plane, eyes if role == "eyelids" else mouth)
        images = {"rest": base}
        images.update(
            {role: planes[role] for role in ("small", "open") if role in planes}
        )
        if "eyelids" in planes:
            images["blink_rest"] = planes["eyelids"]
            for role in ("small", "open"):
                if role in planes:
                    image = Image.composite(planes["eyelids"], planes[role], eyes)
                    stack.callback(image.close)
                    images["blink_" + role] = image
        frames = tuple(
            _frame(images[role], role) for role in FRAME_ROLES if role in images
        )
        if sum(len(frame.png) for frame in frames) > MAX_SET_BYTES:
            raise ValueError("portrait_performance_set_bytes_exceeded")
        return frames


def recover_components(
    base_png: bytes,
    recipe: PortraitCompositionRecipe,
    frames: tuple[PerformanceFrame, ...],
) -> dict[ComponentRole, bytes]:
    """Recover finished planes only after validating the entire stored matrix."""
    validate_recipe_geometry(recipe)
    if (
        type(frames) is not tuple
        or len(frames) != 6
        or any(type(f) is not PerformanceFrame for f in frames)
        or tuple(f.role for f in frames) != FRAME_ROLES
        or any(
            type(f.png) is not bytes
            or hashlib.sha256(f.png).hexdigest() != f.sha256
            or (f.width, f.height) != (recipe.width, recipe.height)
            for f in frames
        )
        or sum(len(f.png) for f in frames) > MAX_SET_BYTES
    ):
        raise ValueError("portrait_composition_frames_invalid")
    lookup = {f.role: f.png for f in frames}
    components = {
        "eyelids": lookup["blink_rest"],
        "small": lookup["small"],
        "open": lookup["open"],
    }
    expected = assemble_components(base_png, recipe, components)
    with _workspace(base_png, 16) as (_, base):
        for actual, wanted in zip(frames, expected, strict=True):
            with ExitStack() as stack:
                a = _load(actual.png, base.size, stack)
                b = _load(wanted.png, base.size, stack)
                if not _equal(a, b):
                    raise ValueError("portrait_composition_pair_mismatch")
    return components
