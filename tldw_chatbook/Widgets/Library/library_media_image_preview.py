"""Narrow local-original image preview helpers for Library Media."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Literal, Mapping
from urllib.parse import urlparse

from textual.widget import Widget
from textual.widgets import Static

from tldw_chatbook.Utils import optional_deps

SUPPORTED_IMAGE_MIME_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/webp"}
)
SUPPORTED_IMAGE_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})
IMAGE_PREVIEW_MAX_DIMENSION = 1024

PreviewEligibilityReason = Literal[
    "eligible", "external", "remote", "unavailable", "unsupported"
]


@dataclass(frozen=True)
class PreviewEligibility:
    """Result of checking one detail and its local original-file receipt."""

    eligible: bool
    reason: PreviewEligibilityReason
    content_type: str = ""


def _normalized_content_type(value: Any) -> str:
    return str(value or "").partition(";")[0].strip().lower()


def _detail_type_hint(detail: Mapping[str, Any]) -> str:
    for key in ("mime_type", "content_type", "media_type", "type"):
        value = _normalized_content_type(detail.get(key))
        aliases = {
            "png": "image/png",
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "webp": "image/webp",
        }
        if value in aliases:
            return aliases[value]
        if value.startswith("image/"):
            return value
    return ""


def image_preview_eligibility(
    detail: Mapping[str, Any] | None,
    file_check: Mapping[str, Any] | None,
    *,
    backend: str,
) -> PreviewEligibility:
    """Return whether an existing local original is a supported raster image.

    Args:
        detail: Loaded media detail, or None when unavailable.
        file_check: Local original-file availability receipt.
        backend: Detail backend; only ``"local"`` is eligible.

    Returns:
        Eligibility decision with a stable reason and normalized MIME type.
    """
    if backend != "local":
        return PreviewEligibility(False, "external")
    if not isinstance(detail, Mapping):
        return PreviewEligibility(False, "unsupported")
    source_url = str(detail.get("url") or "").strip()
    if urlparse(source_url).scheme.lower() in {"http", "https"}:
        return PreviewEligibility(False, "remote")

    type_hint = _detail_type_hint(detail)
    if not isinstance(file_check, Mapping) or not bool(file_check.get("available")):
        return PreviewEligibility(False, "unavailable", type_hint)
    if str(file_check.get("source") or "") != "file_path":
        return PreviewEligibility(False, "unavailable", type_hint)

    content_type = _normalized_content_type(file_check.get("content_type"))
    if not content_type:
        content_type = type_hint
    if content_type not in SUPPORTED_IMAGE_MIME_TYPES:
        return PreviewEligibility(False, "unsupported", content_type)
    return PreviewEligibility(True, "eligible", content_type)


def decode_media_image(content: bytes) -> Any:
    """Decode and detach one supported image for terminal display.

    Args:
        content: Encoded raster-image bytes.

    Returns:
        Detached Pillow image bounded to the preview dimension limit.

    Raises:
        ImportError: If Pillow is unavailable.
        ValueError: If content is empty or its raster format is unsupported.
    """
    if not isinstance(content, bytes) or not content:
        raise ValueError("image content must be non-empty bytes")
    if not optional_deps.check_dependency("PIL", "pillow"):
        raise ImportError("Pillow is unavailable")
    from PIL import Image

    with Image.open(BytesIO(content)) as source:
        source.load()
        format_name = str(source.format or "").upper()
        if format_name not in SUPPORTED_IMAGE_FORMATS:
            raise ValueError(f"unsupported image format: {format_name or 'unknown'}")
        decoded = source.copy()
    decoded.format = format_name
    if max(decoded.size) > IMAGE_PREVIEW_MAX_DIMENSION:
        decoded.thumbnail(
            (IMAGE_PREVIEW_MAX_DIMENSION, IMAGE_PREVIEW_MAX_DIMENSION),
            Image.Resampling.LANCZOS,
        )
    return decoded


def build_media_image_widget(
    image: Any,
    *,
    app_config: Mapping[str, Any],
    box_cols: int,
    box_lines: int,
) -> Widget:
    """Build the existing graphics widget or universal mosaic fallback.

    Args:
        image: Decoded image object.
        app_config: Application image-rendering configuration.
        box_cols: Available preview width in terminal cells.
        box_lines: Available preview height in terminal cells.

    Returns:
        Graphics-protocol image when available, otherwise a mosaic Static.
    """
    from tldw_chatbook.Chat.console_image_view import (
        fit_image_cell_size,
        resolve_default_mode,
    )

    cols = max(1, int(box_cols))
    lines = max(1, int(box_lines))
    if resolve_default_mode(app_config) == "graphics":
        try:
            if not optional_deps.check_dependency("textual_image"):
                raise ImportError("textual_image is unavailable")
            from textual_image.widget import Image as GraphicsImage

            widget: Widget = GraphicsImage(image)
            width, height = fit_image_cell_size(
                image.width, image.height, cols, lines
            )
            widget.styles.width = width
            widget.styles.height = height
            return widget
        except Exception:
            pass

    from tldw_chatbook.Utils.mosaic_render import mosaic_from_image

    mosaic = mosaic_from_image(image, cols, lines, fit="contain")
    rendered_lines = mosaic.plain.splitlines() or [""]
    widget = Static(mosaic)
    widget.styles.width = max(1, max(len(line) for line in rendered_lines))
    widget.styles.height = max(1, len(rendered_lines))
    return widget
