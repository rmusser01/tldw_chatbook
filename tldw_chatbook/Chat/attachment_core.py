"""Shared, UI-agnostic attachment processing for legacy chat and native Console.

Extracted from ChatAttachmentHandler so both the legacy chat window and the
native Console consume one validation/processing/vision-gating pipeline.
No Textual imports allowed in this module.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from tldw_chatbook.Utils.file_handlers import ProcessedFile, file_handler_registry
from tldw_chatbook.Utils.path_validation import is_safe_path
from tldw_chatbook.model_capabilities import (
    get_model_capabilities as _get_capabilities_registry,
    is_vision_capable,
)

MAX_ATTACHMENT_BYTES = 100 * 1024 * 1024  # matches the legacy handler's 100MB cap
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # matches ChatImageHandler.MAX_IMAGE_SIZE
DEFAULT_MAX_HISTORY_IMAGES = 10  # used when model capabilities omit max_images

DEFAULT_RESIZE_MAX_DIMENSION = 2048  # matches ChatImageHandler's legacy literal

DEFAULT_SUPPORTED_IMAGE_FORMATS: tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".svg",
)

# Non-image picker rows; the image rows are derived at call time by
# attachment_filter_specs(). The "All Supported Files" non-image tail is the
# legacy literal verbatim (it was never the union of the rows below — do not
# "fix" that here).
_ALL_FILES_NON_IMAGE_PATTERNS = (
    "*.txt;*.md;*.log;*.py;*.js;*.ts;*.java;*.cpp;*.c;*.h;*.cs;*.rb;*.go;*.rs;"
    "*.json;*.yaml;*.yml;*.csv;*.tsv;*.pdf;*.doc;*.docx;*.rtf;*.odt;"
    "*.epub;*.mobi;*.azw;*.azw3;*.fb2"
)
_NON_IMAGE_FILTER_SPECS: tuple[tuple[str, str], ...] = (
    ("Document Files", "*.pdf;*.doc;*.docx;*.rtf;*.odt"),
    ("E-book Files", "*.epub;*.mobi;*.azw;*.azw3;*.fb2"),
    ("Text Files", "*.txt;*.md;*.log;*.text;*.rst"),
    ("Code Files", "*.py;*.js;*.ts;*.java;*.cpp;*.c;*.h;*.cs;*.rb;*.go;*.rs;*.swift;*.kt;*.php;*.r;*.m;*.lua;*.sh;*.bash;*.ps1;*.sql;*.html;*.css;*.xml"),
    ("Data Files", "*.json;*.yaml;*.yml;*.csv;*.tsv"),
)

# (label, semicolon-separated glob patterns) — single source for both UIs' pickers.
ATTACHMENT_FILTER_SPECS: tuple[tuple[str, str], ...] = (
    ("All Supported Files", "*.png;*.jpg;*.jpeg;*.gif;*.webp;*.bmp;*.tiff;*.tif;*.svg;*.txt;*.md;*.log;*.py;*.js;*.ts;*.java;*.cpp;*.c;*.h;*.cs;*.rb;*.go;*.rs;*.json;*.yaml;*.yml;*.csv;*.tsv;*.pdf;*.doc;*.docx;*.rtf;*.odt;*.epub;*.mobi;*.azw;*.azw3;*.fb2"),
    ("Image Files", "*.png;*.jpg;*.jpeg;*.gif;*.webp;*.bmp;*.tiff;*.tif;*.svg"),
    ("Document Files", "*.pdf;*.doc;*.docx;*.rtf;*.odt"),
    ("E-book Files", "*.epub;*.mobi;*.azw;*.azw3;*.fb2"),
    ("Text Files", "*.txt;*.md;*.log;*.text;*.rst"),
    ("Code Files", "*.py;*.js;*.ts;*.java;*.cpp;*.c;*.h;*.cs;*.rb;*.go;*.rs;*.swift;*.kt;*.php;*.r;*.m;*.lua;*.sh;*.bash;*.ps1;*.sql;*.html;*.css;*.xml"),
    ("Data Files", "*.json;*.yaml;*.yml;*.csv;*.tsv"),
)


def _format_size(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / 1024 / 1024:.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size} B"


def svg_rendering_available() -> bool:
    """Capability seam for the SVG gate; tests monkeypatch this name."""
    from tldw_chatbook.Utils.optional_deps import ensure_svg_rendering

    return ensure_svg_rendering()


def supported_image_formats() -> tuple[str, ...]:
    """Effective image extension allowlist from [chat.images].supported_formats.

    Entries are normalized (lowercased, dotted, deduped in order); .svg is
    dropped when cairosvg is unavailable. Invalid or empty config values fall
    back to DEFAULT_SUPPORTED_IMAGE_FORMATS.
    """
    from tldw_chatbook.config import get_cli_setting

    raw = get_cli_setting(
        "chat.images", "supported_formats", list(DEFAULT_SUPPORTED_IMAGE_FORMATS)
    )
    formats: list[str] = []
    if isinstance(raw, (list, tuple)):
        for entry in raw:
            if not isinstance(entry, str) or not entry.strip():
                logger.warning(
                    f"[chat.images].supported_formats: ignoring entry {entry!r}"
                )
                continue
            ext = entry.strip().lower()
            if not ext.startswith("."):
                ext = f".{ext}"
            if ext not in formats:
                formats.append(ext)
    if not formats:
        logger.warning(
            "[chat.images].supported_formats invalid or empty; using defaults"
        )
        formats = list(DEFAULT_SUPPORTED_IMAGE_FORMATS)
    if ".svg" in formats and not svg_rendering_available():
        formats.remove(".svg")
    return tuple(formats)


def max_image_bytes() -> int:
    """Image byte cap from [chat.images].max_size_mb (default 10 MB)."""
    from tldw_chatbook.config import get_cli_setting

    raw = get_cli_setting("chat.images", "max_size_mb", MAX_IMAGE_BYTES / (1024 * 1024))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.0
    if value <= 0:
        logger.warning(f"[chat.images].max_size_mb invalid ({raw!r}); using 10.0")
        return MAX_IMAGE_BYTES
    return int(value * 1024 * 1024)


def image_resize_max_dimension() -> int:
    """Resize bound from [chat.images].resize_max_dimension (default 2048)."""
    from tldw_chatbook.config import get_cli_setting

    raw = get_cli_setting(
        "chat.images", "resize_max_dimension", DEFAULT_RESIZE_MAX_DIMENSION
    )
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 0
    if value <= 0:
        logger.warning(
            f"[chat.images].resize_max_dimension invalid ({raw!r}); using "
            f"{DEFAULT_RESIZE_MAX_DIMENSION}"
        )
        return DEFAULT_RESIZE_MAX_DIMENSION
    return value


def attachment_filter_specs() -> tuple[tuple[str, str], ...]:
    """Picker filter rows with image patterns derived from the effective formats."""
    image_patterns = ";".join(f"*{ext}" for ext in supported_image_formats())
    return (
        ("All Supported Files", f"{image_patterns};{_ALL_FILES_NON_IMAGE_PATTERNS}"),
        ("Image Files", image_patterns),
        *_NON_IMAGE_FILTER_SPECS,
    )


@dataclass
class PendingAttachment:
    """One processed, not-yet-sent attachment staged on a chat session."""

    file_path: str
    display_name: str
    file_type: str
    insert_mode: Literal["inline", "attachment"]
    data: bytes | None = None
    mime_type: str | None = None
    text_content: str | None = None
    original_size: int = 0
    processed_size: int = 0

    @property
    def label(self) -> str:
        """Return the user-facing chip/indicator label."""
        size = self.processed_size or self.original_size
        return f"{self.display_name} · {_format_size(size)}"


async def load_processed_file(
    file_path: str,
    *,
    allowed_root: str | None = None,
) -> ProcessedFile:
    """Validate and process a file attachment (moved intact from ChatAttachmentHandler).

    Args:
        file_path: Path to the file to validate and process.
        allowed_root: Directory the file must live under; defaults to the
            user's home directory.

    Returns:
        The registry's ProcessedFile for the dispatched handler.

    Raises:
        ValueError: If the path is outside the allowed root or the file
            exceeds MAX_ATTACHMENT_BYTES.
        FileNotFoundError: If the file does not exist.
    """
    root = allowed_root or os.path.expanduser("~")
    logger.info(f"Processing file attachment: {file_path}")
    if not is_safe_path(file_path, root):
        raise ValueError("File path is outside allowed directories")
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    file_size = os.path.getsize(file_path)
    if file_size > MAX_ATTACHMENT_BYTES:
        raise ValueError(
            f"File too large: {file_size / 1024 / 1024:.1f}MB "
            f"(max {MAX_ATTACHMENT_BYTES / 1024 / 1024:.0f}MB)"
        )
    return await file_handler_registry.process_file(file_path)


async def process_attachment_path(
    file_path: str,
    *,
    allowed_root: str | None = None,
) -> PendingAttachment:
    """Validate, process, and normalize a file into a PendingAttachment.

    Args:
        file_path: Path to the file to attach.
        allowed_root: Directory the file must live under; defaults to the
            user's home directory.

    Returns:
        A PendingAttachment carrying either binary attachment data (image
        files) or inline text content (text/code/data files).

    Raises:
        ValueError: If the path is outside the allowed root, the file is
            too large, or the format is unsupported by its handler.
        FileNotFoundError: If the file does not exist.
    """
    processed = await load_processed_file(file_path, allowed_root=allowed_root)
    data = processed.attachment_data
    return PendingAttachment(
        file_path=str(file_path),
        display_name=processed.display_name or Path(file_path).name,
        file_type=processed.file_type,
        insert_mode=processed.insert_mode,
        data=data,
        mime_type=processed.attachment_mime_type,
        text_content=processed.content,
        original_size=os.path.getsize(file_path),
        processed_size=len(data) if data is not None else len(processed.content or ""),
    )


async def process_attachment_bytes(
    data: bytes,
    *,
    display_name: str,
    mime_type: str = "image/png",
) -> PendingAttachment:
    """Build an image PendingAttachment from raw bytes (clipboard path).

    Runs the same validate/resize pipeline as file-based image attachments
    (10 MB cap, PIL validation, resize via ChatImageHandler) with no temp
    files.

    Args:
        data: Raw image bytes (e.g. PNG-encoded clipboard grab).
        display_name: User-facing name (e.g. ``clipboard-20260713-120000.png``).
        mime_type: MIME type of ``data``.

    Returns:
        An attachment-mode image PendingAttachment with ``file_path=""``.

    Raises:
        ValueError: If the bytes exceed the image cap or are not a valid image.
    """
    from io import BytesIO

    from PIL import Image as PILImage

    from tldw_chatbook.Event_Handlers.Chat_Events.chat_image_events import (
        PAYLOAD_FORMAT_MIME,
        ChatImageHandler,
    )

    size_cap = max_image_bytes()
    if len(data) > size_cap:
        raise ValueError(
            f"Image too large ({len(data) / 1024 / 1024:.1f}MB). "
            f"Maximum size: {size_cap / 1024 / 1024:.0f}MB"
        )
    try:
        probe = PILImage.open(BytesIO(data))
        probe.verify()
        probed_format = (probe.format or "").upper()
    except Exception as exc:
        raise ValueError("Clipboard data is not a valid image.") from exc
    extension = ".png" if "png" in mime_type else ".jpg"
    try:
        processed, mime_type = await ChatImageHandler.prepare_image_payload(
            data, extension
        )
    except Exception:
        logger.opt(exception=True).warning(
            "Failed to process clipboard image data, using original bytes."
        )
        processed = data
        mime_type = PAYLOAD_FORMAT_MIME.get(probed_format, mime_type)
    return PendingAttachment(
        file_path="",
        display_name=display_name,
        file_type="image",
        insert_mode="attachment",
        data=processed,
        mime_type=mime_type,
        text_content=None,
        original_size=len(data),
        processed_size=len(processed),
    )


def vision_block_reason(provider: str, model: str | None) -> str | None:
    """Return user-facing blocked-send copy when the model can't accept images.

    Args:
        provider: Provider key for the capability lookup (e.g. "llama_cpp").
        model: Selected model identifier, or None when no model is chosen.

    Returns:
        None when the model is vision-capable; otherwise the blocked-send
        copy, which names the model and the [model_capabilities.models]
        config override escape hatch.
    """
    if model and is_vision_capable(provider, model):
        return None
    model_label = model or "the selected model"
    return (
        f"Console send blocked: {model_label} can't accept images. "
        "Remove the attachment, switch to a vision model, or mark this model as "
        "vision-capable under [model_capabilities.models] in config.toml."
    )


def max_history_images(provider: str, model: str | None) -> int:
    """Return how many recent session images to resend for this model.

    Args:
        provider: Provider key for the capability lookup.
        model: Selected model identifier, or None when no model is chosen.

    Returns:
        The model's max_images capability when present and positive,
        otherwise DEFAULT_MAX_HISTORY_IMAGES.
    """
    if not model:
        return DEFAULT_MAX_HISTORY_IMAGES
    capabilities = _get_capabilities_registry().get_model_capabilities(provider, model)
    value = capabilities.get("max_images")
    if isinstance(value, int) and value > 0:
        return value
    return DEFAULT_MAX_HISTORY_IMAGES


def image_url_part(image_data: bytes, mime_type: str) -> dict[str, Any]:
    """Build one OpenAI-style image_url content part (base64 data URL).

    Args:
        image_data: Raw image bytes.
        mime_type: MIME type for the data URL.

    Returns:
        A single ``{"type": "image_url", ...}`` content part dict.
    """
    encoded = base64.b64encode(image_data).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }


def image_content_parts(
    text: str,
    image_data: bytes,
    mime_type: str,
) -> list[dict[str, Any]]:
    """Build OpenAI-style multimodal content parts with a base64 data URL.

    Args:
        text: Message text; emitted as a leading text part when non-empty.
        image_data: Raw image bytes to encode.
        mime_type: MIME type used in the data URL (e.g. "image/png").

    Returns:
        Content-part dicts: an optional {"type": "text"} part followed by
        one {"type": "image_url"} part with a data URL.
    """
    parts: list[dict[str, Any]] = []
    if text:
        parts.append({"type": "text", "text": text})
    parts.append(image_url_part(image_data, mime_type))
    return parts
