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
DEFAULT_MAX_HISTORY_IMAGES = 10  # used when model capabilities omit max_images

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
    """Validate and process a file attachment (moved intact from ChatAttachmentHandler)."""
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
    """Validate, process, and normalize a file into a PendingAttachment."""
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


def vision_block_reason(provider: str, model: str | None) -> str | None:
    """Return user-facing blocked-send copy when the model can't accept images."""
    if model and is_vision_capable(provider, model):
        return None
    model_label = model or "the selected model"
    return (
        f"Console send blocked: {model_label} can't accept images. "
        "Remove the attachment, switch to a vision model, or mark this model as "
        "vision-capable under [model_capabilities.models] in config.toml."
    )


def max_history_images(provider: str, model: str | None) -> int:
    """Return how many recent session images to resend for this model."""
    if not model:
        return DEFAULT_MAX_HISTORY_IMAGES
    capabilities = _get_capabilities_registry().get_model_capabilities(provider, model)
    value = capabilities.get("max_images")
    if isinstance(value, int) and value > 0:
        return value
    return DEFAULT_MAX_HISTORY_IMAGES


def image_content_parts(
    text: str,
    image_data: bytes,
    mime_type: str,
) -> list[dict[str, Any]]:
    """Build OpenAI-style multimodal content parts with a base64 data URL."""
    encoded = base64.b64encode(image_data).decode("ascii")
    parts: list[dict[str, Any]] = []
    if text:
        parts.append({"type": "text", "text": text})
    parts.append(
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}
    )
    return parts
