"""Detect drag-drop path pastes and grab clipboard images for Console attach.

Terminal reality this module encodes: dropping a file onto a terminal pastes
its PATH as text, and clipboard IMAGES produce no terminal event at all —
they are read explicitly via PIL.ImageGrab (macOS/Windows; unavailable on
most Linux setups).

Pure module: no Textual imports; ``grab_clipboard_image`` is sync/blocking
and callers must run it off the event loop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from fnmatch import fnmatch
from io import BytesIO
from typing import Literal
from urllib.parse import unquote, urlparse

from loguru import logger

from tldw_chatbook.Chat.attachment_core import ATTACHMENT_FILTER_SPECS
from tldw_chatbook.Utils.path_validation import is_safe_path

_SUPPORTED_PATTERNS: tuple[str, ...] = tuple(
    pattern
    for _label, patterns in ATTACHMENT_FILTER_SPECS
    for pattern in patterns.split(";")
)


@dataclass(frozen=True)
class DroppedPaste:
    """A path-paste candidate extracted from pasted text."""

    path: str
    total_dropped: int


@dataclass(frozen=True)
class ClipboardGrab:
    """Result of reading the OS clipboard for attachable content."""

    kind: Literal["image", "paths", "empty", "unavailable"]
    png_bytes: bytes | None = None
    paths: tuple[str, ...] = ()


def _decode_token(token: str) -> str | None:
    """Return the path a single pasted token denotes, or None.

    Args:
        token: One whitespace-trimmed line of pasted text.

    Returns:
        An absolute path string when the token is path-like (quoted,
        backslash-escaped, ``file://`` URI, or plain absolute), else None.
    """
    if not token:
        return None
    if token.startswith("file://"):
        parsed = urlparse(token)
        candidate = unquote(parsed.path)
        return candidate if candidate.startswith("/") else None
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        candidate = token[1:-1]
        return candidate if candidate.startswith("/") else None
    if "\\ " in token:
        candidate = token.replace("\\ ", " ")
        return candidate if candidate.startswith("/") else None
    if token.startswith("/"):
        # A bare absolute path must not contain unescaped spaces — pasted
        # prose like "what does /etc/hosts do?" never reaches here because
        # the caller splits per line and rejects lines with spaces.
        return token if " " not in token else None
    return None


def extract_dropped_path(pasted_text: str) -> DroppedPaste | None:
    """Return the first dropped path when a paste is purely path-like.

    Args:
        pasted_text: Raw text from the terminal paste event.

    Returns:
        The first path plus the total number of dropped paths, or None when
        any line of the paste is not path-like (prose stays text).
    """
    lines = [line.strip() for line in pasted_text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return None
    decoded: list[str] = []
    for line in lines:
        candidate = _decode_token(line)
        if candidate is None:
            return None
        decoded.append(candidate)
    return DroppedPaste(path=decoded[0], total_dropped=len(decoded))


def looks_attachable(path: str, allowed_root: str | None = None) -> bool:
    """Return whether a dropped path is safe and supported for auto-attach.

    Args:
        path: Absolute path extracted from a paste.
        allowed_root: Directory the file must live under; defaults to the
            user's home directory (matching ``attachment_core``).

    Returns:
        True when the file exists, is inside the allowed root, and matches
        the shared picker filter patterns. (Known inherited wart: the specs
        advertise tiff/svg the image pipeline rejects — TASK-222.)
    """
    root = allowed_root or os.path.expanduser("~")
    if not os.path.isfile(path):
        return False
    if not is_safe_path(path, root):
        return False
    name = os.path.basename(path)
    return any(fnmatch(name, pattern) for pattern in _SUPPORTED_PATTERNS)


def _grabclipboard():
    """Read the OS clipboard via PIL.ImageGrab (module seam for tests)."""
    from PIL import ImageGrab

    return ImageGrab.grabclipboard()


def grab_clipboard_image() -> ClipboardGrab:
    """Read the OS clipboard, classifying what it holds.

    Returns:
        ``image`` with PNG bytes for a clipboard image; ``paths`` for
        copied files (e.g. Finder); ``empty`` when the clipboard holds no
        image; ``unavailable`` when the platform/backend cannot be read.
    """
    try:
        grabbed = _grabclipboard()
    except Exception:
        logger.opt(exception=True).info("Clipboard image grab unavailable.")
        return ClipboardGrab(kind="unavailable")
    if grabbed is None:
        return ClipboardGrab(kind="empty")
    if isinstance(grabbed, list):
        return ClipboardGrab(
            kind="paths", paths=tuple(str(item) for item in grabbed)
        )
    buffer = BytesIO()
    grabbed.save(buffer, format="PNG")
    return ClipboardGrab(kind="image", png_bytes=buffer.getvalue())
