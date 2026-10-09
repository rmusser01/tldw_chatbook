"""The Markdown source line a note takes from a Library media item.

TASK-34000.25 (reading-desk design §5.1, "Text is the truth"): a note that
names its source document carries the relationship as ordinary Markdown --
``[<title>](media://<media-uuid>)`` -- so it survives Notes sync, export,
lasting folder sync and Obsidian without a new server domain (ADR-105
unchanged). ``Media.uuid`` is ``UNIQUE NOT NULL`` and stable across
devices, which is why the link names the uuid and never the integer row
id: MCP's existing ``media://<integer-id>`` resources keep their meaning,
and the in-app resolver (slice D1) tells the two apart by shape.

Pure: no I/O, no widget. The Media reader's Note action (``n``) writes this
line as the first line of the new note; slice D5 appends quote attributions
in the same grammar; slice D2's derived index scans for this link form.
"""

from __future__ import annotations

import re

__all__ = ["MediaSourceLinkError", "media_source_line", "media_source_uuid"]

#: RFC 4122 text form, the shape ``Client_Media_DB_v2`` writes to ``Media.uuid``.
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class MediaSourceLinkError(ValueError):
    """The item has no stable identity a portable link may name."""


def media_source_uuid(value: object) -> str:
    """Return ``value`` as the uuid text a source link may carry.

    Args:
        value: The loaded detail's ``uuid`` field, whatever the mapping holds.

    Returns:
        The uuid text, lower-cased, whitespace stripped.

    Raises:
        MediaSourceLinkError: When ``value`` is not a uuid -- including an
            integer row id, which is deliberately refused rather than written
            as ``media://<int>``: a link that silently named a device-local
            row would read as portable and break on every other device.
    """
    if isinstance(value, bool) or not isinstance(value, str):
        raise MediaSourceLinkError("The media item has no stable uuid to link.")
    text = value.strip()
    if not _UUID_RE.match(text):
        raise MediaSourceLinkError("The media item has no stable uuid to link.")
    return text.lower()


def _link_title(title: object) -> str:
    """One-line link text: Markdown's link brackets and newlines escaped."""
    text = " ".join(str(title or "").split())
    if not text:
        text = "Untitled media"
    return text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def media_source_line(title: object, uuid: object) -> str:
    """Render the ``[<title>](media://<media-uuid>)`` source line (§5.1).

    Args:
        title: The document's title as the reader shows it; collapsed to one
            line and bracket-escaped so the link text cannot end the link early.
        uuid: The item's ``Media.uuid``; see :func:`media_source_uuid`.

    Returns:
        The Markdown line, with no trailing newline.

    Raises:
        MediaSourceLinkError: When ``uuid`` is not a stable uuid.
    """
    return f"[{_link_title(title)}](media://{media_source_uuid(uuid)})"
