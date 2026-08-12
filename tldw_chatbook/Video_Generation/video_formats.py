"""Canonical generated-video container, MIME, and extension vocabulary."""

from __future__ import annotations


SUPPORTED_VIDEO_FORMATS = frozenset(
    {
        ("mp4", "video/mp4", "mp4"),
        ("webm", "video/webm", "webm"),
    }
)


def normalize_video_mime(value: object) -> str:
    """Normalize an HTTP video MIME without accepting aliases."""
    if not isinstance(value, str):
        raise ValueError("unsupported video MIME")
    return value.split(";", 1)[0].strip().lower()


def canonical_video_extension(container: object) -> str:
    """Return the one canonical extension for a supported container."""
    for known_container, _mime, extension in SUPPORTED_VIDEO_FORMATS:
        if container == known_container:
            return extension
    raise ValueError("unsupported video container")


def video_container_for_mime(value: object) -> str:
    """Return the canonical container identified by an observed MIME."""
    normalized = normalize_video_mime(value)
    for container, mime, _extension in SUPPORTED_VIDEO_FORMATS:
        if normalized == mime:
            return container
    raise ValueError("unsupported video MIME")
