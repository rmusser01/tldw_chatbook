"""Canonical generated-video container, MIME, and extension vocabulary."""

from __future__ import annotations


SUPPORTED_VIDEO_FORMATS = frozenset(
    {
        ("mp4", "video/mp4", "mp4"),
        ("webm", "video/webm", "webm"),
    }
)


def normalize_video_mime(value: object) -> str:
    """Normalize an HTTP video MIME without accepting aliases.

    Args:
        value: Observed MIME value.

    Returns:
        The lowercase MIME without parameters.

    Raises:
        ValueError: If the value is not a string.
    """
    if not isinstance(value, str):
        raise ValueError("unsupported video MIME")
    return str.__str__(value).split(";", 1)[0].strip().lower()


def canonical_video_extension(container: object) -> str:
    """Return the canonical extension for a supported container.

    Args:
        container: Canonical container name, including string-enum values.

    Returns:
        The canonical filename extension.

    Raises:
        ValueError: If the container is not supported.
    """
    if not isinstance(container, str):
        raise ValueError("unsupported video container")
    container = str.__str__(container)
    for known_container, _mime, extension in SUPPORTED_VIDEO_FORMATS:
        if container == known_container:
            return extension
    raise ValueError("unsupported video container")


def video_container_for_mime(value: object) -> str:
    """Return the canonical container identified by an observed MIME.

    Args:
        value: Observed MIME value.

    Returns:
        The canonical container name.

    Raises:
        ValueError: If the MIME is malformed or unsupported.
    """
    normalized = normalize_video_mime(value)
    for container, mime, _extension in SUPPORTED_VIDEO_FORMATS:
        if normalized == mime:
            return container
    raise ValueError("unsupported video MIME")
