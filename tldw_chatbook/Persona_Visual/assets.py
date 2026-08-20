"""Bounded, path-safe loading for profile-owned Persona Visual rasters."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath

from PIL import Image

from .contracts import (
    ALLOWED_ASSET_MIME_TYPES,
    MAX_ASSET_COUNT,
    MAX_ASSET_DIMENSION,
    MAX_ASSET_TOTAL_BYTES,
    MAX_FRAME_DURATION_MS,
    MAX_FRAMES_PER_ANIMATION,
)


ASSET_INVALID_REASON = "persona_visual_asset_invalid"

_ASSET_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FORMATS = {
    "image/png": ("PNG", frozenset({".png"})),
    "image/jpeg": ("JPEG", frozenset({".jpg", ".jpeg"})),
    "image/webp": ("WEBP", frozenset({".webp"})),
    "image/gif": ("GIF", frozenset({".gif"})),
}
_WINDOWS_DEVICES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)
_READ_CHUNK_BYTES = 64 * 1024
_MAX_STORAGE_KEY_BYTES = 1024


class PersonaVisualAssetError(ValueError):
    """A stable, path-free failure at the Persona Visual asset boundary."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(ASSET_INVALID_REASON)


@dataclass(frozen=True, slots=True)
class PersonaVisualAssetMetadata:
    """Immutable path-free metadata for one version-bound raster."""

    asset_key: str
    role: str
    mime_type: str
    byte_count: int
    sha256: str
    width: int
    height: int
    frame_count: int | None = None
    duration_ms: int | None = None


@dataclass(frozen=True, slots=True)
class PersonaVisualAsset:
    """One validated raster and the frame decoded during admission."""

    metadata: PersonaVisualAssetMetadata
    data: bytes
    selected_frame: int


def validate_persona_visual_asset_set(
    assets: Sequence[PersonaVisualAssetMetadata],
) -> tuple[PersonaVisualAssetMetadata, ...]:
    """Return normalized immutable metadata after enforcing pack-wide budgets."""

    try:
        if not isinstance(assets, Sequence) or isinstance(assets, (str, bytes)):
            raise ValueError
        if not assets or len(assets) > MAX_ASSET_COUNT:
            raise ValueError
        normalized = tuple(_metadata(asset) for asset in assets)
        if len({asset.asset_key for asset in normalized}) != len(normalized):
            raise ValueError
        if sum(asset.byte_count for asset in normalized) > MAX_ASSET_TOTAL_BYTES:
            raise ValueError
        return normalized
    except Exception:
        raise PersonaVisualAssetError() from None


def load_persona_visual_asset(
    profile_root: os.PathLike[str] | str,
    *,
    storage_key: str,
    metadata: PersonaVisualAssetMetadata,
    selected_frame: int = 0,
) -> PersonaVisualAsset:
    """Safely read and decode one declared frame without exposing a local path."""

    try:
        normalized = validate_persona_visual_asset_set((metadata,))[0]
        parts = _storage_parts(storage_key, normalized.mime_type)
        if (
            type(selected_frame) is not int
            or selected_frame < 0
            or selected_frame >= MAX_FRAMES_PER_ANIMATION
        ):
            raise ValueError
        data = _read_profile_file(profile_root, parts, normalized.byte_count)
        if hashlib.sha256(data).hexdigest() != normalized.sha256:
            raise ValueError
        _decode_selected_frame(data, normalized, selected_frame)
        return PersonaVisualAsset(normalized, data, selected_frame)
    except PersonaVisualAssetError:
        raise
    except Exception:
        raise PersonaVisualAssetError() from None


def _metadata(value: object) -> PersonaVisualAssetMetadata:
    if not isinstance(value, PersonaVisualAssetMetadata):
        raise ValueError
    values = (
        value.asset_key,
        value.role,
        value.mime_type,
        value.sha256,
    )
    if any(not isinstance(item, str) for item in values):
        raise ValueError
    for item in values:
        item.encode("utf-8")
    if (
        _ASSET_KEY.fullmatch(value.asset_key) is None
        or value.role != "sprite"
        or value.mime_type not in ALLOWED_ASSET_MIME_TYPES
        or value.mime_type not in _FORMATS
        or _SHA256.fullmatch(value.sha256) is None
        or not _positive_int(value.byte_count)
        or value.byte_count > MAX_ASSET_TOTAL_BYTES
        or not _positive_int(value.width)
        or not _positive_int(value.height)
        or value.width > MAX_ASSET_DIMENSION
        or value.height > MAX_ASSET_DIMENSION
        or (
            value.frame_count is not None
            and (
                not _positive_int(value.frame_count)
                or value.frame_count > MAX_FRAMES_PER_ANIMATION
            )
        )
        or (
            value.duration_ms is not None
            and (
                not _positive_int(value.duration_ms)
                or value.duration_ms > MAX_FRAME_DURATION_MS
            )
        )
    ):
        raise ValueError
    return PersonaVisualAssetMetadata(
        asset_key=value.asset_key,
        role=value.role,
        mime_type=value.mime_type,
        byte_count=value.byte_count,
        sha256=value.sha256,
        width=value.width,
        height=value.height,
        frame_count=value.frame_count,
        duration_ms=value.duration_ms,
    )


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _storage_parts(storage_key: object, mime_type: str) -> tuple[str, ...]:
    if not isinstance(storage_key, str) or not storage_key or "\x00" in storage_key:
        raise ValueError
    storage_key.encode("utf-8")
    if len(storage_key.encode("utf-8")) > _MAX_STORAGE_KEY_BYTES or "\\" in storage_key:
        raise ValueError
    path = PurePosixPath(storage_key)
    parts = tuple(storage_key.split("/"))
    if (
        path.is_absolute()
        or path.as_posix() != storage_key
        or any(part in {"", ".", ".."} for part in parts)
        or (len(parts[0]) >= 2 and parts[0][1] == ":")
    ):
        raise ValueError
    for part in parts:
        if len(part.encode("utf-8")) > 255:
            raise ValueError
        device_name = part.rstrip(" .").split(".", 1)[0].upper()
        if device_name in _WINDOWS_DEVICES:
            raise ValueError
    extensions = _FORMATS[mime_type][1]
    if path.suffix.lower() not in extensions:
        raise ValueError
    return parts


def _read_profile_file(
    profile_root: os.PathLike[str] | str,
    parts: tuple[str, ...],
    expected_bytes: int,
) -> bytes:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory:
        raise ValueError
    root = os.fspath(profile_root)
    if not isinstance(root, str) or "\x00" in root:
        raise ValueError
    flags = os.O_RDONLY | nofollow
    directory_flags = flags | directory
    opened: list[int] = []
    try:
        current = os.open(root, directory_flags)
        opened.append(current)
        for component in parts[:-1]:
            current = os.open(component, directory_flags, dir_fd=current)
            opened.append(current)
        file_fd = os.open(parts[-1], flags, dir_fd=current)
        opened.append(file_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
            raise ValueError

        chunks: list[bytes] = []
        remaining = expected_bytes + 1
        while remaining:
            chunk = os.read(file_fd, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(file_fd)
        final = os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
        identity = (before.st_dev, before.st_ino, before.st_size)
        if (
            len(data) != expected_bytes
            or (after.st_dev, after.st_ino, after.st_size) != identity
            or (final.st_dev, final.st_ino, final.st_size) != identity
            or not stat.S_ISREG(final.st_mode)
        ):
            raise ValueError
        return data
    finally:
        for descriptor in reversed(opened):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _decode_selected_frame(
    data: bytes,
    metadata: PersonaVisualAssetMetadata,
    selected_frame: int,
) -> None:
    with Image.open(BytesIO(data)) as image:
        expected_format = _FORMATS[metadata.mime_type][0]
        if image.format != expected_format:
            raise ValueError
        if image.width != metadata.width or image.height != metadata.height:
            raise ValueError
        if image.width > MAX_ASSET_DIMENSION or image.height > MAX_ASSET_DIMENSION:
            raise ValueError
        frame_count = int(getattr(image, "n_frames", 1))
        if frame_count < 1 or frame_count > MAX_FRAMES_PER_ANIMATION:
            raise ValueError
        if metadata.frame_count is not None and frame_count != metadata.frame_count:
            raise ValueError
        if selected_frame >= frame_count:
            raise ValueError

        duration_ms = 0
        for frame_index in range(frame_count):
            image.seek(frame_index)
            duration = image.info.get("duration", 0)
            if type(duration) is not int or duration < 0:
                raise ValueError
            duration_ms += duration
            if duration_ms > MAX_FRAME_DURATION_MS:
                raise ValueError
        if metadata.duration_ms is not None and duration_ms != metadata.duration_ms:
            raise ValueError

        image.seek(selected_frame)
        image.load()


__all__ = [
    "ASSET_INVALID_REASON",
    "PersonaVisualAsset",
    "PersonaVisualAssetError",
    "PersonaVisualAssetMetadata",
    "load_persona_visual_asset",
    "validate_persona_visual_asset_set",
]
