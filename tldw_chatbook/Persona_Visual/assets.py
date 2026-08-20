"""Bounded, path-safe loading for profile-owned Persona Visual rasters."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath

from PIL import Image

from .contracts import (
    ALLOWED_ASSET_MIME_TYPES,
    ALLOWED_ASSET_ROLES,
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
MAX_ASSET_DECODED_PIXELS = MAX_ASSET_DIMENSION**2 * 4


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
        normalized: list[PersonaVisualAssetMetadata] = []
        asset_keys: set[str] = set()
        total_bytes = 0
        iterator = iter(assets)
        for _index in range(MAX_ASSET_COUNT + 1):
            try:
                asset = _metadata(next(iterator))
            except StopIteration:
                break
            normalized.append(asset)
            if asset.asset_key in asset_keys:
                raise ValueError
            asset_keys.add(asset.asset_key)
            total_bytes += asset.byte_count
            if total_bytes > MAX_ASSET_TOTAL_BYTES:
                raise ValueError
        if not normalized or len(normalized) > MAX_ASSET_COUNT:
            raise ValueError
        return tuple(normalized)
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
    if type(value) is not PersonaVisualAssetMetadata:
        raise ValueError
    values = (
        value.asset_key,
        value.role,
        value.mime_type,
        value.sha256,
    )
    if any(type(item) is not str for item in values):
        raise ValueError
    for item in values:
        item.encode("utf-8")
    if (
        _ASSET_KEY.fullmatch(value.asset_key) is None
        or value.role not in ALLOWED_ASSET_ROLES
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
    if type(storage_key) is not str or not storage_key or "\x00" in storage_key:
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
    root = os.fspath(profile_root)
    if type(root) is not str or "\x00" in root:
        raise ValueError
    root_path = Path(root)
    if not root_path.is_absolute() or str(root_path) != root:
        raise ValueError
    if _supports_secure_descriptor_walk():
        return _read_profile_file_secure(root, parts, expected_bytes)
    return _read_profile_file_fallback(root_path, parts, expected_bytes)


def _supports_secure_descriptor_walk() -> bool:
    return (
        os.name == "posix"
        and getattr(os, "O_NOFOLLOW", 0) > 0
        and getattr(os, "O_DIRECTORY", 0) > 0
        and getattr(os, "O_NONBLOCK", 0) > 0
        and os.open in getattr(os, "supports_dir_fd", ())
        and os.stat in getattr(os, "supports_dir_fd", ())
        and os.stat in getattr(os, "supports_follow_symlinks", ())
    )


def _read_profile_file_secure(
    root: str,
    parts: tuple[str, ...],
    expected_bytes: int,
) -> bytes:
    nofollow = os.O_NOFOLLOW
    directory = os.O_DIRECTORY
    nonblock = os.O_NONBLOCK
    flags = os.O_RDONLY | nofollow
    directory_flags = flags | directory
    opened: list[int] = []
    try:
        current = _open_profile_root(root, directory_flags)
        opened.append(current)
        for component in parts[:-1]:
            current = os.open(component, directory_flags, dir_fd=current)
            opened.append(current)
        file_fd = os.open(parts[-1], flags | nonblock, dir_fd=current)
        opened.append(file_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_bytes:
            raise ValueError

        data = _read_fd_bounded(file_fd, expected_bytes)
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


def _read_profile_file_fallback(
    root: Path,
    parts: tuple[str, ...],
    expected_bytes: int,
) -> bytes:
    candidate = root.joinpath(*parts)
    snapshot = _snapshot_directories(_fallback_directories(root, parts))
    leaf_before = os.lstat(candidate)
    if not stat.S_ISREG(leaf_before.st_mode) or leaf_before.st_size != expected_bytes:
        raise ValueError
    _verify_directory_snapshot(snapshot)

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    nonblock = getattr(os, "O_NONBLOCK", 0)
    if isinstance(nonblock, int) and nonblock > 0:
        flags |= nonblock
    descriptor = os.open(candidate, flags)
    try:
        opened = os.fstat(descriptor)
        _verify_fallback_leaf(candidate, leaf_before, opened)
        _verify_directory_snapshot(snapshot)
        data = _read_fd_bounded(descriptor, expected_bytes)
        after = os.fstat(descriptor)
        _verify_fallback_leaf(candidate, leaf_before, after)
        _verify_directory_snapshot(snapshot)
        return data
    finally:
        os.close(descriptor)


def _fallback_directories(root: Path, parts: tuple[str, ...]) -> tuple[Path, ...]:
    directories = [*reversed(root.parents), root]
    current = root
    for component in parts[:-1]:
        current /= component
        directories.append(current)
    return tuple(directories)


def _snapshot_directories(
    directories: tuple[Path, ...],
) -> tuple[tuple[Path, int, int], ...]:
    snapshot: list[tuple[Path, int, int]] = []
    for directory in directories:
        metadata = os.lstat(directory)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError
        snapshot.append((directory, metadata.st_dev, metadata.st_ino))
    return tuple(snapshot)


def _verify_directory_snapshot(snapshot: tuple[tuple[Path, int, int], ...]) -> None:
    for directory, device, inode in snapshot:
        metadata = os.lstat(directory)
        if not stat.S_ISDIR(metadata.st_mode) or (
            metadata.st_dev,
            metadata.st_ino,
        ) != (device, inode):
            raise ValueError


def _verify_fallback_leaf(
    candidate: Path,
    expected: os.stat_result,
    opened: os.stat_result,
) -> None:
    named = os.lstat(candidate)
    identity = (expected.st_dev, expected.st_ino, expected.st_size)
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(named.st_mode)
        or (opened.st_dev, opened.st_ino, opened.st_size) != identity
        or (named.st_dev, named.st_ino, named.st_size) != identity
    ):
        raise ValueError


def _read_fd_bounded(descriptor: int, expected_bytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = expected_bytes + 1
    while remaining:
        chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
    if len(data) != expected_bytes:
        raise ValueError
    return data


def _open_profile_root(root: str, directory_flags: int) -> int:
    components = Path(root).parts[1:]

    current = os.open(os.sep, directory_flags)
    try:
        for component in components:
            child = os.open(component, directory_flags, dir_fd=current)
            os.close(current)
            current = child
        return current
    except Exception:
        os.close(current)
        raise


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
        if image.width * image.height * frame_count > MAX_ASSET_DECODED_PIXELS:
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
