"""Bounded immutable metadata/art sources; downloaded text is never executable."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from PIL import Image

from tldw_chatbook.Character_Chat.artwork_attribution import (
    MAX_NOTICE_BYTES,
    artwork_context,
)
from tldw_chatbook.Utils.filesystem_identity import directory_identity_from_stat

MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_PACKAGE_BYTES = 32 * 1024 * 1024
MAX_MEMBERS = 128
_READ_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class PetdexSource:
    title: str
    description: str
    metadata_json: str
    image_bytes: bytes = field(repr=False)
    image_name: str
    source_sha256: str
    artwork: Mapping[str, Any]
    _guard: Callable[[], bool] = field(repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "artwork", MappingProxyType(dict(self.artwork)))

    def is_current(self) -> bool:
        try:
            return self._guard() is True
        except Exception:  # noqa: BLE001 - a source guard must fail closed
            return False


def _invalid() -> ValueError:
    return ValueError("petdex_source_invalid")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _invalid()
        result[key] = value
    return result


def _reject_constant(value):
    raise _invalid()


def read_metadata(data: bytes) -> dict[str, Any]:
    """Parse bounded JSON without duplicate keys or non-JSON numbers."""
    if type(data) is not bytes or not 0 < len(data) <= MAX_METADATA_BYTES:
        raise _invalid()
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique,
            parse_constant=_reject_constant,
        )
    except (ValueError, UnicodeError, RecursionError):
        raise _invalid() from None
    if type(value) is not dict:
        raise _invalid()
    return value


def _text(value: Any, limit: int, *, empty: bool = False) -> str:
    if (
        type(value) is not str
        or (not empty and not value.strip())
        or len(value.encode("utf-8")) > limit
    ):
        raise _invalid()
    if any(ord(c) < 32 and c not in "\n\r\t" for c in value):
        raise _invalid()
    return value


def _relative(name: str) -> str:
    if type(name) is not str or not name or "\\" in name or "\x00" in name:
        raise _invalid()
    parts = name.split("/")
    if (
        any(part in {"", ".", ".."} for part in parts)
        or PurePosixPath(name).is_absolute()
    ):
        raise _invalid()
    if len(name.encode("utf-8")) > 512 or any(
        ord(c) < 32 or ord(c) == 127 for c in name
    ):
        raise _invalid()
    return name


def _declared_image(document: Mapping[str, Any]) -> str | None:
    values = [
        _relative(document[key])
        for key in ("spritesheet", "spritesheetPath")
        if key in document
    ]
    if len(set(values)) > 1:
        raise ValueError("petdex_sprite_ambiguous")
    return values[0] if values else None


def source_from_bytes(
    metadata: bytes,
    image: bytes,
    image_name: str,
    *,
    registry_entry: Mapping[str, Any] | None = None,
    notices: str = "",
    guard: Callable[[], bool] = lambda: True,
) -> PetdexSource:
    """Validate supplied art and metadata without fetching any embedded links."""
    document = read_metadata(metadata)
    registry = dict(registry_entry or {})
    _relative(image_name)
    declared = _declared_image(document)
    if declared is not None and declared != image_name:
        raise _invalid()
    title = _text(
        document.get("displayName")
        or document.get("name")
        or registry.get("displayName")
        or document.get("id"),
        256,
    )
    description = _text(document.get("description", ""), 2000, empty=True)
    version = document.get(
        "spriteVersionNumber", registry.get("spriteVersionNumber", 1)
    )
    if type(version) is not int or version not in (1, 2):
        raise _invalid()
    if "spriteVersionNumber" in registry and registry["spriteVersionNumber"] != version:
        raise _invalid()
    document["spriteVersionNumber"] = version
    if type(image) is not bytes or not 0 < len(image) <= MAX_IMAGE_BYTES:
        raise _invalid()
    try:
        with Image.open(io.BytesIO(image)) as raster:
            if (
                raster.format not in {"PNG", "WEBP"}
                or getattr(raster, "n_frames", 1) != 1
                or not 0 < raster.width <= 4096
                or not 0 < raster.height <= 4096
            ):
                raise _invalid()
            if (
                Path(image_name).suffix.lower()
                != {"PNG": ".png", "WEBP": ".webp"}[raster.format]
            ):
                raise _invalid()
            raster.load()
    except (OSError, SyntaxError, Image.DecompressionBombError):
        raise _invalid() from None
    combined = document.get("notices", "")
    combined = _text(combined, MAX_NOTICE_BYTES, empty=True)
    extras = [
        document.get("terms", ""),
        registry.get("terms", ""),
        registry.get("notices", ""),
        notices,
    ]
    for extra in extras:
        _text(extra, MAX_NOTICE_BYTES, empty=True)
    credits = {
        "creator": document.get("creator")
        or document.get("author")
        or registry.get("creator")
        or registry.get("submittedBy"),
        "license": document.get("license") or registry.get("license"),
        "source_url": registry.get("sourceUrl")
        or document.get("source_url")
        or document.get("sourceUrl"),
    }
    # Validate even non-selected statements and retain differing provenance.
    # Registry submitters may be uploaders rather than the original artists.
    candidates = (
        ("creator", "Metadata creator", document.get("creator")),
        ("creator", "Metadata author", document.get("author")),
        ("creator", "Registry creator", registry.get("creator")),
        ("creator", "Registry submitter", registry.get("submittedBy")),
        ("license", "Metadata license", document.get("license")),
        ("license", "Registry license", registry.get("license")),
        ("source_url", "Metadata source", document.get("source_url")),
        ("source_url", "Metadata source", document.get("sourceUrl")),
        ("source_url", "Registry source", registry.get("sourceUrl")),
    )
    for key, label, value in candidates:
        artwork_context(
            {
                "tldw/artwork": {
                    "version": 1,
                    "creator": None,
                    "license": None,
                    "source_url": None,
                    "notices": "",
                    key: value,
                }
            }
        )
        if value and value != credits[key]:
            extras.append(f"{label}: {value}")
    combined = "\n\n".join(value for value in (combined, *extras) if value)
    record = artwork_context(
        {
            "tldw/artwork": {
                "version": 1,
                **credits,
                "notices": combined,
            }
        }
    )["tldw/artwork"]
    canonical = json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    material = json.dumps(
        {
            "metadata": document,
            "image": hashlib.sha256(image).hexdigest(),
            "artwork": record,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    source = PetdexSource(
        title,
        description,
        canonical,
        image,
        image_name,
        hashlib.sha256(material).hexdigest(),
        record,
        guard,
    )
    if not source.is_current():
        raise ValueError("petdex_source_stale")
    return source


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read_fd(fd: int, cap: int) -> tuple[bytes, tuple]:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= cap:
        raise _invalid()
    chunks, total = [], 0
    while True:
        data = os.read(fd, min(_READ_CHUNK_BYTES, cap + 1 - total))
        if not data:
            break
        chunks.append(data)
        total += len(data)
        if total > cap:
            raise _invalid()
    if _identity(os.fstat(fd)) != _identity(before):
        raise ValueError("petdex_source_stale")
    return b"".join(chunks), _identity(before)


def _supports_secure_descriptor_walk() -> bool:
    """Return whether this runtime supports the complete no-follow walk."""
    return (
        os.name == "posix"
        and getattr(os, "O_NOFOLLOW", 0) > 0
        and getattr(os, "O_DIRECTORY", 0) > 0
        and getattr(os, "O_NONBLOCK", 0) > 0
        and os.open in getattr(os, "supports_dir_fd", ())
        and os.stat in getattr(os, "supports_dir_fd", ())
        and os.stat in getattr(os, "supports_follow_symlinks", ())
        and os.scandir in getattr(os, "supports_fd", ())
    )


def _read_path(path: Path, cap: int) -> tuple[bytes, tuple]:
    if not _supports_secure_descriptor_walk():
        return _read_path_fallback(path, cap)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        return _read_fd(fd, cap)
    finally:
        os.close(fd)


def _folder_file(root: Path, name: str, cap: int) -> tuple[bytes, tuple]:
    parts = _relative(name).split("/")
    if not _supports_secure_descriptor_walk():
        return _read_path_fallback(root.joinpath(*parts), cap)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            next_fd = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = next_fd
        child = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
        )
        try:
            return _read_fd(child, cap)
        finally:
            os.close(child)
    finally:
        os.close(fd)


def _fallback_directories(path: Path) -> tuple[Path, ...]:
    parent = path.parent
    return (*reversed(parent.parents), parent)


def _snapshot_directories(
    directories: tuple[Path, ...],
) -> tuple[tuple[Path, tuple], ...]:
    snapshot = []
    for directory in directories:
        info = os.lstat(directory)
        if not stat.S_ISDIR(info.st_mode) or directory_identity_from_stat(info).reparse:
            raise _invalid()
        snapshot.append((directory, _identity(info)))
    return tuple(snapshot)


def _verify_directory_snapshot(snapshot: tuple[tuple[Path, tuple], ...]) -> None:
    for directory, identity in snapshot:
        info = os.lstat(directory)
        if (
            not stat.S_ISDIR(info.st_mode)
            or directory_identity_from_stat(info).reparse
            or _identity(info) != identity
        ):
            raise ValueError("petdex_source_stale")


def _verify_fallback_leaf(path: Path, expected: tuple, opened: os.stat_result) -> None:
    named = os.lstat(path)
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(named.st_mode)
        or directory_identity_from_stat(named).reparse
        or _identity(opened) != expected
        or _identity(named) != expected
    ):
        raise ValueError("petdex_source_stale")


def _read_path_fallback(path: Path, cap: int) -> tuple[bytes, tuple]:
    """Read a regular file with lstat/open/recheck on limited runtimes."""
    snapshot = _snapshot_directories(_fallback_directories(path))
    before = os.lstat(path)
    expected = _identity(before)
    if (
        not stat.S_ISREG(before.st_mode)
        or directory_identity_from_stat(before).reparse
        or not 0 <= before.st_size <= cap
    ):
        raise _invalid()
    _verify_directory_snapshot(snapshot)

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    nonblock = getattr(os, "O_NONBLOCK", 0)
    if isinstance(nonblock, int) and nonblock > 0:
        flags |= nonblock
    fd = os.open(path, flags)
    try:
        _verify_fallback_leaf(path, expected, os.fstat(fd))
        _verify_directory_snapshot(snapshot)
        data, opened_identity = _read_fd(fd, cap)
        if opened_identity != expected:
            raise ValueError("petdex_source_stale")
        _verify_fallback_leaf(path, expected, os.fstat(fd))
        _verify_directory_snapshot(snapshot)
        return data, expected
    finally:
        os.close(fd)


def _image_name(document: Mapping, names: set[str]) -> str:
    declared = _declared_image(document)
    if declared is not None:
        selected = _relative(declared)
        if selected not in names or Path(selected).suffix.lower() not in {
            ".png",
            ".webp",
        }:
            raise _invalid()
        return selected
    candidates = names & {"spritesheet.png", "spritesheet.webp"}
    if len(candidates) != 1:
        raise ValueError("petdex_sprite_ambiguous")
    return next(iter(candidates))


def _notice(name: str) -> bool:
    return Path(name).name.upper().split(".")[0] in {"LICENSE", "NOTICE", "COPYING"}


def _folder_inventory(root: Path) -> dict[str, tuple]:
    """Pin a bounded tree without following links, including nested notices."""
    if not _supports_secure_descriptor_walk():
        return _folder_inventory_fallback(root)
    entries = {}
    total_bytes = 0

    def visit(fd, prefix, depth):
        nonlocal total_bytes
        if depth > 16:
            raise _invalid()
        with os.scandir(fd) as children:
            for child in children:
                if len(entries) >= MAX_MEMBERS:
                    raise _invalid()
                name = _relative(prefix + child.name)
                info = child.stat(follow_symlinks=False)
                directory = stat.S_ISDIR(info.st_mode)
                if not directory and not stat.S_ISREG(info.st_mode):
                    raise _invalid()
                entries[name] = (_identity(info), directory)
                if directory:
                    nested = os.open(
                        child.name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=fd,
                    )
                    try:
                        if _identity(os.fstat(nested)) != _identity(info):
                            raise ValueError("petdex_source_stale")
                        visit(nested, name + "/", depth + 1)
                    finally:
                        os.close(nested)
                else:
                    total_bytes += info.st_size
                    if total_bytes > MAX_PACKAGE_BYTES:
                        raise _invalid()

    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        visit(fd, "", 0)
    finally:
        os.close(fd)
    return entries


def _folder_inventory_fallback(root: Path) -> dict[str, tuple]:
    """Pin a bounded tree using path checks when descriptor walks are absent."""
    entries = {}
    total_bytes = 0
    root_snapshot = _snapshot_directories((*reversed(root.parents), root))

    def visit(directory: Path, prefix: str, depth: int) -> None:
        nonlocal total_bytes
        if depth > 16:
            raise _invalid()
        before = os.lstat(directory)
        if (
            not stat.S_ISDIR(before.st_mode)
            or directory_identity_from_stat(before).reparse
        ):
            raise _invalid()
        with os.scandir(directory) as children:
            for child in children:
                if len(entries) >= MAX_MEMBERS:
                    raise _invalid()
                name = _relative(prefix + child.name)
                child_path = directory / child.name
                info = os.lstat(child_path)
                is_directory = stat.S_ISDIR(info.st_mode)
                if (
                    not is_directory and not stat.S_ISREG(info.st_mode)
                ) or directory_identity_from_stat(info).reparse:
                    raise _invalid()
                entries[name] = (_identity(info), is_directory)
                if is_directory:
                    visit(child_path, name + "/", depth + 1)
                else:
                    total_bytes += info.st_size
                    if total_bytes > MAX_PACKAGE_BYTES:
                        raise _invalid()
        after = os.lstat(directory)
        if (
            not stat.S_ISDIR(after.st_mode)
            or directory_identity_from_stat(after).reparse
            or _identity(after) != _identity(before)
        ):
            raise ValueError("petdex_source_stale")

    visit(root, "", 0)
    _verify_directory_snapshot(root_snapshot)
    return entries


def _folder_source(root: Path) -> PetdexSource:
    if root.is_symlink() or not root.is_dir():
        raise _invalid()
    identity = _identity(root.stat())
    metadata, meta_id = _folder_file(root, "pet.json", MAX_METADATA_BYTES)
    document = read_metadata(metadata)
    inventory = _folder_inventory(root)
    names = {name for name, (_, directory) in inventory.items() if not directory}
    selected = _image_name(document, names)
    image, image_id = _folder_file(root, selected, MAX_IMAGE_BYTES)
    pinned = {"pet.json": (metadata, meta_id), selected: (image, image_id)}
    notice_parts = []
    notice_bytes = 0
    for name in sorted(names):
        if _notice(name):
            data, identity_file = _folder_file(
                root, name, MAX_NOTICE_BYTES - notice_bytes
            )
            notice_bytes += len(data)
            if len(metadata) + len(image) + notice_bytes > MAX_PACKAGE_BYTES:
                raise _invalid()
            pinned[name] = (data, identity_file)
            notice_parts.append(name + "\n" + data.decode("utf-8"))

    def current():
        if root.is_symlink() or _identity(root.stat()) != identity:
            return False
        if _folder_inventory(root) != inventory:
            return False
        return all(
            _folder_file(root, name, max(len(data), 1)) == (data, item_id)
            for name, (data, item_id) in pinned.items()
        )

    return source_from_bytes(
        metadata, image, selected, notices="\n\n".join(notice_parts), guard=current
    )


def _zip_source(path: Path) -> PetdexSource:
    data, identity = _read_path(path, MAX_PACKAGE_BYTES)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        if (
            len(infos) > MAX_MEMBERS
            or sum(i.file_size for i in infos) > MAX_PACKAGE_BYTES
        ):
            raise _invalid()
        names = {}
        for info in infos:
            if info.orig_filename != info.filename:
                raise _invalid()
            name = _relative(info.filename[:-1] if info.is_dir() else info.filename)
            if name in names or info.flag_bits & 1:
                raise _invalid()
            mode = info.external_attr >> 16
            if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR) or (
                stat.S_ISDIR(mode) and not info.is_dir()
            ):
                raise _invalid()
            names[name] = info
        manifests = [
            name
            for name, info in names.items()
            if PurePosixPath(name).name == "pet.json" and not info.is_dir()
        ]
        if len(manifests) != 1:
            raise _invalid()
        manifest = manifests[0]
        prefix = manifest[: -len("pet.json")]
        local = {
            name[len(prefix) :]
            for name, info in names.items()
            if name.startswith(prefix) and not info.is_dir()
        }

        def read(name, cap):
            info = names[prefix + name]
            if info.file_size > cap:
                raise _invalid()
            with archive.open(info) as stream:
                value = stream.read(cap + 1)
            if len(value) > cap:
                raise _invalid()
            return value

        metadata = read("pet.json", MAX_METADATA_BYTES)
        selected = _image_name(read_metadata(metadata), local)
        image = read(selected, MAX_IMAGE_BYTES)
        notice_parts = []
        notice_size = 0
        for name, info in sorted(names.items()):
            if info.is_dir() or not _notice(name):
                continue
            cap = MAX_NOTICE_BYTES - notice_size
            if info.file_size > cap:
                raise _invalid()
            with archive.open(info) as stream:
                notice_data = stream.read(cap + 1)
            if len(notice_data) > cap:
                raise _invalid()
            notice_size += len(notice_data)
            notice_parts.append(name + "\n" + notice_data.decode("utf-8"))
        notices = "\n\n".join(notice_parts)
    digest = hashlib.sha256(data).hexdigest()

    def current():
        fresh, fresh_identity = _read_path(path, MAX_PACKAGE_BYTES)
        return (
            fresh_identity == identity and hashlib.sha256(fresh).hexdigest() == digest
        )

    return source_from_bytes(metadata, image, selected, notices=notices, guard=current)


def read_local_package(path: str | os.PathLike[str]) -> PetdexSource:
    """Read one folder, pet.json, or ZIP without following package links."""
    try:
        selected = Path(path).absolute()
        if selected.is_symlink():
            raise _invalid()
        if selected.name == "pet.json":
            selected = selected.parent
        return _folder_source(selected) if selected.is_dir() else _zip_source(selected)
    except (OSError, UnicodeError, zipfile.BadZipFile, KeyError, RuntimeError):
        raise _invalid() from None
