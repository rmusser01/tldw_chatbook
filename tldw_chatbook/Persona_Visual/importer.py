"""Review-first import for pinned Persona Visual pack archives."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import unicodedata
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from PIL import Image

from tldw_chatbook.Utils.private_paths import secure_private_directory

from .assets import (
    PersonaVisualAssetMetadata,
    load_persona_visual_asset,
    validate_persona_visual_asset_set,
)
from .authoring import (
    PersonaVisualAuthoringDraft,
    PersonaVisualDraftAsset,
    create_persona_visual_import_draft,
)
from .contracts import (
    ALLOWED_ASSET_MIME_TYPES,
    ALLOWED_ASSET_ROLES,
    MAX_ASSET_COUNT,
    MAX_ASSET_DIMENSION,
    MAX_ASSET_TOTAL_BYTES,
    MAX_FRAMES_PER_ANIMATION,
)
from .repository import PersonaVisualIdentity


PERSONA_VISUAL_PACK_SCHEMA = "tldw.persona_visual_pack.v1"
_REQUIRED_MEMBERS = frozenset(
    {
        "manifest.json",
        "metadata/pack.json",
        "metadata/assets.json",
        "checksums/sha256.json",
    }
)
_OPTIONAL_MEMBERS = frozenset({"README.md", "signatures/README.md"})
_ALLOWED_ROOTS = frozenset({"assets", "metadata", "checksums", "signatures"})
_NESTED_SUFFIXES = (
    ".zip",
    ".tar",
    ".tgz",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".tldw-persona-vpack",
    ".tldw-actor-pack",
)
_WINDOWS_DEVICES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_FORMAT_BY_MIME = {
    "image/png": ("PNG", ".png"),
    "image/jpeg": ("JPEG", ".jpg"),
    "image/webp": ("WEBP", ".webp"),
    "image/gif": ("GIF", ".gif"),
}
_MAX_ARCHIVE_BYTES = MAX_ASSET_TOTAL_BYTES + 10 * 1024 * 1024
_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_MEMBER_COUNT = MAX_ASSET_COUNT + 16
_MAX_COMPRESSION_RATIO = 200
_READ_CHUNK = 64 * 1024
_MARKER_NAME = ".persona-visual-import"
_CAPABILITY = re.compile(r"\Apvi1:([0-9a-f]{64}):(\.import-[0-9a-f]{32})\Z")
_SAFE_ASSET_NAME = re.compile(r"[0-9]{3}\.(?:png|jpg|webp|gif)\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class PersonaVisualImportError(ValueError):
    """Stable path-free import failure with optional cleanup authority."""

    __slots__ = ("category", "cleanup_candidate")

    def __init__(self, category: str, *, cleanup_candidate: str | None = None) -> None:
        if category not in {
            "persona_visual_import_invalid",
            "persona_visual_import_unsupported",
            "persona_visual_import_cancelled",
            "persona_visual_import_stale",
            "persona_visual_import_cleanup_denied",
            "persona_visual_import_failed",
        }:
            category = "persona_visual_import_failed"
        self.category = category
        self.cleanup_candidate = cleanup_candidate
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class PersonaVisualImportReview:
    """Path-free review metadata plus an isolated unpublished draft."""

    schema_version: str
    archive_sha256: str
    pack_title: str
    asset_count: int
    state_count: int
    draft: PersonaVisualAuthoringDraft
    cleanup_candidate: str = field(repr=False)
    _candidate_name: str = field(repr=False)
    _candidate_identity: tuple[int, int] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _SourceSnapshot:
    path: Path = field(repr=False)
    identity: tuple[int, int, int, int, int]
    sha256: str
    data: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class _Candidate:
    root: Path = field(repr=False)
    name: str
    identity: tuple[int, int]
    secret: str = field(repr=False)
    asset_names: tuple[str, ...]

    @property
    def capability(self) -> str:
        return f"pvi1:{self.secret}:{self.name}"


def import_persona_visual_pack(
    archive_path: os.PathLike[str] | str,
    *,
    staging_root: os.PathLike[str] | str,
    persona_id: str,
    persona_revision: int,
    expected_identity: PersonaVisualIdentity | None,
    cancel_event: object | None = None,
) -> PersonaVisualImportReview:
    """Validate one server-compatible archive into an unpublished review draft."""

    candidate: _Candidate | None = None
    try:
        cancelled = _cancel_checker(cancel_event)
        _raise_if_cancelled(cancelled)
        source = _pin_source(archive_path)
        root = _private_staging_root(staging_root)
        _raise_if_cancelled(cancelled)
        with zipfile.ZipFile(BytesIO(source.data), "r") as archive:
            members = _validated_members(archive)
            outer = _json_member(archive, members, "manifest.json")
            checksums = _checksums(
                _json_member(archive, members, "checksums/sha256.json")
            )
            pack = _pack(_json_member(archive, members, "metadata/pack.json"))
            asset_records = _assets(
                _json_member(archive, members, "metadata/assets.json")
            )
            _validate_declarations(
                archive,
                members,
                outer,
                checksums,
                asset_records,
                cancelled,
            )
            _preflight_space(root, members)
            candidate = _create_candidate(root)
            draft_assets = _extract_assets(
                archive,
                members,
                asset_records,
                candidate,
                cancelled,
            )
            candidate = _candidate_with_assets(candidate, draft_assets)
            manifest_document = pack["visual_manifest"]
            manifest_json = _canonical_text(manifest_document)
            _write_private(candidate.root / "manifest.json", manifest_json.encode())
            draft = create_persona_visual_import_draft(
                persona_id=persona_id,
                persona_revision=persona_revision,
                expected_identity=expected_identity,
                title=pack["title"],
                description="Imported Persona Visual pack",
                manifest_json=manifest_json,
                assets=draft_assets,
            )
        _raise_if_cancelled(cancelled)
        if not _source_identity_current(
            source.path,
            source.identity,
            source.sha256,
        ):
            raise PersonaVisualImportError("persona_visual_import_stale")
        _candidate_current(candidate)
        return PersonaVisualImportReview(
            schema_version=PERSONA_VISUAL_PACK_SCHEMA,
            archive_sha256=source.sha256,
            pack_title=draft.title,
            asset_count=len(draft.assets),
            state_count=len(json.loads(draft.manifest_json)["states"]),
            draft=draft,
            cleanup_candidate=candidate.capability,
            _candidate_name=candidate.name,
            _candidate_identity=candidate.identity,
        )
    except PersonaVisualImportError as exc:
        cleanup_candidate = _cleanup_failed_candidate(candidate)
        if cleanup_candidate is not None and exc.cleanup_candidate is None:
            exc.cleanup_candidate = cleanup_candidate
        raise
    except (KeyboardInterrupt, SystemExit):
        _cleanup_failed_candidate(candidate)
        raise
    except Exception:
        cleanup_candidate = _cleanup_failed_candidate(candidate)
        raise PersonaVisualImportError(
            "persona_visual_import_invalid",
            cleanup_candidate=cleanup_candidate,
        ) from None


def persona_visual_import_source_root(
    review: PersonaVisualImportReview,
    *,
    staging_root: os.PathLike[str] | str,
) -> Path:
    """Return the private publication source only while its exact lease is current."""

    candidate = _review_candidate(review, staging_root)
    _candidate_current(candidate)
    return candidate.root


def cleanup_persona_visual_import_review(
    review: PersonaVisualImportReview,
    *,
    staging_root: os.PathLike[str] | str,
) -> bool:
    """Delete only the exact module-issued staging identity for one review."""

    candidate = _review_candidate(review, staging_root)
    if not _delete_candidate(candidate):
        raise PersonaVisualImportError("persona_visual_import_cleanup_denied")
    return True


def _pin_source(value: os.PathLike[str] | str) -> _SourceSnapshot:
    raw = os.fspath(value)
    if type(raw) is not str or "\x00" in raw:
        raise ValueError
    path = Path(raw)
    if not path.is_absolute() or str(path) != raw:
        raise ValueError
    before = os.lstat(path)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > _MAX_ARCHIVE_BYTES
    ):
        raise ValueError
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        data = _read_fd(descriptor, before.st_size)
        after = os.fstat(descriptor)
        named = os.lstat(path)
    finally:
        os.close(descriptor)
    identity = _file_identity(before)
    if (
        _file_identity(opened) != identity
        or _file_identity(after) != identity
        or _file_identity(named) != identity
        or not stat.S_ISREG(named.st_mode)
        or len(data) != before.st_size
    ):
        raise ValueError
    return _SourceSnapshot(
        path=path,
        identity=identity,
        sha256=hashlib.sha256(data).hexdigest(),
        data=data,
    )


def _source_identity_current(
    path: Path,
    identity: tuple[int, int, int, int, int],
    expected_sha256: str,
) -> bool:
    try:
        current = _pin_source(path)
        return current.identity == identity and current.sha256 == expected_sha256
    except Exception:
        return False


def _validated_members(
    archive: zipfile.ZipFile,
) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if len(infos) > _MAX_MEMBER_COUNT:
        raise ValueError
    members: dict[str, zipfile.ZipInfo] = {}
    collision_keys: set[str] = set()
    total = 0
    for info in infos:
        raw = getattr(info, "orig_filename", info.filename)
        if info.is_dir():
            _member_name(raw.removesuffix("/"), directory=True)
            continue
        name = _member_name(raw, directory=False)
        collision = unicodedata.normalize("NFC", name).casefold()
        if name in members or collision in collision_keys:
            raise ValueError
        mode = (info.external_attr >> 16) & 0o170000
        if mode not in {0, stat.S_IFREG} or info.flag_bits & 0x1:
            raise ValueError
        if name.lower().endswith(_NESTED_SUFFIXES):
            raise ValueError
        if info.file_size < 0 or info.compress_size < 0:
            raise ValueError
        if (
            info.file_size
            and info.file_size > max(info.compress_size, 1) * _MAX_COMPRESSION_RATIO
        ):
            raise ValueError
        total += info.file_size
        if total > MAX_ASSET_TOTAL_BYTES + 3 * _MAX_JSON_BYTES:
            raise ValueError
        members[name] = info
        collision_keys.add(collision)
    if not _REQUIRED_MEMBERS.issubset(members):
        raise ValueError
    return members


def _member_name(value: object, *, directory: bool) -> str:
    if type(value) is not str or not value or "\x00" in value or "\\" in value:
        raise ValueError
    value.encode("utf-8")
    if value.startswith("/") or not value.isascii():
        raise ValueError
    path = PurePosixPath(value)
    parts = tuple(value.split("/"))
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in parts)
        or any(":" in part for part in parts)
    ):
        raise ValueError
    for part in parts:
        if (
            len(part.encode()) > 255
            or part.rstrip(" .").split(".", 1)[0].upper() in _WINDOWS_DEVICES
        ):
            raise ValueError
    if (
        parts[0] not in _ALLOWED_ROOTS
        and value not in _OPTIONAL_MEMBERS
        and value != "manifest.json"
    ):
        raise ValueError
    if directory and parts[0] not in _ALLOWED_ROOTS:
        raise ValueError
    return value


def _json_member(
    archive: zipfile.ZipFile,
    members: Mapping[str, zipfile.ZipInfo],
    name: str,
) -> object:
    info = members[name]
    if info.file_size > _MAX_JSON_BYTES:
        raise ValueError
    data = archive.read(info)
    if len(data) != info.file_size:
        raise ValueError
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError from None


def _checksums(value: object) -> dict[str, str]:
    if type(value) is not dict or len(value) > _MAX_MEMBER_COUNT:
        raise ValueError
    result: dict[str, str] = {}
    for raw_name, digest in value.items():
        name = _member_name(raw_name, directory=False)
        if type(digest) is not str or _SHA256.fullmatch(digest) is None:
            raise ValueError
        result[name] = digest
    return result


def _pack(value: object) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("pack")) is not dict:
        raise ValueError
    pack = value["pack"]
    title = pack.get("title")
    manifest = pack.get("visual_manifest")
    if (
        type(title) is not str
        or not title
        or len(title) > 256
        or type(manifest) is not dict
        or pack.get("renderer_type") != "sprite_frames"
        or pack.get("manifest_version") != 1
    ):
        raise ValueError
    title.encode("utf-8")
    return {"title": title, "visual_manifest": manifest}


def _assets(value: object) -> tuple[dict[str, Any], ...]:
    if type(value) is not dict or type(value.get("assets")) is not list:
        raise ValueError
    records = value["assets"]
    if not records or len(records) > MAX_ASSET_COUNT:
        raise ValueError
    result: list[dict[str, Any]] = []
    keys: set[str] = set()
    paths: set[str] = set()
    for record in records:
        if type(record) is not dict:
            raise ValueError
        key = record.get("source_asset_id")
        role = record.get("asset_role")
        mime = record.get("mime_type")
        path = record.get("asset_path")
        digest = record.get("asset_sha256") or record.get("checksum_sha256")
        size = record.get("asset_size_bytes")
        if size is None:
            size = record.get("byte_size")
        width = record.get("width")
        height = record.get("height")
        if (
            record.get("asset_bytes_status") != "present"
            or type(key) is not str
            or key in keys
            or type(role) is not str
            or role not in ALLOWED_ASSET_ROLES
            or type(mime) is not str
            or mime not in ALLOWED_ASSET_MIME_TYPES
            or type(path) is not str
            or path in paths
            or not path.startswith("assets/")
            or type(digest) is not str
            or _SHA256.fullmatch(digest) is None
            or type(size) is not int
            or size <= 0
            or type(width) is not int
            or width <= 0
            or width > MAX_ASSET_DIMENSION
            or type(height) is not int
            or height <= 0
            or height > MAX_ASSET_DIMENSION
        ):
            raise ValueError
        _member_name(path, directory=False)
        result.append(
            {
                "asset_key": key,
                "role": role,
                "mime_type": mime,
                "asset_path": path,
                "sha256": digest,
                "byte_count": size,
                "width": width,
                "height": height,
                "duration_ms": record.get("duration_ms"),
            }
        )
        keys.add(key)
        paths.add(path)
    return tuple(result)


def _validate_declarations(
    archive: zipfile.ZipFile,
    members: Mapping[str, zipfile.ZipInfo],
    outer: object,
    checksums: Mapping[str, str],
    assets: tuple[dict[str, Any], ...],
    cancelled: Any,
) -> None:
    if (
        type(outer) is not dict
        or outer.get("schema_version") != PERSONA_VISUAL_PACK_SCHEMA
    ):
        raise PersonaVisualImportError("persona_visual_import_unsupported")
    encryption = outer.get("encryption")
    if type(encryption) is not dict or encryption.get("encrypted") is not False:
        raise ValueError
    sections = outer.get("sections")
    if type(sections) is not list:
        raise ValueError
    section_map: dict[str, str] = {}
    for section in sections:
        if type(section) is not dict:
            raise ValueError
        name = _member_name(section.get("path"), directory=False)
        digest = section.get("sha256")
        if (
            type(digest) is not str
            or _SHA256.fullmatch(digest) is None
            or name in section_map
        ):
            raise ValueError
        section_map[name] = digest
    declared_assets = {record["asset_path"] for record in assets}
    expected_sections = {"metadata/pack.json", "metadata/assets.json"} | declared_assets
    if set(section_map) != expected_sections:
        raise ValueError
    expected_checksum_names = expected_sections | {"manifest.json"}
    if set(checksums) != expected_checksum_names:
        raise ValueError
    regular_members = set(members)
    allowed = expected_checksum_names | {"checksums/sha256.json"} | _OPTIONAL_MEMBERS
    if regular_members - allowed:
        raise ValueError
    if not declared_assets.issubset(members):
        raise ValueError
    for name, expected in checksums.items():
        _raise_if_cancelled(cancelled)
        info = members.get(name)
        if info is None or _member_digest(archive, info, cancelled) != expected:
            raise ValueError
    if any(section_map[name] != checksums[name] for name in section_map):
        raise ValueError
    for record in assets:
        info = members[record["asset_path"]]
        if (
            info.file_size != record["byte_count"]
            or checksums[record["asset_path"]] != record["sha256"]
        ):
            raise ValueError


def _preflight_space(root: Path, members: Mapping[str, zipfile.ZipInfo]) -> None:
    required = sum(info.file_size for info in members.values()) + 1024 * 1024
    if shutil.disk_usage(root).free < required:
        raise ValueError


def _create_candidate(root: Path) -> _Candidate:
    name = f".import-{uuid4().hex}"
    candidate = root / name
    candidate.mkdir(mode=0o700)
    metadata = os.lstat(candidate)
    if not _private_directory(metadata):
        raise ValueError
    secret = os.urandom(32).hex()
    identity = (metadata.st_dev, metadata.st_ino)
    marker = _marker(secret, name, identity)
    _write_private(candidate / _MARKER_NAME, marker.encode())
    (candidate / "assets").mkdir(mode=0o700)
    return _Candidate(candidate, name, identity, secret, ())


def _candidate_with_assets(
    candidate: _Candidate,
    assets: tuple[PersonaVisualDraftAsset, ...],
) -> _Candidate:
    names = tuple(PurePosixPath(asset.source_storage_key).name for asset in assets)
    return _Candidate(
        candidate.root,
        candidate.name,
        candidate.identity,
        candidate.secret,
        names,
    )


def _extract_assets(
    archive: zipfile.ZipFile,
    members: Mapping[str, zipfile.ZipInfo],
    records: tuple[dict[str, Any], ...],
    candidate: _Candidate,
    cancelled: Any,
) -> tuple[PersonaVisualDraftAsset, ...]:
    draft_assets: list[PersonaVisualDraftAsset] = []
    for index, record in enumerate(records):
        _raise_if_cancelled(cancelled)
        extension = _FORMAT_BY_MIME[record["mime_type"]][1]
        target_name = f"{index:03d}{extension}"
        target = candidate.root / "assets" / target_name
        digest = hashlib.sha256()
        written = 0
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            with archive.open(members[record["asset_path"]], "r") as source:
                while True:
                    _raise_if_cancelled(cancelled)
                    chunk = source.read(_READ_CHUNK)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > record["byte_count"]:
                        raise ValueError
                    digest.update(chunk)
                    _write_all(descriptor, chunk)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if written != record["byte_count"] or digest.hexdigest() != record["sha256"]:
            raise ValueError
        frame_count, duration_ms = _inspect_image(target, record)
        metadata = PersonaVisualAssetMetadata(
            asset_key=record["asset_key"],
            role=record["role"],
            mime_type=record["mime_type"],
            byte_count=written,
            sha256=digest.hexdigest(),
            width=record["width"],
            height=record["height"],
            frame_count=frame_count,
            duration_ms=duration_ms,
        )
        metadata = validate_persona_visual_asset_set((metadata,))[0]
        storage_key = f"assets/{target_name}"
        load_persona_visual_asset(
            candidate.root,
            storage_key=storage_key,
            metadata=metadata,
        )
        draft_assets.append(PersonaVisualDraftAsset(storage_key, metadata))
    validate_persona_visual_asset_set(tuple(asset.metadata for asset in draft_assets))
    return tuple(draft_assets)


def _inspect_image(path: Path, record: Mapping[str, Any]) -> tuple[int, int | None]:
    with Image.open(path) as image:
        if (
            image.format != _FORMAT_BY_MIME[record["mime_type"]][0]
            or image.width != record["width"]
            or image.height != record["height"]
        ):
            raise ValueError
        frame_count = int(getattr(image, "n_frames", 1))
        if frame_count < 1 or frame_count > MAX_FRAMES_PER_ANIMATION:
            raise ValueError
        duration = 0
        for index in range(frame_count):
            image.seek(index)
            image.load()
            value = image.info.get("duration", 0)
            if type(value) is not int or value < 0:
                raise ValueError
            duration += value
        actual_duration = duration or None
        declared_duration = record.get("duration_ms")
        if declared_duration is not None and declared_duration != actual_duration:
            raise ValueError
        return frame_count, actual_duration


def _private_staging_root(value: os.PathLike[str] | str) -> Path:
    raw = os.fspath(value)
    if type(raw) is not str or "\x00" in raw:
        raise ValueError
    root = Path(raw)
    if not root.is_absolute() or str(root) != raw:
        raise ValueError
    privacy = secure_private_directory(root, create=True, application_owned=True)
    if not privacy.verified_private:
        raise ValueError
    return root


def _review_candidate(
    review: object,
    staging_root: os.PathLike[str] | str,
) -> _Candidate:
    if type(review) is not PersonaVisualImportReview:
        raise PersonaVisualImportError("persona_visual_import_cleanup_denied")
    root = _private_staging_root(staging_root)
    parsed = _CAPABILITY.fullmatch(review.cleanup_candidate)
    if (
        parsed is None
        or parsed.group(2) != review._candidate_name
        or type(review._candidate_identity) is not tuple
        or len(review._candidate_identity) != 2
    ):
        raise PersonaVisualImportError("persona_visual_import_cleanup_denied")
    return _Candidate(
        root / review._candidate_name,
        review._candidate_name,
        review._candidate_identity,
        parsed.group(1),
        tuple(
            PurePosixPath(asset.source_storage_key).name
            for asset in review.draft.assets
        ),
    )


def _candidate_current(candidate: _Candidate) -> None:
    root_fd = -1
    candidate_fd = -1
    try:
        root_fd, candidate_fd = _open_candidate(candidate)
        if _read_marker(candidate_fd) != _marker(
            candidate.secret,
            candidate.name,
            candidate.identity,
        ):
            raise ValueError
    except Exception:
        raise PersonaVisualImportError("persona_visual_import_cleanup_denied") from None
    finally:
        if candidate_fd >= 0:
            os.close(candidate_fd)
        if root_fd >= 0:
            os.close(root_fd)


def _cleanup_failed_candidate(candidate: _Candidate | None) -> str | None:
    if candidate is None:
        return None
    return (
        None
        if _delete_candidate(candidate, allow_partial=True)
        else candidate.capability
    )


def _delete_candidate(candidate: _Candidate, *, allow_partial: bool = False) -> bool:
    root_fd = -1
    candidate_fd = -1
    assets_fd = -1
    try:
        root_fd, candidate_fd = _open_candidate(candidate)
        if _read_marker(candidate_fd) != _marker(
            candidate.secret,
            candidate.name,
            candidate.identity,
        ):
            return False
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        assets_fd = os.open("assets", directory_flags, dir_fd=candidate_fd)
        if not _private_directory(os.fstat(assets_fd)):
            return False
        actual_assets = tuple(os.listdir(assets_fd))
        if not allow_partial and set(actual_assets) != set(candidate.asset_names):
            return False
        if allow_partial and len(actual_assets) > MAX_ASSET_COUNT:
            return False
        for name in actual_assets:
            if _SAFE_ASSET_NAME.fullmatch(name) is None:
                return False
            metadata = os.stat(name, dir_fd=assets_fd, follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                return False
        allowed_root = {_MARKER_NAME, "manifest.json", "assets"}
        actual_root = set(os.listdir(candidate_fd))
        if not actual_root.issubset(allowed_root):
            return False
        for name in actual_assets:
            os.unlink(name, dir_fd=assets_fd)
        os.close(assets_fd)
        assets_fd = -1
        os.rmdir("assets", dir_fd=candidate_fd)
        if "manifest.json" in actual_root:
            manifest = os.stat(
                "manifest.json",
                dir_fd=candidate_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISREG(manifest.st_mode) or manifest.st_nlink != 1:
                return False
            os.unlink("manifest.json", dir_fd=candidate_fd)
        os.unlink(_MARKER_NAME, dir_fd=candidate_fd)
        named = os.stat(
            candidate.name,
            dir_fd=root_fd,
            follow_symlinks=False,
        )
        opened = os.fstat(candidate_fd)
        if (named.st_dev, named.st_ino) != candidate.identity or (
            opened.st_dev,
            opened.st_ino,
        ) != candidate.identity:
            return False
        os.rmdir(candidate.name, dir_fd=root_fd)
        return True
    except Exception:
        return False
    finally:
        if assets_fd >= 0:
            os.close(assets_fd)
        if candidate_fd >= 0:
            os.close(candidate_fd)
        if root_fd >= 0:
            os.close(root_fd)


def _open_candidate(candidate: _Candidate) -> tuple[int, int]:
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd = os.open(candidate.root.parent, directory_flags)
    try:
        candidate_fd = os.open(candidate.name, directory_flags, dir_fd=root_fd)
        opened = os.fstat(candidate_fd)
        named = os.stat(
            candidate.name,
            dir_fd=root_fd,
            follow_symlinks=False,
        )
        if (
            not _private_directory(opened)
            or not _private_directory(named)
            or (opened.st_dev, opened.st_ino) != candidate.identity
            or (named.st_dev, named.st_ino) != candidate.identity
        ):
            raise ValueError
        return root_fd, candidate_fd
    except Exception:
        os.close(root_fd)
        raise


def _read_marker(candidate_fd: int) -> str:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(_MARKER_NAME, flags, dir_fd=candidate_fd)
    try:
        opened = os.fstat(descriptor)
        named = os.stat(
            _MARKER_NAME,
            dir_fd=candidate_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or opened.st_nlink != 1
            or named.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (named.st_dev, named.st_ino, named.st_size)
            or opened.st_size != 64
        ):
            raise ValueError
        data = _read_fd(descriptor, 64)
        return data.decode("ascii")
    finally:
        os.close(descriptor)


def _marker(secret: str, name: str, identity: tuple[int, int]) -> str:
    payload = f"{name}:{identity[0]}:{identity[1]}".encode()
    return hashlib.blake2b(
        payload,
        digest_size=32,
        key=bytes.fromhex(secret),
    ).hexdigest()


def _private_directory(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(metadata.st_mode)
        and metadata.st_nlink >= 1
        and (not hasattr(os, "geteuid") or metadata.st_uid == os.geteuid())
        and stat.S_IMODE(metadata.st_mode) == 0o700
    )


def _write_private(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        _write_all(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError
        view = view[written:]


def _member_digest(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    cancelled: Any,
) -> str:
    digest = hashlib.sha256()
    total = 0
    with archive.open(info, "r") as source:
        while chunk := source.read(_READ_CHUNK):
            _raise_if_cancelled(cancelled)
            total += len(chunk)
            if total > info.file_size:
                raise ValueError
            digest.update(chunk)
    if total != info.file_size:
        raise ValueError
    return digest.hexdigest()


def _read_fd(descriptor: int, expected: int) -> bytes:
    chunks: list[bytes] = []
    remaining = expected + 1
    while remaining:
        chunk = os.read(descriptor, min(_READ_CHUNK, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
    if len(data) != expected:
        raise ValueError
    return data


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _canonical_text(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise ValueError
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ValueError


def _cancel_checker(value: object | None) -> Any:
    if value is None:
        return lambda: False
    checker = getattr(value, "is_set", None)
    if not callable(checker):
        raise ValueError
    return checker


def _raise_if_cancelled(checker: Any) -> None:
    if checker():
        raise PersonaVisualImportError("persona_visual_import_cancelled")


__all__ = [
    "PERSONA_VISUAL_PACK_SCHEMA",
    "PersonaVisualImportError",
    "PersonaVisualImportReview",
    "cleanup_persona_visual_import_review",
    "import_persona_visual_pack",
    "persona_visual_import_source_root",
]
