"""Review-first import of untrusted portable Actor Pack archives."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import unicodedata
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

from PIL import Image

from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.Utils.private_paths import secure_private_directory

from .contracts import (
    MAX_FILES,
    MAX_MANIFEST_BYTES,
    MAX_MEMBER_BYTES,
    MAX_PORTRAIT_BYTES,
    MAX_TOTAL_BYTES,
    ZIP_CREATE_SYSTEM,
    ActorPackValidationError,
    canonical_json_bytes,
    canonical_member_order,
    validate_actor_pack_document,
    validate_actor_portrait,
)
from .repository import ActorPackRepository


_MAX_ARCHIVE_BYTES = MAX_TOTAL_BYTES + 16 * 1024 * 1024
_MAX_RATIO = 100
_READ_CHUNK = 64 * 1024
_MARKER = ".actor-pack-import"
_NESTED_SUFFIXES = (
    ".zip",
    ".tar",
    ".tgz",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".tldw-actor-pack",
    ".tldw-persona-vpack",
)
_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class ActorPackImportError(ValueError):
    """Stable path-free import failure."""

    __slots__ = ("category", "cleanup_candidate")

    def __init__(self, category: str, *, cleanup_candidate: str | None = None) -> None:
        if category not in {
            "actor_pack_import_invalid",
            "actor_pack_import_unsupported",
            "actor_pack_import_cancelled",
            "actor_pack_import_review_stale",
            "actor_pack_import_identity_conflict",
            "actor_pack_import_cleanup_denied",
            "actor_pack_import_disk_unavailable",
            "actor_pack_import_failed",
        }:
            category = "actor_pack_import_failed"
        self.category = category
        self.cleanup_candidate = cleanup_candidate
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class ActorPackPortraitReview:
    """Bounded path-free portrait metadata."""

    mime_type: str
    width: int
    height: int
    byte_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ActorPackPortraitPreview:
    """Lease-authorized portrait bytes for trusted UI rendering."""

    mime_type: str
    width: int
    height: int
    data: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class _StagedFile:
    member: str
    identity: tuple[int, int, int, int, int, int]
    sha256: str


@dataclass(frozen=True, slots=True)
class _ActorPackImportMaterial:
    actor_fields: Mapping[str, Any]
    portrait: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class ActorPackImportReview:
    """Immutable consent model; filesystem authority is private and repr-free."""

    actor_kind: str
    portable_uuid: str
    actor_fields: tuple[tuple[str, Any], ...]
    portrait: ActorPackPortraitReview
    sections: tuple[str, ...]
    section_effects: tuple[tuple[str, str], ...]
    uuid_match: str
    allowed_actions: tuple[str, ...]
    archive_sha256: str
    content_digest: str
    _candidate_name: str = field(repr=False)
    _candidate_identity: tuple[int, int] = field(repr=False)
    _secret: str = field(repr=False)
    _files: tuple[_StagedFile, ...] = field(repr=False)
    _portrait_member: str = field(repr=False)
    _actor_member: str = field(repr=False)
    _matched_actor_kind: str | None = field(repr=False)
    _matched_local_actor_id: str | None = field(repr=False)
    _matched_identity_version: int | None = field(repr=False)
    _matched_actor_version: int | None = field(repr=False)
    _matched_actor_digest: str | None = field(repr=False)


class ActorPackImportService:
    """Validate archives into private staging without mutating live actors."""

    def __init__(
        self,
        repository: ActorPackRepository,
        *,
        staging_root: os.PathLike[str] | str,
        profile_root: os.PathLike[str] | str,
        local_service: LocalCharacterPersonaService | None = None,
    ) -> None:
        if not isinstance(repository, ActorPackRepository):
            raise ActorPackImportError("actor_pack_import_invalid")
        self.repository = repository
        self._local_service = local_service
        self._staging_root = _absolute_path(staging_root)
        self._profile_root = _absolute_path(profile_root)

    def inspect_archive(
        self,
        archive_path: os.PathLike[str] | str,
        *,
        cancel_requested: Callable[[], bool] = lambda: False,
    ) -> ActorPackImportReview:
        """Inspect and stage one archive while exposing only path-free metadata."""

        candidate: Path | None = None
        try:
            _cancel(cancel_requested)
            source_path = _absolute_path(archive_path)
            source, source_identity, archive_sha256, archive_bytes = _pin_source(
                source_path
            )
            try:
                with source, zipfile.ZipFile(source, "r") as archive:
                    infos = _validated_members(archive)
                    root_bytes = _read_member(
                        archive,
                        infos["actor-pack.json"],
                        MAX_MANIFEST_BYTES,
                        cancel_requested,
                    )
                    manifest = _canonical_object(root_bytes)
                    required = manifest.get("required_features")
                    if type(required) is not list or any(
                        feature
                        not in {
                            "shared-visual-identity/v1",
                            "persona-runtime/sprite-frames-v1",
                        }
                        for feature in required
                    ):
                        raise ActorPackImportError("actor_pack_import_unsupported")
                    inventory = _inventory(manifest)
                    declared = {item[0] for item in inventory}
                    if set(infos) != declared | {"actor-pack.json"}:
                        raise ValueError
                    _preflight_space(
                        self._staging_root,
                        archive_bytes + sum(item[1] for item in inventory),
                    )
                    candidate, identity, secret = _create_candidate(
                        self._staging_root
                    )
                    staged = [_stage_root(candidate, root_bytes)]
                    files: dict[str, bytes] = {}
                    for member, expected_bytes, expected_digest in inventory:
                        _cancel(cancel_requested)
                        record = _extract_member(
                            archive,
                            infos[member],
                            candidate,
                            member,
                            expected_bytes,
                            expected_digest,
                            cancel_requested,
                        )
                        staged.append(record)
                        # Pure contract validation currently consumes bytes. Keep the
                        # bounded V1 call while section loaders move large assets from
                        # their staged descriptors in the section-validation step.
                        files[member] = _read_staged(candidate, record)
                    document = validate_actor_pack_document(manifest, files)
                    actor = _canonical_object(files[document.payload_path])
                    actor_fields = tuple(sorted(actor["data"].items()))
                    portrait_data = files[document.portrait_path]
                    validate_actor_portrait(document.portrait_path, portrait_data)
                    portrait = _portrait_review(
                        document.portrait_path, portrait_data
                    )
                    sections = tuple(section.kind for section in document.sections)
                    section_effects = tuple(
                        (kind, "Included — imported visuals will be activated")
                        for kind in sections
                    )
                    matched = self.repository.get_identity_by_portable_uuid(
                        document.portable_uuid
                    )
                    if matched is None:
                        uuid_match = "none"
                        allowed_actions = ("create_new", "create_copy")
                    elif matched.actor_kind == document.actor_kind:
                        uuid_match = "same_kind"
                        allowed_actions = ("create_copy", "update_existing")
                    else:
                        raise ActorPackImportError(
                            "actor_pack_import_identity_conflict"
                        )
                    actor_authority = (
                        None if matched is None else self._actor_authority(matched)
                    )
            finally:
                if not source.closed:
                    source.close()
            _cancel(cancel_requested)
            if _source_identity(source_path) != source_identity:
                raise ActorPackImportError("actor_pack_import_review_stale")
            _candidate_current(candidate, identity, secret)
            return ActorPackImportReview(
                actor_kind=document.actor_kind,
                portable_uuid=document.portable_uuid,
                actor_fields=actor_fields,
                portrait=portrait,
                sections=sections,
                section_effects=section_effects,
                uuid_match=uuid_match,
                allowed_actions=allowed_actions,
                archive_sha256=archive_sha256,
                content_digest=document.content_digest,
                _candidate_name=candidate.name,
                _candidate_identity=identity,
                _secret=secret,
                _files=tuple(staged),
                _portrait_member=document.portrait_path,
                _actor_member=document.payload_path,
                _matched_actor_kind=None if matched is None else matched.actor_kind,
                _matched_local_actor_id=(
                    None if matched is None else matched.local_actor_id
                ),
                _matched_identity_version=None if matched is None else matched.version,
                _matched_actor_version=(
                    None if actor_authority is None else actor_authority[0]
                ),
                _matched_actor_digest=(
                    None if actor_authority is None else actor_authority[1]
                ),
            )
        except ActorPackImportError:
            if candidate is not None:
                _cleanup_candidate(candidate)
            raise
        except ActorPackValidationError as exc:
            if candidate is not None:
                _cleanup_candidate(candidate)
            category = (
                "actor_pack_import_unsupported"
                if exc.category
                in {"actor_pack_schema_unsupported", "actor_pack_feature_unsupported"}
                else "actor_pack_import_invalid"
            )
            raise ActorPackImportError(category) from None
        except (KeyboardInterrupt, SystemExit):
            if candidate is not None:
                _cleanup_candidate(candidate)
            raise
        except Exception:
            if candidate is not None:
                _cleanup_candidate(candidate)
            raise ActorPackImportError("actor_pack_import_invalid") from None

    def read_portrait_preview(
        self, review: ActorPackImportReview
    ) -> ActorPackPortraitPreview:
        """Read bounded portrait bytes only while the exact review lease is current."""

        candidate = self._review_candidate(review)
        record = next(
            (item for item in review._files if item.member == review._portrait_member),
            None,
        )
        if record is None:
            raise ActorPackImportError("actor_pack_import_review_stale")
        try:
            data = _read_staged(candidate, record)
            if len(data) > MAX_PORTRAIT_BYTES:
                raise ValueError
        except Exception:
            raise ActorPackImportError("actor_pack_import_review_stale") from None
        return ActorPackPortraitPreview(
            review.portrait.mime_type,
            review.portrait.width,
            review.portrait.height,
            data,
        )

    def cleanup_review(self, review: ActorPackImportReview) -> bool:
        """Delete only the authenticated candidate owned by this review."""

        candidate = self._review_candidate(review)
        if not _cleanup_candidate(candidate):
            raise ActorPackImportError("actor_pack_import_cleanup_denied")
        return True

    def revalidate_review(
        self,
        review: ActorPackImportReview,
        *,
        alternate_actor_authorities: tuple[tuple[int, str], ...] = (),
    ) -> None:
        """Require the complete staged and UUID authority represented by review."""

        candidate = self._review_candidate(review)
        try:
            for record in review._files:
                _read_staged(candidate, record)
            current = self.repository.get_identity_by_portable_uuid(
                review.portable_uuid
            )
            if review.uuid_match == "none":
                if current is not None:
                    raise ValueError
            elif (
                current is None
                or current.actor_kind != review._matched_actor_kind
                or current.local_actor_id != review._matched_local_actor_id
                or current.version != review._matched_identity_version
            ):
                raise ValueError
            if current is not None:
                actor_authority = self._actor_authority(current)
                if actor_authority not in (
                    (
                        review._matched_actor_version,
                        review._matched_actor_digest,
                    ),
                    *alternate_actor_authorities,
                ):
                    raise ValueError
        except Exception:
            raise ActorPackImportError("actor_pack_import_review_stale") from None

    def _activation_material(
        self, review: ActorPackImportReview
    ) -> _ActorPackImportMaterial:
        self.revalidate_review(review)
        candidate = self._review_candidate(review)
        records = {record.member: record for record in review._files}
        try:
            actor = _canonical_object(
                _read_staged(candidate, records[review._actor_member])
            )
            portrait = _read_staged(candidate, records[review._portrait_member])
            fields = actor["data"]
            if type(fields) is not dict:
                raise ValueError
        except Exception:
            raise ActorPackImportError("actor_pack_import_review_stale") from None
        return _ActorPackImportMaterial(dict(fields), portrait)

    def _actor_authority(self, identity: Any) -> tuple[int, str]:
        if identity.actor_kind == "character":
            actor = self.repository.db.get_character_card_by_id(
                int(identity.local_actor_id)
            )
            if actor is None:
                raise ActorPackImportError("actor_pack_import_review_stale")
            version = actor.get("version")
            data = {
                key: actor.get(key)
                for key in (
                    "name",
                    "description",
                    "personality",
                    "scenario",
                    "post_history_instructions",
                    "first_message",
                    "message_example",
                    "creator_notes",
                    "system_prompt",
                    "alternate_greetings",
                    "tags",
                    "creator",
                    "character_version",
                    "extensions",
                )
            }
            image = actor.get("image")
            if isinstance(image, bytes):
                data["portrait_sha256"] = hashlib.sha256(image).hexdigest()
        else:
            if self._local_service is None:
                raise ActorPackImportError("actor_pack_import_review_stale")
            actor = self._local_service._find_persona_profile(
                identity.local_actor_id
            )
            version = actor.get("version")
            data = dict(actor)
        if type(version) is not int or version < 1:
            raise ActorPackImportError("actor_pack_import_review_stale")
        try:
            digest = hashlib.sha256(canonical_json_bytes(data)).hexdigest()
        except (ActorPackValidationError, TypeError, ValueError):
            raise ActorPackImportError("actor_pack_import_review_stale") from None
        return version, digest

    def _review_candidate(self, review: object) -> Path:
        if type(review) is not ActorPackImportReview:
            raise ActorPackImportError("actor_pack_import_cleanup_denied")
        candidate = self._staging_root / review._candidate_name
        try:
            _candidate_current(
                candidate,
                review._candidate_identity,
                review._secret,
            )
        except Exception:
            raise ActorPackImportError("actor_pack_import_review_stale") from None
        return candidate


def _absolute_path(value: os.PathLike[str] | str) -> Path:
    raw = os.fspath(value)
    if type(raw) is not str or not raw or "\x00" in raw:
        raise ActorPackImportError("actor_pack_import_invalid")
    path = Path(raw)
    if not path.is_absolute() or str(path) != raw:
        raise ActorPackImportError("actor_pack_import_invalid")
    return path


def _pin_source(path: Path) -> tuple[BinaryIO, tuple[int, ...], str, int]:
    before = os.lstat(path)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > _MAX_ARCHIVE_BYTES
    ):
        raise ValueError
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(before):
            raise ValueError
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, _READ_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_ARCHIVE_BYTES:
                raise ValueError
            digest.update(chunk)
        after = os.fstat(descriptor)
        if total != before.st_size or _file_identity(after) != _file_identity(before):
            raise ValueError
        os.lseek(descriptor, 0, os.SEEK_SET)
        stream = os.fdopen(descriptor, "rb", closefd=True)
        descriptor = -1
        return stream, _file_identity(before), digest.hexdigest(), total
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _source_identity(path: Path) -> tuple[int, ...]:
    return _file_identity(os.lstat(path))


def _validated_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if not 2 <= len(infos) <= MAX_FILES + 1:
        raise ValueError
    names: list[str] = []
    members: dict[str, zipfile.ZipInfo] = {}
    collision_keys: set[str] = set()
    total = 0
    for info in infos:
        if info.is_dir():
            raise ValueError
        name = getattr(info, "orig_filename", info.filename)
        if name == "actor-pack.json":
            canonical = name
        else:
            canonical = canonical_member_order((name,))[0]
        collision = unicodedata.normalize("NFC", canonical).casefold()
        if canonical in members or collision in collision_keys:
            raise ValueError
        mode = (info.external_attr >> 16) & 0o170000
        if mode not in {0, stat.S_IFREG} or info.flag_bits & 0x1:
            raise ValueError
        if info.create_system not in {0, ZIP_CREATE_SYSTEM}:
            raise ValueError
        if canonical.casefold().endswith(_NESTED_SUFFIXES):
            raise ValueError
        if (
            info.file_size < 0
            or info.compress_size < 0
            or info.file_size > MAX_MEMBER_BYTES
            or (
                info.file_size
                and info.file_size > max(info.compress_size, 1) * _MAX_RATIO
            )
        ):
            raise ValueError
        total += info.file_size
        if total > MAX_TOTAL_BYTES + MAX_MANIFEST_BYTES:
            raise ValueError
        names.append(canonical)
        members[canonical] = info
        collision_keys.add(collision)
    if tuple(names) != canonical_member_order(tuple(names)):
        raise ValueError
    if "actor-pack.json" not in members:
        raise ValueError
    return members


def _read_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    limit: int,
    cancel_requested: Callable[[], bool],
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    with archive.open(info, "r") as source:
        while True:
            _cancel(cancel_requested)
            chunk = source.read(min(_READ_CHUNK, limit - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise ValueError
            chunks.append(chunk)
    if total != info.file_size:
        raise ValueError
    return b"".join(chunks)


def _canonical_object(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise ValueError from None
    if type(value) is not dict or canonical_json_bytes(value) != data:
        raise ValueError
    return value


def _inventory(manifest: Mapping[str, Any]) -> tuple[tuple[str, int, str], ...]:
    raw = manifest.get("files")
    if type(raw) is not list or not 2 <= len(raw) <= MAX_FILES:
        raise ValueError
    result: list[tuple[str, int, str]] = []
    for item in raw:
        if type(item) is not dict or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError
        member = canonical_member_order((item["path"],))[0]
        byte_count = item["bytes"]
        digest = item["sha256"]
        if (
            type(byte_count) is not int
            or not 0 < byte_count <= MAX_MEMBER_BYTES
            or type(digest) is not str
            or len(digest) != 64
        ):
            raise ValueError
        result.append((member, byte_count, digest))
    if [item[0] for item in result] != sorted(item[0] for item in result):
        raise ValueError
    return tuple(result)


def _preflight_space(root: Path, required: int) -> None:
    privacy = secure_private_directory(root, create=True, application_owned=True)
    if not privacy.verified_private:
        raise ValueError
    if shutil.disk_usage(root).free < required * 2 + 1024 * 1024:
        raise ActorPackImportError("actor_pack_import_disk_unavailable")


def _create_candidate(root: Path) -> tuple[Path, tuple[int, int], str]:
    name = f".import-{uuid4().hex}"
    candidate = root / name
    candidate.mkdir(mode=0o700)
    metadata = os.lstat(candidate)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ValueError
    identity = (metadata.st_dev, metadata.st_ino)
    secret = os.urandom(32).hex()
    marker = f"api1:{secret}:{name}:{identity[0]}:{identity[1]}".encode()
    descriptor = os.open(
        candidate / _MARKER,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        _write_all(descriptor, marker)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return candidate, identity, secret


def _stage_root(candidate: Path, data: bytes) -> _StagedFile:
    target = candidate / "actor-pack.json"
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        _write_all(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return _staged_file("actor-pack.json", target, hashlib.sha256(data).hexdigest())


def _extract_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    candidate: Path,
    member: str,
    expected_bytes: int,
    expected_digest: str,
    cancel_requested: Callable[[], bool],
) -> _StagedFile:
    target = candidate.joinpath(*member.split("/"))
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    digest = hashlib.sha256()
    written = 0
    try:
        with archive.open(info, "r") as source:
            while True:
                _cancel(cancel_requested)
                chunk = source.read(_READ_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > expected_bytes:
                    raise ValueError
                digest.update(chunk)
                _write_all(descriptor, chunk)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if written != expected_bytes or digest.hexdigest() != expected_digest:
        raise ValueError
    return _staged_file(member, target, expected_digest)


def _staged_file(member: str, path: Path, digest: str) -> _StagedFile:
    metadata = os.lstat(path)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError
    return _StagedFile(member, _file_identity(metadata), digest)


def _read_staged(candidate: Path, record: _StagedFile) -> bytes:
    path = candidate.joinpath(*record.member.split("/"))
    before = os.lstat(path)
    if _file_identity(before) != record.identity:
        raise ValueError
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if _file_identity(opened) != record.identity:
            raise ValueError
        data = bytearray()
        while True:
            chunk = os.read(descriptor, _READ_CHUNK)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > record.identity[4]:
                raise ValueError
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    result = bytes(data)
    if (
        _file_identity(after) != record.identity
        or len(result) != record.identity[4]
        or hashlib.sha256(result).hexdigest() != record.sha256
    ):
        raise ValueError
    return result


def _portrait_review(member: str, data: bytes) -> ActorPackPortraitReview:
    suffix = Path(member).suffix
    mime_type = _MIME_BY_SUFFIX.get(suffix)
    if mime_type is None:
        raise ValueError
    from io import BytesIO

    with Image.open(BytesIO(data)) as image:
        width, height = image.size
    return ActorPackPortraitReview(
        mime_type,
        width,
        height,
        len(data),
        hashlib.sha256(data).hexdigest(),
    )


def _candidate_current(candidate: Path, identity: tuple[int, int], secret: str) -> None:
    metadata = os.lstat(candidate)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or (metadata.st_dev, metadata.st_ino) != identity
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise ValueError
    marker = candidate / _MARKER
    expected = f"api1:{secret}:{candidate.name}:{identity[0]}:{identity[1]}".encode()
    descriptor = os.open(marker, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if os.read(descriptor, len(expected) + 1) != expected:
            raise ValueError
    finally:
        os.close(descriptor)


def _cleanup_candidate(candidate: Path) -> bool:
    try:
        metadata = os.lstat(candidate)
        if not stat.S_ISDIR(metadata.st_mode):
            return False
        for root, directories, files in os.walk(candidate, topdown=False):
            base = Path(root)
            for name in files:
                entry = base / name
                if not stat.S_ISREG(os.lstat(entry).st_mode):
                    return False
                entry.unlink()
            for name in directories:
                entry = base / name
                if not stat.S_ISDIR(os.lstat(entry).st_mode):
                    return False
                entry.rmdir()
        candidate.rmdir()
        return True
    except OSError:
        return False


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
    )


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError
        view = view[written:]


def _cancel(checker: Callable[[], bool]) -> None:
    try:
        cancelled = checker()
    except Exception:
        cancelled = True
    if cancelled is True:
        raise ActorPackImportError("actor_pack_import_cancelled")


__all__ = [
    "ActorPackImportError",
    "ActorPackImportReview",
    "ActorPackImportService",
    "ActorPackPortraitPreview",
    "ActorPackPortraitReview",
]
