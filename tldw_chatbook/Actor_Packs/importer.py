"""Review-first import of untrusted portable Actor Pack archives."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import struct
import unicodedata
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
from tldw_chatbook.Utils.private_paths import secure_private_directory
from tldw_chatbook.Utils.path_validation import validate_path

# TASK-21200: ``Persona_Visual.*`` and ``Character_Chat.visual_identity``
# (module-level ``from PIL import Image``) are imported inside the two functions
# that need them, never at module scope. ``app.py`` imports this module at module
# scope, so a module-level import here puts PIL and most of Persona_Visual back on
# the ``import tldw_chatbook.app`` path -- the exact TASK-21103 regression this
# module re-introduced. Guarded by
# ``Tests/Packaging/test_persona_buddy_import_closure.py``.

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
    validate_actor_pack_manifest,
    validate_actor_payload,
    validate_actor_portrait,
)
from .repository import ActorPackRepository


_MAX_ARCHIVE_BYTES = MAX_TOTAL_BYTES + 16 * 1024 * 1024
_MAX_RATIO = 100
_MAX_CENTRAL_DIRECTORY_BYTES = (MAX_FILES + 1) * 2048
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

    __slots__ = ("category",)

    def __init__(self, category: str) -> None:
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
class ActorPackFieldDifference:
    """One bounded portable-field difference for explicit review."""

    field_name: str
    current_value: str
    incoming_value: str


@dataclass(frozen=True, slots=True)
class _StagedFile:
    member: str
    identity: tuple[int, int, int, int, int, int]
    sha256: str


@dataclass(frozen=True, slots=True)
class _ActorPackImportMaterial:
    actor_fields: Mapping[str, Any]
    portrait: bytes = field(repr=False)
    sections: tuple[_ActorPackSectionMaterial, ...] = field(repr=False, default=())


@dataclass(frozen=True, slots=True)
class _ActorPackSectionAsset:
    asset_key: str
    member: str
    mime_type: str
    width: int
    height: int
    byte_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _ActorPackSectionMaterial:
    kind: str
    manifest: Mapping[str, Any]
    assets: tuple[tuple[_ActorPackSectionAsset, bytes], ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class ActorPackImportReview:
    """Immutable consent model; filesystem authority is private and repr-free."""

    actor_kind: str
    portable_uuid: str
    actor_fields: tuple[tuple[str, Any], ...]
    portrait: ActorPackPortraitReview
    sections: tuple[str, ...]
    section_effects: tuple[tuple[str, str], ...]
    license: tuple[tuple[str, str], ...]
    provenance: tuple[tuple[str, str], ...]
    warnings: tuple[str, ...]
    differences: tuple[ActorPackFieldDifference, ...]
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
    _shared_visual_authority: tuple[Any, ...] | None = field(repr=False)
    _persona_visual_authority: tuple[Any, ...] | None = field(repr=False)
    _required_free_bytes: int = field(repr=False)
    _section_assets: tuple[tuple[str, tuple[_ActorPackSectionAsset, ...]], ...] = field(
        repr=False
    )
    _source_path: Path = field(repr=False)
    _source_identity: tuple[int, ...] = field(repr=False)
    _source_sha256: str = field(repr=False)


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
        self.sweep_staging()

    def sweep_staging(self, *, max_candidates: int = 32) -> int:
        """Remove a bounded set of authenticated crash-left staging candidates."""

        if type(max_candidates) is not int or not 1 <= max_candidates <= 128:
            raise ActorPackImportError("actor_pack_import_invalid")
        try:
            privacy = secure_private_directory(
                self._staging_root, create=True, application_owned=True
            )
            if not privacy.verified_private:
                if privacy.usable:
                    return 0
                raise ValueError
            removed = 0
            examined = 0
            with os.scandir(self._staging_root) as entries:
                candidates = (entry.name for entry in entries)
                for name in candidates:
                    if examined >= max_candidates:
                        break
                    examined += 1
                    if not _candidate_name(name):
                        continue
                    candidate = self._staging_root / name
                    authority = _read_candidate_authority(candidate)
                    if authority is None:
                        continue
                    identity, secret = authority
                    try:
                        _candidate_current(candidate, identity, secret)
                    except Exception:
                        continue
                    removed += int(_cleanup_candidate(candidate, identity))
            return removed
        except ActorPackImportError:
            raise
        except Exception:
            raise ActorPackImportError("actor_pack_import_cleanup_denied") from None

    def inspect_archive(
        self,
        archive_path: os.PathLike[str] | str,
        *,
        cancel_requested: Callable[[], bool] = lambda: False,
    ) -> ActorPackImportReview:
        """Inspect and stage one archive while exposing path-free metadata.

        Args:
            archive_path: Absolute user-selected Actor Pack archive path.
            cancel_requested: Callback returning whether inspection should stop.

        Returns:
            An immutable review lease for a validated staged archive.

        Raises:
            ActorPackImportError: The archive is invalid, unsupported, cancelled,
                unavailable, or cannot be staged safely.
        """

        candidate: Path | None = None
        try:
            _cancel(cancel_requested)
            source_path = _absolute_path(archive_path)
            source, source_identity, archive_sha256, archive_bytes = _pin_source(
                source_path
            )
            try:
                _preflight_zip_directory(source, archive_bytes)
                with source, zipfile.ZipFile(source, "r") as archive:
                    infos = _validated_members(archive)
                    root_bytes = _read_member(
                        archive,
                        infos["actor-pack.json"],
                        MAX_MANIFEST_BYTES,
                        cancel_requested,
                    )
                    manifest = _canonical_object(root_bytes)
                    document = validate_actor_pack_manifest(manifest)
                    inventory = tuple(
                        (item.path, item.byte_count, item.sha256)
                        for item in document.files
                    )
                    declared = {item[0] for item in inventory}
                    if set(infos) != declared | {"actor-pack.json"}:
                        raise ValueError
                    _preflight_space(
                        self._staging_root,
                        archive_bytes + sum(item[1] for item in inventory),
                    )
                    candidate, identity, secret = _create_candidate(self._staging_root)
                    staged = [_stage_root(candidate, root_bytes)]
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
                    staged_records = {record.member: record for record in staged}
                    actor_bytes = _read_staged(
                        candidate, staged_records[document.payload_path]
                    )
                    validate_actor_payload(
                        actor_bytes,
                        actor_kind=document.actor_kind,
                        portable_uuid=document.portable_uuid,
                    )
                    actor = _canonical_object(actor_bytes)
                    actor_fields = tuple(sorted(actor["data"].items()))
                    portrait_data = _read_staged(
                        candidate, staged_records[document.portrait_path]
                    )
                    validate_actor_portrait(document.portrait_path, portrait_data)
                    portrait = _portrait_review(document.portrait_path, portrait_data)
                    sections = tuple(section.kind for section in document.sections)
                    section_assets = _validate_sections(
                        document.sections,
                        frozenset(staged_records),
                        lambda member: _read_staged(candidate, staged_records[member]),
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
                    shared_authority, persona_visual_authority = (
                        self._visual_authorities(matched)
                    )
                    expected_sections = (
                        ("shared-visual-identity", "persona-runtime")
                        if document.actor_kind == "persona"
                        else ("shared-visual-identity",)
                    )
                    section_effects = _section_effects(
                        expected_sections, sections, allowed_actions
                    )
                    differences = (
                        ()
                        if matched is None
                        else self._differences(matched, actor["data"])
                    )
            finally:
                if not source.closed:
                    source.close()
            _cancel(cancel_requested)
            if not _source_is_current(source_path, source_identity, archive_sha256):
                raise ActorPackImportError("actor_pack_import_review_stale")
            _candidate_current(candidate, identity, secret)
            return ActorPackImportReview(
                actor_kind=document.actor_kind,
                portable_uuid=document.portable_uuid,
                actor_fields=actor_fields,
                portrait=portrait,
                sections=sections,
                section_effects=section_effects,
                license=_display_metadata(manifest.get("license")),
                provenance=_display_metadata(manifest.get("provenance")),
                warnings=(),
                differences=differences,
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
                _shared_visual_authority=shared_authority,
                _persona_visual_authority=persona_visual_authority,
                _required_free_bytes=sum(item[1] for item in inventory),
                _section_assets=section_assets,
                _source_path=source_path,
                _source_identity=source_identity,
                _source_sha256=archive_sha256,
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
        except ActorPackImportError:
            raise
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
        if not _cleanup_candidate(candidate, review._candidate_identity):
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
            if not _source_is_current(
                review._source_path,
                review._source_identity,
                review._source_sha256,
            ):
                raise ValueError
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
            shared_authority, persona_visual_authority = self._visual_authorities(
                current
            )
            if (
                shared_authority != review._shared_visual_authority
                or persona_visual_authority != review._persona_visual_authority
            ):
                raise ValueError
            _preflight_space(self._profile_root, review._required_free_bytes)
        except ActorPackImportError:
            raise
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
        section_records = {kind: assets for kind, assets in review._section_assets}
        sections: list[_ActorPackSectionMaterial] = []
        try:
            for kind in review.sections:
                manifest_member = f"{kind}/manifest.json"
                manifest = _canonical_object(
                    _read_staged(candidate, records[manifest_member])
                )
                assets = tuple(
                    (asset, _read_staged(candidate, records[asset.member]))
                    for asset in section_records[kind]
                )
                sections.append(_ActorPackSectionMaterial(kind, manifest, assets))
        except Exception:
            raise ActorPackImportError("actor_pack_import_review_stale") from None
        return _ActorPackImportMaterial(dict(fields), portrait, tuple(sections))

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
            actor = self._local_service._find_persona_profile(identity.local_actor_id)
            return self._persona_actor_authority(actor)
        if type(version) is not int or version < 1:
            raise ActorPackImportError("actor_pack_import_review_stale")
        try:
            digest = hashlib.sha256(canonical_json_bytes(data)).hexdigest()
        except (ActorPackValidationError, TypeError, ValueError):
            raise ActorPackImportError("actor_pack_import_review_stale") from None
        return version, digest

    def _persona_actor_authority(self, actor: Mapping[str, Any]) -> tuple[int, str]:
        version = actor.get("version")
        data = dict(actor)
        character_id = actor.get("character_card_id")
        if type(character_id) is int and character_id > 0:
            linked = self.repository.db.get_character_card_by_id(character_id)
            if linked is None:
                data["portrait_character_missing"] = True
            else:
                if type(linked.get("version")) is not int:
                    raise ActorPackImportError("actor_pack_import_review_stale")
                image = linked.get("image")
                if type(image) is not bytes:
                    raise ActorPackImportError("actor_pack_import_review_stale")
                data["portrait_character_version"] = linked["version"]
                data["portrait_sha256"] = hashlib.sha256(image).hexdigest()
        if type(version) is not int or version < 1:
            raise ActorPackImportError("actor_pack_import_review_stale")
        try:
            digest = hashlib.sha256(canonical_json_bytes(data)).hexdigest()
        except (ActorPackValidationError, TypeError, ValueError):
            raise ActorPackImportError("actor_pack_import_review_stale") from None
        return version, digest

    def _differences(
        self, identity: Any, incoming: Mapping[str, Any]
    ) -> tuple[ActorPackFieldDifference, ...]:
        if identity.actor_kind == "character":
            current = self.repository.db.get_character_card_by_id(
                int(identity.local_actor_id)
            )
        elif self._local_service is not None:
            current = self._local_service._find_persona_profile(identity.local_actor_id)
        else:
            current = None
        if current is None:
            raise ActorPackImportError("actor_pack_import_review_stale")
        return tuple(
            ActorPackFieldDifference(
                key,
                _display_value(current.get(key)),
                _display_value(value),
            )
            for key, value in sorted(incoming.items())
            if current.get(key) != value
        )

    def _visual_authorities(
        self, identity: Any | None
    ) -> tuple[tuple[Any, ...] | None, tuple[Any, ...] | None]:
        # Deferred: see the TASK-21200 note at the top of this module.
        from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository

        if identity is None:
            return None, None
        shared_graph = VisualIdentityRepository(
            self.repository.db
        ).get_active_actor_pack(identity.actor_kind, identity.local_actor_id)
        shared = (
            None
            if shared_graph is None
            else (
                int(shared_graph["binding"]["id"]),
                int(shared_graph["binding"]["version"]),
                int(shared_graph["pack"]["id"]),
                int(shared_graph["pack"]["version"]),
                int(shared_graph["version"]["id"]),
                hashlib.sha256(canonical_json_bytes(shared_graph)).hexdigest(),
            )
        )
        if identity.actor_kind != "persona":
            return shared, None
        persona_graph = PersonaVisualRepository(
            self.repository.db
        ).get_active_persona_pack(identity.local_actor_id)
        if persona_graph is None:
            return shared, None
        visual = persona_graph.identity
        persona = (
            visual.persona_id,
            visual.persona_revision,
            visual.binding_id,
            visual.binding_version,
            visual.pack_id,
            visual.pack_revision,
            visual.pack_version_id,
            visual.version_number,
            visual.manifest_sha256,
        )
        return shared, persona

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
    try:
        return validate_path(path, path.parent, redact_paths=True)
    except ValueError:
        raise ActorPackImportError("actor_pack_import_invalid") from None


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


def _preflight_zip_directory(source: BinaryIO, archive_bytes: int) -> None:
    """Reject an oversized central directory before ``ZipFile`` allocates it."""

    eocd_size = 22
    tail_size = min(archive_bytes, eocd_size + 65_535)
    source.seek(archive_bytes - tail_size)
    tail = source.read(tail_size)
    relative = tail.rfind(b"PK\x05\x06")
    if relative < 0 or relative + eocd_size > len(tail):
        raise ValueError
    eocd_offset = archive_bytes - tail_size + relative
    (
        signature,
        disk_number,
        directory_disk,
        entries_on_disk,
        entries_total,
        directory_bytes,
        directory_offset,
        comment_bytes,
    ) = struct.unpack("<4s4H2LH", tail[relative : relative + eocd_size])
    if (
        signature != b"PK\x05\x06"
        or disk_number != 0
        or directory_disk != 0
        or relative + eocd_size + comment_bytes != len(tail)
    ):
        raise ValueError
    if (
        entries_on_disk == 0xFFFF
        or entries_total == 0xFFFF
        or directory_bytes == 0xFFFFFFFF
        or directory_offset == 0xFFFFFFFF
    ):
        locator_offset = eocd_offset - 20
        if locator_offset < 0:
            raise ValueError
        source.seek(locator_offset)
        locator = source.read(20)
        locator_signature, locator_disk, zip64_offset, total_disks = struct.unpack(
            "<4sLQL", locator
        )
        if (
            locator_signature != b"PK\x06\x07"
            or locator_disk != 0
            or total_disks != 1
            or zip64_offset >= locator_offset
        ):
            raise ValueError
        source.seek(zip64_offset)
        zip64 = source.read(56)
        if len(zip64) != 56:
            raise ValueError
        (
            zip64_signature,
            record_bytes,
            _made_by,
            _required,
            zip64_disk,
            zip64_directory_disk,
            entries_on_disk,
            entries_total,
            directory_bytes,
            directory_offset,
        ) = struct.unpack("<4sQ2H2L4Q", zip64)
        if (
            zip64_signature != b"PK\x06\x06"
            or record_bytes < 44
            or zip64_disk != 0
            or zip64_directory_disk != 0
        ):
            raise ValueError
        directory_end_limit = zip64_offset
    else:
        directory_end_limit = eocd_offset
    if (
        entries_on_disk != entries_total
        or not 2 <= entries_total <= MAX_FILES + 1
        or directory_bytes > _MAX_CENTRAL_DIRECTORY_BYTES
        or directory_offset + directory_bytes > directory_end_limit
    ):
        raise ValueError
    source.seek(0)


def _source_is_current(
    path: Path, identity: tuple[int, ...], expected_sha256: str
) -> bool:
    try:
        stream, current, digest, _size = _pin_source(path)
    except (OSError, TypeError, ValueError):
        return False
    try:
        return current == identity and digest == expected_sha256
    finally:
        stream.close()


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
    try:
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
    except BaseException:
        shutil.rmtree(candidate, ignore_errors=True)
        raise


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
    from PIL import Image

    with Image.open(BytesIO(data)) as image:
        width, height = image.size
    return ActorPackPortraitReview(
        mime_type,
        width,
        height,
        len(data),
        hashlib.sha256(data).hexdigest(),
    )


def _validate_sections(
    sections: tuple[Any, ...],
    archive_members: frozenset[str],
    read_member: Callable[[str], bytes],
) -> tuple[tuple[str, tuple[_ActorPackSectionAsset, ...]], ...]:
    # Deferred: see the TASK-21200 note at the top of this module.
    from tldw_chatbook.Character_Chat.visual_identity import (
        validate_visual_identity_manifest,
    )
    from tldw_chatbook.Persona_Visual.validation import (
        validate_persona_visual_manifest,
    )

    validated: list[tuple[str, tuple[_ActorPackSectionAsset, ...]]] = []
    for section in sections:
        manifest_bytes = read_member(section.manifest_path)
        manifest = _canonical_object(manifest_bytes)
        asset_members = tuple(
            sorted(
                path
                for path in archive_members
                if path.startswith(f"{section.kind}/assets/")
            )
        )
        if not asset_members:
            raise ValueError
        if section.kind == "shared-visual-identity":
            visual = validate_visual_identity_manifest(
                manifest,
                directory_bytes=sum(
                    len(read_member(member)) for member in asset_members
                )
                + len(manifest_bytes),
            )
            expected = tuple(asset.storage_relpath for asset in visual.assets)
            if expected != asset_members:
                raise ValueError
            records = []
            for asset in visual.assets:
                data = read_member(asset.storage_relpath)
                mime, width, height = _section_image(
                    asset.storage_relpath, data, max_frames=512
                )
                if (
                    mime != asset.content_type
                    or width != asset.width
                    or height != asset.height
                    or len(data) != asset.bytes
                    or hashlib.sha256(data).hexdigest() != asset.sha256
                ):
                    raise ValueError
                records.append(
                    _ActorPackSectionAsset(
                        asset.expression_key,
                        asset.storage_relpath,
                        mime,
                        width,
                        height,
                        len(data),
                        asset.sha256,
                    )
                )
        else:
            asset_ids = _persona_asset_ids(manifest)
            if len(asset_ids) != len(asset_members):
                raise ValueError
            records = []
            dimensions: dict[str, tuple[int, int]] = {}
            for asset_id, member in zip(asset_ids, asset_members, strict=True):
                data = read_member(member)
                mime, width, height = _section_image(member, data)
                digest = hashlib.sha256(data).hexdigest()
                records.append(
                    _ActorPackSectionAsset(
                        asset_id,
                        member,
                        mime,
                        width,
                        height,
                        len(data),
                        digest,
                    )
                )
                dimensions[asset_id] = (width, height)
            validate_persona_visual_manifest(manifest, dimensions)
        section_members = {
            path for path in archive_members if path.startswith(f"{section.kind}/")
        }
        if section_members != {section.manifest_path, *asset_members}:
            raise ValueError
        validated.append((section.kind, tuple(records)))
    return tuple(validated)


def _section_image(
    member: str, data: bytes, *, max_frames: int = 240
) -> tuple[str, int, int]:
    mime = _MIME_BY_SUFFIX.get(Path(member).suffix)
    if mime is None:
        raise ValueError
    from io import BytesIO
    from PIL import Image

    with Image.open(BytesIO(data)) as image:
        width, height = image.size
        expected_format = {
            "image/png": "PNG",
            "image/jpeg": "JPEG",
            "image/gif": "GIF",
            "image/webp": "WEBP",
        }[mime]
        frame_count = int(getattr(image, "n_frames", 1) or 1)
        if (
            image.format != expected_format
            or type(width) is not int
            or type(height) is not int
            or not 1 <= width <= 4096
            or not 1 <= height <= 4096
            or not 1 <= frame_count <= max_frames
            or width * height * frame_count > 4096**2 * 4
        ):
            raise ValueError
        image.load()
    return mime, width, height


def _section_effects(
    expected_sections: tuple[str, ...],
    included_sections: tuple[str, ...],
    allowed_actions: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    labels = {
        "create_new": "Create New",
        "create_copy": "Create Copy",
        "update_existing": "Update Existing",
    }
    effects: list[tuple[str, str]] = []
    for kind in expected_sections:
        included = kind in included_sections
        descriptions = []
        for action in allowed_actions:
            if included:
                effect = (
                    "imported visuals will replace the current binding"
                    if action == "update_existing"
                    else "imported visuals will be activated"
                )
            else:
                effect = (
                    "Not included — existing visuals will be preserved"
                    if action == "update_existing"
                    else "Not included — no visual binding will be created"
                )
            descriptions.append(f"{labels[action]}: {effect}")
        effects.append((kind, "; ".join(descriptions)))
    return tuple(effects)


def _persona_asset_ids(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    animations = manifest.get("animations")
    if type(animations) is not dict:
        raise ValueError
    asset_ids: set[str] = set()
    for animation in animations.values():
        if type(animation) is not dict or type(animation.get("frames")) is not list:
            raise ValueError
        preview = animation.get("preview_asset_id")
        if preview is not None:
            if type(preview) is not str:
                raise ValueError
            asset_ids.add(preview)
        for frame in animation["frames"]:
            if type(frame) is not dict or type(frame.get("asset_id")) is not str:
                raise ValueError
            asset_ids.add(frame["asset_id"])
    return tuple(sorted(asset_ids))


def _display_metadata(value: object) -> tuple[tuple[str, str], ...]:
    if type(value) is not dict or not 1 <= len(value) <= 8:
        raise ValueError
    result: list[tuple[str, str]] = []
    for key, item in sorted(value.items()):
        if type(key) is not str or not key or len(key) > 64 or type(item) is not str:
            raise ValueError
        rendered = _display_value(item, quote_strings=False)
        if not rendered or len(rendered) > 256:
            raise ValueError
        result.append((key, rendered))
    return tuple(result)


def _display_value(value: object, *, quote_strings: bool = True) -> str:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (RecursionError, TypeError, UnicodeError, ValueError):
        return "Complex value"
    if not quote_strings and type(value) is str:
        rendered = rendered[1:-1]
    if len(rendered) > 512:
        return f"{rendered[:509]}…"
    return rendered


def _candidate_name(name: str) -> bool:
    suffix = name.removeprefix(".import-")
    return (
        len(suffix) == 32
        and name.startswith(".import-")
        and all(character in "0123456789abcdef" for character in suffix)
    )


def _read_candidate_authority(
    candidate: Path,
) -> tuple[tuple[int, int], str] | None:
    directory_descriptor = marker_descriptor = -1
    try:
        metadata = os.lstat(candidate)
        if (
            not _candidate_name(candidate.name)
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            return None
        directory_descriptor = os.open(
            candidate,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        marker_descriptor = os.open(
            _MARKER,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        marker_metadata = os.fstat(marker_descriptor)
        if (
            not stat.S_ISREG(marker_metadata.st_mode)
            or marker_metadata.st_nlink != 1
            or marker_metadata.st_size > 256
        ):
            return None
        marker = os.read(marker_descriptor, 257).decode("ascii")
        scheme, secret, name, device, inode = marker.split(":")
        if (
            scheme != "api1"
            or name != candidate.name
            or len(secret) != 64
            or any(character not in "0123456789abcdef" for character in secret)
            or int(device) != metadata.st_dev
            or int(inode) != metadata.st_ino
        ):
            return None
        return (metadata.st_dev, metadata.st_ino), secret
    except (OSError, UnicodeError, ValueError):
        return None
    finally:
        if marker_descriptor >= 0:
            os.close(marker_descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)


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


def _cleanup_candidate(
    candidate: Path, expected_identity: tuple[int, int] | None = None
) -> bool:
    parent_descriptor = candidate_descriptor = -1
    try:
        if expected_identity is None:
            authority = _read_candidate_authority(candidate)
            if authority is None:
                return False
            expected_identity = authority[0]
        parent_descriptor = os.open(
            candidate.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.stat(
            candidate.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != expected_identity
            or not _candidate_name(candidate.name)
        ):
            return False
        candidate_descriptor = os.open(
            candidate.name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(candidate_descriptor)
        if (opened.st_dev, opened.st_ino) != expected_identity:
            return False
        if not _remove_directory_contents(candidate_descriptor):
            return False
        os.close(candidate_descriptor)
        candidate_descriptor = -1
        os.rmdir(candidate.name, dir_fd=parent_descriptor)
        return True
    except OSError:
        return False
    finally:
        if candidate_descriptor >= 0:
            os.close(candidate_descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)


def _remove_directory_contents(directory_descriptor: int) -> bool:
    for name in os.listdir(directory_descriptor):
        metadata = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if stat.S_ISREG(metadata.st_mode):
            os.unlink(name, dir_fd=directory_descriptor)
            continue
        if not stat.S_ISDIR(metadata.st_mode):
            return False
        child = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        try:
            if not _remove_directory_contents(child):
                return False
        finally:
            os.close(child)
        os.rmdir(name, dir_fd=directory_descriptor)
    return True


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
    "ActorPackFieldDifference",
    "ActorPackImportError",
    "ActorPackImportReview",
    "ActorPackImportService",
    "ActorPackPortraitPreview",
    "ActorPackPortraitReview",
]
