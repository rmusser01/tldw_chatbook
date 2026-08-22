"""Consistent local actor snapshots for deterministic Actor Pack export."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, BinaryIO

from tldw_chatbook import __version__

from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.Character_Chat.visual_identity import (
    compute_pack_content_sha256,
    load_visual_identity_asset,
    parse_visual_identity_manifest_json,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
from tldw_chatbook.Persona_Visual.assets import (
    PersonaVisualAssetError,
    PersonaVisualAssetMetadata,
    load_persona_visual_asset,
)
from tldw_chatbook.Persona_Visual.repository import (
    PersonaVisualRepository,
)

from .contracts import (
    ActorPackValidationError,
    ZIP_COMPRESSION,
    ZIP_CREATE_SYSTEM,
    ZIP_EXTERNAL_ATTR,
    ZIP_GENERAL_PURPOSE_FLAGS,
    ZIP_TIMESTAMP,
    actor_pack_content_digest,
    build_file_inventory,
    canonical_json_bytes,
    canonical_member_order,
    canonicalize_actor_payload,
    validate_actor_pack_document,
    validate_actor_portrait,
)
from .repository import (
    ActorPackRepository,
    ActorPackRepositoryError,
    PortableActorIdentity,
)


_VALIDATION_UUID = "123e4567-e89b-42d3-a456-426614174000"
_WRITE_CHUNK_BYTES = 64 * 1024


class ActorPackExportError(ValueError):
    """One stable, path-free Actor Pack export failure."""

    def __init__(self, category: str, *, user_message: str | None = None) -> None:
        self.category = category
        self.user_message = user_message
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class ActorPackExportSnapshot:
    """Immutable actor/portrait/portable-identity export authority."""

    actor_kind: str
    actor_revision: int
    portable_uuid: str
    identity_version: int
    portrait_name: str
    portrait_sha256: str
    local_actor_id: str = field(repr=False)
    actor_payload: bytes = field(repr=False)
    portrait_bytes: bytes = field(repr=False)
    sections: tuple[ActorPackExportSection, ...] = ()


@dataclass(frozen=True, slots=True)
class ActorPackExportFile:
    """One verified archive member before deterministic ZIP construction."""

    path: str
    sha256: str
    data: bytes = field(repr=False)
    source_identity: tuple[tuple[int, ...], ...] = field(repr=False, default=())


@dataclass(frozen=True, slots=True)
class ActorPackExportSection:
    """One self-contained visual section and its exact graph authority."""

    kind: str
    manifest_path: str
    graph_identity: object = field(repr=False)
    license: str | None
    provenance: str | None
    manifest_bytes: bytes = field(repr=False)
    assets: tuple[ActorPackExportFile, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class ActorPackExportResult:
    """Path-free result reserved for archive/publication phases."""

    archive_sha256: str
    committed: bool
    durability: str


class ActorPackExportService:
    """Capture one exact local actor before archive materialization."""

    def __init__(
        self,
        database: CharactersRAGDB,
        local_service: LocalCharacterPersonaService,
        repository: ActorPackRepository,
        *,
        persona_visual_repository: PersonaVisualRepository | None = None,
        visual_identity_repository: VisualIdentityRepository | None = None,
        profile_root: Path | None = None,
    ) -> None:
        self.database = database
        self.local_service = local_service
        self.repository = repository
        self.persona_visual_repository = persona_visual_repository
        self.visual_identity_repository = visual_identity_repository
        self.profile_root = profile_root

    def capture_snapshot(
        self,
        actor_kind: str,
        local_actor_id: str,
        *,
        source: str,
        phase_hook: Callable[[str], None] | None = None,
    ) -> ActorPackExportSnapshot:
        """Validate, assign portable identity, and freeze a reread snapshot."""

        if type(source) is not str or source != "local":
            raise ActorPackExportError(
                "actor_pack_source_not_local",
                user_message="Save a local copy first",
            )
        actor_id = _actor_id(actor_kind, local_actor_id)
        initial, initial_portrait = self._read_candidate(actor_kind, actor_id)
        self._validate_candidate(actor_kind, initial, initial_portrait)
        try:
            identity = self.repository.assign_identity(
                actor_kind, actor_id, source=source
            )
        except ActorPackRepositoryError as exc:
            raise ActorPackExportError("actor_pack_export_failed") from exc
        if phase_hook is not None:
            phase_hook("identity_assigned")
        current, current_portrait = self._read_candidate(actor_kind, actor_id)
        self._validate_candidate(actor_kind, current, current_portrait)
        if _candidate_digest(
            actor_kind, initial, initial_portrait
        ) != _candidate_digest(actor_kind, current, current_portrait):
            raise ActorPackExportError("actor_pack_export_authority_changed")
        sections = self._capture_sections(actor_kind, actor_id)
        if phase_hook is not None:
            phase_hook("visuals_loaded")
        final, final_portrait = self._read_candidate(actor_kind, actor_id)
        self._validate_candidate(actor_kind, final, final_portrait)
        final_sections = self._capture_sections(actor_kind, actor_id)
        try:
            final_identity = self.repository.get_identity(actor_kind, actor_id)
        except ActorPackRepositoryError:
            raise ActorPackExportError("actor_pack_export_failed") from None
        if (
            _candidate_digest(actor_kind, current, current_portrait)
            != _candidate_digest(actor_kind, final, final_portrait)
            or final_sections != sections
            or final_identity != identity
        ):
            raise ActorPackExportError("actor_pack_export_authority_changed")
        return _snapshot(
            actor_kind,
            actor_id,
            final,
            final_portrait,
            identity,
            sections=sections,
        )

    def _capture_sections(
        self, actor_kind: str, actor_id: int | str
    ) -> tuple[ActorPackExportSection, ...]:
        sections: list[ActorPackExportSection] = []
        shared = self._capture_shared_visual(actor_kind, actor_id)
        if shared is not None:
            sections.append(shared)
        persona = self._capture_persona_visual(actor_kind, actor_id)
        if persona is not None:
            sections.append(persona)
        return tuple(sections)

    def _capture_persona_visual(
        self, actor_kind: str, actor_id: int | str
    ) -> ActorPackExportSection | None:
        if actor_kind != "persona" or self.persona_visual_repository is None:
            return None
        try:
            export_graph = (
                self.persona_visual_repository.get_active_persona_pack_for_export(
                    str(actor_id)
                )
            )
        except ValueError:
            raise ActorPackExportError("actor_pack_export_visual_invalid") from None
        if export_graph is None:
            return None
        if self.profile_root is None:
            raise ActorPackExportError("actor_pack_export_asset_unavailable")
        files: list[ActorPackExportFile] = []
        try:
            for index, export_asset in enumerate(export_graph.assets, start=1):
                record = export_asset.record
                source_identity = _source_identity(
                    self.profile_root, export_asset.storage_key
                )
                loaded = load_persona_visual_asset(
                    self.profile_root,
                    storage_key=export_asset.storage_key,
                    metadata=PersonaVisualAssetMetadata(
                        asset_key=record.asset_key,
                        role=record.role,
                        mime_type=record.mime_type,
                        byte_count=record.byte_count,
                        sha256=record.sha256,
                        width=record.width,
                        height=record.height,
                        frame_count=record.frame_count,
                        duration_ms=record.duration_ms,
                    ),
                )
                if (
                    _source_identity(self.profile_root, export_asset.storage_key)
                    != source_identity
                ):
                    raise PersonaVisualAssetError
                files.append(
                    ActorPackExportFile(
                        path=(
                            f"persona-runtime/assets/asset-{index:04d}"
                            f"{_mime_suffix(record.mime_type)}"
                        ),
                        sha256=record.sha256,
                        data=loaded.data,
                        source_identity=source_identity,
                    )
                )
        except PersonaVisualAssetError:
            raise ActorPackExportError("actor_pack_export_asset_unavailable") from None
        context = dict(export_graph.source_context)
        return ActorPackExportSection(
            kind="persona-runtime",
            manifest_path="persona-runtime/manifest.json",
            graph_identity=export_graph.graph.identity,
            license=context.get("license"),
            provenance=context.get("provenance"),
            manifest_bytes=export_graph.manifest_bytes,
            assets=tuple(files),
        )

    def _capture_shared_visual(
        self, actor_kind: str, actor_id: int | str
    ) -> ActorPackExportSection | None:
        if self.visual_identity_repository is None:
            return None
        try:
            graph = self.visual_identity_repository.get_active_actor_pack(
                actor_kind, actor_id
            )
            if graph is None:
                return None
            pack = graph["pack"]
            version = graph["version"]
            binding = graph["binding"]
            rows = graph["assets"]
            manifest = parse_visual_identity_manifest_json(version["manifest_json"])
            if len(rows) != len(manifest.assets):
                raise ValueError
            context = json.loads(
                pack["source_context_json"],
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
            provenance = _bounded_provenance(context)
            raw_manifest = json.loads(version["manifest_json"])
            raw_assets = raw_manifest["assets"]
            if type(raw_assets) is not list or len(raw_assets) != len(manifest.assets):
                raise ValueError
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise ActorPackExportError("actor_pack_export_visual_invalid") from None
        if self.profile_root is None and pack["source_kind"] != "builtin":
            raise ActorPackExportError("actor_pack_export_asset_unavailable")
        files: list[ActorPackExportFile] = []
        try:
            for index, (asset, row, raw_asset) in enumerate(
                zip(manifest.assets, rows, raw_assets, strict=True), start=1
            ):
                if not _shared_asset_matches(asset, row):
                    raise ValueError
                source_root = (
                    Path(str(resources.files("tldw_chatbook").joinpath("assets")))
                    if pack["source_kind"] == "builtin"
                    else self.profile_root / "visual_identities"
                )
                source_identity = _source_identity(source_root, asset.storage_relpath)
                loaded = load_visual_identity_asset(
                    asset,
                    source_kind=pack["source_kind"],
                    user_data_dir=self.profile_root,
                )
                if (
                    _source_identity(source_root, asset.storage_relpath)
                    != source_identity
                ):
                    raise ValueError
                path = (
                    f"shared-visual-identity/assets/asset-{index:04d}"
                    f"{_mime_suffix(asset.content_type)}"
                )
                raw_asset["storage_relpath"] = path
                files.append(
                    ActorPackExportFile(
                        path, asset.sha256, loaded.data, source_identity
                    )
                )
        except (KeyError, TypeError, ValueError):
            raise ActorPackExportError("actor_pack_export_asset_unavailable") from None
        raw_manifest["pack_content_sha256"] = compute_pack_content_sha256(raw_manifest)
        manifest_bytes = canonical_json_bytes(raw_manifest)
        graph_identity = (
            int(binding["id"]),
            int(binding["version"]),
            int(pack["id"]),
            int(pack["version"]),
            int(version["id"]),
            int(version["version_number"]),
            hashlib.sha256(manifest_bytes).hexdigest(),
        )
        return ActorPackExportSection(
            kind="shared-visual-identity",
            manifest_path="shared-visual-identity/manifest.json",
            graph_identity=graph_identity,
            license=manifest.license,
            provenance=provenance,
            manifest_bytes=manifest_bytes,
            assets=tuple(files),
        )

    def _read_candidate(
        self, actor_kind: str, actor_id: int | str
    ) -> tuple[dict[str, Any], bytes]:
        try:
            if actor_kind == "character":
                actor = dict(self.local_service.get_character(int(actor_id)))
            else:
                actor = dict(self.local_service.get_persona_profile(str(actor_id)))
        except (CharactersRAGDBError, KeyError, TypeError, ValueError):
            raise ActorPackExportError("actor_pack_actor_unavailable") from None
        if actor_kind == "character":
            portrait = actor.get("image")
        else:
            character_id = actor.get("character_card_id")
            linked = (
                self.database.get_character_card_by_id(character_id)
                if type(character_id) is int and character_id > 0
                else None
            )
            portrait = None if linked is None else linked.get("image")
        if type(portrait) is not bytes:
            raise ActorPackExportError("actor_pack_portrait_invalid")
        return actor, portrait

    @staticmethod
    def _validate_candidate(
        actor_kind: str, actor: dict[str, Any], portrait: bytes
    ) -> None:
        name = _portrait_name(portrait)
        try:
            validate_actor_portrait(name, portrait)
            canonicalize_actor_payload(actor_kind, _VALIDATION_UUID, actor)
        except ActorPackValidationError as exc:
            category = (
                "actor_pack_portrait_invalid"
                if exc.category == "actor_pack_portrait_invalid"
                else "actor_pack_actor_invalid"
            )
            raise ActorPackExportError(category) from None


def write_actor_pack_archive(snapshot: ActorPackExportSnapshot, sink: BinaryIO) -> str:
    """Write one deterministic Actor Pack and return its archive SHA-256."""

    if type(snapshot) is not ActorPackExportSnapshot:
        raise ActorPackExportError("actor_pack_export_snapshot_invalid")
    try:
        files = _snapshot_files(snapshot)
        inventory = build_file_inventory(files)
        root: dict[str, object] = {
            "schema": "tldw.actor-pack/v1",
            "actor": {
                "kind": snapshot.actor_kind,
                "portable_uuid": snapshot.portable_uuid,
                "payload": "actor/actor.json",
                "portrait": f"actor/{snapshot.portrait_name}",
            },
            "sections": [
                {"kind": section.kind, "manifest": section.manifest_path}
                for section in snapshot.sections
            ],
            "producer": {"name": "tldw_chatbook", "version": __version__},
            "license": {"value": "unspecified"},
            "provenance": {"source": "local"},
            "required_features": [
                feature
                for kind, feature in (
                    (
                        "shared-visual-identity",
                        "shared-visual-identity/v1",
                    ),
                    ("persona-runtime", "persona-runtime/sprite-frames-v1"),
                )
                if any(section.kind == kind for section in snapshot.sections)
            ],
            "files": [
                {"path": item.path, "bytes": item.byte_count, "sha256": item.sha256}
                for item in inventory
            ],
        }
        root["content_digest"] = actor_pack_content_digest(root)
        validate_actor_pack_document(root, files)
        archive_files = {"actor-pack.json": canonical_json_bytes(root), **files}
        sink.seek(0)
        sink.truncate(0)
        with zipfile.ZipFile(
            sink,
            mode="w",
            compression=ZIP_COMPRESSION,
            allowZip64=True,
        ) as archive:
            for path in canonical_member_order(tuple(archive_files)):
                info = zipfile.ZipInfo(path, date_time=ZIP_TIMESTAMP)
                info.compress_type = ZIP_COMPRESSION
                info.create_system = ZIP_CREATE_SYSTEM
                info.flag_bits = ZIP_GENERAL_PURPOSE_FLAGS
                info.external_attr = ZIP_EXTERNAL_ATTR
                with archive.open(info, mode="w", force_zip64=True) as member:
                    data = archive_files[path]
                    for offset in range(0, len(data), _WRITE_CHUNK_BYTES):
                        member.write(data[offset : offset + _WRITE_CHUNK_BYTES])
        sink.flush()
        sink.seek(0)
        digest = hashlib.sha256()
        while chunk := sink.read(_WRITE_CHUNK_BYTES):
            digest.update(chunk)
        sink.seek(0, os.SEEK_END)
        return digest.hexdigest()
    except ActorPackExportError:
        raise
    except (
        ActorPackValidationError,
        OSError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
    ):
        raise ActorPackExportError("actor_pack_export_archive_failed") from None


def _snapshot_files(snapshot: ActorPackExportSnapshot) -> dict[str, bytes]:
    files = {
        "actor/actor.json": snapshot.actor_payload,
        f"actor/{snapshot.portrait_name}": snapshot.portrait_bytes,
    }
    for section in snapshot.sections:
        if section.manifest_path in files:
            raise ActorPackExportError("actor_pack_export_snapshot_invalid")
        files[section.manifest_path] = section.manifest_bytes
        for asset in section.assets:
            if asset.path in files:
                raise ActorPackExportError("actor_pack_export_snapshot_invalid")
            files[asset.path] = asset.data
    return files


def _actor_id(actor_kind: object, value: object) -> int | str:
    if actor_kind == "character":
        if type(value) is not str or not value.isdigit() or int(value) < 1:
            raise ActorPackExportError("actor_pack_actor_unavailable")
        return int(value)
    if actor_kind == "persona":
        if type(value) is not str or not value:
            raise ActorPackExportError("actor_pack_actor_unavailable")
        return value
    raise ActorPackExportError("actor_pack_actor_unavailable")


def _portrait_name(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "portrait.png"
    if data.startswith(b"\xff\xd8\xff"):
        return "portrait.jpg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "portrait.gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "portrait.webp"
    return "portrait.invalid"


def _candidate_digest(actor_kind: str, actor: dict[str, Any], portrait: bytes) -> str:
    payload = canonicalize_actor_payload(actor_kind, _VALIDATION_UUID, actor)
    return hashlib.sha256(payload + hashlib.sha256(portrait).digest()).hexdigest()


def _snapshot(
    actor_kind: str,
    actor_id: int | str,
    actor: dict[str, Any],
    portrait: bytes,
    identity: PortableActorIdentity,
    *,
    sections: tuple[ActorPackExportSection, ...] = (),
) -> ActorPackExportSnapshot:
    revision = actor.get("version", 1)
    if type(revision) is not int or revision < 1:
        raise ActorPackExportError("actor_pack_actor_invalid")
    return ActorPackExportSnapshot(
        actor_kind=actor_kind,
        actor_revision=revision,
        portable_uuid=identity.portable_uuid,
        identity_version=identity.version,
        portrait_name=_portrait_name(portrait),
        portrait_sha256=hashlib.sha256(portrait).hexdigest(),
        local_actor_id=str(actor_id),
        actor_payload=canonicalize_actor_payload(
            actor_kind, identity.portable_uuid, actor
        ),
        portrait_bytes=portrait,
        sections=sections,
    )


def _mime_suffix(mime_type: str) -> str:
    try:
        return {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/gif": ".gif",
            "image/webp": ".webp",
        }[mime_type]
    except KeyError:
        raise ActorPackExportError("actor_pack_export_asset_unavailable") from None


def _bounded_provenance(value: object) -> str | None:
    if type(value) is not dict:
        raise ValueError
    provenance = value.get("provenance")
    if provenance is None:
        return None
    if (
        type(provenance) is not str
        or not provenance
        or len(provenance) > 256
        or any(character in provenance for character in ("/", "\\", "\x00", "\n"))
    ):
        raise ValueError
    return provenance


def _shared_asset_matches(asset: Any, row: Any) -> bool:
    try:
        return (
            row["expression_key"] == asset.expression_key
            and row["original_expression_key"] == asset.original_label
            and row["display_label"] == asset.display_label
            and row["storage_relpath"] == asset.storage_relpath
            and row["content_type"] == asset.content_type
            and type(row["bytes"]) is int
            and row["bytes"] == asset.bytes
            and row["sha256"] == asset.sha256
            and type(row["width"]) is int
            and row["width"] == asset.width
            and type(row["height"]) is int
            and row["height"] == asset.height
            and type(row["is_animated"]) is int
            and row["is_animated"] in {0, 1}
            and bool(row["is_animated"]) == asset.is_animated
            and row["frame_count"] == asset.frame_count
            and row["duration_ms"] == asset.duration_ms
        )
    except (KeyError, TypeError):
        return False


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ValueError


def _source_identity(root: Path, storage_key: str) -> tuple[tuple[int, ...], ...]:
    try:
        if not root.is_absolute() or type(storage_key) is not str:
            raise ValueError
        current = root
        root_metadata = os.lstat(root)
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise ValueError
        identities: list[tuple[int, ...]] = [
            (
                root_metadata.st_dev,
                root_metadata.st_ino,
                root_metadata.st_mode,
                root_metadata.st_size,
                root_metadata.st_mtime_ns,
                root_metadata.st_ctime_ns,
            )
        ]
        parts = storage_key.split("/")
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise ValueError
        for index, part in enumerate(parts):
            current = current / part
            metadata = os.lstat(current)
            if index < len(parts) - 1:
                if not stat.S_ISDIR(metadata.st_mode):
                    raise ValueError
            elif not stat.S_ISREG(metadata.st_mode):
                raise ValueError
            identities.append(
                (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mode,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                )
            )
        return tuple(identities)
    except (OSError, TypeError, ValueError):
        raise ActorPackExportError("actor_pack_export_asset_unavailable") from None
